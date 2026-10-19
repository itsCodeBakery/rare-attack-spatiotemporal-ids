"""HARDREP-001: GPU trainable confusion-aware residual tabular IDS.

Exactly original 140,328 train and 35,005 development-validation target rows.
Official 82,332 record UNSW test NEVER opened. Single-seed exploratory model.
"""
import argparse,hashlib,json,random,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_recall_fscore_support
import joblib
from run_ctrl003_tabular_fast import prepare

def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(4194304),b""):h.update(b)
 return h.hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")

class Residual(nn.Module):
 def __init__(self,dim,drop):
  super().__init__()
  self.block=nn.Sequential(nn.Linear(dim,dim),nn.LayerNorm(dim),nn.GELU(),
                           nn.Dropout(drop),nn.Linear(dim,dim),nn.LayerNorm(dim))
 def forward(self,x):return torch.nn.functional.gelu(x+self.block(x))

class HardRep(nn.Module):
 def __init__(self,d,h,z,drop):
  super().__init__()
  self.stem=nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.GELU())
  self.blocks=nn.Sequential(Residual(h,drop),Residual(h,drop))
  self.embedding=nn.Linear(h,z)
  self.head=nn.Linear(z,10)
 def forward(self,x):
  z=torch.nn.functional.normalize(self.embedding(self.blocks(self.stem(x))),dim=1)
  return self.head(z)*10,z

def transform(X,V,cats):
 numeric=[x for x in X.columns if x not in cats]
 imputer=SimpleImputer(strategy="median",add_indicator=False)
 scaler=StandardScaler()
 a=scaler.fit_transform(imputer.fit_transform(X[numeric]))
 b=scaler.transform(imputer.transform(V[numeric]))
 a=np.clip(a,-12,12);b=np.clip(b,-12,12)
 enc=OneHotEncoder(handle_unknown="ignore",min_frequency=2,sparse_output=False,
                   dtype=np.float32)
 a_cat=enc.fit_transform(X[cats].astype(str))
 b_cat=enc.transform(V[cats].astype(str))
 A=np.concatenate([a.astype(np.float32),a_cat],axis=1)
 B=np.concatenate([b.astype(np.float32),b_cat],axis=1)
 if not np.isfinite(A).all() or not np.isfinite(B).all():
  raise ValueError("Nonfinite encoded network inputs")
 return A,B,{"numeric":numeric,"categorical":list(cats),
             "imputer":imputer,"scaler":scaler,"encoder":enc}

def contrastive(z,y,temp):
 """Batch supervised contrastive loss, self pairs excluded."""
 similarities=z@z.T/temp
 similarities=similarities-similarities.max(dim=1,keepdim=True).values.detach()
 n=len(y);valid=~torch.eye(n,device=y.device,dtype=torch.bool)
 positive=(y[:,None]==y[None,:])&valid
 counts=positive.sum(1)
 log_prob=similarities-torch.logsumexp(similarities.masked_fill(~valid,-1e9),dim=1,keepdim=True)
 loss=-(positive*log_prob).sum(1)/counts.clamp_min(1)
 return loss[counts>0].mean() if torch.any(counts>0) else z.sum()*0

def pair_margin(z,y,pairs,margin):
 """Hard negative nearest-class margin: sampled true pair-class prototypes."""
 loss=z.sum()*0;used=0
 for a,b in pairs:
  ia=torch.where(y==a)[0];ib=torch.where(y==b)[0]
  if len(ia)<2 or len(ib)<2:continue
  pa=torch.nn.functional.normalize(z[ia].mean(0),dim=0)
  pb=torch.nn.functional.normalize(z[ib].mean(0),dim=0)
  intra=((z[ia]@pa).mean()+(z[ib]@pb).mean())*.5
  cross=((z[ia]@pb).mean()+(z[ib]@pa).mean())*.5
  loss=loss+torch.relu(margin-(intra-cross));used+=1
 return loss/max(used,1)

@torch.no_grad()
def probabilities(model,data,device):
 model.eval();parts=[]
 for (batch,) in DataLoader(TensorDataset(torch.from_numpy(data)),batch_size=2048):
  logits,_=model(batch.to(device))
  parts.append(logits.softmax(dim=1).cpu().numpy())
 return np.concatenate(parts)

def assess(y,p):
 label=p.argmax(1)
 precision,recall,f,s=precision_recall_fscore_support(
  y,label,labels=list(range(10)),zero_division=0)
 cm=confusion_matrix(y,label,labels=list(range(10)))
 return {"accuracy":float(accuracy_score(y,label)),
         "macro_f1":float(f1_score(y,label,labels=list(range(10)),average="macro",zero_division=0)),
         "weighted_f1":float(f1_score(y,label,labels=list(range(10)),average="weighted",zero_division=0)),
         "dos_recall":float(recall[2]),"analysis_recall":float(recall[0]),
         "backdoor_recall":float(recall[1]),"n":int(len(y))},cm,pd.DataFrame(
         {"class":range(10),"support":s,"precision":precision,"recall":recall,"f1":f})

def run(args):
 repo,ref,baseline,out=map(Path,(args.repo,args.reference,args.baseline,args.output))
 cfg=json.loads(Path(args.config).read_text())
 if cfg["official_test"]!="FORBIDDEN":raise ValueError("Official test must be locked")
 source=json.loads((repo/"configs/ctrl003_fast_tabular.json").read_text())
 X,y,V,yv,classes,cats,lineage=prepare(source,ref)
 if len(y)!=cfg["training_examples"] or len(yv)!=cfg["target_validation_examples"]:
  raise ValueError("Original target rows changed")
 out.mkdir(parents=True,exist_ok=True)
 save(out/"configuration.json",cfg);save(out/"data_lineage.json",lineage)
 (out/"code_commit.txt").write_text(subprocess.check_output(
  ["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()+"\n")
 train,val,preprocess=transform(X,V,cats)
 joblib.dump(preprocess,out/"training_only_preprocessing_PRIVATE.joblib")
 print("ENCODED TRAIN SHAPE:",train.shape,"VALIDATION:",val.shape,flush=True)
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 print("TRAINING DEVICE:",device,"GPU:",torch.cuda.get_device_name(0) if device.type=="cuda" else "None",flush=True)
 random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
 if device.type=="cuda":torch.cuda.manual_seed_all(cfg["seed"])
 torch.set_num_threads(4)
 counts=np.bincount(y,minlength=10)
 # Balanced sampling + blended natural-prior weighted loss to avoid overwhelming FPs.
 weights=np.sqrt(counts.max()/counts).astype(np.float32)
 sampler_weights=np.sqrt(1/np.maximum(counts[y],1).astype(np.float64))
 sampler=WeightedRandomSampler(torch.from_numpy(sampler_weights),
                              num_samples=len(y),replacement=True)
 ds=TensorDataset(torch.from_numpy(train),torch.from_numpy(y.astype(np.int64)))
 loader=DataLoader(ds,batch_size=cfg["batch_size"],sampler=sampler,num_workers=0,pin_memory=(device.type=="cuda"))
 model=HardRep(train.shape[1],cfg["hidden_dim"],cfg["embedding_dim"],cfg["dropout"]).to(device)
 optim=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=cfg["epochs"],eta_min=cfg["lr"]*.08)
 natural_weights=torch.from_numpy(weights).to(device)
 history=[];best=-1;best_epoch=0;stale=0;started=time.monotonic()
 checkpoint=out/"best_model_PRIVATE.pt"
 for epoch in range(1,cfg["epochs"]+1):
  model.train();running=0;seen=0
  for bx,by in loader:
   bx=bx.to(device,non_blocking=True);by=by.to(device,non_blocking=True)
   optim.zero_grad(set_to_none=True)
   logits,z=model(bx)
   ce=torch.nn.functional.cross_entropy(logits,by)
   weighted=torch.nn.functional.cross_entropy(logits,by,weight=natural_weights)
   sup=contrastive(z,by,cfg["temperature"])
   pair=pair_margin(z,by,cfg["class_pair_ids"],cfg["margin"])
   loss=cfg["ce_weight"]*ce+cfg["balanced_ce_weight"]*weighted+cfg["contrastive_weight"]*sup+cfg["pair_margin_weight"]*pair
   loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
   optim.step()
   running+=loss.item()*len(by);seen+=len(by)
  scheduler.step()
  if epoch%cfg["eval_every"] and epoch!=cfg["epochs"]:continue
  pred=probabilities(model,val,device)
  metrics,_,_=assess(yv,pred)
  record={"epoch":epoch,"train_loss":running/seen,**metrics,"time_sec":round(time.monotonic()-started,1)}
  history.append(record)
  print("HARDREP EPOCH",epoch,"LOSS",round(running/seen,4),
        "MACRO-F1",round(metrics["macro_f1"],5),
        "ACC",round(metrics["accuracy"],5),
        "DOS-R",round(metrics["dos_recall"],4),flush=True)
  if metrics["macro_f1"]>best+0.00001:
   best=metrics["macro_f1"];best_epoch=epoch;stale=0
   torch.save({"model":model.state_dict(),"epoch":epoch,"input_dim":train.shape[1],
               "config":cfg,"classes":classes},checkpoint)
   np.savez_compressed(out/"best_validation_predictions_PRIVATE.npz",
      y_true=np.asarray(yv,dtype=np.uint8),probabilities=pred.astype(np.float32))
  else:stale+=cfg["eval_every"]
  pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
  if stale>=cfg["patience"]:
   print("EARLY STOP PATIENCE",stale,flush=True);break
 state=torch.load(checkpoint,map_location=device,weights_only=False)
 model.load_state_dict(state["model"])
 probs=probabilities(model,val,device)
 own,cm,per=assess(yv,probs)
 own["checkpoint_epoch"]=best_epoch
 own["checkpoint_sha256"]=digest(checkpoint)
 np.savez_compressed(out/"best_validation_predictions_PRIVATE.npz",
      y_true=np.asarray(yv,dtype=np.uint8),probabilities=probs.astype(np.float32))
 own["private_predictions_sha256"]=digest(out/"best_validation_predictions_PRIVATE.npz")
 pd.DataFrame(cm,index=classes,columns=classes).rename_axis("true_class").to_csv(out/"confusion.csv")
 per["class"]=classes
 per.to_csv(out/"per_class.csv",index=False)
 with np.load(baseline/"lgbm_rare_sqrt"/"predictions_PRIVATE.npz",allow_pickle=False) as f:
  if not np.array_equal(f["y_true"].astype(int),yv):raise ValueError("Frozen parent validation changed")
  baseline_pred=f["probabilities"].astype(float)
 with np.load(baseline/"lgbm_unweighted"/"predictions_PRIVATE.npz",allow_pickle=False) as f:
  if not np.array_equal(f["y_true"].astype(int),yv):raise ValueError("Frozen unweighted targets changed")
  std_pred=f["probabilities"].astype(float)
 allvariants={"hardrep":probs,"rare_lightgbm":baseline_pred,"standard_lightgbm":std_pred,
              "hardrep_rare_equal":.5*probs+.5*baseline_pred,
              "hardrep_standard_equal":.5*probs+.5*std_pred}
 variants={}
 for name,p in allvariants.items():
  score,_,_=assess(yv,p)
  variants[name]=score
  print("MEASURED",name,"ACC",round(score["accuracy"],5),
        "MACRO-F1",round(score["macro_f1"],5),"DoS",round(score["dos_recall"],5),flush=True)
 save(out/"summary.json",{"experiment_id":cfg["experiment_id"],
      "official_test_used":False,"train_count":len(y),"validation_count":len(yv),
      "best_epoch":best_epoch,"checkpoint":own,
      "preprocessing_sha256":digest(out/"training_only_preprocessing_PRIVATE.joblib"),
      "variants":variants,"experiment_scope":"single-seed internal development-validation; no superiority claim"})
 pd.DataFrame([{"variant":name,**score} for name,score in variants.items()]).to_csv(out/"variant_comparison.csv",index=False)
 print("\nHARDREP-001 STRONGER-MODEL EXPERIMENT COMPLETE",flush=True)
 print(pd.read_csv(out/"variant_comparison.csv").to_string(index=False))
 print("Best checkpoint:",checkpoint,flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser()
 for name in ("repo","reference","baseline","config","output"):
  p.add_argument("--"+name,required=True)
 run(p.parse_args())
