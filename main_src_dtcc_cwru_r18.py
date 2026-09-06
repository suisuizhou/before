#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete DtCC source training on balanced CWRU with ResNet18."""
from __future__ import annotations
from pathlib import Path
import time
import hydra, omegaconf, torch
from omegaconf import open_dict
from torch.utils.data import DataLoader
import Dataset
from Dataset.CWRU import CWRU as CWRUDataset
Dataset.CWRU = CWRUDataset
from Lib.model import get_model
from Lib.protocol_b_common import method_checkpoint_dir, save_training_metadata, set_optimizer_lr
from Lib.protocol_b_project import build_model_name, parse_seed_runs
from Lib.protocol_b_source import evaluate_classifier, train_dtcc_source_epoch
from Lib.train_utils import seed_torch


def _sources(cfg):
    only=getattr(cfg,"only_source",None)
    return [int(only)] if only is not None else [0,1,2,3]


def force_cwru_cfg(cfg):
    with open_dict(cfg):
        cfg.Dataset.data_name = "CWRU"
        cfg.Dataset.TL_list = [0,1,2,3]
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "mean-std"
        cfg.Model.bottleneck_num = 128
        cfg.Model.use_spectral_adapter = False
        cfg.batch_size = int(getattr(cfg,"batch_size",128))
        cfg.num_workers = int(getattr(cfg,"num_workers",4))
        cfg.dtcc_src_epoch = 50
        cfg.label_smoothing = 0.1
        cfg.Opt.lr_src = 1e-3
        cfg.Opt.weight_decay_src = 1e-4
        if not hasattr(cfg,"ProtocolB"): cfg.ProtocolB = {}
        cfg.ProtocolB.checkpoint_root = str(getattr(cfg.ProtocolB,"checkpoint_root","TTA_Model_CWRU"))


def train_one(cfg, source:int, seed:int):
    seed_torch(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fallback=next(x for x in [0,1,2,3] if x!=source)
    with open_dict(cfg):
        cfg.seed_run=int(seed); cfg.Dataset.TL_Task=(int(source),int(fallback)); cfg.model_name=build_model_name(cfg)
    source_data,_=CWRUDataset(**cfg.Dataset).data_generator(); num_classes=10
    train_loader=DataLoader(source_data,batch_size=int(cfg.batch_size),shuffle=True,num_workers=int(cfg.num_workers),pin_memory=device.type=="cuda",drop_last=True)
    eval_loader=DataLoader(source_data,batch_size=int(cfg.batch_size),shuffle=False,num_workers=int(cfg.num_workers),pin_memory=device.type=="cuda",drop_last=False)
    model=get_model(num_classes=num_classes,cfg=cfg,**cfg.Model).to(device)
    initial_lr=float(cfg.Opt.lr_src)
    optimizer=torch.optim.AdamW(model.parameters(),lr=initial_lr,weight_decay=float(cfg.Opt.weight_decay_src))
    epochs=int(cfg.dtcc_src_epoch); epsilon=float(cfg.label_smoothing)
    out_dir=method_checkpoint_dir(Path(str(cfg.ProtocolB.checkpoint_root)),"DTCC","CWRU",source,seed); out_dir.mkdir(parents=True,exist_ok=True)
    final_path=out_dir/cfg.model_name; best_path=out_dir/f"best_source_{cfg.model_name}"
    best_acc=-1.0; best_epoch=-1; started=time.time()
    print(f"[CWRU DTCC SOURCE] source={source} seed={seed} epochs={epochs} batch={cfg.batch_size} epsilon={epsilon}")
    for epoch in range(1,epochs+1):
        progress=0.0 if epochs<=1 else (epoch-1)/(epochs-1); lr=set_optimizer_lr(optimizer,initial_lr,progress)
        tr=train_dtcc_source_epoch(model,train_loader,optimizer,device,num_classes,epsilon)
        ev=evaluate_classifier(model,eval_loader,device,num_classes)
        print(f"Epoch [{epoch}/{epochs}] train_loss={tr['loss']:.6f} train_acc={tr['accuracy']:.2f}% source_acc={ev['accuracy']:.2f}% source_f1={ev['macro_f1']:.2f}% lr={lr:.8f}")
        if ev['accuracy']>=best_acc:
            best_acc=ev['accuracy']; best_epoch=epoch; torch.save(model.state_dict(),best_path); print(f"[SAVE] {best_path}")
    torch.save(model.state_dict(),final_path)
    save_training_metadata(out_dir/"source_training_summary.json",{"protocol":"CWRU","method":"DTCC","source":source,"seed":seed,"epochs":epochs,"best_epoch":best_epoch,"best_source_accuracy":best_acc,"elapsed_seconds":time.time()-started,"checkpoint":str(best_path),"selection_uses_target_labels":False})


@hydra.main(version_base=None,config_path="./Configs",config_name="defaults")
def run(cfg:omegaconf.DictConfig):
    force_cwru_cfg(cfg)
    for seed in parse_seed_runs(getattr(cfg,"seed_runs",None)):
        for source in _sources(cfg): train_one(cfg,source,seed)

if __name__=="__main__": run()
