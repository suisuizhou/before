"""Source training for the HUST gearbox benchmark (ordinary and robust routes)."""
from __future__ import annotations
import json, random, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from Lib.model import get_model
from Lib.pu4d_common_source import forward_parts
from Lib.train_utils import seed_torch
from Lib.wtpg_source_training import _forward_robust
from Dataset.HUSTGearboxStrict import HUSTGearboxStrict

def train(root="Dataset/HUST_GEARBOX_STRICT_CACHE_V1", out="TTA_Model_HUST_GEARBOX_V1", variant="ordinary", epochs=50, device_id=0, source_ids=None, force=False):
    seed_torch(2025); random.seed(2025)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Strict DtCC source architecture: modified 1-D ResNet-18 with bottleneck
    # classifier, without the project-specific spectral adapter (PR=0).
    cfg=type("Cfg",(),{})(); cfg.Model=dict(model_name="ResNet18_1D_SDE",use_spectral_adapter=False,band_num=256,input_len=512,bottleneck=True,bottleneck_num=128,model_type="linear")
    for source in (range(6) if source_ids is None else [int(x) for x in source_ids]):
        path=Path(out)/variant/f"source_{source}"/"seed_2025"/"ResNet18_1D_SDE2025fft_Linear.pt"
        if path.exists() and not force: print("exists",path); continue
        ds=HUSTGearboxStrict(root, (source,(source+1)%6))._load(source)
        loader=DataLoader(ds,batch_size=128,shuffle=True,num_workers=2)
        model=get_model(model_name="ResNet18_1D_SDE",num_classes=3,cfg=cfg,**{k:v for k,v in cfg.Model.items() if k!="model_name"}).to(device)
        # keep the 0711 carriers identity during source training
        for n,p in model.named_parameters():
            if any(k in n for k in ("band_scale","band_bias","warp_ctrl")): p.requires_grad=False
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
        start=time.monotonic(); seen=correct=0
        for ep in range(epochs):
            model.train()
            for x,y,_ in loader:
                x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
                if variant=="robust":
                    # Lightweight class-preserving spectral perturbation for the initial
                    # 0711 source route; avoids consuming target labels.
                    _, logits = forward_parts(model, x)
                    x_aug = x + 0.015 * torch.randn_like(x)
                    _, logits_aug = forward_parts(model, x_aug)
                    loss=(torch.nn.functional.cross_entropy(logits,y)+torch.nn.functional.cross_entropy(logits_aug,y))*0.5
                else:
                    _,logits=forward_parts(model,x); loss=torch.nn.functional.cross_entropy(logits,y)
                loss.backward(); opt.step(); seen+=y.numel(); correct+=(logits.argmax(1)==y).sum().item()
        path.parent.mkdir(parents=True,exist_ok=True)
        payload={"contract":{"version":1,"dataset":"HUSTGearboxStrict","route":variant,"source":source,"seed":2025,"epoch":epochs},"state_dict":{n:p.detach().cpu() for n,p in model.state_dict().items()}}
        torch.save(payload,path)
        meta={"dataset":"HUSTGearboxStrict","route":variant,"source":source,"seed":2025,"epochs":epochs,"source_accuracy":100.0*correct/max(seen,1),"checkpoint":path.name,"target_labels_consumed":False,"robust_profile":"generic_v1" if variant=="robust" else "not_applicable"}
        (path.parent/"source_training_summary.json").write_text(json.dumps(meta,indent=2))
        print(f"SOURCE {variant} {source}: {meta['source_accuracy']:.2f}%",flush=True)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("variant",choices=["ordinary","robust"]); ap.add_argument("--source",type=int); ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--out",default="TTA_Model_HUST_GEARBOX_V1"); ap.add_argument("--force",action="store_true"); args=ap.parse_args(); train(variant=args.variant, out=args.out, epochs=args.epochs, force=args.force, source_ids=None if args.source is None else [args.source])
