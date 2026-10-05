"""GPU-only short pretrained SmolVLA adaptation; never silently falls back to CPU."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from prepare_libero_smolvla import IMAGE, STATE


class DemoDataset(Dataset):
    def __init__(self, root, horizon=50):
        self.root=Path(root); self.horizon=horizon
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.episodes=[{k:np.load(self.root/e['directory']/(k+'.npy'),mmap_mode='r')
                        for k in (IMAGE,STATE,'action')} for e in self.manifest['episodes']]
        self.ends=np.cumsum([e['length'] for e in self.manifest['episodes']])

    def __len__(self): return int(self.ends[-1])

    def __getitem__(self,index):
        if not 0 <= index < len(self): raise IndexError(index)
        ep=int(np.searchsorted(self.ends,index,side='right'))
        t=index-(int(self.ends[ep-1]) if ep else 0)
        e=self.episodes[ep]; n=len(e['action']); indices=np.arange(t,t+self.horizon)
        return {IMAGE:torch.from_numpy(np.array(e[IMAGE][t])).permute(2,0,1).float()/255,
                STATE:torch.from_numpy(np.array(e[STATE][t])),
                'action':torch.from_numpy(np.array(e['action'][indices.clip(max=n-1)])),
                'action_is_pad':torch.from_numpy(indices>=n),
                'task':self.manifest['episodes'][ep]['task']}

    def stats(self):
        result={}
        for k,floor in [(STATE,.01),('action',.001)]:
            a=np.concatenate([e[k] for e in self.episodes])
            result[k]={'mean':torch.tensor(a.mean(0)), 'std':torch.tensor(a.std(0).clip(min=floor))}
        return result

    def probe_indices(self):
        indices=[]; offset=0
        for e in self.episodes:
            event=int(np.flatnonzero(np.diff(e['action'][:,6])>0)[0]+1)
            indices.extend(offset+t for t in (0,event-1,event,event+9))
            offset+=len(e['action'])
        return indices


def train(args):
    # Check before LeRobot imports, downloads, output creation, or any optimizer work.
    if not torch.cuda.is_available():
        raise SystemExit('No usable CUDA GPU. Training intentionally skipped; use the prepared GPU handoff.')
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from huggingface_hub import model_info
    if args.output.exists(): raise ValueError('Refusing to repeat/overwrite a training run')
    torch.manual_seed(0); np.random.seed(0)
    dataset=DemoDataset(args.dataset)
    revision=model_info('lerobot/smolvla_base').sha
    config=SmolVLAConfig.from_pretrained('lerobot/smolvla_base',revision=revision)
    config.input_features={IMAGE:PolicyFeature(type=FeatureType.VISUAL,shape=(3,128,128)),
                           STATE:PolicyFeature(type=FeatureType.STATE,shape=(18,))}
    config.output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))}
    config.device='cuda'; config.push_to_hub=False; config.n_action_steps=4
    config.freeze_vision_encoder=True; config.train_expert_only=True; config.train_state_proj=True
    # Full SmolVLA checkpoint supplies all pretrained weights, including the VLM.
    config.load_vlm_weights=False
    dataset.horizon=config.chunk_size
    policy=SmolVLAPolicy.from_pretrained('lerobot/smolvla_base',revision=revision,config=config,strict=True)
    pre,post=make_smolvla_pre_post_processors(config,dataset.stats())
    trainable=[p for p in policy.parameters() if p.requires_grad]
    assert trainable and any(not p.requires_grad for p in policy.parameters())
    optimizer=torch.optim.AdamW(trainable,lr=1e-4,betas=(.9,.95),weight_decay=1e-10)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
    probe_indices=dataset.probe_indices()
    probe=torch.utils.data.default_collate([dataset[i] for i in probe_indices])
    args.output.mkdir(parents=True)
    stats={k:{n:v.tolist() for n,v in s.items()} for k,s in dataset.stats().items()}
    (args.output/'dataset_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    (args.output/'source_manifest.json').write_text(json.dumps(dataset.manifest,indent=2)+'\n')

    @torch.no_grad()
    def measure():
        policy.eval(); policy.reset()
        # Same noise/time for small training probes; not held-out or simulator evaluation.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(123)
            batch=pre({k:v.clone() if torch.is_tensor(v) else list(v) for k,v in probe.items()})
            loss,_=policy(batch)
            policy.reset()
            predicted=post(policy.predict_action_chunk(batch)).cpu()
        target=probe['action']; valid=~probe['action_is_pad']
        return dict(probe_flow_loss=float(loss),probe_action_mse=float((predicted-target)[valid].square().mean()),
            gripper=[dict(index=i,demonstrated=target[j,:4,6].tolist(),predicted=predicted[j,:4,6].tolist()) for j,i in enumerate(probe_indices)])

    initial=measure(); history=[dict(step=0,**initial)]; start=time.monotonic(); iterator=iter(loader)
    reason='step budget'; step=0
    for step in range(1,args.steps+1):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader); batch=next(iterator)
        policy.train(); loss,details=policy(pre(batch))
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; stopping without evaluation')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,10.);optimizer.step()
        record=dict(step=step,training_flow_loss=float(loss.detach()))
        if step%25==0 or step==args.steps:
            record.update(measure()); print(json.dumps(record),flush=True)
        history.append(record)
        with (args.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        if step>=50 and 'probe_action_mse' in record and record['probe_flow_loss']<.7*initial['probe_flow_loss'] and record['probe_action_mse']<.7*initial['probe_action_mse']:
            reason='Both fixed training-probe losses improved by >=30%';break
        if time.monotonic()-start>=args.minutes*60:
            reason='wall-time budget';break
    final=measure(); policy.save_pretrained(args.output/'pretrained_model')
    report=dict(status='trained',pretrained='lerobot/smolvla_base',revision=revision,device=torch.cuda.get_device_name(),
                steps=step,batch_size=args.batch_size,seed=0,stop_reason=reason,seconds=time.monotonic()-start,
                trainable_parameters=sum(p.numel() for p in trainable),initial=initial,final=final,
                note='Flow-matching objective and fixed training-probe decoded action MSE; not validation or task success.')
    (args.output/'training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('results/libero_smolvla/dataset'))
    parser.add_argument('--output',type=Path,default=Path('results/libero_smolvla/training'))
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--minutes',type=float,default=15)
    args=parser.parse_args()
    if not 1<=args.steps<=200 or not 0<args.minutes<=15 or args.batch_size<1:
        parser.error('Rapid test limits: 1–200 updates, at most 15 minutes, positive batch size')
    train(args)
