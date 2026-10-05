"""Minimal 12-action chunk extension of the two-demonstration SmallBC baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_libero_bc import ROOT, SmallBC, load_data


class SmallChunkBC(SmallBC):
    def __init__(self, chunk_size=12):
        super().__init__()
        self.chunk_size = chunk_size
        self.head[-1] = nn.Linear(64, 7 * chunk_size)

    def forward(self, images, states):
        return super().forward(images, states).reshape(-1, self.chunk_size, 7)


def chunk_targets(actions, lengths, horizon):
    """Mask episode-end padding; never train targets across episode boundaries."""
    targets, masks = [], []
    start = 0
    for length in lengths:
        indices = torch.arange(length)[:, None] + torch.arange(horizon)[None, :]
        masks.append(indices < length)
        targets.append(actions[start + indices.clamp_max(length - 1)])
        start += length
    assert start == len(actions)
    return torch.cat(targets), torch.cat(masks)


@torch.no_grad()
def measure(model, images, states, targets, mask, weight):
    pred = torch.cat([model.predict(images[i:i+256], states[i:i+256]) for i in range(0, len(images), 256)])
    residual = pred - targets
    standardized = residual / model.action_scale
    weights = torch.tensor([1.] * 6 + [weight])
    metrics = dict(normalized_mse=float(standardized[mask].square().mean()),
                   weighted_loss=float((standardized.square()*weights)[mask].mean()),
                   action_mse=float(residual[mask].square().mean()),
                   first_action_mse=float(residual[:, 0].square().mean()),
                   gripper_sign_accuracy=float(((pred[:,:,6]>=0)==(targets[:,:,6]>=0))[mask].float().mean()))
    return metrics, pred


def gripper_audit(actions, predictions, provenance, horizon):
    episodes = []
    offset = 0
    for item in provenance:
        length = item['transitions']
        a = actions[offset:offset+length, 6].numpy()
        p = predictions[offset:offset+length, :, 6].numpy()
        switches = (np.flatnonzero(np.diff(a)!=0)+1).tolist()
        closing = [t for t in switches if a[t]>0]
        windows = []
        for event in closing:
            rows = []
            for t in range(event-horizon+1, event+5):
                rows.append(dict(observation_t=t, demonstrated_current=float(a[t]),
                                 predicted_chunk=p[t].tolist(),
                                 demonstrated_chunk=a[t:t+horizon].tolist(),
                                 predicts_any_close=bool((p[t]>0).any())))
            windows.append(dict(close_transition=event, rows=rows,
                                immediate_preclose_predicts_close=bool((p[event-1]>0).any()),
                                close_on_transition_first_action=bool(p[event,0]>0)))
        bounds = [0]+switches+[length]
        episodes.append(dict(**item, open_count=int((a<0).sum()), closed_count=int((a>0).sum()),
                             runs=[dict(start=s, stop=e, command=float(a[s]), steps=e-s, seconds=(e-s)*.05) for s,e in zip(bounds[:-1],bounds[1:])], grasp_windows=windows))
        offset += length
    return dict(open_count=int((actions[:,6]<0).sum()), closed_count=int((actions[:,6]>0).sum()),
                open_fraction=float((actions[:,6]<0).float().mean()),
                note='-1 open, +1 close. Windows align to commanded closing onset, not independently measured physical grasp.', episodes=episodes)


def train(paths, output, epochs=40, chunk_size=12, gripper_weight=1.):
    if not 8 <= chunk_size <= 16 or epochs < 1 or gripper_weight <= 0:
        raise ValueError('Use chunk size 8–16, positive epochs, and positive gripper weight')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'policy.pt').exists():
        raise ValueError('Refusing to overwrite a policy')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    images, states, actions, provenance = load_data(paths)
    targets, mask = chunk_targets(actions, [p['transitions'] for p in provenance], chunk_size)
    model = SmallChunkBC(chunk_size)
    model.state_mean.copy_(states.mean(0)); model.state_scale.copy_(states.std(0,unbiased=False).clamp_min(.01))
    model.action_mean.copy_(actions.mean(0)); model.action_scale.copy_(actions.std(0,unbiased=False).clamp_min(.001))
    normalized = (targets-model.action_mean)/model.action_scale
    weights = torch.tensor([1.]*6+[gripper_weight])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        for batch in torch.randperm(len(actions)).split(128):
            error = (model(images[batch],states[batch])-normalized[batch]).square()*weights
            loss = error[mask[batch]].mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        metrics, predictions = measure(model,images,states,targets,mask,gripper_weight)
        history.append(dict(epoch=epoch, **metrics))
        print(json.dumps(history[-1]),flush=True)
        if metrics['normalized_mse'] < .025:
            break
    torch.save(dict(state_dict=model.state_dict(), architecture='SmallChunkBC-v1',chunk_size=chunk_size),output/'policy.pt')
    np.savez_compressed(output/'training_predictions.npz', predicted=predictions.numpy(),target=targets.numpy(),valid=mask.numpy())
    report = dict(training_episodes=provenance,chunk_size=chunk_size,execute_steps=4,gripper_weight=gripper_weight,
                  seed=0,parameters=sum(p.numel() for p in model.parameters()),epochs=epoch,
                  objective='Masked per-dimension standardized action MSE, with optional gripper multiplier',
                  optimizer='Adam lr=.001 batch=128',final=metrics,history=history)
    (output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'gripper_audit.json').write_text(json.dumps(gripper_audit(actions,predictions,provenance,chunk_size),indent=2)+'\n')
    # Verify serialized inference against predictions before any rollout.
    saved=torch.load(output/'policy.pt',map_location='cpu',weights_only=True)
    restored=SmallChunkBC(saved['chunk_size']); restored.load_state_dict(saved['state_dict']); restored.eval()
    with torch.no_grad():
        actual=restored.predict(images[:16],states[:16])
    torch.testing.assert_close(actual,predictions[:16])
    (output/'checkpoint_validation.json').write_text(json.dumps(dict(passed=True,max_absolute_error=float((actual-predictions[:16]).abs().max())),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=Path,nargs='+',default=[ROOT/f'results/libero_robot_dataset/episodes/episode_{i:03d}.h5' for i in (1,2)])
    parser.add_argument('--output',type=Path,default=ROOT/'results/libero_chunk_bc')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--chunk-size',type=int,default=12)
    parser.add_argument('--gripper-weight',type=float,default=1.)
    args=parser.parse_args()
    train(args.episodes,args.output,args.epochs,args.chunk_size,args.gripper_weight)
