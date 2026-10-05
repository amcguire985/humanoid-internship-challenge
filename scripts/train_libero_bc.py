"""Small image + proprioception behavior-cloning sanity baseline; no RL."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from inspect_libero_dataset import validate_episode

ROOT = Path(__file__).resolve().parents[1]


class SmallBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 24, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 + 18, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7))
        self.register_buffer('state_mean', torch.zeros(18))
        self.register_buffer('state_scale', torch.ones(18))
        self.register_buffer('action_mean', torch.zeros(7))
        self.register_buffer('action_scale', torch.ones(7))

    def forward(self, images, states):
        features = self.image_encoder(images.float() / 255.)
        state = (states - self.state_mean) / self.state_scale
        return self.head(torch.cat([features, state], dim=1))

    def predict(self, images, states):
        return self(images, states) * self.action_scale + self.action_mean


def prepare_images(images):
    """Identical area downsampling for recorded and online RGB images."""
    tensor = torch.as_tensor(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()
    return F.interpolate(tensor, size=(64, 64), mode='area').round().to(torch.uint8)


def load_data(paths):
    images, states, actions, provenance = [], [], [], []
    for path in paths:
        report = validate_episode(path)
        with h5py.File(path, 'r') as f:
            images.append(prepare_images(f['observations/agentview_rgb'][:-1]))
            states.append(torch.from_numpy(f['observations/proprio'][:-1].astype(np.float32)))
            actions.append(torch.from_numpy(f['actions'][:].astype(np.float32)))
        provenance.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               episode_id=report['episode_id'], transitions=report['transitions']))
    return torch.cat(images), torch.cat(states), torch.cat(actions), provenance


@torch.no_grad()
def measure(model, images, states, actions):
    predictions = torch.cat([model.predict(images[i:i+256], states[i:i+256]) for i in range(0, len(actions), 256)])
    residual = predictions - actions
    return dict(normalized_mse=float(((residual / model.action_scale)**2).mean()),
                action_mse=float((residual**2).mean()),
                rmse_per_dimension=(residual.square().mean(0).sqrt()).tolist(),
                gripper_sign_accuracy=float(((predictions[:, 6] >= 0) == (actions[:, 6] >= 0)).float().mean())), predictions


def train(paths, output, epochs=40):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'policy.pt').exists():
        raise ValueError('Refusing to overwrite a trained policy')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    images, states, actions, provenance = load_data(paths)
    model = SmallBC()
    model.state_mean.copy_(states.mean(0))
    model.state_scale.copy_(states.std(0, unbiased=False).clamp_min(.01))
    model.action_mean.copy_(actions.mean(0))
    model.action_scale.copy_(actions.std(0, unbiased=False).clamp_min(.001))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = (actions - model.action_mean) / model.action_scale
    initial, _ = measure(model, images, states, actions)
    history = [dict(epoch=0, **initial)]
    print('Initial:', initial, flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(actions))
        total_loss = 0.
        for batch in indices.split(128):
            loss = F.mse_loss(model(images[batch], states[batch]), target[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        model.eval()
        metrics, predictions = measure(model, images, states, actions)
        history.append(dict(epoch=epoch, minibatch_mean_loss=total_loss/len(actions), **metrics))
        print(f'Epoch {epoch}: normalized MSE={metrics["normalized_mse"]:.6f}, action MSE={metrics["action_mse"]:.6f}, gripper accuracy={metrics["gripper_sign_accuracy"]:.4f}', flush=True)
        if metrics['normalized_mse'] < .025:
            break
    torch.save(dict(state_dict=model.state_dict(), architecture='SmallBC-v1', image_size=64,
                    state_dimension=18, action_dimension=7), output / 'policy.pt')
    np.savez_compressed(output / 'training_predictions.npz', target=actions.numpy(),
                        predicted=predictions.numpy(), episode_lengths=np.array([p['transitions'] for p in provenance]))
    report = dict(seed=0, training_episodes=provenance, transitions=len(actions),
                  parameters=sum(p.numel() for p in model.parameters()), device='cpu',
                  architecture='Conv(3,16,5,s2), Conv(16,24,3,s2), Conv(24,32,3,s2), FC(2048,64); concatenate normalized 18D proprio; MLP(82,128,64,7); ReLU',
                  objective='MSE in per-dimension standardized action space; scales floored at .001',
                  optimizer='Adam lr=0.001, batch=128', epochs=epoch, max_epochs=epochs,
                  early_stop_normalized_mse=.025, initial=initial, final=metrics,
                  language_used=False, eef_pose_used=False, timestep_or_phase_used=False,
                  evaluation_scope='Training-set action reproduction only; no held-out demonstration claim', history=history)
    (output / 'training.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].semilogy([h['epoch'] for h in history], [h['normalized_mse'] for h in history])
    axes[0].set(xlabel='Epoch', ylabel='Training normalized MSE')
    axes[1].plot(actions[:, 6], label='Demonstration gripper')
    axes[1].plot(predictions[:, 6], alpha=.65, label='Predicted gripper')
    axes[1].set(xlabel='Concatenated training transition', ylabel='Action'); axes[1].legend()
    fig.tight_layout(); fig.savefig(output / 'training.png'); plt.close(fig)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', nargs='+', type=Path, default=[ROOT/f'results/libero_robot_dataset/episodes/episode_{i:03d}.h5' for i in (1, 2)])
    parser.add_argument('--output', type=Path, default=ROOT/'results/libero_bc_baseline')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    train(args.episodes, args.output, args.epochs)
