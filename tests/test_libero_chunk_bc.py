"""Chunk labels must preserve observation/action alignment and episode boundaries."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_libero_chunk_bc import chunk_targets, SmallChunkBC


class ChunkTests(unittest.TestCase):
    def test_episode_boundaries_and_mask(self):
        actions = torch.arange(5.).view(5, 1).repeat(1, 7)
        targets, mask = chunk_targets(actions, [3, 2], 3)
        self.assertEqual(targets[:, :, 0].tolist(), [[0,1,2],[1,2,2],[2,2,2],[3,4,4],[4,4,4]])
        self.assertEqual(mask.tolist(), [[True,True,True],[True,True,False],[True,False,False],[True,True,False],[True,False,False]])
        self.assertEqual(int(mask.sum()), 9)

    def test_chunk_shape_and_normalization(self):
        model = SmallChunkBC(12)
        model.action_mean.fill_(2.)
        model.action_scale.fill_(3.)
        images = torch.zeros(2, 3, 64, 64, dtype=torch.uint8)
        states = torch.zeros(2, 18)
        self.assertEqual(model(images, states).shape, (2,12,7))
        torch.testing.assert_close(model.predict(images,states), model(images,states)*3+2)


if __name__ == '__main__':
    unittest.main()
