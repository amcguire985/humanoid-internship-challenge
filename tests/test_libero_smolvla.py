"""Local tests require only the existing LIBERO environment, not SmolVLA weights."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_libero_smolvla import DemoDataset
from prepare_libero_smolvla import ROOT,IMAGE,STATE


class SmolVLADatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=DemoDataset(ROOT/'results/libero_smolvla/dataset')

    def test_boundary_padding_never_crosses_episodes(self):
        end=self.data[2108];start=self.data[2109]
        self.assertEqual(int((~end['action_is_pad']).sum()),1)
        self.assertTrue(torch.equal(end['action'],end['action'][0].expand(50,7)))
        np.testing.assert_array_equal(start[STATE].numpy(),self.data.episodes[1][STATE][0])
        np.testing.assert_array_equal(start['action'][0].numpy(),self.data.episodes[1]['action'][0])
        self.assertEqual(int((~self.data[4229]['action_is_pad']).sum()),1)

    def test_image_and_grasp_alignment(self):
        sample=self.data[839]
        self.assertEqual(sample[IMAGE].shape,(3,128,128))
        self.assertEqual(sample['action'][0,6].item(),-1)
        self.assertEqual(sample['action'][1,6].item(),1)
        self.assertTrue(sample['task'].startswith('pick the akita black bowl'))
        np.testing.assert_array_equal((sample[IMAGE].permute(1,2,0).numpy()*255).round().astype('uint8'),self.data.episodes[0][IMAGE][839])
        self.assertEqual(len(self.data),4230)

    def test_normalization_and_batch(self):
        stats=self.data.stats()
        self.assertTrue(all(torch.isfinite(s['std']).all() and (s['std']>0).all() for s in stats.values()))
        batch=torch.utils.data.default_collate([self.data[0],self.data[2109]])
        self.assertEqual(batch['action'].shape,(2,50,7))
        self.assertEqual(batch[STATE].shape,(2,18))
        self.assertEqual(batch['action_is_pad'].dtype,torch.bool)


if __name__=='__main__': unittest.main()
