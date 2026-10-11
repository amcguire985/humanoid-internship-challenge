import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_matched_b import validate_c,compare,ALGORITHM

class MatchedBTests(unittest.TestCase):
    def fixtures(self,root):
        for name in ('B','C'):
            d=root/name;e=d/'ground_truth/episode_000';e.mkdir(parents=True)
            summary=dict(condition=name,libero_success=True,placement_success=True,controller_completed=True,
                max_tilt_deg=13.,max_acceleration_m_s2=.6,transport_duration_s=1.,grasp_lost=False,failure_reason=None)
            manifest=dict(condition=name,transport_mode='phone_constraints' if name=='C' else 'trajectory',
                guidance_algorithm=ALGORITHM,phone_constraints={'acceleration_limit_m_s2':1.7},
                checkpoint_sha256={'model':'same'},settings={},placement_settings={},policy_instruction_transport='same')
            init=dict(seed=0,init_state_index=0,init_state_sha256='same',bddl_sha256='same',local_opening_axis=[0,0,1])
            for path,value in [(d/'summary.json',summary),(d/'experiment.json',manifest),(e/'initialization.json',init),(e/'controller.json',{'bounds':'same'})]:path.write_text(json.dumps(value))
            (e/'trajectory.jsonl').write_text(json.dumps({'constraints_active':False,'guidance_active':False})+'\n')
        return root/'C',root/'B'

    def test_comparison_uses_saved_C_and_separate_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,b=self.fixtures(root);compare(c,b,root)
            self.assertTrue((root/'comparison.md').exists())
            self.assertTrue(json.loads((root/'comparison.json').read_text())['matching_verified'])

    def test_failed_C_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,b=self.fixtures(Path(tmp));s=json.loads((c/'summary.json').read_text());s['controller_completed']=False
            (c/'summary.json').write_text(json.dumps(s))
            with self.assertRaises(ValueError):validate_c(c)

    def test_mismatched_state_and_unexpected_B_constraints_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,b=self.fixtures(root);p=b/'ground_truth/episode_000/initialization.json'
            s=json.loads(p.read_text());s['init_state_sha256']='different';p.write_text(json.dumps(s))
            with self.assertRaises(ValueError):compare(c,b,root)
            s['init_state_sha256']='same';p.write_text(json.dumps(s))
            (b/'ground_truth/episode_000/trajectory.jsonl').write_text(json.dumps({'constraints_active':True}))
            with self.assertRaises(ValueError):compare(c,b,root)
