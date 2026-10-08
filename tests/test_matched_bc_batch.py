import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_matched_bc_batch import reduction,summarize,reports,patch_selection

class BatchTests(unittest.TestCase):
    def test_unavailable_and_zero(self):
        self.assertIsNone(reduction(None,1));self.assertIsNone(reduction(0,1))
        self.assertEqual(reduction(2,1),50);self.assertEqual(reduction(1,2),-100)
    def test_failures_retained(self):
        rows=[{'condition':'B','libero_success':True,'max_acceleration_m_s2':2}, {'condition':'B','libero_success':False,'max_acceleration_m_s2':None}, {'condition':'C','libero_success':None,'max_acceleration_m_s2':None}]
        s=summarize(rows)
        self.assertEqual(s['B']['task_success_rate_all_scheduled'],.5)
        self.assertEqual(s['B']['max_acceleration_m_s2']['mean'],2)
        self.assertIsNone(s['C']['max_acceleration_m_s2']['mean'])
    def test_reports_missing_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[dict(initial_state_index=i,condition=c,output=str(root/'absent'),libero_success=False,grasp_success=False,placement_success=False,controller_completed=False,max_tilt_deg=None,max_acceleration_m_s2=None,transport_duration_s=None,grasp_lost=False,failure_reason='no grasp') for i in range(1,6) for c in 'BC']
            reports(root,rows)
            self.assertEqual(len((root/'rollouts.csv').read_text().splitlines()),11)
            self.assertTrue((root/'paired_acceleration.svg').exists())
            self.assertTrue((root/'rollout_table.png').exists())
    def test_selection_adapter_only(self):
        import subprocess,tarfile
        from run_matched_b import FROZEN_REVISION
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); archive=root/'source.tar'
            with archive.open('wb') as f:subprocess.run(['git','archive',FROZEN_REVISION],stdout=f,check=True)
            with tarfile.open(archive) as t:t.extractall(root,filter='data')
            controller=(root/'scripts/bowl_placement.py').read_bytes()
            patch_selection(root)
            self.assertEqual(controller,(root/'scripts/bowl_placement.py').read_bytes())
            compile((root/'scripts/evaluate_bowl_hybrid.py').read_text(),'hybrid','exec')
            compile((root/'scripts/evaluate_bowl_liquid.py').read_text(),'liquid','exec')

if __name__=='__main__':unittest.main()
