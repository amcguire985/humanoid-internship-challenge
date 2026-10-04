"""Task extraction: goal-frame geometry, missing data, phases and overrides."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from process_task_demo import (validate_config, compute_goal_position, clean_trajectory,
                               segment_task_phases, compute_task_metrics, export_processed_demo)


class TaskDemoTests(unittest.TestCase):
    def test_rotated_goal_offset_and_rejected_observation(self):
        row=dict(status='tracked',x_m='1',y_m='2',z_m='3',error_px='1',
                 relative_rx_rad='0',relative_ry_rad='0',relative_rz_rad=str(np.pi/2))
        goal,info=compute_goal_position([row,dict(row,error_px='9',x_m='100')],[.2,0,.1])
        np.testing.assert_allclose(goal,[1,2.2,3.1],atol=1e-12)
        self.assertEqual(info['accepted_samples'],1)
        with self.assertRaises(ValueError): compute_goal_position([], [0,0,0])

    def test_coplanar_goal_projects_tilt_and_locks_before_pickup(self):
        import cv2
        yaw=.5
        rz=cv2.Rodrigues(np.array([0.,0.,yaw]))[0]
        tilt=cv2.Rodrigues(np.array([.2,0.,0.]))[0]
        rv=cv2.Rodrigues(rz@tilt)[0].ravel()
        row=dict(status='tracked',time_s='0',x_m='1',y_m='2',z_m='.08',error_px='1',
                 **{'relative_r'+a+'_rad':str(rv[j]) for j,a in enumerate('xyz')})
        # A later displaced goal must not change the pre-pickup anchor.
        later=dict(row,time_s='3',x_m='10')
        goal,info=compute_goal_position([row,later],[-.09,.09,0],coplanar=True,
                                       center_height=.0225,anchor_before=1.)
        expected=np.array([1.,2.,0.])+rz@np.array([-.09,.09,0.])
        np.testing.assert_allclose(info['goal_area_center'],expected,atol=1e-12)
        np.testing.assert_allclose(goal,expected+[0,0,.0225],atol=1e-12)
        self.assertEqual(info['accepted_samples'],1)
        self.assertAlmostEqual(info['raw_plane_z_median_m'],.08)

    def test_circle_membership_ignores_height_but_reports_it(self):
        c=validate_config(dict(goal_coplanar=True,goal_radius=.06))
        t=np.arange(3)*.1; goal=np.array([0.,0.,.0225])
        bounds=dict(pickup_time=0.,transport_start=0.,transport_end=.2,release_time=.2)
        p=np.tile([.059,0.,.2],(3,1))
        m=compute_task_metrics(t,p,goal,np.zeros(3),bounds,c)
        self.assertTrue(m['final_inside_goal_region'])
        self.assertAlmostEqual(m['final_planar_placement_error'],.059)
        self.assertAlmostEqual(m['final_goal_height_error'],.1775)
        p[:,0]=.061
        m=compute_task_metrics(t,p,goal,np.zeros(3),bounds,c)
        self.assertFalse(m['final_inside_goal_region'])
        self.assertAlmostEqual(m['final_distance_outside_goal_region'],.001)

    def test_cleaning_rejects_spike_fills_only_short_gaps(self):
        t=np.arange(30)*.03; raw=np.zeros((30,3)); raw[4,0]=1
        raw[10:12]=np.nan; raw[18:25]=np.nan
        raw[27,0]=.1
        errors=np.ones(30); errors[27]=10
        clean,p,source,rejected,reasons,gaps=clean_trajectory(t,raw,errors,validate_config({}))
        self.assertTrue(rejected[4]); self.assertTrue(rejected[27])
        self.assertEqual(source[4],'interpolated'); self.assertEqual(source[10],'interpolated')
        self.assertTrue(np.isnan(p[18:25]).all())
        np.testing.assert_allclose(p[:18],0,atol=1e-12)
        self.assertTrue(np.isnan(clean[4]).all())
        self.assertEqual(raw[4,0],1)

    def demo(self):
        t=np.arange(0,5.01,.05); p=np.zeros((len(t),3))
        p[:,2]=np.where(t<1,0,np.where(t<2,.1*(t-1),np.where(t<3,.1,np.where(t<4,.1*(4-t),0))))
        p[:,0]=.1*np.clip(t-2,0,1)
        return t,p,np.array([.1,0,0])

    def test_phases_and_manual_override(self):
        t,p,goal=self.demo(); c=validate_config({})
        labels,speed,bounds,estimates,progress,notes=segment_task_phases(t,p,goal,c)
        self.assertAlmostEqual(bounds['pickup_time'],1,delta=.1)
        self.assertAlmostEqual(bounds['release_time'],4,delta=.1)
        self.assertTrue({'stationary_pre_pickup','pickup_lift','transport','placement_lowering','stationary_post_release'}<=set(labels))
        self.assertAlmostEqual(np.nanmin(progress),0)
        self.assertAlmostEqual(np.nanmax(progress),1)
        c['phase_overrides']={'pickup_time':.9,'transport_start':2.,'transport_end':3.,'release_time':4.1}
        _,_,manual,automatic,_,_=segment_task_phases(t,p,goal,c)
        self.assertEqual(manual['pickup_time'],.9)
        self.assertNotEqual(automatic['pickup_time'],.9)
        c['phase_overrides']['transport_end']=.5
        with self.assertRaises(ValueError): segment_task_phases(t,p,goal,c)

    def test_stationary_has_no_task_and_gap_does_not_add_path(self):
        t=np.arange(8)*.1; p=np.zeros((8,3)); c=validate_config({})
        labels,speed,bounds,_,_,_=segment_task_phases(t,p,np.zeros(3),c)
        self.assertIsNone(bounds['pickup_time']); self.assertEqual(set(labels),{'stationary'})
        p[3:5]=np.nan; p[5:,0]=10
        bounds=dict(pickup_time=0.,transport_start=0.,transport_end=.7,release_time=.7)
        m=compute_task_metrics(t,p,np.zeros(3),speed,bounds,c)
        self.assertEqual(m['transport_path_length'],0)
        self.assertFalse(m['transport_complete'])

    def test_video_annotations_snap_to_source_frames(self):
        t,p,goal=self.demo()
        c=validate_config(dict(phase_overrides=dict(transport_start=2.013,
                              transport_end=3.013,release_time=4.113)))
        labels,_,bounds,automatic,progress,_=segment_task_phases(t,p,goal,c)
        self.assertEqual(bounds['transport_start'],t[40])
        self.assertEqual(bounds['transport_end'],t[60])
        self.assertEqual(bounds['release_time'],t[82])
        self.assertEqual(progress[40],0.)
        self.assertEqual(progress[60],1.)
        self.assertEqual(labels[60],'transport')
        self.assertNotEqual(bounds['release_time'],automatic['release_time'])
        # Reject a bad recording-time annotation before snapping could hide it.
        c['phase_overrides']['release_time']=t[-1]+.001
        with self.assertRaises(ValueError): segment_task_phases(t,p,goal,c)
        keep=(t<=2.) | (t>=3.)
        c['phase_overrides']=dict(transport_start=2.5,transport_end=3.5,release_time=4.1)
        with self.assertRaises(ValueError): segment_task_phases(t[keep],p[keep],goal,c)

    def test_reviewed_test007_includes_slow_lowering_and_hold(self):
        import json
        root=Path(__file__).resolve().parents[1]
        c=validate_config(json.loads((root/'config/test_007_demo.json').read_text()))
        with (root/'results/test_007_block_only_raw/task_demo/processed_demo.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        t=np.array([float(r['time']) for r in rows])
        p=np.array([[float(r['object_'+a]) for a in 'xyz'] for r in rows])
        goal=np.array([float(rows[0]['goal_'+a]) for a in 'xyz'])
        labels,_,bounds,automatic,progress,_=segment_task_phases(t,p,goal,c)
        self.assertAlmostEqual(bounds['transport_start'],4.336666666666667)
        self.assertAlmostEqual(bounds['transport_end'],10.306666666666667)
        self.assertGreater(bounds['release_time'],bounds['transport_end'])
        self.assertLess(automatic['transport_end'],6.5)
        for time in (4.5,6.5,8.5,10.2):
            self.assertEqual(labels[np.argmin(abs(t-time))],'transport')
        self.assertEqual(np.isfinite(progress).sum(),180)
        self.assertEqual(np.nanmin(progress),0.)
        self.assertEqual(np.nanmax(progress),1.)

    def test_export_preserves_invalid_transport_rows(self):
        t,p,goal=self.demo(); p[45:50]=np.nan; c=validate_config({})
        c['phase_overrides']=dict(pickup_time=1.,transport_start=2.,transport_end=3.,release_time=4.)
        labels,speed,bounds,_,progress,_=segment_task_phases(t,p,goal,c)
        m=compute_task_metrics(t,p,goal,speed,bounds,c)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            export_processed_demo(output,t,p,p,p,goal,labels,speed,progress,
                                  np.where(np.isfinite(p).all(axis=1),'observed','missing'),
                                  np.zeros(len(t),dtype=bool),np.full(len(t),'',dtype=object),m,bounds)
            with (output/'transport.csv').open() as handle: rows=list(csv.DictReader(handle))
        missing=[r for r in rows if r['phase']=='invalid']
        self.assertEqual(len(missing),5)
        self.assertEqual(missing[0]['object_x'],'')
        self.assertEqual(missing[0]['valid_measurement'],'0')

    def test_configuration_rejects_invalid_and_unknown_settings(self):
        for config in ({'vertical_sign':0},{'smoothing_window':4},{'max_gap_frames':-1},
                       {'goal_object_offset':[1,2]},{'stationary_speed':0},{'typo':1}):
            with self.assertRaises(ValueError): validate_config(config)


if __name__=='__main__': unittest.main()
