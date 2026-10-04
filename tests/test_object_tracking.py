import sys
import unittest
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from track_apriltags import object_pose

class ObjectTests(unittest.TestCase):
    def test_detector_corner_order(self):
        from track_apriltags import Standard41Detector
        from pupil_apriltags import Detector
        adapter=Standard41Detector()
        adapter.detector=Detector(families="tag25h9",quad_decimate=1)
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
        image=np.full((400,400),255,np.uint8)
        image[50:350,50:350]=cv2.aruco.generateImageMarker(dictionary,0,300)
        for turns in range(4):
            frame=np.ascontiguousarray(np.rot90(image,turns))
            expected,_,_=cv2.aruco.ArucoDetector(dictionary).detectMarkers(frame)
            actual,ids,_=adapter.detectMarkers(frame)
            self.assertEqual(ids[0,0],0)
            np.testing.assert_allclose(actual[0],expected[0],atol=2)

    def test_rigid_carry_and_reacquisition(self):
        world=np.eye(4)
        hand=np.eye(4); hand[:3,3]=[.1,.2,.3]
        offset=np.eye(4); offset[:3,3]=[.06,0,0]
        offset[:3,:3]=cv2.Rodrigues(np.array([.2,0,0]))[0]
        obj=hand@offset
        pose,status,cache=object_pose(world,hand,obj,None,True)
        self.assertEqual(status,'tracked')
        np.testing.assert_allclose(cache,offset,atol=1e-12)
        hand[:3,:3]=cv2.Rodrigues(np.array([0,0,1.]))[0]
        pose,status,cache=object_pose(world,hand,None,cache,False)
        self.assertEqual(status,'inferred_hand')
        np.testing.assert_allclose(pose,hand@offset,atol=1e-12)
        pose,status,cache=object_pose(world,hand,obj,cache,True)
        self.assertEqual(status,'tracked')
        np.testing.assert_allclose(pose,obj)

    def test_requires_initialization_current_world_and_hand(self):
        eye=np.eye(4)
        for world,hand,cache in [(eye,eye,None),(None,eye,eye),(eye,None,eye)]:
            self.assertIsNone(object_pose(world,hand,None,cache,False)[0])
        pose,status,cache=object_pose(eye,eye,None,eye,True)
        self.assertIsNone(pose)
        self.assertIsNone(cache)

if __name__=='__main__': unittest.main()
