import os
import cv2
import numpy as np
from enum import Enum

class DataObj(Enum):
    binary_ims_raw = 'binary_ims_raw'
    binary_ims_tf = 'binary_ims_tf'
    binary_thumbs = 'binary_thumbs'
    depth_ims_raw = 'depth_ims_raw'
    depth_ims_tf = 'depth_ims_tf'
    depth_ims_tf_table = 'depth_ims_tf_table'
    depth_thumbs = 'depth_thumbs'
    ferrari_canny = 'ferrari_canny'
    force_closure = 'force_closure'
    hand_poses = 'hand_poses'
    image_labels = 'image_labels'
    object_labels = 'object_labels'
    pose_labels = 'pose_labels'
    robust_ferrari_canny = 'robust_ferrari_canny'
    robust_force_closure = 'robust_force_closure'
    table_mask = 'table_mask'

objnames = ['binary_ims_raw', 'binary_ims_tf', 'binary_thumbs', 'depth_ims_raw', 'depth_ims_tf', 'depth_ims_tf_table', 'depth_thumbs', 'ferrari_canny', 'force_closure', 'hand_poses', 'image_labels', 'object_labels', 'pose_labels', 'robust_ferrari_canny', 'robust_force_closure', 'table_mask']

class DatasetSlice(object):

    def inspect(self, objs=objnames):
        """TODO: Docstring for ddInspectNpz.

        :file: TODO
        :returns: TODO

        """
        for obj in objs:
            print('== %s ==========================' % obj)
            file = self.getObjFile(obj)
            print('== file ' + file)
            content = self.getObj(obj)
            print('== dims %s' % str(content.shape))

    @staticmethod
    def npzExtractorDefault(file):
        """extractor that takes the first payload it finds in a numpy-loadable file

        :file: a numpy-parsable file
        :returns: the payload

        """
        return np.load(file).items()[0][1]

    def getObjFile(self, obj):
        pattern = '%s/%s_%s.npz'
        return pattern % (self.basepath, obj, format(self.sliceNr, '05d'))
    
    def getObj(self, obj):
        """TODO: Docstring for getObj.

        :ob: TODO
        :returns: TODO

        """
        return DatasetSlice.npzExtractorDefault(self.getObjFile(obj))

    def __init__(self, sliceNr, basepath='/home/simon/sandbox/graspitmod/catkin_ws/src/gqcnn/custom/data/adv_synth', maxIdx=189):
        if sliceNr < 0 or sliceNr > maxIdx:
            raise RuntimeError('invalid slice nr ' + sliceNr)
        self.basepath = basepath
        self.sliceNr = sliceNr
        self.maxIdx = maxIdx
        

def iiBrowse(images):
    """shows multiple images with browsing buttons jk, quitting with q
    :param List images: an array of images to browse
    :returns: the index where the browser was quit
    :rtype: int
    """
    global iidx, iiData
    keycode = None
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    iidx = 0
    iiData = images
    iiRotate(0)
    print('use q to exit, j and k to navigate forward and backward')
    while keycode != ord('q'):
        if keycode == ord('k'):
            iiRotate(-1)
        if keycode == ord( 'j' ):
            iiRotate(1)
        nrcodes = [ ord(str(i)) for i in range(0, 9) ]
        if keycode in nrcodes:
            nr = nrcodes.index(keycode)
            iiGo(nr * 100)
            
        keycode = cv2.waitKey(0)

    cv2.destroyAllWindows()
    return iidx

# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
def iiRotate(shift):
    """TODO: Docstring for iiRotate.

    :shift: TODO
    :returns: TODO

    """
    global iidx, iiData
    iidx = ( iidx + shift ) % len(iiData)
    print('showing image %d' % iidx)
    cv2.imshow('image', iiData[iidx])

def iiGo(idx):
    """TODO: Docstring for iiRotate.

    :shift: TODO
    :returns: TODO

    """
    global iidx, iiData
    iidx = idx
    print('showing image %d' % iidx)
    cv2.imshow('image', iiData[iidx])

def iShow(image):
    """simple cv2 viewer

    :image: TODO
    :returns: TODO

    """
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getEnv(name, predicates=[os.path.exists], failIfNot=True):
    """gets an env variable, applying some optional unary predicates. by default [os.path.exists] for a file argument

    :param String name: the environment variable
    :rtype: String or None

    """
    val = os.environ.get(name)
    if val is None:
        return val
    for pred in predicates:
        if not pred(val):
            if failIfNot:
                raise RuntimeError("predicate %s not met for env variable %s with value %s" % (str(pred), name, val))
            else:
                return None
    
    return val

