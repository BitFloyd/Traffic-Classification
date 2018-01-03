import numpy as np
import scipy.io as spio
import os
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from skimage.io import imsave

def get_frames_from_vid(video_name, category_id):
    # Initialize FFMPEG_VideoReader
    fvr = FFMPEG_VideoReader(filename=video_name)
    fvr.initialize()
    shape_for_stack = (1, fvr.size[0], fvr.size[1], fvr.depth)
    img_stack = np.zeros(shape_for_stack)

    for i in range(0, fvr.nframes):
        frame = fvr.read_frame()
        frame = frame.reshape(shape_for_stack)
        img_stack = np.vstack((img_stack, frame))

    img_stack = img_stack[1:]
    cat_stack = np.ones((len(img_stack), 1)) * category_id

    return img_stack, cat_stack


def save_frames_from_vid(video_name, category_id, train_or_test, row=0):

    make_dir_structure(row)

    # Initialize FFMPEG_VideoReader
    fvr = FFMPEG_VideoReader(filename=video_name)
    fvr.initialize()
    vid = os.path.split(video_name)[1]
    for i in range(0, fvr.nframes):
        frame_name = vid + '_' + str(i)
        frame = fvr.read_frame()
        imsave(os.path.join('trafficdb', 'eval_'+ str(row), train_or_test, str(category_id), frame_name + '.jpg'),
               frame)

    return True


def get_category_numeric_id(category):
    if (category == 'light'):
        return 0
    elif (category == 'medium'):
        return 1
    elif (category == 'heavy'):
        return 2
    else:
        return 3


def get_category_ohe(category):
    if (category == 0):
        return np.array([1,0,0])
    elif (category == 1):
        return np.array([0, 1, 0])
    elif (category == 2):
        return np.array([0, 0, 1])
    else:
        return False


def make_dir_structure(row):

    if(os.path.exists(os.path.join('trafficdb','eval_'+str(row)))):
        return True

    else:
        os.mkdir(os.path.join('trafficdb','eval_'+str(row)))

        os.mkdir(os.path.join('trafficdb', 'eval_' + str(row),'train'))
        os.mkdir(os.path.join('trafficdb', 'eval_' + str(row), 'test'))

        os.mkdir(os.path.join('trafficdb', 'eval_' + str(row), 'train',str(0)))
        os.mkdir(os.path.join('trafficdb', 'eval_' + str(row), 'train', str(1)))
        os.mkdir(os.path.join('trafficdb', 'eval_' + str(row), 'train', str(2)))

        os.mkdir(os.path.join('trafficdb', 'eval_' + str(row), 'test', str(0)))
        os.mkdir(os.path.join('trafficdb', 'eval_' + str(row), 'test', str(1)))
        os.mkdir(os.path.join('trafficdb', 'eval_' + str(row), 'test', str(2)))


def save_train_test_from_db(row=0):
    # Load ImageMaster.mat
    imgmstr_loc = os.path.join('trafficdb', 'ImageMaster.mat')
    imagemstr = spio.loadmat(imgmstr_loc, squeeze_me=True)

    # Get EvalSet_train
    text_file = open(os.path.join('trafficdb', 'EvalSet_train'), 'r')
    text_file.seek(0)
    lines = text_file.readlines()
    idx_rows_train = lines[row].rstrip("\r\n").split(',')
    text_file.close()

    # Get Evalset_test
    text_file = open(os.path.join('trafficdb', 'EvalSet_test'), 'r')
    text_file.seek(0)
    lines = text_file.readlines()
    idx_rows_test = lines[row].rstrip("\r\n").split(',')
    text_file.close()

    trainstack, cat_train_stack, teststack, cat_test_stack = (False, False, False, False)

    # Get training set images
    for idx, i in enumerate(idx_rows_train):

        # Get video name and category
        video_name = os.path.join('trafficdb', 'video', str(imagemstr['imagemaster'][int(i)]['root']) + '.avi')
        category = str(imagemstr['imagemaster'][int(i)]['class'])
        category_id = get_category_numeric_id(category)
        if (category_id > 2):
            raise ValueError('Invalid Category for video ' + video_name)

        save_frames_from_vid(video_name, category_id, 'train', row)

    # Get test set images
    for idx, i in enumerate(idx_rows_test):

        # Get video name and category
        video_name = os.path.join('trafficdb', 'video', str(imagemstr['imagemaster'][int(i)]['root']) + '.avi')
        category = str(imagemstr['imagemaster'][int(i)]['class'])
        category_id = get_category_numeric_id(category)
        if (category_id > 2):
            raise ValueError('Invalid Category for video ' + video_name)

        save_frames_from_vid(video_name, category_id, 'test', row)




