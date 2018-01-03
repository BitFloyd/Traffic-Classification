import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

import matplotlib as mpl
mpl.use('Agg')
from data_pkg import image_fns as imgf
import numpy as np
from data_pkg.graphing import plot_graph
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.models import Sequential
from keras.layers import Dense,Dropout,GaussianNoise
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3 , preprocess_input
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter


def eval_test(eval_id):

    train_images, train_labels, test_images, test_labels = imgf.return_train_test_images(eval_id)

    cat_train_labels = to_categorical(train_labels, num_classes=3)
    cat_test_labels = to_categorical(test_labels, num_classes=3)

    print ""
    print "PREPROCESS IMAGES"
    print ""

    train_images = preprocess_input(train_images)
    test_images = preprocess_input(test_images)

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(200, 200, 3), pooling='avg')

    train_images_feats = base_model.predict(train_images, verbose=1)
    test_images_feats = base_model.predict(test_images, verbose=1)

    del train_images
    del test_images

    # this is the model we will train

    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=train_images_feats.shape[1]))
    model.add(GaussianNoise(0.01))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=16, activation='relu'))
    model.add(GaussianNoise(0.01))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=3, activation='softmax'))

    adam = Adam(lr=1e-5)

    model.compile(optimizer=adam, loss='categorical_crossentropy')

    print "MODEL SUMMARY"
    model.summary()

    #Class imbalance problem
    class_0_images_idx = np.where(train_labels == 0)
    class_0_images_feats = train_images_feats[class_0_images_idx]
    class_0_labels = cat_train_labels[class_0_images_idx]

    class_1_images_idx = np.where(train_labels == 1)
    class_1_images_feats = train_images_feats[class_1_images_idx]
    class_1_labels = cat_train_labels[class_1_images_idx]

    class_2_images_idx = np.where(train_labels == 2)
    class_2_images_feats = train_images_feats[class_2_images_idx]
    class_2_labels = cat_train_labels[class_2_images_idx]

    losslist = []

    for i in tqdm(range(0, 1000000)):
        class_0_idx = np.random.randint(0, len(class_0_images_feats), 32)
        class_1_idx = np.random.randint(0, len(class_1_images_feats), 32)
        class_2_idx = np.random.randint(0, len(class_2_images_feats), 32)

        batch_x = np.vstack(
            (class_0_images_feats[class_0_idx], class_1_images_feats[class_1_idx], class_2_images_feats[class_2_idx]))
        batch_y = np.vstack((class_0_labels[class_0_idx], class_1_labels[class_1_idx], class_2_labels[class_2_idx]))

        loss = model.train_on_batch(batch_x, batch_y)
        losslist.append(loss)

    losslist = gaussian_filter(losslist,sigma=1.0)

    plot_graph(losslist, eval_id+'_lossgraph.png', 'Graph of the loss per minibatch', 'minibatch-index', 'loss-value')

    predictions = model.predict_classes(test_images_feats)

    accu_score = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels,predictions)

    print ""
    print "ACCURACY SCORE: ", accu_score
    print ""
    print "CONFUSION MATRIX : "
    print cm

    return accu_score,cm



list_accuracy_score = []
list_confusion_matrix = []

for i in range(0,4):
    eval_id = 'eval_'+str(i)
    ac,cm = eval_test(eval_id)
    list_accuracy_score.append(ac)
    list_confusion_matrix.append(cm)


print "ACCURACY SCORES PER EVAL_SET "
for idx,i in enumerate(list_accuracy_score):
    print "EVAL_",idx," : ",i

print "MEAN ACCURACY OF ALL EVAL_SETS : ", np.mean(list_accuracy_score)

print "CONFUSION MATRICES PER EVAL_SET "
for idx,i in enumerate(list_confusion_matrix):

    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "EVAL_",idx," : "
    print i
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"


print "MEAN ACCURACY OF ALL EVAL_SETS : ", np.mean(list_accuracy_score)

print "MEAN OF ALL CONFUSION MATRICES : ", np.mean(np.array(list_confusion_matrix),axis=0)
