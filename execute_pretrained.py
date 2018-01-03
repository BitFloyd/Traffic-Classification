from data_pkg import image_fns as imgf
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3 , preprocess_input


train_images, train_labels, test_images, test_labels = imgf.return_train_test_images('eval_2')

cat_train_labels = to_categorical(train_labels, num_classes=3)
cat_test_labels = to_categorical(test_labels, num_classes=3)

print ""


print "PREPROCESS IMAGES"

train_images = preprocess_input(train_images)
test_images = preprocess_input(test_images)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(200, 200, 3),pooling=None)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

adam = Adam(lr=1e-10)

model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])


print "MODEL SUMMARY"
model.summary()


es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='model_saved.h5',monitor='val_loss',verbose=1,save_best_only=True)

model.fit(train_images, cat_train_labels, shuffle=True, epochs=200, callbacks=[es, rlr,mcp],
          validation_split=0.2, verbose=1,batch_size=64)


print "LOADING THE MODEL"
model = load_model('model_saved.h5')
predictions = model.predict_classes(test_images)

print ""
print "ACCURACY SCORE: ", accuracy_score(test_labels, predictions)

print ""
print "CONFUSION MATRIX : "
print confusion_matrix(test_labels,predictions)