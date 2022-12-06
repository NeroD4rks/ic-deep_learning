from tensorflow import keras
from modelos import *
from auxfunc import *
from keras.utils import np_utils

test_folder = "datasets\CWT-BINARY\BeetleFly\TEST"
train_folder  = "datasets\CWT-BINARY\BeetleFly\TRAIN"

del_augment_images(train_folder)

"""shape = (32, 32, 3)
n_class = get_n_class(train_folder)
x_train, y_train = get_test_train(train_folder, shape)
x_test, y_test = get_test_train(test_folder, shape)

x_test = x_test + x_test
y_test = y_test + y_test

def decay(epoch):
    return 0.001 / (1 + 1 * epoch)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(n_class)

print("Iniciando augment, x_train")
x_train = preprocess(x_train, shape)


print("Iniciando preprocess, x_test")
x_test = preprocess(x_test, shape)


callbacks = []
callbacks += [keras.callbacks.LearningRateScheduler(decay, verbose=1)]

model = get_model_densenet(shape, n_class)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=x_train, y=y_train,
                    validation_data=(x_test, y_test),
                    epochs=200,
                    callbacks=callbacks,
                    verbose=1)


"""