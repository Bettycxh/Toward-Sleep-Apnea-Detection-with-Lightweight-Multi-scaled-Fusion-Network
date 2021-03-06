"""NOTES: Batch data is different each time in keras, which result in slight differences in results."""
import pickle
import keras
import numpy as np
import pandas as pd
import os
from keras.layers import Dropout, MaxPooling1D, Reshape, multiply, Conv1D, GlobalAveragePooling1D, Dense
from keras.models import Input, Model, load_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, f1_score
import random

base_dir = "./dataset"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
ir = 3  # interpolate interval
before = 2
after = 2
# normalize
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def load_data(path):
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
    with open(os.path.join(base_dir, path), 'rb') as f:  # read preprocessing result
        apnea_ecg = pickle.load(f)
    x_train1,x_train2,x_train3 = [],[],[]
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
        # Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train1.append([rri_interp_signal, ampl_interp_signal])  # 5-minute-long segment
        x_train2.append([rri_interp_signal[180:720], ampl_interp_signal[180:720]])  # 3-minute-long segment
        x_train3.append([rri_interp_signal[360:540], ampl_interp_signal[360:540]])  # 1-minute-long segment
    x_training1,x_training2,x_training3,y_training,groups_training = [],[],[],[],[]
    x_val1,x_val2,x_val3,y_val,groups_val = [],[],[],[],[]

    trainlist = random.sample(range(len(o_train)),int(len(o_train)*0.7))
    num = [i for i in range(16709)]
    vallist = set(num) - set(trainlist)
    vallist = list(vallist)
    for i in trainlist:
        x_training1.append(x_train1[i])
        x_training2.append(x_train2[i])
        x_training3.append(x_train3[i])
        y_training.append(y_train[i])
        groups_training.append(groups_train[i])
    for i in vallist:
        x_val1.append(x_train1[i])
        x_val2.append(x_train2[i])
        x_val3.append(x_train3[i])
        y_val.append(y_train[i])
        groups_val.append(groups_train[i])
    x_training1 = np.array(x_training1, dtype="float32").transpose((0, 2, 1))
    x_training2 = np.array(x_training2, dtype="float32").transpose((0, 2, 1))
    x_training3 = np.array(x_training3, dtype="float32").transpose((0, 2, 1))
    y_training = np.array(y_training, dtype="float32")
    x_val1 = np.array(x_val1, dtype="float32").transpose((0, 2, 1))
    x_val2 = np.array(x_val2, dtype="float32").transpose((0, 2, 1))
    x_val3 = np.array(x_val3, dtype="float32").transpose((0, 2, 1))
    y_val = np.array(y_val, dtype="float32")
    x_test1,x_test2,x_test3 = [],[],[]
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
        # Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test1.append([rri_interp_signal, ampl_interp_signal])
        x_test2.append([rri_interp_signal[180:720], ampl_interp_signal[180:720]])
        x_test3.append([rri_interp_signal[360:540], ampl_interp_signal[360:540]])
    x_test1 = np.array(x_test1, dtype="float32").transpose((0, 2, 1))
    x_test2 = np.array(x_test2, dtype="float32").transpose((0, 2, 1))
    x_test3 = np.array(x_test3, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return x_training1, x_training2, x_training3, y_training, groups_training, x_val1, x_val2, \
           x_val3, y_val, groups_val, x_test1, x_test2, x_test3, y_test, groups_test


def lr_schedule(epoch, lr):
    if epoch > 70 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr


def create_model(input_a_shape, input_b_shape, input_c_shape, weight=1e-3):
    # SA-CNN-3
    input1 = Input(shape=input_a_shape)
    x1 = Conv1D(16, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(input1)
    x1 = Conv1D(24, kernel_size=11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling1D(pool_size=3, padding="same")(x1)
    x1 = Conv1D(32, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x1 = MaxPooling1D(pool_size=5, padding="same")(x1)

    # SA-CNN-2
    input2 = Input(shape=input_b_shape)
    x2 = Conv1D(16, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(input2)
    x2 = Conv1D(24, kernel_size=11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x2)
    x2 = MaxPooling1D(pool_size=3, padding="same")(x2)
    x2 = Conv1D(32, kernel_size=11, strides=3, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x2)

    # SA-CNN-1
    input3 = Input(shape=input_c_shape)
    x3 = Conv1D(16, kernel_size=11, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(input3)
    x3 = Conv1D(24, kernel_size=11, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x3)
    x3 = MaxPooling1D(pool_size=3, padding="same")(x3)
    x3 = Conv1D(32, kernel_size=1, strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x3)

    # Channel-wise attention module
    concat = keras.layers.concatenate([x1, x2, x3], name="Concat_Layer", axis=-1)
    squeeze = GlobalAveragePooling1D()(concat)
    excitation = Dense(48, activation='relu')(squeeze)
    excitation = Dense(96, activation='sigmoid')(excitation)
    excitation = Reshape((1, 96))(excitation)
    scale = multiply([concat, excitation])
    x = GlobalAveragePooling1D()(scale)
    dp = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax', name="Output_Layer")(dp)
    model = Model(inputs=[input1, input2, input3], outputs=outputs)
    return model


if __name__ == "__main__":
    # load_data
    path = "apnea-ecg.pkl"
    x_train1, x_train2, x_train3, y_train, groups_train, x_val1, x_val2,\
    x_val3, y_val, groups_val, x_test1, x_test2, x_test3, y_test, groups_test = load_data(path)
    y_train = keras.utils.to_categorical(y_train, num_classes=2)  # Convert to two categories
    y_val = keras.utils.to_categorical(y_val, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    print('input_shape', x_train1.shape, x_train2.shape, x_train3.shape)

    # training
    model = create_model(x_train1.shape[1:], x_train2.shape[1:], x_train3.shape[1:])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = 'weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    callbacks_list = [lr_scheduler, checkpoint]
    history = model.fit([x_train1, x_train2, x_train3], y_train, batch_size=128, epochs=100,
                        validation_data=([x_val1, x_val2, x_val3], y_val), callbacks=callbacks_list)

    # test
    filepath = './weights.best.hdf5'
    model = load_model(filepath)
    loss, accuracy = model.evaluate([x_test1, x_test2, x_test3], y_test)
    # save prediction score
    y_score = model.predict([x_test1, x_test2, x_test3])
    output = pd.DataFrame({"y_true": y_test[:, 1], "y_score": y_score[:, 1], "subject": groups_test})
    output.to_csv("./utils/code_for_calculating_per-recording/output/SE-MSCNN.csv", index=False)
    y_true, y_pred = np.argmax(y_test, axis=-1), np.argmax(model.predict([x_test1, x_test2, x_test3], batch_size=1024, verbose=1), axis=-1)
    C = confusion_matrix(y_true, y_pred, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    f1 = f1_score(y_true, y_pred, average='binary')
    print("acc: {}, sn: {}, sp: {}, f1: {}".format(acc, sn, sp, f1))
