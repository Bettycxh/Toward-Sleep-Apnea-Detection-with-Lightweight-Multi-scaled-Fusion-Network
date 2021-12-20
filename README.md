# SE-MSCNN-A-Lightweight-Multi-scaled-Fusion-Network-for-Sleep-Apnea-Detection-Using-Single-Lead-ECG-

## Abstract
Sleep apnea (SA) is a common sleep disorder that occurs during sleep and its symptom is the reduction or disappearance of respiratory airflow caused by upper airway collapse. The SA would cause a variety of diseases like diabetes, chronic kidney disease, depression, cardiovascular diseases, or even sudden death. Early detecting SA and intervention can help individuals to prevent malignant events induced by SA. In this study, we propose a multi-scaled fusion network named SE-MSCNN for SA detection based on single-lead ECG signals acquired from wearable devices. The proposed SE-MSCNN mainly has two modules: multi-scaled convolutional neural network (CNN) module and channel-wise attention module. To utilize adjacent ECG segments information to facilitate the SA detection performance, the multi-scaled CNN module consists of three streams of shallow neural networks with segments with various length as inputs to extract different scaled features. To overcome the problem of feature information local concentration for feature fusion with concatenation, a channel-wise attention module with squeeze-to-excitation block is employed to fuse the different scaled features adaptively. Experiment results on PhysioNet Apnea-ECG dataset show that the proposed SE-MSCNN can achieve the best per-segment accuracy of 90.64 % and the best per-recording accuracy of 100 %, which is superior to state-of-the-art SA detection methods with a big margin. The SE-MSCNN with merits of quick response and lightweight parameters can be potentially embedded to a wearable device to provide a SA detection service for individuals in home sleep test.

## Dataset
[Apnea-ECG Database](https://physionet.org/content/apnea-ecg/1.0.0/)

## Usage

## Requirements
Python==3.6
Keras==2.3.1
TensorFlow==1.14.0


## Cite
Not yet published
<!-- If our work is helpful to you, please cite: -->

## Email
If you have any questions, please email to: [xhchen@m.scnu.edu.cn](mailto:xhchen@m.scnu.edu.cn)
