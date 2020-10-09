# Object-detection-in-real-time-using-tensorflow
STUDENTSPLAYGROUND
.
Technical Documentation

Phase 1:

Understanding Image Recognition using  CIFAR 10 

TABLE OF CONTENTS

0	Preface
0.1	Purpose of this document
This document is a generic technical documentation for use by the project STUDENTSPLAYGROUND under the id : studentsone. It provides guidance for the members associated with this project as it shows the errors and mistakes while performing basic steps in image processing and object detection and also assists better ways to avoid wrong methods adopted while practising image processing and classification.
0.2	Overview
This documentation is basically a record of all the problems encountered during the initial learning phase of object detection which is image classification along with their solutions. 
0.3	Basis of this Document
This documentation is based on the practical implementation of basic image processing algorithm and image classification by the members of the group: Prasoon Kumar (leader), Hritik Kumar, Shivam Shubham    under the esteemed guidance and direction of Dr. Sudhir Kumar Singh           (Founder CEO of  INVII.AI). 
⦁	Introduction
⦁	Purpose of the project
The project Studentsplayground, unlike to what the name suggests, actually suggests the play area for all its members who are actually intended to play with various algorithms and then come up with an efficient one for the main project.
In the initial phase, the team worked on CIFAR-10 dataset and and training testing and evaluating the model through tensorflow checking different possibilities to find an even greater algorithm in terms of accuracy and an economical one with respect to time. 
⦁	Acronyms and Abbreviations
This section should define all terms, acronyms and abbreviations used in this document:

c10	CIFAR -10 (basic data set the team worked on initially)
tflow	Tensorflow (a machine learning library)
gcp	Google Cloud Platform (a google based platform serving as a remote server provider for storage, computing and networking)
inst	Instance (a virtual machine under gcp projects)
bkt	Bucket(everything that’s stored in google cloud are contained in a bucket)
ML	Machine learning
SSD	Single Shot Detector
⦁	References
This section lists all the applicable and reference documents and learning materials :
https://www.tensorflow.org/
https://askubuntu.com/questions/1072683/how-can-i-install-protoc-on-ubuntu-16-04
https://linuxize.com/
https://www.youtube.com/watch?v=eYj3a9RsA5s&list=PL9ooVrP1hQOFUm7TmkH1zk5xy75GAxV44
https://www.youtube.com/watch?v=3BXfw_1_TF4
https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148
https://stackoverflow.com/
⦁	tools Overview
This section should briefly introduce the tools and discuss the background to the project. 
⦁	TENSORFLOW
Tflow is an open source ML library for research and production by providing APIs to develop. The high-level Keras API provides building blocks to create and train deep learning models.
https://www.tensorflow.org/tutorials/
⦁	IMAGE RECOGNITION
Image recognition is used to perform a large no. of  machine based visual tasks like labelling images with tags, guiding autonomous robots and what not. It is basically the ability to identify objects and motions. 
https://searchenterpriseai.techtarget.com/definition/image-recognition
⦁	Google cloud platform 
Gcp, is a suite of cloud computing services that runs on the same infrastructure that google uses internally for its end-user products such as google maps, YouTube etc.
Advantages of Google Cloud
⦁	Higher Productivity is gained through Quick Access to Innovation: Google’s systems can distribute updates efficiently and deliver functionality on a weekly basis or even faster.
⦁	Less Disruption is Caused When Users Adopt New Functionality: Rather than large disruptive batches of change, Google delivers manageable improvements in a continuous stream.
⦁	Employees Can Work From Anywhere: They can gain full access to information across devices from anywhere in the world through web-based apps powered by Google cloud.
⦁	Google Cloud Allows Quick Collaboration: Many users can contribute to and access projects at the same time as data is stored in the cloud instead of their computers.
⦁	Customers are protected by Google’s Investments in Security: They are benefited by the process-based and physical security investments made by Google. Google hires the leading security experts in the world.
⦁	Less Data has to be stored on Vulnerable Devices: Minimal data is stored on computers that may get compromised after a user stops using web-based apps on the cloud.
⦁	Customers get Higher Uptime and Reliability: If a data center is not available for some reason, the system immediately falls back on the secondary center without any service interruption being visible to users.
⦁	Control and Flexibility Available to Users: They have control over technology and data and have ownership over their data in Google apps. If they decide not to use the service any more, they can get their data out of Google cloud.
⦁	Google’s Economies of Scale Let Customers Spend Less: Google minimizes overheads and consolidates a small number of server configurations. It manages these through an efficient ratio of people to computers.
⦁	https://console.cloud.google.com/getting-started?authuser=2&project=studentsplayground
⦁	cifar 10
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
https://www.tensorflow.org/tutorials/images/deep_cnn
⦁	Phase (I) of the project
This section is intended to provide a detailed explanation of the steps involved, the errors encountered, the solutions to those error while applying the tflow model at every stage of the implementation.
⦁	Ml using tflow
Basic Classification
Our 1st neural network was trained through tflow using tf.keras, a high level API for training tflow models.
https://linuxize.com/post/how-to-install-tensorflow-on-ubuntu-18-04/        t
This tutorial and guide was followed to install python3 on ubuntu and then install tflow in the terminal using pip.
MNIST dataset was used to build, train and evaluate the accuracy of the tflow model.

 
Ran the following code on the terminal to make sure that the tflow is working fine:

from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
it did run fine and gave the output:
1.14.0
So, we proceeded with further steps involved by following the tflow guide:
https://www.tensorflow.org/tutorials/keras/basic_classification


Overfitting & Underfitting
After evaluating the results and the accuracy of the model, it was concluded that the accuracy of our model on the validation data would peak after training for a number of epochs, and would then start decreasing.
In other words, our model would overfit to the training data. Learning how to deal with overfitting is important. Although it's often possible to achieve high accuracy on the training set, what we really want is to develop models that generalize well to a testing set (or data they haven't seen before).
The opposite of overfitting is underfitting. Underfitting occurs when there is still room for improvement on the test data. This can happen for a number of reasons: If the model is not powerful enough, is over-regularized, or has simply not been trained long enough. This means the network has not learned the relevant patterns in the training data. If you train for too long though, the model will start to overfit and learn patterns from the training data that don't generalize to the test data. We need to strike a balance. Understanding how to train for an appropriate number of epochs as we'll explore below is a useful skill. To prevent overfitting, the best solution is to use more training data. A model trained on more data will naturally generalize better. When that is no longer possible, the next best solution is to use techniques like regularization. These place constraints on the quantity and type of information your model can store. If a network can only afford to memorize a small number of patterns, the optimization process will force it to focus on the most prominent patterns, which have a better chance of generalizing well.

60000/60000 [==============================] - 4s 75us/sample - loss: 0.5018 - acc: 0.8241
Epoch 2/5
60000/60000 [==============================] - 4s 71us/sample - loss: 0.3763 - acc: 0.8643
Epoch 3/5
60000/60000 [==============================] - 4s 71us/sample - loss: 0.3382 - acc: 0.8777
Epoch 4/5
60000/60000 [==============================] - 4s 72us/sample - loss: 0.3138 - acc: 0.8846
Epoch 5/5
60000/60000 [==============================] - 4s 72us/sample - loss: 0.2967 - acc: 0.8897

<tensorflow.python.keras.callbacks.History at 0x7f65fb64b5c0>


⦁	Advanced cnn
Advanced CNN involved understanding and implementing C10 classification problem where we intended to classify RGB 32 x 32 pixel images across 10 categories:
               airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
⦁	Goals
The goal of this tutorial is to build a relatively small convolutional neural network(CNN) for recognizing images. In the process, this tutorial:
⦁	Highlights a canonical organization for network architecture, training and evaluation.
⦁	Provides a template for constructing larger and more sophisticated models.
The reason CIFAR-10 was selected was that it is complex enough to exercise much of TensorFlow's ability to scale to large models. At the same time, the model is small enough to train fast, which is ideal for trying out new ideas and experimenting with new techniques.
⦁	Highlights of the Tutorial
The CIFAR-10 tutorial demonstrates several important constructs for designing larger and more sophisticated models in TensorFlow:
⦁	Core mathematical components including tf.nn.conv2d (wiki), tf.nn.relu(wiki), tf.nn.max_pool (wiki) and tf.nn.local_response_normalization(section 3.3 in AlexNet paper).
⦁	Visualization of network activities during training, including input images, losses and distributions of activations and gradients.
⦁	Routines for calculating the tf.train.ExponentialMovingAverage of learned parameters and using these averages during evaluation to boost predictive performance.
⦁	Implementation of a tf.train.exponential_decay that systematically decrements over time.
⦁	Prefetching input data to isolate the model from disk latency and expensive image pre-processing.

⦁	c10 tutorial
The code for this tutorial resides in https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/
the following sequence must be followed in order to successfully build the model, train the model and evaluate the model:

cifar10_input.py
	Loads CIFAR-10 dataset using tensorflow-datasets library.
cifar10.py
	Builds the CIFAR-10 model.
cifar10_train.py
	Trains a CIFAR-10 model on a CPU or GPU.
cifar10_multi_gpu_train.py
	Trains a CIFAR-10 model on multiple GPUs.
cifar10_eval.py
	Evaluates the predictive performance of a CIFAR-10 model.

We studied and tried to understand the codes as much as we could and then decided how to implement using gcp
⦁	Working on gcp
In the account ‘ iinvi.ai ’ formed in gcp we created an inst and named it instance2 using this video:  https://www.youtube.com/watch?v=_Q0tRI5hMnc&list=PL9ooVrP1hQOFUm7TmkH1zk5xy75GAxV44&index=3&t=22s
 Installation of tensorflow-gpu in  google cloud platform in ubuntu-18.04 

After creating a instance of ubuntu in GCP, we have to install nvidia-driver, cuda , cudNN and then tensorflow-gpu

update ubuntu

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install gnome-shell

install nvidia driver (https://www.nvidia.com/en-gb/data-center/gpu-accelerated-applications/tensorflow/)

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-430

install cuda-toolkit 
(https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork)

Download cuda deb network file from above link

sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb(compatible so we downgrading to cuda 10.0)
sudo apt-key adv --fetch-keys (https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub)
sudo apt-get update
sudo apt-get install cuda

The details of our newly created inst is as shown in the screenshots below:
 
 
 

















after creating the inst, we installed python3, venv, pip and tflow in the compute engine created in the virtual machine that is instance2 like we did in the ubuntu terminal.
Before downloading cudNN.tgz file to your local PC, create a bucket in GCP so that we can directly import large file to our instance-shell
Create a bucket – multi regional – set object level and bucket level policy transfer bucket file to instance.
we created  a tflow directory in the inst and 
to load the data into the directory for training the model we then created a bkt:

 


on connecting the bkt to the inst we uploaded those datasets into the inst  from that bkt and then started working on the C10 datasets and tflow codes:
gsutil ls ( to check your bucket is accessible or not, if not create a new one)
gcloud init ( pick project_name – pick zone)

⦁	install cudNN ( for downloading cudNN, create a new nvidia account https://developer.nvidia.com/cudnn )
Copying cudnn file from bucket to instance
gsutil cp [file_source_URL] [file_destination_URL]

            storage error (400 Bucket is requester pays bucket but no user project provided.)
            gsutil -u [PROJECT_ID] cp gs://[bucket_name]/[obj_name]   [obj_destination]

⦁	unzip cudNN
tar -xzvf [cudnn filename .tgz]

⦁	Copy the following files into the CUDA Toolkit directory, and change the file permissions
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

⦁	Prepare for TensorFlow dependencies and required packages.
sudo apt-get install libcupti-dev
⦁	install tensorflow-gpu
first install pip for python3( sudo apt install python3-pip)
python3 -m pip install tensorflow-gpu

⦁	open python3 and Verify tensorflow-gpu

import tensorflow as tf
print(tf.__version__)                  
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))       O/P – 1.14     Hello, TensorFlow!'

⦁	Install python libraries(numpy,pandas,matplotlib,scikit-learn,scipy,seaborn)
python3  -m pip install [lib – name]
⦁	check tensorflow-gpu
tf.test.is_gpu_available(
           cuda_only=False,
           min_cuda_compute_capability=None)        O/P – TRUE


These are the library files installed and needed:
 
1. To load cif10 dataset from tflow-datasets library, we ran the code in the file cifar10_input.py  from models folder.
(took 6 seconds to run)
2. Running cifar10.py built the predefined model which will work on the loaded tfow dataset.
3. Next we ran cifar10_train.py to train the built model on the dataset in the GPU provided in GCP.
4. At last to evaluate the predictive performance of the tflow model we ran cifar10_eval.py.

The convergence of the loss value is observed while training the model.
We get the accuracy after evaluating.
The value of loss can be considered for evaluating the accuracy of the model only when the value of loss converges after some no. of steps.

As the no. of steps for which the model has been trained increases the chances of the loss value converging increases :
 

No. of Steps     Loss (Training)             Accuracy (evaluation)

* 500 steps           arbitrary loss=3.6                     accuracy = 56%
* 2000 steps         arbitrary loss=2.2                     accuracy =68%
* 5000 steps         arbitrary loss=1.6                     accuracy =72% 
* 10000 steps      partly converged loss=1.1         accuracy =74%
* 20000 steps     partly converged loss=1.0          accuracy =79%
* 50000 steps     converged loss=0.7                    accuracy = 83%

when the training part of the model runs, it appears like the image below:

 




Conclusion:

Phase 1 of the project provided a good, hands-on, beginner level experience on training, testing and evaluation of ML models as it lead the learning through several obstacles challenges, be it technology based or knowledge based and thus, making the understanding of machine learning algorithms more precise and their implementation more clear.





Now taking this experience on a greater level in Phase 2:
Phase 2:


Object Detection Using Single Shot Detector



TABLE OF CONTENTS
0	Phase 2 : OVERVIEW	1

1	Introduction TO SSD	5

2	SYSTEM INSTALLATION	7
2.1	Package Installation	7
2.2	Downloading TF models	7
2.3 Protobuf Installation………………………………………………………………8
2.4	COCO API Installation	9
3	structuring work area	11

4	Label Map Creation	15
5          creating tf records……………………………………………………….16
5.1	Converting .xml to .csv	7
5.2	Converting .csv to .record	7

6	Configuring training pipeline	5
7	model training	5
8	evaluation	5

9	conclusion	5

   Overview    

In this documentation, we are going to discuss brief knowledge about Object Detection and learn the whole procedure to run any objection detection model into a Linux based system. 
Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class in digital images and videos. Well-researched domains of object detection include face detection and pedestrian detection.
As declared this documentation is for the Linux based System, we will discuss each and every point clearly about the installation needed to train and evaluate the model.
Our model is based on the technique called SSD for Object Detection. So, we are going to know the details related to this topic in next topic.









What is SSD?

SSD or Single Shot Detector, SSD is designed for object detection in real-time. As different CNNs  has many bottleneck like:-
1. Training the data is unwieldy too long.
2. Training happens in multiple phases (e.g. training reason proposal vs       classifier).
3. Network is too slow at inference time (i.e when dealing with non-training data).
Therefore, to address these bottlenecks many architectures were created in these few years , enabling real-time object detection. The most famous ones are YOLO and SSD. We are going to explore SSD in this after understanding this it will be easy to understand YOLO too.
To better understand SSD, let's start explaining each term of its name:
Single Shot- 
This means that the tasks of object localization and classification are done in a single forward pass of the network.

Multibox-
 This is the approach for bounding box regression developed by szegedy.

Detector- 
The network is an object detector that also classifies those detected objects.

	
	Architecture- 
	The VGG-16 architecture is used in SSD because of its best performance in high quality image classification tasks and its popularity for problems where transfer learning helps in improving results. Instead of the original VGG fully connected layers, a set of auxiliary convolutional layers were added, thus enabling to extract features at multiple scales and progressively decrease the size of the input to each subsequent layer.
SSD does not use a delegated region proposal network. Instead, it resolves to a very simple method. It computes both the location and class scores using small convolution filters. After extracting the feature maps, SSD applies 3 × 3 convolution filters for each cell to make predictions. (These filters compute the results just like the regular CNN filters.) Each filter outputs 25 channels: 21 scores for each class plus one boundary box.
Just like Deep Learning, we can start with random predictions and use gradient descent to optimize the model. However, during the initial training, the model may fight with each other to determine what shapes (pedestrians or cars) to be optimized for which predictions.
For more resources
⦁	https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab
⦁	https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11
⦁	https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06
⦁	https://arxiv.org/abs/1611.10012



	Installation

Installation of TensorFlow for gpu , cpu and NVIDIA Drivers and Toolkit is defined in our last version of documentation which is related to image classification on the CIFAR10 model. Link of that documentation is below: -

Package Installation
After creating new virtual environment, we had to install some packages before installing different models. So given below are the few packages to install only we have to do is change the package name after python3 -m pip install <package name> <=version>.
⦁	python3 -m pip install pillow
⦁	python3 -m pip install lxml
⦁	python3 -m pip install contextlib2
⦁	python3 -m pip install opencv – python
⦁	python3 -m pip install coco
⦁	python3 -m pip install protobuf - compiler
  NOTE: - According to different versions of ubuntu, there should be some more packages to install which will be asked to install during these packages install so same process to install them also as written above. Ex: Like in coco installation it is asked to install crontab.

Test of Installation
As there is important to crosscheck what we did till now is correct or not. As above installation codes are correct otherwise, we can’t proceed ahead but to check we provided few codes in this we are just trying to run some basic code of TensorFlow to check it is working or not.
⦁	python3
⦁	import tensorflow as tf

   Downloading of TensorFlow Models         
   Now it’s time to clone our TensorFlow models for the all the tasks to be done in this object detection models.
⦁	git clone https://github.com/tensorflow/models.git

Protobuf Installation
The TensorFlow Object Detection API uses protobufs to configure models and training parameters. Therefore, we have to install protobuf for it so first make a path to the research directory via models.
⦁	Move to the research directory (cd/models/research) and compile the Protobuf libraries as follows: -
First make a directory of protoc_3.0
⦁	mkdir protoc_3.0
Then move inside the directory.
⦁	cd protoc_3.0
Inside the directory we have to install the latest protobuf according to our use.
⦁	wget https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
⦁	unzip protobuf.zip
After unzipping the zipped file we have to run the following code given below
⦁	Add libraries to PYTHONPATH—export PYTHONPATH= $PYHTONPATH:’pwd’:’pwd’/slim

As TensorFlow models are core package for object detection, it’s convenient to add specific folder to our environment variables.
⦁	./protoc_3.0/bin/protoc object_detection/protos/*.proto –python out=.

Setup completion
⦁	python setup.py build
⦁	python setup.py install

Testing the Installation
⦁	python object_detection/builders/model_builder_test.py

COCO API Installation(optional)
Coco is a large image dataset designed for object detection, segmentation, person keypoints detection and caption generation. This package provides Matlab, Python and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO.
For COCO API installation I am providing you a video link(https://www.youtube.com/watch?v=COlbP62-B-U HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s" HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"feature=youtu.be HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"t=7m23s" HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s" HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"feature=youtu.be HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"t=7m23s" HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"feature=youtu.be HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s" HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"feature=youtu.be HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"t=7m23s" HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s" HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"feature=youtu.be HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"& HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"t=7m23s" HYPERLINK "https://www.youtube.com/watch?v=COlbP62-B-U&feature=youtu.be&t=7m23s"t=7m23s) by which we can easily understand the installation process.
In our model, we have created our own dataset for training and testing purpose. So, after this whole process of data conversion from .xml to .csv and .csv to .record is written. 

  Creating Tfrecord files

⦁	For the purpose of training and testing, Divide images into two new different folders Train and Test. Divide the total number of images in your required ratio. Most of the times the ratio should be 80:20 and 90:10 if the number of images is 100. 

Converting  .xml to .csv   

⦁	make a file ‘xml_to csv.py’ by using  ‘gedit’ in terminal and paste code from ( https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#converting-xml-to-csv)
 
To Create train data:
⦁	python3 xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv
     
 To Create test data:
⦁	python3 xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv

         Converting from .csv to .record
⦁	make a file ‘generate_tfrecord.py’ by using  ‘gedit’ in terminal and paste code from ( https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#converting-from-csv-to-record )
⦁	set image path in main function for train images and test images ( path = os.path.join(os.getcwd(), FLAGS.img_path) ) change ‘  FLAGS.img_path’  to image directories.

⦁	for error (TypeError: None has type NoneType, but expected one of: int, long)   goto ( def class_text_to_int(row_label): ) and return 0 for ‘else’ block.

⦁	python3 generate_tfrecord.py --label= <LABEL> --csv_input=train_labels.csv  --output_path=train.record

⦁	python3 generate_tfrecord.py --label= <LABEL> --csv_input=test_labels.csv  --output_path=test.record

Configuring a Training Pipeline
A machine learning pipeline is used to help automate machine learning workflows. They operate by enabling a sequence of data to be transformed and correlated together in a model that can be tested and evaluated to achieve an outcome, whether positive or negative.
In this model, we are going to reuse one of the pre-trained models to train our model. If you would try to run a new model then have a look at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md
In this we are going to use ssd_inception_v2_coco to train our model due to its speed and mAP. If you want to use a  different models then have a look at this list https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models
First of all, we need to get ourselves a same pipeline configuration file we want to use so download it from the link given just above.
The file downloaded will be on tar.gz format then uncompress it to have access to files inside this and go through the code of pre-trained model then do the changes in the code you want to do like changing the value of num-classes, decay factor etc.


Model Training

⦁	goto cd models/research
⦁	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
⦁	./protoc_3.0/bin/protoc object_detection/protos/*.proto –-python_out=.
⦁	goto cd object_detection

⦁	Download the  SSD network (we use ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz )

⦁	wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

⦁	Extract a file run a command  tar xvzf ⦁	ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

⦁	cd training

⦁	make a same name network config file in gedit ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.config
⦁	copy the network code from https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_300x300_coco14_sync.config
and paste to your own file ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.config
⦁	move the ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.config to training directory
⦁	open ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.config file
⦁	if you only want to train for ‘Tip’ ten (num_classes=1) otherwise equal to number of row_label i.e. 3 in this case
⦁	inside config file find fine_tune_checkpoint: "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt"   below this line there is num_steps: 20000 you can set as your need. (in my case 20,000 steps)  
⦁	train_input_reader: {
  tf_record_input_reader {
    input_path: "data/train.record"    # train.reecord location
  }
  label_map_path: "data/objt.pbtxt"   # pbtxt file location
}
⦁	 same for eval_input_reader , input_path is test.record and pbtxt file location is same
⦁	 for training upto 20k steps run the code python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config
⦁	for evaluation  python3 eval.py --logtostderr --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config --checkpoint_dir=training/ --eval_dir=training/

Making a pbtxt file
⦁	gedit objt.pbtxt
⦁	paste the below code in a file
⦁	item {
 id: 1
 name: 'tip'}
⦁	item {
 id: 2
name: 'top'}
⦁	item {
 id: 3
 name: 'all'}

Exporting a Trained Inference Graph

⦁	Goto cd models/research/object_detection/training  
⦁	simply sort all the files inside training by descending time and pick themodel.ckpt-*file that comes first in the list.
⦁	cd ..
⦁	run the command  python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v2_quantized_300x300_coco.config --trained_checkpoint_prefix training/model.ckpt-20000 --output_directory trained-inference-graphs/output_inference_graph_v1.pb

Changing IOU value from  0.5 to 0.75 & 0.90

⦁	goto cd models/research/object_detection/utills
⦁	open object_detection_evaluation.py
⦁	change all text of   matching_iou_threshold=0.5 to matching_iou_threshold=0.75 or 0.90
⦁	save the file and run cd ..
⦁	In object_detection directory run the command   python3 eval.py --logtostderr --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config --checkpoint_dir=training/ --eval_dir=training/



Monitoring the job using Tensorboard
First activate the TensorFlow gpu using the code: -
activate tensorflow_gpu

Then go to the training folder and run this code: -
tensorboard --logdir=training\







