# GTC2016 Lab
# L6136 - Deep Learning on GPUs: From Large Scale Training to Embedded Deployment

## Slides

Slides with partial instructions are available [here](slides.pdf).

## Part 3: Install a few necessary programs and make sure caffe is working.

### Dependencies
Get the required dependencies and a few useful tools
```
sudo apt-get install cmake git aptitude screen g++ libboost-all-dev \
    libgflags-dev libgoogle-glog-dev protobuf-compiler libprotobuf-dev \
    bc libblas-dev libatlas-dev libhdf5-dev libleveldb-dev liblmdb-dev \
    libsnappy-dev libatlas-base-dev python-numpy libgflags-dev \
    libgoogle-glog-dev python-skimage python-protobuf python-pandas \
    libopencv4tegra-python
```

### Caffe
We will use the experimental branch of caffe used in the [whitepaper](http://www.nvidia.com/content/tegra/embedded-systems/pdf/jetson_tx1_whitepaper.pdf).

It was set up and provided to you in the github material that you should have it available in: /home/ubuntu/tx1-lab2/caffe/

First, pull the latest changes from GitHub:
```
cd ~/tx1-lab2
git pull
```

Set up a few environment variables
```
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ubuntu/tx1-lab2/caffe/3rdparty/cnmem/build:/home/ubuntu/tx1-lab2/caffe/distribute/lib' >> ~/.bashrc
echo 'export PYTHONPATH=${PYTHONPATH}:/home/ubuntu/tx1-lab2/caffe/distribute/python' >> ~/.bashrc
bash
```

### FP16 eval
Let's check our version of caffe is working by reproing the [whitepaper](http://www.nvidia.com/content/tegra/embedded-systems/pdf/jetson_tx1_whitepaper.pdf) numbers. 

### Setting clocks
First, check the clocks on the TX1:
```
sudo bash jetson_max_l4t.sh --show
```

Let's set maximum clocks on the TX1 for best performance:
```
sudo bash jetson_max_l4t.sh
```
The fan should start.

### Running caffe fp16 inference with batch size 1
Caffe prototxt files for [Alexnet](caffe_files/deploy_alexnet_b1.prototxt) and [Googlenet](caffe_files/deploy_googlenet_b1.prototxt) are available.

Timings can be obtained with:
```
/home/ubuntu/tx1-lab2/caffe/build/tools/caffe_fp16 time --model=/home/ubuntu/tx1-lab2/caffe_files/deploy_alexnet_b1.prototxt -gpu 0 -iterations 100
/home/ubuntu/tx1-lab2/caffe/build/tools/caffe_fp16 time --model=/home/ubuntu/tx1-lab2/caffe_files/deploy_googlenet_b1.prototxt -gpu 0 -iterations 100
```

Compare numbers with the ones presented in the [whitepaper](http://www.nvidia.com/content/tegra/embedded-systems/pdf/jetson_tx1_whitepaper.pdf).


## Part 4: Deploy the classification model on the TX1

### Download the model
Setup a directory to put models in:
```
cd
mkdir deploy_files
```

Download a model with the [provided python script](digits_connect/download-digits-model.py):
```
python tx1-lab2/digits_connect/download-digits-model.py \
  -n <your amazon instance>.compute-1.amazonaws.com -p 5000 deploy_files/my_model.tar.gz
```

Extract files from the archive:
```
cd deploy_files
tar xzvf my_model.tar.gz
cd
```

### Classify an image
Use the same image you use in Part 2 with Digits. Download it and save it in the _Pictures_ folder.

Classify it using the classification binary available in caffe, example:
```
/home/ubuntu/tx1-lab2/caffe/build/examples/cpp_classification/classification.bin /home/ubuntu/deploy_files/deploy.prototxt  /home/ubuntu/deploy_files/snapshot_iter_54400.caffemodel /home/ubuntu/deploy_files/mean.binaryproto /home/ubuntu/deploy_files/labels.txt /home/ubuntu/Pictures/Bananas.jpg 
---------- Prediction for /home/ubuntu/Pictures/Bananas.jpg ----------
1.0000 - "banana"
0.0000 - "lemon"
0.0000 - "pineapple"
0.0000 - "sunglasses"
0.0000 - "keyboard"
```

## Part 5 : Webcam
The previous classification model can also be run from the webcam:
```
cd /home/ubuntu/tx1-lab2/webcam
python webCamClassify.py --gpu --mean_file mean.npy  --pretrained_model /home/ubuntu/deploy_files/snapshot_iter_54400.caffemodel --labels_file /home/ubuntu/deploy_files/labels.txt --model_def /home/ubuntu/deploy_files/deploy.prototxt  
```
