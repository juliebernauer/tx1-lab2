import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
import cv2

import numpy as np
import matplotlib.pyplot as plt

import caffe
from threading import Thread
from time import sleep

from scipy.misc import imresize

def main(argv):

	pycaffe_dir = os.path.dirname(__file__)

	parser = argparse.ArgumentParser()
	# Optional arguments.
	parser.add_argument(
	    "--model_def",
	    default=os.path.join(pycaffe_dir,
	            "/Users/jbarker/caffe/models/bvlc_reference_caffenet/deploy.prototxt"),
	    help="Model definition file."
	)
	parser.add_argument(
	    "--pretrained_model",
	    default=os.path.join(pycaffe_dir,
	            "/Users/jbarker/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
	    help="Trained model weights file."
	)
	parser.add_argument(
	    "--gpu",
	default=False,
	    action='store_true',
	    help="Switch for gpu computation."
	)
	parser.add_argument(
	    "--mean_file",
	    default=os.path.join(pycaffe_dir,
	                         '/Users/jbarker/caffe/data/ilsvrc12/mean.npy'),
	    help="Data set image mean of [Channels x Height x Width] dimensions " +
	         "(numpy array). Set to '' for no mean subtraction."
	)
	parser.add_argument(
	    "--raw_scale",
	    type=float,
	    default=255.0,
	    help="Multiply raw input by this scale before preprocessing."
	)
	parser.add_argument(
	    "--channel_swap",
	    default='2,1,0',
	    help="Order to permute input channels. The default converts " +
	         "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

	)
	parser.add_argument(
	    "--labels_file",
	    default=os.path.join(pycaffe_dir,
	            "/Users/jbarker/caffe/data/ilsvrc12/synset_words.txt"),
	    help="Readable label definition file."
	)
	args = parser.parse_args()

	mean, channel_swap = None, None
	if args.mean_file:
	    mean = np.load(args.mean_file)
	if args.channel_swap:
	    channel_swap = [int(s) for s in args.channel_swap.split(',')]

	if args.gpu:
	    caffe.set_mode_gpu()
	    print("GPU mode")
	else:
	    caffe.set_mode_cpu()
	    print("CPU mode")

	classifier = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)
	transformer = caffe.io.Transformer({'data': classifier.blobs['data'].data.shape})
	#transformer.set_raw_scale('data', 255)
	transformer.set_transpose('data', (2,0,1))
	#transformer.set_channel_swap('data', (2,1,0))
	transformer.set_mean('data',mean.mean(1).mean(1))

	print("Reading frames from webcam...")

	time.sleep(3)

	semaphore = False

	stream = cv2.VideoCapture(0)

	with open(args.labels_file) as f:
		rawlabels = f.read().splitlines()

		labels = [r for r in rawlabels]
		print labels
		
	while semaphore == False:
		(grabbed, frame) = stream.read()

		data = transformer.preprocess('data', frame)

		classifier.blobs['data'].data[...] = data 
		start = time.time()
		out = classifier.forward()
		end = (time.time() - start)*1000

		#cv2.rectangle(frame,(5,10),(450,70),(0,0,0),-1)
		#cv2.putText(frame,"FF time: %dms/%dFPS" % (end,1000/end),
		#	(10,30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
		print("Main: Done in %.4f s." % (time.time() - start))
		
		scores = out['softmax']

		indices = scores.argmax(axis=1)[0,:,:]
		# Remove low confidence predictions
		indices[scores.max(axis=1)[0]<0.5] = 10
		
		frame_size = frame.shape[:2]
		step_size = (float(frame.shape[0])/indices.shape[0],
					 float(frame.shape[1])/indices.shape[1])
		
		for i in range(0,indices.shape[0]):
			for j in range(0,indices.shape[1]):
				pred = labels[indices[i,j]]
				position = (int((j + .5) * step_size[1]),
							int((i + .5) * step_size[0]))
				
				cv2.putText(frame,pred,
					position, cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)	
		
		cv2.imshow('test',frame)
		cv2.waitKey(1)

    	if cv2.waitKey(1) & 0xFF == ord('q'):
    		semaphore = True

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)

