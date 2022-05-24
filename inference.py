"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
upda
"""
import argparse

import os
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from PIL import Image
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.layers import Input,Lambda
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from Model.models.darknet_v3_detection_head import preprocess_true_boxes,  yolo_body, tiny_yolo_body, yolo_loss,yolo_eval
from Model.utils.draw_boxes import draw_boxes




def _main(args):
    data_path = os.path.expanduser(args.data_path)  
    classes_path = os.path.expanduser(args.classes_path) 
    anchors_path = os.path.expanduser(args.anchors_path)
    weight_path = os.path.expanduser(args.weight_path)
    model_name = args.model_name
    image_size = (416,416)
    if model_name == "mobilnetV1":
        image_size = (224,320)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    anchors = YOLO_ANCHORS


    alpha = 0.5
    model_body, model = create_tiny_model(model_name = model_name, alpha=alpha, input_shape=image_size, anchors=anchors, num_classes=num_classes, freeze_body=False, light_head=False)


    draw(model_body,
        class_names,
        anchors,
        data_path,
        image_size,
        weights_name=weight_path,
        save_all=True)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(input_shape[0], input_shape[1], 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model



def save_results(image,box_classes,class_names,boxes,scores, name):
    image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))
    out_file = open('out_label/{}.txt'.format(name), 'w')
    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        if isinstance(scores, np.ndarray):
            score = scores[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        out_file.write(box_class+" "+str(score) + " " +str(left)+" "+str(top)+" "+str(right)+" "+str(bottom) + '\n')
        #print(label, (left, top), (right, bottom))

def Eval(model_body, class_names, anchors, data_path, image_size, weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    assert image_size[0]%32 == 0, 'Multiples of 32 required'
    assert image_size[1]%32 == 0, 'Multiples of 32 required'
	
	

    model_body.load_weights(weights_name)
	
    input_image_shape = K.placeholder(shape=(2, ))
	
    boxes, scores, classes = yolo_eval(model_body.output, anchors, len(class_names), input_image_shape, score_threshold=0.8, iou_threshold=0.2)	
				

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
		
    img_list = os.listdir(data_path)
    img_list = [i for i in img_list if not "sub" in i]
	
    for index, name in enumerate(img_list):
        image = Image.open(data_path+name)
        h, w = image_size
		




        image = image.resize((w,h))
        image = np.array(image)
        if len(image_data.shape)==2:
            image_data = np.expand_dims(image_data, axis=2)
            image_data = np.concatenate((image_data,image_data,image_data),axis=2)	
        image = image/255.

		
        out_boxes, out_scores, out_classes = sess.run( [boxes, scores, classes],
                                                      feed_dict={model_body.input: image,
                                                                 input_image_shape: [image.shape[1], image.shape[2]],
                                                                 K.learning_phase(): 0})
        save_results(image,out_classes,class_names,out_boxes,out_scores, name.split(".")[0])	


        #-----------------		
        image_with_boxes = draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
        # Save the image:
        if save_all:
            image = Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,'{}.png'.format(index)))


