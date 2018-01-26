import os, tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
import base64


txt_img_path = "/Users/danayu/TensorFlow/hedge_detector/LabelBB/LabelBoundingBox/Labels/hedgehog/"
img_path = "/Users/danayu/TensorFlow/hedge_detector/LabelBB/LabelBoundingBox/Images/hedgehog_40/"

def read_file(filen):

    txtfile = txt_img_path+filen+".txt"
    txtfile_contents = open(txtfile, "r")
    txtlines = txtfile_contents.readlines()
    return txtlines


def create_tf_example(img_name):

    imgfile = img_path+img_name+".jpg"

    im = Image.open(imgfile)

    filename = str.encode(img_name)  # Filename of the image. Empty if image is not from file

    height = im.height
    width = im.width

    #img_file = open(imgfile, 'rb')
    #encoded_image_data = base64.b64encode(img_file.read())  # Encoded image bytes

    image_data = tf.gfile.FastGFile(imgfile,'rb').read()

    #image_format = b'jpeg'

    test = imgfile.split('.')
    stripped = test[1]
    stripped =stripped.strip()

    image_format = str.encode(stripped)


    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes_int = []

    # open corresponding txt file

    txtlines= read_file(img_name)

    for line in txtlines:
        l_array = line.split()

        xmins.append(float(l_array[2])/width)
        xmaxs.append(float(l_array[4])/width)
        ymins.append(float(l_array[3])/height)
        ymaxs.append(float(l_array[5])/height)

        classes_text.append(str.encode(l_array[1]))
        classes_int.append(1)

    print(filename, height, width, xmins, xmaxs, ymins, ymaxs, classes_text, classes_int)

    tf_example = tf.train.Example(features = tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes_int),

    })

    )

    return tf_example

#create_tf_example("pic_012", img_path)


# def read_images_dir():
#
#     img_path = "/Users/danayu/TensorFlow/hedge_detector/LabelBB/LabelBoundingBox/Images/test/"
#
#     images = os.listdir(img_path)
#
#     # read each image from images dir
#     for im in images:
#
#         split_line = im.split('.')
#
#         print(split_line[0])


# create tf example for each image and its accompanying txt label info
# serialize and write each tfexample to tfrecord writer

def main():

    img_path = "/Users/danayu/TensorFlow/hedge_detector/LabelBB/LabelBoundingBox/Images/hedgehog_40/"

    tfr_filename ='val3.tfrecords'

    writer = tf.python_io.TFRecordWriter('/Users/danayu/TensorFlow/hedge_detector/LabelBB/LabelBoundingBox/tfrecord/'+tfr_filename)

    images = os.listdir(img_path)

    # read each image from images dir
    for im in images:
        split_line = im.split('.')
        tf_example = create_tf_example(split_line[0])
        writer.write(tf_example.SerializeToString())

    writer.close()

main()