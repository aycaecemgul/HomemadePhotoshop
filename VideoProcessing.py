import os
import cv2
import moviepy as moviepy
import moviepy.video.io.ImageSequenceClip
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import load_img


style_path=input("Enter the style image file path (please use / backslash when typing in directory path):")
video_path=input("Enter the video file path (please use / backslash when typing in directory path):")


style = load_img(style_path)

def reshape_image(image_path):
    dim = 450
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [dim, dim])
    img = img[tf.newaxis, :]
    return img

style = reshape_image(style_path)

content_layers = ['block4_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style * 255)
style_extractor.save("styleExtractor")


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


extractor = StyleContentModel(style_layers, content_layers)

# gradient descent
style_targets = extractor(style)['style']

#define a function to keep the pixel values between 0 and 1:
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Custom weights for style and content updates
style_weight = 200
content_weight = 1e6

# The loss function to optimize
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    content_targets = extractor(content)['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

total_variation_weight=800

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


def tensor_to_image(tensor,path):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    tf.keras.preprocessing.image.save_img(path,tensor)

def video_to_frames(input_loc, output_loc):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    cap = cv2.VideoCapture(input_loc)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    frame_count=0
    while cap.isOpened():
        ret, frame = cap.read()
        if (count % 4 == 1):
            cv2.imwrite(output_loc + "/%2d0.jpeg" % (frame_count), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            frame_count=frame_count+1
        count=count+1
        if (count > (video_length - 1)):
            cap.release()
            break

    return frame_count

file_path="Frames\\"

frames=video_to_frames(video_path,file_path)

epochs = 10
steps_per_epoch = 70

def load_content(file_path,image_path):
    content_path = os.path.join(file_path,image_path)
    content = reshape_image(content_path)
    return content

index = 0

for image_path in os.listdir(file_path):
    content=load_content(file_path,image_path)
    image = tf.Variable(content)
    print(image_path)
    for n in range(epochs):
        print("epoch: %s" % n )
        for m in range(steps_per_epoch):
            train_step(image)
    # plt.imshow(np.squeeze(image.read_value(), 0))
    # plt.show()
    file_name = 'results\\' + "%s" % (index) + ".jpeg"
    tensor_to_image(image, file_name)
    index= index +1

def frames_to_video(input_path,fps):
    image_files = [input_path+'/'+img for img in os.listdir(input_path)]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile('result.mp4')

input_path = 'results'

fps = 10
frames_to_video(input_path,fps)