import base64
import base64
import io
import os
import PIL
from PIL import Image
import PySimpleGUI as sg
import cv2 as cv
import numpy as np
from numpy import asarray
from skimage.transform import rotate
import Main
from io import BytesIO

def main():
    sg.theme("Black")

    # Define the window layout
    layout = [
        [sg.HorizontalSeparator(color="White")],
        [sg.Text("Image Processing Project 1 By Ayça Ecem Gül", size=(60, 1), justification="center")],
        [sg.HorizontalSeparator(color="White")],
        [sg.Text('Choose an image', size=(14, 1)), sg.Input(key='-FILE-', enable_events=True), sg.FileBrowse()],
        [sg.Image(key='-IMAGE-',enable_events=True)],
        [sg.Image(key='-IMAGE1-',enable_events=True)],
        [sg.HorizontalSeparator(color="White")],
        [
            sg.Text("Rotate",size=(14,1)),
            sg.Button("90",size=(8,1),key="-ROT-90-",enable_events=True),
            sg.Button("180", size=(8, 1),key="-ROT-180-", enable_events=True),
            sg.Button("270",size=(8, 1),key="-ROT-270-", enable_events=True)
        ],
        [
            sg.Text("Resize", size=(14, 1)),
            sg.Input(key="-X-",size=(8,1)),sg.Input(key="-Y-",size=(8,1)),
            sg.Button("Apply", size=(8, 1), key="-RESIZE-APPLY-", enable_events=True)
        ],
        [
            sg.Text("Rescale",size=(14, 1)),
            sg.Input(key="-RESCALE-AMOUNT-", size=(8, 1)),
            sg.Button("Apply", size=(8, 1), key="-RESCALE-APPLY-", enable_events=True)
        ],
        [
            sg.Text("Flip",size=(14,1)),
            sg.Button("Horizontal",key="-HOR-FLIP-", size=(8, 1), enable_events=True),
            sg.Button("Vertical",key="-VER-FLIP-" ,size=(8, 1), enable_events=True)
        ],
        [
            sg.Text("Crop", size=(14, 1)),
            sg.Input(key="-CROPX1-",size=(5,1)),sg.Input(key="-CROPX2-",size=(5,1)),sg.Input(key="-CROPY1-",size=(5,1)),sg.Input(key="-CROPY2-",size=(5,1)),
            sg.Button("Apply", size=(8, 1), key="-CROP-APPLY-", enable_events=True)
        ],
        [
            sg.Text("Swirl", size=(14, 1)),
            sg.Slider(
                (0, 255),
                128,
                1,
                orientation="h",
                size=(20, 10),
                key="-SWIRL SLIDER-",
            ),
            sg.Button("Apply", size=(8, 1), key="-SWIRL-APPLY-", enable_events=True)
        ],

        [
            sg.HorizontalSeparator(color="White")
        ],
        [
            sg.Text("Histogram Equalization",size=(24,1))
        ],
        [
            sg.Text("Choose an image to equalize" ,size=(24, 1)), sg.Input(key='-FILE2-', enable_events=True), sg.FileBrowse()
        ],
        [
            sg.HorizontalSeparator(color="White")
        ],
        [
            sg.Text("Görüntü İyileştirme İşlemleri",size=(24, 1))],
        [sg.Text("Choose a filter:",size=(12, 1)),
         sg.Combo(['Wiener',"Prewitt V","Prewitt H", "Hessian",'Median ', "Meijering","Frangi","Laplacian", "Gaussian",'Sato'], enable_events=True,size=(17, 4), key='-IYI-COMBO-'),
         sg.Slider(
             (0, 255),
             128,
             1,
             orientation="h",
             size=(20, 10),
             key="-IYI-SLIDER-",
         ),
         sg.Button("Apply", size=(8, 1), key="-IYILESTIRME-APPLY-", enable_events=True)],
        [
            sg.HorizontalSeparator(color="White")
        ],
        [
            sg.Text("Yoğunluk Dönüşümü İşlemleri")
        ],
        [
            sg.Combo(["","","","","",""],size=(17,3),enable_events=True,key="-YOG-COMBO-"),
            sg.Button("Apply", size=(10, 1), key="-YOG-APPLY-", enable_events=True)
        ],
        [
            sg.HorizontalSeparator(color="White")
        ],
        [sg.Text("Morphological Operations", size=(18, 1))],
        [
            sg.Text("Choose an operation:", size=(18, 1)),
            sg.Combo(['Dilation', 'Erosion', 'Thin','Skeletonize','Skeletonize-3D',
                      'Opening',"Closing","Convex Hull","White Tophat","Black Tophat"],
                     size=(17, 4), key='-MORP-COMBO-',enable_events=True),
         sg.Button("Apply", size=(10, 1), key="-IYILESTIRME-APPLY-", enable_events=True)
         ],
        [
            sg.HorizontalSeparator(color="White")
        ],
        [sg.Text("Try Ayça's special instagram filter!", size=(27, 1)),
         sg.Button("Apply", size=(8, 1), key="-INSTA-APPLY-", enable_events=True)
         ],
        [
            sg.Button("Save", size=(8, 1)),
         sg.Button("Exit", size=(8, 1))
        ]
    ]

    # Create the window and show it without the plot
    window = sg.Window("OpenCV Integration", layout, location=(800, 400),resizable=True)

    def image_to_bytes(img):
        im_file = BytesIO()
        img.save(im_file, format="PNG")
        im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
        im_b64 = base64.b64encode(im_bytes)
        return im_b64
    def convert_to_bytes(file_or_bytes, resize=None):
        '''
        Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
        Turns into  PNG format in the process so that can be displayed by tkinter
        :param file_or_bytes: either a string filename or a bytes base64 image object
        :type file_or_bytes:  (Union[str, bytes])
        :param resize:  optional new size
        :type resize: (Tuple[int, int] or None)
        :return: (bytes) a byte-string object
        :rtype: (bytes)
        '''
        if isinstance(file_or_bytes, str):
            img = PIL.Image.open(file_or_bytes)
        else:
            try:
                img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
            except Exception as e:
                dataBytesIO = io.BytesIO(file_or_bytes)
                img = PIL.Image.open(dataBytesIO)

        cur_width, cur_height = img.size
        if resize:
            new_width, new_height = resize
            scale = min(new_height / cur_height, new_width / cur_width)
            img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.ANTIALIAS)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-FILE-':
            filename = values['-FILE-'] #!!!!!!!!!!
            image=Image.open(filename)
            filename=filename[:-4] + '-converted.png'
            image.save(filename)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400,400)))
        elif event == "-ROT-90-" :
            image=Image.open(filename)
            image=asarray(image)
            rotated_image=cv.rotate(image,cv.ROTATE_90_CLOCKWISE)
            rotated_image=Image.fromarray(rotated_image)
            rotated_image.save(filename)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400,400)))
        # elif values["-THRESH-"]:
        #     if filename != None:
        #         image=Image.open(filename)
        #         frame = asarray(image)
        #         window['-IMAGE-'].update(data=convert_to_bytes(frame, resize=(400, 400)))
        # elif values["-CANNY-"]:
        #     frame = cv2.Canny(
        #         frame, values["-CANNY SLIDER A-"], values["-CANNY SLIDER B-"]
        #     )



        # elif values["-HUE-"]:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #     frame[:, :, 0] += int(values["-HUE SLIDER-"])
        #     frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        # elif values["-ENHANCE-"]:
        #     enh_val = values["-ENHANCE SLIDER-"] / 40
        #     clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
        #     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        #     lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        #     frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        #
        # imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        # window["-IMAGE-"].update(data=imgbytes)

    window.close()

main()