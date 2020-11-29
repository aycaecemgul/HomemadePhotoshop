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

    layout = [
        [sg.HorizontalSeparator(color="White")],
        [sg.Text("Image Processing Project 1 By Ayça Ecem Gül", size=(60, 1), justification="center")],
        [sg.HorizontalSeparator(color="White")],
        [sg.Text('Choose an image', size=(14, 1)), sg.Input(key='-FILE-', enable_events=True), sg.FileBrowse()],
        [sg.Image(key='-IMAGE-',enable_events=True),
         sg.Image(key='-IMAGE1-',enable_events=True)],
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
            sg.Button("Horizontal",key="-H-FLIP-", size=(8, 1), enable_events=True),
            sg.Button("Vertical",key="-V-FLIP-" ,size=(8, 1), enable_events=True)
        ],
        [
            sg.Text("Crop", size=(14, 1)),
            sg.Input(key="-CROPX1-",size=(5,1)),sg.Input(key="-CROPX2-",size=(5,1)),sg.Input(key="-CROPY1-",size=(5,1)),sg.Input(key="-CROPY2-",size=(5,1)),
            sg.Button("Apply", size=(8, 1), key="-CROP-APPLY-", enable_events=True)
        ],
        [
            sg.Text("Swirl", size=(14, 1)),
            sg.Text("Strength", size=(10, 1)),
            sg.Slider(
                (0, 500),
                250,
                1,
                orientation="h",
                size=(10, 10),
                key="-SWIRL-SLIDER-",
            ),
            sg.Text("Radius", size=(10, 1)),
            sg.Slider(
                (0, 500),
                250,
                1,
                orientation="h",
                size=(10, 10),
                key="-SWIRL-SLIDER2-",
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
            sg.Text("Choose an image to equalize" ,size=(24, 1)), sg.Input(key='-FILE2-', enable_events=True), sg.FileBrowse(),
            sg.Button("Equalize", size=(8, 1), key="-HISTO-APPLY-", enable_events=True)
        ],
        [sg.Text("Histogram Equalization Plot"),
         sg.Button("Create",size=(8, 1), key="-HISTO-PLOT-APPLY-", enable_events=True)],
        [
            sg.HorizontalSeparator(color="White")
        ],
        [
            sg.Text("Görüntü İyileştirme İşlemleri",size=(24, 1))],
        [sg.Text("Choose a filter:",size=(12, 1)),
         sg.Combo(['Wiener',"Prewitt V","Prewitt H", "Hessian",'Median', "Meijering","Frangi","Laplacian", "Gaussian",'Sato'], enable_events=True,size=(17, 4), key='-IYI-COMBO-'),
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

    window = sg.Window("OpenCV Integration",resizable=True).Layout(
        [[sg.Column(layout,size=(800,1200), scrollable=True,grab=True,justification="c")]])

    def convert_to_bytes(file_or_bytes, resize=None):

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
        secondImage=None
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
            filename= Main.rotate_image_90(filename)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400,400)))

        elif event == "-ROT-180-":
            filename = Main.rotate_image_180(filename)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

        elif event == "-ROT-270-":
            filename = Main.rotate_image_270(filename)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

        elif event == "-H-FLIP-":
            filename= Main.h_flip(filename)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

        elif event == "-V-FLIP-":
            filename= Main.v_flip(filename)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

        elif event=="-RESIZE-APPLY-":
            if(int(values["-X-"])>0 and int(values["-X-"])>0):
                X=int(values["-X-"])
                Y=int(values["-Y-"])
                filename=Main.resize_image(filename,X,Y)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(X, Y)))
                # elif values["-THRESH-"]:
        elif event=="-SWIRL-APPLY-":
            strength=float(values["-SWIRL-SLIDER-"])
            radius=float(values["-SWIRL-SLIDER2-"])
            filename=Main.swirl_image(filename,strength,radius)
            window['-IMAGE-'].update(data=convert_to_bytes(filename,resize=(400, 400)))

        elif event=="-CROP-APPLY-":
            x1=int(values["-CROPX1-"])
            x2 = int(values["-CROPX2-"])
            y1=int(values["-CROPY1-"])
            y2=int(values["-CROPY2-"])

            filename=Main.crop_image(filename,x1,x2,y1,y2)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

        # 0 ile 1 arası bir değer alır.
        elif event == "-RESCALE-APPLY-":
            amount=float(values["-RESCALE-AMOUNT-"])
            if(0<amount<=1):
                filename = Main.rescale_image(filename, amount)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(
                400 // amount, 400 // amount)))  # GUI de gosterimi kolay olsun diye
        elif event =="-FILE2-":
            filename2 = values['-FILE2-']
            image2 = Image.open(filename2)
            filename2 = filename2[:-4] + '-converted-histogram.png'
            image2.save(filename2)
            secondImage=True
            window['-IMAGE1-'].update(data=convert_to_bytes(filename2, resize=(400, 400)))

        elif event=="-HISTO-APPLY-":
            filename,histogram_plot=Main.equalize_histogram(filename, filename2)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400,400)))
            histo=True
        elif event=="-HISTO-PLOT-APPLY-":
            histogram_plot.show()

        elif event=="-IYILESTIRME-APPLY-" :
            value=float(values["-IYI-SLIDER-"])
            if values['-IYI-COMBO-'] == "Wiener":
                print("here")
                filename=Main.wiener_filter(filename,value)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

            elif values['-IYI-COMBO-']=="Prewitt V":
                filename = Main.prewitt_V(filename)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

            elif values['-IYI-COMBO-']=="Prewitt H":
                filename = Main.prewitt_H(filename)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

            elif values['-IYI-COMBO-']=="Hessian":
                filename = Main.hessian_filter(filename, value)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

            elif values['-IYI-COMBO-']=='Median':
                filename = Main.median_filter(filename, value)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

            elif values['-IYI-COMBO-']=="Meijering":
                filename = Main.meijering_filter(filename, value)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

            elif values['-IYI-COMBO-']=="Frangi":
                filename = Main.frangi_filter(filename, value)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

            elif values['-IYI-COMBO-']=="Laplacian":
                filename = Main.laplacian_filter(filename, value)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

            elif values['-IYI-COMBO-']=="Gaussian":
                filename = Main.gaussian_filter(filename, value)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))

            elif values['-IYI-COMBO-']=='Sato':
                filename = Main.sato_filter(filename, value)
                window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(400, 400)))


    window.close()

main()