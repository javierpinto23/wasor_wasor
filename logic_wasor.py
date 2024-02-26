from tkinter import *
from PIL import Image,ImageTk
import cv2, imutils
import numpy as np
from ultralytics import YOLO
import math

# Show color recycling containers and additional information
def Recyling_Containers(img, img_txt):
    img = img
    img_txt = img_txt

    # Image detect
    img = np.array(img, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    img_ = ImageTk.PhotoImage(image=img)
    label_img.configure(image=img_)
    label_img.image = img_

    img_txt = np.array(img_txt, dtype='uint8')
    img_txt = cv2.cvtColor(img_txt, cv2.COLOR_BGR2RGB)
    img_txt = Image.fromarray(img_txt)

    txt_img = ImageTk.PhotoImage(image=img_txt)
    label_img_txt.configure(image=txt_img)
    label_img_txt.image = txt_img

# Scanning function
def Scanning():
    global label_img, label_img_txt

    # Interface
    label_img = Label(window_main)
    label_img.place(x=75,y=260)
    label_img_txt = Label(window_main)
    label_img_txt.place(x=995,y=310)

    # Read video capture
    if cap is not None:
        ret, frame = cap.read()
        frame_show = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if ret == True:

            # Results
            results = model(frame, stream=True, verbose=False, max_det=1,conf=0.5)
            for res in results:
                # Box
                boxes = res.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Fix Error
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 < 0: x2 = 0
                    if y2 < 0: y2 = 0

                    # Class
                    cls = int(box.cls[0])

                    # Confidence
                    conf = math.ceil(box.conf[0])

                    if conf > 0.6:
                        # ORGANIC
                        if cls == 0:
                            # Draw rectangle
                            cv2.rectangle(frame_show,(x1,y1),(x2,y2), (0, 128, 0),2)

                            # Text
                            text = f'{clasess_name[0]} {int(conf)*100}%'
                            sizetext = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            cv2.rectangle(frame_show,(x1,y1 -dim[1] - baseline), (x1 + dim[0],y1 + baseline),(0,0,0), cv2.FILLED)
                            cv2.putText(frame_show,text,(x1,y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 128, 0),2)

                            # Image container
                            Recyling_Containers(ima_organic,ima_organic_txt)

                        # BATTERIES
                        if cls == 1:
                            # Draw rectangle
                            cv2.rectangle(frame_show,(x1,y1),(x2,y2), (128, 128, 0),2)

                            # Text
                            text = f'{clasess_name[1]} {int(conf)*100}%'
                            sizetext = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            cv2.rectangle(frame_show,(x1,y1 -dim[1] - baseline), (x1 + dim[0],y1 + baseline),(0,0,0), cv2.FILLED)
                            cv2.putText(frame_show,text,(x1,y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,1,(128, 128, 0),2)

                            # Image container
                            Recyling_Containers(ima_batteries, ima_batteries_txt)

                        # GLASS
                        if cls == 2:
                            # Draw rectangle
                            cv2.rectangle(frame_show,(x1,y1),(x2,y2), (0, 255, 255),2)

                            # Text
                            text = f'{clasess_name[2]} {int(conf)*100}%'
                            sizetext = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            cv2.rectangle(frame_show,(x1,y1 -dim[1] - baseline), (x1 + dim[0],y1 + baseline),(0,0,0), cv2.FILLED)
                            cv2.putText(frame_show,text,(x1,y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2)

                            # Image container
                            Recyling_Containers(ima_glass, ima_glass_txt)

                        # METAL
                        if cls == 3:
                            # Draw rectangle
                            cv2.rectangle(frame_show,(x1,y1),(x2,y2), (128, 0, 0),2)

                            # Text
                            text = f'{clasess_name[3]} {int(conf)*100}%'
                            sizetext = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            cv2.rectangle(frame_show,(x1,y1 -dim[1] - baseline), (x1 + dim[0],y1 + baseline),(0,0,0), cv2.FILLED)
                            cv2.putText(frame_show,text,(x1,y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,1,(128, 0, 0),2)

                            # Image container
                            Recyling_Containers(ima_metal, ima_metal_txt)

                        # PAPER
                        if cls == 4:
                            # Draw rectangle
                            cv2.rectangle(frame_show,(x1,y1),(x2,y2), (255,255,0),2)

                            # Text
                            text = f'{clasess_name[4]} {int(conf)*100}%'
                            sizetext = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            cv2.rectangle(frame_show,(x1,y1 -dim[1] - baseline), (x1 + dim[0],y1 + baseline),(0,0,0), cv2.FILLED)
                            cv2.putText(frame_show,text,(x1,y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

                            # Image container
                            Recyling_Containers(ima_paper, ima_paper_txt)

                        # PLASTIC
                        if cls == 5:
                            # Draw rectangle
                            cv2.rectangle(frame_show,(x1,y1),(x2,y2), (255, 0, 0),2)

                            # Text
                            text = f'{clasess_name[5]} {int(conf)*100}%'
                            sizetext = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            cv2.rectangle(frame_show,(x1,y1 -dim[1] - baseline), (x1 + dim[0],y1 + baseline),(0,0,0), cv2.FILLED)
                            cv2.putText(frame_show,text,(x1,y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)

                            # Image container
                            Recyling_Containers(ima_plastic, ima_plastic_txt)

                        # BIOLOGICAL
                        if cls == 6:
                            # Draw rectangle
                            cv2.rectangle(frame_show,(x1,y1),(x2,y2), (255, 0, 0),2)

                            # Text
                            text = f'{clasess_name[6]} {int(conf)*100}%'
                            sizetext = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            cv2.rectangle(frame_show,(x1,y1 -dim[1] - baseline), (x1 + dim[0],y1 + baseline),(0,0,0), cv2.FILLED)
                            cv2.putText(frame_show,text,(x1,y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)

                            # Image container
                            Recyling_Containers(ima_biological, ima_biological_txt)

            # Resize
            frame_show = imutils.resize(frame_show, width=640)

            # Covert video
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)

            # Display
            label_video.configure(image=img)
            label_video.image = img
            label_video.after(10, Scanning)
        else:
            cap.release()



def Main_Window():
    global model, clasess_name, ima_organic,ima_batteries,ima_glass,ima_metal,ima_paper,ima_plastic,ima_biological
    global ima_organic_txt,ima_batteries_txt,ima_glass_txt,ima_metal_txt,ima_paper_txt,ima_plastic_txt,ima_biological_txt,cap,label_video,window_main

    window_main = Tk()
    window_main.title("WASTE SORTING")
    window_main.geometry("1280x720")

    # Background
    canva_img = PhotoImage(file="/Users/pintojav/Desktop/wasor_wasor/wasor_wasor/layout/canva_adidas.png")
    background = Label(image=canva_img)
    background.place(x=0,y=0,relwidth=1,relheight=1)

    # Model
    model = YOLO('/Users/pintojav/Desktop/wasor_wasor/wasor_wasor/wasor_wasor/runs2/weights/best.pt')

    # Classes
    clasess_name = ['Organic', 'Batteries', 'Glass', 'Metal', 'Paper', 'Plastic', 'Biological']


    # Images classes
    ima_organic = cv2.imread("layout/organic.png")
    ima_batteries = cv2.imread("layout/batteries.png")
    ima_glass = cv2.imread("layout/glass.png")
    ima_metal = cv2.imread("layout/metal.png")
    ima_paper = cv2.imread("layout/paper.png")
    ima_plastic = cv2.imread("layout/plastic.png")
    ima_biological = cv2.imread("layout/plastic.png")

    ima_organic_txt = cv2.imread("layout/organic.png")
    ima_batteries_txt = cv2.imread("layout/batteries.png")
    ima_glass_txt = cv2.imread("layout/glass.png")
    ima_metal_txt = cv2.imread("layout/metal.png")
    ima_paper_txt = cv2.imread("layout/paper.png")
    ima_plastic_txt = cv2.imread("layout/plastic.png")
    ima_biological_txt = cv2.imread("layout/plastic.png")

    # video label
    label_video = Label(window_main)
    label_video.place(x=320, y=130)

    # Cam
    cap = cv2.VideoCapture(0)
    cap.set(3,590)
    cap.set(4,500)

    # Scanning
    Scanning()

    # loop
    window_main.mainloop()

if __name__ == '__main__':
    Main_Window()