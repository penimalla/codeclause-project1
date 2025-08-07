# codeclause-project1
#codeclause project 1

import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img, faces

def show_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    detected_img, faces = detect_faces(file_path)
    b,g,r = cv2.split(detected_img)
    img_rgb = cv2.merge((r,g,b))
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk
    status_text.set(f"Detected {len(faces)} face(s).")

root = tk.Tk()
root.title("Image Recognition with OpenCV")
root.geometry("600x600")

Label(root, text="Image Recognition System", font=("Helvetica", 18, "bold")).pack(pady=10)
Button(root, text="Upload Image", command=show_image, font=("Helvetica", 14)).pack(pady=10)

panel = Label(root)
panel.pack(pady=10)

status_text = tk.StringVar()
status_label = Label(root, textvariable=status_text, font=("Helvetica", 12))
status_label.pack()

root.mainloop()
