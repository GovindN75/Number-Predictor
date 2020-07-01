# Dependencies needed for the actual GUI
import tensorflow as tf
import numpy as np
import cv2 
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import PIL
from PIL import ImageGrab, Image, ImageDraw
import os

# define variables for the GUI
WIDTH = 560
HEIGHT = 560
BRUSH_SIZE = 10
WHITE = (255, 255, 255)

# Load our saved model back in
model=tf.keras.models.load_model('digit_recognition.model')

# Define some methods to make our code clean

# Processes the image of the canvas and inputs it into the model
def predict_digit():
    img=cv2.imread('img.png',0)
    # Our image right now has a white background and black lines. Looking at the original MNIST dataset, it has a black background and white lines.
    # The bitwise_not function converts the normal picture to the one that looks like the ones in the MNIST dataset.
    img=cv2.bitwise_not(img) 
    img=cv2.resize(img,(28,28))
    img=img.reshape(1,28,28,1)
    img=img.astype('float32')
    img=img/255.0

    ans=model.predict(img)[0]
    return np.argmax(ans), max(ans)


# Drawing method that allows the user to draw lines on the canvas
def draw_digit(event):
    oldx, oldy = (event.x-BRUSH_SIZE), (event.y-BRUSH_SIZE)
    newx, newy = (event.x+BRUSH_SIZE), (event.y + BRUSH_SIZE)
    canvas.create_oval(oldx, oldy, newx, newy, fill="black", width=30)
    drawing.line([oldx, oldy, newx, newy], fill="black", width=30)

# Clears the screen
def clear():
    canvas.delete("all")
    drawing.rectangle((0, 0, WIDTH, HEIGHT), fill=(255, 255, 255, 0))

# this is the function that takes the overlaying PIL image and feeds it into the preprocessing method defined above
def predicting_drawing():
    filename = "img.png"
    image_overlay.save(filename)
    prediction, accuracy=predict_digit()
    messagebox.showinfo("Prediction", "I predict this number is " + str(prediction) + " with an accuracy of " + str(accuracy*100))
    #print(str(prediction) + " " + str(accuracy*100))

master=Tk()
master.resizable(0, 0)
canvas = Canvas(master, width=WIDTH, height=HEIGHT, bg="white")
canvas.grid(row=0, column=0, columnspan=2)
image_overlay = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
drawing = ImageDraw.Draw(image_overlay)

# Tracks the mouse motion on the canvas and lets it draw lines
canvas.bind("<B1-Motion>", draw_digit)

# UI for the GUI
predict_button=Button(master, text="Predict", padx=20, pady=20, command=predicting_drawing)
clear_button=Button(master, text="Clear", padx=20, pady=20, command=clear)
predict_button.grid(row=1, column=0)
clear_button.grid(row=1, column=1)

master.title('Number Predictor')
master.mainloop()


# THANK YOU LINK: https://www.semicolonworld.com/question/55284/how-can-i-convert-canvas-content-to-an-image I got the idea for the secondary PIL image on top of the canvas from this link. 