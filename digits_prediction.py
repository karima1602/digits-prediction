import tkinter as tk 
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2 as cv
import joblib


global img
img = None

# Load the trained model
model = joblib.load('model_digits.pkl')


#Fonction pour charger et afficher l'image
def load_image():
    global img
    path_img = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif")])
    if path_img:
        img = cv.imread(path_img)
        # Convertir l'image de BGR à RGB
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        tk_image = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        # Afficher l'image chargée sur le premier label
        lbl_img.config(image=tk_image)
        lbl_img.image = tk_image  # Gardez une référence pour éviter que l'image ne soit effacée par le garbage collector


def predict_digit():  # Passer l'image en tant qu'argument
    global img
    if img is None:  # Vérifier si une image a été passée
        messagebox.showerror("Error", "No image loaded.")
        return

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    resized_img = cv.resize(gray_img, (8, 8))
    input_data = resized_img.flatten().reshape(1, -1)
    prediction = model.predict(input_data)
    messagebox.showinfo("Prediction", f"Prediction : {prediction[0]}")

window = tk.Tk()
window.title("Digits prediction")
window.geometry("900x700")
window.configure(bg="#aebaff")
window.resizable(False, False)



btn_upload = tk.Button(window, text="Upload image", command=load_image, bg="#268290", fg="white", font=("Arial", 20))
btn_upload.pack()

lbl_img = tk.Label(window)
lbl_img.pack()

predict_button = tk.Button(window, text="Predict" , font=("Arial", 20), bg = "#268290", fg="white", command=lambda: predict_digit())
predict_button.pack()

window.mainloop()
