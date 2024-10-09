from fastai.vision.all import *
import gradio as gr
from PIL import Image

# Load the trained model
model = load_learner('pet-breed-recognizer-v2.pkl')

# Get the labels
cap_labels = model.dls.vocab

# Define the prediction function
def recognize_image(image):
    image = Image.fromarray(image).resize((192, 192))
    pred, idx, probs = model.predict(image)
    return dict(zip(cap_labels, map(float, probs)))

image = gr.Image(type="numpy")  
label = gr.Label(num_top_classes=5)

# Add example images
examples = [
    'unknown00.jpg',
    'unknown01.jpg',
    'unknown02.jpg',
    'unknown03.jpg'
]

# Create the Gradio interface
iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)
