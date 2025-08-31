import torch
import gradio as gr
from transformers import ConvNextForImageClassification, AutoImageProcessor
from PIL import Image

class_names = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos',
             'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases',
               'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles',
                 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis',
               'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors',
                 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']

model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224")

# Redefine classifier for 23 classes
model.classifier = torch.nn.Linear(in_features=1024, out_features=23)

# Load model configuration and weights manually
model.load_state_dict(torch.load("convnext_base_finetuned.pth", map_location="cpu"))  # Load your finetuned weights
model.eval()

# Load the processor
processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224")

# Define a function to predict the class from an image
def predict(image):
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return class_names[predicted_class]


# Create Gradio interface for user input
iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Textbox())

iface.launch()
