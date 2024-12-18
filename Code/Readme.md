# Image Caption Generation Project

## Overview
This project implements an image caption generation system using the Flickr 8k dataset. It includes scripts for data preprocessing, model training, evaluation, and a Streamlit-based UI for generating captions for user-uploaded images.

## Pre-requistes

Run the below command to download all the respective libraries

pip install -r requirements.txt

Trained model weights, requirements.txt and tokenizer.pkl file can be found in this drive link 
https://drive.google.com/drive/folders/160x5k6eGySmgP777AifIwEu87zpbpaaj?usp=sharing
---

## Steps to Use

### 1. **Download the Dataset**
Download the Flickr 8k dataset from [this link](https://www.kaggle.com/datasets/adityajn105/flickr8k). Ensure that the `Images` folder and `captions.txt` file are accessible in your local directory.

### 2. **Modify File Paths**
In the `image_caption_generation.py` file, update the paths for:
- **`Images` folder**: Point to where the dataset images are stored.
- **`captions.txt` file**: Point to the location of the captions file.

### 3. **Run the Model Training Script**
Once the file paths are updated, run the `image_caption_generation.py` script. This script:
- Preprocesses the data.
- Defines the model architecture.
- Trains the model and saves the checkpoints.
- Evaluates the model's performance.
- Generates checkpoints of best models, set the path for this as well in the code 

### 4. **Run the Streamlit Application**
To launch the user interface:
- Open the `streamlit_app.py` file.
- Ensure the correct paths to the saved model checkpoints are provided.
- (Optional) If you need to regenerate the `tokenizer.pkl` file, uncomment the relevant line in `image_caption_generation.py` and rerun the script.
- Run the Streamlit app to start the UI where users can upload an image and view the generated captions.

---

## File Descriptions

### `image_caption_generation.py`
- Contains the full workflow: data preprocessing, model definition, training, and evaluation.
- Outputs model checkpoints for later use.

### `streamlit_app.py`
- A Streamlit-based user interface for the project.
- Allows users to upload images and view caption predictions.

---

