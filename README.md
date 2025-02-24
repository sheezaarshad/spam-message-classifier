# Spam Detection System

## Project Overview
This project is a **Spam Detection System** built using **Python, Scikit-Learn, and Streamlit**. It uses **Natural Language Processing (NLP)** techniques to classify messages as **spam** or **ham** (not spam). The system is trained on a dataset and provides a simple web interface using Streamlit.

## Features
- Preprocesses text data by removing stopwords and punctuation.
- Converts text into numerical features using **TF-IDF Vectorization**.
- Trains a **Naïve Bayes classifier** to detect spam messages.
- Provides a **Streamlit-based web interface** for user-friendly interaction.
- Displays model accuracy after training.

## Installation
### Prerequisites
Ensure you have Python installed. You can download it from [Python.org](https://www.python.org/downloads/).

### Install Dependencies
Run the following command to install required Python libraries:
```sh
pip install -r requirements.txt
```

## Running the Project
### Step 1: Train the Model
Before using the frontend, train the spam detection model by running:
```sh
python spam.py
```
This will generate `model.pkl` and `vectorizer.pkl` files.

### Step 2: Run the Streamlit App
Once the model is trained, start the Streamlit web app with:
```sh
streamlit run app.py
```
This will open a browser window where you can input messages to check if they are spam or not.


## Technologies Used
- **Python**
- **Scikit-Learn**
- **NLTK** (Natural Language Toolkit)
- **Pandas & NumPy**
- **Streamlit** (for frontend UI)

## Example Usage
1. Enter a message in the input field.
2. Click the **Check** button.
3. The app will classify the message as **Spam** or **Ham**.

## Acknowledgments
This project was built using public spam datasets and various open-source tools.


