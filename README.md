## **Deployed App on Streamlit link :-** [click here](https://yashkumbalkar-emotion-detection-from-text-app-48vzim.streamlit.app/)

# Emotion Detection from Text using DistilBERT

### Overview :-

This project uses **DistilBERT**, a smaller, faster, and lighter version of BERT, for detecting emotions from text. 
The model is fine-tuned on a text dataset and deployed as an interactive web application using **Streamlit**.

### Data Source :-

The dataset used for this project is sourced from HuggingFace:- [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)

### Project Description :-

The model is based on DistilBERT, which has been fine-tuned on a custom dataset. The model currently detects the following 
emotions:- `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`.

## Technologies Used :-

- **DistilBERT**: A pre-trained transformer model fine-tuned for text classification tasks.
- **Streamlit**: A powerful tool for creating interactive web apps with minimal code.
- **Python**: The main programming language for building the app and model.
- **Hugging Face Transformers**: Provides pre-trained models and tools for text processing.
- **scikit-learn**: For any additional preprocessing or performance evaluation.

### Example Usage :-

- Open the deployed Streamlit app.
- Enter an English sentence in the text input field.
- Click the "Detect Emotion" button to get the prediction.



