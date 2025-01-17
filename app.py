import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from googletrans import Translator, LANGUAGES
from transformers import CLIPProcessor, CLIPModel
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX  # Using SARIMAX for forecasting
import pandas as pd  # Add this line to import pandas
from sklearn.ensemble import IsolationForest  # For fraud detection
import numpy as np

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini pro vision model
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize CLIP model and processor from Hugging Face
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Function to get Gemini response
def get_gemini_response(input, image, user_prompt):
    response = model.generate_content([input, image[0], user_prompt])
    return response.text


# Function to get image details from uploaded file
def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


# Function: Multimodal Analysis (Text and Image Matching using CLIP)
def multimodal_analysis(image_path, user_query):
    """
    Perform multimodal analysis by matching user query (text) with image data (invoice).
    This function uses CLIP model from Hugging Face to evaluate how the text matches the image.
    """
    # Open the image file and preprocess it for the CLIP model
    image = Image.open(image_path)
    inputs = processor(
        text=[user_query], images=image, return_tensors="pt", padding=True
    )

    # Run the CLIP model to get similarity scores between the image and text
    outputs = clip_model(**inputs)
    logits_per_text = outputs.logits_per_text
    scores = logits_per_text.softmax(dim=1).tolist()  # Convert logits to probabilities

    return scores


# Function for Predictive Analytics (Forecasting using SARIMAX)
def forecast_invoice_trends_sarimax(historical_data):
    """
    Predict future invoice trends using the SARIMAX model.
    historical_data should be a list of dictionaries with 'ds' (date) and 'y' (value).
    """
    # Convert data to DataFrame
    df = pd.DataFrame(historical_data)
    df["ds"] = pd.to_datetime(df["ds"])  # Ensure 'ds' is in datetime format
    df.set_index("ds", inplace=True)

    # Fit SARIMAX model (adjust the parameters as necessary)
    model = SARIMAX(
        df["y"],
        order=(1, 1, 1),  # AR, I, MA terms (adjust as necessary)
        seasonal_order=(1, 1, 1, 12),  # Seasonality of 12 months (adjust if needed)
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit()

    # Forecast the next 12 months
    forecast = results.get_forecast(steps=12)
    forecast_index = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1), periods=12, freq="M"
    )

    # Get forecasted values and confidence intervals
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Create a DataFrame for the forecasted results
    forecast_df = pd.DataFrame(
        {
            "ds": forecast_index,
            "yhat": forecast_values,
            "yhat_lower": conf_int.iloc[:, 0],
            "yhat_upper": conf_int.iloc[:, 1],
        }
    )

    return forecast_df


# Function for Fraud Detection using Isolation Forest
def detect_invoice_fraud(historical_data):
    """
    Detect fraud in the invoice data using Isolation Forest.
    """
    # Convert historical data to DataFrame
    df = pd.DataFrame(historical_data)
    df["ds"] = pd.to_datetime(df["ds"])  # Ensure 'ds' is in datetime format

    # Prepare features for fraud detection (e.g., using invoice amounts)
    X = df[["y"]].values  # Using 'y' as the feature (invoice amounts)

    # Initialize Isolation Forest model for anomaly detection
    model = IsolationForest(contamination=0.05)  # 5% contamination assumed
    df["fraud_score"] = model.fit_predict(X)

    # Mark potential frauds with a fraud score of -1 (indicating anomalies)
    fraud_df = df[df["fraud_score"] == -1]

    return fraud_df


# Initialize Streamlit app
st.set_page_config(page_title="MultiLanguage Invoice Extractor")
st.header("MultiLanguage Invoice Extractor")

input = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader(
    "Choose an image of the invoice...", type=["jpg", "jpeg", "png"]
)

# Create a mapping of language names to their codes
language_code_mapping = {LANGUAGES[key]: key for key in LANGUAGES.keys()}

# Language selection dropdown
language = st.selectbox(
    "Select language for response:", options=list(language_code_mapping.keys())
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me about the invoice")

input_prompt = """
You are an expert in understanding invoices. We will upload an image as invoice in different languages
and you will have to answer any questions based on the uploaded invoice image in English language only in detail.
"""

# If submit button is clicked
if submit:
    image_data = input_image_details(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)

    # Get the language code from the selected language name
    selected_language_code = language_code_mapping[language]

    # Translate the response
    translator = Translator()
    translated_response = translator.translate(
        response, dest=selected_language_code
    ).text

    # Display the AI-generated response
    st.subheader("The Response is")
    st.write(translated_response)

    # Perform multimodal analysis
    st.subheader("Multimodal Analysis Results:")
    multimodal_scores = multimodal_analysis(uploaded_file, input)
    st.write(f"Similarity score between the query and image: {multimodal_scores[0]}")

    # Predictive Analytics (Forecasting)
    st.subheader("Forecasting Invoice Trends:")
    # Example historical data (past invoice amounts)
    historical_data = [
        {"ds": "2023-01-01", "y": 200},
        {"ds": "2023-02-01", "y": 150},
        {"ds": "2023-03-01", "y": 300},
        {"ds": "2023-04-01", "y": 250},
        {"ds": "2023-05-01", "y": 350},
        {"ds": "2023-06-01", "y": 400},
    ]

    forecast = forecast_invoice_trends_sarimax(historical_data)

    # Display the forecasted data
    st.write("Forecasted Invoice Trends:")
    st.write(forecast)

    # Plot the forecast
    st.line_chart(forecast.set_index("ds")[["yhat"]])

    # Fraud Detection (Flagging Potential Fraud)
    st.subheader("Fraud Detection Results:")
    fraud_df = detect_invoice_fraud(historical_data)

    if not fraud_df.empty:
        st.write("Potential Fraud Detected in the Following Invoices:")
        st.write(fraud_df[["ds", "y", "fraud_score"]])
    else:
        st.write("No fraud detected.")
