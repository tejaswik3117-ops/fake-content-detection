import gradio as gr
import requests
import json
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000/detect"

def analyze_text(text):
    text = text.strip() if text else ""
    
    if not text:
        return "Error: Input text cannot be empty. Please provide real content."
    
    if text == "{content}":
        return "Error: Placeholder '{content}' detected. Please provide REAL user input."

    logger.info(f"UI INPUT RECEIVED: {text}")

    try:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            result = response.json()
            return json.dumps(result, indent=2)
        else:
            return f"Error from API: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Failed to connect to the backend. Please ensure the FastAPI server is running."
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft(), title="Fake Content Detection System") as demo:
    gr.Markdown("# 🕵️ Fake Content Detection System")
    gr.Markdown("Enter any claim, news article, or social media post to verify its authenticity.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Content to Evaluate", 
                placeholder="e.g., Aliens landed in Bangalore yesterday", 
                lines=5
            )
            submit_btn = gr.Button("Analyze Content", variant="primary")
            
            gr.Examples(
                examples=[
                    "Aliens landed in Bangalore yesterday",
                    "The stock market experienced a slight dip today due to inflation fears",
                    "Drink this magic potion to lose 50 pounds in one day!",
                    "{content}"
                ],
                inputs=input_text
            )
        
        with gr.Column(scale=1):
            output_json = gr.Code(label="Analysis Result (JSON)", language="json")
            
    submit_btn.click(fn=analyze_text, inputs=input_text, outputs=output_json)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
