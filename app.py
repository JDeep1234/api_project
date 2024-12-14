# app.py  

from fastapi import FastAPI, UploadFile, File  
from fastapi.responses import HTMLResponse  
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor  
from PIL import Image  
import io  
import re  
from datetime import datetime  
import torch  

# Load the model and processor  
model = Qwen2VLForConditionalGeneration.from_pretrained(  
    "Qwen/Qwen2-VL-2B-Instruct",  
    torch_dtype="auto",  
    device_map="auto",  
)  

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")  

app = FastAPI()  

# HTML form for image upload  
html_content = """  
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Product Image Upload</title>  
</head>  
<body>  
    <h1>Upload a Product Image</h1>  
    <form action="/upload" method="post" enctype="multipart/form-data">  
        <input type="file" name="file" accept="image/*" required>  
        <button type="submit">Upload</button>  
    </form>  
    <div id="result"></div>  
</body>  
</html>  
"""  

@app.get("/", response_class=HTMLResponse)  
async def main():  
    return html_content  

@app.post("/upload")  
async def upload(file: UploadFile = File(...)):  
    # Load the image  
    contents = await file.read()  
    image = Image.open(io.BytesIO(contents))  

    # Define the user message  
    messages = [  
        {  
            "role": "user",  
            "content": [  
                {"type": "image"},  
                {"type": "text", "text": "Extract brand name, expiry date, expired status, expected life span in days, and object counts."}  
            ]  
        }  
    ]  

    # Create a text prompt with the updated task  
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  

    # Process both the text and image inputs  
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")  
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")  

    # Generate output text  
    output_ids = model.generate(**inputs, max_new_tokens=1024)  
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]  
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]  

    # Extract information using regex  
    expiry_date, brand_name, expired, life_span_days, object_count = extract_information(output_text)  

    # Prepare results for display  
    result_html = f"""  
    <h2>Results:</h2>  
    <p><strong>Brand Name:</strong> {brand_name}</p>  
    <p><strong>Expiry Date:</strong> {expiry_date.strftime('%d/%m/%Y') if expiry_date else 'Not found'}</p>  
    <p><strong>Expired:</strong> {'Yes' if expired else 'No'}</p>  
    <p><strong>Expected Life Span in Days:</strong> {life_span_days if life_span_days is not None else 'N/A'}</p>  
    <p><strong>Object Count:</strong> {object_count if object_count is not None else 'Not found'}</p>  
    <a href="/">Upload another image</a>  
    """  
    return HTMLResponse(content=result_html)  

def extract_information(output_text):  
    # Regex patterns for extracting information  
    date_patterns = [  
        r'\b(\d{2}/\d{2}/\d{4})\b',  # MM/DD/YYYY  
        r'\b(\d{2}-\d{2}-\d{4})\b',  # MM-DD-YYYY  
        r'\b(\d{2}/\d{2}/\d{2})\b',  # MM/DD/YY  
        r'\b(\d{2}-\d{2}-\d{2})\b',  # MM-DD-YY  
        r'\b(\d{2} \w+ \d{4})\b',    # DD Month YYYY (e.g., 12 May 2024)  
        r'\b(\d{2} \d{2} \d{4})\b'   # DD MM YYYY (e.g., 23 10 2021)  
    ]  

    # Extract expiry date  
    expiry_date = None  
    for pattern in date_patterns:  
        match = re.findall(pattern, output_text)  
        if match:  
            expiry_date = match[0]  
            break  

    # Extract brand name  
    brand_name = None  
    brand_patterns = [  
        r"brand[\s:]*([A-Za-z0-9\s]+)",  # Look for a 'brand' keyword followed by a name  
        r"([A-Za-z]+)[\s]+brand"  # Assume a word before the keyword 'brand' might be the brand name  
    ]  
    for pattern in brand_patterns:  
        match = re.findall(pattern, output_text)  
        if match:  
            brand_name = match[0]  
            break  

    # Determine if the product is expired  
    expired = None  
    if expiry_date:  
        try:  
            if " " in expiry_date:  
                expiry_date = datetime.strptime(expiry_date, "%d %m %Y")  
            elif "/" in expiry_date or "-" in expiry_date:  
                expiry_date = datetime.strptime(expiry_date, "%d/%m/%Y")  

            current_date = datetime.now()  
            expired = expiry_date < current_date  
        except ValueError:  
            print(f"Could not parse the expiry date format: {expiry_date}")  

    # Calculate expected life span if not expired  
    life_span_days = None  
    if not expired and expiry_date:  
        life_span_days = (expiry_date - current_date).days  

    # Object count extraction  
    object_count = None  
    count_pattern = r"(\d+)\s*objects?|(\d+)\s*items?"  
    count_match = re.findall(count_pattern, output_text)  
    if count_match:  
        object_count = int(count_match[0][0])  # Taking the first match group  

    return expiry_date, brand_name, expired, life_span_days, object_count  

if __name__ == "__main__":  
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8000)
