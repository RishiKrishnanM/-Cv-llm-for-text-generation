from PIL import Image
from ultralytics import YOLO
import groq
import argparse
import os
from dotenv import load_dotenv

load_dotenv()
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def detect_objects(image_path, threshold=0.3):
    model = YOLO("yolov8s.pt")
    results = model(image_path)
    return [
        (model.names[int(box.cls[0])], round(box.conf[0].item(), 2))
        for r in results
        for box in r.boxes
        if box.conf[0].item() >= threshold
    ]

def generate_text_response(prompt, detected_objects):
    object_summary = ", ".join([f"{obj} ({score})" for obj, score in detected_objects]) or "no objects detected"
    full_prompt = (
        f"Objects detected in the image: {object_summary}. "
        f"Now, considering this, respond to the following prompt: {prompt}"
    )
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def main(image_path, prompt):
    load_image(image_path)  
    print("Detecting objects with YOLOv8...")
    objects = detect_objects(image_path)

    print("Detected Objects:")
    for obj, score in objects:
        print(f"- {obj}: {score}")

    print("\n Generating response from LLaMA 3 via Groq...")
    response = generate_text_response(prompt, objects)
    print("\n Generated Response:\n", response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + LLaMA3 via Groq API")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", required=True, help="Text prompt for the LLM")
    args = parser.parse_args()
    main(args.image, args.prompt)
