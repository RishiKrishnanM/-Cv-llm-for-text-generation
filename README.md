
#  Computer Vision + LLM Integration (YOLO + Groq)
This Python application detects objects in an image using a pre-trained **YOLOv8** model and generates a meaningful text response using **LLaMA 3 (via Groq API)**.

---

# Features

- Object detection using YOLOv8 (`ultralytics`)
- Text generation using LLaMA 3 (`llama3-70b-8192`) via Groq API
- Input: Image + custom prompt
- Output: Detected objects + a coherent response from LLM

---

# Installation

```bash
git clone https://github.com/your-repo/cv-llm-integration.git
cd cv-llm-integration

# Install required packages
pip install -r requirements.txt


#how to Run
save jpg image in this folder

# Run this command in terminal for outputs
python main.py --image sample.jpg --prompt "Describe what might be happening in this scene."
