
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
#Clone the Repository
git clone https://github.com/your-repo/cv-llm-integration.git
cd cv-llm-integration

# Install required packages
pip install -r requirements.txt

#Configure Environment Variables
Create a .env file in the project root and add your Groq API key


#Image
Save your image file (e.g., sample.jpg) in the root directory.

# Run this command in terminal for outputs
python main.py --image sample.jpg --prompt "Describe what might be happening in this scene."

'''
"""
#Example Output:
Detecting objects with YOLOv8...
Detected Objects:
- person: 0.92
- bicycle: 0.78


Generating response from LLaMA 3 via Groq...
Generated Response:
It appears that a person is riding a bicycle, possibly commuting or enjoying a ride outdoors.
"""

#Dependencies
As listed in requirements.txt:
*python_dotenv
*ultralytics
*groq
*pillow


 #Architecture Overview
Input Image
     │
     ▼
[ YOLOv8 Detection ]
     │
     ▼
[ Object Labels + Confidence Scores ]
     │
     ▼
[ Groq LLaMA 3 - Prompt Enrichment ]
     │
     ▼
[ Contextual LLM Output ]


#contact
For inquiries, please reach out via rishimrk003@gmail.com.

