import os
import random
import sys
import torch
from pathlib import Path

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

app = FastAPI(title="Math Topic Classification")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", None)
USE_LOCAL_MODEL = LOCAL_MODEL_PATH is not None and os.path.exists(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else False

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
HF_API_URL = f"https://router.huggingface.co/models/{HF_MODEL_ID}"
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

model = None
tokenizer = None
device = None

TOPICS = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}

SAMPLE_QUESTIONS = [
    "Solve the quadratic equation x^2 - 5x + 6 = 0.",
    "Factor the expression 2x³ - 4x² - 22x + 24.",
    "Simplify the expression (3x⁴y²)³ ÷ (9x²y⁵)².",

    "Find the area of a triangle with vertices at (0,0), (3,4), and (-1,2).",
    "Calculate the volume of a cone with radius 5 cm and height 12 cm.",
    "If sin(θ) = 0.6, what is the value of cos(θ)?",

    "Find the limit of (1+1/n)^n as n approaches infinity.",
    "Calculate the derivative of f(x) = x^3 - 3x^2 + 2x - 1.",
    "Evaluate the integral of x*ln(x) from x=1 to x=e.",

    "If P(A) = 0.3 and P(B) = 0.4 and A and B are independent events, what is P(A and B)?",
    "What is the probability of rolling a sum of 8 with two dice?",
    "Calculate the mean and standard deviation of the dataset: 4, 7, 8, 11, 12, 14.",

    "Find all prime numbers p such that p+2 is also prime.",
    "Prove that the square root of 3 is irrational.",
    "Find the greatest common divisor of 168 and 180.",

    "How many ways can 5 books be arranged on a shelf?",
    "In how many ways can a committee of 3 be chosen from 9 people?",
    "Solve the recurrence relation an = an-1 + 2an-2 with a0 = 1 and a1 = 3.",

    "For the matrix A = [[1,2,3],[4,5,6],[7,8,9]], find the eigenvalues and eigenvectors.",
    "Determine if the vectors (1,2,3), (2,3,4), and (3,4,5) are linearly independent.",
    "Find the eigenvalues of the matrix [[4,1],[6,-1]].",

    "Prove that the set of continuous functions on [0,1] is a vector space.",
    "Show that the set of all rational numbers is dense in the real numbers.",
    "Prove that every compact metric space is complete."
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def load_local_model():
    global model, tokenizer, device

    if model is not None and tokenizer is not None:
        return

    print(f"Loading trained model from: {LOCAL_MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    try:
        peft_config = PeftConfig.from_pretrained(LOCAL_MODEL_PATH, token=hf_token)

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            print("[INFO] CPU detected: Disabling quantization for faster loading")
            bnb_config = None

        if torch.cuda.is_available():
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                num_labels=8,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token
            )
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                num_labels=8,
                quantization_config=bnb_config,
                dtype=torch.float32,
                trust_remote_code=True,
                token=hf_token
            )

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, token=hf_token)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"
                tokenizer.add_special_tokens({'pad_token': "[PAD]"})

        base_model.config.pad_token_id = tokenizer.pad_token_id

        model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)
        model = model.to(device)
        model.eval()

        print("[OK] Trained model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def predict_with_local_model(question: str):
    global model, tokenizer, device

    if model is None or tokenizer is None:
        load_local_model()

    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_label].item()

    topic_name = TOPICS.get(predicted_label, f"Topic {predicted_label}")

    return topic_name, confidence

def predict_with_hf_api(question: str):
    import requests

    headers = {
        "Content-Type": "application/json"
    }

    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {"inputs": question}

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type or response.text.strip().startswith('<!DOCTYPE'):
            raise Exception("The Hugging Face API endpoint is not available. Please use a local trained model or check your API configuration.")

        result = response.json()

        if isinstance(result, list):
            if len(result) > 0:
                best_pred = result[0]
                label = best_pred.get("label", "")
                score = best_pred.get("score", 0.0)

                if label.startswith("LABEL_"):
                    label_id = int(label.replace("LABEL_", ""))
                    topic_name = TOPICS.get(label_id, label)
                else:
                    topic_name = label

                return topic_name, score
        elif isinstance(result, dict):
            label = result.get("label", "")
            score = result.get("score", 0.0)
            if label.startswith("LABEL_"):
                label_id = int(label.replace("LABEL_", ""))
                topic_name = TOPICS.get(label_id, label)
            else:
                topic_name = label
            return topic_name, score

        return "Unknown", 0.0

    except requests.exceptions.HTTPError as e:
        error_msg = ""
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 503:
                error_msg = "Model is loading, please wait a moment and try again."
            elif 'text/html' in e.response.headers.get('content-type', '').lower():
                error_msg = "The Hugging Face API endpoint returned an error page. The model may not be available via the API. Please use a local trained model instead."
            else:
                try:
                    error_json = e.response.json()
                    error_msg = f"API error: {error_json.get('error', str(e))}"
                except:
                    error_msg = f"API error (Status {e.response.status_code}): The endpoint may not be available."
        else:
            error_msg = f"API connection error: {str(e)}"
        raise Exception(error_msg)
    except Exception as e:
        error_msg = str(e)
        if "HTML" in error_msg or "DOCTYPE" in error_msg:
            error_msg = "The API endpoint is not available. Please configure a local trained model or check your Hugging Face API settings."
        raise Exception(error_msg)

def predict_with_demo_mode(question: str):

    question_lower = question.lower()

    topic_keywords = {
        "Algebra": ["equation", "solve", "factor", "polynomial", "quadratic", "linear", "variable", "x^", "algebra"],
        "Geometry and Trigonometry": ["triangle", "circle", "angle", "area", "volume", "sin", "cos", "tan", "geometry", "trigonometry"],
        "Calculus and Analysis": ["derivative", "integral", "limit", "calculus", "differentiate", "integrate"],
        "Probability and Statistics": ["probability", "statistics", "mean", "standard deviation", "dice", "random", "distribution"],
        "Number Theory": ["prime", "gcd", "divisor", "modulo", "number theory", "irrational"],
        "Combinatorics and Discrete Math": ["permutation", "combination", "arrange", "choose", "recurrence", "combinatorics"],
        "Linear Algebra": ["matrix", "eigenvalue", "eigenvector", "vector", "linear", "determinant"],
        "Abstract Algebra and Topology": ["group", "ring", "field", "topology", "continuous", "compact", "metric space"]
    }

    scores = {}
    for topic, keywords in topic_keywords.items():
        score = sum(1 for keyword in keywords if keyword in question_lower)
        scores[topic] = score

    if max(scores.values()) > 0:
        predicted_topic = max(scores, key=scores.get)
        confidence = min(0.85, 0.5 + (max(scores.values()) / len(question.split())) * 0.35)
    else:
        predicted_topic = "Algebra"
        confidence = 0.65

    return predicted_topic, confidence

@app.post("/predict", response_class=JSONResponse)
async def predict(question: str = Form(...)):
    try:
        if USE_LOCAL_MODEL:
            topic_name, raw_confidence = predict_with_local_model(question)
        else:
            try:
                topic_name, raw_confidence = predict_with_hf_api(question)
            except Exception as api_error:
                print(f"[INFO] API unavailable, using demo mode: {str(api_error)}")
                topic_name, raw_confidence = predict_with_demo_mode(question)

        return {
            "success": True,
            "question": question,
            "prediction": topic_name,
            "confidence": round(raw_confidence * 100, 2)
        }
    except Exception as e:
        return {
            "success": False,
            "question": question,
            "error": str(e)
        }

@app.get("/random-question", response_class=JSONResponse)
async def get_random_question():
    question = random.choice(SAMPLE_QUESTIONS)
    return {"question": question}

@app.on_event("startup")
async def startup_event():
    if USE_LOCAL_MODEL:
        print("=" * 60)
        print("Using LOCAL TRAINED MODEL")
        print(f"Model path: {LOCAL_MODEL_PATH}")
        print("=" * 60)
        try:
            load_local_model()
            print("[OK] Model ready for predictions!")
        except Exception as e:
            print(f"[WARNING] Could not load model on startup: {e}")
            print("Model will be loaded on first prediction request.")
    else:
        print("=" * 60)
        print("Using Hugging Face Inference API")
        print(f"Model: {HF_MODEL_ID}")
        print(f"API URL: {HF_API_URL}")
        if HF_TOKEN:
            print("[OK] Hugging Face token is configured")
        else:
            print("[WARNING] No Hugging Face token found.")
        print("=" * 60)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 
