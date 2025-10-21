from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Chào Kiên, đây là ứng dụng FastAPI đầu tiên của bạn!"}
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Medical NER API")

class InputText(BaseModel):
    text: str

# Dùng mô hình NER (có thể thay bằng mô hình y tế riêng sau)
model_name = "dslim/bert-base-NER"
nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")

@app.get("/")
def home():
    return {"message": "Medical NER API is running!"}

@app.post("/ner")
def get_entities(data: InputText):
    result = nlp(data.text)
    return {"entities": result}
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "d4data/biomedical-ner-all"  # Mô hình NER y tế có sẵn
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
#%% from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc thay "*" bằng domain cụ thể nếu muốn an toàn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# %%
