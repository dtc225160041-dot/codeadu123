from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# -------------------------------
# 1️⃣ Khởi tạo FastAPI app
# -------------------------------
app = FastAPI(title="Vietnamese Medical NER API")

# -------------------------------
# 2️⃣ Cấu hình CORS (cho phép frontend gọi API)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể thay "*" bằng URL thật của v0.app nếu muốn bảo mật hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 3️⃣ Định nghĩa model đầu vào
# -------------------------------
class InputText(BaseModel):
    text: str

# -------------------------------
# 4️⃣ Nạp mô hình NER
# -------------------------------
model_name = "d4data/biomedical-ner-all"  # Mô hình NER y tế có sẵn trên Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=-1   # Ép dùng CPU để tránh lỗi DLL
)


# -------------------------------
# 5️⃣ Các endpoint chính
# -------------------------------
@app.get("/")
def home():
    return {"message": "Vietnamese Medical NER API is running!"}

@app.post("/ner")
def get_entities(data: InputText):
    result = nlp(data.text)
    return {"entities": result}
