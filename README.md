# **BFI Tech Test – Data Analytics Operation**

API ini memprediksi kualitas wine berdasarkan fitur-fitur kimiawi menggunakan model *machine learning* yang telah dilatih.

---

## 🚀 Features

- 🔮 Prediksi kualitas wine (skor 0–10) dari input data kimia
- ⚙️ Model ML berbasis `RandomForestClassifier`
- 🧪 Dibangun dengan **FastAPI**
- 🐳 Dikemas dalam Docker container
- 📦 Siap untuk di-deploy dan digunakan sebagai REST API

---

## 📁 Project Structure

```
.
├── app.py                
├── train_model.py        
├── predict.py            
├── model.pkl             
├── requirements.txt      
├── Dockerfile            
├── curl_test.sh          
├── README.md             
└── WineQT.csv            
```
---

## ⚙️ Setup & Installation

### 1. 🔧 Build Docker Image

```bash
docker build -t wine-api .
```

### 2. ▶️ Run Docker Container

```bash
docker run -p 8000:8000 wine-api
```

### 3. 🧪 Test API Endpoint

Gunakan `curl_test.sh` atau langsung dengan `curl`:

```bash
sh curl_test.sh
```

Atau manual:

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}}'
```

---

## 📨 API Usage

### Endpoint
```
POST /predict
```

### Request Body

```json
{{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}}
```

### Response

```json
{{
  "prediction": 5
}}
```

---

## 🧠 Model Information

- **Model**: Random Forest Classifier
- **Dataset**: Wine Quality (Red Wine)
- **Fitur input**: 11 fitur kimia (acidity, sugar, alcohol, dll)
- **Output**: Skor kualitas wine (0–10)

---

## 📦 Dependencies

- `fastapi`
- `uvicorn`
- `scikit-learn`
- `pandas`
- `pydantic`

Instalasi manual (jika tidak pakai Docker):

```bash
pip install -r requirements.txt
```

---

## 📬 Contact

Jika ada pertanyaan terkait proyek ini, silakan hubungi melalui email atau buat issue di repositori ini.
