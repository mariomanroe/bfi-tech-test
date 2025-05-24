# BFI Tech Test - Data Analytics Operation

## Setup & Run

```bash
docker build -t wine-api .
docker run -p 8000:8000 wine-api
```

## API Usage

POST /predict

Input JSON:
```json
{
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
}
```

Response:
```json
{
  "prediction": 5
}
```

## Dependencies
- FastAPI
- Uvicorn
- scikit-learn
- pandas
- pydantic