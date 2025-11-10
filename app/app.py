import os
import re
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient

from intent_classifier import IntentClassifier
from db.engine import get_mongo_collection
from app.auth import verify_token

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ENV = os.getenv("ENV", "prod").lower()
logger.info(f"Running in {ENV} mode")

app = FastAPI(
    title="Basic ML App",
    description="A basic ML app",
    version="1.0.0",
)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "https://meusite.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
try:
    collection = get_mongo_collection(f"{ENV.upper()}_intent_logs")
    logger.info("Database connection established")
except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    logger.error(traceback.format_exc())


async def conditional_auth():
    """Returns user based on environment mode"""
    global ENV
    if ENV == "dev":
        logger.info("Development mode: skipping authentication")
        return "dev_user"
    else:
        try:
            return await verify_token()
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise HTTPException(status_code=401, detail="Authentication failed")


# Load models
MODELS = {}
try:
    logger.info("Loading confusion model...")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "intent_classifier", "models")
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model_name = model_file.replace(".keras", "")
        MODELS[model_name] = IntentClassifier(load_model=model_path)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    logger.error(traceback.format_exc())
app.MODELS = MODELS

"""
Routes
"""

@app.get("/")
async def root():
    return {"message": f"Basic ML App is running in {ENV} mode"}


@app.get("/status")
def status():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB")

    try:
        client = MongoClient(mongo_uri)
        client.server_info()  # força verificação de conexão
        return {"status": "ok", "database": db_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/predict")
async def predict(text: str, owner: str = Depends(conditional_auth)):
    predictions = {}
    for model_name, model in MODELS.items():
        top_intent, all_probs = model.predict(text)
        predictions[model_name] = {
            "top_intent": top_intent,
            "all_probs": all_probs
        }

    results = {
        "text": text,
        "owner": owner,
        "predictions": predictions,
        "timestamp": int(datetime.now(timezone.utc).timestamp())
    }

    # ✅ usa função mockável
    collection = get_mongo_collection(f"{ENV.upper()}_intent_logs")
    collection.insert_one(results)

    return JSONResponse(content=results)


    return JSONResponse(content=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
