import os
import json
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, List
import uvicorn

# Import quant core engine
from quant_engine import QuantEngine
# Database schema (SQLAlchemy)
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = "sqlite:///./quant.db"

Base = declarative_base()

class OptimizationHistory(Base):
    __tablename__ = "optimization_history"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now)
    max_weight_setting = Column(Float, nullable=False)
    top_allocated_asset = Column(String(20), nullable=True)
    top_asset_weight = Column(Float, nullable=True)
    portfolio_weights = Column(String, nullable=True)
    status = Column(String(20), default="SUCCESS")

# Initialize database connection
try:
    db_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    Base.metadata.create_all(bind=db_engine)
except Exception as e:
    print(f"⚠️ Database engine connection warning: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI app initialization and CORS
app = FastAPI(
    title="Investment Portfolio Expert API",
    description="Backend for intelligent asset allocation using LSTM time-series forecasting and Markowitz optimization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Initializing quant engine and loading local LSTM model weights...")
try:
    quant_engine = QuantEngine()
    print("✅ LSTM core and scaler assets loaded; API is ready.")
except Exception as e:
    print(f"❌ Quant engine initialization failed. Ensure 4 weight files are in the current directory. Error: {e}")

# Request/response schemas
class OptimizeRequest(BaseModel):
    max_weight: float = Field(
        default=0.4, 
        ge=0.1, 
        le=1.0, 
        description="Maximum weight cap per single stock"
    )
    tickers: List[str] = Field(
        default=["MRK", "NFLX", "CRM", "HON", "IBM"], 
        description="User-selected asset pool ticker list"
    )

class OptimizeResponse(BaseModel):
    status: str
    timestamp: str
    weights: Dict[str, float]

class PortfolioHistoryResponse(BaseModel):
    id: int
    created_at: datetime
    max_weight: float
    weights_json: dict


# Core API routes
@app.post("/api/optimize_portfolio", response_model=OptimizeResponse)
def do_portfolio_optimization(request: OptimizeRequest, db: Session = Depends(get_db)):
    try:
        print(f"📥 Received portfolio optimization request, max weight cap: {request.max_weight}")
        print(f"📥 Selected asset pool tickers: {request.tickers}") 
        
        optimized_weights = quant_engine.run_optimization(
            max_weight=request.max_weight,
            tickers=request.tickers
        )
        
        # Return empty dict when model recommends staying in cash
        if not optimized_weights or len(optimized_weights) == 0:
            print("⚠️ Optimization returned empty; issuing cash-only directive.")
            return OptimizeResponse(
                status="empty",
                timestamp=datetime.now().isoformat(),
                weights={}
            )
        
        top_asset = max(optimized_weights, key=optimized_weights.get)
        top_weight = optimized_weights[top_asset]
        
        try:
            log_entry = OptimizationHistory(
                max_weight_setting=request.max_weight,
                top_allocated_asset=top_asset,
                top_asset_weight=top_weight,
                portfolio_weights=json.dumps(optimized_weights),
                status="SUCCESS"
            )
            db.add(log_entry)
            db.commit()
            print("💾 Portfolio allocation history saved to database.")
        except Exception as db_err:
            print(f"⚠️ Database write interrupted: {db_err}")
            db.rollback()

        return OptimizeResponse(
            status="success",
            timestamp=datetime.now().isoformat(),
            weights=optimized_weights
        )
        
    except Exception as e:
        print(f"❌ Critical runtime error in API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quant server internal error: {str(e)}")


@app.get("/api/portfolio_history", response_model=List[PortfolioHistoryResponse])
def get_portfolio_history(db: Session = Depends(get_db)):
    try:
        records = db.query(OptimizationHistory).order_by(OptimizationHistory.timestamp.desc()).all()
        
        response_data = []
        for r in records:
            weights = {}
            if r.portfolio_weights:
                try:
                    weights = json.loads(r.portfolio_weights)
                except Exception:
                    weights = {r.top_allocated_asset: r.top_asset_weight} if r.top_allocated_asset else {}
            else:
                weights = {r.top_allocated_asset: r.top_asset_weight} if r.top_allocated_asset else {}

            response_data.append({
                "id": r.id,
                "created_at": r.timestamp,
                "max_weight": r.max_weight_setting,
                "weights_json": weights
            })
        return response_data
    except Exception as e:
        print(f"❌ Failed to read portfolio history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database read failed: {str(e)}")

# Server Execution Entry Point
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)