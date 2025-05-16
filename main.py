from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import Column, Float, Integer, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import joblib

# --- Database setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./homes.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- SQLAlchemy model ---
class Home(Base):
    __tablename__ = "homes"
    id = Column(Integer, primary_key=True, index=True)
    rm = Column(Float, nullable=False)
    lstat = Column(Float, nullable=False)
    dis = Column(Float, nullable=False)
    tax = Column(Float, nullable=False)
    ptratio = Column(Float, nullable=False)
    age = Column(Float, nullable=False)
    indus = Column(Float, nullable=False)
    medv = Column(Float, nullable=False)  # Actual median value


Base.metadata.create_all(bind=engine)


# --- Pydantic schemas ---
class HomeBase(BaseModel):
    rm: float
    lstat: float
    dis: float
    tax: float
    ptratio: float
    age: float
    indus: float


class HomeCreate(HomeBase):
    medv: float  # Include actual median value when creating


class HomeOut(HomeBase):
    id: int
    medv: float

    class Config:
        orm_mode = True


class Prediction(BaseModel):
    predicted_price_dh: float


# --- Load trained model ---
model = joblib.load("pipeline.pkl")

# --- FastAPI instance ---
app = FastAPI(
    title="Boston Housing Predictor API", docs_url="/docs", redoc_url="/redoc"
)


# --- Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Routes ---
@app.get("/", tags=["root"])
def read_root():
    return {"message": "Welcome to the Boston Housing Predictor API"}


@app.post("/homes/", response_model=HomeOut)
def create_home(home: HomeCreate, db: Session = Depends(get_db)):
    # Store new home record including actual median value
    db_home = Home(**home.dict())
    db.add(db_home)
    db.commit()
    db.refresh(db_home)
    return db_home


@app.get("/homes/", response_model=list[HomeOut])
def list_homes(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(Home).offset(skip).limit(limit).all()


@app.post("/predict/", response_model=Prediction)
def predict_price(home: HomeBase):
    vals = [
        [home.rm, home.lstat, home.dis, home.tax, home.ptratio, home.age, home.indus]
    ]
    pred = model.predict(vals)[0]
    # model predicts median value in $1000s
    dollar_price = float(pred * 1000)
    # Convert dollar to dirham (assuming rate *10)
    dirham_price = round(dollar_price, 1) * 10
    return {"predicted_price_dh": dirham_price}


@app.get("/recommendation/", response_model=list[HomeOut])
def recommendation(price: float, limit: int = 20, db: Session = Depends(get_db)):
    """
    Recommend homes with median values closest to the given price (in dirhams).
    - price: target median home price in dirhams
    - limit: number of recommendations to return
    """
    # medv stored in thousands, convert price to same scale
    target = price / 1000
    target = target / 10
    homes = db.query(Home).order_by(func.abs(Home.medv - target)).limit(limit).all()
    if not homes:
        raise HTTPException(status_code=404, detail="No homes found for recommendation")
    return homes
