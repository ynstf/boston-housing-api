import pandas as pd
from sqlalchemy import Column, Float, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Database setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./homes.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
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
    medv = Column(Float, nullable=False)

# create table if it doesn't already exist
Base.metadata.create_all(bind=engine)

# --- Load and filter CSV ---
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Keep only the desired columns
keep_cols = ['rm', 'lstat', 'dis', 'tax', 'ptratio', 'age', 'indus', 'medv']
df = df[keep_cols]

# --- Insert into database ---
session = SessionLocal()
for _, row in df.iterrows():
    home = Home(
        rm=row.rm,
        lstat=row.lstat,
        dis=row.dis,
        tax=row.tax,
        ptratio=row.ptratio,
        age=row.age,
        indus=row.indus,
        medv=row.medv
    )
    session.add(home)
try:
    session.commit()
    print(f"Inserted {len(df)} records into 'homes' table.")
except Exception as e:
    session.rollback()
    print(f"Error inserting records: {e}")
finally:
    session.close()
