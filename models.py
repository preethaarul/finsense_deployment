from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from datetime import datetime
from database import Base


# -------------------------------
# USER MODEL (AUTH)
# -------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# -------------------------------
# TRANSACTION MODEL
# -------------------------------

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    type = Column(String, nullable=False)        # income | expense
    amount = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    date = Column(String, nullable=False)        # YYYY-MM-DD
    description = Column(String, nullable=True)


# -------------------------------
# MONTHLY BUDGET MODEL  ✅ NEW
# -------------------------------

class Budget(Base):
    __tablename__ = "budgets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    month = Column(Integer, nullable=False)      # 1 - 12
    year = Column(Integer, nullable=False)       # 2024, 2025, etc
    amount = Column(Float, nullable=False)       # Monthly budget amount

    created_at = Column(DateTime, default=datetime.utcnow)