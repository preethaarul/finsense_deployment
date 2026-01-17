from pydantic import BaseModel, EmailStr, Field
from typing import Optional


# ----------------------------
# AUTH SCHEMAS
# ----------------------------

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user_name: str


# ----------------------------
# TRANSACTION SCHEMAS
# ----------------------------

class TransactionCreate(BaseModel):
    type: str            # "income" or "expense"
    amount: float
    category: str
    date: str            # YYYY-MM-DD
    description: Optional[str] = None


class TransactionResponse(TransactionCreate):
    id: int

    class Config:
        orm_mode = True


# ----------------------------
# MONTHLY BUDGET SCHEMAS ✅ NEW
# ----------------------------

class BudgetCreate(BaseModel):
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2020)
    amount: float = Field(..., gt=0)


class BudgetResponse(BudgetCreate):
    id: int
    user_id: int
    month: int
    year: int
    amount: float
    #created_at: str

    class Config:
        orm_mode = True


# ----------------------------
# PASSWORD CHANGE
# ----------------------------

class PasswordChange(BaseModel):
    current_password: str
    new_password: str