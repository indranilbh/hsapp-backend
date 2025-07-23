from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import random
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()
DATABASE_URL = os.getenv("DATABASE_URL")


# Pydantic Models
class SignupRequest(BaseModel):
    name: str
    phone: str
    email: str | None = None
    interests: list[str]
    is_senior: bool = False
    is_driver: bool = False
    is_travel_agent: bool = False
    is_homestay_owner: bool = False

class OTPRequest(BaseModel):
    phone: str

class OTPVerify(BaseModel):
    phone: str
    otp: str


# Utility
def generate_otp():
    return str(random.randint(100000, 999999))


@router.post("/signup")
def signup(data: SignupRequest):
    if len(data.interests) == 0:
        raise HTTPException(status_code=400, detail="At least one interest required")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if phone exists
            cur.execute("SELECT name FROM users WHERE phone = %s", (data.phone,))
            existing = cur.fetchone()
            if existing:
                raise HTTPException(status_code=409, detail=existing["name"])

            otp = generate_otp()
            expiry = datetime.utcnow() + timedelta(minutes=10)

            cur.execute("""
                INSERT INTO users (name, phone, email, interests,
                is_senior, is_driver, is_travel_agent, is_homestay_owner, otp, otp_expiry)
                VALUES (%s, %s, %s, %s,%s, %s, %s, %s, %s, %s)

            """, (
                data.name,
                data.phone,
                data.email,
                data.interests,
                data.is_senior,
                data.is_driver,
                data.is_travel_agent,
                data.is_homestay_owner,
                otp,
                expiry
            ))
            conn.commit()
            print(f"[DEBUG] OTP for {data.phone}: {otp}")
            return {"message": "User created. OTP sent to phone."}
    finally:
        conn.close()


@router.post("/request-otp")
def request_otp(data: OTPRequest):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE phone = %s", (data.phone,))
            user = cur.fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            otp = generate_otp()
            expiry = datetime.utcnow() + timedelta(minutes=10)

            cur.execute("""
                UPDATE users SET otp = %s, otp_expiry = %s WHERE phone = %s
            """, (otp, expiry, data.phone))
            conn.commit()
            print(f"[DEBUG] OTP for {data.phone}: {otp}")
            return {"message": "OTP sent"}
    finally:
        conn.close()


@router.post("/verify-otp")
def verify_otp(data: OTPVerify):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, otp, otp_expiry FROM users WHERE phone = %s
            """, (data.phone,))
            user = cur.fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            # if user["otp"] != data.otp:
            #     raise HTTPException(status_code=401, detail="Invalid OTP")

            if user["otp_expiry"] < datetime.utcnow():
                raise HTTPException(status_code=401, detail="OTP expired")

            # Clear OTP and set last login
            cur.execute("""
                UPDATE users SET otp = NULL, otp_expiry = NULL, last_login = %s WHERE phone = %s
            """, (datetime.utcnow(), data.phone))
            conn.commit()

            return {
                "message": "Login successful",
                "user_id": user["id"],
                "expires_in": "24 hours"
            }
    finally:
        conn.close()

@router.get("/user-by-phone/{phone}")
def get_user_by_phone(phone: str):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT name FROM users WHERE phone = %s", (phone,))
            user = cur.fetchone()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            return user
    finally:
        conn.close()
