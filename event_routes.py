from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from datetime import date, timedelta

load_dotenv()
router = APIRouter()
DATABASE_URL = os.getenv("DATABASE_URL")

class Event(BaseModel):
    eventdate: date
    area_id: int
    location: str
    description: str
    specialnote: Optional[str] = None
    pocname: str
    poccontact1: str
    poccontact2: Optional[str] = None
    pocemail: Optional[str] = None
    social1: Optional[str] = None
    social2: Optional[str] = None
    attendiscount: Optional[str] = None
    status: Optional[str] = "active"

@router.post("/events")
def create_event(event: Event):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO events (
                    event_date, area_id, location, description, special_note,
                    poc_name, poc_contact_number1, poc_contact_number2, poc_email,
                    social_media_link1, social_media_link2, attendee_discount, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                event.eventdate, event.area_id, event.location, event.description, event.specialnote,
                event.pocname, event.poccontact1, event.poccontact2, event.pocemail,
                event.social1, event.social2, event.attendiscount, event.status
            ))
            conn.commit()
            return {"message": "✅ Event created successfully"}
    except Exception as e:
        print("Error inserting event:", e)
        return JSONResponse(content={"error": "Failed to create event"}, status_code=500)
    finally:
        conn.close()


@router.get("/upcoming")
def get_upcoming_events():
    try:
        print ("get_upcoming_events")
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            today = date.today()
            six_months_later = today + timedelta(days=180)
            query = """
                SELECT e.*, a.country, a.state, a.district
                FROM events e
                LEFT JOIN areas a ON e.area_id = a.id
                WHERE e.status = 'active' AND e.event_date BETWEEN %s AND %s
                ORDER BY e.event_date ASC
                LIMIT 1000;
            """
            print("📄 SQL Query:", query)
            print("📅 Parameters:", today, six_months_later)
            cur.execute(query, (today, six_months_later))
            events = cur.fetchall()
            return events

    except Exception as e:
        print("Error fetching upcoming events:", e)
        return JSONResponse(content={"error": "Failed to fetch upcoming events"}, status_code=500)
    finally:
        conn.close()

@router.post("/interest")
def register_interest(user_id: int = Body(...), event_id: int = Body(...)):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users_event_interest (user_id, event_id)
                VALUES (%s, %s)
            """, (user_id, event_id))
            conn.commit()
        return {"message": "✅ Interest recorded."}
    except Exception as e:
        print("❌ Error saving interest:", e)
        return JSONResponse(content={"error": "Failed to save interest"}, status_code=500)
    finally:
        conn.close()