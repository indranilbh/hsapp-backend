from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psycopg2
import os
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()
router = APIRouter()
DATABASE_URL = os.getenv("DATABASE_URL")

class Driver(BaseModel):
    name: str
    contact1: str
    contact2: str = None
    social_media1: str = None
    social_media2: str = None
    bio: str = None
    notes: str = None
    photo_url: str = None
    area_id: int = None
    region: str = None
    car_type: str = None     
    car_model: str = None

@router.post("/drivers")
def create_driver(driver: Driver):
    try:
        print("Received driver data:", driver.dict())  # 🔍 DEBUG LINE
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO driver_detail (
                    name, contact1, contact2, social_media1, social_media2,
                    bio, notes, photo_url, area_id, region, car_type, car_model
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                driver.name, driver.contact1, driver.contact2, driver.social_media1, driver.social_media2,
                driver.bio, driver.notes, driver.photo_url, driver.area_id, driver.region,driver.car_type,driver.car_model
            ))
            conn.commit()
            return {"message": "✅ Driver created successfully"}
    except Exception as e:
        print("Error inserting driver:", e)
        return JSONResponse(content={"error": "Failed to create driver"}, status_code=500)
    finally:
        conn.close()

@router.get("/drivers")
def get_drivers():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.id, d.name, d.contact1, d.contact2, d.social_media1, d.social_media2,
                       d.bio, d.notes, d.photo_url, d.area_id, d.region,d.car_type, d.car_model,
                       a.country, a.state, a.district
                FROM driver_detail d
                LEFT JOIN areas a ON d.area_id = a.id
                ORDER BY d.id DESC
            """)
            rows = cur.fetchall()
            drivers = []
            for row in rows:
                drivers.append({
                    "id": row[0],
                    "name": row[1],
                    "contact1": row[2],
                    "contact2": row[3],
                    "social_media1": row[4],
                    "social_media2": row[5],
                    "bio": row[6],
                    "notes": row[7],
                    "photo_url": row[8],
                    "area_id": row[9],
                    "region": row[10],
                    "car_type": row[11],
                    "car_model": row[12],
                    "country": row[13],
                    "state": row[14],
                    "district": row[15],
                })
            return drivers
    except Exception as e:
        print("Error fetching drivers:", e)
        return JSONResponse(content={"error": "Failed to fetch drivers"}, status_code=500)
    finally:
        conn.close()

@router.put("/drivers/{driver_id}")
def update_driver(driver_id: int, driver: Driver):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE driver_detail SET
                    name=%s, contact1=%s, contact2=%s, social_media1=%s,
                    social_media2=%s, bio=%s, notes=%s, photo_url=%s, area_id=%s, region=%s,car_type=%s,car_model=%s
                WHERE id=%s
            """, (
                driver.name, driver.contact1, driver.contact2, driver.social_media1, driver.social_media2,
                driver.bio, driver.notes, driver.photo_url, driver.area_id, driver.region,driver.car_type,driver.car_model, driver_id
            ))
            conn.commit()
            return {"message": "✏️ Driver updated successfully"}
    except Exception as e:
        print("Error updating driver:", e)
        return JSONResponse(content={"error": "Failed to update driver"}, status_code=500)
    finally:
        conn.close()

@router.delete("/drivers/{driver_id}")
def delete_driver(driver_id: int):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM driver_detail WHERE id = %s", (driver_id,))
            conn.commit()
            return {"message": "🗑️ Driver deleted successfully"}
    except Exception as e:
        print("Error deleting driver:", e)
        return JSONResponse(content={"error": "Failed to delete driver"}, status_code=500)
    finally:
        conn.close()

@router.get("/homestay_drivers/{homestay_id}")
def get_drivers_for_homestay(homestay_id: int):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            # Step 1: Get milestone names for the homestay
            cur.execute("""
                SELECT mt.name
                FROM homestay_milestones hm
                JOIN milestone_tags mt ON hm.milestone_tag_id = mt.id
                WHERE hm.homestay_id = %s
            """, (homestay_id,))
            milestone_names = [row[0].lower() for row in cur.fetchall()]
            if not milestone_names:
                return []

            # Step 2: Get all drivers
            cur.execute("""
                SELECT d.id, d.name, d.contact1, d.contact2, d.social_media1, d.social_media2,
                       d.bio, d.notes, d.photo_url, d.area_id, d.region, d.car_type, d.car_model,
                       a.country, a.state, a.district
                FROM driver_detail d
                LEFT JOIN areas a ON d.area_id = a.id
            """)
            rows = cur.fetchall()

            # Step 3: Filter drivers whose region matches any milestone
            matching_drivers = []
            for row in rows:
                driver_region = row[10] or ""
                if any(milestone in driver_region.lower() for milestone in milestone_names):
                    matching_drivers.append({
                        "id": row[0],
                        "name": row[1],
                        "contact1": row[2],
                        "contact2": row[3],
                        "social_media1": row[4],
                        "social_media2": row[5],
                        "bio": row[6],
                        "notes": row[7],
                        "photo_url": row[8],
                        "area_id": row[9],
                        "region": row[10],
                        "car_type": row[11],
                        "car_model": row[12],
                        "country": row[13],
                        "state": row[14],
                        "district": row[15],
                    })

            return matching_drivers
    except Exception as e:
        print("Error fetching drivers for homestay:", e)
        return JSONResponse(content={"error": "Failed to fetch drivers"}, status_code=500)
    finally:
        conn.close()
