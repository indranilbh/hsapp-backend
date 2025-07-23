# lexical_routes.py
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

router = APIRouter()

class LexicalSearchRequest(BaseModel):
    country: str
    state: str
    districts: List[str] = []
    areas: List[str] = []
    tags: List[str] = []
    facilities: List[str] = []
    price: List[int] = []
    milestones: Optional[List[str]] = []

@router.get("/filters")
def get_filters(country: Optional[str] = None, state: Optional[str] = None, district: Optional[str] = None):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            # ✅ Independent: fetch all countries
            cur.execute("SELECT DISTINCT country FROM areas WHERE country IS NOT NULL")
            countries = [r[0] for r in cur.fetchall()]

            # ✅ Dependent: fetch states based on selected country
            if country:
                cur.execute("SELECT DISTINCT state FROM areas WHERE country = %s", (country,))
                states = [r[0] for r in cur.fetchall()]
            else:
                states = []

            # ✅ Dependent: fetch districts based on selected state
            if state:
                cur.execute("SELECT DISTINCT district FROM areas WHERE state = %s", (state,))
                districts = [r[0] for r in cur.fetchall()]
            else:
                districts = []

            # ✅ Independent: fetch all tags
            cur.execute("SELECT name FROM tags")
            tags = [r[0] for r in cur.fetchall()]

            # ✅ Independent: fetch all facilities
            cur.execute("SELECT name FROM facilities")
            facilities = [r[0] for r in cur.fetchall()]


            # ✅ Filtered milestone tags
            milestone_query = """
                SELECT DISTINCT mt.tag
                FROM milestone_tags mt
                JOIN areas a ON mt.area_id = a.id
                WHERE (%s IS NULL OR a.country = %s)
                  AND (%s IS NULL OR a.state = %s)
                  AND (%s IS NULL OR a.district = %s)
            """
            cur.execute(milestone_query, (country, country, state, state, district, district))
            milestones = [r[0] for r in cur.fetchall()]

        return {
            "countries": countries,
            "states": states,
            "districts": districts,
            "tags": tags,
            "milestones": [],
            "facilities": facilities
        }
    finally:
        conn.close()


@router.post("/lexical_search")
def lexical_search(req: LexicalSearchRequest):
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            base_query = """
                SELECT DISTINCT h.id, h.name, h.cost, a.state, a.district
                FROM homestays h
                JOIN areas a ON h.area_id = a.id
                LEFT JOIN homestay_tags ht ON ht.homestay_id = h.id
                LEFT JOIN tags t ON t.id = ht.tag_id
                WHERE a.country = %s AND a.state = %s
            """
            params = [req.country, req.state]

            if req.districts:
                base_query += " AND a.district = ANY(%s)"
                params.append(req.districts)


            if req.tags:
                base_query += " AND t.name = ANY(%s)"
                params.append(req.tags)

            if req.price:
                base_query += " AND h.cost <= %s"
                params.append(max(req.price))

            cur.execute(base_query, tuple(params))
            print("base_query for lexical search =", base_query)
            rows = cur.fetchall()

            matches = [
                {
                    "id": r[0],
                    "name": r[1],
                    "cost": r[2],
                    "state": r[3],
                    "district": r[4]
                }
                for r in rows
            ]

            return {
                "message": f"Found {len(matches)} homestays using lexical search.",
                "matches": matches
            }
    finally:
        conn.close()
