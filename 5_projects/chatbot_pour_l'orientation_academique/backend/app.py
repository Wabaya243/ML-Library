"""
Backend FastAPI simple pour authentifier un utilisateur sur PostgreSQL.

Routes exposées:
- POST /login       : vérifie email + mot de passe (mdp) dans la table users
- POST /auth/login  : alias de /login pour compatibilité avec le client Flutter

ATTENTION: ce code lit des mots de passe en clair (colonne mdp). En production,
il faut stocker des mots de passe hachés (bcrypt/argon2) et comparer avec un hash.

cd "C:\Users\OMEN\Documents\Cours I.A (Machine Learning)\5. Projet\chatbot_pour_l'orientation_academique\backend>"
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

import os
import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "test_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "123456789")

app = FastAPI(title="Academia Login API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginIn(BaseModel):
    email: EmailStr
    mdp: str


def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        cursor_factory=RealDictCursor,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
def login(payload: LoginIn):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, nom, email
                FROM users
                WHERE email = %s AND mdp = %s
                LIMIT 1
                """,
                (payload.email, payload.mdp),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
            return {"ok": True, "user": row}
    finally:
        conn.close()


@app.post("/auth/login")
def login_alias(payload: LoginIn):
    return login(payload)

