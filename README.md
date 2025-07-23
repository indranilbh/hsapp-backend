# 🏕️ HSAPP Backend

This is the **FastAPI** backend for the Offbeat Homestay Discovery App – built to serve search, chat, and homestay information via APIs. The backend interacts with a PostgreSQL database hosted on **Supabase**, and supports both semantic and lexical filtering.

---

## 📁 Folder Structure

HSAPP-backend/
├── main.py # FastAPI entrypoint
├── requirements.txt # Python dependencies
├── .env # API keys and DB credentials (excluded from git)
├── Procfile # For deployment (Render/Fly.io)
├── models/ # SQLAlchemy models (optional)
├── routers/ # Modular API routes
├── utils/ # Utility functions

yaml
Copy
Edit

---

## 🚀 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/HSAPP-backend.git
cd HSAPP-backend
2. Create a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate     # For Linux/macOS
venv\Scripts\activate        # For Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Add a .env file
Create a .env file in the root folder and define the following:

env
Copy
Edit
DATABASE_URL=postgresql://username:password@host:port/dbname
OPENAI_API_KEY=your_openai_key_here
OTHER_API_KEY=your_optional_keys
▶️ Run the server locally
bash
Copy
Edit
uvicorn main:app --reload
Access at: http://localhost:8000

🧪 Key API Endpoints
Endpoint	Description
/chat	Handles NLP-based queries
/lexical_search	Performs structured dropdown search
/homestay/{id}	Fetch homestay details by ID
/guides, /drivers	Community guide/driver info
/filters	Returns cascading filters

☁️ Deployment Notes
Add Procfile with:

bash
Copy
Edit
web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000}
Use platforms like Render, Fly.io, or Railway.

Make sure to set environment variables securely in the hosting dashboard (never commit .env).

📦 Tech Stack
Python 3.10+

FastAPI

Uvicorn

PostgreSQL (Supabase)

OpenAI (for chat response)

FAISS (for vector search)

🤝 Contributing
PRs are welcome! If you spot a bug or have ideas to improve the logic, feel free to open an issue or pull request.

📄 License
This project is under MIT License.

yaml
Copy
Edit

---

Let me know if you want to generate a similar README for the frontend as well.
