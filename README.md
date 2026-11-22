# STATGUARD backend (FastAPI)

## Run locally
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
Open http://localhost:8000/docs to try the endpoints.

## Deploy on Render (free)
1. Put these files in a GitHub repo (main.py, engine.py, requirements.txt, render.yaml at the repo root).
2. Render dashboard -> New -> Web Service -> connect the repo.
3. Build: `pip install -r requirements.txt`   Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add env var PYTHON_VERSION = 3.12.3. Deploy.
5. Open https://YOUR-SERVICE.onrender.com/health  -> {"status":"ok"}

## Notes
- Data is held in memory (last 3 uploads). A restart or free-tier sleep clears it; just re-upload.
- Free tier sleeps after ~15 min idle; the first request can take 30-60 s.
- Set ALLOWED_ORIGINS=https://statguard.vercel.app to restrict CORS (default: any origin).
- Upload limit: MAX_UPLOAD_MB (default 30).
