import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import torch
from pinecone import Pinecone, ServerlessSpec
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
import uvicorn

load_dotenv()

# FastAPI App
app = FastAPI()

# Load environment variables
MONGO_URI = os.getenv('MONGO_URI')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = 'job-posting-embeddings2'

# Load the SentenceTransformer model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# MongoDB Connection
client = MongoClient(MONGO_URI)
db = client['jobPortalDB']
job_collection = db['jobposts']

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists; if not, create it
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1'),
    )

# Connect to the index
index = pc.Index(INDEX_NAME)


def create_embedding_for_new_job(job):
    """Generate and store embedding for a single job posting and debug the process."""
    job_id = str(job['_id'])
    job_text = f"{job.get('job_title', '')} {' '.join(job.get('skills', []))} {job.get('description', '')} {job.get('location', '')}"
    
    print(f"Processing job: {job_id}")  # Debugging print
    
    # Encode text
    with torch.no_grad():
        embedding = model.encode([job_text], batch_size=1, convert_to_tensor=True).cpu().numpy().tolist()[0]

    print(f"Generated embedding for job: {job_id}")  # Debugging print

    # Store in Pinecone
    metadata = {
        "job_id": job_id,
        "title": job.get('job_title', ''),
        "skills": job.get('skills', []),
        "description": job.get('description', ''),
        "location": job.get('location', '')
    }
    
    try:
        index.upsert([(job_id, embedding, metadata)])
        print(f"Successfully stored embedding in Pinecone for job: {job_id}")  # Debugging print
    except Exception as e:
        print(f"Error inserting into Pinecone for job {job_id}: {e}")  # Debugging print

    # Mark as processed in MongoDB
    job_collection.update_one({"_id": ObjectId(job_id)}, {"$set": {"processed": True}})
    print(f"Marked job {job_id} as processed in MongoDB")  # Debugging print


def watch_new_jobs():
    """Continuously watch MongoDB for new job postings."""
    print("Watching for new job postings...")
    with job_collection.watch() as stream:
        for change in stream:
            if change["operationType"] == "insert":
                new_job = change["fullDocument"]
                print(f"New job detected: {new_job['_id']}")  # Debugging print
                if not new_job.get("processed", False):
                    create_embedding_for_new_job(new_job)


@app.get("/start-watching")
def start_watching(background_tasks: BackgroundTasks):
    """Start background process to watch for new jobs."""
    background_tasks.add_task(watch_new_jobs)
    return {"message": "Started watching for new jobs in MongoDB!"}


@app.post("/process-new-jobs")
def process_new_jobs():
    """Find unprocessed jobs and store embeddings in Pinecone."""
    unprocessed_jobs = job_collection.find({"processed": False})
    count = 0

    for job in unprocessed_jobs:
        create_embedding_for_new_job(job)
        count += 1

    return {"message": f"{count} new jobs processed and stored in Pinecone."}


@app.get("/debug-unprocessed-jobs")
def debug_unprocessed_jobs():
    """Check unprocessed jobs in MongoDB for debugging."""
    unprocessed_jobs = list(job_collection.find({"processed": False}))
    return {"unprocessed_jobs": [str(job["_id"]) for job in unprocessed_jobs]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
