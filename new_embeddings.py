import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import torch
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
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
    """Generate and store embedding for a single job posting."""
    job_id = str(job['_id'])
    job_text = f"{job.get('job_title', '')} {' '.join(job.get('skills', []))} {job.get('description', '')} {job.get('location', '')}"
    
    # Encode text
    with torch.no_grad():
        embedding = model.encode([job_text], batch_size=1, convert_to_tensor=True).cpu().numpy().tolist()[0]

    # Store in Pinecone
    metadata = {
        "job_id": job_id,
        "title": job.get('job_title', ''),
        "skills": job.get('skills', []),
        "description": job.get('description', ''),
        "location": job.get('location', '')
    }
    index.upsert([(job_id, embedding, metadata)])

    # Mark as processed in MongoDB
    job_collection.update_one({"_id": ObjectId(job_id)}, {"$set": {"processed": True}})
    print(f"Embedding stored for job: {job_id}")


def watch_new_jobs():
    """Continuously watch MongoDB for new job postings."""
    print("Watching for new job postings...")
    with job_collection.watch() as stream:
        for change in stream:
            if change["operationType"] == "insert":
                new_job = change["fullDocument"]
                if not new_job.get("processed", False):
                    create_embedding_for_new_job(new_job)


# FastAPI Endpoint to Manually Trigger Watching for Jobs
@app.get("/start-watching")
def start_watching(background_tasks: BackgroundTasks):
    background_tasks.add_task(watch_new_jobs)
    return {"message": "Started watching for new jobs in MongoDB!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
