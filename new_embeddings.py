import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import torch
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from bson import ObjectId
import time 
from dotenv import load_dotenv

load_dotenv()

# Load environment variabledo
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


# Fetch only new job descriptions from MongoDB
def fetch_new_job_descriptions():
    print("Fetching new job descriptions from MongoDB...")
    cursor = job_collection.find(
        {"processed": {"$ne": True}}, 
        {'_id': 1, 'job_title': 1, 'skills': 1, 'description': 1, 'location': 1}
    )
    jobs = [
        {
            'id': str(job['_id']),
            'title': job.get('job_title', ''),
            'skills': job.get('skills', []),
            'description': job.get('description', ''),
            'location': job.get('location', '')
        }
        for job in cursor
    ]
    
    print(f"Found {len(jobs)} new job descriptions.")
    return jobs


# Fetch existing job IDs from Pinecone in batches
def fetch_existing_embeddings(job_ids, batch_size=100):
    """Check which job IDs already have embeddings in Pinecone in batches."""
    print("Checking existing embeddings in Pinecone...")
    existing_ids = set()

    for i in range(0, len(job_ids), batch_size):
        batch_ids = job_ids[i:i + batch_size]
        query_results = index.fetch(ids=batch_ids)  # Fetch existing embeddings

        # Extract existing job IDs properly
        if query_results.vectors:  
            existing_ids.update(query_results.vectors.keys())

    print(f"Found {len(existing_ids)} existing embeddings in Pinecone.")
    return existing_ids


# Create embeddings only for new jobs
def create_new_embeddings(batch_size=32):
    print("\nStarting embedding creation process...\n")
    job_descriptions = fetch_new_job_descriptions()
    if not job_descriptions:
        print("No new job postings found. Exiting...")
        return

    job_ids = [job['id'] for job in job_descriptions]
    existing_ids = fetch_existing_embeddings(job_ids)

    # Filter out jobs that already have embeddings
    new_jobs = [job for job in job_descriptions if job['id'] not in existing_ids]

    if not new_jobs:
        print("All jobs already have embeddings. No new embeddings to create.")
        return

    print(f"Creating embeddings for {len(new_jobs)} new job postings...")

    job_texts, job_ids, metadata_list = [], [], []

    for job in new_jobs:
        job_text = f"{job['title']} {' '.join(job['skills'])} {job['description']} {job['location']}"
        metadata = {
            "job_id": job['id'],
            "title": job['title'],
            "skills": job['skills'],
            "description": job['description'],
            "location": job['location']
        }
        job_ids.append(job['id'])
        job_texts.append(job_text)
        metadata_list.append(metadata)

    # Process in batches
    for i in tqdm(range(0, len(job_texts), batch_size), desc="Creating New Embeddings"):
        batch_ids = job_ids[i:i + batch_size]
        batch_texts = job_texts[i:i + batch_size]
        batch_metadata = metadata_list[i:i + batch_size]

        # Encode the batch
        with torch.no_grad():
            embeddings = model.encode(batch_texts, batch_size=batch_size, convert_to_tensor=True).cpu().numpy().tolist()

        # Batch upsert into Pinecone
        vectors = [(batch_ids[j], embeddings[j], batch_metadata[j]) for j in range(len(batch_ids))]
        index.upsert(vectors)

    # Convert job IDs back to ObjectId and mark as processed in MongoDB
    job_collection.update_many(
        {"_id": {"$in": [ObjectId(job['id']) for job in new_jobs]}}, 
        {"$set": {"processed": True}}
    )

    print("Embedding creation complete. Jobs marked as processed in MongoDB.")


if __name__ == "__main__":
    while True:
        create_new_embeddings()
        print("Sleeping for 1 hour...")
        time.sleep(3600)  # Sleep for 1 hour
