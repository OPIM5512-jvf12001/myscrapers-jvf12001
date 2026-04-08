import os
from google.cloud import pubsub_v1
from google.cloud import storage

# --- CONFIGURATION ---
PROJECT_ID = "myscrapers-jvf12001"
TOPIC_ID = "craigslist-backlog-topic"
BUCKET_NAME = "myscrapers-jvf12001"

def trigger_backlog():
    # Initialize Clients
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    storage_client = storage.Client(project=PROJECT_ID)
    
    print(f"Scanning bucket {BUCKET_NAME} for runs...")

    # List all run folders in 'scrapes/' using delimiter to get virtual 'folders'
    blobs = storage_client.list_blobs(BUCKET_NAME, prefix="scrapes/", delimiter="/")
    
    # Iterate through the blobs to populate the 'prefixes' property
    for _ in blobs:
        pass
    
    run_folders = blobs.prefixes

    if not run_folders:
        print("No run folders found in scrapes/.")
        return

    print(f"Found {len(run_folders)} runs. Sending to Pub/Sub...")

    for folder_path in run_folders:
        # Clean up the path to get just the run_id string
        run_id = folder_path.strip("/").split("/")[-1]
        
        data = run_id.encode("utf-8")
        
        # Trigger the cloud function
        future = publisher.publish(topic_path, data)
        print(f"Enqueued: {run_id} | Message ID: {future.result()}")


if __name__ == "__main__":
    trigger_backlog()