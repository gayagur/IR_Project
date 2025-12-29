#!/bin/bash
# build_indices_on_gcp.sh
# Script to build indices on GCP instance - pulls code from GCS bucket

set -e

INSTANCE_NAME="indexing-instance"
ZONE="us-central1-c"
PROJECT_NAME="ir-project-481821"
BUCKET_NAME="matiasgaya333"

echo "=========================================="
echo "Building Indices on GCP Instance"
echo "=========================================="

# 1. Create instance (if not exists)
echo "Step 1: Creating/Checking instance..."
if ! gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_NAME &>/dev/null; then
    echo "Creating instance $INSTANCE_NAME..."
    gcloud compute instances create $INSTANCE_NAME \
      --zone=$ZONE \
      --project=$PROJECT_NAME \
      --machine-type=e2-highmem-8 \
      --boot-disk-size=200GB \
      --image-family=ubuntu-2204-lts \
      --image-project=ubuntu-os-cloud \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --tags=http-server
    
    echo "Waiting for instance to be ready..."
    sleep 30
else
    echo "Instance $INSTANCE_NAME already exists"
fi

# 2. Run setup and indexing on instance
echo "Step 2: Setting up instance and running indexing..."
gcloud compute ssh $INSTANCE_NAME \
  --zone=$ZONE \
  --project=$PROJECT_NAME << ENDSSH
set -e

echo "=========================================="
echo "Setting up environment..."
echo "=========================================="

# Auto-shutdown after 6 hours (safety)
sudo shutdown -h +360 || true

# Install dependencies
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv -qq

# Create project directory
mkdir -p ~/IR_Project
cd ~/IR_Project

# Download code from GCS bucket
echo "Downloading code from gs://${BUCKET_NAME}/code/..."
gsutil -m cp -r gs://matiasgaya333/IR_Project/* .

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip -q
pip install -q google-cloud-storage pyarrow pandas mwparserfromhell

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
fi

# Make sure config is set to write locally
sed -i 's/WRITE_TO_GCS = True/WRITE_TO_GCS = False/' config.py
sed -i 's/READ_FROM_GCS = True/READ_FROM_GCS = False/' config.py

echo "=========================================="
echo "Starting index builds..."
echo "=========================================="

# Build body, title, anchor indices
echo "Building body, title, anchor indices..."
nohup python3 -m indexing.build_indices \
  --dump "gs://${BUCKET_NAME}/raw/wikidata20210801_preprocessed/" \
  --build all \
  --parquet > build_all.log 2>&1 &

ALL_PID=\$!
echo "Body/title/anchor build started (PID: \$ALL_PID)"

# Build PageViews (doesn't need dump, downloads from Wikimedia)
echo ""
echo "Building PageViews..."
nohup python3 -m indexing.build_indices \
  --dump dummy \
  --build pageviews > build_pageviews.log 2>&1 &

PAGEVIEWS_PID=\$!
echo "PageViews build started (PID: \$PAGEVIEWS_PID)"

# Build PageRank (needs dump and titles.pkl)
echo ""
echo "Building PageRank..."
nohup python3 -m indexing.build_indices \
  --dump "gs://${BUCKET_NAME}/raw/wikidata20210801_preprocessed/" \
  --build pagerank \
  --parquet > build_pagerank.log 2>&1 &

PAGERANK_PID=\$!
echo "PageRank build started (PID: \$PAGERANK_PID)"

echo ""
echo "=========================================="
echo "All builds running in background!"
echo "=========================================="
echo ""
echo "Processes:"
echo "  - Body/title/anchor: PID \$ALL_PID"
echo "  - PageViews: PID \$PAGEVIEWS_PID"
echo "  - PageRank: PID \$PAGERANK_PID"
echo ""
echo "You can disconnect - processes will continue."
echo ""
echo "To check progress:"
echo "  tail -f ~/IR_Project/build_all.log"
echo "  tail -f ~/IR_Project/build_pageviews.log"
echo "  tail -f ~/IR_Project/build_pagerank.log"
echo ""
echo "To check if processes are still running:"
echo "  ps aux | grep build_indices | grep -v grep"
echo ""
echo "When done, upload to GCS:"
echo "  gsutil -m rsync -r ~/IR_Project/indices gs://${BUCKET_NAME}/indices"
echo "  gsutil -m rsync -r ~/IR_Project/aux gs://${BUCKET_NAME}/aux"
echo ""
ENDSSH

echo ""
echo "=========================================="
echo "Done! Instance is building indices."
echo "=========================================="
echo ""
echo "To reconnect:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_NAME"
echo ""
echo "To delete instance when done:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_NAME"