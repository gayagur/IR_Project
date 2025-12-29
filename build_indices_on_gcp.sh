#!/bin/bash
# build_indices_on_gcp.sh
# Script to build indices on GCP instance efficiently

set -e

INSTANCE_NAME="indexing-instance"
ZONE="us-central1-c"
PROJECT_NAME="ir-project-481821"
GOOGLE_ACCOUNT_NAME="gayag"

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

# 2. Upload code
echo "Step 2: Uploading code to instance..."
gcloud compute scp --recurse . \
  ${GOOGLE_ACCOUNT_NAME}@${INSTANCE_NAME}:~/IR_Project \
  --zone=$ZONE \
  --project=$PROJECT_NAME

# 3. Run indexing on instance
echo "Step 3: Running indexing on instance..."
gcloud compute ssh ${GOOGLE_ACCOUNT_NAME}@${INSTANCE_NAME} \
  --zone=$ZONE \
  --project=$PROJECT_NAME << 'ENDSSH'
set -e

cd ~/IR_Project

# Stop the VM automatically after 3 hours (even if indexing is still running)
# This saves cost if you fall asleep.
sudo shutdown -h +300 || true

# Setup environment
echo "Setting up Python environment..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv -qq

# Create venv if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip -q

# Install deps (requirements may or may not exist)
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
fi

# Parquet + GCS support
pip install -q google-cloud-storage pyarrow pandas mwparserfromhell

# Build indices locally on VM disk (GCS is only for reading input data)
echo "Building indices locally on VM disk (this may take several hours)..."
echo "Reading input data from GCS: gs://matiasgaya333/raw/wikidata20210801_preprocessed/"

python3 -m indexing.build_indices \
  --dump "gs://matiasgaya333/raw/wikidata20210801_preprocessed/" \
  --build body \
  --parquet

echo ""
echo "Index building completed!"
echo "To upload results to GCS:"
echo "  gsutil -m rsync -r indices gs://matiasgaya333/indices"
echo "  gsutil -m rsync -r aux     gs://matiasgaya333/aux"
ENDSSH


echo ""
echo "=========================================="
echo "Index building completed!"
echo "=========================================="
echo ""
echo "To delete the instance (to save costs):"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_NAME"
echo ""

