#!/bin/bash
# ============================================================
# Ready-to-run script from Cloud Shell
# Run: cd ~/IR_Project && bash run_frontend_in_gcp.sh
# ============================================================

set -e  # Stop on any error

# Configuration
INSTANCE_NAME="instance-1"
REGION="us-central1"
ZONE="us-central1-c"
PROJECT_NAME="ir-project-481821"
IP_NAME="${PROJECT_NAME}-ip"
GOOGLE_ACCOUNT_NAME="gayag"
BUCKET_NAME="matiasgaya333"
PROJECT_DIR="/home/${GOOGLE_ACCOUNT_NAME}/IR_Project"

echo "============================================"
echo "Starting deployment..."
echo "============================================"

# 1. Reserve static IP (ignore error if exists)
echo "[1/7] Setting up static IP..."
gcloud compute addresses create $IP_NAME --project=$PROJECT_NAME --region=$REGION 2>/dev/null || echo "IP already exists"
INSTANCE_IP=$(gcloud compute addresses describe $IP_NAME --region=$REGION --format="get(address)")
echo "External IP: $INSTANCE_IP"

# 2. Create firewall rule (ignore error if exists)
echo "[2/7] Setting up firewall..."
gcloud compute firewall-rules create default-allow-http-8080 \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --target-tags http-server 2>/dev/null || echo "Firewall rule already exists"

# 3. Create VM instance
echo "[3/7] Creating VM instance (this takes ~60 seconds)..."
gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=e2-standard-2 \
  --network-interface=address=$INSTANCE_IP,network-tier=PREMIUM,subnet=default \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=http-server \
  --boot-disk-size=50GB 2>/dev/null || echo "Instance might already exist"

echo "Waiting 60 seconds for instance to be ready..."
sleep 60

# 4. Install Python and dependencies on VM
echo "[4/7] Installing dependencies on VM..."
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME --zone $ZONE --command="
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-pip python3-venv
  python3 -m venv ~/venv
  ~/venv/bin/pip install --quiet flask google-cloud-storage pandas numpy
"

# 5. Create directories and download indices from GCS
echo "[5/7] Downloading indices from GCS to VM (this may take a while)..."
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME --zone $ZONE --command="
  mkdir -p ${PROJECT_DIR}/indices/body
  mkdir -p ${PROJECT_DIR}/indices/title
  mkdir -p ${PROJECT_DIR}/indices/anchor
  mkdir -p ${PROJECT_DIR}/aux
  mkdir -p ${PROJECT_DIR}/ranking
  
  echo 'Downloading indices...'
  gsutil -m cp -r gs://${BUCKET_NAME}/indices/body/* ${PROJECT_DIR}/indices/body/ 2>/dev/null || echo 'Body index not found in GCS'
  gsutil -m cp -r gs://${BUCKET_NAME}/indices/title/* ${PROJECT_DIR}/indices/title/ 2>/dev/null || echo 'Title index not found in GCS'
  gsutil -m cp -r gs://${BUCKET_NAME}/indices/anchor/* ${PROJECT_DIR}/indices/anchor/ 2>/dev/null || echo 'Anchor index not found in GCS'
  gsutil -m cp -r gs://${BUCKET_NAME}/aux/* ${PROJECT_DIR}/aux/ 2>/dev/null || echo 'Aux files not found in GCS'
  
  echo 'Files downloaded:'
  ls -la ${PROJECT_DIR}/indices/
  ls -la ${PROJECT_DIR}/aux/
"

# 6. Copy Python code from Cloud Shell to VM
echo "[6/7] Copying Python code to VM..."
gcloud compute scp --recurse \
  ~/IR_Project/search_frontend.py \
  ~/IR_Project/search_runtime.py \
  ~/IR_Project/config.py \
  ~/IR_Project/inverted_index_gcp.py \
  ~/IR_Project/text_processing.py \
  ~/IR_Project/parser_utils.py \
  ~/IR_Project/ranking \
  ${GOOGLE_ACCOUNT_NAME}@${INSTANCE_NAME}:${PROJECT_DIR}/ \
  --zone ${ZONE}

# 7. Start the server
echo "[7/7] Starting the search server..."
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME --zone $ZONE --command="
  cd ${PROJECT_DIR}
  nohup ~/venv/bin/python search_frontend.py > ~/frontend.log 2>&1 &
  sleep 5
  echo 'Server started! Testing...'
  curl -s 'http://localhost:8080/search?query=hello' || echo 'Server might need more time to load indices'
"

echo ""
echo "============================================"
echo "DEPLOYMENT COMPLETE!"
echo "============================================"
echo ""
echo "Your search engine is running at:"
echo "  http://${INSTANCE_IP}:8080"
echo ""
echo "Test it with:"
echo "  curl 'http://${INSTANCE_IP}:8080/search?query=computer+science'"
echo ""
echo "To check logs:"
echo "  gcloud compute ssh ${GOOGLE_ACCOUNT_NAME}@${INSTANCE_NAME} --zone ${ZONE} --command='tail -f ~/frontend.log'"
echo ""
echo "To SSH into the VM:"
echo "  gcloud compute ssh ${GOOGLE_ACCOUNT_NAME}@${INSTANCE_NAME} --zone ${ZONE}"
echo ""
echo "============================================"
