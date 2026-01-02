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
echo "[1/9] Setting up static IP..."
gcloud compute addresses create $IP_NAME --project=$PROJECT_NAME --region=$REGION 2>/dev/null || echo "IP already exists"
INSTANCE_IP=$(gcloud compute addresses describe $IP_NAME --region=$REGION --format="get(address)")
echo "External IP: $INSTANCE_IP"

# 2. Create firewall rule (ignore error if exists)
echo "[2/9] Setting up firewall..."
gcloud compute firewall-rules create default-allow-http-8080 \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --target-tags http-server 2>/dev/null || echo "Firewall rule already exists"

# 3. Create VM instance
echo "[3/9] Creating VM instance (this takes ~60 seconds)..."
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
echo "[4/9] Installing dependencies on VM..."
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME --zone $ZONE --command="
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-pip python3-venv
  python3 -m venv ~/venv
  ~/venv/bin/pip install --quiet flask google-cloud-storage pandas numpy scikit-learn scipy
"

# 5. Create directories and download indices from GCS
echo "[5/9] Downloading indices from GCS to VM (this may take a while)..."
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

# 6. Download latest code from GCS bucket to Cloud Shell (to ensure it's up to date)
echo "[6/9] Downloading latest code from GCS bucket to Cloud Shell..."
mkdir -p ~/IR_Project
gsutil -m cp -r gs://${BUCKET_NAME}/IR_Project/* ~/IR_Project/ 2>/dev/null || {
  echo "⚠ Warning: Could not download code from gs://${BUCKET_NAME}/IR_Project/"
  echo "Will use local code in ~/IR_Project/"
}

# 7. Update config.py with the correct IP and copy Python code to VM
echo "[7/9] Updating config.py with instance IP and copying Python code to VM..."
# Update INSTANCE_IP in config.py before copying (if file exists locally)
if [ -f ~/IR_Project/config.py ]; then
  # Backup original config.py
  cp ~/IR_Project/config.py ~/IR_Project/config.py.bak 2>/dev/null || true
  # Update INSTANCE_IP in config.py
  sed -i "s|INSTANCE_IP = \".*\"|INSTANCE_IP = \"${INSTANCE_IP}\"|" ~/IR_Project/config.py
  # Update BASE_URL as well (escape the f-string properly)
  sed -i "s|BASE_URL = f\"http://.*\"|BASE_URL = f\"http://${INSTANCE_IP}:8080\"|" ~/IR_Project/config.py
fi

gcloud compute scp --recurse \
  ~/IR_Project/search_frontend.py \
  ~/IR_Project/search_runtime.py \
  ~/IR_Project/config.py \
  ~/IR_Project/inverted_index_gcp.py \
  ~/IR_Project/text_processing.py \
  ~/IR_Project/parser_utils.py \
  ~/IR_Project/ranking \
  ~/IR_Project/requirements.txt \
  ${GOOGLE_ACCOUNT_NAME}@${INSTANCE_NAME}:${PROJECT_DIR}/ \
  --zone ${ZONE} 2>/dev/null || {
  echo "⚠ Warning: Some files might not exist locally"
  echo "Continuing with available files..."
}

# Restore original config.py if we modified it
if [ -f ~/IR_Project/config.py.bak ]; then
  mv ~/IR_Project/config.py.bak ~/IR_Project/config.py 2>/dev/null || true
fi

# 8. Install additional dependencies from requirements.txt (if exists)
echo "[8/9] Installing additional dependencies from requirements.txt..."
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME --zone $ZONE --command="
  if [ -f ${PROJECT_DIR}/requirements.txt ]; then
    ~/venv/bin/pip install --quiet -r ${PROJECT_DIR}/requirements.txt
  fi
"

# 9. Start the server
echo "[9/9] Starting the search server..."
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME --zone $ZONE --command="
  cd ${PROJECT_DIR} || { echo 'Error: Failed to cd to ${PROJECT_DIR}'; exit 1; }
  pwd
  ls -la search_frontend.py || { echo 'Error: search_frontend.py not found in ${PROJECT_DIR}'; ls -la; exit 1; }
  nohup ~/venv/bin/python ${PROJECT_DIR}/search_frontend.py > ~/frontend.log 2>&1 &
  sleep 5
  echo 'Server started! Testing with external IP from config...'
  cd ${PROJECT_DIR}
  INSTANCE_IP=\$(~/venv/bin/python -c 'import config; print(config.INSTANCE_IP)')
  curl -s \"http://\${INSTANCE_IP}:8080/search?query=hello\" || echo 'Server might need more time to load indices'
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
