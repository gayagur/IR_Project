# GCP Deployment Guide

מדריך מפורט להעלאת מנוע החיפוש ל-Google Cloud Platform.

## שלב 1: הכנת הפרויקט

### 1.1 בדיקת קבצים נדרשים

וודא שיש לך את כל הקבצים הבאים:
- ✅ `search_frontend.py` - אפליקציית Flask
- ✅ `requirements.txt` - תלויות Python
- ✅ `config.py` - הגדרות נתיבים
- ✅ כל הקבצים ב-`indexing/`, `ranking/`, `experiments/`
- ✅ `inverted_index_gcp.py` - תמיכה ב-GCP Storage

### 1.2 בניית האינדקסים

**חשוב**: האינדקסים צריכים להיבנות לפני ההעלאה ל-GCP.

```bash
# בניית כל האינדקסים
python -m indexing.build_indices --dump path/to/enwiki-latest-pages-articles.xml.bz2 --build all
```

זה יוצר:
- `data/indices/body/` - אינדקס body
- `data/indices/title/` - אינדקס title
- `data/indices/anchor/` - אינדקס anchor
- `data/aux/` - קבצים עזר (titles.pkl, doc_norms.pkl, וכו')

## שלב 2: העלאת אינדקסים ל-Google Storage

### 2.1 יצירת Bucket

```bash
# הגדר את PROJECT_ID שלך
export PROJECT_ID="healthy-highway-469706-h9"  # או PROJECT_ID שלך

# צור bucket חדש
gsutil mb -p $PROJECT_ID -l us-central1 gs://your-bucket-name-ir-project
```

### 2.2 העלאת קבצי אינדקס

```bash
# העלה את כל תיקיית data
gsutil -m cp -r data/ gs://your-bucket-name-ir-project/

# או העלה כל אינדקס בנפרד:
gsutil -m cp -r data/indices/body/ gs://your-bucket-name-ir-project/indices/body/
gsutil -m cp -r data/indices/title/ gs://your-bucket-name-ir-project/indices/title/
gsutil -m cp -r data/indices/anchor/ gs://your-bucket-name-ir-project/indices/anchor/
gsutil -m cp -r data/aux/ gs://your-bucket-name-ir-project/aux/
```

### 2.3 הפיכת Bucket לציבורי (לצורך הגשה)

```bash
# הפוך את ה-bucket לציבורי
gsutil iam ch allUsers:objectViewer gs://your-bucket-name-ir-project

# או רק קבצים ספציפיים:
gsutil -m acl ch -u AllUsers:R gs://your-bucket-name-ir-project/**
```

### 2.4 עדכון config.py ל-GCP (אופציונלי)

אם אתה רוצה לקרוא ישירות מ-GCP Storage, עדכן את `inverted_index_gcp.py`:

```python
# ב-inverted_index_gcp.py, שים את שם ה-bucket שלך:
BUCKET_NAME = "your-bucket-name-ir-project"
```

## שלב 3: יצירת Compute Engine Instance

### 3.1 יצירת Instance

```bash
# צור instance חדש
gcloud compute instances create ir-search-engine \
    --project=$PROJECT_ID \
    --zone=us-central1-a \
    --machine-type=e2-standard-4 \
    --network-tier=PREMIUM \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=YOUR_SERVICE_ACCOUNT@$PROJECT_ID.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server \
    --create-disk=auto-delete=yes,boot=yes,device-name=ir-search-engine,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20240110,mode=rw,size=50,type=projects/$PROJECT_ID/zones/us-central1-a/diskTypes/pd-standard \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
```

### 3.2 שמירת IP ציבורי

```bash
# שמור IP ציבורי קבוע
gcloud compute addresses create ir-search-engine-ip \
    --project=$PROJECT_ID \
    --region=us-central1

# קשר את ה-IP ל-instance
gcloud compute instances add-access-config ir-search-engine \
    --project=$PROJECT_ID \
    --zone=us-central1-a \
    --address=ir-search-engine-ip
```

### 3.3 פתיחת Firewall

```bash
# פתח פורט 8080
gcloud compute firewall-rules create allow-http-8080 \
    --project=$PROJECT_ID \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:8080 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=http-server
```

## שלב 4: העלאת קוד ל-Instance

### 4.1 העלאת קבצים

```bash
# העלה את כל הפרויקט
gcloud compute scp --recurse . ir-search-engine:~/IR_Project \
    --zone=us-central1-a

# או העלה קבצים ספציפיים:
gcloud compute scp search_frontend.py config.py requirements.txt ir-search-engine:~/IR_Project \
    --zone=us-central1-a
gcloud compute scp --recurse ranking/ indexing/ experiments/ ir-search-engine:~/IR_Project \
    --zone=us-central1-a
```

### 4.2 התחברות ל-Instance

```bash
gcloud compute ssh ir-search-engine --zone=us-central1-a
```

## שלב 5: התקנה והרצה ב-GCP

### 5.1 התקנת תלויות

```bash
# בתוך ה-instance
cd ~/IR_Project

# עדכן את המערכת
sudo apt-get update
sudo apt-get install -y python3 python3-pip

# התקן תלויות
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### 5.2 הורדת אינדקסים מ-Google Storage

```bash
# הורד את האינדקסים מה-bucket
gsutil -m cp -r gs://your-bucket-name-ir-project/data/ ./data/
```

### 5.3 הרצת מנוע החיפוש

```bash
# הרץ את האפליקציה ברקע
nohup python3 search_frontend.py > search_engine.log 2>&1 &

# או עם screen (מומלץ):
screen -S search_engine
python3 search_frontend.py
# לחץ Ctrl+A ואז D כדי לנתק
```

### 5.4 בדיקה

```bash
# בדוק שהשרת רץ
curl http://localhost:8080/search?query=test

# או מבחוץ (החלף ב-EXTERNAL_IP):
curl http://EXTERNAL_IP:8080/search?query=test
```

## שלב 6: הגדרת Startup Script (אופציונלי)

יצירת `startup_script_gcp.sh`:

```bash
#!/bin/bash
cd ~/IR_Project
pip3 install -r requirements.txt
python3 search_frontend.py
```

ואז הוסף ל-instance:

```bash
gcloud compute instances add-metadata ir-search-engine \
    --metadata-from-file startup-script=startup_script_gcp.sh \
    --zone=us-central1-a
```

## שלב 7: בדיקות סופיות

### 7.1 בדיקת כל ה-Endpoints

```bash
# בדוק את כל ה-endpoints
EXTERNAL_IP="YOUR_EXTERNAL_IP"

curl "http://$EXTERNAL_IP:8080/search?query=artificial intelligence"
curl "http://$EXTERNAL_IP:8080/search_body?query=machine learning"
curl "http://$EXTERNAL_IP:8080/search_title?query=python"
curl "http://$EXTERNAL_IP:8080/search_anchor?query=deep learning"
curl "http://$EXTERNAL_IP:8080/search_pagerank?query=computer science"
curl "http://$EXTERNAL_IP:8080/search_pageview?query=technology"
```

### 7.2 בדיקת ביצועים

```bash
# הרץ הערכה
python3 experiments/run_evaluation.py \
    --base-url http://$EXTERNAL_IP:8080 \
    --queries queries_train.json
```

## פתרון בעיות

### בעיה: Instance לא מגיב
```bash
# בדוק את הלוגים
tail -f search_engine.log

# בדוק שהפורט פתוח
sudo netstat -tlnp | grep 8080
```

### בעיה: אין גישה ל-Google Storage
```bash
# בדוק הרשאות
gcloud auth application-default login

# או הגדר service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### בעיה: אינדקסים לא נטענים
```bash
# בדוק שהקבצים קיימים
ls -lh data/indices/body/
ls -lh data/aux/

# בדוק הרשאות
chmod -R 755 data/
```

## הערות חשובות

1. **גודל דיסק**: ודא שיש מספיק מקום (לפחות 50GB) לאינדקסים
2. **זיכרון**: מומלץ לפחות 4GB RAM (e2-standard-4)
3. **עלויות**: Compute Engine עלול לעלות כסף - זכור לכבות את ה-instance כשלא בשימוש
4. **אבטחה**: שקול להוסיף HTTPS עם Load Balancer
5. **Backup**: שמור גיבוי של האינדקסים ב-Google Storage

## קישורים שימושיים

- **IP ציבורי**: `http://YOUR_EXTERNAL_IP:8080`
- **Google Storage Bucket**: `gs://your-bucket-name-ir-project`
- **Compute Engine Console**: https://console.cloud.google.com/compute/instances

## סיכום

לאחר השלמת כל השלבים, המנוע שלך אמור להיות:
- ✅ נגיש דרך IP ציבורי
- ✅ מחזיר תוצאות לכל ה-endpoints
- ✅ עובד תוך פחות מ-35 שניות לכל שאילתה
- ✅ מוכן לבדיקות

**URL להגשה**: `http://YOUR_EXTERNAL_IP:8080`


