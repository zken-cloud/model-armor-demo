# GenAI Chatbot with Advanced Model Armor Integration

This web application demonstrates a sophisticated integration of Google Cloud's Vertex AI foundation models with Model Armor. It provides a highly configurable and performant interface for testing prompt/response sanitization and multimodal interactions.

## Key Features

* **Text and File Input Types**: Supports file uploads (PDF, DOCX, PPTX, XLSX) in addition to text prompts, enabling analysis of document-based content.
* **Advanced Model Armor Analysis**:
    * Dynamically loads Model Armor templates based on the selected GCP region.
    * Provides a detailed breakdown of which Model Armor filters (Responsible AI, Sensitive Data, Jailbreak/PI, etc.) passed or failed for both the prompt and the model's response.
    * Option to view the raw JSON output from the Model Armor API for in-depth analysis.
* **Configurable Safeguards**:
    * **System Instructions**: Set a persistent system-level instruction to guide the model's behavior across a conversation.
    * **Default Responses**: Configure the application to return a pre-defined default response if either the prompt or the model's output violates a policy.
* **Performance Optimizations**:
    * **Asynchronous Processing**: The backend leverages `asyncio` to perform prompt analysis and model generation in parallel for a faster user experience.
    * **Backend Caching**: Caches Model Armor API results and templates to reduce latency and redundant API calls.
    * **Optimized HTTP Client**: Uses an HTTP session with connection pooling and automatic retries for more resilient communication with Google Cloud APIs.

## Deployment Guide

Follow these step-by-step instructions to deploy the Model Armor Demo application to Google Cloud.

### Prerequisites

1.  **Google Cloud Project**: You need a Google Cloud project with billing enabled.
2.  **Google Cloud SDK (gcloud)**: Ensure you have the `gcloud` CLI installed and authenticated.
    ```bash
    gcloud auth login
    gcloud config set project YOUR_PROJECT_ID
    ```

### Step 1: Enable Required APIs

Enable the APIs for Vertex AI, Model Armor, Artifact Registry, and Cloud Run in your project.

```bash
gcloud services enable aiplatform.googleapis.com \
  modelarmor.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com
```

### Step 2: Create a Service Account

It is recommended to run the application with a dedicated service account with least privileges.

1.  **Create the service account:**
    ```bash
    gcloud iam service-accounts create model-armor-demo-sa \
      --display-name="Model Armor Demo Service Account"
    ```

2.  **Grant Model Armor Admin role** (required to create and manage templates):
    ```bash
    gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
      --member="serviceAccount:model-armor-demo-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
      --role="roles/modelarmor.admin"
    ```

3.  **Grant Vertex AI User role** (required for model inference):
    ```bash
    gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
      --member="serviceAccount:model-armor-demo-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
      --role="roles/aiplatform.user"
    ```

### Step 3: Local Setup & Testing (Optional)

If you want to test the application locally before deploying:

1.  **Clone the repository** and navigate to the directory.
2.  **Create a `.env` file**:
    ```bash
    cat <<EOF > .env
    GCP_PROJECT_ID=YOUR_PROJECT_ID
    GCP_LOCATION=us-central1
    EOF
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the app**:
    ```bash
    python app.py
    ```
    Access the app at `http://127.0.0.1:8080`.

### Step 4: Deploy to Cloud Run

You can deploy the application to Cloud Run using either Cloud Build or manual commands.

#### Option A: Using Cloud Build (Recommended)

A sample `cloudbuild.yaml.example` file is provided in the repository.

1.  **Copy the example file** to `cloudbuild.yaml`:
    ```bash
    cp cloudbuild.yaml.example cloudbuild.yaml
    ```
2.  **Submit the build** using the configuration file:
    ```bash
    gcloud builds submit --config cloudbuild.yaml .
    ```
    You can override the default values by passing substitutions:
    ```bash
    gcloud builds submit --config cloudbuild.yaml \
      --substitutions=_LOCATION=us-east1,_SERVICE_NAME=my-custom-chat-app .
    ```

#### Option B: Manual Deployment

1.  **Create an Artifact Registry repository** (if you don't have one):
    ```bash
    gcloud artifacts repositories create model-armor-repo \
      --repository-format=docker \
      --location=us-central1 \
      --description="Model Armor Demo Docker Repository"
    ```

2.  **Build the container image:**
    ```bash
    gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/model-armor-repo/model-armor-demo .
    ```

3.  **Deploy to Cloud Run:**
    ```bash
    gcloud run deploy model-armor-demo \
      --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/model-armor-repo/model-armor-demo \
      --region us-central1 \
      --platform managed \
      --service-account model-armor-demo-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
      --allow-unauthenticated \
      --set-env-vars="GCP_PROJECT_ID=YOUR_PROJECT_ID,GCP_LOCATION=global,MODEL_ARMOR_ENABLED=true"
    ```

After deployment, Cloud Run will provide a URL where you can access the application.
