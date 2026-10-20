#!/usr/bin/env bash
# ==============================================================================
# Deploy Brooklyn Urban Heat Resilience Planner to Google Cloud Run
# Usage:
#   bash scripts/deploy_cloud_run.sh [SERVICE_NAME] [REGION]
# ==============================================================================
set -euo pipefail

SERVICE_NAME="${1:-brooklyn-heat-planner}"
REGION="${2:-us-central1}"
PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: No active GCP project. Run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

echo "==> Deploying ${SERVICE_NAME} to Cloud Run in project ${PROJECT_ID} (${REGION})..."
echo "==> Using --min-instances=1 to eliminate cold-start latency during live demo judging."

gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --min-instances=1 \
  --max-instances=5 \
  --memory=1Gi \
  --cpu=1 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},BQ_DATASET=a4i_climate"

echo "==> Deployment complete!"
gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format="value(status.url)"
