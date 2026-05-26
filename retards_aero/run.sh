# 2. Lancement avec ton secrets.sh
source secrets.sh

docker run --rm \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
  ppml-api:latest
