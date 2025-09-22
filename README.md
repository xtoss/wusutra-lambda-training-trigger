# Wusutra Lambda Training Trigger

AWS Lambda function that triggers SageMaker training jobs for Whisper model fine-tuning.

## Structure

```
wusutra-lambda-training-trigger/
├── training_trigger/         # Lambda function code
│   ├── lambda_function.py   # Main handler
│   ├── requirements.txt     # SageMaker training dependencies
│   └── test_lambda_local.py # Local testing
├── training_scripts/        # Scripts deployed to SageMaker
│   ├── train_whisper.py     # Full/incremental training
│   └── train_whisper_lora.py # LoRA training
├── serverless.yml          # Serverless deployment config
├── requirements.txt        # Lambda runtime dependencies
└── README.md
```

## Functionality

### Lambda Function
- **Endpoint**: `POST /v1/training/trigger`
- **Query Parameters**: `mode` (incremental|full|lora)
- **Purpose**: Triggers SageMaker training jobs with different training modes

### Training Modes
- **incremental**: Updates existing model with new audio files
- **full**: Complete retraining from scratch
- **lora**: Parameter-efficient training using LoRA

## Environment Variables
- `S3_BUCKET`: S3 bucket for audio files and models
- `SAGEMAKER_ROLE_ARN`: IAM role for SageMaker execution
- `SAGEMAKER_IMAGE_URI`: Docker image for training (optional)

## Deployment

```bash
# Install serverless
npm install -g serverless
npm install serverless-python-requirements

# Deploy
serverless deploy

# Deploy to specific stage
serverless deploy --stage prod
```

## Usage

```bash
# Trigger incremental training
curl -X POST https://your-api-gateway/v1/training/trigger?mode=incremental

# Trigger full training
curl -X POST https://your-api-gateway/v1/training/trigger?mode=full

# Trigger LoRA training
curl -X POST https://your-api-gateway/v1/training/trigger?mode=lora
```

## Dependencies

The Lambda function requires minimal dependencies (only boto3). Training scripts are uploaded to S3 and executed in SageMaker containers with their own dependencies.