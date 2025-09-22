import json
import os
import boto3
import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker')

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', 'wusutra-audio-files')
SAGEMAKER_ROLE = os.environ.get('SAGEMAKER_ROLE_ARN')
SAGEMAKER_IMAGE_URI = os.environ.get('SAGEMAKER_IMAGE_URI', '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker')

def lambda_handler(event, context):
    """
    AWS Lambda handler for triggering SageMaker training
    
    Expected event format:
    {
        "httpMethod": "POST" | "OPTIONS",
        "queryStringParameters": {
            "mode": "incremental" | "full" | "lora"
        }
    }
    """
    try:
        # CORS headers for all responses
        cors_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Content-Type': 'application/json'
        }
        
        # Parse request
        http_method = event.get('httpMethod', 'POST')
        
        # Handle preflight OPTIONS request
        if http_method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': json.dumps({'message': 'CORS preflight'})
            }
        
        if http_method != 'POST':
            return {
                'statusCode': 405,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Method not allowed'})
            }
        
        # Get training mode
        query_params = event.get('queryStringParameters', {}) or {}
        mode = query_params.get('mode', 'incremental')
        
        if mode not in ['incremental', 'full', 'lora']:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Invalid mode. Use "incremental", "full", or "lora"'})
            }
        
        logger.info(f"Training trigger received for mode: {mode}")
        
        # For incremental mode, check if there are new files
        if mode == 'incremental':
            new_files_count = check_for_new_files()
            if new_files_count == 0:
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({
                        'error': 'No new audio files to train on since last model was created.'
                    })
                }
            logger.info(f"Found {new_files_count} new files for incremental training")
        
        # Trigger SageMaker training job
        job_name = trigger_sagemaker_training(mode)
        
        return {
            'statusCode': 202,
            'headers': cors_headers,
            'body': json.dumps({
                'status': 'accepted',
                'message': f'Training started for mode: {mode}',
                'mode': mode,
                'job_name': job_name,
                'note': 'Training is running on SageMaker. Use job_name to check status.'
            })
        }
        
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': f'Training failed: {str(e)}'
            })
        }

def check_for_new_files():
    """Check if there are new audio files since the last model"""
    try:
        # List audio files in S3
        audio_response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='audio/',
            MaxKeys=1000
        )
        
        if 'Contents' not in audio_response:
            return 0
        
        audio_files = [obj for obj in audio_response['Contents'] if obj['Key'].endswith('.wav')]
        
        # Find latest model
        models_response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix='models/',
            MaxKeys=1000
        )
        
        if 'Contents' not in models_response:
            # No models exist, all files are new
            return len(audio_files)
        
        # Find latest model timestamp
        model_timestamps = []
        for obj in models_response['Contents']:
            if 'whisper_finetuned_output_' in obj['Key']:
                try:
                    # Extract timestamp from model name
                    parts = obj['Key'].split('whisper_finetuned_output_')[1].split('/')[0]
                    timestamp = datetime.strptime(parts, "%Y%m%d-%H%M%S")
                    model_timestamps.append(timestamp)
                except:
                    pass
        
        if not model_timestamps:
            return len(audio_files)
        
        latest_model_time = max(model_timestamps)
        
        # Count files newer than the latest model
        new_files = 0
        for audio_file in audio_files:
            try:
                # Extract timestamp from filename
                filename = audio_file['Key'].split('/')[-1]
                file_timestamp_str = filename[:15]  # YYYYMMDD-HHMMSS
                file_timestamp = datetime.strptime(file_timestamp_str, "%Y%m%d-%H%M%S")
                
                if file_timestamp > latest_model_time:
                    new_files += 1
            except:
                # Count unparseable files as new
                new_files += 1
        
        return new_files
        
    except Exception as e:
        logger.error(f"Error checking for new files: {e}")
        raise

def trigger_sagemaker_training(mode):
    """Trigger SageMaker training job"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"wusutra-whisper-{mode}-{timestamp}"
        
        # Configure training with proper entry point
        script_name = 'train_whisper_lora.py' if mode == 'lora' else 'train_whisper.py'
        training_params = {
            'TrainingJobName': job_name,
            'RoleArn': SAGEMAKER_ROLE,
            'AlgorithmSpecification': {
                'TrainingImage': SAGEMAKER_IMAGE_URI,
                'TrainingInputMode': 'File',
                'ContainerEntrypoint': [
                    'python3',
                    f'/opt/ml/input/data/code/{script_name}'
                ]
            },
            'OutputDataConfig': {
                'S3OutputPath': f"s3://{S3_BUCKET}/models/"
            },
            'ResourceConfig': {
                'InstanceType': 'ml.p3.2xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600 * 3  # 3 hours max
            },
            'HyperParameters': {
                'training_type': mode,
                'num_train_epochs': '15' if mode == 'full' else ('8' if mode == 'lora' else '10'),
                'audio_path': '/opt/ml/input/data/training',
                'model_output_path': '/opt/ml/model',
                'validation_path': '/opt/ml/input/data/validation',
                'run_validation': 'True',
                'push_to_hub': 'False',
                'lora_r': '16' if mode == 'lora' else '0',
                'lora_alpha': '32' if mode == 'lora' else '0',
                'lora_dropout': '0.05' if mode == 'lora' else '0.1'
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f"s3://{S3_BUCKET}/audio/",
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                },
                {
                    'ChannelName': 'validation',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f"s3://{S3_BUCKET}/validation_data/",
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                },
                {
                    'ChannelName': 'code',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f"s3://{S3_BUCKET}/code/training_scripts/",
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                }
            ],
            'Environment': {
                'TRAINING_MODE': mode,
                'S3_BUCKET': S3_BUCKET
            }
        }
        
        # Add previous model input for incremental training only
        if mode == 'incremental':
            training_params['InputDataConfig'].append({
                'ChannelName': 'model',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f"s3://{S3_BUCKET}/models/latest/",
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                }
            })
        
        # Start training job
        response = sagemaker.create_training_job(**training_params)
        
        logger.info(f"Started SageMaker training job: {job_name}")
        return job_name
        
    except Exception as e:
        logger.error(f"Error triggering SageMaker training: {e}")
        raise