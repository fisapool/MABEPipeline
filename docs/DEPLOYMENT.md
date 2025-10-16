# MABE Pipeline Deployment Guide

## Overview

This guide covers deploying the MABE Pipeline in various environments, from local development to production systems.

## Local Development

### Prerequisites

- Python 3.8+
- Git
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd MABEPipeline
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python bin/run_pipeline.py --help
   ```

### Configuration

1. **Update configuration:**
   ```bash
   cp configs/default.yaml configs/local.yaml
   # Edit configs/local.yaml with your paths
   ```

2. **Set environment variables (optional):**
   ```bash
   export MABE_DATASET__PATH="/path/to/your/data"
   export MABE_DEVICE__USE_CUDA=true
   ```

### Running the Pipeline

```bash
# Full pipeline
python bin/run_pipeline.py all --config configs/local.yaml

# Individual stages
python bin/run_pipeline.py preprocess --config configs/local.yaml
python bin/run_pipeline.py train --config configs/local.yaml
python bin/run_pipeline.py infer --config configs/local.yaml
```

## Docker Deployment

### Building the Image

```bash
# Build the Docker image
docker build -t mabe-pipeline .

# Tag for registry
docker tag mabe-pipeline your-registry/mabe-pipeline:latest
```

### Running with Docker

```bash
# Basic usage
docker run -v /path/to/data:/app/data mabe-pipeline

# With environment variables
docker run -e COMMAND=train -e EPOCHS=10 -e MAX_VIDEOS=5 mabe-pipeline

# With custom configuration
docker run -v /path/to/config:/app/configs mabe-pipeline
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mabe-pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
      - ./outputs:/app/outputs
    environment:
      - CONFIG_FILE=configs/production.yaml
      - COMMAND=all
      - MAX_VIDEOS=10
      - EPOCHS=50
    command: ["./docker/entrypoint.sh"]
```

Run with:
```bash
docker-compose up
```

## Cloud Deployment

### AWS EC2

1. **Launch EC2 instance:**
   - AMI: Ubuntu 20.04 LTS
   - Instance type: g4dn.xlarge (for GPU) or t3.large (for CPU)
   - Storage: 50GB+ EBS volume

2. **Install dependencies:**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python 3.9
   sudo apt install python3.9 python3.9-pip python3.9-venv -y
   
   # Install CUDA (for GPU instances)
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo apt-key add /var/cuda-repository-ubuntu2004-11-8-local/7fa2af80.pub
   sudo apt update
   sudo apt install cuda -y
   ```

3. **Deploy application:**
   ```bash
   # Clone repository
   git clone <repository-url>
   cd MABEPipeline
   
   # Create virtual environment
   python3.9 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure application
   cp configs/default.yaml configs/production.yaml
   # Edit configs/production.yaml
   ```

4. **Run as service:**
   ```bash
   # Create systemd service
   sudo nano /etc/systemd/system/mabe-pipeline.service
   ```

   Service file content:
   ```ini
   [Unit]
   Description=MABE Pipeline Service
   After=network.target
   
   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/MABEPipeline
   Environment=PATH=/home/ubuntu/MABEPipeline/venv/bin
   ExecStart=/home/ubuntu/MABEPipeline/venv/bin/python bin/run_pipeline.py all --config configs/production.yaml
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start:
   ```bash
   sudo systemctl enable mabe-pipeline
   sudo systemctl start mabe-pipeline
   sudo systemctl status mabe-pipeline
   ```

### Google Cloud Platform

1. **Create Compute Engine instance:**
   ```bash
   gcloud compute instances create mabe-pipeline \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=50GB
   ```

2. **Install dependencies:**
   ```bash
   # Install NVIDIA drivers
   curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo apt-key add /var/cuda-repository-ubuntu2004-11-8-local/7fa2af80.pub
   sudo apt update
   sudo apt install cuda -y
   ```

3. **Deploy application:**
   ```bash
   # Clone and setup
   git clone <repository-url>
   cd MABEPipeline
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Azure

1. **Create Virtual Machine:**
   ```bash
   az vm create \
     --resource-group myResourceGroup \
     --name mabe-pipeline \
     --image Ubuntu2004 \
     --size Standard_NC6s_v3 \
     --admin-username azureuser \
     --generate-ssh-keys
   ```

2. **Install dependencies:**
   ```bash
   # Install NVIDIA drivers
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo apt-key add /var/cuda-repository-ubuntu2004-11-8-local/7fa2af80.pub
   sudo apt update
   sudo apt install cuda -y
   ```

## Kubernetes Deployment

### Create Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mabe-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mabe-pipeline
  template:
    metadata:
      labels:
        app: mabe-pipeline
    spec:
      containers:
      - name: mabe-pipeline
        image: your-registry/mabe-pipeline:latest
        env:
        - name: CONFIG_FILE
          value: "configs/production.yaml"
        - name: COMMAND
          value: "all"
        - name: MAX_VIDEOS
          value: "10"
        - name: EPOCHS
          value: "50"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: config-volume
          mountPath: /app/configs
        - name: outputs-volume
          mountPath: /app/outputs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      - name: config-volume
        configMap:
          name: mabe-config
      - name: outputs-volume
        persistentVolumeClaim:
          claimName: outputs-pvc
```

### Create Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mabe-pipeline-service
spec:
  selector:
    app: mabe-pipeline
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

### Deploy

```bash
# Apply configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get pods
kubectl get services
```

## CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t mabe-pipeline .
        docker tag mabe-pipeline your-registry/mabe-pipeline:${{ github.sha }}
    
    - name: Push to registry
      run: |
        docker push your-registry/mabe-pipeline:${{ github.sha }}
        docker push your-registry/mabe-pipeline:latest
    
    - name: Deploy to production
      run: |
        # Deploy to your production environment
        kubectl set image deployment/mabe-pipeline mabe-pipeline=your-registry/mabe-pipeline:${{ github.sha }}
```

### GitLab CI

Create `.gitlab-ci.yml`:

```yaml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - docker build -t mabe-pipeline .
    - docker tag mabe-pipeline $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

test:
  stage: test
  script:
    - docker run --rm mabe-pipeline ./scripts/ci_run_short.sh

deploy:
  stage: deploy
  script:
    - kubectl set image deployment/mabe-pipeline mabe-pipeline=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
```

## Monitoring and Logging

### Prometheus Metrics

Add metrics collection:

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
training_epochs = Counter('mabe_training_epochs_total', 'Total training epochs')
inference_requests = Counter('mabe_inference_requests_total', 'Total inference requests')
model_accuracy = Gauge('mabe_model_accuracy', 'Model accuracy')
training_duration = Histogram('mabe_training_duration_seconds', 'Training duration')
```

### ELK Stack

Configure log shipping:

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /app/outputs/logs/*.log
  fields:
    service: mabe-pipeline
  fields_under_root: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

### Grafana Dashboard

Create dashboard with:
- Training metrics
- Inference performance
- System resources
- Error rates

## Security Considerations

### Secrets Management

Use Kubernetes secrets:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mabe-secrets
type: Opaque
data:
  api-key: <base64-encoded-key>
  database-url: <base64-encoded-url>
```

### Network Security

- Use TLS for all communications
- Implement network policies
- Use service mesh (Istio) for traffic management

### Data Protection

- Encrypt data at rest
- Use secure data transmission
- Implement access controls
- Regular security audits

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Data loading errors:**
   - Check file paths
   - Verify data format
   - Check permissions

3. **Model convergence issues:**
   - Adjust learning rate
   - Use different optimizers
   - Check data quality

### Debugging

Enable debug logging:

```bash
export MABE_LOGGING__LEVEL=DEBUG
python bin/run_pipeline.py train --config configs/default.yaml --verbose
```

### Performance Optimization

1. **Data loading:**
   - Use multiple workers
   - Enable data caching
   - Use faster storage (SSD)

2. **Training:**
   - Use mixed precision
   - Enable gradient checkpointing
   - Optimize batch size

3. **Inference:**
   - Use model quantization
   - Enable TensorRT optimization
   - Use batch inference

## Backup and Recovery

### Data Backup

```bash
# Backup data
rsync -av /path/to/data/ /backup/data/
rsync -av /path/to/outputs/ /backup/outputs/
```

### Model Backup

```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz outputs/models/
```

### Recovery

```bash
# Restore data
rsync -av /backup/data/ /path/to/data/
rsync -av /backup/outputs/ /path/to/outputs/
```
