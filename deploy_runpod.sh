#!/bin/bash
# Script to deploy OmniTry to RunPod
# اسكريبت لنشر OmniTry على RunPod

set -e

echo "🚀 OmniTry RunPod Deployment Script"
echo "===================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Get Docker Hub username
read -p "Enter your Docker Hub username: " DOCKER_USERNAME
if [ -z "$DOCKER_USERNAME" ]; then
    echo "❌ Error: Docker Hub username is required"
    exit 1
fi

# Image name
IMAGE_NAME="omnitry-runpod"
FULL_IMAGE_NAME="$DOCKER_USERNAME/$IMAGE_NAME:latest"

echo ""
echo "📦 Building Docker image..."
echo "Image: $FULL_IMAGE_NAME"
echo ""

# Build image
docker build -t $FULL_IMAGE_NAME -f Dockerfile .

echo ""
echo "✅ Build completed!"
echo ""

# Ask if user wants to push
read -p "Do you want to push to Docker Hub? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📤 Pushing to Docker Hub..."
    echo ""
    
    # Login to Docker Hub
    echo "Please login to Docker Hub:"
    docker login
    
    # Push image
    docker push $FULL_IMAGE_NAME
    
    echo ""
    echo "✅ Image pushed successfully!"
    echo ""
    
    # Print RunPod instructions
    echo "=" * 60
    echo "📝 Next Steps for RunPod Deployment:"
    echo "=" * 60
    echo ""
    echo "1. Go to RunPod Console: https://www.runpod.io/console/serverless"
    echo ""
    echo "2. Create a new Endpoint:"
    echo "   - Click 'New Endpoint'"
    echo "   - Name: omnitry-tryon"
    echo "   - Select GPU: RTX 4090 or A100 (40GB+ VRAM)"
    echo "   - Docker Image: $FULL_IMAGE_NAME"
    echo "   - Container Disk: 20 GB"
    echo "   - Environment Variables (optional):"
    echo "     - HUGGINGFACE_TOKEN=your_token (if models are private)"
    echo ""
    echo "3. Create a Network Volume for checkpoints:"
    echo "   - Go to Storage > Network Volumes"
    echo "   - Create new volume (50GB recommended)"
    echo "   - Mount to: /app/checkpoints"
    echo ""
    echo "4. Upload model checkpoints to the volume:"
    echo "   - FLUX.1-Fill-dev model"
    echo "   - omnitry_v1_unified.safetensors"
    echo ""
    echo "5. Test your endpoint using the API:"
    echo "   python example_client.py"
    echo ""
    echo "6. Your endpoint URL will be:"
    echo "   https://api.runpod.ai/v2/YOUR_ENDPOINT_ID"
    echo ""
    echo "=" * 60
    echo ""
    
    # Save instructions to file
    cat > runpod_deployment_info.txt << EOF
OmniTry RunPod Deployment Information
====================================

Docker Image: $FULL_IMAGE_NAME
Date: $(date)

RunPod Configuration:
- GPU: RTX 4090 or A100 (40GB+ VRAM recommended)
- Container Disk: 20 GB
- Network Volume: 50 GB (for checkpoints)
- Mount Point: /app/checkpoints

Required Files in Network Volume:
1. checkpoints/FLUX.1-Fill-dev/ (full model)
2. checkpoints/omnitry_v1_unified.safetensors

Endpoint Settings:
- Max Workers: 1-3 (depending on usage)
- Idle Timeout: 60 seconds
- Execution Timeout: 300 seconds

Testing Your Endpoint:
1. Get your endpoint ID and API key from RunPod
2. Use example_client.py:
   
   from example_client import RunPodClient
   
   client = RunPodClient(
       endpoint_id="YOUR_ENDPOINT_ID",
       api_key="YOUR_API_KEY"
   )
   
   result = client.try_on(
       person_image_path="person.jpg",
       clothing_image_path="clothing.jpg",
       category="top clothes"
   )
   result.save("result.png")

Cost Estimation (approximate):
- A100 (40GB): ~$2.89/hour (on-demand)
- RTX 4090: ~$0.50/hour (on-demand)
- Network Storage: ~$0.10/GB/month

Tips:
- Use serverless for sporadic usage
- Use dedicated pods for constant usage
- Enable auto-scaling for variable load
- Monitor your usage in RunPod dashboard

Support:
- RunPod Docs: https://docs.runpod.io
- OmniTry Issues: https://github.com/your-repo/issues
EOF
    
    echo "💾 Deployment info saved to: runpod_deployment_info.txt"
    echo ""
else
    echo ""
    echo "⏭️  Skipping push to Docker Hub"
    echo ""
    echo "To push manually later:"
    echo "  docker login"
    echo "  docker push $FULL_IMAGE_NAME"
    echo ""
fi

echo "✨ Done!"
