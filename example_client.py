"""
Example client for OmniTry API
مثال على استخدام الـ API
"""

import requests
import base64
from PIL import Image
import io
import json

class OmniTryClient:
    """Client for OmniTry Virtual Try-On API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_categories(self):
        """Get available categories"""
        response = requests.get(f"{self.base_url}/categories")
        return response.json()
    
    def try_on(
        self,
        person_image_path,
        clothing_image_path,
        category,
        steps=20,
        guidance_scale=30.0,
        seed=-1,
        return_base64=False
    ):
        """
        Perform virtual try-on
        
        Args:
            person_image_path: Path to person image
            clothing_image_path: Path to clothing/accessory image
            category: Category of the item (e.g., 'top clothes', 'dress')
            steps: Number of inference steps (1-50)
            guidance_scale: Guidance scale (1-50)
            seed: Random seed (-1 for random)
            return_base64: Return image as base64 string
        
        Returns:
            PIL Image if return_base64=False, else dict with base64 image
        """
        # Read images
        with open(person_image_path, "rb") as f:
            person_image = f.read()
        
        with open(clothing_image_path, "rb") as f:
            clothing_image = f.read()
        
        # Prepare request
        files = {
            "person_image": ("person.jpg", person_image, "image/jpeg"),
            "clothing_image": ("clothing.jpg", clothing_image, "image/jpeg")
        }
        data = {
            "category": category,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "return_base64": return_base64
        }
        
        # Send request
        response = requests.post(f"{self.base_url}/try-on", files=files, data=data)
        response.raise_for_status()
        
        if return_base64:
            return response.json()
        else:
            return Image.open(io.BytesIO(response.content))
    
    def try_on_from_base64(
        self,
        person_image_b64,
        clothing_image_b64,
        category,
        steps=20,
        guidance_scale=30.0,
        seed=-1
    ):
        """
        Perform virtual try-on using base64 encoded images
        Useful for RunPod serverless deployment
        
        Args:
            person_image_b64: Base64 encoded person image
            clothing_image_b64: Base64 encoded clothing image
            category: Category of the item
            steps: Number of inference steps
            guidance_scale: Guidance scale
            seed: Random seed
        
        Returns:
            PIL Image
        """
        # Decode base64 to bytes
        person_bytes = base64.b64decode(person_image_b64)
        clothing_bytes = base64.b64decode(clothing_image_b64)
        
        # Prepare request
        files = {
            "person_image": ("person.jpg", person_bytes, "image/jpeg"),
            "clothing_image": ("clothing.jpg", clothing_bytes, "image/jpeg")
        }
        data = {
            "category": category,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "return_base64": True
        }
        
        # Send request
        response = requests.post(f"{self.base_url}/try-on", files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
        img_data = base64.b64decode(result["image"])
        return Image.open(io.BytesIO(img_data))


class RunPodClient:
    """Client for RunPod serverless deployment"""
    
    def __init__(self, endpoint_id, api_key):
        """
        Initialize RunPod client
        
        Args:
            endpoint_id: RunPod endpoint ID
            api_key: RunPod API key
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
    
    def try_on(
        self,
        person_image_path,
        clothing_image_path,
        category,
        steps=20,
        guidance_scale=30.0,
        seed=-1
    ):
        """
        Perform virtual try-on via RunPod
        
        Args:
            person_image_path: Path to person image
            clothing_image_path: Path to clothing/accessory image
            category: Category of the item
            steps: Number of inference steps
            guidance_scale: Guidance scale
            seed: Random seed
        
        Returns:
            PIL Image
        """
        # Read and encode images
        with open(person_image_path, "rb") as f:
            person_b64 = base64.b64encode(f.read()).decode()
        
        with open(clothing_image_path, "rb") as f:
            clothing_b64 = base64.b64encode(f.read()).decode()
        
        # Prepare payload
        payload = {
            "input": {
                "person_image": person_b64,
                "clothing_image": clothing_b64,
                "category": category,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed
            }
        }
        
        # Send request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Run request
        response = requests.post(
            f"{self.base_url}/run",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        # Get job ID
        job_id = response.json()["id"]
        
        # Poll for result
        while True:
            import time
            time.sleep(2)
            
            status_response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=headers
            )
            status_data = status_response.json()
            
            if status_data["status"] == "COMPLETED":
                result = status_data["output"]
                if "error" in result:
                    raise Exception(f"Error: {result['error']}")
                
                img_data = base64.b64decode(result["image"])
                return Image.open(io.BytesIO(img_data))
            
            elif status_data["status"] == "FAILED":
                raise Exception(f"Job failed: {status_data.get('error', 'Unknown error')}")


def example_usage():
    """Example usage of the clients"""
    
    # Example 1: Local API
    print("=" * 60)
    print("Example 1: Using Local API")
    print("=" * 60)
    
    client = OmniTryClient("http://localhost:8000")
    
    # Check health
    print("Checking health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print()
    
    # Get categories
    print("Getting categories...")
    categories = client.get_categories()
    print(f"Available categories: {categories['categories']}")
    print()
    
    # Try on (if demo images exist)
    try:
        print("Performing virtual try-on...")
        result = client.try_on(
            person_image_path="demo_example/person_top_cloth.jpg",
            clothing_image_path="demo_example/object_top_cloth.jpg",
            category="top clothes",
            steps=20,
            guidance_scale=30,
            seed=42
        )
        result.save("example_result.png")
        print("✅ Result saved to example_result.png")
    except FileNotFoundError:
        print("⚠️  Demo images not found, skipping try-on example")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()
    
    # Example 2: RunPod (commented out - requires credentials)
    print("=" * 60)
    print("Example 2: Using RunPod (commented out)")
    print("=" * 60)
    print("To use RunPod:")
    print("1. Deploy your endpoint on RunPod")
    print("2. Get your endpoint ID and API key")
    print("3. Uncomment and run the code below")
    print()
    print("Example code:")
    print("""
    runpod_client = RunPodClient(
        endpoint_id="YOUR_ENDPOINT_ID",
        api_key="YOUR_API_KEY"
    )
    
    result = runpod_client.try_on(
        person_image_path="person.jpg",
        clothing_image_path="clothing.jpg",
        category="top clothes"
    )
    result.save("runpod_result.png")
    """)


if __name__ == "__main__":
    example_usage()
