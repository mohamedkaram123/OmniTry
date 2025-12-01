"""
Script to test the OmniTry API
اسكريبت لاختبار الـ API
"""

import requests
import base64
from PIL import Image
import io
import sys
import os

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_categories():
    """Test categories endpoint"""
    print("🔍 Testing categories endpoint...")
    response = requests.get("http://localhost:8000/categories")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Available categories: {len(data['categories'])}")
    print(f"Categories: {data['categories']}")
    print()

def test_try_on(person_image_path, clothing_image_path, category="top clothes", save_path="result.png"):
    """Test try-on endpoint"""
    print(f"🔍 Testing try-on endpoint...")
    print(f"Person image: {person_image_path}")
    print(f"Clothing image: {clothing_image_path}")
    print(f"Category: {category}")
    
    # Check if files exist
    if not os.path.exists(person_image_path):
        print(f"❌ Error: Person image not found: {person_image_path}")
        return
    
    if not os.path.exists(clothing_image_path):
        print(f"❌ Error: Clothing image not found: {clothing_image_path}")
        return
    
    # Read images
    with open(person_image_path, "rb") as f:
        person_image = f.read()
    
    with open(clothing_image_path, "rb") as f:
        clothing_image = f.read()
    
    # Prepare request
    url = "http://localhost:8000/try-on"
    files = {
        "person_image": ("person.jpg", person_image, "image/jpeg"),
        "clothing_image": ("clothing.jpg", clothing_image, "image/jpeg")
    }
    data = {
        "category": category,
        "steps": 20,
        "guidance_scale": 30,
        "seed": 42  # Fixed seed for reproducibility
    }
    
    print("⏳ Sending request...")
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        # Save result
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Success! Result saved to: {save_path}")
        
        # Show image info
        img = Image.open(io.BytesIO(response.content))
        print(f"Result size: {img.size}")
        print(f"Result format: {img.format}")
    else:
        print(f"❌ Error: {response.status_code}")
        try:
            print(f"Details: {response.json()}")
        except:
            print(f"Response: {response.text}")
    print()

def test_try_on_base64(person_image_path, clothing_image_path, category="top clothes"):
    """Test try-on endpoint with base64 response"""
    print(f"🔍 Testing try-on endpoint with base64...")
    
    # Check if files exist
    if not os.path.exists(person_image_path):
        print(f"❌ Error: Person image not found: {person_image_path}")
        return
    
    if not os.path.exists(clothing_image_path):
        print(f"❌ Error: Clothing image not found: {clothing_image_path}")
        return
    
    # Read images
    with open(person_image_path, "rb") as f:
        person_image = f.read()
    
    with open(clothing_image_path, "rb") as f:
        clothing_image = f.read()
    
    # Prepare request
    url = "http://localhost:8000/try-on"
    files = {
        "person_image": ("person.jpg", person_image, "image/jpeg"),
        "clothing_image": ("clothing.jpg", clothing_image, "image/jpeg")
    }
    data = {
        "category": category,
        "steps": 15,
        "guidance_scale": 30,
        "seed": 42,
        "return_base64": True
    }
    
    print("⏳ Sending request...")
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success!")
        print(f"Seed used: {result['seed']}")
        print(f"Image format: {result['format']}")
        print(f"Base64 length: {len(result['image'])} characters")
        
        # Decode and save
        img_data = base64.b64decode(result['image'])
        img = Image.open(io.BytesIO(img_data))
        save_path = "result_base64.png"
        img.save(save_path)
        print(f"Result saved to: {save_path}")
    else:
        print(f"❌ Error: {response.status_code}")
        try:
            print(f"Details: {response.json()}")
        except:
            print(f"Response: {response.text}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("OmniTry API Test Script")
    print("=" * 60)
    print()
    
    # Test basic endpoints
    test_health()
    test_categories()
    
    # Test try-on if demo images exist
    if len(sys.argv) >= 3:
        person_image = sys.argv[1]
        clothing_image = sys.argv[2]
        category = sys.argv[3] if len(sys.argv) > 3 else "top clothes"
        
        test_try_on(person_image, clothing_image, category)
        test_try_on_base64(person_image, clothing_image, category)
    else:
        print("ℹ️  To test try-on, run:")
        print("python test_api.py <person_image> <clothing_image> [category]")
        print()
        print("Example:")
        print("python test_api.py demo_example/person_top_cloth.jpg demo_example/object_top_cloth.jpg 'top clothes'")
        print()
        
        # Try with demo images if they exist
        demo_person = "demo_example/person_top_cloth.jpg"
        demo_clothing = "demo_example/object_top_cloth.jpg"
        
        if os.path.exists(demo_person) and os.path.exists(demo_clothing):
            print("📁 Found demo images, testing...")
            print()
            test_try_on(demo_person, demo_clothing, "top clothes")
        else:
            print("⚠️  Demo images not found. Please download them first or provide your own images.")
