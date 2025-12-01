# OmniTry API Documentation

API للـ Virtual Try-On باستخدام OmniTry - جرّب أي ملابس أو إكسسوارات على أي شخص!

## 📋 المتطلبات

- **VRAM**: على الأقل 28GB للاستخدام العادي
- **CUDA**: مطلوب GPU مع دعم CUDA
- **Python**: 3.10 أو أعلى

## 🚀 طرق التشغيل

### 1. التشغيل المباشر (FastAPI)

```bash
# تثبيت المتطلبات
pip install -r requirements.txt
pip install -r requirements_api.txt

# تنزيل الموديلات (مطلوب مرة واحدة فقط)
mkdir checkpoints
# تنزيل FLUX.1-Fill-dev من HuggingFace
# تنزيل omnitry_v1_unified.safetensors من HuggingFace

# تشغيل API
python api_server.py
```

سيعمل السيرفر على `http://localhost:8000`

### 2. التشغيل باستخدام Docker

```bash
# بناء الصورة
docker build -t omnitry-api .

# تشغيل الكونتينر
docker run -p 8000:8000 --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  omnitry-api
```

### 3. التشغيل باستخدام Docker Compose

```bash
docker-compose up -d
```

### 4. النشر على RunPod

#### الطريقة الأولى: RunPod Serverless (موصى بها)

1. **تحضير الصورة:**
```bash
docker build -t your-username/omnitry-runpod:latest .
docker push your-username/omnitry-runpod:latest
```

2. **إنشاء Endpoint على RunPod:**
   - اذهب إلى [RunPod Serverless](https://www.runpod.io/console/serverless)
   - اضغط "New Endpoint"
   - اختر GPU (مثلاً RTX 4090 أو A100)
   - ضع صورة Docker: `your-username/omnitry-runpod:latest`
   - ضبط الإعدادات:
     - Container Disk: 20GB
     - Memory: 32GB
     - GPU: 1x A100 (40GB) أو أفضل

3. **رفع الموديلات:**
   - استخدم RunPod Network Volume أو
   - ضمّن الموديلات في الصورة (سيزيد الحجم)

#### الطريقة الثانية: RunPod Pod (GPU Instance)

1. اذهب إلى RunPod Console
2. أنشئ Pod جديد مع GPU (RTX 4090 أو A100)
3. اختر Template: PyTorch
4. افتح الـ Terminal وقم بـ:
```bash
git clone https://github.com/your-repo/OmniTry.git
cd OmniTry
pip install -r requirements.txt -r requirements_api.txt
python api_server.py
```

## 📡 Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### 2. Get Categories
```bash
GET /categories
```

**Response:**
```json
{
  "categories": [
    "top clothes",
    "bottom clothes",
    "dress",
    "shoe",
    "earrings",
    ...
  ],
  "descriptions": {
    "top clothes": "replacing the top cloth",
    ...
  }
}
```

### 3. Virtual Try-On
```bash
POST /try-on
```

**Parameters:**
- `person_image` (file, required): صورة الشخص
- `clothing_image` (file, required): صورة الملابس/الإكسسوار
- `category` (string, required): نوع القطعة (مثل "top clothes", "dress", "shoe")
- `steps` (int, optional): عدد خطوات التوليد (1-50، افتراضي: 20)
- `guidance_scale` (float, optional): قوة التوجيه (1-50، افتراضي: 30)
- `seed` (int, optional): للحصول على نتائج ثابتة (-1 للعشوائية)
- `return_base64` (bool, optional): إرجاع الصورة كـ base64 (افتراضي: false)

## 🧪 أمثلة الاستخدام

### Python (requests)

```python
import requests
import base64
from PIL import Image
import io

# قراءة الصور
with open("person.jpg", "rb") as f:
    person_image = f.read()

with open("shirt.jpg", "rb") as f:
    clothing_image = f.read()

# إرسال الطلب
url = "http://localhost:8000/try-on"
files = {
    "person_image": ("person.jpg", person_image, "image/jpeg"),
    "clothing_image": ("shirt.jpg", clothing_image, "image/jpeg")
}
data = {
    "category": "top clothes",
    "steps": 20,
    "guidance_scale": 30,
    "seed": -1
}

response = requests.post(url, files=files, data=data)

# حفظ النتيجة
if response.status_code == 200:
    with open("result.png", "wb") as f:
        f.write(response.content)
    print("✅ تم التوليد بنجاح!")
else:
    print(f"❌ خطأ: {response.json()}")
```

### Python (مع base64)

```python
import requests
import base64
from PIL import Image
import io

# تحويل الصور لـ base64
with open("person.jpg", "rb") as f:
    person_b64 = base64.b64encode(f.read()).decode()

with open("shirt.jpg", "rb") as f:
    clothing_b64 = base64.b64encode(f.read()).decode()

# استخدام RunPod Handler
payload = {
    "input": {
        "person_image": person_b64,
        "clothing_image": clothing_b64,
        "category": "top clothes",
        "steps": 20,
        "guidance_scale": 30,
        "seed": -1
    }
}

# للـ FastAPI
url = "http://localhost:8000/try-on"
files = {
    "person_image": ("person.jpg", base64.b64decode(person_b64), "image/jpeg"),
    "clothing_image": ("shirt.jpg", base64.b64decode(clothing_b64), "image/jpeg")
}
data = {
    "category": "top clothes",
    "return_base64": True
}

response = requests.post(url, files=files, data=data)
result = response.json()

# حفظ الصورة
img_data = base64.b64decode(result["image"])
img = Image.open(io.BytesIO(img_data))
img.save("result.png")
```

### cURL

```bash
curl -X POST "http://localhost:8000/try-on" \
  -F "person_image=@person.jpg" \
  -F "clothing_image=@shirt.jpg" \
  -F "category=top clothes" \
  -F "steps=20" \
  -F "guidance_scale=30" \
  -F "seed=-1" \
  --output result.png
```

### JavaScript/TypeScript

```javascript
const formData = new FormData();
formData.append('person_image', personImageFile);
formData.append('clothing_image', clothingImageFile);
formData.append('category', 'top clothes');
formData.append('steps', '20');
formData.append('guidance_scale', '30');

const response = await fetch('http://localhost:8000/try-on', {
  method: 'POST',
  body: formData
});

const blob = await response.blob();
const imageUrl = URL.createObjectURL(blob);
// استخدم imageUrl في img src
```

## 📝 الفئات المتاحة (Categories)

- `top clothes` - قمصان، بلوزات، تيشيرتات
- `bottom clothes` - بناطيل، تنانير
- `dress` - فساتين
- `shoe` - أحذية
- `earrings` - أقراط
- `bracelet` - أساور
- `necklace` - قلائد
- `ring` - خواتم
- `sunglasses` - نظارات شمسية
- `glasses` - نظارات طبية
- `belt` - أحزمة
- `bag` - حقائب
- `hat` - قبعات
- `tie` - كرافتة
- `bow tie` - فيونكة

## 🔧 استكشاف الأخطاء

### مشكلة: Out of Memory (CUDA OOM)

**الحل:**
- استخدم GPU بـ VRAM أكبر (28GB على الأقل)
- قلل حجم الصور قبل الإرسال
- تأكد من تفعيل `enable_model_cpu_offload()` في الكود

### مشكلة: البطء في التوليد

**الحل:**
- ثبت flash-attention: `pip install flash-attn==2.6.3`
- قلل عدد الـ steps (مثلاً 15 بدلاً من 20)
- استخدم GPU أسرع

### مشكلة: Checkpoints not found

**الحل:**
```bash
mkdir -p checkpoints
cd checkpoints

# تنزيل FLUX.1-Fill-dev
git clone https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev

# تنزيل OmniTry LoRA
wget https://huggingface.co/Kunbyte/OmniTry/resolve/main/omnitry_v1_unified.safetensors
```

## 💡 نصائح للاستخدام الأمثل

1. **جودة الصور**: استخدم صور عالية الجودة وواضحة
2. **الإضاءة**: تأكد من إضاءة جيدة في الصور
3. **الخلفية**: خلفيات بسيطة تعطي نتائج أفضل
4. **الزاوية**: صور أمامية مباشرة أفضل للملابس
5. **الدقة**: الصور ستُعدّل تلقائياً لـ 1024x1024 كحد أقصى

## 📊 الأداء

- **وقت التوليد**: 10-30 ثانية (حسب GPU و steps)
- **VRAM المستخدم**: 20-28GB
- **الدقة القصوى**: 1024x1024 بكسل

## 🔐 الأمان

- API غير محمي بـ authentication افتراضياً
- للإنتاج، أضف API keys أو OAuth
- استخدم HTTPS في Production
- ضع Rate limiting للطلبات

## 📞 الدعم

للمزيد من المعلومات:
- [OmniTry Paper](http://arxiv.org/abs/2508.13632)
- [HuggingFace Model](https://huggingface.co/Kunbyte/OmniTry)
- [GitHub Issues](https://github.com/your-repo/issues)

## 📄 الترخيص

نفس ترخيص OmniTry الأصلي.
