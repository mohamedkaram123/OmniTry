# 🚀 Quick Start Guide - دليل البدء السريع

## تجهيز الموديل (مطلوب مرة واحدة فقط)

### 1. تنزيل Checkpoints

```bash
# إنشاء مجلد الموديلات
mkdir -p checkpoints
cd checkpoints

# تنزيل FLUX.1-Fill-dev
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev

# تنزيل OmniTry LoRA
wget https://huggingface.co/Kunbyte/OmniTry/resolve/main/omnitry_v1_unified.safetensors

cd ..
```

**البدائل:**
- استخدم Hugging Face Hub:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("black-forest-labs/FLUX.1-Fill-dev", local_dir="checkpoints/FLUX.1-Fill-dev")
  ```

### 2. تثبيت المتطلبات

```bash
# المتطلبات الأساسية
pip install -r requirements.txt

# متطلبات الـ API
pip install -r requirements_api.txt

# (اختياري) Flash Attention للأداء الأفضل
pip install flash-attn==2.6.3
```

## 🎯 طرق التشغيل

### الطريقة 1: تشغيل سريع (FastAPI)

```bash
# باستخدام السكريبت
./start_api.sh

# أو مباشرة
python api_server.py
```

افتح المتصفح على: http://localhost:8000/docs

### الطريقة 2: Docker (موصى به للإنتاج)

```bash
# بناء وتشغيل
docker-compose up -d

# أو يدوياً
docker build -t omnitry-api .
docker run -p 8000:8000 --gpus all -v $(pwd)/checkpoints:/app/checkpoints omnitry-api
```

### الطريقة 3: RunPod Deployment

```bash
# تشغيل سكريبت النشر
./deploy_runpod.sh

# ثم اتبع التعليمات التي ستظهر
```

## 🧪 اختبار الـ API

### اختبار سريع

```bash
# اختبار أساسي
python test_api.py

# اختبار مع صور
python test_api.py demo_example/person_top_cloth.jpg demo_example/object_top_cloth.jpg "top clothes"
```

### استخدام Python

```python
from example_client import OmniTryClient

# إنشاء client
client = OmniTryClient("http://localhost:8000")

# تجربة virtual try-on
result = client.try_on(
    person_image_path="person.jpg",
    clothing_image_path="shirt.jpg",
    category="top clothes"
)

# حفظ النتيجة
result.save("result.png")
```

### استخدام cURL

```bash
curl -X POST "http://localhost:8000/try-on" \
  -F "person_image=@person.jpg" \
  -F "clothing_image=@shirt.jpg" \
  -F "category=top clothes" \
  --output result.png
```

## 📋 الفئات المتاحة

| Category | Description | Example |
|----------|-------------|---------|
| `top clothes` | قمصان، بلوزات | T-shirts, blouses |
| `bottom clothes` | بناطيل، تنانير | Pants, skirts |
| `dress` | فساتين | Dresses |
| `shoe` | أحذية | Shoes |
| `earrings` | أقراط | Earrings |
| `bracelet` | أساور | Bracelets |
| `necklace` | قلائد | Necklaces |
| `ring` | خواتم | Rings |
| `sunglasses` | نظارات شمسية | Sunglasses |
| `glasses` | نظارات طبية | Glasses |
| `belt` | أحزمة | Belts |
| `bag` | حقائب | Bags |
| `hat` | قبعات | Hats |
| `tie` | كرافتة | Ties |
| `bow tie` | فيونكة | Bow ties |

## ⚙️ معلمات التحكم

- **steps** (10-50): عدد خطوات التوليد
  - أقل = أسرع لكن جودة أقل
  - أكثر = أبطأ لكن جودة أعلى
  - الموصى به: 20

- **guidance_scale** (1-50): قوة التوجيه
  - أقل = أكثر إبداعاً
  - أكثر = التزام أكبر بالصورة الأصلية
  - الموصى به: 30

- **seed** (integer أو -1): للحصول على نتائج متشابهة
  - -1 = عشوائي
  - أي رقم = نتائج ثابتة

## 🔧 استكشاف الأخطاء الشائعة

### مشكلة: CUDA out of memory

**الحل:**
```python
# في api_server.py أو gradio_demo.py
pipeline.enable_model_cpu_offload()  # ✓ Already enabled
pipeline.vae.enable_tiling()  # ✓ Already enabled
```

### مشكلة: بطء في التوليد

**الحلول:**
1. ثبت flash-attention
2. استخدم GPU أقوى (RTX 4090, A100)
3. قلل steps إلى 15-18

### مشكلة: Port already in use

```bash
# غير البورت في api_server.py
uvicorn.run(app, host="0.0.0.0", port=8001)  # Instead of 8000
```

### مشكلة: Checkpoints not found

```bash
# تحقق من المسارات
ls -la checkpoints/FLUX.1-Fill-dev
ls -la checkpoints/omnitry_v1_unified.safetensors
```

## 📊 متطلبات النظام

### الحد الأدنى (للتجربة)
- GPU: RTX 3090 (24GB VRAM)
- RAM: 16GB
- Storage: 50GB

### الموصى به (للإنتاج)
- GPU: RTX 4090 (24GB) أو A100 (40GB)
- RAM: 32GB+
- Storage: 100GB SSD

### RunPod المقترح
- GPU: RTX 4090 أو A100 40GB
- Container Disk: 20GB
- Network Volume: 50GB

## 🚀 النشر على RunPod

### خطوات سريعة:

1. **بناء ورفع Docker Image:**
```bash
./deploy_runpod.sh
```

2. **إنشاء Endpoint على RunPod:**
   - اذهب إلى https://www.runpod.io/console/serverless
   - New Endpoint → اختر GPU → أدخل Docker image

3. **تحميل الموديلات:**
   - أنشئ Network Volume
   - حمّل checkpoints للـ volume
   - اربط الـ volume بالـ endpoint

4. **اختبر Endpoint:**
```python
from example_client import RunPodClient

client = RunPodClient(
    endpoint_id="your-endpoint-id",
    api_key="your-api-key"
)

result = client.try_on("person.jpg", "shirt.jpg", "top clothes")
result.save("result.png")
```

## 💡 نصائح للاستخدام

### للحصول على أفضل النتائج:
1. ✅ استخدم صور عالية الجودة
2. ✅ إضاءة جيدة ومتساوية
3. ✅ خلفية بسيطة أو محايدة
4. ✅ صور أمامية مباشرة
5. ✅ قطع ملابس واضحة ومفصولة

### تجنب:
1. ❌ صور مظلمة أو ضبابية
2. ❌ خلفيات معقدة جداً
3. ❌ زوايا غريبة أو ملتوية
4. ❌ صور صغيرة جداً (<512px)

## 📞 الدعم والموارد

- 📖 [API Documentation الكامل](API_README.md)
- 🔧 [Test Script](test_api.py)
- 💻 [Example Client](example_client.py)
- 🐳 [Docker Deployment](docker-compose.yml)
- 🚀 [RunPod Deployment](deploy_runpod.sh)

## 📈 Next Steps

1. جرب الـ API محلياً
2. اختبر مع صورك الخاصة
3. عدّل المعاملات للحصول على أفضل نتيجة
4. انشر على RunPod للاستخدام في الإنتاج

---

**أي أسئلة؟** افتح issue على GitHub أو راجع [API_README.md](API_README.md) للتفاصيل الكاملة.
