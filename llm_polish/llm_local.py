import time
import requests
from typing import Optional
import os
import io
import base64
import mimetypes
from PIL import Image

#安装依赖
#python -m pip install vllm fastapi uvicorn pydantic torch transformers requests

#启动vllm
# CUDA_VISIBLE_DEVICES=0,2 NCCL_SHM_DISABLE=1 \
# python -m vllm.entrypoints.openai.api_server \
#   --model /mnt/mdocr_external_file/Qwen2.5-14B-Instruct \
#   --served-model-name qwen14b \
#   --tensor-parallel-size 2 \
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.9 \
#   --disable-uvicorn-access-log \
#   --host 0.0.0.0 \
#   --port 8000



def local_api_text(
    sys_prompt: str,
    prompt: str,
    input_text: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    prompt_enhancer: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    url: str = "http://127.0.0.1:8000/v1/chat/completions",
    model: str = "qwen14b",
) -> str:
    user_prompt = prompt_enhancer.format(prompt=prompt) if prompt_enhancer else prompt

    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": f"{user_prompt}\n\n{input_text}"
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise last_err

# ====== Vision helpers (copied/adapted from llm_api.py style) ======
DEFAULT_IMAGE_LIMIT_MB = 6


def _encode_data_uri(img_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _file_to_data_uri_with_limit(
    image_path: str,
    limit_bytes: int,
    prefer_format: str = "JPEG",
    min_side: int = 384,
    start_quality: int = 90,
    min_quality: int = 40,
) -> str:
    """
    将本地图片转成 data URI，尽可能在限制内保留最大可用分辨率
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        raw = f.read()

    if len(raw) <= limit_bytes:
        return _encode_data_uri(raw, mime_type)

    with Image.open(image_path) as im0:
        w0, h0 = im0.size

    with Image.open(image_path) as im:
        # 统一转 RGB，去掉透明通道
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        else:
            im = im.convert("RGB")

        low, high = 0.1, 1.0
        best = None

        def try_encode(scale: float):
            w = max(min_side, int(w0 * scale))
            h = max(min_side, int(h0 * scale))
            resized = im.resize((w, h), Image.LANCZOS) if (w, h) != (w0, h0) else im

            for q in range(start_quality, min_quality - 1, -5):
                buf = io.BytesIO()
                resized.save(buf, format=prefer_format, quality=q, optimize=True)
                data = buf.getvalue()
                if len(data) <= limit_bytes:
                    return data
            return None

        # 二分搜索尽量保留分辨率
        for _ in range(14):
            mid = (low + high) / 2
            data = try_encode(mid)
            if data:
                best = data
                low = mid
            else:
                high = mid

        if best is None:
            # 兜底：最小质量 + 当前 low scale
            scale = low
            w = max(min_side, int(w0 * scale))
            h = max(min_side, int(h0 * scale))
            resized = im.resize((w, h), Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format=prefer_format, quality=min_quality, optimize=True)
            best = buf.getvalue()

        return _encode_data_uri(best, "image/jpeg")


def local_api_vision(
    sys_prompt: str,
    prompt: str,
    image_input: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    prompt_enhancer: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_image_mb: float = DEFAULT_IMAGE_LIMIT_MB,
    url: str = "http://127.0.0.1:8000/v1/chat/completions",
    model: str = "qwen_vl",  # 这里写你 vllm --served-model-name 的名字
) -> str:
    """
    本地视觉模型调用（OpenAI-compatible: /v1/chat/completions）
    - image_input: 本地图片路径 / http(s) 图片链接 / data:image/... base64
    """
    user_prompt = prompt_enhancer.format(prompt=prompt) if prompt_enhancer else prompt

    # --- image preprocessing: local file -> data uri ---
    final_image_url = image_input
    if isinstance(image_input, str) and os.path.exists(image_input):
        limit_bytes = int(max_image_mb * 1024 * 1024)
        final_image_url = _file_to_data_uri_with_limit(
            image_input,
            limit_bytes=limit_bytes,
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": final_image_url}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise last_err
