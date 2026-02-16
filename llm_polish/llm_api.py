import base64
import json
import time
import os
import mimetypes
import inspect
import io
from PIL import Image
from typing import Optional
from openai import OpenAI
from openai import APIError, PermissionDeniedError, RateLimitError

#================================================= 文本模型部分 =================================================
def ali_api_text(
    sys_prompt: str,
    prompt: str,
    input_text: str,
    model_number: int,
    key: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,

    prompt_enhancer: Optional[str] = None,

    max_tokens: int = 10240,
    temperature: float = 1.0,
    top_p: float = 0.9,

    is_remove_emoji: bool = True,

    # ✅ 新增：向前兼容扩展（不传则完全等同原逻辑）
    provider: str = "aliyun",                # 默认阿里
    base_url: Optional[str] = None,          # 可覆盖
    api_key: Optional[str] = None,           # 可覆盖（不传则用 key）
    model_name: Optional[str] = None,        # 可直接指定模型名（优先级最高）
    provider_model_map: Optional[dict] = None,  # 可注入自定义 provider->models 映射
):
    """
    向前兼容：
    - 不传 provider/base_url/api_key/model_name/provider_model_map 时：行为与原函数一致（阿里云）。
    - 传 provider 或 base_url 后：可调用任意 OpenAI 兼容接口。
    """

    # 1) 默认阿里云（与你原来一致）
    default_provider_base_urls = {
        "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "siliconflow": "https://api.siliconflow.cn/v1",
        "local": "http://127.0.0.1:8000/v1",
    }

    default_text_model_map = {
        "aliyun": [
            "qwen-max-latest",
            "qwen-plus-latest",
            "qwen-turbo-latest",
            "qwen3-max",
        ],
        # 下面这些只是“示例默认值”，你可以按实际可用模型改/注入 provider_model_map
        "openai": ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "o4-mini"],
        "deepseek": ["deepseek-chat", "deepseek-chat", "deepseek-chat", "deepseek-reasoner"],
        "siliconflow": ["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct"],
        "local": ["qwen2.5:latest", "qwen2.5:latest", "qwen2.5:latest", "qwen2.5:latest"],
    }

    if provider_model_map:
        # 允许你传入自定义映射覆盖默认
        default_text_model_map.update(provider_model_map)

    # 2) 选择 base_url / api_key
    final_base_url = base_url or default_provider_base_urls.get(provider, default_provider_base_urls["aliyun"])
    final_api_key = api_key or key

    # 3) 选择模型
    if model_name:
        final_model = model_name
    else:
        model_list = default_text_model_map.get(provider, default_text_model_map["aliyun"])
        final_model = model_list[min(max(model_number, 0), len(model_list) - 1)]

    client = OpenAI(api_key=final_api_key, base_url=final_base_url)

    user_prompt = prompt_enhancer.format(prompt=prompt) if prompt_enhancer else prompt

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt + input_text},
    ]

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=final_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            response = completion.choices[0].message.content
            if is_remove_emoji:
                response = response.encode("gbk", errors="ignore").decode("gbk")
            return response

        except (PermissionDeniedError, RateLimitError, APIError) as e:
            print(f"[Error] {provider}/{final_model} attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise

#================================================= 视觉模型部分 =================================================

DEFAULT_IMAGE_LIMIT_MB = 6


def _encode_data_uri(img_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _file_to_data_uri_with_limit(
    image_path: str,
    limit_bytes: int,
    prefer_format: str = "JPEG",
    min_side: int = 384,          # ⬅️ 比你原来 256 更稳（文本/界面类图像）
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

    raw_size = len(raw)
    print(f"[Image] 原图大小: {raw_size/(1024*1024):.2f} MB")

    with Image.open(image_path) as im0:
        w0, h0 = im0.size
        print(f"[Image] 原图分辨率: {w0} x {h0}")

    if raw_size <= limit_bytes:
        print("[Image] 未超限，直接使用原图")
        return _encode_data_uri(raw, mime_type)

    with Image.open(image_path) as im:
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
                    return data, (w, h), scale, q
            return None

        for _ in range(14):
            mid = (low + high) / 2
            result = try_encode(mid)
            if result:
                best = result
                low = mid
            else:
                high = mid

        if best is None:
            scale = low
            w = max(min_side, int(w0 * scale))
            h = max(min_side, int(h0 * scale))
            resized = im.resize((w, h), Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format=prefer_format, quality=min_quality, optimize=True)
            best = buf.getvalue(), (w, h), scale, min_quality

        data, (fw, fh), scale, q = best
        print(
            f"[Image] 压缩后分辨率: {fw} x {fh}, "
            f"scale={scale:.3f}, quality={q}, "
            f"size={len(data)/(1024*1024):.2f} MB"
        )
        return _encode_data_uri(data, "image/jpeg")

def ali_api_vision(
    sys_prompt: str,
    prompt: str,
    image_input: str,
    model_number: int,
    api_key: str,
    max_image_mb: float = DEFAULT_IMAGE_LIMIT_MB,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    prompt_enhancer: Optional[str] = None,

    max_tokens: int = 10240,
    temperature: float = 1.0,
    top_p: float = 0.9,

    is_remove_emoji: bool = True,

    # ✅ 新增：向前兼容扩展（不传则完全等同原逻辑）
    provider: str = "aliyun",                # 默认阿里
    base_url: Optional[str] = None,          # 可覆盖
    model_name: Optional[str] = None,        # 可直接指定模型名（优先级最高）
    provider_model_map: Optional[dict] = None,   # 可注入自定义 provider->models 映射
    image_payload_style: str = "openai",     # 默认用 OpenAI 兼容的 image_url 格式
):
    """
    向前兼容：
    - 不传 provider/base_url/model_name/provider_model_map 时：行为与原函数一致（阿里云）。
    - 传 provider 或 base_url 后：可调用任意 OpenAI 兼容视觉接口（前提是对方支持该消息格式）。
    """

    default_provider_base_urls = {
        "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "siliconflow": "https://api.siliconflow.cn/v1",
        "local": "http://127.0.0.1:8000/v1",
    }

    default_vision_model_map = {
        "aliyun": [
            "qwen3-vl-flash",
            "qwen3-vl-plus",
            "qwen-vl-max",
            "qwen-vl-plus",
        ],
        # 示例默认值（按你实际可用的改）
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o4-mini"],
        "deepseek": ["deepseek-vl", "deepseek-vl", "deepseek-vl", "deepseek-vl"],
        "siliconflow": ["Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"],
        "local": ["qwen2.5-vl:latest", "qwen2.5-vl:latest", "qwen2.5-vl:latest", "qwen2.5-vl:latest"],
    }

    if provider_model_map:
        default_vision_model_map.update(provider_model_map)

    final_base_url = base_url or default_provider_base_urls.get(provider, default_provider_base_urls["aliyun"])

    # 选择模型
    if model_name:
        final_model = model_name
    else:
        model_list = default_vision_model_map.get(provider, default_vision_model_map["aliyun"])
        final_model = model_list[min(max(model_number, 0), len(model_list) - 1)]

    limit_bytes = int(max_image_mb * 1024 * 1024)

    final_image_url = image_input
    if os.path.exists(image_input):
        final_image_url = _file_to_data_uri_with_limit(
            image_input,
            limit_bytes=limit_bytes,
        )

    client = OpenAI(api_key=api_key, base_url=final_base_url)

    user_prompt = prompt_enhancer.format(prompt=prompt) if prompt_enhancer else prompt

    if image_payload_style == "openai":
        image_part = {"type": "image_url", "image_url": {"url": final_image_url}}
    elif image_payload_style == "simple":
        image_part = {"type": "image_url", "image_url": final_image_url}
    else:
        raise ValueError(f"Unknown image_payload_style: {image_payload_style}")

    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                image_part,
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=final_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            response = resp.choices[0].message.content
            if is_remove_emoji:
                response = response.encode("gbk", errors="ignore").decode("gbk")
            return response

        except (PermissionDeniedError, RateLimitError, APIError) as e:
            print(f"[Error] {provider}/{final_model} attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise

