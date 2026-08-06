"""
图片编辑模块 - 基于 InstructPix2Pix
支持通过自然语言指令编辑图片（如"把帽子换成蓝色的"）
使用 Mac M3 MPS 加速
"""

import os
import torch
import PIL.Image
from diffusers import StableDiffusionInstructPix2PixPipeline

# ==================== 配置 ====================
# 使用本地已下载的模型路径，避免 HuggingFace 缓存问题
MODEL_ID = os.path.expanduser("~/.cache/huggingface/hub/models--timbrooks--instruct-pix2pix/snapshots/model")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "edited_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 模型懒加载 ====================
_pipeline = None


def _get_pipeline():
    """懒加载模型：首次调用时加载，后续复用"""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print("[image_editor] 正在加载 InstructPix2Pix 模型...")

    # 检测可用设备：MPS (Mac) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    _pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,  # 禁用安全检查器以加速
    )
    _pipeline = _pipeline.to(device)

    # MPS 需要额外设置
    if device == "mps":
        _pipeline.enable_attention_slicing()

    print(f"[image_editor] 模型加载完成（设备: {device}）")
    return _pipeline


# ==================== 核心功能 ====================

def edit_image(image_path: str, instruction: str) -> str:
    """
    根据文字指令编辑图片。

    参数 image_path: 原始图片路径
    参数 instruction: 编辑指令（英文效果最好，如 "Replace the red hat with a blue cap"）
    返回: 修改后的图片保存路径
    """
    # 加载图片
    try:
        image = PIL.Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"❌ 无法打开图片: {e}"

    # 调整图片大小（InstructPix2Pix 要求 512x512 的倍数）
    width, height = image.size
    new_width = (width // 8) * 8  # 确保是 8 的倍数
    new_height = (height // 8) * 8
    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height), PIL.Image.LANCZOS)

    # 限制最大尺寸以控制内存（最大 768x768）
    max_side = 768
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        # 确保是 8 的倍数
        new_size = ((new_size[0] // 8) * 8, (new_size[1] // 8) * 8)
        image = image.resize(new_size, PIL.Image.LANCZOS)

    # 加载模型并执行编辑
    try:
        pipe = _get_pipeline()
        result = pipe(
            prompt=instruction,
            image=image,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
        ).images[0]

        # 保存结果
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{basename}_edited.png")
        result.save(output_path)

        return output_path

    except Exception as e:
        return f"❌ 图片编辑失败: {e}"


def release_model():
    """释放模型内存（可选，用于不需要图片编辑时回收内存）"""
    global _pipeline
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[image_editor] 模型已释放")
