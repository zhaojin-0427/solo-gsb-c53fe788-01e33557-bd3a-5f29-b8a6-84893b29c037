"""The soft-proofing color pipeline.

Stages (every pixel, all transforms executed by LittleCMS through Pillow):

    source pixels --(source profile, source intent)--> Lab          [Lab_src]
    source pixels --(source profile, source intent)--> press CMYK   [device]
    press CMYK    --(press profile)--> Lab                          [Lab_press]
    Lab_press     --(display profile, proof intent)--> display RGB   [proof]
    source pixels --(display profile, source intent)--> display RGB  [preview]

Pixel-level ΔE00 is computed between Lab_src and Lab_press. The browser is
never asked to interpret untagged data: every RGB image we emit is saved with
the chosen display profile embedded... (for the screen-facing previews we tag
sRGB so the file is unambiguous even when a display profile differs).
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageCms

from . import colorscience
from .profiles import ProfileManager

INTENTS = {
    "perceptual": ImageCms.Intent.PERCEPTUAL,
    "relative_colorimetric": ImageCms.Intent.RELATIVE_COLORIMETRIC,
    "saturation": ImageCms.Intent.SATURATION,
    "absolute_colorimetric": ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
}
INTENT_LABELS = {
    "perceptual": "感知",
    "relative_colorimetric": "相对色度",
    "saturation": "饱和度",
    "absolute_colorimetric": "绝对色度",
}

ALLOWED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class PipelineError(ValueError):
    pass


@dataclass
class PipelineInputs:
    image_data: bytes
    source_profile_id: str
    press_profile_id: str
    display_profile_id: str
    source_intent: str
    proof_intent: str
    threshold: float
    use_embedded: bool = False
    max_dimension: int = 3000


def _build(src, dst, in_mode: str, out_mode: str, intent: str):
    try:
        return ImageCms.buildTransform(
            src, dst, in_mode, out_mode, renderingIntent=INTENTS[intent]
        )
    except KeyError:
        raise PipelineError(f"未知渲染意图: {intent}")
    except ImageCms.PyCMSError as exc:
        raise PipelineError(f"无法建立色彩转换 ({in_mode}->{out_mode}, {intent}): {exc}")


def _apply(img: Image.Image, transform) -> Image.Image:
    try:
        return ImageCms.applyTransform(img, transform)
    except ImageCms.PyCMSError as exc:
        raise PipelineError(f"色彩转换执行失败: {exc}")


def load_pixels(data: bytes, max_dimension: int) -> tuple[Image.Image, str, bytes | None]:
    """Validate the upload and return a normalized RGB/CMYK PIL image.

    Returns (image, declared_mode, embedded_icc_bytes).
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise PipelineError(f"无法打开图像: {exc}")

    fmt = (img.format or "").upper()
    if fmt not in ("TIFF", "PNG", "JPEG"):
        raise PipelineError(f"不支持的图像格式: {fmt or '未知'}（仅支持 TIFF/PNG/JPEG）")

    embedded = None
    try:
        icc_block = img.info.get("icc_profile")
        if icc_block:
            embedded = icc_block
    except Exception:
        embedded = None

    mode = img.mode
    # Only the color spaces the spec calls for are accepted; indexed ("P"),
    # 16-bit, LA 等一律拒绝，而不是被隐式重新解释。
    if mode not in ("RGB", "RGBA", "CMYK", "L"):
        raise PipelineError(
            f"不支持的图像位深/模式: {mode}（仅支持 8 位 RGB/RGBA/CMYK/灰度；"
            "请先在印前软件中转换并嵌入 ICC 配置）"
        )

    if mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        img = Image.composite(img, bg.convert("RGBA"), img.split()[-1]).convert("RGB")
        mode = "RGB"

    if max(img.size) > max_dimension:
        raise PipelineError(
            f"图像尺寸 {img.size[0]}x{img.size[1]} 超过上限 {max_dimension}px"
        )
    return img, mode, embedded


class PipelineResult:
    def __init__(self, inputs: PipelineInputs, image_sha: str,
                 img: Image.Image, mode: str,
                 lab_src: np.ndarray, lab_press: np.ndarray,
                 cmyk: np.ndarray, de: np.ndarray,
                 preview: Image.Image, proof: Image.Image,
                 heatmap: np.ndarray, mask: np.ndarray,
                 stats: dict, profile_summaries: dict,
                 embedded_sha: str | None):
        self.inputs = inputs
        self.image_sha = image_sha
        self.img = img
        self.mode = mode
        self.lab_src = lab_src
        self.lab_press = lab_press
        self.cmyk = cmyk
        self.de = de
        self.preview = preview
        self.proof = proof
        self.heatmap = heatmap
        self.mask = mask
        self.stats = stats
        self.profile_summaries = profile_summaries
        self.embedded_sha = embedded_sha


def run_pipeline(inputs: PipelineInputs, pm: ProfileManager,
                 lab_profile: ImageCms.ImageCmsProfile,
                 srgb_profile: ImageCms.ImageCmsProfile) -> PipelineResult:
    if inputs.threshold < 0:
        raise PipelineError("色差阈值不能为负")
    if inputs.source_intent not in INTENTS or inputs.proof_intent not in INTENTS:
        raise PipelineError("渲染意图无效")

    image_sha = hashlib.sha256(inputs.image_data).hexdigest()
    img, mode, embedded_icc = load_pixels(inputs.image_data, inputs.max_dimension)

    # ---- resolve source profile (embedded upload wins when requested) ----
    embedded_sha = None
    if inputs.use_embedded:
        if not embedded_icc:
            raise PipelineError("图像中没有内嵌的 ICC 配置，无法按“使用内嵌源配置”处理")
        try:
            from . import lcms_native
            embedded_header = lcms_native.read_header(embedded_icc)
            src_prof = ImageCms.ImageCmsProfile(io.BytesIO(embedded_icc))
        except Exception as exc:
            raise PipelineError(f"内嵌 ICC 配置无效: {exc}") from exc
        src_cs = embedded_header["color_space"]
        embedded_sha = hashlib.sha256(embedded_icc).hexdigest()
        if src_cs not in ("RGB", "CMYK", "GRAY"):
            raise PipelineError(f"内嵌配置的色彩空间不支持: {src_cs}")
    else:
        src_prof = pm.get(inputs.source_profile_id, "source")
        _, src_header = pm.get_bytes_header(inputs.source_profile_id)
        src_cs = src_header["color_space"]

    press_prof = pm.get(inputs.press_profile_id, "press")
    display_prof = pm.get(inputs.display_profile_id, "display")

    # ---- strict channel/color-space matching ----------------------------
    pixel_mode = mode
    if pixel_mode == "L":
        pixel_cs = "GRAY"
    else:
        pixel_cs = pixel_mode
    if pixel_cs != src_cs:
        raise PipelineError(
            f"图像色彩空间/通道数与源配置不匹配: 图像为 {pixel_cs}"
            f"（{'1' if pixel_cs == 'GRAY' else ('4' if pixel_cs == 'CMYK' else '3')} 通道），"
            f"源配置为 {src_cs}，拒绝按浏览器默认色彩解释"
        )
    in_mode = "L" if pixel_cs == "GRAY" else pixel_cs

    # ---- transforms -------------------------------------------------------
    # Source -> Lab (back-calculation of color values to Lab via source ICC)
    lab_src_img = _apply(img, _build(src_prof, lab_profile, in_mode, "LAB",
                                    inputs.source_intent))
    lab_src = colorscience.lab_image_to_array(lab_src_img)

    # Source -> press device CMYK
    cmyk_img = _apply(img, _build(src_prof, press_prof, in_mode, "CMYK",
                                 inputs.source_intent))
    cmyk = np.asarray(cmyk_img)

    # Press CMYK -> Lab (沿源→印刷→Lab 回算色值)
    lab_press_img = _apply(cmyk_img, _build(press_prof, lab_profile, "CMYK", "LAB",
                                           inputs.source_intent))
    lab_press = colorscience.lab_image_to_array(lab_press_img)

    # Lab_press -> display RGB = the soft proof
    # (route via CMYK again would be lossy; Lab PCS -> display is the proof).
    proof_img = _apply(lab_press_img, _build(lab_profile, display_prof, "LAB", "RGB",
                                             inputs.proof_intent))
    # Source -> display direct preview
    preview_img = _apply(img, _build(src_prof, display_prof, in_mode, "RGB",
                                    inputs.source_intent))

    # ---- ΔE, heatmap, mask ----------------------------------------------
    de = colorscience.ciede2000(lab_src, lab_press)
    threshold = float(inputs.threshold)
    heat = colorscience.heatmap_rgb(de, vmax=max(float(np.percentile(de, 99)), threshold * 2, 1.0))
    mask = colorscience.threshold_mask(de, threshold)

    over = float((de >= threshold).sum()) / de.size
    stats = {
        "mean": float(de.mean()),
        "p95": float(np.percentile(de, 95)),
        "max": float(de.max()),
        "over_ratio": over,
        "threshold": threshold,
        "over_pixels": int((de >= threshold).sum()),
        "total_pixels": int(de.size),
    }

    # ---- profile summaries for the deterministic report ------------------
    summaries = {}
    if inputs.use_embedded:
        summaries["source"] = {
            "profile_id": "<embedded>",
            "embedded": True,
            "sha256": embedded_sha,
            **embedded_header,
        }
    else:
        summaries["source"] = pm.summary(inputs.source_profile_id)
    summaries["press"] = pm.summary(inputs.press_profile_id)
    summaries["display"] = pm.summary(inputs.display_profile_id)

    return PipelineResult(
        inputs=inputs, image_sha=image_sha, img=img, mode=mode,
        lab_src=lab_src, lab_press=lab_press, cmyk=cmyk, de=de,
        preview=preview_img, proof=proof_img, heatmap=heat,
        mask=np.stack([mask] * 3, axis=-1),
        stats=stats, profile_summaries=summaries, embedded_sha=embedded_sha,
    )


# ---------------------------------------------------------------------------
# Export artifacts
# ---------------------------------------------------------------------------

def _png(img: Image.Image, icc_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", icc_profile=icc_bytes)
    return buf.getvalue()


def export_pngs(result: PipelineResult, srgb_profile, display_icc_bytes: bytes) -> dict[str, bytes]:
    """Emit the five PNG artifacts.

    preview/softproof live in the chosen display profile color space and are
    tagged with that profile; the heatmap, mask and their overlay are synthetic
    visualizations without physical colorimetry and are tagged sRGB so browsers
    reproduce their colors deterministically.
    """
    srgb_bytes = srgb_profile.tobytes()
    overlay = colorscience.overlay_heatmap(
        np.asarray(result.proof), result.heatmap, alpha=0.55
    )
    return {
        "preview.png": _png(result.preview, display_icc_bytes),
        "softproof.png": _png(result.proof, display_icc_bytes),
        "heatmap.png": _png(Image.fromarray(result.heatmap, "RGB"), srgb_bytes),
        "overlay.png": _png(Image.fromarray(overlay, "RGB"), srgb_bytes),
        "mask.png": _png(Image.fromarray(result.mask, "RGB"), srgb_bytes),
    }


def compute_job_id(result: PipelineResult) -> str:
    h = hashlib.sha256()
    h.update(result.image_sha.encode())
    parts = [
        result.profile_summaries["source"]["sha256"],
        result.profile_summaries["press"]["sha256"],
        result.profile_summaries["display"]["sha256"],
        result.inputs.source_intent,
        result.inputs.proof_intent,
        f"{result.inputs.threshold:.6f}",
    ]
    for p in parts:
        h.update(p.encode())
    return h.hexdigest()


def build_report(result: PipelineResult, job_id: str) -> dict:
    w, h = result.img.size
    return {
        "job_id": job_id,
        "image": {
            "sha256": result.image_sha,
            "width": w,
            "height": h,
            "mode": result.mode,
            "embedded_source_sha256": result.embedded_sha,
        },
        "profiles": result.profile_summaries,
        "parameters": {
            "source_intent": result.inputs.source_intent,
            "source_intent_label": INTENT_LABELS[result.inputs.source_intent],
            "proof_intent": result.inputs.proof_intent,
            "proof_intent_label": INTENT_LABELS[result.inputs.proof_intent],
            "threshold_delta_e": result.inputs.threshold,
            "use_embedded_source": result.inputs.use_embedded,
        },
        "stats": result.stats,
        "lab_note": (
            "Lab 值取 LittleCMS applyTransform 输出的 ICC v2 8 位 PCS 字节直接解码"
            "(L*=byte*100/255; a/b: byte<=127 取 byte，否则 byte-256; 0 与 255 均为中性)，"
            "像素级量化误差约 ±0.5 ΔE00 (p95<0.5)"
        ),
    }


def export_zip(pngs: dict[str, bytes], report: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(pngs.items()):
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, data)
        report_bytes = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
        zi = zipfile.ZipInfo("report.json", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, report_bytes)
    return buf.getvalue()
