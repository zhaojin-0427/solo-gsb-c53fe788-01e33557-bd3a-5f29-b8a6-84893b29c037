"""Minimal ctypes bindings for the subset of LittleCMS 2 we need.

Pillow's ImageCms is used for the pixel transforms themselves, but we go to
liblcms2 directly for a few things Pillow cannot do:

* create a real Lab identity profile (``cmsCreateLab2Profile``) so the
  pipeline can land in CIELAB PCS with V2 8-bit encoding;
* synthesize a self-contained CMYK demo profile at startup, so the app can
  demonstrate the full source -> press -> display chain without requiring
  the user to ship a press profile;
* read ICC header fields (device class / color space / PCS) for strict
  validation.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import struct
from functools import lru_cache

import numpy as np

# ICC 4-byte signatures (big-endian uint32 on the wire/API).
def sig(value: bytes) -> int:
    return struct.unpack(">I", value)[0]


class _CIEXYZ(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_double),
        ("Y", ctypes.c_double),
        ("Z", ctypes.c_double),
    ]


@lru_cache(maxsize=1)
def lcms() -> ctypes.CDLL:
    lib = ctypes.util.find_library("lcms2")
    if not lib:
        raise RuntimeError("找不到 LittleCMS 2 共享库 (liblcms2)")
    L = ctypes.CDLL(lib)

    L.cmsCreate_sRGBProfile.restype = ctypes.c_void_p
    L.cmsCreate_sRGBProfile.argtypes = []

    L.cmsCreateLab2Profile.restype = ctypes.c_void_p
    L.cmsCreateLab2Profile.argtypes = [ctypes.c_void_p]

    L.cmsCreateProfilePlaceholder.restype = ctypes.c_void_p
    L.cmsCreateProfilePlaceholder.argtypes = [ctypes.c_void_p]

    for name in ("cmsSetPCS", "cmsSetColorSpace", "cmsSetDeviceClass"):
        getattr(L, name).restype = None
        getattr(L, name).argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    L.cmsSetProfileVersion.restype = None
    L.cmsSetProfileVersion.argtypes = [ctypes.c_void_p, ctypes.c_double]

    L.cmsMLUalloc.restype = ctypes.c_void_p
    L.cmsMLUalloc.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    L.cmsMLUsetASCII.restype = ctypes.c_int
    L.cmsMLUsetASCII.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]

    L.cmsWriteTag.restype = ctypes.c_int
    L.cmsWriteTag.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]

    L.cmsCloseProfile.restype = ctypes.c_int
    L.cmsCloseProfile.argtypes = [ctypes.c_void_p]

    L.cmsSaveProfileToMem.restype = ctypes.c_int
    L.cmsSaveProfileToMem.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]

    L.cmsSaveProfileToFile.restype = ctypes.c_int
    L.cmsSaveProfileToFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    L.cmsOpenProfileFromMem.restype = ctypes.c_void_p
    L.cmsOpenProfileFromMem.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    L.cmsGetColorSpace.restype = ctypes.c_uint32
    L.cmsGetColorSpace.argtypes = [ctypes.c_void_p]

    L.cmsGetPCS.restype = ctypes.c_uint32
    L.cmsGetPCS.argtypes = [ctypes.c_void_p]

    L.cmsGetDeviceClass.restype = ctypes.c_uint32
    L.cmsGetDeviceClass.argtypes = [ctypes.c_void_p]

    L.cmsPipelineAlloc.restype = ctypes.c_void_p
    L.cmsPipelineAlloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    L.cmsPipelineFree.restype = None
    L.cmsPipelineFree.argtypes = [ctypes.c_void_p]

    L.cmsStageAllocCLut16bit.restype = ctypes.c_void_p
    L.cmsStageAllocCLut16bit.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint16),
    ]

    L.cmsPipelineInsertStage.restype = ctypes.c_int
    L.cmsPipelineInsertStage.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]

    L.cmsD50_XYZ.restype = ctypes.POINTER(_CIEXYZ)
    L.cmsD50_XYZ.argtypes = []

    # High-precision pixel transforms (TYPE_Lab_DBL keeps CIELAB in native
    # doubles instead of quantizing through the ICC v2 8-bit PCS encoding).
    L.cmsCreateTransform.restype = ctypes.c_void_p
    L.cmsCreateTransform.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32,
    ]
    L.cmsDoTransform.restype = None
    L.cmsDoTransform.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
    ]
    L.cmsDeleteTransform.restype = None
    L.cmsDeleteTransform.argtypes = [ctypes.c_void_p]

    return L


class _CIELab(ctypes.Structure):
    """cmsCIELab: three native doubles -- the memory layout of TYPE_Lab_DBL."""
    _fields_ = [
        ("L", ctypes.c_double),
        ("a", ctypes.c_double),
        ("b", ctypes.c_double),
    ]


# Rendering intent numbers (lcms2.h INTENT_*); identical to ImageCms.Intent.
INTENT_CODES = {
    "perceptual": 0,
    "relative_colorimetric": 1,
    "saturation": 2,
    "absolute_colorimetric": 3,
}

# TYPE_* constants assembled exactly as in lcms2.h.
_FLOAT_SH = lambda a: a << 22
_COLORSPACE_SH = lambda s: s << 16
_CHANNELS_SH = lambda c: c << 3
_BYTES_SH = lambda b: b

# PT_GRAY 3, PT_RGB 4, PT_CMYK 6, PT_Lab 10.
_TYPE_GRAY_8 = _COLORSPACE_SH(3) | _CHANNELS_SH(1) | _BYTES_SH(1)
_TYPE_RGB_8 = _COLORSPACE_SH(4) | _CHANNELS_SH(3) | _BYTES_SH(1)
_TYPE_CMYK_8 = _COLORSPACE_SH(6) | _CHANNELS_SH(4) | _BYTES_SH(1)
_TYPE_LAB_DBL = _FLOAT_SH(1) | _COLORSPACE_SH(10) | _CHANNELS_SH(3) | _BYTES_SH(0)

_PIXEL_FORMATS_8BIT = {
    1: _TYPE_GRAY_8,
    3: _TYPE_RGB_8,
    4: _TYPE_CMYK_8,
}


def _profile_to_bytes(handle: ctypes.c_void_p) -> bytes:
    L = lcms()
    size = ctypes.c_uint32(0)
    L.cmsSaveProfileToMem(handle, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not L.cmsSaveProfileToMem(handle, buf, ctypes.byref(size)):
        raise RuntimeError("保存 ICC profile 到内存失败")
    return buf.raw[: size.value]


def build_lab_profile_bytes() -> bytes:
    """CIELAB identity profile (ICC v2, PCS Lab) used to materialize Lab."""
    L = lcms()
    handle = L.cmsCreateLab2Profile(None)
    if not handle:
        raise RuntimeError("cmsCreateLab2Profile 失败")
    try:
        return _profile_to_bytes(handle)
    finally:
        L.cmsCloseProfile(handle)


def build_srgb_profile_bytes() -> bytes:
    L = lcms()
    handle = L.cmsCreate_sRGBProfile()
    try:
        return _profile_to_bytes(handle)
    finally:
        L.cmsCloseProfile(handle)


# ---------------------------------------------------------------------------
# High-precision pixel transforms (CIELAB as native doubles)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _lab_profile_handle() -> ctypes.c_void_p:
    """Long-lived handle to the Lab identity profile for the Lab PCS legs."""
    L = lcms()
    handle = L.cmsCreateLab2Profile(None)
    if not handle:
        raise RuntimeError("cmsCreateLab2Profile 失败")
    return handle


def _open_handle(data: bytes):
    """Open an ICC blob; return (handle, buffer) keeping the buffer alive."""
    L = lcms()
    buf = ctypes.create_string_buffer(data, len(data))
    handle = L.cmsOpenProfileFromMem(buf, len(data))
    if not handle:
        raise RuntimeError("LittleCMS 无法打开 ICC 配置")
    return handle, buf


def pixels_to_lab_double(profile_bytes: bytes, pixels: np.ndarray,
                         channels: int, intent: int) -> np.ndarray:
    """Transform 8-bit device pixels to CIELAB via LittleCMS TYPE_Lab_DBL.

    Lab values (L in 0..100, a/b signed) come straight out of lcms as native
    doubles -- no v2 8-bit PCS quantization, so every CIEDE2000 result uses the
    full-precision colorimetry.
    """
    return _native_transform(
        profile_bytes, _PIXEL_FORMATS_8BIT[channels],
        _lab_profile_handle(), _TYPE_LAB_DBL,
        pixels, intent, np.float64, 3,
    )


def lab_double_to_rgb8(profile_bytes: bytes, lab: np.ndarray,
                       intent: int) -> np.ndarray:
    """Transform CIELAB doubles to 8-bit display RGB via LittleCMS."""
    return _native_transform(
        _lab_profile_handle(), _TYPE_LAB_DBL,
        profile_bytes, _TYPE_RGB_8,
        lab, intent, np.uint8, 3,
    )


def _native_transform(in_profile, in_format, out_profile, out_format,
                      values: np.ndarray, intent: int,
                      out_dtype, out_channels: int) -> np.ndarray:
    """Run a one-shot lcms transform on flattened pixel rows.

    Profile arguments may be ICC bytes (opened here and closed when done) or an
    already-open cmsHPROFILE (the cached Lab identity, which is never closed).
    """
    L = lcms()
    in_buf = out_buf = None
    in_opened = out_opened = None
    try:
        if isinstance(in_profile, (bytes, bytearray)):
            in_opened, in_buf = _open_handle(bytes(in_profile))
            in_h = in_opened
        else:
            in_h = in_profile
        if isinstance(out_profile, (bytes, bytearray)):
            out_opened, out_buf = _open_handle(bytes(out_profile))
            out_h = out_opened
        else:
            out_h = out_profile

        flat = np.ascontiguousarray(values).reshape(-1, values.shape[-1])
        n = flat.shape[0]
        out = np.empty((n, out_channels), dtype=out_dtype)
        xform = L.cmsCreateTransform(
            in_h, in_format, out_h, out_format, intent, 0
        )
        if not xform:
            raise RuntimeError("LittleCMS 无法建立高精度色彩转换")
        try:
            L.cmsDoTransform(
                xform,
                flat.ctypes.data_as(ctypes.c_void_p),
                out.ctypes.data_as(ctypes.c_void_p),
                n,
            )
        finally:
            L.cmsDeleteTransform(xform)
        return out.reshape(*values.shape[:-1], out_channels)
    finally:
        if in_opened is not None:
            L.cmsCloseProfile(in_opened)
        if out_opened is not None:
            L.cmsCloseProfile(out_opened)


def read_header(data: bytes) -> dict[str, str | int]:
    """Parse the ICC profile header without trusting lcms to load it."""
    if len(data) < 128 or data[36:40] != b"acsp":
        raise ValueError("不是有效的 ICC profile（缺少 acsp 签名）")
    declared = struct.unpack(">I", data[0:4])[0]
    if declared != len(data):
        # lcms tolerates trailing bytes when loading, but a malformed size
        # field is a strong signal of a corrupt file; reject it.
        if declared > len(data):
            raise ValueError("ICC profile 长度字段与文件不一致")
    return {
        "size": declared,
        "version": data[8:12].hex(),
        "device_class": data[12:16].decode("latin1").strip(),
        "color_space": data[16:20].decode("ascii").strip(),
        "pcs": data[20:24].decode("ascii").strip(),
    }


# ---------------------------------------------------------------------------
# Synthetic CMYK demo profile
# ---------------------------------------------------------------------------

# sRGB linear -> CIE XYZ D50 (Bradford-adapted, Bruce Lindbloom, full precision).
_SRGB_TO_XYZ_D50 = np.array(
    [
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139322, 0.0971045, 0.7141733],
    ]
)
_XYZ_TO_SRGB_D50 = np.linalg.inv(_SRGB_TO_XYZ_D50)
_D50 = (0.9642, 1.0, 0.8249)
_GRID_POINTS = 17


def _srgb_eotf(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb_oetf(c: np.ndarray) -> np.ndarray:
    return np.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * np.power(np.clip(c, 0, None), 1.0 / 2.4) - 0.055,
    )


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    Xn, Yn, Zn = _D50
    scaled = xyz / np.array([Xn, Yn, Zn])
    f = np.where(
        scaled > 216.0 / 24389.0,
        np.cbrt(scaled),
        7.787 * scaled + 16.0 / 116.0,
    )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack(
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1
    )


def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def finv(f: np.ndarray) -> np.ndarray:
        return np.where(f**3 > 216.0 / 24389.0, f**3, (f - 16.0 / 116.0) / 7.787)

    Xn, Yn, Zn = _D50
    return np.stack([finv(fx) * Xn, finv(fy) * Yn, finv(fz) * Zn], axis=-1)


def _a2b_table() -> np.ndarray:
    """CMYK device -> Lab PCS, shape (n^4, 3), uint16 (ICC 16-bit Lab)."""
    n = _GRID_POINTS
    v = np.linspace(0.0, 1.0, n)
    c, m, y, k = np.meshgrid(v, v, v, v, indexing="ij")
    # Simple "physical" model: ink removes its primary, black removes all.
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    rgb_lin = _srgb_eotf(np.stack([r, g, b], axis=-1))
    xyz = rgb_lin @ _SRGB_TO_XYZ_D50.T
    lab = _xyz_to_lab(xyz)
    # ICC 16-bit Lab encoding: L*65535/100, (a,b)*256 + 32768.
    enc = np.stack(
        [lab[..., 0] * 655.35, lab[..., 1] * 256.0 + 32768.0, lab[..., 2] * 256.0 + 32768.0],
        axis=-1,
    )
    return np.clip(np.rint(enc), 0, 65535).astype(np.uint16).reshape(-1, 3)


def _b2a_table() -> np.ndarray:
    """Lab PCS -> CMYK device, shape (n^3, 4), uint16."""
    n = _GRID_POINTS
    Lv = np.linspace(0.0, 100.0, n)
    av = np.linspace(-128.0, 127.0, n)
    bv = np.linspace(-128.0, 127.0, n)
    lab = np.stack(np.meshgrid(Lv, av, bv, indexing="ij"), axis=-1).reshape(-1, 3)
    rgb_lin = np.clip(_lab_to_xyz(lab) @ _XYZ_TO_SRGB_D50.T, 0.0, 1.0)
    rgb = _srgb_oetf(rgb_lin)
    c = 1.0 - rgb[..., 0]
    m = 1.0 - rgb[..., 1]
    y = 1.0 - rgb[..., 2]
    k = np.minimum(np.minimum(c, m), y)
    # Full undercolor removal: CMY normalized after black.
    denom = np.clip(1.0 - k, 1e-12, None)
    cmyk = np.stack(
        [(c - k) / denom, (m - k) / denom, (y - k) / denom, k], axis=-1
    )
    cmyk = np.where(k[..., None] >= 1.0, 0.0, cmyk)
    return np.clip(np.rint(cmyk * 65535.0), 0, 65535).astype(np.uint16)


def _write_profile(prof: ctypes.c_void_p) -> bytes:
    return _profile_to_bytes(prof)


def _clut_tag(table: np.ndarray, n_in: int, n_out: int, grid: int = _GRID_POINTS):
    """Attach a CLUT16 pipeline (tag bytes are owned by the profile)."""
    L = lcms()
    pipe = L.cmsPipelineAlloc(None, n_in, n_out)
    if not pipe:
        raise RuntimeError("cmsPipelineAlloc 失败")
    flat = np.ascontiguousarray(table, dtype=np.uint16)
    ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    stage = L.cmsStageAllocCLut16bit(None, grid, n_in, n_out, ptr)
    if not stage or not L.cmsPipelineInsertStage(pipe, 0, stage):
        L.cmsPipelineFree(pipe)
        raise RuntimeError("CLUT stage 构建失败")
    return pipe


def build_demo_gray_profile_bytes() -> bytes:
    """Synthetic sRGB-tone gray monitor profile (GRAY -> Lab PCS)."""
    L = lcms()
    n = _GRID_POINTS
    v = np.linspace(0.0, 1.0, n)
    lin = _srgb_eotf(v)
    # Neutral sRGB device value (R=G=B=g): XYZ = [g,g,g] @ M.T = g * row sums.
    sums = _SRGB_TO_XYZ_D50.sum(axis=1)
    xyz = lin[:, None] * sums[None, :]
    lab = _xyz_to_lab(xyz)
    a2b = np.stack(
        [lab[:, 0] * 655.35, lab[:, 1] * 256.0 + 32768.0, lab[:, 2] * 256.0 + 32768.0],
        axis=-1,
    )
    a2b = np.clip(np.rint(a2b), 0, 65535).astype(np.uint16).reshape(-1, 3)

    Lv = np.linspace(0.0, 100.0, n)
    av = np.linspace(-128.0, 127.0, n)
    bv = np.linspace(-128.0, 127.0, n)
    lab_grid = np.stack(np.meshgrid(Lv, av, bv, indexing="ij"), axis=-1).reshape(-1, 3)
    rgb_lin = np.clip(_lab_to_xyz(lab_grid) @ _XYZ_TO_SRGB_D50.T, 0.0, 1.0)
    rgb = _srgb_oetf(rgb_lin)
    # Gray device value: luminance-equivalent mean of the reconstructed RGB.
    gray = rgb.mean(axis=1)
    b2a = np.clip(np.rint(gray * 65535.0), 0, 65535).astype(np.uint16).reshape(-1)

    prof = L.cmsCreateProfilePlaceholder(None)
    try:
        L.cmsSetProfileVersion(prof, 2.4)
        L.cmsSetDeviceClass(prof, sig(b"mntr"))
        L.cmsSetColorSpace(prof, sig(b"GRAY"))
        L.cmsSetPCS(prof, sig(b"Lab "))
        mlu = L.cmsMLUalloc(None, 1)
        L.cmsMLUsetASCII(mlu, b"en", b"US", b"Soft Proofing Demo sRGB Gray (synthetic)")
        L.cmsWriteTag(prof, sig(b"desc"), mlu)
        cprt = L.cmsMLUalloc(None, 1)
        L.cmsMLUsetASCII(cprt, b"en", b"US", b"Public domain synthetic demo profile")
        L.cmsWriteTag(prof, sig(b"cprt"), cprt)
        L.cmsWriteTag(prof, sig(b"wtpt"), L.cmsD50_XYZ())
        p1 = _clut_tag(a2b, 1, 3)
        ok = L.cmsWriteTag(prof, sig(b"A2B0"), p1)
        L.cmsPipelineFree(p1)
        if not ok:
            raise RuntimeError("gray A2B0 写入失败")
        p2 = _clut_tag(b2a, 3, 1)
        ok = L.cmsWriteTag(prof, sig(b"B2A0"), p2)
        L.cmsPipelineFree(p2)
        if not ok:
            raise RuntimeError("gray B2A0 写入失败")
        return _profile_to_bytes(prof)
    finally:
        L.cmsCloseProfile(prof)


def build_demo_cmyk_profile_bytes(
    description: str = "Soft Proofing Demo SWOP-like CMYK (synthetic, do not use for production)",
) -> bytes:
    """Build a self-contained, valid CMYK output profile (ICC v2.4).

    The gamut model is intentionally simple (complementary inks + sRGB
    chromaticities) -- it exists so the full pipeline runs out of the box;
    users should upload real press profiles for production work.
    """
    L = lcms()
    prof = L.cmsCreateProfilePlaceholder(None)
    if not prof:
        raise RuntimeError("cmsCreateProfilePlaceholder 失败")
    try:
        L.cmsSetProfileVersion(prof, 2.4)
        L.cmsSetDeviceClass(prof, sig(b"prtr"))
        L.cmsSetColorSpace(prof, sig(b"CMYK"))
        L.cmsSetPCS(prof, sig(b"Lab "))

        mlu = L.cmsMLUalloc(None, 1)
        L.cmsMLUsetASCII(mlu, b"en", b"US", description.encode("utf-8"))
        if not L.cmsWriteTag(prof, sig(b"desc"), mlu):
            raise RuntimeError("写入 desc 标签失败")

        cprt = L.cmsMLUalloc(None, 1)
        L.cmsMLUsetASCII(cprt, b"en", b"US", b"Public domain synthetic demo profile")
        if not L.cmsWriteTag(prof, sig(b"cprt"), cprt):
            raise RuntimeError("写入 cprt 标签失败")

        if not L.cmsWriteTag(prof, sig(b"wtpt"), L.cmsD50_XYZ()):
            raise RuntimeError("写入 wtpt 标签失败")

        for tag, table, n_in, n_out in (
            (b"A2B0", _a2b_table(), 4, 3),
            (b"B2A0", _b2a_table(), 3, 4),
        ):
            pipe = L.cmsPipelineAlloc(None, n_in, n_out)
            if not pipe:
                raise RuntimeError("cmsPipelineAlloc 失败")
            flat = np.ascontiguousarray(table, dtype=np.uint16)
            ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            stage = L.cmsStageAllocCLut16bit(
                None, _GRID_POINTS, n_in, n_out, ptr
            )
            if not stage or not L.cmsPipelineInsertStage(pipe, 0, stage):
                L.cmsPipelineFree(pipe)
                raise RuntimeError(f"构建 {tag.decode()} CLUT 失败")
            ok = L.cmsWriteTag(prof, sig(tag), pipe)
            L.cmsPipelineFree(pipe)  # profile owns its dup of the pipeline
            if not ok:
                raise RuntimeError(f"写入 {tag.decode()} 标签失败")

        return _profile_to_bytes(prof)
    finally:
        L.cmsCloseProfile(prof)
