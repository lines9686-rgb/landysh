"""Ретушь фото ресниц в стиле лэш-мастеров.

Кожа сглаживается и выравнивается по тону, ресницы и глаз не трогаются
(только лёгкая резкость). Ресницы НЕ дорисовываются.

python retouch.py вход.jpg выход.jpg [сила 0..1]
"""
import sys
import cv2
import numpy as np


def retouch(src, dst, strength=0.75):
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"не открыть {src}")
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # --- маска «защиты»: ресницы, брови, зрачок, края — всё тёмное и контрастное
    dark = (L < np.percentile(L, 22)).astype(np.uint8) * 255
    edges = cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 40, 110)
    protect = cv2.bitwise_or(dark, edges)
    k = max(3, int(min(h, w) * 0.012)) | 1
    protect = cv2.dilate(protect, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    protect = cv2.GaussianBlur(protect, (0, 0), k / 2).astype(np.float32) / 255.0
    skin = (1.0 - protect)[:, :, None]

    # --- сглаживание кожи (двойной bilateral) + убрать мелкие красные пятна
    d = max(5, int(min(h, w) * 0.012))
    smooth = cv2.bilateralFilter(img, d, 40, d)
    smooth = cv2.bilateralFilter(smooth, d, 30, d)
    # выравнивание тона: сгладить хрому (a,b) сильнее, чем яркость
    slab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    big = max(9, int(min(h, w) * 0.04)) | 1
    slab[:, :, 1] = cv2.GaussianBlur(slab[:, :, 1], (big, big), 0) * 0.6 + slab[:, :, 1] * 0.4
    slab[:, :, 1] = 128 + (slab[:, :, 1] - 128) * 0.85  # меньше красноты
    smooth = cv2.cvtColor(np.clip(slab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    out = img.astype(np.float32) * (1 - skin * strength) + smooth.astype(np.float32) * (skin * strength)

    # --- резкость только на ресницах/глазу
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    sharp = cv2.addWeighted(out, 1.55, blur, -0.55, 0)
    p = protect[:, :, None]
    out = out * (1 - p) + sharp * p

    # --- общий свет: чуть светлее, мягкий контраст, чистый нейтральный тон
    out = np.clip(out, 0, 255).astype(np.uint8)
    olab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    Lc = olab[:, :, 0] / 255.0
    Lc = np.clip(Lc * 1.05 + 0.02, 0, 1)
    Lc = Lc + 0.08 * np.sin(np.pi * Lc) * (Lc - 0.5) * 2  # S-кривая
    olab[:, :, 0] = np.clip(Lc, 0, 1) * 255
    olab[:, :, 2] = 128 + (olab[:, :, 2] - 128) * 0.95  # меньше желтизны
    out = cv2.cvtColor(np.clip(olab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return img, out


if __name__ == "__main__":
    s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
    retouch(sys.argv[1], sys.argv[2], s)
