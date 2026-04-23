import numpy as np


def image_to_gray_np(pil_img):
    img = np.array(pil_img).astype(np.float32) / 255.0
    gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    return gray


def hann2d(h, w):
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx).astype(np.float32)


def fft_band_energies(
    pil_img,
    low_radius_ratio=0.12,
    mid_radius_ratio=0.30,
    eps=1e-8
):
    gray = image_to_gray_np(pil_img)
    gray = gray - gray.mean()

    h, w = gray.shape
    gray = gray * hann2d(h, w)

    F = np.fft.fft2(gray)
    F_shift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(F_shift))

    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_norm = r / (r.max() + eps)

    low_mask = r_norm <= low_radius_ratio
    mid_mask = (r_norm > low_radius_ratio) & (r_norm < mid_radius_ratio)
    high_mask = r_norm >= mid_radius_ratio

    e_low = mag[low_mask].sum()
    e_mid = mag[mid_mask].sum()
    e_high = mag[high_mask].sum()
    e_total = e_low + e_mid + e_high + eps

    return {
        "low": float(e_low / e_total),
        "mid": float(e_mid / e_total),
        "high": float(e_high / e_total),
        "ratio_hl": float(e_high / (e_low + eps)),
    }


def fit_slope(values, steps):
    x = np.array(steps, dtype=np.float32)
    y = np.array(values, dtype=np.float32)

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum() + 1e-8

    return float(numerator / denominator)


def extract_fft_trend_features(stats_by_step):
    steps = sorted(stats_by_step.keys())

    low_vals = [stats_by_step[s]["low"] for s in steps]
    mid_vals = [stats_by_step[s]["mid"] for s in steps]
    high_vals = [stats_by_step[s]["high"] for s in steps]
    ratio_vals = [stats_by_step[s]["ratio_hl"] for s in steps]

    low_slope = fit_slope(low_vals, steps)
    mid_slope = fit_slope(mid_vals, steps)
    high_slope = fit_slope(high_vals, steps)
    ratio_slope = fit_slope(ratio_vals, steps)

    low_std = float(np.std(low_vals))
    mid_std = float(np.std(mid_vals))
    high_std = float(np.std(high_vals))
    ratio_std = float(np.std(ratio_vals))

    score = high_slope - low_slope

    fft_features = np.array([
        low_vals[-1],
        mid_vals[-1],
        high_vals[-1],
        ratio_vals[-1],
        low_slope,
        mid_slope,
        high_slope,
        ratio_slope,
        low_std,
        mid_std,
        high_std,
        ratio_std,
        score,
    ], dtype=np.float32)

    aux = {
        "steps": steps,
        "low_vals": low_vals,
        "mid_vals": mid_vals,
        "high_vals": high_vals,
        "ratio_vals": ratio_vals,
        "low_slope": low_slope,
        "mid_slope": mid_slope,
        "high_slope": high_slope,
        "ratio_slope": ratio_slope,
        "score": score,
    }

    return fft_features, aux