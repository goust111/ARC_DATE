import random

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    solid_rows = [r for r in range(h) if np.all(grid[r] == SEP)]
    solid_cols = [c for c in range(w) if np.all(grid[:, c] == SEP)]
    assert len(solid_rows) == 1, "expected exactly one full grey row"
    assert len(solid_cols) == 1, "expected exactly one full grey column"
    cross_r = solid_rows[0]
    cross_c = solid_cols[0]
    assert 0 < cross_r < h - 1
    assert 0 < cross_c < w - 1
    return cross_r, cross_c


def _dominant_color(region: np.ndarray) -> int:
    colors, counts = np.unique(region, return_counts=True)
    freq = {
        int(color): int(count)
        for color, count in zip(colors.tolist(), counts.tolist(), strict=False)
        if int(color) not in (BG, SEP)
    }
    assert freq, "each quadrant must contain a non-background main color"
    best = max(freq.values())
    winners = [color for color, count in freq.items() if count == best]
    assert len(winners) == 1, "quadrant dominant color must be unique"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cross_r, cross_c = _find_separator_cross(g)
    out = np.full_like(g, BG)
    out[cross_r, :] = SEP
    out[:, cross_c] = SEP

    quadrant_slices = {
        "tl": (slice(0, cross_r), slice(0, cross_c)),
        "tr": (slice(0, cross_r), slice(cross_c + 1, g.shape[1])),
        "bl": (slice(cross_r + 1, g.shape[0]), slice(0, cross_c)),
        "br": (slice(cross_r + 1, g.shape[0]), slice(cross_c + 1, g.shape[1])),
    }
    dominant = {name: _dominant_color(g[slc]) for name, slc in quadrant_slices.items()}
    swapped = {
        "tl": dominant["br"],
        "tr": dominant["bl"],
        "bl": dominant["tr"],
        "br": dominant["tl"],
    }

    for name, slc in quadrant_slices.items():
        region = g[slc]
        mask = region == dominant[name]
        out[slc][mask] = swapped[name]

    return out


def _paint_rect(mask: np.ndarray, r0: int, c0: int, rh: int, cw: int) -> None:
    mask[r0 : r0 + rh, c0 : c0 + cw] = True


def _random_shape_mask(qh: int, qw: int) -> np.ndarray:
    area = qh * qw
    min_fill = max(4, area // 5)
    max_fill = max(min_fill, (area * 2) // 3)

    for _ in range(200):
        mask = np.zeros((qh, qw), dtype=bool)
        shape_type = random.randrange(4)

        if shape_type == 0:
            rh = random.randint(max(2, qh // 2), qh)
            cw = random.randint(max(2, qw // 2), qw)
            r0 = random.randint(0, qh - rh)
            c0 = random.randint(0, qw - cw)
            _paint_rect(mask, r0, c0, rh, cw)
        elif shape_type == 1:
            vrh = random.randint(max(2, qh // 2), qh)
            vcw = random.randint(1, max(1, qw // 3))
            hrh = random.randint(1, max(1, qh // 3))
            hcw = random.randint(max(2, qw // 2), qw)
            vr0 = random.randint(0, qh - vrh)
            vc0 = random.randint(0, qw - vcw)
            hr0 = random.randint(0, qh - hrh)
            hc0 = random.randint(0, qw - hcw)
            _paint_rect(mask, vr0, vc0, vrh, vcw)
            _paint_rect(mask, hr0, hc0, hrh, hcw)
        elif shape_type == 2:
            stem_h = random.randint(max(2, qh // 2), qh)
            stem_w = random.randint(1, max(1, qw // 3))
            foot_h = random.randint(1, max(1, qh // 3))
            foot_w = random.randint(max(2, qw // 2), qw)
            top = random.randint(0, qh - stem_h)
            left = random.randint(0, qw - stem_w)
            align_bottom = random.choice([True, False])
            if align_bottom:
                foot_r0 = top + stem_h - foot_h
            else:
                foot_r0 = top
            foot_c0 = random.randint(0, qw - foot_w)
            _paint_rect(mask, top, left, stem_h, stem_w)
            _paint_rect(mask, foot_r0, foot_c0, foot_h, foot_w)
        else:
            rh1 = random.randint(max(2, qh // 3), max(2, (2 * qh) // 3))
            cw1 = random.randint(max(2, qw // 3), max(2, (2 * qw) // 3))
            r01 = random.randint(0, qh - rh1)
            c01 = random.randint(0, qw - cw1)
            _paint_rect(mask, r01, c01, rh1, cw1)
            rh2 = random.randint(1, max(1, qh // 2))
            cw2 = random.randint(1, max(1, qw // 2))
            r02 = random.randint(max(0, r01 - rh2 + 1), min(qh - rh2, r01 + rh1 - 1))
            c02 = random.randint(max(0, c01 - cw2 + 1), min(qw - cw2, c01 + cw1 - 1))
            _paint_rect(mask, r02, c02, rh2, cw2)

        fill = int(mask.sum())
        if min_fill <= fill <= max_fill:
            return mask

    raise RuntimeError("failed to build a valid dominant mask")


def _paint_quadrant(region: np.ndarray, dominant_color: int, noise_palette: list[int]) -> None:
    qh, qw = region.shape
    mask = _random_shape_mask(qh, qw)
    region[mask] = dominant_color

    empty = [(int(r), int(c)) for r, c in np.argwhere(~mask)]
    random.shuffle(empty)
    dominant_count = int(mask.sum())
    max_noise = min(len(empty), max(0, dominant_count - 1), 6)
    noise_count = random.randint(0, max_noise)

    for idx in range(noise_count):
        r, c = empty[idx]
        region[r, c] = random.choice(noise_palette)


def _build_candidate() -> np.ndarray:
    h = random.randint(9, 20)
    w = random.randint(9, 20)
    cross_r = random.randint(3, h - 4)
    cross_c = random.randint(3, w - 4)

    grid = np.full((h, w), BG, dtype=int)
    grid[cross_r, :] = SEP
    grid[:, cross_c] = SEP

    dominant_colors = random.sample(COLOR_POOL, 4)
    quadrant_specs = [
        ((slice(0, cross_r), slice(0, cross_c)), dominant_colors[0]),
        ((slice(0, cross_r), slice(cross_c + 1, w)), dominant_colors[1]),
        ((slice(cross_r + 1, h), slice(0, cross_c)), dominant_colors[2]),
        ((slice(cross_r + 1, h), slice(cross_c + 1, w)), dominant_colors[3]),
    ]

    for slc, color in quadrant_specs:
        region = grid[slc]
        noise_palette = [c for c in COLOR_POOL if c != color]
        _paint_quadrant(region, color, noise_palette)

    return grid


def _is_valid_instance(grid: np.ndarray) -> bool:
    try:
        cross_r, cross_c = _find_separator_cross(grid)
        quadrants = [
            grid[:cross_r, :cross_c],
            grid[:cross_r, cross_c + 1 :],
            grid[cross_r + 1 :, :cross_c],
            grid[cross_r + 1 :, cross_c + 1 :],
        ]
        dominant = [_dominant_color(region) for region in quadrants]
        if len(set(dominant)) != 4:
            return False
        out = main(grid)
        return out.shape == grid.shape and np.any(out != grid)
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if _is_valid_instance(grid):
            return grid
    raise RuntimeError("failed to generate valid input")
