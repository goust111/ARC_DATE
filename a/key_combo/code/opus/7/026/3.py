from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 5
MAIN_PALETTE = (1, 2, 3, 4)
NOISE_PALETTE = (6, 7, 8, 9)

GRID_H_RANGE = (15, 21)
GRID_W_RANGE = (15, 21)
QUADRANT_MIN_SIZE = 5
MAX_GEN_ATTEMPTS = 400
MAX_REGION_ATTEMPTS = 120
MAX_MASK_ATTEMPTS = 80


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    solid_rows = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if np.all(grid[r, :] == int(grid[r, 0]))
    ]
    solid_cols = [
        (c, int(grid[0, c]))
        for c in range(w)
        if np.all(grid[:, c] == int(grid[0, c]))
    ]
    crosses = [
        (r, c, row_color)
        for r, row_color in solid_rows
        for c, col_color in solid_cols
        if row_color == col_color
    ]
    assert len(crosses) == 1, "expected exactly one solid same-color cross"
    cross_r, cross_c, sep = crosses[0]
    assert 0 < cross_r < h - 1 and 0 < cross_c < w - 1
    return int(cross_r), int(cross_c), int(sep)


def _quadrants(shape: tuple[int, int], cross_r: int, cross_c: int) -> list[tuple[slice, slice]]:
    h, w = shape
    return [
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, w)),
        (slice(cross_r + 1, h), slice(0, cross_c)),
        (slice(cross_r + 1, h), slice(cross_c + 1, w)),
    ]


def _quadrant_info(region: np.ndarray) -> dict[str, object]:
    active = region[(region != BG) & (region != SEP)]
    counts = Counter(map(int, active.tolist()))
    ranked = counts.most_common()
    assert ranked
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    main_color = int(ranked[0][0])
    main_mask = region == main_color
    noise_mask = (region != BG) & (region != SEP) & (region != main_color)
    box = _bbox(main_mask)
    assert box is not None
    return {
        "main_color": main_color,
        "main_mask": main_mask,
        "noise_mask": noise_mask,
        "noise_count": int(noise_mask.sum()),
        "noise_colors": [int(x) for x in region[noise_mask].tolist()],
        "bbox": box,
        "area": int(main_mask.sum()),
    }


def _apply_local_erasure(region: np.ndarray, *, erase_width: int, origin_col: int) -> np.ndarray:
    info = _quadrant_info(region)
    out = region.copy()
    main_color = int(info["main_color"])
    noise_mask = np.asarray(info["noise_mask"], dtype=bool)
    cols = np.arange(region.shape[1])[None, :]
    main_mask = region == main_color
    out[noise_mask] = BG
    out[main_mask & (cols >= origin_col) & (cols < origin_col + erase_width)] = BG
    return out


def _solve_variant(grid: np.ndarray, mode: str, fixed_width: int | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cross_r, cross_c, _sep = _find_cross(g)
    out = g.copy()
    for rows, cols in _quadrants(g.shape, cross_r, cross_c):
        region = g[rows, cols]
        info = _quadrant_info(region)
        if mode == "true":
            erase_width = int(info["noise_count"])
            origin_col = int(info["bbox"][1])
        elif mode == "fixed":
            assert fixed_width is not None
            erase_width = int(fixed_width)
            origin_col = int(info["bbox"][1])
        elif mode == "quadrant_left":
            erase_width = int(info["noise_count"])
            origin_col = 0
        elif mode == "distinct_noise":
            erase_width = len(set(int(x) for x in info["noise_colors"]))
            origin_col = int(info["bbox"][1])
        elif mode == "noise_only":
            erase_width = 0
            origin_col = int(info["bbox"][1])
        else:
            raise ValueError(f"unknown mode: {mode}")
        out[rows, cols] = _apply_local_erasure(region, erase_width=erase_width, origin_col=origin_col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove quadrant noise and erase each dominant object from its local left edge by the quadrant's noise count.

    Input structure:
    - The grid has black background plus one solid gray row and one solid gray column that form a full cross.
    - The cross splits the grid into four nonempty rectangular quadrants.
    - Each quadrant contains one dominant non-gray, non-black color object and a few off-color noise pixels.

    Rule:
    - Detect the unique solid same-color row and column to locate the gray separator cross.
    - In each quadrant, find the unique most frequent non-black, non-gray color as the dominant object color.
    - Count every other non-black, non-gray pixel in that quadrant as noise; erase those noise pixels to black.
    - Compute the dominant object's local bounding box, then erase dominant-color pixels in the leftmost N bbox columns, where N is that quadrant's noise-pixel count.
    - Keep the separator cross unchanged.

    Key insight:
    - Noise plays two roles at once: it must be filtered out, and its pixel count determines how much of the local dominant object is removed from the left.
    """
    return _solve_variant(grid, mode="true")


def _carve_once(mask: np.ndarray) -> np.ndarray:
    trial = mask.copy()
    h, w = trial.shape
    op = random.choice(("top_notch", "bottom_notch", "left_bite", "right_bite", "inner_hole"))
    if op == "top_notch" and h >= 2 and w >= 4:
        depth = random.randint(1, min(2, h - 1))
        width = random.randint(1, min(3, w - 2))
        start = random.randint(1, w - width - 1)
        trial[:depth, start : start + width] = False
    elif op == "bottom_notch" and h >= 2 and w >= 4:
        depth = random.randint(1, min(2, h - 1))
        width = random.randint(1, min(3, w - 2))
        start = random.randint(1, w - width - 1)
        trial[h - depth :, start : start + width] = False
    elif op == "left_bite" and h >= 4 and w >= 2:
        bite_h = random.randint(1, min(3, h - 1))
        bite_w = random.randint(1, min(2, w - 1))
        start_r = random.randint(1, h - bite_h)
        trial[start_r : start_r + bite_h, :bite_w] = False
    elif op == "right_bite" and h >= 4 and w >= 2:
        bite_h = random.randint(1, min(3, h - 1))
        bite_w = random.randint(1, min(2, w - 1))
        start_r = random.randint(0, h - bite_h - 1)
        trial[start_r : start_r + bite_h, w - bite_w :] = False
    elif op == "inner_hole" and h >= 4 and w >= 5:
        hole_h = 1
        hole_w = random.randint(1, min(2, w - 2))
        start_r = random.randint(1, h - hole_h - 1)
        start_c = random.randint(1, w - hole_w - 1)
        trial[start_r : start_r + hole_h, start_c : start_c + hole_w] = False
    return trial


def _dense_object_mask(h: int, w: int) -> np.ndarray:
    for _ in range(MAX_MASK_ATTEMPTS):
        mask = np.ones((h, w), dtype=bool)
        for _ in range(random.randint(1, 4)):
            trial = _carve_once(mask)
            if not trial.any(axis=0).all():
                continue
            if not trial.any(axis=1).all():
                continue
            if not _is_connected(trial):
                continue
            if int(trial.sum()) < max(w + 2, (h * w * 2) // 3):
                continue
            mask = trial
        if _is_connected(mask) and mask.any(axis=0).all() and mask.any(axis=1).all():
            return mask
    return np.ones((h, w), dtype=bool)


def _noise_values(noise_count: int, repeated: bool) -> list[int]:
    if noise_count == 1:
        return [random.choice(NOISE_PALETTE)]
    if repeated:
        repeated_color = random.choice(NOISE_PALETTE)
        values = [repeated_color] * noise_count
        if noise_count == 3 and random.random() < 0.5:
            other = random.choice([c for c in NOISE_PALETTE if c != repeated_color])
            values[-1] = other
        random.shuffle(values)
        return values
    if noise_count <= len(NOISE_PALETTE):
        return random.sample(NOISE_PALETTE, noise_count)
    return [random.choice(NOISE_PALETTE) for _ in range(noise_count)]


def _build_quadrant(
    qh: int,
    qw: int,
    main_color: int,
    noise_count: int,
    *,
    repeated_noise: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    quadrant_area = qh * qw
    for _ in range(MAX_REGION_ATTEMPTS):
        max_top = min(1, qh - 3)
        max_bottom = min(1, qh - 3)
        max_left = min(2, qw - 4)
        max_right = min(2, qw - 4)
        top = random.randint(0, max_top)
        bottom = random.randint(0, max_bottom)
        left = random.randint(0, max_left)
        right = random.randint(0, max_right)
        bbox_h = qh - top - bottom
        bbox_w = qw - left - right
        if bbox_h < 3 or bbox_w < max(4, noise_count + 1):
            continue

        mask = _dense_object_mask(bbox_h, bbox_w)
        object_area = int(mask.sum())
        if object_area <= quadrant_area // 2:
            continue

        region_mask = np.zeros((qh, qw), dtype=bool)
        region_mask[top : top + bbox_h, left : left + bbox_w] = mask
        free = np.argwhere(~region_mask)
        if len(free) < noise_count:
            continue

        region = np.zeros((qh, qw), dtype=int)
        region[region_mask] = main_color
        chosen = random.sample(range(len(free)), noise_count)
        values = _noise_values(noise_count, repeated_noise)
        for idx, value in zip(chosen, values):
            r, c = free[idx]
            region[int(r), int(c)] = int(value)

        info = _quadrant_info(region)
        bbox = info["bbox"]
        width = int(bbox[3] - bbox[1] + 1)
        cols = np.arange(qw)[None, :]
        main_mask = np.asarray(info["main_mask"], dtype=bool)
        erased_band = main_mask & (cols >= bbox[1]) & (cols < bbox[1] + noise_count)
        kept_band = main_mask & (cols >= bbox[1] + noise_count)
        if width < 4 or noise_count >= width:
            continue
        if not erased_band.any() or not kept_band.any():
            continue
        return region, {"left_margin": left, "bbox_left": int(bbox[1])}
    raise RuntimeError("failed to build a valid quadrant")


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if grid.shape != out.shape or np.array_equal(grid, out):
        return False

    try:
        cross_r, cross_c, sep = _find_cross(grid)
    except AssertionError:
        return False
    if sep != SEP:
        return False
    if not np.array_equal(out[cross_r, :], grid[cross_r, :]):
        return False
    if not np.array_equal(out[:, cross_c], grid[:, cross_c]):
        return False

    noise_counts: list[int] = []
    left_margins: list[int] = []
    has_repeated_noise = False

    for rows, cols in _quadrants(grid.shape, cross_r, cross_c):
        region_in = grid[rows, cols]
        region_out = out[rows, cols]
        info = _quadrant_info(region_in)
        main_mask = np.asarray(info["main_mask"], dtype=bool)
        noise_mask = np.asarray(info["noise_mask"], dtype=bool)
        bbox = tuple(int(x) for x in info["bbox"])
        width = bbox[3] - bbox[1] + 1
        noise_count = int(info["noise_count"])
        cols_local = np.arange(region_in.shape[1])[None, :]
        erased_band = main_mask & (cols_local >= bbox[1]) & (cols_local < bbox[1] + noise_count)
        kept_band = main_mask & (cols_local >= bbox[1] + noise_count)

        if not _is_connected(main_mask):
            return False
        if int(info["area"]) <= region_in.size // 2:
            return False
        if not (1 <= noise_count <= 3):
            return False
        if width < 4 or noise_count >= width:
            return False
        if not erased_band.any() or not kept_band.any():
            return False
        if np.any(region_out[noise_mask] != BG):
            return False
        if np.any(region_out[erased_band] != BG):
            return False
        if not np.array_equal(region_out[kept_band], region_in[kept_band]):
            return False
        if not np.array_equal(region_out[region_in == BG], region_in[region_in == BG]):
            return False

        noise_counts.append(noise_count)
        left_margins.append(bbox[1])
        if noise_count >= 2 and len(set(int(x) for x in info["noise_colors"])) < noise_count:
            has_repeated_noise = True

    if not {1, 2, 3}.issubset(set(noise_counts)):
        return False
    if 0 not in left_margins or max(left_margins) == 0:
        return False
    if not has_repeated_noise:
        return False

    if np.array_equal(_solve_variant(grid, mode="quadrant_left"), out):
        return False
    if np.array_equal(_solve_variant(grid, mode="distinct_noise"), out):
        return False
    if np.array_equal(_solve_variant(grid, mode="noise_only"), out):
        return False
    for width in (1, 2, 3):
        if np.array_equal(_solve_variant(grid, mode="fixed", fixed_width=width), out):
            return False
    return True


def _candidate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    cross_r = random.randint(QUADRANT_MIN_SIZE, h - QUADRANT_MIN_SIZE - 1)
    cross_c = random.randint(QUADRANT_MIN_SIZE, w - QUADRANT_MIN_SIZE - 1)
    quadrants = _quadrants((h, w), cross_r, cross_c)
    noise_counts = [1, 2, 3, random.choice((1, 2, 3))]
    random.shuffle(noise_counts)
    repeated_idx = random.choice([i for i, count in enumerate(noise_counts) if count >= 2])
    main_colors = random.sample(MAIN_PALETTE, 4)

    grid = np.zeros((h, w), dtype=int)
    for idx, ((rows, cols), main_color, noise_count) in enumerate(zip(quadrants, main_colors, noise_counts)):
        qh = rows.stop - rows.start
        qw = cols.stop - cols.start
        region, _meta = _build_quadrant(
            qh,
            qw,
            main_color,
            noise_count,
            repeated_noise=idx == repeated_idx,
        )
        grid[rows, cols] = region

    grid[cross_r, :] = SEP
    grid[:, cross_c] = SEP
    if not _validate_candidate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
