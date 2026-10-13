import math
import random

import numpy as np


BG = 0
GREY = 5
YELLOW = 4
SEED_COLORS = (1, 2, 3, 6, 7, 8, 9)

GEN_RETRIES = 400
GRID_MIN = 10
GRID_MAX = 30
FRAME_COUNT_CHOICES = (1, 2, 3, 4)
FRAME_COUNT_WEIGHTS = (2, 3, 2, 1)
FRAME_STYLES = ("single", "flat", "monotone", "zigzag")
SHADOW_STYLES = {"monotone", "zigzag"}


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components4(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_h < 6 or box_w < 5:
            continue

        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(box[0, :] == GREY):
            continue
        if not np.all(box[-1, :] == GREY):
            continue
        if not np.all(box[:, 0] == GREY):
            continue
        if not np.all(box[:, -1] == GREY):
            continue
        if np.any(box[1:-1, 1:-1] == GREY):
            continue
        if len(comp) != 2 * (box_h + box_w) - 4:
            continue

        seed_row = box[1, 1:-1]
        if seed_row.size < 3 or np.any(seed_row == BG) or np.any(seed_row == GREY):
            continue
        frames.append((r0, c0, r1, c1))
    return frames


def _segments(seed_row: np.ndarray) -> list[tuple[int, int, int, int]]:
    segments: list[tuple[int, int, int, int]] = []
    start = 0
    while start < len(seed_row):
        stop = start + 1
        while stop < len(seed_row) and int(seed_row[stop]) == int(seed_row[start]):
            stop += 1
        segments.append((start, stop, int(seed_row[start]), stop - start))
        start = stop
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed-row color segment downward with height equal to its run length.
    At every step boundary between unequal heights, paint a yellow shadow stripe
    on the shorter side.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _find_frames(g):
        inner_top = r0 + 1
        inner_left = c0 + 1
        inner_bottom = r1 - 1
        inner_right = c1 - 1
        inner_h = inner_bottom - inner_top + 1

        out[inner_top : inner_bottom + 1, inner_left : inner_right + 1] = BG
        seed = g[inner_top, inner_left : inner_right + 1]
        segs = _segments(seed)

        for start, stop, color, height in segs:
            if height > inner_h:
                raise AssertionError("seed run length exceeds rectangle interior height")
            out[inner_top : inner_top + height, inner_left + start : inner_left + stop] = color

        for idx in range(len(segs) - 1):
            left_start, left_stop, _, left_h = segs[idx]
            right_start, right_stop, _, right_h = segs[idx + 1]
            if left_h == right_h:
                continue
            if left_h < right_h:
                shade_col = inner_left + left_stop - 1
                out[inner_top + left_h : inner_top + right_h, shade_col] = YELLOW
            else:
                shade_col = inner_left + right_start
                out[inner_top + right_h : inner_top + left_h, shade_col] = YELLOW

    return out


def _sample_lengths(style: str, inner_h: int, rng: random.Random) -> list[int] | None:
    for _ in range(100):
        if style == "single":
            max_len = min(inner_h, 8)
            if max_len < 3:
                return None
            return [rng.randint(3, max_len)]

        if style == "flat":
            base = rng.randint(2, min(inner_h, 4))
            min_k = 2
            max_k = min(5, 12 // base)
            if max_k >= min_k:
                return [base] * rng.randint(min_k, max_k)

        if style == "monotone":
            k = rng.randint(2, min(4, inner_h))
            vals = sorted(rng.sample(range(1, inner_h + 1), k))
            if sum(vals) <= 12:
                if rng.random() < 0.5:
                    vals.reverse()
                return vals

        if style == "zigzag":
            if inner_h < 2:
                return None
            k = rng.randint(3, 5)
            low = rng.randint(1, inner_h - 1)
            high = rng.randint(low + 1, inner_h)
            start_high = rng.random() < 0.5
            vals = []
            for idx in range(k):
                use_high = (idx % 2 == 0) == start_high
                vals.append(high if use_high else low)
            if sum(vals) <= 12:
                return vals
    return None


def _choose_styles(frame_count: int, rng: random.Random) -> list[str]:
    styles = rng.choices(FRAME_STYLES, k=frame_count)
    if rng.random() < 0.75 and not any(style in SHADOW_STYLES for style in styles):
        styles[rng.randrange(frame_count)] = rng.choice(tuple(SHADOW_STYLES))
    if frame_count > 1 and rng.random() < 0.35 and "flat" not in styles:
        styles[rng.randrange(frame_count)] = "flat"
    if frame_count > 1 and rng.random() < 0.25 and "single" not in styles:
        styles[rng.randrange(frame_count)] = "single"
    return styles


def _make_frame_spec(style: str, rng: random.Random) -> dict | None:
    for _ in range(100):
        inner_h = rng.randint(4, 8)
        lengths = _sample_lengths(style, inner_h, rng)
        if not lengths:
            continue
        inner_w = sum(lengths)
        if not (3 <= inner_w <= 12):
            continue
        if max(lengths) > inner_h:
            continue
        colors = rng.sample(SEED_COLORS, len(lengths))
        return {
            "style": style,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "box_h": inner_h + 2,
            "box_w": inner_w + 2,
            "lengths": lengths,
            "colors": colors,
        }
    return None


def _place_specs(specs: list[dict], h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["box_h"] * specs[idx]["box_w"],
        reverse=True,
    )

    for idx in order:
        box_h = specs[idx]["box_h"]
        box_w = specs[idx]["box_w"]
        if h - box_h - 1 < 1 or w - box_w - 1 < 1:
            return None

        placed = False
        for _ in range(400):
            r = rng.randint(1, h - box_h - 1)
            c = rng.randint(1, w - box_w - 1)
            if occupied[r - 1 : r + box_h + 1, c - 1 : c + box_w + 1].any():
                continue
            positions[idx] = (r, c)
            occupied[r - 1 : r + box_h + 1, c - 1 : c + box_w + 1] = True
            placed = True
            break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _draw_frame(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    box_h = spec["box_h"]
    box_w = spec["box_w"]
    bottom = top + box_h - 1
    right = left + box_w - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY

    cursor = left + 1
    seed_row = top + 1
    for color, length in zip(spec["colors"], spec["lengths"]):
        grid[seed_row, cursor : cursor + length] = color
        cursor += length


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    styles = _choose_styles(frame_count, rng)

    specs: list[dict] = []
    for style in styles:
        spec = _make_frame_spec(style, rng)
        if spec is None:
            return None
        specs.append(spec)

    max_h = max(spec["box_h"] for spec in specs)
    max_w = max(spec["box_w"] for spec in specs)
    total_area = sum((spec["box_h"] + 2) * (spec["box_w"] + 2) for spec in specs)
    base = int(math.ceil(math.sqrt(total_area)))
    min_h = max(GRID_MIN, max_h + 2, base)
    min_w = max(GRID_MIN, max_w + 2, base)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    for _ in range(60):
        h = rng.randint(min_h, GRID_MAX)
        w = rng.randint(min_w, GRID_MAX)
        positions = _place_specs(specs, h, w, rng)
        if positions is None or len(positions) != len(specs):
            continue

        grid = np.zeros((h, w), dtype=int)
        for spec, (top, left) in zip(specs, positions):
            _draw_frame(grid, top, left, spec)

        found_frames = _find_frames(grid)
        if len(found_frames) != len(specs):
            continue
        if np.array_equal(main(grid), grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
