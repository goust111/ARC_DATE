import random
from collections import Counter

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
GRID_H_MIN, GRID_H_MAX = 12, 18
GRID_W_MIN, GRID_W_MAX = 10, 16
OBJECT_COUNT_CHOICES = (2, 3)
HEIGHT_RANGE = (3, 6)
WIDTH_RANGE = (3, 6)
GENERATION_ATTEMPTS = 600
FORCE_CLIP_PROB = 0.35
THIRD_SHARED_ROWS_PROB = 0.5


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        rows = sorted(int(r) for r in np.unique(coords[:, 0]))
        segments: dict[int, tuple[int, int]] = {}
        area = 0
        for r in rows:
            cols = sorted(int(c) for c in coords[coords[:, 0] == r, 1])
            assert len(cols) == 2, "each occupied row must have exactly two boundary pixels"
            c0, c1 = cols
            assert c0 < c1, "boundary pixels must be distinct"
            segments[int(r)] = (int(c0), int(c1))
            area += int(c1 - c0 + 1)
        objects.append(
            {
                "color": int(color),
                "rows": rows,
                "segments": segments,
                "area": int(area),
                "top": int(rows[0]),
                "bottom": int(rows[-1]),
            }
        )
    assert objects, "expected at least one outline"
    return objects


def _shifted_intervals(obj: dict, width: int, *, local_shear: bool) -> list[tuple[int, int, int]]:
    intervals: list[tuple[int, int, int]] = []
    for local_row, r in enumerate(obj["rows"]):
        c0, c1 = obj["segments"][int(r)]
        shift = int(local_row) if local_shear else int(r)
        lo = max(0, int(c0) - shift)
        hi = min(int(width) - 1, int(c1) - shift)
        if lo <= hi:
            intervals.append((int(r), int(lo), int(hi)))
    return intervals


def _compose(
    objects: list[dict],
    shape: tuple[int, int],
    *,
    local_shear: bool,
    prefer: str,
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BACKGROUND, dtype=int)
    if prefer == "largest":
        scores = np.full((h, w), -1, dtype=int)
        better = lambda area, cur: area > cur
    elif prefer == "smallest":
        scores = np.full((h, w), 10**9, dtype=int)
        better = lambda area, cur: area < cur
    else:
        raise ValueError(f"unsupported preference: {prefer}")

    for obj in objects:
        color = int(obj["color"])
        area = int(obj["area"])
        for r, c0, c1 in _shifted_intervals(obj, w, local_shear=local_shear):
            for c in range(int(c0), int(c1) + 1):
                if better(area, int(scores[r, c])):
                    scores[r, c] = area
                    out[r, c] = color
    return out


def _coverage(objects: list[dict], shape: tuple[int, int], *, local_shear: bool) -> list[list[list[int]]]:
    h, w = map(int, shape)
    cov = [[[] for _ in range(w)] for _ in range(h)]
    for idx, obj in enumerate(objects):
        for r, c0, c1 in _shifted_intervals(obj, w, local_shear=local_shear):
            for c in range(int(c0), int(c1) + 1):
                cov[r][c].append(idx)
    return cov


def _has_clipping(obj: dict, width: int, *, local_shear: bool) -> bool:
    for local_row, r in enumerate(obj["rows"]):
        c0, c1 = obj["segments"][int(r)]
        shift = int(local_row) if local_shear else int(r)
        if int(c0) - shift < 0 or int(c1) - shift >= int(width):
            return True
    return False


def _solve_from_objects(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    return _compose(objects, shape, local_shear=True, prefer="largest")


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each row span of every sparse outline, then left-shear each filled outline by its own row index and keep the largest-area color on overlaps.

    Input structure:
    - The background is black or otherwise the most common color.
    - There are 2-3 non-background monochrome outlines; for each outline color, every occupied row contains exactly two same-colored pixels marking that row's left and right boundary.
    - Different outlines may share rows, but before filling their row spans are separated in the original grid.

    Rule:
    - For each outline color, fill every occupied row from its left boundary pixel to its right boundary pixel, inclusive, and record the outline's total filled area.
    - Shear each filled outline independently: within that outline's own occupied-row order, its first row shifts left by 0, its second by 1, its third by 2, and so on. The output keeps the same height and width; pixels shifted past the left edge are clipped and new right-side cells become background.
    - If multiple sheared filled outlines cover the same output cell, keep the color of the outline whose pre-shear filled area is larger.

    Key insight:
    - The sparse boundary pixels only define row-wise fill spans; the final visible conflict pattern comes from combining those filled spans with outline-local shear offsets, and overlap winners must be decided using pre-shear area rather than local pixel evidence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _extract_objects(g)
    return _solve_from_objects(objects, g.shape)


def _make_rect_object(color: int, top: int, height: int, left: int, width: int) -> dict:
    rows = list(range(int(top), int(top + height)))
    segments = {int(r): (int(left), int(left + width - 1)) for r in rows}
    return {
        "color": int(color),
        "rows": rows,
        "segments": segments,
        "area": int(height * width),
        "top": int(top),
        "bottom": int(top + height - 1),
    }


def _row_overlap(a: dict, b: dict) -> bool:
    return not (int(a["bottom"]) < int(b["top"]) or int(b["bottom"]) < int(a["top"]))


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


def _input_disjoint(candidate: dict, others: list[dict]) -> bool:
    for other in others:
        for r in set(candidate["rows"]).intersection(other["rows"]):
            a0, a1 = candidate["segments"][int(r)]
            b0, b1 = other["segments"][int(r)]
            if _intervals_overlap(a0, a1, b0, b1):
                return False
    return True


def _pair_objects(force_clip: bool) -> tuple[list[dict], int, int] | None:
    for _ in range(300):
        h1 = random.randint(*HEIGHT_RANGE)
        h2 = random.randint(*HEIGHT_RANGE)
        w1 = random.randint(*WIDTH_RANGE)
        w2 = random.randint(*WIDTH_RANGE)
        if h1 * w1 == h2 * w2:
            continue

        delta = random.randint(1, min(h1, h2) - 1)
        gap = random.randint(0, delta - 1)
        top_margin = random.randint(0, 2)
        bottom_margin = random.randint(0, 2)
        base_height = max(h1, delta + h2)
        h = top_margin + base_height + bottom_margin
        if not (GRID_H_MIN <= h <= GRID_H_MAX):
            continue

        top1 = top_margin
        top2 = top1 + delta

        if force_clip:
            left2_choices = list(range(0, min(3, GRID_W_MAX)))
        else:
            left2_start = max(h2 - 1, 1)
            left2_choices = list(range(left2_start, left2_start + 4))
        random.shuffle(left2_choices)

        for left2 in left2_choices:
            left1 = left2 + w2 + gap
            min_w = left1 + w1 + 1
            if min_w > GRID_W_MAX:
                continue
            w_lo = max(GRID_W_MIN, min_w)
            w_hi = min(GRID_W_MAX, min_w + 2)
            if w_lo > w_hi:
                continue
            w = random.randint(w_lo, w_hi)

            obj1 = _make_rect_object(1, top1, h1, left1, w1)
            obj2 = _make_rect_object(2, top2, h2, left2, w2)
            objects = [obj1, obj2]

            if not _input_disjoint(obj1, [obj2]):
                continue

            local_out = _solve_from_objects(objects, (h, w))
            cov = _coverage(objects, (h, w), local_shear=True)
            if not any(len(cov[r][c]) >= 2 for r in range(h) for c in range(w)):
                continue

            if force_clip and not any(_has_clipping(obj, w, local_shear=True) for obj in objects):
                continue
            if not force_clip and any(_has_clipping(obj, w, local_shear=True) for obj in objects):
                continue

            small_out = _compose(objects, (h, w), local_shear=True, prefer="smallest")
            if np.array_equal(local_out, small_out):
                continue

            global_out = _compose(objects, (h, w), local_shear=False, prefer="largest")
            if np.array_equal(local_out, global_out):
                continue

            return objects, h, w
    return None


def _candidate_third_objects(h: int, w: int, others: list[dict], area_used: set[int], color: int) -> list[dict]:
    candidates: list[dict] = []
    for height in range(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1):
        for width in range(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1):
            area = int(height * width)
            if area in area_used:
                continue
            for top in range(0, h - height + 1):
                for left in range(0, w - width + 1):
                    obj = _make_rect_object(color, top, height, left, width)
                    if _input_disjoint(obj, others):
                        candidates.append(obj)
    random.shuffle(candidates)
    return candidates


def _add_third_object(objects: list[dict], h: int, w: int) -> list[dict] | None:
    area_used = {int(obj["area"]) for obj in objects}
    color = 3
    candidates = _candidate_third_objects(h, w, objects, area_used, color)
    want_shared_rows = random.random() < THIRD_SHARED_ROWS_PROB

    preferred = [
        obj
        for obj in candidates
        if any(_row_overlap(obj, other) for other in objects) == want_shared_rows
    ]
    pool = preferred if preferred else candidates
    for obj in pool:
        trial = objects + [obj]
        local_out = _solve_from_objects(trial, (h, w))
        if len(np.unique(local_out[local_out != BACKGROUND])) < 2:
            continue
        small_out = _compose(trial, (h, w), local_shear=True, prefer="smallest")
        if np.array_equal(local_out, small_out):
            continue
        return trial
    return None


def _render_input(objects: list[dict], shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for obj, color in zip(objects, colors):
        for r in obj["rows"]:
            c0, c1 = obj["segments"][int(r)]
            grid[int(r), int(c0)] = int(color)
            grid[int(r), int(c1)] = int(color)
    return grid


def _semantic_ok(inp: np.ndarray, *, expect_clip: bool) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    objects = _extract_objects(inp)
    h, w = map(int, inp.shape)
    if len(objects) not in OBJECT_COUNT_CHOICES:
        return False

    areas = [int(obj["area"]) for obj in objects]
    if len(set(areas)) != len(areas):
        return False

    local_out = _solve_from_objects(objects, (h, w))
    if np.array_equal(local_out, inp):
        return False

    cov = _coverage(objects, (h, w), local_shear=True)
    if not any(len(cov[r][c]) >= 2 for r in range(h) for c in range(w)):
        return False

    smaller = _compose(objects, (h, w), local_shear=True, prefer="smallest")
    if np.array_equal(local_out, smaller):
        return False

    global_out = _compose(objects, (h, w), local_shear=False, prefer="largest")
    if np.array_equal(local_out, global_out):
        return False

    clipped = any(_has_clipping(obj, w, local_shear=True) for obj in objects)
    if expect_clip != clipped:
        return False

    for obj in objects:
        if len(obj["rows"]) < 3:
            return False
        if not _shifted_intervals(obj, w, local_shear=True):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    force_clip = random.random() < FORCE_CLIP_PROB
    paired = _pair_objects(force_clip)
    if paired is None:
        return None
    objects, h, w = paired

    target_count = random.choice(OBJECT_COUNT_CHOICES)
    if target_count == 3:
        extended = _add_third_object(objects, h, w)
        if extended is None:
            return None
        objects = extended

    colors = random.sample(COLORS, len(objects))
    colored_objects = []
    for obj, color in zip(objects, colors):
        colored_objects.append(
            {
                "color": int(color),
                "rows": list(obj["rows"]),
                "segments": dict(obj["segments"]),
                "area": int(obj["area"]),
                "top": int(obj["top"]),
                "bottom": int(obj["bottom"]),
            }
        )

    inp = _render_input(colored_objects, (h, w), colors)
    if not _semantic_ok(inp, expect_clip=force_clip):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
