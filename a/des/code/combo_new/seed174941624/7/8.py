from collections import deque
import random

import numpy as np


BG = 0
GEN_RETRIES = 300
STRIP_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MOLD_COLOR = 5


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _label_components(mask: np.ndarray, conn8: bool):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps = []
    neigh = _neighbors8 if conn8 else _neighbors4
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or labels[r, c] != -1:
                continue
            cid = len(comps)
            q = deque([(r, c)])
            labels[r, c] = cid
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            comps.append(coords)
    return labels, comps


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _touches_border(coords, h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in coords)


def _find_molds_and_strips(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    labels, nonbg_comps = _label_components(g != BG, conn8=True)
    _, black_comps = _label_components(g == BG, conn8=False)

    mold_cavities = {}
    for comp in black_comps:
        if _touches_border(comp, h, w):
            continue
        adj = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] != -1:
                    adj.add(int(labels[nr, nc]))
        if len(adj) == 1:
            cid = next(iter(adj))
            mold_cavities.setdefault(cid, []).extend(comp)

    molds = []
    for cid, cavity in mold_cavities.items():
        bbox = _bbox(nonbg_comps[cid])
        molds.append({"cid": cid, "coords": nonbg_comps[cid], "cavity": cavity, "bbox": bbox})
    molds.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    mold_ids = {m["cid"] for m in molds}
    strips = []
    for cid, coords in enumerate(nonbg_comps):
        if cid in mold_ids:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        if r0 != r1:
            continue
        if len(coords) != c1 - c0 + 1:
            continue
        strips.append({"cid": cid, "coords": coords, "color": next(iter(colors)), "bbox": bbox})
    strips.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))

    return molds, strips


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use strip order to pour colors into closed mold cavities.

    Input structure:
    - Background is black.
    - The upper portion contains several grey one-pixel-thick closed outlines; each outline surrounds a black cavity.
    - The lower portion contains the same number of one-row colored horizontal strips, separated by black rows.

    Rule:
    - Identify each enclosed cavity and the outline component that surrounds it.
    - Sort the mold components from left to right and sort the colored strips from top to bottom.
    - Pair item k in the mold order with item k in the strip order, erase the original strip, and fill that mold's cavity with the strip color.
    - Keep the outline pixels unchanged and leave all remaining background pixels black.

    Key insight:
    - The correspondence is ordinal across different axes: horizontal order of molds maps to vertical order of strips.
    """
    g = np.asarray(grid, dtype=int)
    molds, strips = _find_molds_and_strips(g)
    assert molds and len(molds) == len(strips)

    out = g.copy()
    for strip in strips:
        for r, c in strip["coords"]:
            out[r, c] = BG
    for mold, strip in zip(molds, strips):
        for r, c in mold["cavity"]:
            out[r, c] = int(strip["color"])
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_partition(rng: random.Random, total: int, parts: int):
    out = [0] * parts
    for _ in range(total):
        out[rng.randrange(parts)] += 1
    return out


def _random_cavity_mask(rng: random.Random):
    for _ in range(100):
        height = rng.randint(3, 7)
        canvas_width = rng.randint(4, 8)
        intervals = []

        width = rng.randint(2, canvas_width - 1)
        start = rng.randint(0, canvas_width - width)
        intervals.append((start, start + width))

        for _row in range(1, height):
            prev_start, prev_end = intervals[-1]
            prev_width = prev_end - prev_start
            width_lo = max(2, prev_width - 1)
            width_hi = min(canvas_width, prev_width + 1)
            width = rng.randint(width_lo, width_hi)

            start_lo = max(0, prev_start - width + 1)
            start_hi = min(canvas_width - width, prev_end - 1)
            if start_lo > start_hi:
                start = max(0, min(canvas_width - width, prev_start))
            else:
                start = rng.randint(start_lo, start_hi)
            intervals.append((start, start + width))

        mask = np.zeros((height, canvas_width), dtype=bool)
        for r, (start, end) in enumerate(intervals):
            mask[r, start:end] = True
        mask = _crop_mask(mask)

        if mask.sum() < 8:
            continue

        lefts = []
        rights = []
        for r in range(mask.shape[0]):
            cols = np.flatnonzero(mask[r])
            lefts.append(int(cols[0]))
            rights.append(int(cols[-1]))
        if len(set(lefts)) == 1 and len(set(rights)) == 1:
            continue

        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        return mask
    return None


def _outline_from_cavity(cavity: np.ndarray):
    cavity = np.pad(cavity.astype(bool), 1, constant_values=False)
    solid = np.zeros_like(cavity, dtype=bool)
    for r, c in np.argwhere(cavity):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                solid[r + dr, c + dc] = True
    outline = solid & (~cavity)
    return outline, cavity


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int):
    for r, c in np.argwhere(mask):
        grid[top + r, left + c] = color


def _build_expected_output(base: np.ndarray, molds, strip_specs):
    out = base.copy()
    for strip in strip_specs:
        r = strip["row"]
        c0 = strip["start"]
        c1 = c0 + strip["length"]
        out[r, c0:c1] = BG
    for mold, strip in zip(molds, strip_specs):
        color = int(strip["color"])
        for r, c in mold["cavity_cells"]:
            out[r, c] = color
    return out


def _gen_attempt(rng: random.Random):
    count = rng.randint(2, 4)

    local_molds = []
    for _ in range(count):
        cavity = _random_cavity_mask(rng)
        if cavity is None:
            return None
        outline, cavity_local = _outline_from_cavity(cavity)
        cav_bbox = _bbox([(int(r), int(c)) for r, c in np.argwhere(cavity_local)])
        local_molds.append(
            {
                "outline": outline,
                "cavity": cavity_local,
                "height": outline.shape[0],
                "width": outline.shape[1],
                "cavity_width": cav_bbox[3] - cav_bbox[1] + 1,
            }
        )

    widths = [m["width"] for m in local_molds]
    heights = [m["height"] for m in local_molds]
    mold_gaps = [rng.randint(2, 4) for _ in range(count - 1)]
    min_width = 2 + sum(widths) + sum(mold_gaps) + 2
    if min_width > 30:
        return None
    width = rng.randint(max(16, min_width), 30)

    bar_gaps = [rng.randint(1, 2) for _ in range(count - 1)]
    upper_needed = max(heights) + 2
    lower_needed = count + sum(bar_gaps) + 2
    feasible_heights = [
        h
        for h in range(max(16, upper_needed + lower_needed), 31)
        if h // 2 >= upper_needed and h - h // 2 >= lower_needed
    ]
    if not feasible_heights:
        return None
    height = rng.choice(feasible_heights)
    split = height // 2

    grid = np.full((height, width), BG, dtype=int)

    horizontal_slack = width - (2 + sum(widths) + sum(mold_gaps) + 2)
    extras = _random_partition(rng, horizontal_slack, count + 1)
    x = 2 + extras[0]
    placed_molds = []
    for idx, mold in enumerate(local_molds):
        max_top = split - mold["height"] - 1
        if max_top < 1:
            return None
        top = rng.randint(1, max_top)
        left = x
        _place_mask(grid, mold["outline"], top, left, MOLD_COLOR)

        cavity_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mold["cavity"])]
        placed_molds.append(
            {
                "left": left,
                "top": top,
                "cavity_cells": cavity_cells,
                "outline": mold["outline"],
                "cavity": mold["cavity"],
            }
        )

        x += mold["width"]
        if idx < count - 1:
            x += mold_gaps[idx] + extras[idx + 1]

    colors = rng.sample(STRIP_COLORS, count)

    lower_rows = height - split
    base_lower = 2 + count + sum(bar_gaps)
    lower_slack = lower_rows - base_lower
    if lower_slack < 0:
        return None
    row_extras = _random_partition(rng, lower_slack, count + 1)
    row = split + 1 + row_extras[0]

    strip_specs = []
    for idx in range(count):
        min_len = min(width - 2, local_molds[idx]["cavity_width"] + 2)
        if min_len > width - 2:
            return None
        length = rng.randint(min_len, width - 2)
        start = rng.randint(1, width - length - 1)
        color = colors[idx]
        grid[row, start : start + length] = color
        strip_specs.append({"row": row, "start": start, "length": length, "color": color})
        if idx < count - 1:
            row += 1 + bar_gaps[idx] + row_extras[idx + 1]

    expected = _build_expected_output(grid, placed_molds, strip_specs)

    molds, strips = _find_molds_and_strips(grid)
    if len(molds) != count or len(strips) != count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
