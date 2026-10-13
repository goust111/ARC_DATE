from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GEN_BG = 8
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _dilate_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _find_bands(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    bands: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        rects: list[dict] = []
        for pts in comps:
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if len(pts) != h * w:
                continue
            orient = None
            if h in (2, 3) and w > h:
                orient = "H"
            elif w in (2, 3) and h > w:
                orient = "V"
            if orient is None:
                continue
            rects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "orient": orient,
                    "cells": set(pts),
                    "area": len(pts),
                }
            )
        if not rects:
            continue
        band = max(rects, key=lambda item: item["area"])
        markers = [
            (int(r), int(c))
            for r, c in np.argwhere(g == color)
            if (int(r), int(c)) not in band["cells"]
        ]
        bands.append(
            {
                "color": color,
                "bbox": band["bbox"],
                "orient": band["orient"],
                "markers": markers,
            }
        )
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace scattered side markers with centered protrusions grown from same-color bands.

    Input structure:
    - The background is a uniform azure-like field, usually also the dominant color.
    - Each non-background color forms one solid horizontal or vertical rectangular band with thickness 2-3.
    - Extra same-color single-pixel markers lie in the band's column span (for horizontal bands) or row span (for vertical bands), separated from the band body.

    Rule:
    - For each color, detect its solid band and collect same-color pixels outside that band as markers, then erase those markers to background.
    - A horizontal band counts markers above versus below its body; from the midpoint column of the band, grow an upward protrusion of the upper count and a downward protrusion of the lower count.
    - A vertical band counts markers left versus right of its body; from the midpoint row of the band, grow leftward and rightward protrusions with those counts.

    Key insight:
    - Marker positions only encode side membership and quantity; the output is produced by counting markers on each side and converting those counts into centered protrusion lengths.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)

    for band in _find_bands(g):
        color = int(band["color"])
        r0, c0, r1, c1 = map(int, band["bbox"])
        markers = [(int(r), int(c)) for r, c in band["markers"]]

        for r, c in markers:
            out[r, c] = bg

        if band["orient"] == "H":
            mid_c = (c0 + c1) // 2
            up = sum(1 for r, c in markers if c0 <= c <= c1 and r < r0)
            down = sum(1 for r, c in markers if c0 <= c <= c1 and r > r1)
            for step in range(1, up + 1):
                out[r0 - step, mid_c] = color
            for step in range(1, down + 1):
                out[r1 + step, mid_c] = color
        else:
            mid_r = (r0 + r1) // 2
            left = sum(1 for r, c in markers if r0 <= r <= r1 and c < c0)
            right = sum(1 for r, c in markers if r0 <= r <= r1 and c > c1)
            for step in range(1, left + 1):
                out[mid_r, c0 - step] = color
            for step in range(1, right + 1):
                out[mid_r, c1 + step] = color

    return out


def _free_run(blocked: np.ndarray, start: tuple[int, int], dr: int, dc: int) -> int:
    h, w = blocked.shape
    r, c = start
    length = 0
    while 0 <= r < h and 0 <= c < w and not blocked[r, c]:
        length += 1
        r += dr
        c += dc
    return length


def _sample_isolated(candidates: list[tuple[int, int]], k: int) -> list[tuple[int, int]] | None:
    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    banned: set[tuple[int, int]] = set()
    for cell in candidates:
        if cell in banned:
            continue
        picked.append(cell)
        r, c = cell
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                banned.add((nr, nc))
        if len(picked) == k:
            return picked
    return None


def _choose_side_counts(max_a: int, max_b: int) -> list[tuple[int, int]]:
    branch_order = ["a_only", "b_only", "both"]
    random.shuffle(branch_order)
    results: list[tuple[int, int]] = []
    for branch in branch_order:
        if branch == "a_only" and max_a >= 1:
            group = [(a, 0) for a in range(1, max_a + 1)]
            random.shuffle(group)
            results.extend(group)
        elif branch == "b_only" and max_b >= 1:
            group = [(0, b) for b in range(1, max_b + 1)]
            random.shuffle(group)
            results.extend(group)
        elif branch == "both" and max_a >= 1 and max_b >= 1:
            group = [(a, b) for a in range(1, max_a + 1) for b in range(1, max_b + 1)]
            random.shuffle(group)
            results.extend(group)
    return results


def _side_candidates(
    blocked: np.ndarray,
    orient: str,
    bbox: tuple[int, int, int, int],
) -> dict[str, list[tuple[int, int]]]:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    if orient == "H":
        mid_c = (c0 + c1) // 2
        top = [
            (r, c)
            for r in range(0, r0 - 1)
            for c in range(c0, c1 + 1)
            if abs(c - mid_c) > 1 and not blocked[r, c]
        ]
        bottom = [
            (r, c)
            for r in range(r1 + 2, h)
            for c in range(c0, c1 + 1)
            if abs(c - mid_c) > 1 and not blocked[r, c]
        ]
        return {"neg": top, "pos": bottom}
    mid_r = (r0 + r1) // 2
    left = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(0, c0 - 1)
        if abs(r - mid_r) > 1 and not blocked[r, c]
    ]
    right = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c1 + 2, w)
        if abs(r - mid_r) > 1 and not blocked[r, c]
    ]
    return {"neg": left, "pos": right}


def _reserve_structure(
    blocked: np.ndarray,
    orient: str,
    bbox: tuple[int, int, int, int],
    count_neg: int,
    count_pos: int,
    markers_neg: list[tuple[int, int]],
    markers_pos: list[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    cells.update(markers_neg)
    cells.update(markers_pos)

    if orient == "H":
        mid_c = (c0 + c1) // 2
        for step in range(1, count_neg + 1):
            rr = r0 - step
            if rr < 0 or blocked[rr, mid_c]:
                return None
            cells.add((rr, mid_c))
        for step in range(1, count_pos + 1):
            rr = r1 + step
            if rr >= h or blocked[rr, mid_c]:
                return None
            cells.add((rr, mid_c))
    else:
        mid_r = (r0 + r1) // 2
        for step in range(1, count_neg + 1):
            cc = c0 - step
            if cc < 0 or blocked[mid_r, cc]:
                return None
            cells.add((mid_r, cc))
        for step in range(1, count_pos + 1):
            cc = c1 + step
            if cc >= w or blocked[mid_r, cc]:
                return None
            cells.add((mid_r, cc))
    return cells


def _place_band(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    orient: str,
) -> bool:
    h, w = grid.shape
    thickness = random.choice([2, 3])
    max_len = min(13, (w - 3) if orient == "H" else (h - 3))
    lengths = [v for v in (5, 7, 9, 11, 13) if v <= max_len]
    if not lengths:
        return False

    for _ in range(160):
        span = random.choice(lengths)
        if orient == "H":
            band_h, band_w = thickness, span
            if h - band_h - 3 < 2 or w - band_w - 2 < 1:
                return False
            r0 = random.randint(2, h - band_h - 3)
            c0 = random.randint(1, w - band_w - 2)
        else:
            band_h, band_w = span, thickness
            if h - band_h - 2 < 1 or w - band_w - 3 < 2:
                return False
            r0 = random.randint(1, h - band_h - 2)
            c0 = random.randint(2, w - band_w - 3)
        r1 = r0 + band_h - 1
        c1 = c0 + band_w - 1

        band_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if any(blocked[r, c] for r, c in band_cells):
            continue

        if orient == "H":
            run_neg = min(4, _free_run(blocked, (r0 - 1, (c0 + c1) // 2), -1, 0))
            run_pos = min(4, _free_run(blocked, (r1 + 1, (c0 + c1) // 2), 1, 0))
        else:
            run_neg = min(4, _free_run(blocked, ((r0 + r1) // 2, c0 - 1), 0, -1))
            run_pos = min(4, _free_run(blocked, ((r0 + r1) // 2, c1 + 1), 0, 1))

        side_cands = _side_candidates(blocked, orient, (r0, c0, r1, c1))
        max_neg = min(run_neg, 4)
        max_pos = min(run_pos, 4)

        for count_neg, count_pos in _choose_side_counts(max_neg, max_pos):
            neg_cells = _sample_isolated(list(side_cands["neg"]), count_neg) if count_neg else []
            if count_neg and neg_cells is None:
                continue
            pos_cells = _sample_isolated(list(side_cands["pos"]), count_pos) if count_pos else []
            if count_pos and pos_cells is None:
                continue
            reserved = _reserve_structure(
                blocked,
                orient,
                (r0, c0, r1, c1),
                count_neg,
                count_pos,
                neg_cells,
                pos_cells,
            )
            if reserved is None:
                continue
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            for rr, cc in neg_cells + pos_cells:
                grid[rr, cc] = color
            block_rows, block_cols = zip(*_dilate_cells(reserved, h, w))
            blocked[block_rows, block_cols] = True
            return True

    return False


def _analyze_validity(grid: np.ndarray, expected_bands: int) -> bool:
    bands = _find_bands(grid)
    if len(bands) != expected_bands:
        return False
    for band in bands:
        r0, c0, r1, c1 = map(int, band["bbox"])
        seen = set()
        for r, c in band["markers"]:
            if (r, c) in seen:
                return False
            seen.add((r, c))
            if band["orient"] == "H":
                if not (c0 <= c <= c1 and (r < r0 or r > r1)):
                    return False
            else:
                if not (r0 <= r <= r1 and (c < c0 or c > c1)):
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_bands = random.randint(2, 4)
    n_horizontal = random.randint(1, n_bands - 1)
    orientations = ["H"] * n_horizontal + ["V"] * (n_bands - n_horizontal)
    random.shuffle(orientations)

    h = random.randint(20, 30)
    w = random.randint(20, 30)
    grid = np.full((h, w), GEN_BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = random.sample(GEN_COLORS, n_bands)
    for color, orient in zip(colors, orientations):
        if not _place_band(grid, blocked, color, orient):
            return None

    if _get_bg(grid) != GEN_BG:
        return None
    if not _analyze_validity(grid, n_bands):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
