from __future__ import annotations

import numpy as np


BG = 0
RED = 2
PALETTE = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)

DIRS = ("up", "down", "left", "right")
DIR_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN = 12
GRID_MAX = 22
OVERLAP_GRID_MAX = 17
PATTERN_SIZE_RANGE = (2, 6)
SHAFT_LEN_RANGE = (1, 6)
OBJECT_COUNT_PROBS = [0.15, 0.35, 0.30, 0.20]
OVERLAP_PROB = 0.35
GEN_RETRIES = 300
PER_OBJECT_TRIES = 500


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr or dc
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _arrow_candidates(coords: np.ndarray) -> list[tuple[str, int, tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    candidates: list[tuple[str, int, tuple[int, int]]] = []

    if h == 3 and w >= 2:
        row_counts = [sum((rr, cc) in rel for cc in range(w)) for rr in range(h)]
        if row_counts == [1, w, 1]:
            col_counts = [sum((rr, cc) in rel for rr in range(h)) for cc in range(w)]
            left_rows = {rr for rr, cc in rel if cc == 0}
            right_rows = {rr for rr, cc in rel if cc == w - 1}
            if left_rows == {1} and col_counts[1] == 3:
                candidates.append(("left", w - 1, (r0 + 1, c1)))
            if right_rows == {1} and col_counts[-2] == 3:
                candidates.append(("right", w - 1, (r0 + 1, c0)))

    if w == 3 and h >= 2:
        col_counts = [sum((rr, cc) in rel for rr in range(h)) for cc in range(w)]
        if col_counts == [1, h, 1]:
            row_counts = [sum((rr, cc) in rel for cc in range(w)) for rr in range(h)]
            top_cols = {cc for rr, cc in rel if rr == 0}
            bottom_cols = {cc for rr, cc in rel if rr == h - 1}
            if top_cols == {1} and row_counts[1] == 3:
                candidates.append(("up", h - 1, (r1, c0 + 1)))
            if bottom_cols == {1} and row_counts[-2] == 3:
                candidates.append(("down", h - 1, (r0, c0 + 1)))

    return candidates


def _attached_cell(direction: str, start: tuple[int, int]) -> tuple[int, int]:
    sr, sc = start
    if direction == "right":
        return sr, sc - 1
    if direction == "left":
        return sr, sc + 1
    if direction == "down":
        return sr - 1, sc
    return sr + 1, sc


def _paste_clipped(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = out.shape
    ph, pw = patch.shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(h, top + ph)
    c1 = min(w, left + pw)
    if r0 >= r1 or c0 >= c1:
        return
    pr0 = r0 - top
    pc0 = c0 - left
    out[r0:r1, c0:c1] = patch[pr0 : pr0 + (r1 - r0), pc0 : pc0 + (c1 - c0)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    pattern_mask = (g != BG) & (g != RED)
    labels = np.full((h, w), -1, dtype=int)
    patterns: list[dict[str, object]] = []
    for idx, comp in enumerate(_components(pattern_mask, connectivity=4)):
        r0, c0, r1, c1 = _bbox(comp)
        labels[comp[:, 0], comp[:, 1]] = idx
        patterns.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
            }
        )

    actions: list[tuple[int, tuple[int, int, int, int], np.ndarray, int, int]] = []
    for comp in _components(g == RED, connectivity=4):
        matches: list[tuple[str, int, int]] = []
        for direction, length, start in _arrow_candidates(comp):
            ar, ac = _attached_cell(direction, start)
            if 0 <= ar < h and 0 <= ac < w and labels[ar, ac] >= 0:
                matches.append((direction, length, int(labels[ar, ac])))

        assert len(matches) == 1, "each arrow must resolve to exactly one adjacent pattern"
        direction, length, pattern_idx = matches[0]
        pattern = patterns[pattern_idx]
        span = int(pattern["w"]) if direction in ("left", "right") else int(pattern["h"])
        shift = length + span
        dr, dc = DIR_STEP[direction]
        actions.append(
            (
                length,
                pattern["bbox"],
                pattern["patch"],
                dr * shift,
                dc * shift,
            )
        )

    out = g.copy()
    out[out == RED] = BG
    actions.sort(key=lambda item: item[0])
    for _length, bbox, patch, dr, dc in actions:
        r0, c0, _r1, _c1 = bbox
        _paste_clipped(out, patch, r0 + dr, c0 + dc)
    return out


def _sample_pattern(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    color_count = int(rng.choice([1, 2, 3], p=[0.20, 0.40, 0.40]))
    colors = rng.choice(PALETTE, size=color_count, replace=False)
    mode = int(rng.integers(0, 6))
    pat = np.empty((h, w), dtype=int)

    if mode == 0 or color_count == 1:
        pat[:] = int(colors[0])
    elif mode == 1:
        for c in range(w):
            pat[:, c] = int(colors[c % color_count])
    elif mode == 2:
        for r in range(h):
            pat[r, :] = int(colors[r % color_count])
    elif mode == 3:
        for r in range(h):
            for c in range(w):
                pat[r, c] = int(colors[(r + c) % color_count])
    elif mode == 4:
        for r in range(h):
            for c in range(w):
                idx = ((r >= h // 2) * 2 + (c >= w // 2)) % color_count
                pat[r, c] = int(colors[idx])
    else:
        pat[:] = int(colors[0])
        for r in range(h):
            for c in range(w):
                if (r + c) % 2 == 0 or r == h // 2 or c == w // 2:
                    pat[r, c] = int(colors[(r * 3 + c) % color_count])

    return pat


def _rect_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _build_spec(
    top: int,
    left: int,
    pattern: np.ndarray,
    direction: str,
    length: int,
    attach: int,
) -> dict[str, object]:
    h, w = pattern.shape
    bottom = top + h - 1
    right = left + w - 1
    arrow_cells: list[tuple[int, int]] = []

    if direction == "right":
        row = attach
        arrow_cells.extend((row, c) for c in range(right + 1, right + length + 1))
        arrow_cells.extend([(row - 1, right + length), (row + 1, right + length), (row, right + length + 1)])
        copy_bbox = (top, left + w + length, bottom, right + w + length)
    elif direction == "left":
        row = attach
        arrow_cells.extend((row, c) for c in range(left - length, left))
        arrow_cells.extend([(row - 1, left - length), (row + 1, left - length), (row, left - length - 1)])
        copy_bbox = (top, left - (w + length), bottom, right - (w + length))
    elif direction == "down":
        col = attach
        arrow_cells.extend((r, col) for r in range(bottom + 1, bottom + length + 1))
        arrow_cells.extend([(bottom + length, col - 1), (bottom + length, col + 1), (bottom + length + 1, col)])
        copy_bbox = (top + h + length, left, bottom + h + length, right)
    else:
        col = attach
        arrow_cells.extend((r, col) for r in range(top - length, top))
        arrow_cells.extend([(top - length, col - 1), (top - length, col + 1), (top - length - 1, col)])
        copy_bbox = (top - (h + length), left, bottom - (h + length), right)

    return {
        "bbox": (top, left, bottom, right),
        "pattern": pattern,
        "direction": direction,
        "length": length,
        "arrow_cells": arrow_cells,
        "copy_bbox": copy_bbox,
    }


def _render_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = spec["pattern"]
        for r, c in spec["arrow_cells"]:
            grid[r, c] = RED
    return grid


def _render_expected(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = _render_input(shape, specs)
    out[out == RED] = BG
    for spec in sorted(specs, key=lambda item: int(item["length"])):
        r0, c0, _r1, _c1 = spec["copy_bbox"]
        _paste_clipped(out, spec["pattern"], r0, c0)
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    need_overlap = bool(rng.random() < OVERLAP_PROB)
    max_side = OVERLAP_GRID_MAX if need_overlap else GRID_MAX
    h = int(rng.integers(GRID_MIN, max_side + 1))
    w = int(rng.integers(GRID_MIN, max_side + 1))
    if need_overlap:
        obj_count = int(rng.integers(2, 5))
    else:
        obj_count = int(rng.choice([1, 2, 3, 4], p=OBJECT_COUNT_PROBS))

    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, object]] = []

    for _ in range(obj_count):
        accepted = None
        for _ in range(PER_OBJECT_TRIES):
            ph = int(rng.integers(*PATTERN_SIZE_RANGE))
            pw = int(rng.integers(*PATTERN_SIZE_RANGE))
            direction = str(rng.choice(DIRS))
            length = int(rng.integers(*SHAFT_LEN_RANGE))
            top = int(rng.integers(0, h - ph + 1))
            left = int(rng.integers(0, w - pw + 1))

            if direction in ("left", "right"):
                anchors = [r for r in range(top, top + ph) if 1 <= r <= h - 2]
            else:
                anchors = [c for c in range(left, left + pw) if 1 <= c <= w - 2]
            if not anchors:
                continue

            spec = _build_spec(
                top,
                left,
                _sample_pattern(rng, ph, pw),
                direction,
                length,
                int(rng.choice(anchors)),
            )
            cr0, cc0, cr1, cc1 = spec["copy_bbox"]
            if not (0 <= cr0 <= cr1 < h and 0 <= cc0 <= cc1 < w):
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in spec["arrow_cells"]):
                continue

            input_cells = _rect_cells(spec["bbox"]) + spec["arrow_cells"]
            if any(blocked[r, c] or reserved[r, c] for r, c in input_cells):
                continue
            if occupied[cr0 : cr1 + 1, cc0 : cc1 + 1].any():
                continue

            accepted = spec
            break

        if accepted is None:
            return None

        specs.append(accepted)
        input_cells = _rect_cells(accepted["bbox"]) + accepted["arrow_cells"]
        for r, c in input_cells:
            occupied[r, c] = True
            blocked[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
        cr0, cc0, cr1, cc1 = accepted["copy_bbox"]
        reserved[cr0 : cr1 + 1, cc0 : cc1 + 1] = True

    has_overlap = False
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a = specs[i]["copy_bbox"]
            b = specs[j]["copy_bbox"]
            if not _rect_overlap(a, b):
                continue
            if int(specs[i]["length"]) == int(specs[j]["length"]):
                return None
            has_overlap = True

    if need_overlap and not has_overlap:
        return None

    grid = _render_input((h, w), specs)
    expected = _render_expected((h, w), specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to find a valid layout")
