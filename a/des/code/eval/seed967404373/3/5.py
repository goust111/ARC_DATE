import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    if connectivity == 4:
        neighbors = DIRS4
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_strip_height(grid: np.ndarray) -> int:
    h = grid.shape[0]
    assert h >= 2 and np.all(grid[0] != BG), "top sample strip must exist"
    if h >= 2 and np.all(grid[1] != BG) and np.array_equal(grid[0], grid[1]):
        return 2
    return 1


def _find_rectangles(grid: np.ndarray, strip_h: int) -> tuple[list[dict[str, int]], dict[tuple[int, int], int]]:
    rects: list[dict[str, int]] = []
    owner: dict[tuple[int, int], int] = {}
    colors = [int(c) for c in np.unique(grid[strip_h:]) if int(c) != BG]
    for color in colors:
        mask = grid == color
        mask[:strip_h, :] = False
        for comp in _components(mask, connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 3 or ww < 3:
                continue
            if len(comp) != hh * ww:
                continue
            rect_id = len(rects)
            rects.append({"id": rect_id, "r0": r0, "c0": c0, "r1": r1, "c1": c1})
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    owner[(rr, cc)] = rect_id
    return rects, owner


def _parse_arrow(
    cells: list[tuple[int, int]],
    rects: list[dict[str, int]],
    owner: dict[tuple[int, int], int],
) -> tuple[int, list[tuple[int, int]]] | None:
    cell_set = set(cells)
    rect_by_id = {rect["id"]: rect for rect in rects}
    for tip in cell_set:
        tr, tc = tip
        for dr, dc in DIRS4:
            perp = (dc, -dr)
            back = (tr - dr, tc - dc)
            arm1 = (back[0] + perp[0], back[1] + perp[1])
            arm2 = (back[0] - perp[0], back[1] - perp[1])
            if cell_set != {tip, arm1, arm2}:
                continue
            target = (tr + dr, tc + dc)
            rect_id = owner.get(target)
            if rect_id is None:
                continue
            rect = rect_by_id[rect_id]
            rr, cc = target
            if rr not in (rect["r0"], rect["r1"]) and cc not in (rect["c0"], rect["c1"]):
                continue
            return rect_id, list(cell_set)
    return None


def _find_arrows(
    grid: np.ndarray,
    strip_h: int,
    rects: list[dict[str, int]],
    owner: dict[tuple[int, int], int],
) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    colors = [int(c) for c in np.unique(grid[strip_h:]) if int(c) != BG]
    for color in colors:
        mask = grid == color
        mask[:strip_h, :] = False
        for comp in _components(mask, connectivity=8):
            if len(comp) != 3:
                continue
            parsed = _parse_arrow(comp, rects, owner)
            if parsed is None:
                continue
            rect_id, cells = parsed
            arrows.append({"color": color, "rect_id": rect_id, "cells": cells})
    return arrows


def _segment_patterns(
    grid: np.ndarray,
    strip_h: int,
    arrow_colors: set[int],
) -> dict[int, np.ndarray]:
    row = np.asarray(grid[0], dtype=int)
    w = row.shape[0]
    for rr in range(strip_h):
        assert np.array_equal(grid[rr], row), "sample strip rows must match"

    candidates: list[tuple[int, dict[int, np.ndarray]]] = []
    for seg_w in range(2, w + 1):
        if w % seg_w != 0:
            continue
        first_colors = []
        patterns: dict[int, np.ndarray] = {}
        ok = True
        for start in range(0, w, seg_w):
            chunk = row[start : start + seg_w]
            if len(set(map(int, chunk.tolist()))) > 2:
                ok = False
                break
            index_color = int(chunk[0])
            first_colors.append(index_color)
            patterns[index_color] = chunk.copy()
        if not ok or len(set(first_colors)) != len(first_colors):
            continue
        first_set = set(first_colors)
        if first_set == arrow_colors:
            return patterns
        if arrow_colors.issubset(first_set):
            candidates.append((seg_w, patterns))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]
    raise AssertionError("failed to recover sample strip segmentation")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    strip_h = _detect_strip_height(g)
    rects, owner = _find_rectangles(g, strip_h)
    assert rects, "expected at least one rectangle"
    arrows = _find_arrows(g, strip_h, rects, owner)
    assert arrows, "expected at least one arrow"

    arrow_colors = {int(arrow["color"]) for arrow in arrows}
    patterns = _segment_patterns(g, strip_h, arrow_colors)

    out = g.copy()
    assignments: dict[int, tuple[int, ...]] = {}
    for arrow in arrows:
        rect_id = int(arrow["rect_id"])
        color = int(arrow["color"])
        pattern = tuple(int(v) for v in patterns[color].tolist())
        if rect_id in assignments:
            assert assignments[rect_id] == pattern, "multiple arrows for one rectangle must agree"
        assignments[rect_id] = pattern
        for r, c in arrow["cells"]:
            out[r, c] = BG

    for rect in rects:
        rect_id = int(rect["id"])
        if rect_id not in assignments:
            continue
        pattern = assignments[rect_id]
        width = int(rect["c1"] - rect["c0"] + 1)
        tiled = np.array([pattern[i % len(pattern)] for i in range(width)], dtype=int)
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = tiled[None, :]

    return out


PATTERN_LIBRARY = {
    3: (
        lambda a, b: [a, a, a],
        lambda a, b: [a, b, a],
        lambda a, b: [a, b, b],
    ),
    4: (
        lambda a, b: [a, a, a, a],
        lambda a, b: [a, b, a, b],
        lambda a, b: [a, a, b, b],
        lambda a, b: [a, b, b, a],
    ),
    5: (
        lambda a, b: [a, a, a, a, a],
        lambda a, b: [a, b, a, b, a],
        lambda a, b: [a, a, b, b, a],
        lambda a, b: [a, b, b, b, a],
        lambda a, b: [a, b, a, a, b],
    ),
}


def _arrow_cells(rect: dict[str, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == "top":
        return [(r0 - 1, mid_c), (r0 - 2, mid_c - 1), (r0 - 2, mid_c + 1)]
    if side == "bottom":
        return [(r1 + 1, mid_c), (r1 + 2, mid_c - 1), (r1 + 2, mid_c + 1)]
    if side == "left":
        return [(mid_r, c0 - 1), (mid_r - 1, c0 - 2), (mid_r + 1, c0 - 2)]
    if side == "right":
        return [(mid_r, c1 + 1), (mid_r - 1, c1 + 2), (mid_r + 1, c1 + 2)]
    raise ValueError(f"unknown side {side}")


def _choose_layout(rng: random.Random, h: int, w: int, n_rect: int, strip_h: int) -> list[dict[str, int]] | None:
    rects: list[dict[str, int]] = []
    attempts = 0
    width_max = 4 if w <= 12 else 5
    height_max = 4 if h <= 14 else 6
    while len(rects) < n_rect and attempts < 400:
        attempts += 1
        hh = rng.randint(3, height_max)
        ww = rng.randint(3, width_max)
        r0_min = strip_h + 2
        r0_max = h - hh - 3
        c0_min = 2
        c0_max = w - ww - 3
        if r0_min > r0_max or c0_min > c0_max:
            return None
        r0 = rng.randint(r0_min, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        cand = {"id": len(rects), "r0": r0, "c0": c0, "r1": r0 + hh - 1, "c1": c0 + ww - 1}
        ok = True
        for prev in rects:
            if not (
                cand["r1"] + 2 < prev["r0"]
                or prev["r1"] + 2 < cand["r0"]
                or cand["c1"] + 2 < prev["c0"]
                or prev["c1"] + 2 < cand["c0"]
            ):
                ok = False
                break
        if ok:
            rects.append(cand)
    if len(rects) != n_rect:
        return None
    return rects


def _build_pattern_row(
    rng: random.Random,
    band_w: int,
    index_colors: list[int],
    companion_pool: list[int],
) -> np.ndarray:
    chunks = []
    for idx_color in index_colors:
        companion = rng.choice(companion_pool)
        builder = rng.choice(PATTERN_LIBRARY[band_w])
        chunks.extend(builder(idx_color, companion))
    return np.array(chunks, dtype=int)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    strip_h = rng.choice([1, 2])
    n_targeted = rng.choice([3, 4])
    band_w = rng.choice([4, 5]) if n_targeted == 3 else rng.choice([3, 4, 5])
    w = n_targeted * band_w

    add_unpointed = rng.random() < 0.5 and w >= 15
    n_rect = n_targeted + (1 if add_unpointed else 0)
    min_h = strip_h + 9 + (n_rect - 3) * 2
    h = rng.randint(max(12, min_h), 20)

    rects = _choose_layout(rng, h, w, n_rect, strip_h)
    if rects is None:
        return None

    palette = list(range(1, 10))
    rng.shuffle(palette)
    index_colors = palette[:n_targeted]
    fill_colors = palette[n_targeted : n_targeted + n_rect]
    companion_pool = fill_colors.copy()
    if not companion_pool:
        return None

    grid = np.zeros((h, w), dtype=int)
    sample_row = _build_pattern_row(rng, band_w, index_colors, companion_pool)
    grid[0, :] = sample_row
    if strip_h == 2:
        grid[1, :] = sample_row

    for rect, color in zip(rects, fill_colors):
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = color

    rect_indices = list(range(n_rect))
    rng.shuffle(rect_indices)
    targeted_ids = sorted(rect_indices[:n_targeted])
    arrow_cells_used: set[tuple[int, int]] = set()
    for rect_id, arrow_color in zip(targeted_ids, index_colors):
        rect = rects[rect_id]
        side_count = 1 if rng.random() < 0.5 else rng.randint(2, 4)
        sides = rng.sample(["top", "bottom", "left", "right"], side_count)
        for side in sides:
            cells = _arrow_cells(rect, side)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                return None
            if any(grid[r, c] != BG for r, c in cells):
                return None
            if any((r, c) in arrow_cells_used for r, c in cells):
                return None
            for r, c in cells:
                grid[r, c] = arrow_color
                arrow_cells_used.add((r, c))

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if strip_h == 2 and not np.array_equal(grid[0], grid[1]):
        return None
    if not np.array_equal(out[:strip_h], grid[:strip_h]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
