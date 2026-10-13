from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _classify_frame(cells: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    pts = set(cells)
    border = _border_cells(r0, c0, r1, c1)
    if any((r, c) not in border for r, c in pts):
        return None

    missing = border - pts
    if not missing:
        return None

    if all(r == r0 for r, _ in missing):
        cols = sorted(c for _, c in missing)
        if cols == list(range(cols[0], cols[-1] + 1)) and c0 < cols[0] and cols[-1] < c1:
            return {"bbox": (r0, c0, r1, c1), "side": "top", "gap": (cols[0], cols[-1])}
    if all(r == r1 for r, _ in missing):
        cols = sorted(c for _, c in missing)
        if cols == list(range(cols[0], cols[-1] + 1)) and c0 < cols[0] and cols[-1] < c1:
            return {"bbox": (r0, c0, r1, c1), "side": "bottom", "gap": (cols[0], cols[-1])}
    if all(c == c0 for _, c in missing):
        rows = sorted(r for r, _ in missing)
        if rows == list(range(rows[0], rows[-1] + 1)) and r0 < rows[0] and rows[-1] < r1:
            return {"bbox": (r0, c0, r1, c1), "side": "left", "gap": (rows[0], rows[-1])}
    if all(c == c1 for _, c in missing):
        rows = sorted(r for r, _ in missing)
        if rows == list(range(rows[0], rows[-1] + 1)) and r0 < rows[0] and rows[-1] < r1:
            return {"bbox": (r0, c0, r1, c1), "side": "right", "gap": (rows[0], rows[-1])}
    return None


def _match_key(
    color: int,
    frame: dict[str, object],
    components_by_color: dict[int, list[dict[str, object]]],
) -> dict[str, object] | None:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    gap0, gap1 = frame["gap"]
    candidates: list[dict[str, object]] = []
    for comp in components_by_color[color]:
        cr0, cc0, cr1, cc1 = comp["bbox"]
        if not all(r0 < r < r1 and c0 < c < c1 for r, c in comp["cells"]):
            continue
        if side == "top" and cr0 == r0 + 1 and cc0 == gap0 and cc1 == gap1:
            candidates.append(comp)
        elif side == "bottom" and cr1 == r1 - 1 and cc0 == gap0 and cc1 == gap1:
            candidates.append(comp)
        elif side == "left" and cc0 == c0 + 1 and cr0 == gap0 and cr1 == gap1:
            candidates.append(comp)
        elif side == "right" and cc1 == c1 - 1 and cr0 == gap0 and cr1 == gap1:
            candidates.append(comp)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _in_bounds(base_r: int, base_c: int, h: int, w: int, shape: tuple[int, int]) -> bool:
    grid_h, grid_w = shape
    return 0 <= base_r and 0 <= base_c and base_r + h <= grid_h and base_c + w <= grid_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each key from its gapped frame and fire repeated key copies out through the gap.

    Input structure:
    - The background is black.
    - Several colors each form a 1-pixel-thick hollow rectangle with exactly one contiguous gap on one side.
    - Inside each frame is one same-colored connected key sprite whose bbox is aligned with the gap and sits just inside that side.

    Rule:
    - Detect every rectangular frame, its missing side segment, and the matching same-colored key inside it.
    - Erase the original key from the frame interior while leaving the frame and its gap unchanged.
    - Let K be the number of colored pixels in the key.
    - Starting immediately outside the gap, copy the key outward in the gap direction, shifting each next copy by K cells along that ray, until the next copy would leave the grid.

    Key insight:
    - The gap is both the pairing cue for the interior key and the launcher direction, while the launch spacing is determined by key area rather than by its width or height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    components_by_color: dict[int, list[dict[str, object]]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        comps = []
        for cells in _components(g == color):
            comps.append({"cells": cells, "bbox": _bbox(cells)})
        components_by_color[color] = comps

    matches: list[dict[str, object]] = []
    for color, comps in components_by_color.items():
        for comp in comps:
            frame = _classify_frame(comp["cells"])
            if frame is None:
                continue
            key = _match_key(color, frame, components_by_color)
            if key is None:
                continue
            matches.append({"color": color, "frame": frame, "key": key})

    assert matches

    out = g.copy()
    for match in matches:
        color = int(match["color"])
        frame = match["frame"]
        key = match["key"]
        r0, c0, r1, c1 = frame["bbox"]
        side = frame["side"]
        kr0, kc0, kr1, kc1 = key["bbox"]
        key_h = kr1 - kr0 + 1
        key_w = kc1 - kc0 + 1
        rel_cells = [(r - kr0, c - kc0) for r, c in key["cells"]]
        spacing = len(rel_cells)

        for r, c in key["cells"]:
            out[r, c] = BG

        if side == "top":
            base_r, base_c = r0 - key_h, kc0
            step_r, step_c = -spacing, 0
        elif side == "bottom":
            base_r, base_c = r1 + 1, kc0
            step_r, step_c = spacing, 0
        elif side == "left":
            base_r, base_c = kr0, c0 - key_w
            step_r, step_c = 0, -spacing
        else:
            base_r, base_c = kr0, c1 + 1
            step_r, step_c = 0, spacing

        cur_r, cur_c = base_r, base_c
        while _in_bounds(cur_r, cur_c, key_h, key_w, (h, w)):
            for dr, dc in rel_cells:
                out[cur_r + dr, cur_c + dc] = color
            cur_r += step_r
            cur_c += step_c

    return out


def _grow_shape(height: int, width: int, target_area: int) -> set[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _random_key_shape(height: int, width: int) -> list[tuple[int, int]]:
    if height == 1 and width == 1:
        return [(0, 0)]
    for _ in range(120):
        max_area = height * width
        min_area = max(height, width)
        if min_area > max_area:
            min_area = max_area
        if random.random() < 0.2:
            area = max_area
        else:
            area = random.randint(min_area, max_area)
        cells = _grow_shape(height, width, area)
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) == height and len(cols) == width:
            return sorted(cells)
    return [(r, c) for r in range(height) for c in range(width)]


def _build_object(color: int, side: str, copies: int) -> dict[str, object]:
    if side in ("top", "bottom"):
        key_h = random.randint(1, 3)
        key_w = random.randint(1, 4)
        left_space = random.randint(1, 3)
        right_space = random.randint(1, 3)
        far_space = random.randint(1, 3)
        frame_h = key_h + far_space + 2
        frame_w = key_w + left_space + right_space + 2
        key_r0 = 1 if side == "top" else 1 + far_space
        key_c0 = 1 + left_space
    else:
        key_h = random.randint(1, 4)
        key_w = random.randint(1, 3)
        top_space = random.randint(1, 3)
        bottom_space = random.randint(1, 3)
        far_space = random.randint(1, 3)
        frame_h = key_h + top_space + bottom_space + 2
        frame_w = key_w + far_space + 2
        key_r0 = 1 + top_space
        key_c0 = 1 if side == "left" else 1 + far_space

    key_shape = _random_key_shape(key_h, key_w)
    spacing = len(key_shape)

    frame_cells = _border_cells(0, 0, frame_h - 1, frame_w - 1)
    if side == "top":
        gap_cells = {(0, c) for c in range(key_c0, key_c0 + key_w)}
        start_r, start_c = -key_h, key_c0
        step_r, step_c = -spacing, 0
    elif side == "bottom":
        gap_cells = {(frame_h - 1, c) for c in range(key_c0, key_c0 + key_w)}
        start_r, start_c = frame_h, key_c0
        step_r, step_c = spacing, 0
    elif side == "left":
        gap_cells = {(r, 0) for r in range(key_r0, key_r0 + key_h)}
        start_r, start_c = key_r0, -key_w
        step_r, step_c = 0, -spacing
    else:
        gap_cells = {(r, frame_w - 1) for r in range(key_r0, key_r0 + key_h)}
        start_r, start_c = key_r0, frame_w
        step_r, step_c = 0, spacing
    frame_cells -= gap_cells

    key_cells = {(key_r0 + r, key_c0 + c) for r, c in key_shape}
    copy_cells: set[tuple[int, int]] = set()
    for idx in range(copies):
        base_r = start_r + step_r * idx
        base_c = start_c + step_c * idx
        for r, c in key_shape:
            copy_cells.add((base_r + r, base_c + c))

    reserve = frame_cells | key_cells | copy_cells
    rows = [r for r, _ in reserve]
    cols = [c for _, c in reserve]
    return {
        "color": color,
        "frame_cells": frame_cells,
        "key_cells": key_cells,
        "reserve": reserve,
        "difficulty": (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1),
        "min_r": min(rows),
        "max_r": max(rows),
        "min_c": min(cols),
        "max_c": max(cols),
    }


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _place_object(grid: np.ndarray, blocked: np.ndarray, obj: dict[str, object]) -> bool:
    h, w = grid.shape
    row_lo = -int(obj["min_r"])
    row_hi = h - 1 - int(obj["max_r"])
    col_lo = -int(obj["min_c"])
    col_hi = w - 1 - int(obj["max_c"])
    if row_lo > row_hi or col_lo > col_hi:
        return False

    positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(positions)
    for base_r, base_c in positions:
        if any(blocked[base_r + r, base_c + c] for r, c in obj["reserve"]):
            continue
        for r, c in obj["frame_cells"]:
            grid[base_r + r, base_c + c] = int(obj["color"])
        for r, c in obj["key_cells"]:
            grid[base_r + r, base_c + c] = int(obj["color"])
        placed_reserve = {(base_r + r, base_c + c) for r, c in obj["reserve"]}
        _mark_blocked(blocked, placed_reserve)
        return True
    return False


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(14, 30)
    width = random.randint(14, 30)
    object_count = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 1])[0]
    colors = random.sample(PALETTE, object_count)

    if object_count == 4 and random.random() < 0.5:
        sides = random.sample(list(SIDES), 4)
    else:
        sides = [random.choice(SIDES) for _ in range(object_count)]

    objects = []
    for color, side in zip(colors, sides):
        copies = random.randint(1, 3)
        objects.append(_build_object(color, side, copies))
    objects.sort(key=lambda obj: int(obj["difficulty"]), reverse=True)

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    for obj in objects:
        if not _place_object(grid, blocked, obj):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-launch frame puzzle")
