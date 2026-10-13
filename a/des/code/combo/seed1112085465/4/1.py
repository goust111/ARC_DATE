from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_contiguous(values: list[int]) -> bool:
    return not values or values == list(range(values[0], values[-1] + 1))


def _analyze_frame(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 4 or width < 4:
        return None

    comp_set = set(comp)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(comp_set):
        return None

    for r, c in comp:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) in comp_set:
                return None

    top_missing = [c for c in range(c0 + 1, c1) if (r0, c) not in comp_set]
    bottom_missing = [c for c in range(c0 + 1, c1) if (r1, c) not in comp_set]
    left_missing = [r for r in range(r0 + 1, r1) if (r, c0) not in comp_set]
    right_missing = [r for r in range(r0 + 1, r1) if (r, c1) not in comp_set]

    if not all(_is_contiguous(missing) for missing in (top_missing, bottom_missing, left_missing, right_missing)):
        return None

    missing_sides = [(side, missing) for side, missing in (
        ("top", top_missing),
        ("bottom", bottom_missing),
        ("left", left_missing),
        ("right", right_missing),
    ) if missing]
    if len(missing_sides) != 1:
        return None

    side, missing = missing_sides[0]
    if side in ("top", "bottom"):
        gap0 = int(missing[0])
        gap1 = int(missing[-1])
        gap_cells = [(r0 if side == "top" else r1, c) for c in range(gap0, gap1 + 1)]
    else:
        gap0 = int(missing[0])
        gap1 = int(missing[-1])
        gap_cells = [(r, c0 if side == "left" else c1) for r in range(gap0, gap1 + 1)]

    if any(int(grid[r, c]) != BLACK for r, c in gap_cells):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    fill_mask = (interior != BLACK) & (interior != GREY)
    fill_count = int(fill_mask.sum())
    if fill_count <= 0:
        return None

    fill_colors = {int(v) for v in interior[fill_mask].tolist()}
    if len(fill_colors) != 1:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "gap": (int(gap0), int(gap1)),
        "color": int(next(iter(fill_colors))),
        "count": int(fill_count),
    }


def _spill_positions(shape: tuple[int, int], spec: dict) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = spec["bbox"]
    gap0, gap1 = spec["gap"]
    side = spec["side"]

    positions: list[tuple[int, int]] = []
    if side == "top":
        for r in range(r0, -1, -1):
            for c in range(gap0, gap1 + 1):
                positions.append((int(r), int(c)))
    elif side == "bottom":
        for r in range(r1, h):
            for c in range(gap0, gap1 + 1):
                positions.append((int(r), int(c)))
    elif side == "left":
        for c in range(c0, -1, -1):
            for r in range(gap0, gap1 + 1):
                positions.append((int(r), int(c)))
    else:
        for c in range(c1, w):
            for r in range(gap0, gap1 + 1):
                positions.append((int(r), int(c)))
    return positions


def _apply_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
        for r, c in _spill_positions(out.shape, spec)[: spec["count"]]:
            out[r, c] = int(spec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each frame's colored contents out through its unique opening.

    Input structure:
    - background is black
    - one-pixel-thick grey rectangular frames
    - exactly one side of each frame has a continuous opening
    - each frame interior contains pixels of exactly one non-black, non-grey color

    Rule:
    - count the interior colored pixels
    - clear the frame interior
    - starting at the opening, draw that many pixels outward in the opening's span
    """
    g = np.asarray(grid, dtype=int)
    specs: list[dict] = []
    for comp in _components(g == GREY):
        spec = _analyze_frame(g, comp)
        if spec is not None:
            specs.append(spec)
    return _apply_specs(g, specs)


# --------------- generator (codex) ---------------

GRID_SIDE_RANGE = (18, 31)
FRAME_COUNT_RANGE = (1, 5)
OUTER_H_RANGE = (4, 9)
OUTER_W_RANGE = (4, 10)
PLACEMENT_TRIES = 120
GENERATION_TRIES = 400
PADDING = 1


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = PADDING) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h - 1, r + pad) + 1):
            for cc in range(max(0, c - pad), min(w - 1, c + pad) + 1):
                expanded.add((int(rr), int(cc)))
    return expanded


def _grow_fill_mask(inner_h: int, inner_w: int, count: int) -> np.ndarray:
    mode = random.choice(("grow", "scatter"))
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    if mode == "scatter":
        idxs = random.sample(range(inner_h * inner_w), count)
        for idx in idxs:
            mask[idx // inner_w, idx % inner_w] = True
        return mask

    start = (random.randrange(inner_h), random.randrange(inner_w))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < count:
        if not frontier:
            remaining = [
                (r, c)
                for r in range(inner_h)
                for c in range(inner_w)
                if (r, c) not in cells
            ]
            frontier.add(random.choice(remaining))
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_gap(side: str, r0: int, c0: int, outer_h: int, outer_w: int) -> tuple[int, int]:
    span = outer_w - 2 if side in ("top", "bottom") else outer_h - 2
    gap_len = random.randint(1, span)
    start = random.randint(1, span - gap_len + 1)
    if side in ("top", "bottom"):
        return int(c0 + start), int(c0 + start + gap_len - 1)
    return int(r0 + start), int(r0 + start + gap_len - 1)


def _available_layers(shape: tuple[int, int], bbox: tuple[int, int, int, int], side: str) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if side == "top":
        return int(r0 + 1)
    if side == "bottom":
        return int(h - r1)
    if side == "left":
        return int(c0 + 1)
    return int(w - c1)


def _band_cells(spec: dict, layers: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    gap0, gap1 = spec["gap"]
    side = spec["side"]
    cells: set[tuple[int, int]] = set()
    if side == "top":
        for r in range(r0, r0 - layers, -1):
            for c in range(gap0, gap1 + 1):
                cells.add((int(r), int(c)))
    elif side == "bottom":
        for r in range(r1, r1 + layers):
            for c in range(gap0, gap1 + 1):
                cells.add((int(r), int(c)))
    elif side == "left":
        for c in range(c0, c0 - layers, -1):
            for r in range(gap0, gap1 + 1):
                cells.add((int(r), int(c)))
    else:
        for c in range(c1, c1 + layers):
            for r in range(gap0, gap1 + 1):
                cells.add((int(r), int(c)))
    return cells


def _place_frame(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    color: int,
) -> dict | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_TRIES):
        side = random.choice(SIDES)
        outer_h = random.randint(*OUTER_H_RANGE)
        outer_w = random.randint(*OUTER_W_RANGE)
        if outer_h >= h or outer_w >= w:
            continue

        r0 = random.randint(0, h - outer_h)
        c0 = random.randint(0, w - outer_w)
        r1 = int(r0 + outer_h - 1)
        c1 = int(c0 + outer_w - 1)
        bbox = (int(r0), int(c0), int(r1), int(c1))

        gap0, gap1 = _make_gap(side, r0, c0, outer_h, outer_w)
        gap_len = int(gap1 - gap0 + 1)
        inner_h = int(outer_h - 2)
        inner_w = int(outer_w - 2)
        inner_area = int(inner_h * inner_w)
        max_layers = _available_layers(grid.shape, bbox, side)
        if max_layers < 2:
            continue

        max_count = min(inner_area, gap_len * max_layers)
        if max_count <= 0:
            continue
        count = random.randint(1, max_count)
        layers = int((count + gap_len - 1) // gap_len)
        spec = {
            "bbox": bbox,
            "side": side,
            "gap": (int(gap0), int(gap1)),
            "color": int(color),
            "count": int(count),
        }

        occupied = {
            (rr, cc)
            for rr in range(r0, r1 + 1)
            for cc in range(c0, c1 + 1)
        }
        occupied.update(_band_cells(spec, layers))
        padded = _expand_cells(occupied, grid.shape)
        if padded & reserved:
            continue

        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        if side == "top":
            grid[r0, gap0 : gap1 + 1] = BLACK
        elif side == "bottom":
            grid[r1, gap0 : gap1 + 1] = BLACK
        elif side == "left":
            grid[gap0 : gap1 + 1, c0] = BLACK
        else:
            grid[gap0 : gap1 + 1, c1] = BLACK

        fill_mask = _grow_fill_mask(inner_h, inner_w, count)
        interior = np.full((inner_h, inner_w), BLACK, dtype=int)
        interior[fill_mask] = int(color)
        grid[r0 + 1 : r1, c0 + 1 : c1] = interior
        reserved.update(padded)
        return spec
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    if frame_count > len(FILL_COLORS):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    reserved: set[tuple[int, int]] = set()
    specs: list[dict] = []
    for color in random.sample(FILL_COLORS, frame_count):
        spec = _place_frame(grid, reserved, color)
        if spec is None:
            return None
        specs.append(spec)

    expected = _apply_specs(grid, specs)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pouring-frame input")
