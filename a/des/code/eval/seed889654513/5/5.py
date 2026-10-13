from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
MAROON = 9
COLOR_ORDER = tuple(range(10))
COLOR_RANK = {color: idx for idx, color in enumerate(COLOR_ORDER)}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 300
GRID_MIN = 18
GRID_MAX = 30
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8]
GAP_LENGTHS = (2, 3, 4)
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))


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
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _analyze_container(cells: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    cell_set = set(cells)
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0, r1 + 1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))
    if not cell_set <= perimeter:
        return None

    side_cells = {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }

    gap_side = None
    gap_cells: list[tuple[int, int]] = []
    for side, pts in side_cells.items():
        missing = [p for p in pts if p not in cell_set]
        if not missing:
            continue
        if gap_side is not None or not (2 <= len(missing) <= 4):
            return None
        if side in ("top", "bottom"):
            coords = sorted(c for _, c in missing)
        else:
            coords = sorted(r for r, _ in missing)
        if coords != list(range(coords[0], coords[0] + len(coords))):
            return None
        gap_side = side
        gap_cells = missing

    if gap_side is None:
        return None
    if len(cell_set) != len(perimeter) - len(gap_cells):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap_cells": gap_cells,
        "outward": SIDE_TO_VEC[gap_side],
        "signature": (_normalize(gap_cells), SIDE_TO_VEC[gap_side]),
    }


def _analyze_key(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict | None:
    h, w = grid.shape
    cell_set = set(cells)
    handles = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == MAROON:
                handles.add((int(nr), int(nc)))
    if len(handles) != 1:
        return None

    handle = next(iter(handles))
    touching = [
        (r, c)
        for r, c in cells
        if abs(int(r - handle[0])) + abs(int(c - handle[1])) == 1
    ]
    if not touching:
        return None

    dirs = {(int(handle[0] - r), int(handle[1] - c)) for r, c in touching}
    if len(dirs) != 1:
        return None
    outward = next(iter(dirs))
    if outward not in SIDE_TO_VEC.values():
        return None

    return {
        "cells": cells,
        "handle": handle,
        "outward": outward,
        "signature": (_normalize(cells), outward),
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _most_common_color(grid)
    grey_comps = _components(np.asarray(grid == GREY, dtype=bool))
    containers: list[dict] = []
    keys: list[dict] = []
    for comp in grey_comps:
        container = _analyze_container(comp)
        if container is not None:
            containers.append(container)
            continue
        key = _analyze_key(grid, comp)
        if key is not None:
            keys.append(key)
            continue
        raise AssertionError("unexpected grey component")
    return bg, containers, keys


def _gap_center(container: dict) -> tuple[int, int]:
    gap_cells = container["gap_cells"]
    dr, dc = container["outward"]
    if dr != 0:
        ordered = sorted(gap_cells, key=lambda p: (p[1], p[0]))
    else:
        ordered = sorted(gap_cells, key=lambda p: (p[0], p[1]))
    return ordered[(len(ordered) - 1) // 2]


def _majority_color(values: list[int]) -> int:
    counts = Counter(int(v) for v in values)
    return min(counts, key=lambda color: (-counts[color], COLOR_RANK[color]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Seal each grey container with its matching grey key, then emit a ray from the original gap
    direction only when the number of colored dots inside the sealed container is odd.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg, containers, keys = _extract_scene(g)
    assert containers, "expected at least one container"
    assert len(containers) == len(keys), "expected one key per container"

    container_sigs = Counter(container["signature"] for container in containers)
    key_sigs = Counter(key["signature"] for key in keys)
    assert container_sigs == key_sigs, "container gaps and keys must match"

    out = g.copy()
    for key in keys:
        for r, c in key["cells"]:
            out[int(r), int(c)] = bg
        hr, hc = key["handle"]
        out[int(hr), int(hc)] = bg

    for container in containers:
        for r, c in container["gap_cells"]:
            out[int(r), int(c)] = GREY

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        dots: list[int] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(out[r, c])
                if value not in (bg, GREY):
                    dots.append(value)
        if len(dots) % 2 == 0:
            continue

        ray_color = _majority_color(dots)
        dr, dc = container["outward"]
        sr, sc = _gap_center(container)
        r = sr + dr
        c = sc + dc
        while 0 <= r < h and 0 <= c < w and int(out[r, c]) != GREY:
            out[r, c] = ray_color
            r += dr
            c += dc

    return out


def _bbox_clear(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc1 = min(w, c1 + pad + 1)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_bbox(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _sample_dot_values(count: int, palette: list[int]) -> list[int]:
    tie_case = count in (3, 5) and random.random() < 0.35 and len(palette) >= 3
    if tie_case and count == 3:
        values = random.sample(palette, 3)
    elif tie_case and count == 5:
        colors = random.sample(palette, 3)
        values = [colors[0], colors[0], colors[1], colors[1], colors[2]]
    else:
        main_color = random.choice(palette)
        others = [color for color in palette if color != main_color]
        values = [main_color]
        for _ in range(count - 1):
            if others and random.random() < 0.6:
                values.append(main_color)
            else:
                values.append(random.choice(others or [main_color]))
    random.shuffle(values)
    return [int(v) for v in values]


def _build_container_spec(side: str, gap_len: int, odd_count: bool, palette: list[int]) -> dict:
    if side in ("top", "bottom"):
        width = random.randint(gap_len + 4, min(gap_len + 7, 10))
        height = random.randint(5, 9)
        inner_span = width - 2
    else:
        height = random.randint(gap_len + 4, min(gap_len + 7, 10))
        width = random.randint(5, 9)
        inner_span = height - 2

    gap_offset = random.randint(1, inner_span - gap_len - 1)
    gap_cells: list[tuple[int, int]] = []
    if side == "top":
        gap_cells = [(0, 1 + gap_offset + i) for i in range(gap_len)]
    elif side == "bottom":
        gap_cells = [(height - 1, 1 + gap_offset + i) for i in range(gap_len)]
    elif side == "left":
        gap_cells = [(1 + gap_offset + i, 0) for i in range(gap_len)]
    else:
        gap_cells = [(1 + gap_offset + i, width - 1) for i in range(gap_len)]

    wall_cells = []
    gap_set = set(gap_cells)
    for r in range(height):
        for c in range(width):
            on_perimeter = r in (0, height - 1) or c in (0, width - 1)
            if on_perimeter and (r, c) not in gap_set:
                wall_cells.append((r, c))

    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    max_dots = min(5, len(interior))
    counts = [k for k in range(1, max_dots + 1) if (k % 2 == 1) == odd_count]
    dot_count = random.choice(counts)
    dot_positions = random.sample(interior, dot_count)
    dot_values = _sample_dot_values(dot_count, palette)
    dots = [(r, c, color) for (r, c), color in zip(dot_positions, dot_values)]

    return {
        "side": side,
        "gap_len": gap_len,
        "height": height,
        "width": width,
        "wall_cells": wall_cells,
        "gap_cells": gap_cells,
        "dots": dots,
        "signature": (_normalize(gap_cells), SIDE_TO_VEC[side]),
    }


def _sample_parities(n: int) -> list[bool]:
    if n == 1:
        return [bool(random.getrandbits(1))]
    if n == 2:
        vals = [True, False]
        random.shuffle(vals)
        return vals
    vals = random.choice([[True, True, False], [True, False, False]])
    random.shuffle(vals)
    return vals


def _grid_size_for_count(count: int) -> tuple[int, int]:
    low = max(GRID_MIN, 12 + 4 * count)
    high = max(low, GRID_MAX)
    return random.randint(low, high), random.randint(low, high)


def _container_candidates(blocked: np.ndarray, spec: dict) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ch = spec["height"]
    cw = spec["width"]
    side = spec["side"]
    choices: list[tuple[int, int]] = []
    for r0 in range(1, h - ch):
        r1 = r0 + ch - 1
        if side == "top" and r0 < 3:
            continue
        if side == "bottom" and r1 > h - 4:
            continue
        for c0 in range(1, w - cw):
            c1 = c0 + cw - 1
            if side == "left" and c0 < 3:
                continue
            if side == "right" and c1 > w - 4:
                continue
            if _bbox_clear(blocked, r0, c0, r1, c1, pad=1):
                choices.append((r0, c0))
    return choices


def _paint_container(grid: np.ndarray, spec: dict, origin: tuple[int, int]) -> None:
    r0, c0 = origin
    for r, c in spec["wall_cells"]:
        grid[r0 + r, c0 + c] = GREY
    for r, c, color in spec["dots"]:
        grid[r0 + r, c0 + c] = int(color)


def _key_cells_for_spec(spec: dict, origin: tuple[int, int], handle_idx: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r0, c0 = origin
    side = spec["side"]
    length = spec["gap_len"]
    if side in ("top", "bottom"):
        cells = [(r0, c0 + i) for i in range(length)]
        handle = (r0 - 1, c0 + handle_idx) if side == "top" else (r0 + 1, c0 + handle_idx)
    else:
        cells = [(r0 + i, c0) for i in range(length)]
        handle = (r0 + handle_idx, c0 - 1) if side == "left" else (r0 + handle_idx, c0 + 1)
    return cells, handle


def _key_candidates(blocked: np.ndarray, spec: dict) -> list[tuple[tuple[int, int], int]]:
    h, w = blocked.shape
    side = spec["side"]
    length = spec["gap_len"]
    choices: list[tuple[tuple[int, int], int]] = []
    if side in ("top", "bottom"):
        for r0 in range(1, h - 1):
            for c0 in range(1, w - length - 1):
                for handle_idx in range(length):
                    cells, handle = _key_cells_for_spec(spec, (r0, c0), handle_idx)
                    union = cells + [handle]
                    if any(not (0 <= r < h and 0 <= c < w) for r, c in union):
                        continue
                    ur0, uc0, ur1, uc1 = _bbox(union)
                    if _bbox_clear(blocked, ur0, uc0, ur1, uc1, pad=1):
                        choices.append(((r0, c0), handle_idx))
    else:
        for r0 in range(1, h - length - 1):
            for c0 in range(1, w - 1):
                for handle_idx in range(length):
                    cells, handle = _key_cells_for_spec(spec, (r0, c0), handle_idx)
                    union = cells + [handle]
                    if any(not (0 <= r < h and 0 <= c < w) for r, c in union):
                        continue
                    ur0, uc0, ur1, uc1 = _bbox(union)
                    if _bbox_clear(blocked, ur0, uc0, ur1, uc1, pad=1):
                        choices.append(((r0, c0), handle_idx))
    return choices


def _paint_key(grid: np.ndarray, spec: dict, origin: tuple[int, int], handle_idx: int) -> tuple[int, int, int, int]:
    cells, handle = _key_cells_for_spec(spec, origin, handle_idx)
    for r, c in cells:
        grid[r, c] = GREY
    grid[handle] = MAROON
    return _bbox(cells + [handle])


def _generate_attempt() -> np.ndarray | None:
    n_containers = random.choices([1, 2, 3], weights=[1, 2, 2])[0]
    h, w = _grid_size_for_count(n_containers)
    bg = random.choice(BG_CHOICES)
    palette = [color for color in COLOR_ORDER if color not in (bg, GREY, MAROON)]

    signatures = random.sample([(side, gap) for side in SIDES for gap in GAP_LENGTHS], n_containers)
    parities = _sample_parities(n_containers)
    specs = [
        _build_container_spec(side=side, gap_len=gap_len, odd_count=odd_count, palette=palette)
        for (side, gap_len), odd_count in zip(signatures, parities)
    ]

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    placed_specs = []
    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        candidates = _container_candidates(blocked, spec)
        if not candidates:
            return None
        origin = random.choice(candidates)
        _paint_container(grid, spec, origin)
        _mark_bbox(blocked, origin[0], origin[1], origin[0] + spec["height"] - 1, origin[1] + spec["width"] - 1)
        spec["origin"] = origin
        placed_specs.append(spec)

    random.shuffle(placed_specs)
    for spec in placed_specs:
        candidates = _key_candidates(blocked, spec)
        if not candidates:
            return None
        origin, handle_idx = random.choice(candidates)
        r0, c0, r1, c1 = _paint_key(grid, spec, origin, handle_idx)
        _mark_bbox(blocked, r0, c0, r1, c1)

    try:
        bg2, containers, keys = _extract_scene(grid)
        if bg2 != bg:
            return None
        if len(containers) != n_containers or len(keys) != n_containers:
            return None
        if Counter(container["signature"] for container in containers) != Counter(key["signature"] for key in keys):
            return None
        out = main(grid)
    except Exception:
        return None

    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
