from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
ORANGE = 7
INPUT_BORDER_COLORS = (1, 4, 5, 6, 8, 9)

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
CONTAINER_COUNT_RANGE = (2, 4)
INNER_H_RANGE = (4, 8)
INNER_W_RANGE = (4, 8)
HOLE_AREA_MIN = 3
HOLE_AREA_CAP = 8
SHAPE_ATTEMPTS = 120
GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.flat).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((int(cr), int(cc)))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(
    coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in np.asarray(coords, dtype=int)))
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _mask_has_internal_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for comp in _components(~mask):
        if not any(int(r) in (0, h - 1) or int(c) in (0, w - 1) for r, c in comp):
            return True
    return False


def _paint_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _extract_container(
    grid: np.ndarray,
    coords: np.ndarray,
    color: int,
    bg: int,
) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if int(len(coords)) != 2 * h + 2 * w - 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if not bool(np.all(sub[border] == int(color))):
        return None
    if h > 2 and w > 2 and bool(np.any(sub[1:-1, 1:-1] == int(color))):
        return None

    inner = sub[1:-1, 1:-1]
    if inner.size == 0:
        return None
    if set(map(int, np.unique(inner))) - {int(bg), ORANGE}:
        return None
    if not bool(np.any(inner == ORANGE)) or not bool(np.any(inner == int(bg))):
        return None

    holes = _components(inner == int(bg))
    if len(holes) != 1:
        return None
    hole = holes[0]
    hole_key, shape_h, shape_w = _normalize_coords(hole)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(color),
        "hole_cells": tuple(sorted((int(r), int(c)) for r, c in hole)),
        "hole_key": hole_key,
        "shape_h": int(shape_h),
        "shape_w": int(shape_w),
    }


def _find_containers(grid: np.ndarray, bg: int) -> list[dict]:
    containers: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (int(bg), ORANGE)):
        for coords in _components(grid == int(color)):
            item = _extract_container(grid, coords, int(color), int(bg))
            if item is not None:
                containers.append(item)
    containers.sort(key=lambda item: item["bbox"])
    assert containers, "expected at least one valid container"
    return containers


def _find_fragments(grid: np.ndarray, containers: list[dict]) -> list[dict]:
    owner = np.full(grid.shape, -1, dtype=int)
    for idx, container in enumerate(containers):
        r0, c0, r1, c1 = container["bbox"]
        owner[r0 + 1 : r1, c0 + 1 : c1] = int(idx)

    fragments: list[dict] = []
    for coords in _components(grid == ORANGE):
        ids = {int(owner[int(r), int(c)]) for r, c in coords}
        if len(ids) == 1 and next(iter(ids)) >= 0:
            continue
        key, shape_h, shape_w = _normalize_coords(coords)
        fragments.append(
            {
                "coords": tuple(sorted((int(r), int(c)) for r, c in coords)),
                "key": key,
                "shape_h": int(shape_h),
                "shape_w": int(shape_w),
            }
        )
    fragments.sort(key=lambda item: item["coords"])
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match orange fragments to equally shaped container holes, fill the holes,
    erase the moved fragments, then recolor each border by completion state.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    containers = _find_containers(g, bg)
    fragments = _find_fragments(g, containers)

    by_shape: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for fragment in fragments:
        by_shape.setdefault(fragment["key"], []).append(fragment)

    out = g.copy()
    for container in containers:
        matches = by_shape.get(container["hole_key"], [])
        if len(matches) == 1:
            fragment = matches.pop()
            for r, c in fragment["coords"]:
                out[int(r), int(c)] = int(bg)
            r0, c0, _r1, _c1 = container["bbox"]
            for rr, cc in container["hole_cells"]:
                out[int(r0 + 1 + rr), int(c0 + 1 + cc)] = ORANGE

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        final_color = GREEN if not bool(np.any(inner == int(bg))) else RED
        _paint_border(out, container["bbox"], int(final_color))

    return out


def _random_hole_cells(rng: random.Random, inner_h: int, inner_w: int) -> tuple[tuple[int, int], ...] | None:
    max_area = min(HOLE_AREA_CAP, (inner_h * inner_w) // 2)
    if max_area < HOLE_AREA_MIN:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        target = rng.randint(HOLE_AREA_MIN, max_area)
        start = (rng.randrange(inner_h), rng.randrange(inner_w))
        cells = {start}

        while len(cells) < target:
            frontier = []
            for r, c in sorted(cells):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            frontier = list(dict.fromkeys(frontier))
            cells.add(rng.choice(frontier))

        if len(cells) != target:
            continue

        norm, shape_h, shape_w = _normalize_coords(tuple(cells))
        if shape_h == 1 or shape_w == 1:
            continue
        mask = np.zeros((shape_h, shape_w), dtype=bool)
        for r, c in norm:
            mask[int(r), int(c)] = True
        if bool(mask.all()):
            continue
        if _mask_has_internal_hole(mask):
            continue
        return tuple(sorted((int(r), int(c)) for r, c in cells))
    return None


def _make_container_specs(rng: random.Random, count: int) -> list[dict] | None:
    colors = list(INPUT_BORDER_COLORS)
    rng.shuffle(colors)
    specs: list[dict] = []
    used_shapes: set[tuple[tuple[int, int], ...]] = set()

    for idx in range(count):
        border_color = int(colors[idx])
        spec = None
        for _ in range(SHAPE_ATTEMPTS):
            inner_h = rng.randint(*INNER_H_RANGE)
            inner_w = rng.randint(*INNER_W_RANGE)
            hole_cells = _random_hole_cells(rng, inner_h, inner_w)
            if hole_cells is None:
                continue
            hole_key, shape_h, shape_w = _normalize_coords(hole_cells)
            if hole_key in used_shapes:
                continue
            spec = {
                "border_color": int(border_color),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
                "hole_cells": tuple(hole_cells),
                "hole_key": hole_key,
                "shape_h": int(shape_h),
                "shape_w": int(shape_w),
            }
            used_shapes.add(hole_key)
            break
        if spec is None:
            return None
        specs.append(spec)
    return specs


def _can_place_box(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    if top < 1 or left < 1 or top + h > H - 1 or left + w > W - 1:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_box(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    H, W = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _place_boxes(
    rng: random.Random,
    occupied: np.ndarray,
    boxes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda idx: -(boxes[idx][0] * boxes[idx][1]))
    H, W = occupied.shape

    for idx in order:
        h, w = boxes[idx]
        positions = [(r, c) for r in range(1, H - h) for c in range(1, W - w)]
        rng.shuffle(positions)
        chosen = None
        for top, left in positions:
            if _can_place_box(occupied, int(top), int(left), int(h), int(w), pad=1):
                _mark_box(occupied, int(top), int(left), int(h), int(w), pad=1)
                chosen = (int(top), int(left))
                break
        if chosen is None:
            return None
        placements[idx] = chosen

    return [item for item in placements if item is not None]


def _draw_container(grid: np.ndarray, spec: dict) -> None:
    top = int(spec["top"])
    left = int(spec["left"])
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])

    grid[top : top + outer_h, left : left + outer_w] = int(spec["border_color"])
    grid[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w] = ORANGE
    for r, c in spec["hole_cells"]:
        grid[top + 1 + int(r), left + 1 + int(c)] = BG


def _draw_fragment(grid: np.ndarray, spec: dict) -> None:
    top = int(spec["frag_top"])
    left = int(spec["frag_left"])
    for r, c in spec["hole_key"]:
        grid[top + int(r), left + int(c)] = ORANGE


def _build_expected_output(grid: np.ndarray, specs: list[dict], matched_ids: set[int], bg: int) -> np.ndarray:
    out = grid.copy()
    for idx in matched_ids:
        spec = specs[int(idx)]
        for r, c in spec["hole_key"]:
            out[int(spec["frag_top"]) + int(r), int(spec["frag_left"]) + int(c)] = int(bg)
        for r, c in spec["hole_cells"]:
            out[int(spec["top"]) + 1 + int(r), int(spec["left"]) + 1 + int(c)] = ORANGE

    for idx, spec in enumerate(specs):
        final_color = GREEN if idx in matched_ids else RED
        bbox = (
            int(spec["top"]),
            int(spec["left"]),
            int(spec["top"] + spec["outer_h"] - 1),
            int(spec["left"] + spec["outer_w"] - 1),
        )
        _paint_border(out, bbox, int(final_color))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    container_count = rng.randint(*CONTAINER_COUNT_RANGE)
    specs = _make_container_specs(rng, int(container_count))
    if specs is None:
        return None

    matched_count = rng.randint(1, container_count - 1)
    matched_ids = set(rng.sample(range(container_count), matched_count))
    matched_specs = [specs[idx] for idx in sorted(matched_ids)]

    max_h = max(spec["outer_h"] for spec in specs) + 2
    max_w = max(spec["outer_w"] for spec in specs) + 2
    max_frag_h = max(spec["shape_h"] for spec in matched_specs) + 2
    max_frag_w = max(spec["shape_w"] for spec in matched_specs) + 2
    packed_area = sum((spec["outer_h"] + 2) * (spec["outer_w"] + 2) for spec in specs)
    packed_area += sum((spec["shape_h"] + 2) * (spec["shape_w"] + 2) for spec in matched_specs)
    side_hint = int(np.ceil(np.sqrt(packed_area * 1.5)))

    min_h = max(GRID_H_RANGE[0], max_h, max_frag_h, side_hint)
    min_w = max(GRID_W_RANGE[0], max_w, max_frag_w, side_hint)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    H = rng.randint(min_h, GRID_H_RANGE[1])
    W = rng.randint(min_w, GRID_W_RANGE[1])
    occupied = np.zeros((H, W), dtype=bool)

    container_boxes = [(int(spec["outer_h"]), int(spec["outer_w"])) for spec in specs]
    container_positions = _place_boxes(rng, occupied, container_boxes)
    if container_positions is None:
        return None
    for spec, (top, left) in zip(specs, container_positions):
        spec["top"] = int(top)
        spec["left"] = int(left)

    fragment_boxes = [(int(spec["shape_h"]), int(spec["shape_w"])) for spec in matched_specs]
    fragment_positions = _place_boxes(rng, occupied, fragment_boxes)
    if fragment_positions is None:
        return None
    for spec, (top, left) in zip(matched_specs, fragment_positions):
        spec["frag_top"] = int(top)
        spec["frag_left"] = int(left)

    grid = np.full((H, W), BG, dtype=int)
    for spec in specs:
        _draw_container(grid, spec)
    for spec in matched_specs:
        _draw_fragment(grid, spec)

    expected = _build_expected_output(grid, specs, matched_ids, BG)
    if not bool(np.any(expected == RED)) or not bool(np.any(expected == GREEN)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate ARC-style inputs with unique hole/fragment matches and
    both completion outcomes present.
    """
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-fragment puzzle input")
