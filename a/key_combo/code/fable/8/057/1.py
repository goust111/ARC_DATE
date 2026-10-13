from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
NON_BLACK = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _color_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": _bbox_from_coords(coords_arr),
                    "area": int(len(coords)),
                }
            )
    return components


def _is_single_cycle(mapping: dict[int, int]) -> bool:
    if not mapping:
        return False
    keys = set(mapping)
    values = set(mapping.values())
    if keys != values:
        return False
    if any(src == dst for src, dst in mapping.items()):
        return False

    start = next(iter(mapping))
    seen = set()
    cur = start
    while cur not in seen:
        seen.add(cur)
        cur = mapping.get(cur, None)
        if cur is None:
            return False
    return cur == start and seen == keys


def _step_color(mapping: dict[int, int], color: int, steps: int) -> int:
    cur = int(color)
    cycle_len = len(mapping)
    for _ in range(int(steps) % cycle_len):
        cur = mapping[cur]
    return cur


def _corner_touches(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, int]]:
    mask = grid != BLACK
    candidates = []
    for coords in _mask_components(mask):
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        if region.shape[0] != 2 or region.shape[1] < 3:
            continue
        if not _corner_touches(bbox, grid.shape):
            continue
        if np.any(region == BLACK):
            continue
        top = [int(x) for x in region[0]]
        bottom = [int(x) for x in region[1]]
        if len(set(top)) != len(top) or len(set(bottom)) != len(bottom):
            continue
        mapping = dict(zip(top, bottom))
        if len(mapping) != region.shape[1]:
            continue
        if not _is_single_cycle(mapping):
            continue
        candidates.append((bbox, mapping))

    assert len(candidates) == 1, "expected exactly one valid corner legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each scene object by stepping through the legend's color cycle by its own area.

    Input structure:
    - The grid has a black background.
    - One corner contains a 2-row connected legend made of vertical dominoes; each column maps top color X to bottom color Y.
    - Away from the border, the scene contains several separated monochrome 4-connected objects whose colors all belong to the legend cycle.

    Rule:
    - Read all domino columns as a directed color mapping and verify that together they form one closed cycle.
    - For each scene object, count its pixels as n, then start from the object's current color and advance n steps along that cycle.
    - Recolor the whole object to that destination color, keep every scene shape in place, and clear the legend to black.

    Key insight:
    - The dominoes define an iterable cycle rather than a one-step lookup, and each object's own area chooses how far to walk around that cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_bbox, mapping = _find_legend(g)
    r0, c0, r1, c1 = legend_bbox

    scene = g.copy()
    scene[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    objects = _color_components(scene, bg=BLACK)
    assert objects, "expected at least one scene object"

    out = scene.copy()
    for obj in objects:
        color = int(obj["color"])
        assert color in mapping, "scene color must appear in legend cycle"
        target = _step_color(mapping, color, obj["area"])
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = target

    return out


def _random_connected_shape(area: int) -> np.ndarray:
    max_side = min(5, max(2, area))
    for _ in range(200):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < area:
            choices = []
            for rr, cc in frontier:
                min_r = min(r for r, _ in cells | {(rr, cc)})
                max_r = max(r for r, _ in cells | {(rr, cc)})
                min_c = min(c for _, c in cells | {(rr, cc)})
                max_c = max(c for _, c in cells | {(rr, cc)})
                if max_r - min_r + 1 <= max_side and max_c - min_c + 1 <= max_side:
                    choices.append((rr, cc))
            if not choices:
                break
            cell = random.choice(choices)
            cells.add(cell)
            frontier.discard(cell)
            for nr, nc in _neighbors4(*cell):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        if h > 5 or w > 5:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr - min_r, cc - min_c] = True
        if area > 2 and (h == 1 or w == 1) and random.random() < 0.75:
            continue
        return mask
    raise RuntimeError(f"failed to sample connected shape of area {area}")


def _legend_bbox_for_corner(height: int, width: int, cycle_len: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, 1, cycle_len - 1
    if corner == "tr":
        return 0, width - cycle_len, 1, width - 1
    if corner == "bl":
        return height - 2, 0, height - 1, cycle_len - 1
    if corner == "br":
        return height - 2, width - cycle_len, height - 1, width - 1
    raise ValueError(corner)


def _paint_legend(grid: np.ndarray, mapping: dict[int, int], corner: str) -> tuple[int, int, int, int]:
    pairs = list(mapping.items())
    random.shuffle(pairs)
    cycle_len = len(pairs)
    r0, c0, r1, c1 = _legend_bbox_for_corner(grid.shape[0], grid.shape[1], cycle_len, corner)
    for offset, (src, dst) in enumerate(pairs):
        grid[r0, c0 + offset] = src
        grid[r1, c0 + offset] = dst
    return r0, c0, r1, c1


def _expand_bbox(r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int], pad: int = 1) -> tuple[int, int, int, int]:
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _mark_forbidden(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    rr0, cc0, rr1, cc1 = _expand_bbox(*bbox, mask.shape, pad=pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_areas(cycle_len: int, count: int) -> list[int]:
    choices = list(range(1, 10))
    wrap_choices = [a for a in choices if a > cycle_len]
    assert wrap_choices
    while True:
        first = random.choice(choices)
        second_candidates = [a for a in wrap_choices if a != first and a % cycle_len != first % cycle_len]
        if not second_candidates:
            continue
        second = random.choice(second_candidates)
        remaining = [a for a in choices if a not in {first, second}]
        extra = random.sample(remaining, count - 2)
        areas = [first, second] + extra
        if len(set(areas)) != count:
            continue
        if not any(a % cycle_len not in (0, 1) for a in areas):
            continue
        random.shuffle(areas)
        return areas


def _sample_object_specs(cycle: list[int], count: int) -> list[dict]:
    cycle_len = len(cycle)
    areas = _sample_areas(cycle_len, count)
    repeated_color = random.choice(cycle)

    repeated_pair = None
    for i in range(count):
        for j in range(i + 1, count):
            if areas[i] % cycle_len != areas[j] % cycle_len:
                repeated_pair = (i, j)
                break
        if repeated_pair is not None:
            break
    assert repeated_pair is not None

    colors = [random.choice(cycle) for _ in range(count)]
    i, j = repeated_pair
    colors[i] = repeated_color
    colors[j] = repeated_color

    return [
        {"area": area, "color": color, "shape": _random_connected_shape(area)}
        for area, color in zip(areas, colors)
    ]


def _place_objects(
    height: int,
    width: int,
    legend_bbox: tuple[int, int, int, int],
    specs: list[dict],
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)
    forbidden[0, :] = True
    forbidden[-1, :] = True
    forbidden[:, 0] = True
    forbidden[:, -1] = True
    _mark_forbidden(forbidden, legend_bbox, pad=1)

    for spec in sorted(specs, key=lambda item: item["shape"].sum(), reverse=True):
        mask = spec["shape"]
        obj_h, obj_w = mask.shape
        positions = [
            (r, c)
            for r in range(1, height - obj_h)
            for c in range(1, width - obj_w)
            if not forbidden[r : r + obj_h, c : c + obj_w].any()
        ]
        random.shuffle(positions)
        if not positions:
            return None

        placed = False
        for r0, c0 in positions[:120]:
            grid[r0 : r0 + obj_h, c0 : c0 + obj_w][mask] = spec["color"]
            _mark_forbidden(forbidden, (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1), pad=1)
            placed = True
            break
        if not placed:
            return None

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        legend_bbox, mapping = _find_legend(grid)
        r0, c0, r1, c1 = legend_bbox
        scene = grid.copy()
        scene[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        objects = _color_components(scene, bg=BLACK)
        if len(objects) < 3:
            return False
        areas = [obj["area"] for obj in objects]
        if len(set(areas)) != len(areas):
            return False
        cycle_len = len(mapping)
        if not any(area > cycle_len for area in areas):
            return False
        if not any(area % cycle_len not in (0, 1) for area in areas):
            return False

        same_color_pair = False
        for i, obj_a in enumerate(objects):
            for obj_b in objects[i + 1 :]:
                if obj_a["color"] == obj_b["color"]:
                    target_a = _step_color(mapping, obj_a["color"], obj_a["area"])
                    target_b = _step_color(mapping, obj_b["color"], obj_b["area"])
                    if target_a != target_b:
                        same_color_pair = True
                        break
            if same_color_pair:
                break
        if not same_color_pair:
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
            return False

        changed = 0
        out_colors = set()
        for obj in objects:
            coords = obj["coords"]
            target = _step_color(mapping, obj["color"], obj["area"])
            vals = set(map(int, out[coords[:, 0], coords[:, 1]]))
            if vals != {target}:
                return False
            out_colors.add(target)
            if target != obj["color"]:
                changed += 1
        if changed == 0 or len(out_colors) < 2:
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    cycle_len = random.randint(3, 6)
    cycle = random.sample(NON_BLACK, cycle_len)
    mapping = {cycle[i]: cycle[(i + 1) % cycle_len] for i in range(cycle_len)}
    count = random.randint(3, 5)

    for _ in range(80):
        height = random.randint(max(10, cycle_len + 5), 22)
        width = random.randint(max(10, cycle_len + 5), 24)
        corner = random.choice(("tl", "tr", "bl", "br"))
        legend_bbox = _legend_bbox_for_corner(height, width, cycle_len, corner)
        specs = _sample_object_specs(cycle, count)
        scene = _place_objects(height, width, legend_bbox, specs)
        if scene is None:
            continue
        grid = scene.copy()
        _paint_legend(grid, mapping, corner)
        if _validate_generated(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-cycle stepping input")
