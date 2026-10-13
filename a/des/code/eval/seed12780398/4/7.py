from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
        (r, c - 1), (r, c + 1),
        (r + 1, c - 1), (r + 1, c), (r + 1, c + 1),
    ]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _is_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return False
    coords = np.argwhere(mask)
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return len(seen) == int(mask.sum())


def _parse_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == int(color))]
        border = [(r, c) for r, c in coords if _is_border(r, c, h, w)]
        interior = {(r, c) for r, c in coords if not _is_border(r, c, h, w)}
        assert len(border) == 1, "each object color must have exactly one border marker"
        assert interior, "each marker color must have one interior object"
        assert _is_connected(_cells_to_mask(interior)), "object cells must be 4-connected"

        marker_r, marker_c = border[0]
        if marker_r == 0:
            direction = "up"
        elif marker_r == h - 1:
            direction = "down"
        elif marker_c == 0:
            direction = "left"
        else:
            direction = "right"

        r0, c0, r1, c1 = _bbox(interior)
        objects.append(
            {
                "color": int(color),
                "direction": direction,
                "marker": (int(marker_r), int(marker_c)),
                "cells": interior,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
            }
        )
    assert 3 <= len(objects) <= 4, "expected three or four marked objects"
    return objects


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _object_rays(obj: dict) -> list[list[tuple[int, int]]]:
    cells = obj["cells"]
    height = int(obj["height"])
    width = int(obj["width"])
    direction = obj["direction"]
    rays: list[list[tuple[int, int]]] = []

    if direction == "up":
        fronts = {}
        for r, c in cells:
            fronts[c] = min(fronts.get(c, r), r)
        for c, r in sorted(fronts.items()):
            rays.append([(r - d, c) for d in range(1, width + 1)])
    elif direction == "down":
        fronts = {}
        for r, c in cells:
            fronts[c] = max(fronts.get(c, r), r)
        for c, r in sorted(fronts.items()):
            rays.append([(r + d, c) for d in range(1, width + 1)])
    elif direction == "left":
        fronts = {}
        for r, c in cells:
            fronts[r] = min(fronts.get(r, c), c)
        for r, c in sorted(fronts.items()):
            rays.append([(r, c - d) for d in range(1, height + 1)])
    else:
        fronts = {}
        for r, c in cells:
            fronts[r] = max(fronts.get(r, c), c)
        for r, c in sorted(fronts.items()):
            rays.append([(r, c + d) for d in range(1, height + 1)])
    return rays


def _cast_shadow_rays(
    grid_shape: tuple[int, int],
    rays: list[list[tuple[int, int]]],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = grid_shape
    shadow: set[tuple[int, int]] = set()
    active = [True] * len(rays)
    max_steps = max((len(ray) for ray in rays), default=0)

    for step in range(max_steps):
        proposals: dict[tuple[int, int], list[int]] = {}
        for idx, ray in enumerate(rays):
            if not active[idx] or step >= len(ray):
                continue
            r, c = ray[step]
            cell = (int(r), int(c))
            if not (0 <= r < h and 0 <= c < w):
                active[idx] = False
                continue
            if cell in blocked or cell in shadow:
                active[idx] = False
                continue
            proposals.setdefault(cell, []).append(idx)

        for cell, owners in proposals.items():
            shadow.add(cell)
            if len(owners) > 1:
                for idx in owners:
                    active[idx] = False
    return shadow


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a black shadow from each colored object toward the border marker of the same color.

    The active contour is the face of the object that points toward the marker edge.
    Each row/column ray extends by the object's bbox size in the perpendicular axis.
    Rays stop early when they hit another non-background cell or an already drawn shadow.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _dominant_color(g)
    objects = _parse_objects(g, bg)

    blocked = {(int(r), int(c)) for r, c in np.argwhere(g != bg)}
    rays: list[list[tuple[int, int]]] = []
    for obj in objects:
        rays.extend(_object_rays(obj))

    shadow = _cast_shadow_rays((h, w), rays, blocked)
    out = g.copy()
    for r, c in shadow:
        out[r, c] = BLACK
    return out


# --------------- generator ---------------

GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (15, 22)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
MIN_SHADOW_PIXELS = 2

SHAPE_TEMPLATES = [
    ("111", "111", "110"),
    ("111", "110", "111"),
    ("011", "111", "110"),
    ("1111", "1110", "1111", "0111"),
    ("1111", "1101", "1111", "1110"),
    ("0111", "1111", "1011", "1111"),
    ("11111", "11110", "11111", "01111", "11111"),
    ("11111", "11011", "11111", "11110", "11111"),
    ("01111", "11111", "11101", "11111", "11110"),
]


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out


def _template_mask(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in lines], dtype=bool)


def _random_shape() -> np.ndarray:
    for _ in range(60):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        mask = np.ones((h, w), dtype=bool)
        edge_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)
        ]
        random.shuffle(edge_cells)
        target = random.randint(1, min(4, len(edge_cells) - 1))
        removed = 0
        for r, c in edge_cells:
            if removed >= target:
                break
            mask[r, c] = False
            if (
                _is_connected(mask)
                and mask.any(axis=0).all()
                and mask.any(axis=1).all()
                and int(mask.sum()) >= max(mask.shape) + 1
            ):
                removed += 1
            else:
                mask[r, c] = True
        if removed >= 1:
            return _transform_mask(mask)
    return _transform_mask(_template_mask(random.choice(SHAPE_TEMPLATES)))


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(mask):
        cells.add((int(top + r), int(left + c)))
    return cells


def _expanded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    halo = set()
    for r, c in cells:
        for nr, nc in _neighbors8(r, c) + [(r, c)]:
            if 0 <= nr < h and 0 <= nc < w:
                halo.add((int(nr), int(nc)))
    return halo


def _position_candidates(h: int, w: int, obj_h: int, obj_w: int, direction: str) -> list[tuple[int, int]]:
    rows = list(range(1, h - obj_h))
    cols = list(range(1, w - obj_w))
    base = [(r, c) for r in rows for c in cols]
    if direction == "up":
        preferred = [(r, c) for r, c in base if r >= h // 2 - 1]
    elif direction == "down":
        preferred = [(r, c) for r, c in base if r <= max(1, (h - obj_h) // 2)]
    elif direction == "left":
        preferred = [(r, c) for r, c in base if c >= w // 2 - 1]
    else:
        preferred = [(r, c) for r, c in base if c <= max(1, (w - obj_w) // 2)]
    if preferred:
        base = preferred
    random.shuffle(base)
    return base


def _sample_layout_specs(h: int, w: int, directions: list[str], colors: list[int]) -> list[dict] | None:
    specs: list[dict] = []
    occupied_halo: set[tuple[int, int]] = set()
    order = list(range(len(directions)))
    random.shuffle(order)

    for idx in order:
        direction = directions[idx]
        color = colors[idx]
        mask = _random_shape()
        obj_h, obj_w = map(int, mask.shape)
        placed = False
        for top, left in _position_candidates(h, w, obj_h, obj_w, direction)[:PLACEMENT_ATTEMPTS]:
            cells = _mask_cells(mask, top, left)
            if cells & occupied_halo:
                continue
            specs.append(
                {
                    "color": int(color),
                    "direction": direction,
                    "mask": mask,
                    "top": int(top),
                    "left": int(left),
                    "cells": cells,
                    "height": int(obj_h),
                    "width": int(obj_w),
                }
            )
            occupied_halo |= _expanded(cells, h, w)
            placed = True
            break
        if not placed:
            return None
    return specs


def _shadow_info_for_spec(
    spec: dict,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
) -> dict:
    rays = _object_rays(spec)
    shadow = set()
    truncated = False
    hit_object = False
    for ray in rays:
        produced = 0
        for r, c in ray:
            cell = (int(r), int(c))
            if not (0 <= r < h and 0 <= c < w):
                truncated = True
                break
            if cell in blocked:
                truncated = True
                hit_object = True
                break
            shadow.add(cell)
            produced += 1
        if produced < len(ray):
            truncated = True
    return {
        "shadow": shadow,
        "rays": rays,
        "truncated": truncated,
        "hit_object": hit_object,
    }


def _choose_markers(
    specs: list[dict],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> dict[int, tuple[int, int]] | None:
    used = set()
    markers: dict[int, tuple[int, int]] = {}
    for spec in specs:
        direction = spec["direction"]
        cells = spec["cells"]
        if direction == "up":
            own_axis = {c for _, c in cells}
            candidates = [(0, c) for c in range(1, w - 1) if c not in own_axis]
        elif direction == "down":
            own_axis = {c for _, c in cells}
            candidates = [(h - 1, c) for c in range(1, w - 1) if c not in own_axis]
        elif direction == "left":
            own_axis = {r for r, _ in cells}
            candidates = [(r, 0) for r in range(1, h - 1) if r not in own_axis]
        else:
            own_axis = {r for r, _ in cells}
            candidates = [(r, w - 1) for r in range(1, h - 1) if r not in own_axis]

        candidates = [cell for cell in candidates if cell not in forbidden and cell not in used]
        if not candidates:
            if direction == "up":
                candidates = [(0, c) for c in range(1, w - 1)]
            elif direction == "down":
                candidates = [(h - 1, c) for c in range(1, w - 1)]
            elif direction == "left":
                candidates = [(r, 0) for r in range(1, h - 1)]
            else:
                candidates = [(r, w - 1) for r in range(1, h - 1)]
            candidates = [cell for cell in candidates if cell not in forbidden and cell not in used]
        if not candidates:
            return None
        marker = random.choice(candidates)
        markers[int(spec["color"])] = marker
        used.add(marker)
    return markers


def _render_input(h: int, w: int, specs: list[dict], markers: dict[int, tuple[int, int]]) -> np.ndarray:
    grid = np.full((h, w), GREY, dtype=int)
    for spec in specs:
        for r, c in spec["cells"]:
            grid[r, c] = int(spec["color"])
        mr, mc = markers[int(spec["color"])]
        grid[mr, mc] = int(spec["color"])
    return grid


def _render_output(
    h: int,
    w: int,
    specs: list[dict],
    markers: dict[int, tuple[int, int]],
    shadow_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = _render_input(h, w, specs, markers)
    for r, c in shadow_cells:
        out[r, c] = BLACK
    return out


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    obj_count = random.choice((3, 4))
    missing = random.choice(DIRECTIONS) if obj_count == 3 else None
    directions = [d for d in DIRECTIONS if d != missing]
    random.shuffle(directions)
    colors = random.sample(PALETTE, len(directions))

    specs = _sample_layout_specs(h, w, directions, colors)
    if specs is None:
        return None

    blocked = set().union(*(spec["cells"] for spec in specs))
    infos = [_shadow_info_for_spec(spec, h, w, blocked) for spec in specs]

    shadow_sets = [info["shadow"] for info in infos]
    all_shadow = set().union(*shadow_sets) if shadow_sets else set()
    if len(all_shadow) < max(MIN_SHADOW_PIXELS, len(specs)):
        return None
    if any(not shadow for shadow in shadow_sets):
        return None
    if sum(len(s) for s in shadow_sets) != len(all_shadow):
        return None

    want_truncation = random.random() < 0.5
    has_truncation = any(info["truncated"] for info in infos)
    if want_truncation and not has_truncation:
        return None

    markers = _choose_markers(specs, h, w, blocked | all_shadow)
    if markers is None:
        return None

    for spec in specs:
        spec["marker"] = markers[int(spec["color"])]

    inp = _render_input(h, w, specs, markers)
    expected = _render_output(h, w, specs, markers, all_shadow)

    if _dominant_color(inp) != GREY:
        return None
    if np.any(inp == BLACK):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection input")
