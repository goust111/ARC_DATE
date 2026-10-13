from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOLE_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))

GRID_ROWS = (18, 28)
GRID_COLS = (18, 28)
OBJECT_COUNT_OPTIONS = (3, 4)
MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECT_PLACEMENT_TRIES = 500
MAX_OBJECT_SHAPE_TRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_holes(mask: np.ndarray) -> int:
    bg = ~mask
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(bg.shape, nr, nc) and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_border = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _component_for_color(grid: np.ndarray, color: int) -> np.ndarray:
    comps = _components(grid == int(color))
    assert len(comps) == 1, "each object color must form exactly one component"
    return comps[0]


def _local_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each holed object toward its grey head by exactly its hole count, then erase the heads.

    Input structure:
    - The background is a uniform canvas containing several separated monochrome 4-connected objects.
    - Each object has one to four enclosed background holes inside it, and each object color is unique.
    - Exactly one grey single-pixel head sits outside each object, 4-neighbor adjacent to that object on one side.

    Rule:
    - For each object, count the number of enclosed 4-connected background holes in that object's shape.
    - Use the grey head only to read the movement direction: move the whole object in the direction from the object toward its head by that hole count.
    - Erase all grey heads and clear every object's old location back to the background.

    Key insight:
    - The head specifies direction only, while the distance is encoded internally by the object's own hole count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == BG), "expected black background"
    bg = BG

    head_coords = np.argwhere(g == GREY)
    assert len(head_coords) > 0

    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, GREY)]
    assert len(colors) == len(head_coords), "one object per head"

    objects = {color: _component_for_color(g, color) for color in colors}

    head_by_color: dict[int, tuple[int, int, int, int]] = {}
    for hr, hc in head_coords:
        adj = [
            (nr, nc)
            for nr, nc in _neighbors4(int(hr), int(hc))
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) not in (bg, GREY)
        ]
        assert len(adj) == 1, "each grey head must touch exactly one object pixel"
        nr, nc = adj[0]
        color = int(g[nr, nc])
        dr = int(hr) - nr
        dc = int(hc) - nc
        assert (dr, dc) in DIRS4
        assert color not in head_by_color, "each object must have exactly one head"
        head_by_color[color] = (int(hr), int(hc), dr, dc)

    assert set(head_by_color) == set(objects)

    out = np.full_like(g, bg)
    occupied = np.zeros_like(g, dtype=bool)
    for color, coords in objects.items():
        _hr, _hc, dr, dc = head_by_color[color]
        holes = _count_holes(_local_mask_from_coords(coords))
        assert 1 <= holes <= 4
        shift = np.array([dr * holes, dc * holes], dtype=int)
        target = coords + shift
        assert np.all(target[:, 0] >= 0) and np.all(target[:, 0] < g.shape[0])
        assert np.all(target[:, 1] >= 0) and np.all(target[:, 1] < g.shape[1])
        assert not np.any(occupied[target[:, 0], target[:, 1]])
        occupied[target[:, 0], target[:, 1]] = True
        out[target[:, 0], target[:, 1]] = color
    return out


def _hole_options(height: int, width: int) -> list[tuple[tuple[int, int], ...]]:
    options = []
    for hh, ww in HOLE_SHAPES:
        for r0 in range(1, height - hh):
            for c0 in range(1, width - ww):
                cells = tuple((r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww))
                options.append(cells)
    return options


def _hole_separated(cells: tuple[tuple[int, int], ...], used: tuple[tuple[int, int], ...]) -> bool:
    for r0, c0 in cells:
        for r1, c1 in used:
            if abs(r0 - r1) + abs(c0 - c1) < 2:
                return False
    return True


def _search_holes(
    options: list[tuple[tuple[int, int], ...]],
    remaining: int,
    used: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[int, int], ...]] | None:
    if remaining == 0:
        return []
    pool = options[:]
    random.shuffle(pool)
    for cells in pool:
        if not _hole_separated(cells, used):
            continue
        tail = _search_holes(options, remaining - 1, used + cells)
        if tail is not None:
            return [cells] + tail
    return None


def _feasible_dims(n_holes: int) -> list[tuple[int, int]]:
    cache = getattr(_feasible_dims, "_cache", {})
    if n_holes not in cache:
        dims = []
        for height in range(5, 11):
            for width in range(5, 11):
                if _search_holes(_hole_options(height, width), n_holes, ()) is not None:
                    dims.append((height, width))
        cache[n_holes] = dims
        _feasible_dims._cache = cache
    return list(cache[n_holes])


def _make_object_mask(n_holes: int) -> np.ndarray:
    dims = _feasible_dims(n_holes)
    random.shuffle(dims)
    for height, width in dims:
        options = _hole_options(height, width)
        for _ in range(MAX_OBJECT_SHAPE_TRIES):
            holes = _search_holes(options, n_holes, ())
            if holes is None:
                break
            mask = np.ones((height, width), dtype=bool)
            for hole in holes:
                for r, c in hole:
                    mask[r, c] = False
            if _count_holes(mask) == n_holes and len(_components(mask)) == 1:
                return mask
    raise RuntimeError(f"failed to build object with {n_holes} holes")


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([top, left], dtype=int)


def _head_for_side(mask: np.ndarray, top: int, left: int, direction: tuple[int, int]) -> tuple[int, int]:
    height, width = mask.shape
    dr, dc = direction
    if dr == -1:
        return top - 1, left + random.randrange(width)
    if dr == 1:
        return top + height, left + random.randrange(width)
    if dc == -1:
        return top + random.randrange(height), left - 1
    return top + random.randrange(height), left + width


def _far_enough(existing: list[tuple[int, int]], candidate: list[tuple[int, int]], gap: int = 1) -> bool:
    for r0, c0 in existing:
        for r1, c1 in candidate:
            if max(abs(r0 - r1), abs(c0 - c1)) <= gap:
                return False
    return True


def _render_scene(
    shape: tuple[int, int],
    placements: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    for spec in placements:
        color = spec["color"]
        coords = spec["coords"]
        target = spec["target"]
        head = spec["head"]
        inp[coords[:, 0], coords[:, 1]] = color
        inp[head] = GREY
        out[target[:, 0], target[:, 1]] = color
    return inp, out


def _projections_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _has_axis_distractor(placements: list[dict]) -> bool:
    for i, spec in enumerate(placements):
        r0, c0, r1, c1 = spec["bbox"]
        dr, dc = spec["direction"]
        for j, other in enumerate(placements):
            if i == j:
                continue
            or0, oc0, or1, oc1 = other["bbox"]
            if dr == 0 and _projections_overlap(r0, r1, or0, or1):
                return True
            if dc == 0 and _projections_overlap(c0, c1, oc0, oc1):
                return True
    return False


def _semantic_scene_ok(inp: np.ndarray, expected: np.ndarray, placements: list[dict]) -> bool:
    if int(np.count_nonzero(inp == BG)) <= inp.size // 2:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    dirs = {tuple(spec["direction"]) for spec in placements}
    if not any(dr == 0 for dr, dc in dirs):
        return False
    if not any(dc == 0 for dr, dc in dirs):
        return False
    hole_counts = {int(spec["holes"]) for spec in placements}
    if len(hole_counts) < len(placements):
        return False
    if not _has_axis_distractor(placements):
        return False
    return True


def _place_objects(
    shape: tuple[int, int],
    specs: list[tuple[np.ndarray, int, int, tuple[int, int]]],
) -> list[dict] | None:
    h, w = shape
    placements: list[dict] = []
    input_taken: list[tuple[int, int]] = []
    output_taken: list[tuple[int, int]] = []

    order = list(specs)
    random.shuffle(order)
    order.sort(key=lambda item: int(item[0].sum()), reverse=True)

    for mask, holes, color, direction in order:
        mh, mw = mask.shape
        dr, dc = direction
        placed = False
        for _ in range(MAX_OBJECT_PLACEMENT_TRIES):
            top = random.randrange(h - mh + 1)
            left = random.randrange(w - mw + 1)
            coords = _coords_from_mask(mask, top, left)
            head = _head_for_side(mask, top, left, direction)
            if not _in_bounds(shape, head[0], head[1]):
                continue
            shift = np.array([dr * holes, dc * holes], dtype=int)
            target = coords + shift
            if np.any(target[:, 0] < 0) or np.any(target[:, 0] >= h):
                continue
            if np.any(target[:, 1] < 0) or np.any(target[:, 1] >= w):
                continue

            cand_input = [tuple(rc) for rc in coords.tolist()]
            cand_input.append(head)
            cand_output = [tuple(rc) for rc in target.tolist()]
            if not _far_enough(input_taken, cand_input, gap=1):
                continue
            if not _far_enough(output_taken, cand_output, gap=1):
                continue

            bbox = _bbox(coords)
            placements.append(
                {
                    "mask": mask,
                    "holes": holes,
                    "color": color,
                    "direction": direction,
                    "coords": coords,
                    "target": target,
                    "head": head,
                    "bbox": bbox,
                }
            )
            input_taken.extend(cand_input)
            output_taken.extend(cand_output)
            placed = True
            break
        if not placed:
            return None
    return placements


def _generation_attempt() -> np.ndarray | None:
    rows = random.randrange(*GRID_ROWS)
    cols = random.randrange(*GRID_COLS)
    n_objects = random.choices(OBJECT_COUNT_OPTIONS, weights=(1, 4), k=1)[0]

    if n_objects == 4:
        hole_counts = [1, 2, 3, 4]
    else:
        hole_counts = random.sample([1, 2, 3, 4], 3)

    colors = random.sample(OBJECT_COLORS, n_objects)

    while True:
        directions = [random.choice(DIRS4) for _ in range(n_objects)]
        if len(set(directions)) >= 2 and any(dr == 0 for dr, dc in directions) and any(dc == 0 for dr, dc in directions):
            break

    specs = []
    for holes, color, direction in zip(hole_counts, colors, directions):
        mask = _make_object_mask(holes)
        specs.append((mask, holes, color, direction))

    placements = _place_objects((rows, cols), specs)
    if placements is None:
        return None

    inp, expected = _render_scene((rows, cols), placements)
    if not _semantic_scene_ok(inp, expected, placements):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
