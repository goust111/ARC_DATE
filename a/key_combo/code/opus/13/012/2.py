from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

OBJECT_COLORS = {GREEN, RED}
GRID_SIZE_RANGE = (18, 24)
RECT_COUNT_RANGE = (4, 6)
GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _object_components(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = np.isin(grid, list(OBJECT_COLORS))
    boxes = [_bbox_coords(coords) for coords in _components_mask(mask)]
    boxes.sort()
    return boxes


def _layer_fill(region: np.ndarray, marker_mask: np.ndarray) -> np.ndarray:
    h, w = region.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    depth = np.minimum(
        np.minimum(rows, h - 1 - rows),
        np.minimum(cols, w - 1 - cols),
    )

    out = np.full((h, w), GREEN, dtype=int)
    out[(depth > 0) & (depth % 2 == 1)] = YELLOW
    out[(depth > 0) & (depth % 2 == 0)] = BLUE
    out[marker_mask] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase over-marked rectangles; otherwise repaint them as concentric layers whose depth also set the survival threshold.

    Input structure:
    - The grid has a black background and several separated axis-aligned rectangular green objects.
    - Sparse red marker pixels appear only inside those rectangles' bounding boxes.
    - Each rectangle is isolated from the others by background cells, so each object can be processed independently.

    Rule:
    - For each connected rectangle object, take the bounding box covering its green and red pixels and compute D = floor(min(height, width) / 2).
    - Count the red marker pixels inside that bounding box.
    - If the red count is greater than D, erase that rectangle object to background.
    - Otherwise fill the whole bounding box by rectangular layer depth: border stays green, odd positive depths become yellow, positive even depths become blue, and the original red markers stay red on top.

    Key insight:
    - The same geometric quantity D, derived from the rectangle's smaller side, simultaneously controls whether the object survives and how many alternating concentric layers it can display.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG if np.any(g == BG) else _get_bg(g)
    out = g.copy()

    for r0, c0, r1, c1 in _object_components(g):
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        marker_mask = region == RED
        depth_limit = min(region.shape) // 2
        marker_count = int(marker_mask.sum())

        if marker_count > depth_limit:
            object_mask = np.isin(region, list(OBJECT_COLORS))
            out[r0 : r1 + 1, c0 : c1 + 1][object_mask] = bg
            continue

        out[r0 : r1 + 1, c0 : c1 + 1] = _layer_fill(region, marker_mask)

    return out


def _d_from_dims(h: int, w: int) -> int:
    return min(h, w) // 2


def _interior_cells(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]


def _sample_dims_for_d(rng: random.Random, d: int, min_markers: int) -> tuple[int, int] | None:
    if d == 1:
        min_side_choices = (2, 3)
    elif d == 2:
        min_side_choices = (4, 5)
    elif d == 3:
        min_side_choices = (6, 7)
    else:
        return None

    for _ in range(100):
        min_side = rng.choice(min_side_choices)
        max_side = rng.randint(max(min_side, min_side + 1), 8)
        if rng.random() < 0.5:
            h, w = min_side, max_side
        else:
            h, w = max_side, min_side
        if _d_from_dims(h, w) != d:
            continue
        if len(_interior_cells(h, w)) < min_markers:
            continue
        return h, w
    return None


def _sample_rect_spec(
    rng: random.Random,
    *,
    must_survive: bool,
    d_choices: tuple[int, ...],
    prefer_exact_threshold: bool = False,
) -> dict | None:
    target_d = rng.choice(d_choices)
    min_markers = 0 if must_survive else target_d + 1
    dims = _sample_dims_for_d(rng, target_d, min_markers)
    if dims is None:
        return None

    h, w = dims
    d = _d_from_dims(h, w)
    interior = _interior_cells(h, w)
    max_markers = min(5, len(interior))

    if must_survive:
        cap = min(d, max_markers)
        if cap <= 0:
            marker_count = 0
        elif prefer_exact_threshold and rng.random() < 0.7:
            marker_count = cap
        else:
            marker_count = rng.randint(0, cap)
    else:
        if max_markers <= d:
            return None
        marker_count = rng.randint(d + 1, max_markers)

    return {"h": h, "w": w, "d": d, "marker_count": marker_count, "survives": must_survive}


def _sample_specs(rng: random.Random) -> list[dict] | None:
    n_rects = rng.randint(*RECT_COUNT_RANGE)
    specs: list[dict] = []

    first = _sample_rect_spec(rng, must_survive=True, d_choices=(1,), prefer_exact_threshold=True)
    second = _sample_rect_spec(rng, must_survive=True, d_choices=(2, 3), prefer_exact_threshold=True)
    removed = _sample_rect_spec(rng, must_survive=False, d_choices=(1, 2, 3))
    if first is None or second is None or removed is None:
        return None

    specs.extend([first, second, removed])

    while len(specs) < n_rects:
        survives = rng.random() < 0.6
        d_choices = (1, 2, 3)
        spec = _sample_rect_spec(
            rng,
            must_survive=survives,
            d_choices=d_choices,
            prefer_exact_threshold=survives and rng.random() < 0.4,
        )
        if spec is None:
            return None
        specs.append(spec)

    survivor_ds = {spec["d"] for spec in specs if spec["survives"]}
    if len(survivor_ds) < 2:
        return None
    if not any(not spec["survives"] for spec in specs):
        return None
    if not any(spec["marker_count"] == spec["d"] for spec in specs if spec["survives"]):
        return None

    rng.shuffle(specs)
    return specs


def _fits_with_gap(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _place_specs(
    rng: random.Random,
    h: int,
    w: int,
    specs: list[dict],
) -> list[dict] | None:
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["h"] * specs[idx]["w"], reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_specs: list[dict | None] = [None] * len(specs)

    for idx in order:
        spec = specs[idx]
        positions = []
        for r0 in range(h - spec["h"] + 1):
            for c0 in range(w - spec["w"] + 1):
                box = (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1)
                if _fits_with_gap(box, placed_boxes):
                    positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        box = (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1)
        placed_boxes.append(box)
        placed_specs[idx] = {**spec, "box": box}

    return [spec for spec in placed_specs if spec is not None]


def _choose_marker_cells(rng: random.Random, h: int, w: int, count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    interior = _interior_cells(h, w)
    assert count <= len(interior)
    return rng.sample(interior, count)


def _expected_output(grid: np.ndarray, placed_specs: list[dict]) -> np.ndarray:
    out = np.full_like(grid, BG)
    out[:, :] = grid
    for spec in placed_specs:
        r0, c0, r1, c1 = spec["box"]
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        marker_mask = region == RED
        if spec["marker_count"] > spec["d"]:
            object_mask = np.isin(region, list(OBJECT_COLORS))
            out[r0 : r1 + 1, c0 : c1 + 1][object_mask] = BG
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = _layer_fill(region, marker_mask)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    specs = _sample_specs(rng)
    if specs is None:
        return None

    total_area = sum(spec["h"] * spec["w"] for spec in specs)
    coverage = total_area / float(h * w)
    if not 0.18 <= coverage <= 0.48:
        return None

    placed_specs = _place_specs(rng, h, w, specs)
    if placed_specs is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for spec in placed_specs:
        r0, c0, r1, c1 = spec["box"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREEN
        for rr, cc in _choose_marker_cells(rng, spec["h"], spec["w"], spec["marker_count"]):
            grid[r0 + rr, c0 + cc] = RED

    boxes = _object_components(grid)
    if len(boxes) != len(placed_specs):
        return None

    expected = _expected_output(grid, placed_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None

    survivor_ds = {spec["d"] for spec in placed_specs if spec["marker_count"] <= spec["d"]}
    if len(survivor_ds) < 2:
        return None
    if not any(spec["marker_count"] > spec["d"] for spec in placed_specs):
        return None

    color_counts = Counter(map(int, grid.ravel()))
    if color_counts[BG] <= color_counts[GREEN]:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
