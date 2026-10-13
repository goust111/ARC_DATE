from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
LEGEND = np.array([[RED, BLUE], [GREEN, YELLOW]], dtype=int)

# --------------- solver ---------------


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_legend(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            if not np.array_equal(grid[r0 : r0 + 2, c0 : c0 + 2], LEGEND):
                continue
            if r0 == 0 or c0 == 0 or r0 == h - 2 or c0 == w - 2:
                candidates.append((r0, c0))
    assert len(candidates) == 1, f"expected one edge legend, got {candidates}"
    return candidates[0]


def _parity_color(legend: np.ndarray, height: int, width: int) -> int:
    row_idx = 0 if height % 2 == 1 else 1
    col_idx = 0 if width % 2 == 1 else 1
    return int(legend[row_idx, col_idx])


def _extract_objects(grid: np.ndarray, legend_pos: tuple[int, int]) -> list[dict]:
    r0, c0 = legend_pos
    fg = grid != BG
    fg = fg.copy()
    fg[r0 : r0 + 2, c0 : c0 + 2] = False

    objects: list[dict] = []
    for coords in _components(fg):
        br0, bc0, br1, bc1 = _bbox(coords)
        rel = coords - np.array([br0, bc0], dtype=int)
        objects.append(
            {
                "coords": coords,
                "rel": rel,
                "bbox": (br0, bc0, br1, bc1),
                "height": br1 - br0 + 1,
                "width": bc1 - bc0 + 1,
                "area": int(len(coords)),
            }
        )
    assert objects, "expected at least one foreground object besides the legend"
    return objects


def _stack_objects(objects: list[dict], legend: np.ndarray, canvas_shape: tuple[int, int]) -> np.ndarray:
    out = np.full(canvas_shape, BG, dtype=int)
    h, w = canvas_shape
    ordered = sorted(objects, key=lambda obj: (-obj["area"], obj["bbox"][0], obj["bbox"][1]))

    for obj in ordered:
        top = (h - obj["height"]) // 2
        left = (w - obj["width"]) // 2
        color = _parity_color(legend, obj["height"], obj["width"])
        rr = top + obj["rel"][:, 0]
        cc = left + obj["rel"][:, 1]
        out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object according to bbox parity and center-stack the result by area.

    Input structure:
    - Black background with several disjoint 4-connected foreground objects.
    - A 2x2 legend on the grid edge encodes colors for
      odd-by-odd, odd-by-even, even-by-odd, and even-by-even bbox sizes.

    Rule:
    - Find each foreground object excluding the legend.
    - Measure the object's bounding-box height and width.
    - Recolor the whole object using the corresponding legend entry.
    - Sort objects by area descending.
    - Center every recolored object on a blank canvas of the same size, painting
      large objects first and small objects last.

    Key insight:
    - The 2x2 legend is a direct lookup table for bbox height/width parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_pos = _find_legend(g)
    legend = g[legend_pos[0] : legend_pos[0] + 2, legend_pos[1] : legend_pos[1] + 2]
    objects = _extract_objects(g, legend_pos)
    return _stack_objects(objects, legend, g.shape)


# --------------- generator ---------------

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (16, 22)
OBJECT_COUNT_RANGE = (4, 5)
ODD_SIZES = (3, 5)
EVEN_SIZES = (4, 6)
EXTRA_OBJECT_COMBO_PROB = 0.5
OBJECT_COLORS = (5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _random_path(
    start: tuple[int, int], end: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    while (r, c) != (tr, tc):
        choices: list[tuple[int, int]] = []
        if r < tr:
            choices.append((r + 1, c))
        elif r > tr:
            choices.append((r - 1, c))
        if c < tc:
            choices.append((r, c + 1))
        elif c > tc:
            choices.append((r, c - 1))
        r, c = rng.choice(choices)
        path.append((r, c))
    return path


def _shape_frontier(cells: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return list(frontier)


def _generate_shape(height: int, width: int, rng: random.Random) -> np.ndarray:
    center = (rng.randrange(height), rng.randrange(width))
    anchors = [
        (0, rng.randrange(width)),
        (height - 1, rng.randrange(width)),
        (rng.randrange(height), 0),
        (rng.randrange(height), width - 1),
    ]
    cells = {center}
    for anchor in anchors:
        cells.update(_random_path(center, anchor, rng))

    min_area = max(len(cells), (height * width + 1) // 2)
    if rng.random() < 0.25:
        target_area = height * width
    else:
        target_area = rng.randint(min_area, height * width)

    while len(cells) < target_area:
        frontier = _shape_frontier(cells, height, width)
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_dims(parity: tuple[int, int], rng: random.Random) -> tuple[int, int]:
    height = rng.choice(ODD_SIZES if parity[0] else EVEN_SIZES)
    width = rng.choice(ODD_SIZES if parity[1] else EVEN_SIZES)
    return int(height), int(width)


def _sample_objects(rng: random.Random) -> list[dict] | None:
    base_combos = [(1, 1), (1, 0), (0, 1), (0, 0)]
    combos = base_combos[:]
    if rng.random() < EXTRA_OBJECT_COMBO_PROB:
        combos.append(rng.choice(base_combos))
    rng.shuffle(combos)

    colors = rng.sample(list(OBJECT_COLORS), len(combos))
    objects: list[dict] = []
    used_areas: set[int] = set()

    for color, combo in zip(colors, combos):
        for _ in range(40):
            height, width = _sample_dims(combo, rng)
            mask = _generate_shape(height, width, rng)
            area = int(mask.sum())
            if area in used_areas:
                continue
            used_areas.add(area)
            objects.append(
                {
                    "mask": mask,
                    "height": height,
                    "width": width,
                    "area": area,
                    "color": int(color),
                }
            )
            break
        else:
            return None
    return objects


def _place_legend(
    grid: np.ndarray, blocked: np.ndarray, side: str, rng: random.Random
) -> tuple[int, int]:
    h, w = grid.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - 2)
    elif side == "bottom":
        r0, c0 = h - 2, rng.randint(0, w - 2)
    elif side == "left":
        r0, c0 = rng.randint(0, h - 2), 0
    else:
        r0, c0 = rng.randint(0, h - 2), w - 2

    grid[r0 : r0 + 2, c0 : c0 + 2] = LEGEND
    blocked[max(0, r0 - 1) : min(h, r0 + 3), max(0, c0 - 1) : min(w, c0 + 3)] = True
    return r0, c0


def _candidate_positions(blocked: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - height + 1):
        for c0 in range(w - width + 1):
            if blocked[r0 : r0 + height, c0 : c0 + width].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_object(grid: np.ndarray, obj: dict, top: int, left: int) -> None:
    patch = grid[top : top + obj["height"], left : left + obj["width"]]
    patch[obj["mask"]] = obj["color"]


def _expected_output(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    solver_objects = []
    for obj in objects:
        coords = np.argwhere(obj["mask"]).astype(int)
        solver_objects.append(
            {
                "rel": coords,
                "bbox": (0, 0, obj["height"] - 1, obj["width"] - 1),
                "height": obj["height"],
                "width": obj["width"],
                "area": obj["area"],
            }
        )
    return _stack_objects(solver_objects, LEGEND, shape)


def _background_is_dominant(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts)) == BG


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    side = rng.choice(["top", "bottom", "left", "right"])

    objects = _sample_objects(rng)
    if objects is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    _place_legend(grid, blocked, side, rng)

    placed_objects: list[dict] = []
    for obj in sorted(objects, key=lambda item: item["height"] * item["width"], reverse=True):
        positions = _candidate_positions(blocked, obj["height"], obj["width"])
        if not positions:
            return None
        top, left = rng.choice(positions)
        _paint_object(grid, obj, top, left)
        blocked[
            max(0, top - 1) : min(height, top + obj["height"] + 1),
            max(0, left - 1) : min(width, left + obj["width"] + 1),
        ] = True
        placed_objects.append(obj)

    if not _background_is_dominant(grid):
        return None

    expected = _expected_output(placed_objects, grid.shape)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
