import random
from collections import deque

import numpy as np


DIRS = ("up", "right", "down", "left")
MISSING_CORNER_TO_DIR = {
    (0, 0): "up",
    (0, 1): "right",
    (1, 1): "down",
    (1, 0): "left",
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tile_template(template: np.ndarray, height: int, width: int) -> np.ndarray:
    reps_r = (height + template.shape[0] - 1) // template.shape[0]
    reps_c = (width + template.shape[1] - 1) // template.shape[1]
    return np.tile(template, (reps_r, reps_c))[:height, :width]


def _find_edge_templates(grid: np.ndarray, bg: int) -> dict[str, np.ndarray]:
    h, w = grid.shape
    templates: dict[str, np.ndarray] = {}
    for coords in _components_from_mask(grid != bg):
        if coords.shape[0] != 9:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            continue
        sides = []
        if r0 == 0:
            sides.append("up")
        if r1 == h - 1:
            sides.append("down")
        if c0 == 0:
            sides.append("left")
        if c1 == w - 1:
            sides.append("right")
        if len(sides) == 1:
            templates[sides[0]] = patch.copy()
    assert set(templates) == set(DIRS), "expected one 3x3 template on each edge"
    return templates


def _find_regions(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    regions: list[tuple[int, int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components_from_mask(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 5 or width < 5:
                continue
            if coords.shape[0] != 2 * (height + width) - 4:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                np.all(patch[0, :] == color)
                and np.all(patch[-1, :] == color)
                and np.all(patch[:, 0] == color)
                and np.all(patch[:, -1] == color)
            ):
                continue
            if np.any(patch[1:-1, 1:-1] == color):
                continue
            regions.append((r0, c0, r1, c1))
    assert regions, "expected at least one rectangular border region"
    regions.sort()
    return regions


def _line_direction(coords: np.ndarray, interior_shape: tuple[int, int]) -> str:
    h, w = interior_shape
    if np.all(coords[:, 1] == coords[0, 1]):
        top = int(coords[:, 0].min())
        bottom = int(coords[:, 0].max())
        dist_top = top
        dist_bottom = (h - 1) - bottom
        if dist_top == dist_bottom:
            return "down" if top < h / 2 else "up"
        return "down" if dist_top < dist_bottom else "up"
    left = int(coords[:, 1].min())
    right = int(coords[:, 1].max())
    dist_left = left
    dist_right = (w - 1) - right
    if dist_left == dist_right:
        return "right" if left < w / 2 else "left"
    return "right" if dist_left < dist_right else "left"


def _seed_direction(coords: np.ndarray, interior_shape: tuple[int, int]) -> str:
    rows = len(set(map(int, coords[:, 0])))
    cols = len(set(map(int, coords[:, 1])))
    if rows == 1 or cols == 1:
        return _line_direction(coords, interior_shape)

    r0, c0, r1, c1 = _bbox(coords)
    assert (r1 - r0, c1 - c0) == (1, 1), "L seed must fit inside a 2x2 box"
    block = np.zeros((2, 2), dtype=bool)
    block[coords[:, 0] - r0, coords[:, 1] - c0] = True
    missing = np.argwhere(~block)
    assert missing.shape == (1, 2), "L seed must occupy exactly three cells of a 2x2 block"
    return MISSING_CORNER_TO_DIR[tuple(map(int, missing[0]))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangle with the 3x3 edge template selected by the seed's direction.

    The input contains:
    - a dominant background color,
    - four 3x3 multicolor templates attached to the four grid edges,
    - several hollow rectangular border regions,
    - exactly one small seed shape inside each region.

    The output preserves the outer scene and borders, and tiles the chosen edge
    template across each rectangle interior from its top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    templates = _find_edge_templates(g, bg)
    out = g.copy()

    for r0, c0, r1, c1 in _find_regions(g, bg):
        ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
        interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        seed_components = _components_from_mask(interior != bg)
        assert len(seed_components) == 1, "each region must contain exactly one seed component"
        seed = seed_components[0]
        direction = _seed_direction(seed, interior.shape)
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = _tile_template(
            templates[direction], interior.shape[0], interior.shape[1]
        )

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_SEED = 2
GEN_BORDER = 5
GEN_TEMPLATE_COLORS = (1, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
REGION_COUNT_RANGE = (4, 5)
INTERIOR_H_RANGE = (3, 7)
INTERIOR_W_RANGE = (3, 8)
PLACEMENT_RETRIES = 180
GEN_RETRIES = 400


def _random_template(used: set[bytes]) -> np.ndarray | None:
    for _ in range(100):
        n_colors = random.randint(2, 4)
        colors = random.sample(GEN_TEMPLATE_COLORS, n_colors)
        patch = np.array([random.choice(colors) for _ in range(9)], dtype=int).reshape(3, 3)
        if len(set(map(int, patch.flat))) < 2:
            continue
        sig = patch.tobytes()
        if sig in used:
            continue
        used.add(sig)
        return patch
    return None


def _template_boxes(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "up": (0, random.randint(3, w - 6), 2, 0),
        "down": (h - 3, random.randint(3, w - 6), 2, 0),
        "left": (random.randint(3, h - 6), 0, 0, 2),
        "right": (random.randint(3, h - 6), w - 3, 0, 2),
    }


def _paint_template(grid: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    grid[top : top + 3, left : left + 3] = patch


def _reserve(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _line_seed(direction: str, ih: int, iw: int) -> list[tuple[int, int]]:
    if direction in {"left", "right"}:
        row = random.randint(0, ih - 1)
        starts = []
        for start in range(iw - 1):
            dist_left = start
            dist_right = (iw - 1) - (start + 1)
            if direction == "right" and dist_left < dist_right:
                starts.append(start)
            if direction == "left" and dist_right < dist_left:
                starts.append(start)
        start = random.choice(starts)
        return [(row, start), (row, start + 1)]

    col = random.randint(0, iw - 1)
    starts = []
    for start in range(ih - 1):
        dist_top = start
        dist_bottom = (ih - 1) - (start + 1)
        if direction == "down" and dist_top < dist_bottom:
            starts.append(start)
        if direction == "up" and dist_bottom < dist_top:
            starts.append(start)
    start = random.choice(starts)
    return [(start, col), (start + 1, col)]


def _l_seed(direction: str, ih: int, iw: int) -> list[tuple[int, int]]:
    top = random.randint(0, ih - 2)
    left = random.randint(0, iw - 2)
    missing = {
        "up": (0, 0),
        "right": (0, 1),
        "down": (1, 1),
        "left": (1, 0),
    }[direction]
    cells = []
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                cells.append((top + dr, left + dc))
    return cells


def _make_seed(direction: str, shape: str, ih: int, iw: int) -> list[tuple[int, int]]:
    if shape == "line":
        return _line_seed(direction, ih, iw)
    return _l_seed(direction, ih, iw)


def _expected_output(
    grid: np.ndarray,
    templates: dict[str, np.ndarray],
    regions: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        direction = region["direction"]
        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_template(templates[direction], ih, iw)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), GEN_BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    boxes = _template_boxes(h, w)
    templates: dict[str, np.ndarray] = {}
    used_templates: set[bytes] = set()
    for side in DIRS:
        patch = _random_template(used_templates)
        if patch is None:
            return None
        templates[side] = patch
        box = boxes[side]
        top = box[0]
        left = box[1]
        _paint_template(grid, top, left, patch)
        _reserve(occ, top, left, top + 2, left + 2, pad=1)

    n_regions = random.randint(*REGION_COUNT_RANGE)
    directions = list(DIRS)
    random.shuffle(directions)
    while len(directions) < n_regions:
        directions.append(random.choice(DIRS))
    random.shuffle(directions)

    shapes = ["L", "line", "L", "line"]
    while len(shapes) < n_regions:
        shapes.append(random.choice(["L", "line"]))
    random.shuffle(shapes)

    regions: list[dict[str, object]] = []
    for direction, shape in zip(directions, shapes):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            ih = random.randint(*INTERIOR_H_RANGE)
            iw = random.randint(*INTERIOR_W_RANGE)
            oh = ih + 2
            ow = iw + 2
            r0 = random.randint(1, h - oh - 1)
            c0 = random.randint(1, w - ow - 1)
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            if occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            seed = _make_seed(direction, shape, ih, iw)
            grid[r0 : r1 + 1, c0] = GEN_BORDER
            grid[r0 : r1 + 1, c1] = GEN_BORDER
            grid[r0, c0 : c1 + 1] = GEN_BORDER
            grid[r1, c0 : c1 + 1] = GEN_BORDER
            for sr, sc in seed:
                grid[r0 + 1 + sr, c0 + 1 + sc] = GEN_SEED
            _reserve(occ, r0, c0, r1, c1, pad=1)
            regions.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "direction": direction,
                    "shape": shape,
                }
            )
            placed = True
            break
        if not placed:
            return None

    if _most_common_color(grid) != GEN_BG:
        return None

    expected = _expected_output(grid, templates, regions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
