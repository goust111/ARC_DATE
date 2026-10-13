import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
ARC_MAX = 30
GEN_RETRIES = 400

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

WALL_CASES = {
    "single": [("U",), ("D",), ("L",), ("R",)],
    "adjacent": [("U", "L"), ("U", "R"), ("D", "L"), ("D", "R")],
    "opposite": [("U", "D"), ("L", "R")],
    "triple": [("U", "D", "L"), ("U", "D", "R"), ("U", "L", "R"), ("D", "L", "R")],
    "quad": [("U", "D", "L", "R")],
}
CASE_NAMES = tuple(WALL_CASES)

BASE_SHAPES = [
    np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (1, 2)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)], dtype=int),
]


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def normalize(coords: np.ndarray) -> np.ndarray:
    base = coords.min(axis=0)
    return np.array(sorted(map(tuple, coords - base)), dtype=int)


def transform_shape(coords: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    pts = coords.copy()
    if flip:
        pts[:, 1] *= -1
    for _ in range(rot % 4):
        pts = np.column_stack([pts[:, 1], -pts[:, 0]])
    return normalize(pts)


def infer_grey(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    border_positions = [(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
    border_positions += [(r, 0) for r in range(1, h - 1)] + [(r, w - 1) for r in range(1, h - 1)]
    counts: dict[int, int] = {}
    for r, c in border_positions:
        color = int(grid[r, c])
        if color == bg:
            continue
        counts[color] = counts.get(color, 0) + 1
    assert counts, "expected a non-background border marker"
    grey = max(counts.items(), key=lambda item: item[1])[0]
    return int(grey)


def palette_span(grid: np.ndarray, bg: int, grey: int) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    assert h >= 3 and w >= 5, "grid too small for palette"
    colors: list[int] = []
    coords: set[tuple[int, int]] = set()
    c = 0
    while c < w:
        color = int(grid[1, c])
        if color in (bg, grey):
            break
        colors.append(color)
        coords.add((1, c))
        c += 1
    assert 3 <= len(colors) <= 4, "expected a 3-4 cell palette at row 1"
    assert len(set(colors)) == len(colors), "palette colors must be distinct"
    return colors, coords


def visible_walls(grid: np.ndarray, coords: np.ndarray, grey: int) -> tuple[str, ...]:
    h, w = grid.shape
    hits: list[str] = []
    for name, (dr, dc) in DIRS.items():
        found = False
        for r, c in coords:
            rr, cc = int(r) + dr, int(c) + dc
            while 0 <= rr < h and 0 <= cc < w:
                if int(grid[rr, cc]) == grey:
                    found = True
                    break
                rr += dr
                cc += dc
            if found:
                hits.append(name)
                break
    return tuple(hits)


def arrangement_direction(walls: Iterable[str]) -> tuple[int, int]:
    wall_set = set(walls)
    n = len(wall_set)
    if n == 0:
        return (0, 1)
    if n == 1:
        only = next(iter(wall_set))
        dr, dc = DIRS[only]
        return (-dr, -dc)
    if n == 2:
        if wall_set == {"U", "D"} or wall_set == {"L", "R"}:
            return (0, 1)
        dr = 1 if "U" in wall_set else -1
        dc = 1 if "L" in wall_set else -1
        return (dr, dc)
    if n == 4:
        return (0, 1)
    for name, vec in DIRS.items():
        if name not in wall_set:
            return vec
    raise AssertionError("unreachable")


def copy_step(coords: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    dr, dc = direction
    step_r = 0 if dr == 0 else dr * (h + 1)
    step_c = 0 if dc == 0 else dc * (w + 1)
    return step_r, step_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate each interior object as many times as the number of grey walls it can see, using the corner palette for copy colors.

    Input structure:
    - The background is black and grey wall segments lie on the outer border.
    - A horizontal palette of 3-4 distinct colors starts at row 1, column 0 and does not overlap any grey wall.
    - The remaining non-background interior pixels form several disjoint monochrome 4-connected objects of area 3-5.

    Rule:
    - For each interior object, test the four cardinal directions: if some object pixel reaches a grey wall by stepping cell-by-cell straight in that direction before leaving the grid, that direction counts as a visible wall.
    - Let N be the number of visible wall directions. Choose the copy direction as the direction away from the wall set: the opposite direction for one wall, the diagonal away from two adjacent walls, rightward for two opposite walls, and the unique wall-free direction for three walls (rightward again for four walls).
    - Keep the original object unchanged, then place N copies of the same shape along that copy direction with a one-cell gap between consecutive bounding boxes; recolor copy k with palette color k.
    - Preserve the grey walls and the palette exactly.

    Key insight:
    - The copy count comes from directional line-of-sight to grey border segments rather than physical contact, and the wall geometry independently determines the placement direction while the palette determines copy colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    grey = infer_grey(g, bg)
    palette, palette_cells = palette_span(g, bg, grey)

    mask = np.ones_like(g, dtype=bool)
    for r, c in palette_cells:
        mask[r, c] = False
    object_mask = mask & (g != bg) & (g != grey)
    objects = components_mask(object_mask)

    out = g.copy()
    for coords in sorted(objects, key=lambda pts: bbox(pts)[:2]):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        walls = visible_walls(g, coords, grey)
        n = len(walls)
        if n == 0:
            continue
        assert n <= len(palette), "palette too short for wall count"
        direction = arrangement_direction(walls)
        step_r, step_c = copy_step(coords, direction)
        for k in range(1, n + 1):
            color = int(palette[k - 1])
            shift = np.array([k * step_r, k * step_c], dtype=int)
            placed = coords + shift
            assert np.all((0 <= placed[:, 0]) & (placed[:, 0] < g.shape[0]))
            assert np.all((0 <= placed[:, 1]) & (placed[:, 1] < g.shape[1]))
            assert np.all(out[placed[:, 0], placed[:, 1]] == bg)
            out[placed[:, 0], placed[:, 1]] = color
    return out


def choose_shape(rng: random.Random) -> np.ndarray:
    base = BASE_SHAPES[rng.randrange(len(BASE_SHAPES))]
    return transform_shape(base, rng.randrange(4), bool(rng.randrange(2)))


def choose_walls(rng: random.Random) -> tuple[str, ...]:
    case_name = CASE_NAMES[rng.randrange(len(CASE_NAMES))]
    options = WALL_CASES[case_name]
    return tuple(options[rng.randrange(len(options))])


def placement_envelope(top: int, left: int, shape: np.ndarray, direction: tuple[int, int], copies: int) -> tuple[int, int, int, int]:
    shifted = []
    step_r, step_c = copy_step(shape, direction)
    for k in range(copies + 1):
        delta = np.array([k * step_r, k * step_c], dtype=int)
        shifted.append(shape + np.array([top, left], dtype=int) + delta)
    all_pts = np.vstack(shifted)
    return bbox(all_pts)


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def build_walls(grid: np.ndarray, specs: list[dict[str, object]]) -> None:
    top_ranges = []
    bottom_ranges = []
    left_ranges = []
    right_ranges = []
    for spec in specs:
        walls = set(spec["walls"])
        r0, c0, r1, c1 = spec["bbox"]
        if "U" in walls:
            top_ranges.append((c0, c1))
        if "D" in walls:
            bottom_ranges.append((c0, c1))
        if "L" in walls:
            left_ranges.append((r0, r1))
        if "R" in walls:
            right_ranges.append((r0, r1))
    for c0, c1 in top_ranges:
        grid[0, c0 : c1 + 1] = GREY
    for c0, c1 in bottom_ranges:
        grid[-1, c0 : c1 + 1] = GREY
    for r0, r1 in left_ranges:
        grid[r0 : r1 + 1, 0] = GREY
    for r0, r1 in right_ranges:
        grid[r0 : r1 + 1, -1] = GREY


def validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > ARC_MAX:
        return False
    bg = get_bg(grid)
    grey = infer_grey(grid, bg)
    palette, palette_cells = palette_span(grid, bg, grey)
    if any(int(grid[1, len(palette) + offset]) != bg for offset in range(0, max(0, min(2, grid.shape[1] - len(palette))))):
        return False
    border_non_bg = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if (r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)) and int(grid[r, c]) not in (bg, grey)
    ]
    if set(border_non_bg) - {(1, c) for c in range(len(palette)) if c == 0}:
        return False

    mask = np.ones_like(grid, dtype=bool)
    for pos in palette_cells:
        mask[pos] = False
    objs = components_mask(mask & (grid != bg) & (grid != grey))
    if not objs:
        return False
    palette_colors = set(palette)
    object_colors = []
    seen_rows = []
    seen_cols = []
    for comp in objs:
        if not (3 <= len(comp) <= 5):
            return False
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            return False
        color = next(iter(colors))
        if color in palette_colors:
            return False
        object_colors.append(color)
        r0, c0, r1, c1 = bbox(comp)
        if r0 <= 0 or c0 <= 0 or r1 >= grid.shape[0] - 1 or c1 >= grid.shape[1] - 1:
            return False
        for rr0, rr1 in seen_rows:
            if not (r1 < rr0 or rr1 < r0):
                return False
        for cc0, cc1 in seen_cols:
            if not (c1 < cc0 or cc1 < c0):
                return False
        seen_rows.append((r0, r1))
        seen_cols.append((c0, c1))
        walls = visible_walls(grid, comp, grey)
        if len(walls) == 0 or len(walls) > len(palette):
            return False
    if len(set(object_colors)) != len(object_colors):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = rng.randint(2, 4)
    max_wall_need = 4
    palette_len = 4 if rng.random() < 0.7 else 3

    chosen_specs = []
    wall_counts = []
    for _ in range(obj_count):
        walls = choose_walls(rng)
        wall_counts.append(len(walls))
        chosen_specs.append({"walls": walls, "shape": choose_shape(rng)})
    if max(wall_counts) > palette_len:
        if palette_len == 3:
            chosen_specs = []
            wall_counts = []
            for _ in range(obj_count):
                family = CASE_NAMES[rng.randrange(len(CASE_NAMES) - 1)]
                walls = WALL_CASES[family][rng.randrange(len(WALL_CASES[family]))]
                chosen_specs.append({"walls": tuple(walls), "shape": choose_shape(rng)})
                wall_counts.append(len(walls))
        palette_len = max(palette_len, max(wall_counts))
    if palette_len not in (3, 4):
        return None

    h = rng.randint(18, 31)
    w = rng.randint(18, 31)
    if h > ARC_MAX or w > ARC_MAX:
        return None

    envelopes: list[tuple[int, int, int, int]] = []
    row_ranges: list[tuple[int, int]] = []
    col_ranges: list[tuple[int, int]] = []
    placed_specs: list[dict[str, object]] = []
    for spec in chosen_specs:
        shape = spec["shape"]
        walls = tuple(spec["walls"])
        direction = arrangement_direction(walls)
        copies = len(walls)
        sr0, sc0, sr1, sc1 = bbox(shape)
        shape_h = sr1 - sr0 + 1
        shape_w = sc1 - sc0 + 1
        placed = False
        for _ in range(200):
            top = rng.randint(2, h - shape_h - 1)
            left = rng.randint(1, w - shape_w - 1)
            comp_bbox = (top, left, top + shape_h - 1, left + shape_w - 1)
            if any(not (comp_bbox[2] < rr0 or rr1 < comp_bbox[0]) for rr0, rr1 in row_ranges):
                continue
            if any(not (comp_bbox[3] < cc0 or cc1 < comp_bbox[1]) for cc0, cc1 in col_ranges):
                continue
            env = placement_envelope(top, left, shape, direction, copies)
            if env[0] < 2 or env[1] < 1 or env[2] >= h - 1 or env[3] >= w - 1:
                continue
            if any(overlaps(env, other, margin=1) for other in envelopes):
                continue
            if any((1, cc) in {(1, i) for i in range(palette_len)} for cc in range(env[1], env[3] + 1) if env[0] <= 1 <= env[2]):
                continue
            placed_specs.append(
                {
                    "shape": shape,
                    "walls": walls,
                    "direction": direction,
                    "copies": copies,
                    "top": top,
                    "left": left,
                    "bbox": comp_bbox,
                }
            )
            envelopes.append(env)
            row_ranges.append((comp_bbox[0], comp_bbox[2]))
            col_ranges.append((comp_bbox[1], comp_bbox[3]))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), BG, dtype=int)
    all_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(all_colors)
    palette = all_colors[:palette_len]
    object_colors = all_colors[palette_len : palette_len + obj_count]
    if len(object_colors) < obj_count:
        return None

    for c, color in enumerate(palette):
        grid[1, c] = color

    build_walls(grid, placed_specs)
    if grid[1, 0] == GREY:
        return None

    for spec, color in zip(placed_specs, object_colors):
        shape = spec["shape"] + np.array([spec["top"], spec["left"]], dtype=int)
        grid[shape[:, 0], shape[:, 1]] = color

    if not validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
