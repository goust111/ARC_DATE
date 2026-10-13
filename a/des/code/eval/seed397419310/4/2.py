import random
from collections import deque

import numpy as np


GREY = 5
COLOR_POOL = tuple(range(10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _interior_cells(coords: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch_h = r1 - r0 + 1
    patch_w = c1 - c0 + 1
    solid = np.zeros((patch_h, patch_w), dtype=bool)
    solid[coords[:, 0] - r0, coords[:, 1] - c0] = True

    exterior = np.zeros_like(solid, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for rr in range(patch_h):
        for cc in (0, patch_w - 1):
            if not solid[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                q.append((rr, cc))
    for cc in range(patch_w):
        for rr in (0, patch_h - 1):
            if not solid[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                q.append((rr, cc))

    while q:
        cr, cc = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < patch_h and 0 <= nc < patch_w and not solid[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    inside = np.argwhere(~solid & ~exterior)
    return [(int(r0 + rr), int(c0 + cc)) for rr, cc in inside]


def _find_axis_line(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    row_candidates = [
        r
        for r in range(h)
        if int(grid[r, 0]) != bg
        and np.all(grid[r, :] == grid[r, 0])
        and r > 0
        and r < h - 1
    ]
    col_candidates = [
        c
        for c in range(w)
        if int(grid[0, c]) != bg
        and np.all(grid[:, c] == grid[0, c])
        and c > 0
        and c < w - 1
    ]

    if len(row_candidates) == 1 and not col_candidates:
        axis = row_candidates[0]
        return "H", axis, int(grid[axis, 0])
    if len(col_candidates) == 1 and not row_candidates:
        axis = col_candidates[0]
        return "V", col_candidates[0], int(grid[0, col_candidates[0]])
    raise AssertionError("expected exactly one full grey divider")


def _half_arrays(
    grid: np.ndarray, orientation: str, axis: int
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    if orientation == "H":
        return [(grid[:axis, :], (0, 0)), (grid[axis + 1 :, :], (axis + 1, 0))]
    return [(grid[:, :axis], (0, 0)), (grid[:, axis + 1 :], (0, axis + 1))]


def _half_kind(half: np.ndarray, bg: int, divider: int) -> str:
    colors = [int(c) for c in np.unique(half) if int(c) not in {bg, divider}]
    if not colors:
        return "empty"

    max_area = 0
    wall_like = 0
    for color in colors:
        for comp in _components(half == color):
            area = len(comp)
            max_area = max(max_area, area)
            if area > 1 and _interior_cells(comp):
                wall_like += 1

    if wall_like > 0:
        return "template"
    if max_area == 1:
        return "seed"
    return "other"


def _mirror_coord(r: int, c: int, orientation: str, axis: int) -> tuple[int, int]:
    if orientation == "H":
        return 2 * axis - r, c
    return r, 2 * axis - c


def _extract_walls(
    template_half: np.ndarray,
    offset: tuple[int, int],
    bg: int,
    divider: int,
) -> list[dict[str, object]]:
    off_r, off_c = offset
    walls: list[dict[str, object]] = []
    colors = [int(c) for c in np.unique(template_half) if int(c) not in {bg, divider}]
    for color in colors:
        for comp in _components(template_half == color):
            interior = _interior_cells(comp)
            if not interior:
                continue
            wall_cells = [(int(r + off_r), int(c + off_c)) for r, c in comp]
            inside_cells = [(int(r + off_r), int(c + off_c)) for r, c in interior]
            walls.append({"color": color, "wall": wall_cells, "interior": inside_cells})
    return walls


def _seed_pixels(
    seed_half: np.ndarray,
    offset: tuple[int, int],
    bg: int,
    divider: int,
) -> list[tuple[tuple[int, int], int]]:
    off_r, off_c = offset
    colors = [int(c) for c in np.unique(seed_half) if int(c) not in {bg, divider}]
    seeds: list[tuple[tuple[int, int], int]] = []
    for color in colors:
        comps = _components(seed_half == color)
        assert len(comps) == 1
        assert len(comps[0]) == 1
        r, c = map(int, comps[0][0])
        seeds.append(((r + off_r, c + off_c), color))
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror template walls onto the seed side and fill them with the seed colors."""
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    orientation, axis, divider = _find_axis_line(g, bg)
    halves = _half_arrays(g, orientation, axis)

    kinds = [_half_kind(half, bg, divider) for half, _ in halves]
    if kinds == ["template", "seed"]:
        template_half, template_offset = halves[0]
        seed_half, seed_offset = halves[1]
    elif kinds == ["seed", "template"]:
        template_half, template_offset = halves[1]
        seed_half, seed_offset = halves[0]
    else:
        raise AssertionError("could not distinguish template side from seed side")

    walls = _extract_walls(template_half, template_offset, bg, divider)
    assert walls

    owner: dict[tuple[int, int], int] = {}
    for idx, wall in enumerate(walls):
        for cell in wall["interior"]:
            assert cell not in owner
            owner[cell] = idx

    seeds = _seed_pixels(seed_half, seed_offset, bg, divider)
    assert seeds

    out = g.copy()
    for (r, c), _color in seeds:
        out[r, c] = bg

    claimed: set[int] = set()
    for (r, c), color in seeds:
        mr, mc = _mirror_coord(r, c, orientation, axis)
        wall_idx = owner.get((mr, mc))
        assert wall_idx is not None, "every seed must mirror into exactly one wall interior"
        assert wall_idx not in claimed, "multiple seeds mapped to the same template wall"
        claimed.add(wall_idx)
        wall = walls[wall_idx]
        for rr, cc in wall["wall"] + wall["interior"]:
            tr, tc = _mirror_coord(rr, cc, orientation, axis)
            out[tr, tc] = color

    return out


def _random_frame_patch(side_h: int, side_w: int) -> dict[str, object] | None:
    max_outer_h = min(7, side_h - 2)
    max_outer_w = min(8, side_w - 2)
    if max_outer_h < 4 or max_outer_w < 4:
        return None

    outer_h = random.randint(4, max_outer_h)
    outer_w = random.randint(4, max_outer_w)

    for _ in range(40):
        hole_h = random.randint(1, outer_h - 2)
        hole_w = random.randint(1, outer_w - 2)
        if hole_h * hole_w < 2:
            continue
        hole_r0 = random.randint(1, outer_h - hole_h - 1)
        hole_c0 = random.randint(1, outer_w - hole_w - 1)

        wall_cells = []
        interior_cells = []
        for r in range(outer_h):
            for c in range(outer_w):
                inside_hole = hole_r0 <= r < hole_r0 + hole_h and hole_c0 <= c < hole_c0 + hole_w
                if inside_hole:
                    interior_cells.append((r, c))
                else:
                    wall_cells.append((r, c))
        return {
            "shape": (outer_h, outer_w),
            "wall": wall_cells,
            "interior": interior_cells,
        }
    return None


def _place_walls(side_h: int, side_w: int, wall_count: int) -> list[dict[str, object]] | None:
    blocked = np.zeros((side_h, side_w), dtype=bool)
    walls: list[dict[str, object]] = []

    for _ in range(wall_count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            patch = _random_frame_patch(side_h, side_w)
            if patch is None:
                return None

            patch_h, patch_w = patch["shape"]
            min_r = 1 if side_h - patch_h >= 2 else 0
            max_r = side_h - patch_h - 1 if side_h - patch_h >= 2 else side_h - patch_h
            min_c = 1 if side_w - patch_w >= 2 else 0
            max_c = side_w - patch_w - 1 if side_w - patch_w >= 2 else side_w - patch_w
            if max_r < min_r or max_c < min_c:
                continue

            r0 = random.randint(min_r, max_r)
            c0 = random.randint(min_c, max_c)
            br0 = max(0, r0 - 1)
            bc0 = max(0, c0 - 1)
            br1 = min(side_h, r0 + patch_h + 1)
            bc1 = min(side_w, c0 + patch_w + 1)
            if np.any(blocked[br0:br1, bc0:bc1]):
                continue

            blocked[br0:br1, bc0:bc1] = True
            walls.append(
                {
                    "wall": [(r0 + r, c0 + c) for r, c in patch["wall"]],
                    "interior": [(r0 + r, c0 + c) for r, c in patch["interior"]],
                }
            )
            placed = True
            break
        if not placed:
            return None

    return walls


def _local_to_global(
    r: int,
    c: int,
    orientation: str,
    axis: int,
    template_first: bool,
) -> tuple[int, int]:
    if orientation == "H":
        return (r, c) if template_first else (axis + 1 + r, c)
    return (r, c) if template_first else (r, axis + 1 + c)


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["H", "V"])
    template_first = random.choice([False, True])

    if orientation == "H":
        side_h = random.randint(8, 11)
        width = random.randint(14, 20)
        height = 2 * side_h + 1
        axis = side_h
        side_shape = (side_h, width)
    else:
        height = random.randint(14, 20)
        side_w = random.randint(8, 11)
        width = 2 * side_w + 1
        axis = side_w
        side_shape = (height, side_w)

    palette = [c for c in COLOR_POOL if c != GREY]
    bg = random.choice(palette)
    remaining = [c for c in COLOR_POOL if c not in {bg, GREY}]
    wall_color = random.choice(remaining)
    seed_colors = [c for c in remaining if c != wall_color]

    side_h, side_w = side_shape
    count_choices = [2, 3] + ([4] if side_h * side_w >= 170 else [])
    wall_count = random.choice(count_choices)
    walls = _place_walls(side_h, side_w, wall_count)
    if walls is None or len(seed_colors) < wall_count:
        return None

    grid = np.full((height, width), bg, dtype=int)
    if orientation == "H":
        grid[axis, :] = GREY
    else:
        grid[:, axis] = GREY

    chosen_seed_colors = random.sample(seed_colors, wall_count)
    for wall, seed_color in zip(walls, chosen_seed_colors):
        for r, c in wall["wall"]:
            gr, gc = _local_to_global(r, c, orientation, axis, template_first)
            grid[gr, gc] = wall_color

        anchor_r, anchor_c = random.choice(wall["interior"])
        anchor_gr, anchor_gc = _local_to_global(anchor_r, anchor_c, orientation, axis, template_first)
        seed_r, seed_c = _mirror_coord(anchor_gr, anchor_gc, orientation, axis)
        if grid[seed_r, seed_c] != bg:
            return None
        grid[seed_r, seed_c] = seed_color

    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if len(order) < 2 or counts[order[0]] == counts[order[1]]:
        return None
    if int(values[order[0]]) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
