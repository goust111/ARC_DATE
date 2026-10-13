from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

DIR_BY_HOLES = {
    1: (-1, 0),
    2: (0, 1),
    3: (1, 0),
    4: (0, -1),
}

_GEN_COUNTER = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_holes(mask: np.ndarray) -> list[np.ndarray]:
    padded = np.pad(mask, 1, constant_values=False)
    bg_parts = _components(~padded)
    holes: list[np.ndarray] = []
    ph, pw = padded.shape
    for part in bg_parts:
        rr = part[:, 0]
        cc = part[:, 1]
        touches_outer = np.any((rr == 0) | (rr == ph - 1) | (cc == 0) | (cc == pw - 1))
        if touches_outer:
            continue
        holes.append(part - 1)
    return holes


def _is_square_hole(hole_coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_from_coords(hole_coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w:
        return False
    area = hole_coords.shape[0]
    return area == h * w


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        holes = _find_holes(local_mask)
        hole_count = len(holes)
        if hole_count == 0:
            continue
        square_holes = all(_is_square_hole(h) for h in holes)
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        objects.append(
            {
                "coords": coords,
                "coord_set": {tuple(map(int, p)) for p in coords},
                "bbox": (r0, c0, r1, c1),
                "center": center,
                "hole_count": hole_count,
                "square_holes": square_holes,
                "ray_color": RED if square_holes else BLUE,
            }
        )
    return objects


def _cast_ray(grid: np.ndarray, obj: dict) -> tuple[list[tuple[int, int]], str]:
    dr, dc = DIR_BY_HOLES[obj["hole_count"]]
    own = obj["coord_set"]
    r, c = obj["center"]
    cells: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, r, c):
        pos = (r, c)
        if pos in own:
            pass
        elif int(grid[r, c]) == BLACK:
            cells.append(pos)
        else:
            return cells, "object"
        r += dr
        c += dc
    return cells, "boundary"


def _semantic_output(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for obj in _analyze_objects(out):
        ray_cells, _ = _cast_ray(out, obj)
        for r, c in ray_cells:
            out[r, c] = obj["ray_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit one colored ray from each gray holed object, with hole count choosing direction and hole geometry choosing color.

    Input structure:
    - The grid has a black background and several 4-connected gray objects.
    - Each gray object encloses one to four black holes inside its bounding box.
    - Different objects have different hole counts, and every object's holes are either all square or all non-square.

    Rule:
    - Find each gray object and count its enclosed black holes.
    - Map hole count to one ray direction: 1->up, 2->right, 3->down, 4->left.
    - If all holes of that object are square, color its ray red; otherwise color the ray blue.
    - Start from the object's bounding-box center, ignore the object's own gray pixels while marching in the chosen direction, paint traversed black cells, and stop before the first other non-black pixel or at the grid boundary.

    Key insight:
    - The same internal hole structure encodes two independent ray parameters: topology gives direction, while hole geometry gives color.
    """
    g = np.asarray(grid, dtype=int)
    return _semantic_output(g)


def _row_starts(lengths: list[int]) -> list[int]:
    starts = []
    pos = 1
    for length in lengths:
        starts.append(pos)
        pos += length + 1
    return starts


def _build_object(hole_count: int, square_holes: bool) -> dict:
    if hole_count == 1:
        rows, cols, keep = 1, 1, {(0, 0)}
    elif hole_count == 2:
        if int(np.random.randint(2)) == 0:
            rows, cols = 1, 2
        else:
            rows, cols = 2, 1
        keep = {(r, c) for r in range(rows) for c in range(cols)}
    elif hole_count == 3:
        rows, cols = 2, 2
        missing = int(np.random.randint(4))
        keep = {(r, c) for r in range(2) for c in range(2)}
        keep.remove((missing // 2, missing % 2))
    elif hole_count == 4:
        rows, cols = 2, 2
        keep = {(r, c) for r in range(2) for c in range(2)}
    else:
        raise ValueError("hole count must be in 1..4")

    if square_holes:
        side = 1 if np.random.random() < 0.65 else 3
        hole_h = side
        hole_w = side
    else:
        hole_h, hole_w = ((1, 3) if int(np.random.randint(2)) == 0 else (3, 1))

    row_heights = [hole_h] * rows
    col_widths = [hole_w] * cols
    h = sum(row_heights) + rows + 1
    w = sum(col_widths) + cols + 1
    sprite = np.full((h, w), GREY, dtype=int)

    rs = _row_starts(row_heights)
    cs = _row_starts(col_widths)
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in keep:
                continue
            r0 = rs[r]
            c0 = cs[c]
            sprite[r0 : r0 + row_heights[r], c0 : c0 + col_widths[c]] = BLACK

    center = (h // 2, w // 2)
    return {
        "hole_count": hole_count,
        "square_holes": square_holes,
        "sprite": sprite,
        "h": h,
        "w": w,
        "center_is_grey": int(sprite[center]) == GREY,
    }


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_sep = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_sep = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return max(row_sep, col_sep)


def _sprite_bbox(center: tuple[int, int], spec: dict) -> tuple[int, int, int, int]:
    r, c = center
    return (r - spec["h"] // 2, c - spec["w"] // 2, r + spec["h"] // 2, c + spec["w"] // 2)


def _sample_center(
    spec: dict,
    shape: tuple[int, int],
    *,
    row_band: tuple[float, float],
    col_band: tuple[float, float],
    fixed_row: int | None = None,
    fixed_col: int | None = None,
) -> tuple[int, int] | None:
    h, w = shape
    half_h = spec["h"] // 2
    half_w = spec["w"] // 2
    row_lo = half_h
    row_hi = h - half_h - 1
    col_lo = half_w
    col_hi = w - half_w - 1
    row_lo = max(row_lo, int(np.floor((h - 1) * row_band[0])))
    row_hi = min(row_hi, int(np.ceil((h - 1) * row_band[1])))
    col_lo = max(col_lo, int(np.floor((w - 1) * col_band[0])))
    col_hi = min(col_hi, int(np.ceil((w - 1) * col_band[1])))
    if fixed_row is not None:
        if not (row_lo <= fixed_row <= row_hi):
            return None
        row_lo = row_hi = fixed_row
    if fixed_col is not None:
        if not (col_lo <= fixed_col <= col_hi):
            return None
        col_lo = col_hi = fixed_col
    if row_lo > row_hi or col_lo > col_hi:
        return None
    rr = int(np.random.randint(row_lo, row_hi + 1))
    cc = int(np.random.randint(col_lo, col_hi + 1))
    return rr, cc


def _try_place(
    placed: dict[int, tuple[int, int]],
    counts: list[int],
    specs: dict[int, dict],
    shape: tuple[int, int],
    hole_count: int,
    *,
    row_band: tuple[float, float],
    col_band: tuple[float, float],
    fixed_row: int | None = None,
    fixed_col: int | None = None,
    tries: int = 120,
) -> bool:
    spec = specs[hole_count]
    for _ in range(tries):
        center = _sample_center(
            spec,
            shape,
            row_band=row_band,
            col_band=col_band,
            fixed_row=fixed_row,
            fixed_col=fixed_col,
        )
        if center is None:
            continue
        box = _sprite_bbox(center, spec)
        ok = True
        for other in counts:
            if other not in placed:
                continue
            other_box = _sprite_bbox(placed[other], specs[other])
            if _bbox_gap(box, other_box) < 2:
                ok = False
                break
        if ok:
            placed[hole_count] = center
            return True
    return False


def _render_scene(shape: tuple[int, int], counts: list[int], specs: dict[int, dict], centers: dict[int, tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for hole_count in counts:
        spec = specs[hole_count]
        sprite = spec["sprite"]
        r0, c0, _, _ = _sprite_bbox(centers[hole_count], spec)
        grid[r0 : r0 + spec["h"], c0 : c0 + spec["w"]] = sprite
    return grid


def _choose_counts(call_idx: int) -> list[int]:
    pattern = call_idx % 4
    if pattern == 0:
        return [1, 2, 3, 4]
    missing = [4, 2, 3][pattern - 1]
    return [count for count in [1, 2, 3, 4] if count != missing]


def _independent_validate(grid: np.ndarray, expected_counts: list[int]) -> np.ndarray | None:
    objects = _analyze_objects(grid)
    if len(objects) != len(expected_counts):
        return None

    hole_counts = sorted(obj["hole_count"] for obj in objects)
    if hole_counts != sorted(expected_counts):
        return None
    if len(set(hole_counts)) != len(hole_counts):
        return None

    square_flags = [obj["square_holes"] for obj in objects]
    if all(square_flags) or not any(square_flags):
        return None

    center_values = [int(grid[obj["center"]]) for obj in objects]
    if GREY not in center_values or BLACK not in center_values:
        return None

    painted: dict[tuple[int, int], int] = {}
    hit_types = []
    for obj in objects:
        ray_cells, hit_type = _cast_ray(grid, obj)
        if not ray_cells:
            return None
        hit_types.append(hit_type)
        for cell in ray_cells:
            if int(grid[cell]) != BLACK:
                return None
            if cell in painted:
                return None
            painted[cell] = obj["ray_color"]

    if "object" not in hit_types or "boundary" not in hit_types:
        return None
    if RED not in painted.values() or BLUE not in painted.values():
        return None

    out = np.array(grid, copy=True)
    for cell, color in painted.items():
        out[cell] = color
    if not np.array_equal(main(grid), out):
        return None
    return out


def _attempt_scene(call_idx: int) -> np.ndarray | None:
    counts = _choose_counts(call_idx)
    h = int(np.random.randint(16, 23))
    w = int(np.random.randint(16, 23))
    shape = (h, w)

    flags = {count: bool(np.random.randint(2)) for count in counts}
    if all(flags.values()) or not any(flags.values()):
        flip_count = counts[int(np.random.randint(len(counts)))]
        flags[flip_count] = not flags[flip_count]

    specs = {count: _build_object(count, flags[count]) for count in counts}
    placed: dict[int, tuple[int, int]] = {}

    horizontal_possible = 2 in counts and any(count in counts for count in (1, 3))
    vertical_possible = 3 in counts and any(count in counts for count in (2, 4))
    if horizontal_possible and vertical_possible:
        use_horizontal = (call_idx % 2 == 0)
    else:
        use_horizontal = horizontal_possible
    if use_horizontal:
        source = 2
        target = 3 if 3 in counts else 1
    else:
        source = 3
        target = 2 if 2 in counts else 4

    if use_horizontal:
        if not _try_place(placed, counts, specs, shape, source, row_band=(0.28, 0.62), col_band=(0.10, 0.28)):
            return None
        src_r, _ = placed[source]
        if not _try_place(
            placed,
            counts,
            specs,
            shape,
            target,
            row_band=(0.25, 0.68),
            col_band=(0.68, 0.88),
            fixed_row=src_r,
        ):
            return None
    else:
        if not _try_place(placed, counts, specs, shape, source, row_band=(0.10, 0.28), col_band=(0.35, 0.72)):
            return None
        _, src_c = placed[source]
        if not _try_place(
            placed,
            counts,
            specs,
            shape,
            target,
            row_band=(0.68, 0.88),
            col_band=(0.28, 0.78),
            fixed_col=src_c,
        ):
            return None

    for hole_count in counts:
        if hole_count in placed:
            continue
        dr, dc = DIR_BY_HOLES[hole_count]
        if dr < 0:
            row_band, col_band = (0.68, 0.90), (0.10, 0.86)
        elif dr > 0:
            row_band, col_band = (0.10, 0.30), (0.14, 0.90)
        elif dc < 0:
            row_band, col_band = (0.12, 0.88), (0.68, 0.90)
        else:
            row_band, col_band = (0.14, 0.86), (0.10, 0.30)
        if not _try_place(placed, counts, specs, shape, hole_count, row_band=row_band, col_band=col_band):
            return None

    grid = _render_scene(shape, counts, specs, placed)
    out = _independent_validate(grid, counts)
    if out is None:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    base_call_idx = _GEN_COUNTER
    for extra in range(400):
        grid = _attempt_scene(base_call_idx + extra)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
