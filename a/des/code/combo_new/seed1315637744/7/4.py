from __future__ import annotations

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
BODY_COLORS = tuple(range(1, 10))
POINTER_AXIS_ODD = (3, 5, 7)
OTHER_AXIS = (2, 3, 4, 5, 6)
MAX_GEN_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((int(cr), int(cc)))
            for dr, dc in DIR4:
                nr, nc = int(cr + dr), int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _parse_object(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict[str, object]:
    pts = np.array(comp, dtype=int)
    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert len(comp) == h * w, "object must fill its bounding box"

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()
    colors, counts = np.unique(patch, return_counts=True)
    color_counts = {int(color): int(count) for color, count in zip(colors, counts)}
    assert len(color_counts) == 2, "object must have one body color and one pointer color"

    pointer_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(pointer_colors) == 1, "pointer must be a single pixel"
    pointer_color = int(pointer_colors[0])
    body_color = int(next(color for color, count in color_counts.items() if count > 1))

    pr_rel, pc_rel = map(int, np.argwhere(patch == pointer_color)[0])
    pr = int(r0 + pr_rel)
    pc = int(c0 + pc_rel)

    directions: list[str] = []
    if w % 2 == 1 and pr == r0 and pc == c0 + w // 2:
        directions.append("up")
    if w % 2 == 1 and pr == r1 and pc == c0 + w // 2:
        directions.append("down")
    if h % 2 == 1 and pc == c0 and pr == r0 + h // 2:
        directions.append("left")
    if h % 2 == 1 and pc == c1 and pr == r0 + h // 2:
        directions.append("right")
    assert len(directions) == 1, "pointer must sit on exactly one edge midpoint"

    return {
        "bbox": (r0, c0, r1, c1),
        "shape": (h, w),
        "patch": patch,
        "body_color": body_color,
        "pointer_color": pointer_color,
        "pointer": (pr, pc),
        "direction": directions[0],
    }


def _find_beacon(
    singletons_by_color: dict[int, list[tuple[int, int]]],
    body_color: int,
    direction: str,
    pointer: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], int]:
    r0, c0, r1, c1 = map(int, bbox)
    pr, pc = map(int, pointer)
    candidates = singletons_by_color.get(int(body_color), [])
    best_beacon: tuple[int, int] | None = None
    best_gap: int | None = None

    for br, bc in candidates:
        br = int(br)
        bc = int(bc)
        gap: int | None = None
        if direction == "up" and bc == pc and br < r0:
            gap = int(r0 - br - 1)
        elif direction == "down" and bc == pc and br > r1:
            gap = int(br - r1 - 1)
        elif direction == "left" and br == pr and bc < c0:
            gap = int(c0 - bc - 1)
        elif direction == "right" and br == pr and bc > c1:
            gap = int(bc - c1 - 1)
        if gap is None or gap < 1:
            continue
        if best_gap is None or gap < best_gap:
            best_gap = int(gap)
            best_beacon = (br, bc)

    assert best_beacon is not None and best_gap is not None, "missing aligned beacon"
    return best_beacon, int(best_gap)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each pointer-marked rectangle until it touches its same-colored beacon.

    Input structure:
    - The background is black.
    - Foreground contains several solid rectangles; each rectangle has one body color plus one embedded single-pixel pointer color.
    - The pointer sits at the midpoint of exactly one rectangle edge and indicates a cardinal direction.
    - Along that outward row or column there is a separate single-pixel beacon of the same color as the rectangle body.

    Rule:
    - Detect every multicolor rectangle and read its direction from the pointer edge midpoint.
    - On the pointer's row or column, find the nearest same-colored singleton beacon in that direction.
    - Let D be the number of black cells between the beacon and the rectangle edge; translate the whole rectangle, including the pointer, by D cells toward the beacon.
    - Keep all beacons fixed and clear the rectangles' original positions to black.

    Key insight:
    - The pointer location encodes direction, while the nearest aligned same-colored singleton encodes the translation distance through the visible gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components4(g != BG)
    objects: list[dict[str, object]] = []
    singletons_by_color: dict[int, list[tuple[int, int]]] = {}

    for comp in comps:
        if len(comp) == 1:
            r, c = comp[0]
            singletons_by_color.setdefault(int(g[r, c]), []).append((int(r), int(c)))
            continue
        objects.append(_parse_object(g, comp))

    if not objects:
        return g.copy()

    out = g.copy()
    for obj in objects:
        r0, c0, r1, c1 = map(int, obj["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for obj in objects:
        _beacon, dist = _find_beacon(
            singletons_by_color=singletons_by_color,
            body_color=int(obj["body_color"]),
            direction=str(obj["direction"]),
            pointer=tuple(obj["pointer"]),
            bbox=tuple(obj["bbox"]),
        )
        dr, dc = DIR_VECS[str(obj["direction"])]
        r0, c0, _r1, _c1 = map(int, obj["bbox"])
        patch = np.asarray(obj["patch"], dtype=int)
        h, w = map(int, patch.shape)
        nr0 = int(r0 + dr * dist)
        nc0 = int(c0 + dc * dist)
        out[nr0 : nr0 + h, nc0 : nc0 + w] = patch

    return out


def _paint_patch(grid: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = map(int, patch.shape)
    grid[top : top + h, left : left + w] = patch


def _pointer_rel(h: int, w: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return 0, w // 2
    if direction == "down":
        return h - 1, w // 2
    if direction == "left":
        return h // 2, 0
    return h // 2, w - 1


def _make_patch(h: int, w: int, body_color: int, pointer_color: int, direction: str) -> np.ndarray:
    patch = np.full((h, w), int(body_color), dtype=int)
    pr, pc = _pointer_rel(h, w, direction)
    patch[int(pr), int(pc)] = int(pointer_color)
    return patch


def _candidate_cells(top: int, left: int, h: int, w: int, beacon: tuple[int, int]) -> set[tuple[int, int]]:
    cells = {(int(top + r), int(left + c)) for r in range(h) for c in range(w)}
    cells.add((int(beacon[0]), int(beacon[1])))
    return cells


def _can_add(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if (nr, nc) in occupied and (nr, nc) not in cells:
                return False
    return True


def _sample_dims(rng: np.random.Generator, direction: str, h: int, w: int) -> tuple[int, int] | None:
    if direction in ("up", "down"):
        w_choices = [value for value in POINTER_AXIS_ODD if value <= w]
        h_choices = [value for value in OTHER_AXIS if value <= h]
        if not w_choices or not h_choices:
            return None
        return int(rng.choice(h_choices)), int(rng.choice(w_choices))
    h_choices = [value for value in POINTER_AXIS_ODD if value <= h]
    w_choices = [value for value in OTHER_AXIS if value <= w]
    if not h_choices or not w_choices:
        return None
    return int(rng.choice(h_choices)), int(rng.choice(w_choices))


def _sample_layout(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    obj_h: int,
    obj_w: int,
    direction: str,
    dist: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if direction == "up":
        max_fr0 = grid_h - obj_h - dist
        if max_fr0 < 1:
            return None
        fr0 = int(rng.integers(1, max_fr0 + 1))
        fc0 = int(rng.integers(0, grid_w - obj_w + 1))
        in_top = int(fr0 + dist)
        in_left = int(fc0)
        beacon = (int(fr0 - 1), int(fc0 + obj_w // 2))
        return (in_top, in_left), (int(fr0), int(fc0)), beacon
    if direction == "down":
        min_fr0 = dist
        max_fr0 = grid_h - obj_h - 1
        if min_fr0 > max_fr0:
            return None
        fr0 = int(rng.integers(min_fr0, max_fr0 + 1))
        fc0 = int(rng.integers(0, grid_w - obj_w + 1))
        in_top = int(fr0 - dist)
        in_left = int(fc0)
        beacon = (int(fr0 + obj_h), int(fc0 + obj_w // 2))
        return (in_top, in_left), (int(fr0), int(fc0)), beacon
    if direction == "left":
        max_fc0 = grid_w - obj_w - dist
        if max_fc0 < 1:
            return None
        fr0 = int(rng.integers(0, grid_h - obj_h + 1))
        fc0 = int(rng.integers(1, max_fc0 + 1))
        in_top = int(fr0)
        in_left = int(fc0 + dist)
        beacon = (int(fr0 + obj_h // 2), int(fc0 - 1))
        return (in_top, in_left), (int(fr0), int(fc0)), beacon
    min_fc0 = dist
    max_fc0 = grid_w - obj_w - 1
    if min_fc0 > max_fc0:
        return None
    fr0 = int(rng.integers(0, grid_h - obj_h + 1))
    fc0 = int(rng.integers(min_fc0, max_fc0 + 1))
    in_top = int(fr0)
    in_left = int(fc0 - dist)
    beacon = (int(fr0 + obj_h // 2), int(fc0 + obj_w))
    return (in_top, in_left), (int(fr0), int(fc0)), beacon


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(12, 31))
    grid_w = int(rng.integers(12, 31))
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    n_objects = int(rng.integers(2, 5))
    body_colors = [int(color) for color in rng.choice(BODY_COLORS, size=n_objects, replace=False)]
    directions = [str(direction) for direction in rng.choice(np.array(list(DIR_VECS.keys())), size=n_objects, replace=False)]

    input_occupied: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()

    for index in range(n_objects):
        direction = directions[index]
        body_color = body_colors[index]
        pointer_choices = [color for color in BODY_COLORS if color != body_color]
        pointer_color = int(rng.choice(pointer_choices))

        placed = False
        for _ in range(120):
            dims = _sample_dims(rng, direction, grid_h - 2, grid_w - 2)
            if dims is None:
                return None
            obj_h, obj_w = dims
            if direction in ("up", "down"):
                max_dist = min(6, grid_h - obj_h - 1)
            else:
                max_dist = min(6, grid_w - obj_w - 1)
            if max_dist < 1:
                continue
            dist = int(rng.integers(1, max_dist + 1))
            layout = _sample_layout(rng, grid_h, grid_w, obj_h, obj_w, direction, dist)
            if layout is None:
                continue
            (in_top, in_left), (out_top, out_left), beacon = layout
            input_cells = _candidate_cells(in_top, in_left, obj_h, obj_w, beacon)
            output_cells = _candidate_cells(out_top, out_left, obj_h, obj_w, beacon)
            if not _can_add(input_cells, input_occupied, grid_h, grid_w):
                continue
            if not _can_add(output_cells, output_occupied, grid_h, grid_w):
                continue

            patch = _make_patch(obj_h, obj_w, body_color, pointer_color, direction)
            _paint_patch(grid, in_top, in_left, patch)
            grid[beacon] = body_color
            input_occupied.update(input_cells)
            output_occupied.update(output_cells)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
