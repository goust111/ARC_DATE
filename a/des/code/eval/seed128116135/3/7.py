import random
from collections import Counter, deque

import numpy as np


FENCE_COLOR = 5
ALL_COLORS = tuple(range(10))
DIRECTIONS = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Base templates point to the right; rotations provide the other head directions.
BASE_SHAPES_EAST = (
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 0]], dtype=bool),
)

OUTER_SIZE_OPTIONS = (7, 9, 11)
GRID_SIDE_RANGE = (22, 30)
OBJECT_COUNT_RANGE = (2, 4)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR_VECS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rotate_to_direction(mask: np.ndarray, direction: str) -> np.ndarray:
    turns = {"right": 0, "up": 1, "left": 2, "down": 3}[direction]
    return np.rot90(mask, turns)


def _shape_head_direction(mask: np.ndarray) -> str:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    counts = {
        "up": int(cropped[0, :].sum()),
        "down": int(cropped[-1, :].sum()),
        "left": int(cropped[:, 0].sum()),
        "right": int(cropped[:, -1].sum()),
    }
    heads = [direction for direction, count in counts.items() if count == 1]
    assert len(heads) == 1, "shape must have exactly one head edge"
    return heads[0]


def _gap_for_box(r0: int, c0: int, r1: int, c1: int, side: str) -> tuple[int, int]:
    if side == "up":
        return r0, c0 + (c1 - c0) // 2
    if side == "down":
        return r1, c0 + (c1 - c0) // 2
    if side == "left":
        return r0 + (r1 - r0) // 2, c0
    return r0 + (r1 - r0) // 2, c1


def _extract_fences(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    fences = []
    for comp in _components(g == FENCE_COLOR):
        r0, c0, r1, c1 = _bbox(comp)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        assert box_h >= 5 and box_w >= 5
        local = np.zeros((box_h, box_w), dtype=bool)
        local[comp[:, 0] - r0, comp[:, 1] - c0] = True

        border = np.zeros_like(local, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(local <= border), "fence pixels must stay on the rectangle border"

        missing = np.argwhere(border & ~local)
        assert len(missing) == 1, "each fence must have exactly one border gap"
        mr, mc = map(int, missing[0])
        if mr == 0:
            exit_side = "up"
            assert 0 < mc < box_w - 1 and mc == box_w // 2
        elif mr == box_h - 1:
            exit_side = "down"
            assert 0 < mc < box_w - 1 and mc == box_w // 2
        elif mc == 0:
            exit_side = "left"
            assert 0 < mr < box_h - 1 and mr == box_h // 2
        else:
            exit_side = "right"
            assert 0 < mr < box_h - 1 and mr == box_h // 2

        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        shape_mask = (inside != bg) & (inside != FENCE_COLOR)
        shape_components = _components(shape_mask)
        assert len(shape_components) == 1, "each fence interior must contain one shape"
        shape_coords = shape_components[0]
        abs_rows = shape_coords[:, 0] + r0 + 1
        abs_cols = shape_coords[:, 1] + c0 + 1
        shape_colors = np.unique(g[abs_rows, abs_cols])
        assert len(shape_colors) == 1, "shape must be monochrome"
        shape_color = int(shape_colors[0])
        area = int(len(shape_coords))

        local_shape = np.zeros_like(inside, dtype=bool)
        local_shape[shape_coords[:, 0], shape_coords[:, 1]] = True
        head_side = _shape_head_direction(local_shape)

        gap_r, gap_c = _gap_for_box(r0, c0, r1, c1, exit_side)
        fences.append(
            {
                "bbox": (r0, c0, r1, c1),
                "exit_side": exit_side,
                "gap": (gap_r, gap_c),
                "head_side": head_side,
                "shape_color": shape_color,
                "shape_area": area,
            }
        )
    assert fences, "expected at least one fence"
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit a colored ray from a fence gap when the enclosed shape points at that gap.

    Input structure:
    - The background is the most common color.
    - Grey pixels form rectangular one-cell-thick fences with exactly one one-cell gap.
    - Each fence contains one monochrome connected shape whose head direction is given
      by the unique bbox edge containing exactly one shape pixel.

    Rule:
    - Find the side of each fence gap.
    - Find the head direction of the enclosed shape.
    - If the two directions match, draw a ray outside the fence from the first cell
      beyond the gap, using the shape color and a length equal to the shape area.
    - Otherwise keep the fence unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for fence in _extract_fences(g):
        if fence["exit_side"] != fence["head_side"]:
            continue
        dr, dc = DIR_VECS[fence["exit_side"]]
        rr = fence["gap"][0] + dr
        cc = fence["gap"][1] + dc
        for _ in range(fence["shape_area"]):
            assert 0 <= rr < h and 0 <= cc < w, "ray must stay inside the grid"
            out[rr, cc] = fence["shape_color"]
            rr += dr
            cc += dc
    return out


def _branch_flags(count: int) -> list[bool]:
    if count == 2:
        flags = [True, False]
    elif count == 3:
        flags = [True, False, random.choice([True, False])]
    else:
        flags = [True, True, False, False]
    random.shuffle(flags)
    return flags


def _placement_offsets(inner: int, span: int) -> list[int]:
    slack = inner - span
    assert slack >= 0
    if slack >= 2:
        return list(range(1, slack))
    return list(range(slack + 1))


def _reserve_cells(
    h: int,
    w: int,
    top: int,
    left: int,
    box_h: int,
    box_w: int,
    ray_cells: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(top - 1, top + box_h + 1):
        for c in range(left - 1, left + box_w + 1):
            if 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
    for r, c in ray_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    cells.add((rr, cc))
    return cells


def _draw_fence(grid: np.ndarray, top: int, left: int, box_h: int, box_w: int, exit_side: str) -> tuple[int, int]:
    bottom = top + box_h - 1
    right = left + box_w - 1
    grid[top, left : right + 1] = FENCE_COLOR
    grid[bottom, left : right + 1] = FENCE_COLOR
    grid[top : bottom + 1, left] = FENCE_COLOR
    grid[top : bottom + 1, right] = FENCE_COLOR
    gap = _gap_for_box(top, left, bottom, right, exit_side)
    grid[gap] = _most_common_color(grid)
    return gap


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        if not spec["match"]:
            continue
        dr, dc = DIR_VECS[spec["exit_side"]]
        rr = spec["gap"][0] + dr
        cc = spec["gap"][1] + dc
        for _ in range(spec["area"]):
            out[rr, cc] = spec["color"]
            rr += dr
            cc += dc
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    bg = random.choice([color for color in ALL_COLORS if color != FENCE_COLOR])
    grid = np.full((h, w), bg, dtype=int)

    count = random.randint(*OBJECT_COUNT_RANGE)
    flags = _branch_flags(count)
    palette = [color for color in ALL_COLORS if color not in {bg, FENCE_COLOR}]
    colors = random.sample(palette, count)

    reserved = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []

    for index in range(count):
        exit_side = random.choice(DIRECTIONS)
        match = flags[index]
        head_side = exit_side if match else random.choice([d for d in DIRECTIONS if d != exit_side])
        base = random.choice(BASE_SHAPES_EAST)
        shape = _rotate_to_direction(base, head_side)
        shape_h, shape_w = shape.shape
        area = int(shape.sum())

        box_h_choices = [size for size in OUTER_SIZE_OPTIONS if size - 2 >= shape_h + 1]
        box_w_choices = [size for size in OUTER_SIZE_OPTIONS if size - 2 >= shape_w + 1]
        if not box_h_choices or not box_w_choices:
            return None
        box_h = random.choice(box_h_choices)
        box_w = random.choice(box_w_choices)
        inner_h = box_h - 2
        inner_w = box_w - 2
        row_offsets = _placement_offsets(inner_h, shape_h)
        col_offsets = _placement_offsets(inner_w, shape_w)

        placed = False
        for _ in range(300):
            max_top = h - box_h - 1
            max_left = w - box_w - 1
            if max_top < 1 or max_left < 1:
                return None
            top = random.randint(1, max_top)
            left = random.randint(1, max_left)
            bottom = top + box_h - 1
            right = left + box_w - 1
            gap = _gap_for_box(top, left, bottom, right, exit_side)

            ray_cells: list[tuple[int, int]] = []
            if match:
                dr, dc = DIR_VECS[exit_side]
                rr = gap[0] + dr
                cc = gap[1] + dc
                ok = True
                for _step in range(area):
                    if not (0 <= rr < h and 0 <= cc < w):
                        ok = False
                        break
                    ray_cells.append((rr, cc))
                    rr += dr
                    cc += dc
                if not ok:
                    continue

            reserve_cells = _reserve_cells(h, w, top, left, box_h, box_w, ray_cells)
            if any(reserved[r, c] for r, c in reserve_cells):
                continue

            shape_top = top + 1 + random.choice(row_offsets)
            shape_left = left + 1 + random.choice(col_offsets)
            color = colors[index]

            _draw_fence(grid, top, left, box_h, box_w, exit_side)
            for rr, cc in np.argwhere(shape):
                grid[shape_top + int(rr), shape_left + int(cc)] = color

            for r, c in reserve_cells:
                reserved[r, c] = True
            specs.append(
                {
                    "exit_side": exit_side,
                    "head_side": head_side,
                    "match": match,
                    "gap": gap,
                    "area": area,
                    "color": color,
                }
            )
            placed = True
            break
        if not placed:
            return None

    if not any(spec["match"] for spec in specs):
        return None
    if not any(not spec["match"] for spec in specs):
        return None

    expected = _expected_output(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
