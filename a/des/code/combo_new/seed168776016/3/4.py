from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GEN_RETRIES = 300
MAX_GRID_SIZE = 30
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
HOLE_TO_SIDES = {
    0: (),
    1: ("up",),
    2: ("up", "down"),
    3: ("up", "left", "right"),
    4: ("up", "down", "left", "right"),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q = deque([(rr, cc)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    holes = 0
    h, w = mask.shape
    for comp in _components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _shape_info(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    shapes: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            local_mask = g[r0 : r1 + 1, c0 : c1 + 1] == color
            shapes.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                    "holes": _hole_count(local_mask),
                }
            )
    shapes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return shapes


def _edge_midpoint(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mid_r = r0 + (r1 - r0 + 1) // 2
    mid_c = c0 + (c1 - c0 + 1) // 2
    if side == "up":
        return r0, mid_c
    if side == "down":
        return r1, mid_c
    if side == "left":
        return mid_r, c0
    if side == "right":
        return mid_r, c1
    raise ValueError(f"unknown side {side}")


def _ray_cells(
    bbox: tuple[int, int, int, int],
    side: str,
    length: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    start_r, start_c = _edge_midpoint(bbox, side)
    dr, dc = SIDE_TO_DELTA[side]
    cells: list[tuple[int, int]] = []
    for step in range(length):
        rr = start_r + dr * step
        cc = start_c + dc * step
        if not (0 <= rr < height and 0 <= cc < width):
            break
        cells.append((rr, cc))
    return cells


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    for shape in _shape_info(g):
        color = int(shape["color"])
        area = int(shape["area"])
        holes = int(shape["holes"])
        assert holes in HOLE_TO_SIDES, "expected 0..4 enclosed holes per shape"
        for side in HOLE_TO_SIDES[holes]:
            for index, (r, c) in enumerate(_ray_cells(shape["bbox"], side, area, h, w)):
                cell_value = int(g[r, c])
                if index == 0:
                    if cell_value not in (BG, color):
                        break
                    out[r, c] = color
                    continue
                if cell_value != BG:
                    break
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-color rays from each shape according to its enclosed-hole count, with ray length set by the shape area.

    Input structure:
    - The background is black.
    - The grid contains several separated 4-connected monochrome shapes, each with a different color.
    - Some shapes contain 0 to 4 enclosed black holes, and no shape touches the grid border.

    Rule:
    - For each colored shape, count its enclosed black 4-connected holes and count the shape's own colored pixels.
    - Map the hole count to ray directions: 0 -> none, 1 -> up, 2 -> up and down, 3 -> up plus left and right, 4 -> all four cardinal directions.
    - Use the shape area as the ray length, including the starting pixel.
    - Start each ray at the midpoint of the corresponding side of the shape's bounding box; when the midpoint lies between cells, round downward or rightward.
    - Extend the ray in that direction until the planned length is exhausted, the grid edge is reached, or another original shape pixel would be covered. Keep the original shapes and their holes unchanged.

    Key insight:
    - Two independent shape properties control two different output attributes: hole count selects the ray-direction combination, while shape area sets the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= MAX_GRID_SIZE, "ARC grids must stay within 30x30"
    return _transform(g)


def _add_protrusions(mask: np.ndarray, rng: random.Random, max_count: int = 2) -> np.ndarray:
    canvas = np.pad(np.asarray(mask, dtype=bool), 2, constant_values=False)
    for _ in range(rng.randint(0, max_count)):
        rows, cols = np.where(canvas)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        side = rng.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            seg_len = rng.randint(1, min(3, c1 - c0 + 1))
            start = rng.randint(c0, c1 - seg_len + 1)
            depth = rng.randint(1, 2)
            if side == "top" and r0 - depth >= 0:
                canvas[r0 - depth : r0, start : start + seg_len] = True
            if side == "bottom" and r1 + depth < canvas.shape[0]:
                canvas[r1 + 1 : r1 + depth + 1, start : start + seg_len] = True
        else:
            seg_len = rng.randint(1, min(3, r1 - r0 + 1))
            start = rng.randint(r0, r1 - seg_len + 1)
            depth = rng.randint(1, 2)
            if side == "left" and c0 - depth >= 0:
                canvas[start : start + seg_len, c0 - depth : c0] = True
            if side == "right" and c1 + depth < canvas.shape[1]:
                canvas[start : start + seg_len, c1 + 1 : c1 + depth + 1] = True
    return _crop_mask(canvas)


def _grow_polyomino(target_area: int, rng: random.Random) -> np.ndarray | None:
    cells: set[tuple[int, int]] = {(0, 0)}
    for _ in range(target_area * 4):
        if len(cells) >= target_area:
            break
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in DIRS4
            if (r + dr, c + dc) not in cells
        }
        candidates = list(frontier)
        rng.shuffle(candidates)
        placed = False
        for candidate in candidates:
            trial = set(cells)
            trial.add(candidate)
            mask = _mask_from_cells(trial)
            if mask.shape[0] <= 6 and mask.shape[1] <= 6:
                cells = trial
                placed = True
                break
        if not placed:
            return None
    if len(cells) != target_area:
        return None
    return _mask_from_cells(cells)


def _core_holey_mask(holes: int, rng: random.Random) -> np.ndarray:
    if holes == 1:
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        return mask
    if holes == 2:
        if rng.random() < 0.5:
            h = rng.randint(4, 6)
            w = rng.randint(5, 6)
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            wall_col = rng.randint(2, w - 3)
            mask[:, wall_col] = True
        else:
            h = rng.randint(5, 6)
            w = rng.randint(4, 6)
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            wall_row = rng.randint(2, h - 3)
            mask[wall_row, :] = True
        return mask
    if holes == 3:
        h = rng.randint(5, 6)
        w = rng.randint(5, 6)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        wall_col = rng.randint(2, w - 3)
        wall_row = rng.randint(2, h - 3)
        mask[:, wall_col] = True
        if rng.random() < 0.5:
            mask[wall_row, : wall_col + 1] = True
        else:
            mask[wall_row, wall_col:] = True
        return np.rot90(mask, rng.randrange(4)).copy()
    if holes == 4:
        h = rng.randint(5, 6)
        w = rng.randint(5, 6)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        wall_col = rng.randint(2, w - 3)
        wall_row = rng.randint(2, h - 3)
        mask[:, wall_col] = True
        mask[wall_row, :] = True
        return mask
    raise ValueError(f"unsupported hole count {holes}")


def _make_shape(holes: int, rng: random.Random) -> np.ndarray:
    for _ in range(80):
        if holes == 0:
            if rng.random() < 0.5:
                mask = np.ones((rng.randint(2, 4), rng.randint(2, 5)), dtype=bool)
            else:
                target_area = rng.randint(4, 9)
                mask = _grow_polyomino(target_area, rng)
                if mask is None:
                    continue
        else:
            mask = _core_holey_mask(holes, rng)
        mask = _add_protrusions(mask, rng, max_count=2 if holes > 0 else 1)
        mask = _crop_mask(mask)
        if mask.size == 0:
            continue
        if mask.shape[0] > 8 or mask.shape[1] > 8:
            continue
        if len(_components(mask)) != 1:
            continue
        if _hole_count(mask) != holes:
            continue
        area = int(mask.sum())
        if holes == 0 and area < 4:
            continue
        if holes > 0 and area < 8:
            continue
        return mask
    raise RuntimeError(f"failed to build a shape with {holes} holes")


def _expand8(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _shape_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(np.asarray(mask, dtype=bool))
    }


def _relative_shape_data(mask: np.ndarray, holes: int) -> dict[str, object]:
    mask = np.asarray(mask, dtype=bool)
    shape_h, shape_w = mask.shape
    coords_rel = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    shape_set = set(coords_rel)
    pad_rel = sorted(
        {
            (r + dr, c + dc)
            for r, c in coords_rel
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        }
    )
    bbox = (0, 0, shape_h - 1, shape_w - 1)
    ray_rel: set[tuple[int, int]] = set()
    for side in HOLE_TO_SIDES[holes]:
        start_r, start_c = _edge_midpoint(bbox, side)
        dr, dc = SIDE_TO_DELTA[side]
        for step in range(len(coords_rel)):
            rr = start_r + dr * step
            cc = start_c + dc * step
            if (rr, cc) not in shape_set:
                ray_rel.add((rr, cc))
    return {
        "coords_rel": coords_rel,
        "pad_rel": pad_rel,
        "ray_rel": sorted(ray_rel),
    }


def _placement_score(
    bbox: tuple[int, int, int, int],
    sides: tuple[str, ...],
    area: int,
    height: int,
    width: int,
) -> int:
    if not sides:
        return 0
    r0, c0, r1, c1 = bbox
    distances = {
        "up": r0,
        "down": height - 1 - r1,
        "left": c0,
        "right": width - 1 - c1,
    }
    return sum(min(area - 1, distances[side]) for side in sides)


def _try_place_shapes(
    specs: list[dict[str, object]], height: int, width: int, rng: random.Random
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied_shapes: set[tuple[int, int]] = set()
    occupied_padded: set[tuple[int, int]] = set()
    reserved_rays: set[tuple[int, int]] = set()

    ordering = list(specs)
    rng.shuffle(ordering)
    ordering.sort(
        key=lambda spec: (
            -len(HOLE_TO_SIDES[int(spec["holes"])]),
            -int(spec["area"]),
            int(spec["holes"]),
        )
    )

    placements: list[tuple[dict[str, object], int, int, set[tuple[int, int]]]] = []
    for spec in ordering:
        mask = np.asarray(spec["mask"], dtype=bool)
        shape_h, shape_w = mask.shape
        if height - shape_h < 2 or width - shape_w < 2:
            return None

        positions = [
            (top, left)
            for top in range(1, height - shape_h)
            for left in range(1, width - shape_w)
        ]
        rng.shuffle(positions)
        positions.sort(
            key=lambda pos: _placement_score(
                (pos[0], pos[1], pos[0] + shape_h - 1, pos[1] + shape_w - 1),
                HOLE_TO_SIDES[int(spec["holes"])],
                int(spec["area"]),
                height,
                width,
            ),
            reverse=True,
        )

        coords_rel = spec["coords_rel"]
        pad_rel = spec["pad_rel"]
        ray_rel = spec["ray_rel"]

        placed = False
        for top, left in positions:
            bad = False
            cells: set[tuple[int, int]] = set()
            for dr, dc in coords_rel:
                rc = (top + dr, left + dc)
                if rc in occupied_padded or rc in reserved_rays:
                    bad = True
                    break
                cells.add(rc)
            if bad:
                continue

            new_ray_cells: set[tuple[int, int]] = set()
            for dr, dc in ray_rel:
                rr = top + dr
                cc = left + dc
                if not (0 <= rr < height and 0 <= cc < width):
                    continue
                rc = (rr, cc)
                if rc in occupied_shapes or rc in reserved_rays:
                    bad = True
                    break
                new_ray_cells.add(rc)
            if bad:
                continue

            placements.append((spec, top, left, cells))
            occupied_shapes |= cells
            for dr, dc in pad_rel:
                rr = top + dr
                cc = left + dc
                if 0 <= rr < height and 0 <= cc < width:
                    occupied_padded.add((rr, cc))
            reserved_rays |= new_ray_cells
            placed = True
            break
        if not placed:
            return None

    for spec, _top, _left, cells in placements:
        color = int(spec["color"])
        for r, c in cells:
            grid[r, c] = color

    if reserved_rays & occupied_shapes:
        return None
    return grid


def _valid_generated_grid(grid: np.ndarray, expected_holes: list[int]) -> bool:
    if grid is None or grid.ndim != 2 or max(grid.shape) > MAX_GRID_SIZE:
        return False
    shapes = _shape_info(grid)
    if len(shapes) != len(expected_holes):
        return False
    if sorted(int(shape["holes"]) for shape in shapes) != sorted(expected_holes):
        return False
    for shape in shapes:
        r0, c0, r1, c1 = shape["bbox"]
        if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
            return False
    out = _transform(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _fixed_slot_place_shapes(
    specs: list[dict[str, object]], height: int, width: int, rng: random.Random
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied_padded: set[tuple[int, int]] = set()
    slots = [
        (2, 2),
        (2, width - 10),
        (height - 10, 2),
        (height - 10, width - 10),
        ((height - 8) // 2, (width - 8) // 2),
    ]
    rng.shuffle(slots)

    for spec, (base_top, base_left) in zip(specs, slots):
        mask = np.asarray(spec["mask"], dtype=bool)
        shape_h, shape_w = mask.shape
        candidates: list[tuple[int, int]] = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                top = base_top + dr
                left = base_left + dc
                if 1 <= top <= height - shape_h - 1 and 1 <= left <= width - shape_w - 1:
                    candidates.append((top, left))
        rng.shuffle(candidates)

        placed = False
        for top, left in candidates:
            cells = _shape_cells(mask, top, left)
            padded = _expand8(cells, height, width)
            if cells & occupied_padded:
                continue
            color = int(spec["color"])
            for r, c in cells:
                grid[r, c] = color
            occupied_padded |= padded
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.choice([4, 5])
    hole_values = [0, 1, 2, 3, 4]
    if object_count == 4:
        missing = rng.randrange(5)
        hole_values.pop(missing)
    rng.shuffle(hole_values)

    colors = rng.sample(list(range(1, 10)), len(hole_values))
    specs: list[dict[str, object]] = []
    try:
        for color, holes in zip(colors, hole_values):
            mask = _make_shape(holes, rng)
            specs.append(
                {
                    "color": color,
                    "holes": holes,
                    "mask": mask,
                    "area": int(mask.sum()),
                    **_relative_shape_data(mask, holes),
                }
            )
    except RuntimeError:
        return None

    height = rng.randint(28, MAX_GRID_SIZE)
    width = rng.randint(28, MAX_GRID_SIZE)

    grid = _fixed_slot_place_shapes(specs, height, width, rng)
    if grid is None:
        return None
    if not _valid_generated_grid(grid, hole_values):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
