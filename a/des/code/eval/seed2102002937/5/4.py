from __future__ import annotations

import random

import numpy as np


BG = 0
ANCHOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GRID_SIZES = (19, 21, 23, 25, 27)
SHAPE_COUNT_RANGE = (3, 5)
MAX_HALF_SIZE = 4
GEN_ATTEMPTS = 400
PLACEMENT_TRIES = 180


def _sign(value: float, eps: float = 1e-9) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _shape_center(coords: np.ndarray) -> tuple[int, int, float, float]:
    rows = coords[:, 0].astype(float)
    cols = coords[:, 1].astype(float)
    mean_r = float(rows.mean())
    mean_c = float(cols.mean())

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    if int(r0 + r1) % 2 == 0:
        center_r = int((int(r0) + int(r1)) // 2)
    else:
        center_r = int(round(mean_r))

    if int(c0 + c1) % 2 == 0:
        center_c = int((int(c0) + int(c1)) // 2)
    else:
        center_c = int(round(mean_c))

    return center_r, center_c, mean_r, mean_c


def _extract_shapes(grid: np.ndarray) -> list[dict]:
    shapes: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, ANCHOR)):
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        center_r, center_c, mean_r, mean_c = _shape_center(coords)
        shapes.append(
            {
                "color": int(color),
                "coords": coords.astype(int),
                "area": int(coords.shape[0]),
                "center_r": int(center_r),
                "center_c": int(center_c),
                "mean_r": float(mean_r),
                "mean_c": float(mean_c),
            }
        )
    return shapes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack scattered convex fragments by area and offset each layer by its source-side direction.

    The generator keeps every shape center-compatible with integer translations, but the
    solver itself only assumes:
    - black background,
    - one central grey anchor,
    - one connected monochrome shape per non-black/non-grey color.
    """
    g = np.asarray(grid, dtype=int)
    anchor_cells = np.argwhere(g == ANCHOR)
    assert anchor_cells.shape[0] == 1, "expected exactly one grey anchor"
    anchor_r, anchor_c = map(int, anchor_cells[0])

    shapes = _extract_shapes(g)
    assert shapes, "expected at least one colored shape"

    placed: list[dict] = []
    for shape in shapes:
        dr = _sign(shape["mean_r"] - float(anchor_r))
        dc = _sign(shape["mean_c"] - float(anchor_c))
        rel = shape["coords"].copy()
        rel[:, 0] = rel[:, 0] - int(shape["center_r"]) + int(dr)
        rel[:, 1] = rel[:, 1] - int(shape["center_c"]) + int(dc)
        placed.append(
            {
                "color": int(shape["color"]),
                "area": int(shape["area"]),
                "coords": rel,
            }
        )

    placed.sort(key=lambda item: (-int(item["area"]), int(item["color"])))

    all_rows = np.concatenate([item["coords"][:, 0] for item in placed])
    all_cols = np.concatenate([item["coords"][:, 1] for item in placed])
    min_r, max_r = int(all_rows.min()), int(all_rows.max())
    min_c, max_c = int(all_cols.min()), int(all_cols.max())

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for item in placed:
        rows = item["coords"][:, 0] - min_r
        cols = item["coords"][:, 1] - min_c
        out[rows, cols] = int(item["color"])

    return out


def _build_mask(half_h: int, half_w: int, mode: str) -> np.ndarray:
    top_min = max(0, half_w - half_h)
    if mode == "rectangle":
        top = half_w
    elif mode == "diamond":
        top = top_min
    else:
        top = random.randint(top_min, half_w)

    increments = half_w - top
    if increments == 0:
        raise_steps: set[int] = set()
    elif mode == "diamond":
        raise_steps = set(range(increments))
    elif mode == "plateau":
        raise_steps = set(range(half_h - increments, half_h))
    else:
        raise_steps = set(random.sample(range(half_h), increments))

    top_to_center = [top]
    current = top
    for step in range(half_h):
        if step in raise_steps:
            current += 1
        top_to_center.append(current)

    full_profile = top_to_center + top_to_center[-2::-1]
    mask = np.zeros((2 * half_h + 1, 2 * half_w + 1), dtype=bool)
    mid = half_w
    for r, width in enumerate(full_profile):
        mask[r, mid - width : mid + width + 1] = True
    return mask


def _random_mask() -> np.ndarray:
    half_h = random.randint(1, MAX_HALF_SIZE)
    half_w = random.randint(1, MAX_HALF_SIZE)
    if half_h == half_w:
        mode = random.choice(("rectangle", "diamond", "plateau", "random"))
    else:
        mode = random.choice(("rectangle", "plateau", "random"))
    return _build_mask(half_h, half_w, mode)


def _mask_cells(mask: np.ndarray, center_r: int, center_c: int) -> list[tuple[int, int]]:
    r0 = int(mask.shape[0] // 2)
    c0 = int(mask.shape[1] // 2)
    pts = np.argwhere(mask)
    return [(int(center_r + r - r0), int(center_c + c - c0)) for r, c in pts]


def _add_blocked(blocked: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((int(r + dr), int(c + dc)))


def _axis_choices(anchor: int, size: int, half_extent: int, direction: int) -> list[int]:
    border_lo = half_extent + 1
    border_hi = size - half_extent - 2
    if border_lo > border_hi:
        return []
    if direction < 0:
        hi = anchor - half_extent - 2
        if border_lo > hi:
            return []
        return list(range(border_lo, hi + 1))
    if direction > 0:
        lo = anchor + half_extent + 2
        if lo > border_hi:
            return []
        return list(range(lo, border_hi + 1))
    if not (border_lo <= anchor <= border_hi):
        return []
    return [int(anchor)]


def _place_shapes(specs: list[dict], h: int, w: int) -> np.ndarray | None:
    anchor_r, anchor_c = h // 2, w // 2
    grid = np.full((h, w), BG, dtype=int)
    grid[anchor_r, anchor_c] = ANCHOR

    blocked: set[tuple[int, int]] = set()
    _add_blocked(blocked, [(anchor_r, anchor_c)])

    for spec in sorted(specs, key=lambda item: -int(item["mask"].sum())):
        mask = spec["mask"]
        dr, dc = spec["direction"]
        half_h = int(mask.shape[0] // 2)
        half_w = int(mask.shape[1] // 2)

        row_choices = _axis_choices(anchor_r, h, half_h, int(dr))
        col_choices = _axis_choices(anchor_c, w, half_w, int(dc))
        if not row_choices or not col_choices:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            center_r = random.choice(row_choices)
            center_c = random.choice(col_choices)
            cells = _mask_cells(mask, int(center_r), int(center_c))
            if any((r, c) in blocked for r, c in cells):
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = int(spec["color"])
            _add_blocked(blocked, cells)
            placed = True
            break

        if not placed:
            return None

    return grid


def _make_shape_specs(count: int) -> list[dict] | None:
    colors = random.sample(PALETTE, count)
    directions = random.sample(DIRECTIONS, count)

    specs: list[dict] = []
    seen_areas: set[int] = set()
    seen_signatures: set[bytes] = set()

    for color, direction in zip(colors, directions):
        chosen = None
        for _ in range(80):
            mask = _random_mask()
            area = int(mask.sum())
            signature = mask.astype(np.uint8).tobytes() + bytes(mask.shape)
            if area in seen_areas or signature in seen_signatures:
                continue
            chosen = {
                "color": int(color),
                "direction": tuple(map(int, direction)),
                "mask": mask,
            }
            seen_areas.add(area)
            seen_signatures.add(signature)
            break
        if chosen is None:
            return None
        specs.append(chosen)

    return specs


def _valid_input(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if h % 2 == 0 or w % 2 == 0:
        return False
    if int(grid[h // 2, w // 2]) != ANCHOR:
        return False
    if int(np.sum(grid == ANCHOR)) != 1:
        return False

    shapes = _extract_shapes(grid)
    if not shapes:
        return False
    areas = [int(shape["area"]) for shape in shapes]
    if len(set(areas)) != len(areas):
        return False

    seen_dirs = set()
    for shape in shapes:
        dr = _sign(shape["mean_r"] - float(h // 2))
        dc = _sign(shape["mean_c"] - float(w // 2))
        if (dr, dc) == (0, 0):
            return False
        seen_dirs.add((dr, dc))
    if len(seen_dirs) != len(shapes):
        return False

    out = main(grid)
    return isinstance(out, np.ndarray) and out.size > 0


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*SHAPE_COUNT_RANGE)
    specs = _make_shape_specs(count)
    if specs is None:
        return None

    for _ in range(60):
        h = random.choice(GRID_SIZES)
        w = random.choice(GRID_SIZES)
        grid = _place_shapes(specs, int(h), int(w))
        if grid is not None and _valid_input(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
