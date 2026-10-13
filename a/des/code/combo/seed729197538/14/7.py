from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
MAX_OFFSET = 4

GENERATE_ATTEMPTS = 300
CENTER_ATTEMPTS = 80
EDGE_CENTER_PROB = 0.4


def _sign(value: int) -> int:
    return int((value > 0) - (value < 0))


def _cross_cells(h: int, w: int, center_r: int, center_c: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = center_r + dr
        cc = center_c + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.append((rr, cc))
    return cells


def _shifted_cell(r: int, c: int, center_r: int, center_c: int) -> tuple[int, int]:
    dr = center_r - r
    dc = center_c - c
    if max(abs(dr), abs(dc)) <= 1:
        return r, c
    return r + _sign(dr), c + _sign(dc)


def main(grid: np.ndarray) -> np.ndarray:
    """Pull each color one step toward its centroid and draw a same-color cross there."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    colors = sorted(int(color) for color in np.unique(g) if int(color) != BG)
    for color in colors:
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue

        center_r = int(coords[:, 0].sum() // len(coords))
        center_c = int(coords[:, 1].sum() // len(coords))

        for r, c in coords:
            rr, cc = _shifted_cell(int(r), int(c), center_r, center_c)
            out[rr, cc] = color

        for rr, cc in _cross_cells(h, w, center_r, center_c):
            out[rr, cc] = color

    return out


def _moved_offset(dr: int, dc: int) -> tuple[int, int]:
    if max(abs(dr), abs(dc)) <= 1:
        return dr, dc
    return dr - _sign(dr), dc - _sign(dc)


def _offsets_ok(
    offsets: list[tuple[int, int]],
    *,
    require_adjacent: bool,
    require_all_far: bool,
) -> bool:
    count = len(offsets)
    sum_r = sum(dr for dr, _ in offsets)
    sum_c = sum(dc for _, dc in offsets)
    if not (0 <= sum_r < count and 0 <= sum_c < count):
        return False

    dists = [max(abs(dr), abs(dc)) for dr, dc in offsets]
    if max(dists) < 2:
        return False
    if require_adjacent and 1 not in dists:
        return False
    if require_all_far and any(dist == 1 for dist in dists):
        return False

    rows = {dr for dr, _ in offsets}
    cols = {dc for _, dc in offsets}
    if len(rows) + len(cols) < 3:
        return False

    row_span = max(dr for dr, _ in offsets) - min(dr for dr, _ in offsets)
    col_span = max(dc for _, dc in offsets) - min(dc for _, dc in offsets)
    if row_span + col_span < 3:
        return False

    moved = {_moved_offset(dr, dc) for dr, dc in offsets}
    if len(moved) == 1:
        return False

    return True


INTERIOR_ADJACENT_TEMPLATES = (
    ((-2, 0), (2, 0), (0, -2), (1, 0), (-1, 2)),
    ((-2, -2), (2, 1), (0, 2), (1, 0), (-1, -1)),
    ((-3, 0), (3, 0), (0, -2), (1, 2), (-1, 0)),
    ((-2, 0), (2, 0), (0, -3), (0, 2), (-1, 1), (1, 0)),
)

INTERIOR_FAR_DIAG_TEMPLATES = (
    ((-2, -1), (2, 1), (-1, 2), (1, -2)),
    ((-2, 0), (0, 2), (2, -2)),
    ((-3, -1), (3, 1), (-1, 3), (1, -3), (0, 2), (0, -2)),
)

INTERIOR_FAR_ORTH_TEMPLATES = (
    ((-2, 0), (2, 0), (0, -2), (0, 2)),
    ((-3, 0), (2, 0), (0, -2), (1, 2)),
)

EDGE_TOP_ADJACENT_TEMPLATES = (
    ((0, -3), (0, 3), (1, -1), (1, 1)),
    ((0, -2), (0, 2), (1, -3), (1, 1), (1, 2)),
)

EDGE_TOP_FAR_TEMPLATES = (
    ((0, -4), (0, -2), (0, 2), (0, 4), (1, -3), (1, 3)),
    ((0, -3), (0, 3), (1, -2), (1, 2), (2, 0)),
)


def _flip_rows(offsets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((-dr, dc) for dr, dc in offsets)


def _flip_cols(offsets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((dr, -dc) for dr, dc in offsets)


def _transpose(offsets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((dc, dr) for dr, dc in offsets)


def _transform_interior(template: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    offsets = template
    if random.random() < 0.5:
        offsets = _flip_rows(offsets)
    if random.random() < 0.5:
        offsets = _flip_cols(offsets)
    if random.random() < 0.5:
        offsets = _transpose(offsets)
    if random.random() < 0.5:
        offsets = _flip_rows(offsets)
    if random.random() < 0.5:
        offsets = _flip_cols(offsets)
    return list(offsets)


def _transform_top(template: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    offsets = template
    if random.random() < 0.5:
        offsets = _flip_cols(offsets)
    return list(offsets)


def _transform_left(template: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    offsets = _transpose(tuple(_transform_top(template)))
    if random.random() < 0.5:
        offsets = _flip_rows(offsets)
    return list(offsets)


def _sample_offsets(role: str, edge_mode: str | None) -> list[tuple[int, int]]:
    if edge_mode == "top":
        if role == "adjacent":
            return _transform_top(random.choice(EDGE_TOP_ADJACENT_TEMPLATES))
        return _transform_top(random.choice(EDGE_TOP_FAR_TEMPLATES))

    if edge_mode == "left":
        if role == "adjacent":
            return _transform_left(random.choice(EDGE_TOP_ADJACENT_TEMPLATES))
        return _transform_left(random.choice(EDGE_TOP_FAR_TEMPLATES))

    if role == "adjacent":
        return _transform_interior(random.choice(INTERIOR_ADJACENT_TEMPLATES))
    if role == "far_diag":
        return _transform_interior(random.choice(INTERIOR_FAR_DIAG_TEMPLATES))
    if role == "far_orth":
        return _transform_interior(random.choice(INTERIOR_FAR_ORTH_TEMPLATES))

    pool = (
        INTERIOR_ADJACENT_TEMPLATES
        + INTERIOR_FAR_DIAG_TEMPLATES
        + INTERIOR_FAR_ORTH_TEMPLATES
    )
    return _transform_interior(random.choice(pool))


def _sample_center_for_offsets(
    h: int,
    w: int,
    offsets: list[tuple[int, int]],
    edge_mode: str | None,
) -> tuple[int, int] | None:
    min_dr = min(dr for dr, _ in offsets)
    max_dr = max(dr for dr, _ in offsets)
    min_dc = min(dc for _, dc in offsets)
    max_dc = max(dc for _, dc in offsets)

    if edge_mode == "top":
        if min_dr < 0:
            return None
        low_c = max(1, -min_dc)
        high_c = min(w - 2, w - 1 - max_dc)
        if low_c > high_c:
            return None
        return 0, random.randint(low_c, high_c)

    if edge_mode == "left":
        if min_dc < 0:
            return None
        low_r = max(1, -min_dr)
        high_r = min(h - 2, h - 1 - max_dr)
        if low_r > high_r:
            return None
        return random.randint(low_r, high_r), 0

    low_r = max(1, -min_dr)
    high_r = min(h - 2, h - 1 - max_dr)
    low_c = max(1, -min_dc)
    high_c = min(w - 2, w - 1 - max_dc)
    if low_r > high_r or low_c > high_c:
        return None
    return random.randint(low_r, high_r), random.randint(low_c, high_c)


def _group_footprint(
    h: int,
    w: int,
    center_r: int,
    center_c: int,
    offsets: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]]:
    cells: set[tuple[int, int]] = set(_cross_cells(h, w, center_r, center_c))
    for dr, dc in offsets:
        cells.add((center_r + dr, center_c + dc))
        mr, mc = _moved_offset(dr, dc)
        cells.add((center_r + mr, center_c + mc))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    bbox = (
        max(0, min(rows) - 1),
        max(0, min(cols) - 1),
        min(h - 1, max(rows) + 1),
        min(w - 1, max(cols) + 1),
    )
    return cells, bbox


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _build_group(
    h: int,
    w: int,
    color: int,
    reserved_boxes: list[tuple[int, int, int, int]],
    *,
    edge_mode: str | None,
    role: str,
) -> dict[str, object] | None:
    for _ in range(CENTER_ATTEMPTS):
        offsets = _sample_offsets(role, edge_mode)
        center = _sample_center_for_offsets(h, w, offsets, edge_mode)
        if center is None:
            continue
        center_r, center_c = center
        footprint, bbox = _group_footprint(h, w, center_r, center_c, offsets)
        if any(_boxes_overlap(bbox, other) for other in reserved_boxes):
            continue

        positions = [(center_r + dr, center_c + dc) for dr, dc in offsets]
        return {
            "color": color,
            "center": (center_r, center_c),
            "offsets": offsets,
            "positions": positions,
            "footprint": footprint,
            "bbox": bbox,
        }

    return None


def _group_features(groups: list[dict[str, object]]) -> dict[str, bool]:
    has_adjacent = False
    has_all_far = False
    has_orth_move = False
    has_diagonal_move = False
    has_edge_center = False

    for group in groups:
        center_r, center_c = group["center"]  # type: ignore[misc]
        offsets = group["offsets"]  # type: ignore[misc]
        has_edge_center |= center_r == 0 or center_c == 0

        dists = [max(abs(dr), abs(dc)) for dr, dc in offsets]
        has_adjacent |= any(dist == 1 for dist in dists)
        has_all_far |= all(dist >= 2 for dist in dists)

        for dr, dc in offsets:
            if max(abs(dr), abs(dc)) <= 1:
                continue
            has_orth_move |= (dr == 0 or dc == 0)
            has_diagonal_move |= (dr != 0 and dc != 0)

    return {
        "has_adjacent": has_adjacent,
        "has_all_far": has_all_far,
        "has_orth_move": has_orth_move,
        "has_diagonal_move": has_diagonal_move,
        "has_edge_center": has_edge_center,
    }


def _group_count_for_size(h: int, w: int) -> int:
    area = h * w
    if area < 180:
        return 2
    if area < 250:
        return random.randint(2, 3)
    return random.randint(3, 4)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    group_count = _group_count_for_size(h, w)

    colors = random.sample(PALETTE, group_count)
    roles = ["adjacent", "far_diag"]
    if group_count >= 3:
        roles.append("far_orth")
    while len(roles) < group_count:
        roles.append(random.choice(("adjacent", "far_diag", "far_orth")))
    random.shuffle(roles)

    edge_index = random.randrange(group_count) if random.random() < EDGE_CENTER_PROB else None

    groups: list[dict[str, object]] = []
    reserved_boxes: list[tuple[int, int, int, int]] = []

    for idx, (color, role) in enumerate(zip(colors, roles)):
        edge_mode = None
        if idx == edge_index:
            edge_mode = random.choice(("top", "left"))

        group = _build_group(
            h,
            w,
            color,
            reserved_boxes,
            edge_mode=edge_mode,
            role=role,
        )
        if group is None:
            return None

        groups.append(group)
        reserved_boxes.append(group["bbox"])  # type: ignore[arg-type]

    features = _group_features(groups)
    assert features["has_adjacent"]
    assert features["has_all_far"]
    assert features["has_orth_move"]
    assert features["has_diagonal_move"]
    if edge_index is not None:
        assert features["has_edge_center"]

    grid = np.full((h, w), BG, dtype=int)
    for group in groups:
        center_r, center_c = group["center"]  # type: ignore[misc]
        assert grid[center_r, center_c] == BG
        color = int(group["color"])
        for r, c in group["positions"]:  # type: ignore[misc]
            if grid[r, c] != BG:
                return None
            grid[r, c] = color

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid constellation grid")


for _template in INTERIOR_ADJACENT_TEMPLATES:
    assert _offsets_ok(list(_template), require_adjacent=True, require_all_far=False)
for _template in INTERIOR_FAR_DIAG_TEMPLATES + INTERIOR_FAR_ORTH_TEMPLATES:
    assert _offsets_ok(list(_template), require_adjacent=False, require_all_far=True)
for _template in EDGE_TOP_ADJACENT_TEMPLATES:
    assert _offsets_ok(list(_template), require_adjacent=True, require_all_far=False)
for _template in EDGE_TOP_FAR_TEMPLATES:
    assert _offsets_ok(list(_template), require_adjacent=False, require_all_far=True)
