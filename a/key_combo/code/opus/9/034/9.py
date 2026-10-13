import random

import numpy as np


BACKGROUND = 0
RECT_COLOR = 2
FILL_COLORS = (3, 4)
OBSTACLE_COLOR = 5
TEMPLATE_BORDER = 8
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_from_corners(grid: np.ndarray) -> tuple[int, int, int, int]:
    corners = np.argwhere(grid == RECT_COLOR)
    assert corners.shape == (4, 2), "expected exactly four red corner markers"
    rows = sorted({int(r) for r, _ in corners})
    cols = sorted({int(c) for _, c in corners})
    assert len(rows) == 2 and len(cols) == 2, "red markers must occupy two rows and two cols"
    expected = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    actual = {(int(r), int(c)) for r, c in corners}
    assert actual == expected, "red markers must be the four rectangle corners"
    return rows[0], cols[0], rows[1], cols[1]


def _extract_template(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    border = np.argwhere(grid == TEMPLATE_BORDER)
    assert border.size > 0, "expected one cyan template frame"
    r0, c0, r1, c1 = _bbox(border)
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "template must enclose a non-trivial interior"

    frame_h = r1 - r0 + 1
    frame_w = c1 - c0 + 1
    expected_border = 2 * frame_h + 2 * frame_w - 4
    assert border.shape[0] == expected_border, "all cyan cells must belong to one rectangular border"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch[0, :] == TEMPLATE_BORDER)
    assert np.all(patch[-1, :] == TEMPLATE_BORDER)
    assert np.all(patch[:, 0] == TEMPLATE_BORDER)
    assert np.all(patch[:, -1] == TEMPLATE_BORDER)
    assert not np.any(patch[1:-1, 1:-1] == TEMPLATE_BORDER), "template border must be one cell thick"

    tile = patch[1:-1, 1:-1].copy()
    assert tile.shape[0] in (3, 4) and tile.shape[1] in (3, 4), "template interior must be 3x3 or 4x4"
    assert set(map(int, np.unique(tile))) == set(FILL_COLORS), "template tile must use green and yellow"
    return (r0, c0, r1, c1), tile


def _draw_rectangle_edges(out: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    for c in range(c0, c1 + 1):
        if int(out[r0, c]) != OBSTACLE_COLOR:
            out[r0, c] = RECT_COLOR
        if int(out[r1, c]) != OBSTACLE_COLOR:
            out[r1, c] = RECT_COLOR
    for r in range(r0, r1 + 1):
        if int(out[r, c0]) != OBSTACLE_COLOR:
            out[r, c0] = RECT_COLOR
        if int(out[r, c1]) != OBSTACLE_COLOR:
            out[r, c1] = RECT_COLOR


def _tile_fill(out: np.ndarray, tile: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    tile_h, tile_w = tile.shape
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(out[r, c]) != BACKGROUND:
                continue
            out[r, c] = int(tile[(r - (r0 + 1)) % tile_h, (c - (c0 + 1)) % tile_w])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the four red rectangle corners, fill the interior by tiling the remote template, then erase the template.

    Input structure:
    - Black background with exactly four red corner markers that define one axis-aligned rectangle.
    - Some grey pixels may lie on the would-be rectangle edges as obstacles, but not inside the rectangle.
    - A separate cyan one-cell-thick rectangular frame in a corner encloses a 3x3 or 4x4 green/yellow template tile.

    Rule:
    - Use the four red markers as the rectangle corners and draw the four red edges, skipping over any grey obstacle cells.
    - Extract the template interior inside the cyan frame and tile it from the rectangle interior's top-left corner across the whole interior.
    - Only black interior cells may be overwritten by the tiled pattern, and the cyan template region is erased back to black afterward.

    Key insight:
    - The rectangle geometry and the fill texture are encoded by two disjoint structures: red corner markers specify where to draw, while the remote cyan-framed tile specifies what to repeat.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rr0, cc0, rr1, cc1 = _rect_from_corners(g)
    template_bbox, tile = _extract_template(g)

    out = g.copy()
    _draw_rectangle_edges(out, rr0, cc0, rr1, cc1)
    _tile_fill(out, tile, rr0, cc0, rr1, cc1)

    tr0, tc0, tr1, tc1 = template_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BACKGROUND
    return out


def _make_tile(size: int, pattern_kind: str) -> np.ndarray:
    tile = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if pattern_kind == "checker":
                idx = (r + c) % 2
            elif pattern_kind == "horizontal":
                idx = r % 2
            else:
                idx = c % 2
            tile[r, c] = FILL_COLORS[idx]
    return tile


def _template_positions(height: int, width: int, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    offsets_r = [0, 1] if height - outer_h >= 1 else [0]
    offsets_c = [0, 1] if width - outer_w >= 1 else [0]
    positions = []
    for dr in offsets_r:
        for dc in offsets_c:
            positions.extend(
                [
                    (dr, dc),
                    (dr, width - outer_w - dc),
                    (height - outer_h - dr, dc),
                    (height - outer_h - dr, width - outer_w - dc),
                ]
            )
    return sorted(set(positions))


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _boundary_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = []
    for c in range(c0 + 1, c1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _render_input(case: dict) -> np.ndarray:
    grid = np.zeros((case["height"], case["width"]), dtype=int)

    tr0, tc0, tr1, tc1 = case["template_bbox"]
    grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = TEMPLATE_BORDER
    grid[tr0 + 1 : tr1, tc0 + 1 : tc1] = case["tile"]

    r0, c0, r1, c1 = case["rect"]
    grid[r0, c0] = RECT_COLOR
    grid[r0, c1] = RECT_COLOR
    grid[r1, c0] = RECT_COLOR
    grid[r1, c1] = RECT_COLOR

    for r, c in case["obstacles"]:
        grid[r, c] = OBSTACLE_COLOR
    return grid


def _expected_output(case: dict) -> np.ndarray:
    out = _render_input(case)
    r0, c0, r1, c1 = case["rect"]
    _draw_rectangle_edges(out, r0, c0, r1, c1)
    _tile_fill(out, case["tile"], r0, c0, r1, c1)
    tr0, tc0, tr1, tc1 = case["template_bbox"]
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BACKGROUND
    return out


def _validate_case(case: dict, inp: np.ndarray, expected: np.ndarray) -> bool:
    r0, c0, r1, c1 = case["rect"]
    tr0, tc0, tr1, tc1 = case["template_bbox"]
    tile = case["tile"]
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1

    if interior_h * interior_w < 2 * tile.size:
        return False
    if _boxes_overlap(case["rect"], case["template_bbox"], gap=1):
        return False
    if not set(map(int, np.unique(tile))) == set(FILL_COLORS):
        return False
    if np.array_equal(tile, np.full_like(tile, tile[0, 0])):
        return False

    obstacle_cells = set(case["obstacles"])
    if not obstacle_cells.issubset(set(_boundary_cells(case["rect"]))):
        return False
    if case["with_obstacles"] and not obstacle_cells:
        return False
    if not case["with_obstacles"] and obstacle_cells:
        return False

    corner_ok = (
        tr0 in (0, 1)
        or tc0 in (0, 1)
        or tr1 in (inp.shape[0] - 1, inp.shape[0] - 2)
        or tc1 in (inp.shape[1] - 1, inp.shape[1] - 2)
    )
    if not corner_ok:
        return False

    if case["needs_crop"] and interior_h % tile.shape[0] == 0 and interior_w % tile.shape[1] == 0:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    if np.any(expected[tr0 : tr1 + 1, tc0 : tc1 + 1] != BACKGROUND):
        return False
    for r, c in obstacle_cells:
        if int(expected[r, c]) != OBSTACLE_COLOR:
            return False
    return True


def _sample_rect(
    height: int,
    width: int,
    template_bbox: tuple[int, int, int, int],
    tile: np.ndarray,
    needs_crop: bool,
) -> tuple[int, int, int, int] | None:
    tile_area = int(tile.size)
    row_pairs = [(r0, r1) for r0 in range(2, height - 4) for r1 in range(r0 + 3, height - 1)]
    col_pairs = [(c0, c1) for c0 in range(2, width - 4) for c1 in range(c0 + 3, width - 1)]
    random.shuffle(row_pairs)
    random.shuffle(col_pairs)

    for r0, r1 in row_pairs[:120]:
        interior_h = r1 - r0 - 1
        for c0, c1 in col_pairs[:160]:
            interior_w = c1 - c0 - 1
            rect = (r0, c0, r1, c1)
            if interior_h * interior_w < 2 * tile_area:
                continue
            exact_fit = interior_h % tile.shape[0] == 0 and interior_w % tile.shape[1] == 0
            if needs_crop and exact_fit:
                continue
            if not needs_crop and not exact_fit:
                continue
            if _boxes_overlap(rect, template_bbox, gap=1):
                continue
            return rect
    return None


def _sample_obstacles(rect: tuple[int, int, int, int], with_obstacles: bool) -> list[tuple[int, int]]:
    cells = _boundary_cells(rect)
    if not with_obstacles:
        return []

    target = random.randint(1, min(6, len(cells)))
    random.shuffle(cells)
    chosen = cells[:target]
    edge_ids = set()
    r0, c0, r1, c1 = rect
    for r, c in chosen:
        if r == r0:
            edge_ids.add("top")
        elif r == r1:
            edge_ids.add("bottom")
        elif c == c0:
            edge_ids.add("left")
        else:
            edge_ids.add("right")
    if len(edge_ids) == 1 and len(cells) > target:
        for r, c in cells[target:]:
            if r == r0 and "top" not in edge_ids:
                chosen.append((r, c))
                break
            if r == r1 and "bottom" not in edge_ids:
                chosen.append((r, c))
                break
            if c == c0 and "left" not in edge_ids:
                chosen.append((r, c))
                break
            if c == c1 and "right" not in edge_ids:
                chosen.append((r, c))
                break
    return sorted(set(chosen))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 24)
    width = random.randint(16, 24)

    tile_size = random.choice((3, 4))
    pattern_kind = random.choice(("checker", "horizontal", "vertical"))
    tile = _make_tile(tile_size, pattern_kind)
    outer_h = tile_size + 2
    outer_w = tile_size + 2

    positions = _template_positions(height, width, outer_h, outer_w)
    random.shuffle(positions)

    needs_crop = random.random() < 0.65
    with_obstacles = random.random() < 0.7

    for tr0, tc0 in positions:
        template_bbox = (tr0, tc0, tr0 + outer_h - 1, tc0 + outer_w - 1)
        rect = _sample_rect(height, width, template_bbox, tile, needs_crop)
        if rect is None:
            continue

        case = {
            "height": height,
            "width": width,
            "tile": tile,
            "pattern_kind": pattern_kind,
            "template_bbox": template_bbox,
            "rect": rect,
            "with_obstacles": with_obstacles,
            "needs_crop": needs_crop,
        }
        case["obstacles"] = _sample_obstacles(rect, with_obstacles)

        inp = _render_input(case)
        expected = _expected_output(case)
        if _validate_case(case, inp, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
