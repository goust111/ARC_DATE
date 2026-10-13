import numpy as np


BG = 0
MAROON = 9
RECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRECTIONS = ("up", "down", "left", "right")
RNG = np.random.default_rng()

GRID_MIN = 18
GRID_MAX = 30
GENERATE_RETRIES = 400
PAIR_SPEC_RETRIES = 80
PAIR_ANCHOR_RETRIES = 40
EXTRA_RECT_RETRIES = 120


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_arrow(arrow_coords: np.ndarray, rect_bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int, int]]:
    assert len(arrow_coords) == 4, "each arrow must have exactly four maroon cells"

    rows = arrow_coords[:, 0]
    cols = arrow_coords[:, 1]
    row_vals, row_counts = np.unique(rows, return_counts=True)
    col_vals, col_counts = np.unique(cols, return_counts=True)
    line_rows = [int(v) for v, ct in zip(row_vals, row_counts) if int(ct) == 3]
    line_cols = [int(v) for v, ct in zip(col_vals, col_counts) if int(ct) == 3]
    r0, c0, r1, c1 = rect_bbox

    assert (len(line_rows) == 1) ^ (len(line_cols) == 1), "arrow must be one straight 3-cell segment plus one tip"

    if line_rows:
        line_row = line_rows[0]
        line = arrow_coords[arrow_coords[:, 0] == line_row]
        seg_cols = sorted(int(c) for c in line[:, 1])
        assert seg_cols[1] == seg_cols[0] + 1 and seg_cols[2] == seg_cols[1] + 1, "horizontal arrow stem must be consecutive"

        extra = arrow_coords[arrow_coords[:, 0] != line_row]
        assert len(extra) == 1
        extra_r, extra_c = map(int, extra[0])
        assert extra_r in {line_row - 1, line_row + 1}, "arrow tip must be one cell off the stem"
        assert extra_c in {seg_cols[0], seg_cols[-1]}, "arrow tip must attach to one stem endpoint"

        if extra_r < line_row:
            direction = "up"
            assert line_row == r0 - 1, "up arrow stem must sit directly above the rectangle"
        else:
            direction = "down"
            assert line_row == r1 + 1, "down arrow stem must sit directly below the rectangle"

        assert c0 <= seg_cols[0] and seg_cols[-1] <= c1, "horizontal arrow stem must align to the source edge"
        return direction, (seg_cols[0], seg_cols[1], seg_cols[2])

    line_col = line_cols[0]
    line = arrow_coords[arrow_coords[:, 1] == line_col]
    seg_rows = sorted(int(r) for r in line[:, 0])
    assert seg_rows[1] == seg_rows[0] + 1 and seg_rows[2] == seg_rows[1] + 1, "vertical arrow stem must be consecutive"

    extra = arrow_coords[arrow_coords[:, 1] != line_col]
    assert len(extra) == 1
    extra_r, extra_c = map(int, extra[0])
    assert extra_c in {line_col - 1, line_col + 1}, "arrow tip must be one cell off the stem"
    assert extra_r in {seg_rows[0], seg_rows[-1]}, "arrow tip must attach to one stem endpoint"

    if extra_c < line_col:
        direction = "left"
        assert line_col == c0 - 1, "left arrow stem must sit directly left of the rectangle"
    else:
        direction = "right"
        assert line_col == c1 + 1, "right arrow stem must sit directly right of the rectangle"

    assert r0 <= seg_rows[0] and seg_rows[-1] <= r1, "vertical arrow stem must align to the source edge"
    return direction, (seg_rows[0], seg_rows[1], seg_rows[2])


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects: list[dict] = []
    rect_id = np.full(g.shape, -1, dtype=int)

    for coords in _components(g != BG):
        colors = {int(g[r, c]) for r, c in coords}
        rect_colors = colors - {MAROON}
        assert len(rect_colors) == 1, "each component must contain exactly one rectangle color"
        rect_color = next(iter(rect_colors))

        rect_mask = np.array([int(g[r, c]) != MAROON for r, c in coords], dtype=bool)
        rect_coords = coords[rect_mask]
        arrow_coords = coords[~rect_mask]
        assert len(rect_coords) > 0, "each component needs a non-maroon rectangle body"

        r0, c0, r1, c1 = _bbox(rect_coords)
        assert len(rect_coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "rectangle body must be solid"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == rect_color), "rectangle bbox must be fully filled by its color"

        obj = {
            "index": len(objects),
            "color": rect_color,
            "bbox": (r0, c0, r1, c1),
            "has_arrow": False,
            "direction": None,
            "band": (),
        }

        if len(arrow_coords):
            assert colors == {rect_color, MAROON}, "arrowed component must be a rectangle plus maroon arrow"
            direction, band = _parse_arrow(arrow_coords, (r0, c0, r1, c1))
            obj["has_arrow"] = True
            obj["direction"] = direction
            obj["band"] = band

        rect_id[r0 : r1 + 1, c0 : c1 + 1] = obj["index"]
        objects.append(obj)

    return objects, rect_id


def _find_target(obj: dict, rect_id: np.ndarray) -> int:
    r0, c0, r1, c1 = obj["bbox"]
    source_id = int(obj["index"])
    direction = obj["direction"]
    band = tuple(int(v) for v in obj["band"])

    if direction == "up":
        for r in range(r0 - 1, -1, -1):
            hits = {int(rect_id[r, c]) for c in band if int(rect_id[r, c]) >= 0}
            hits.discard(source_id)
            if hits:
                assert len(hits) == 1, "an arrow band must meet at most one rectangle at the first hit distance"
                return next(iter(hits))
    elif direction == "down":
        for r in range(r1 + 1, rect_id.shape[0]):
            hits = {int(rect_id[r, c]) for c in band if int(rect_id[r, c]) >= 0}
            hits.discard(source_id)
            if hits:
                assert len(hits) == 1, "an arrow band must meet at most one rectangle at the first hit distance"
                return next(iter(hits))
    elif direction == "left":
        for c in range(c0 - 1, -1, -1):
            hits = {int(rect_id[r, c]) for r in band if int(rect_id[r, c]) >= 0}
            hits.discard(source_id)
            if hits:
                assert len(hits) == 1, "an arrow band must meet at most one rectangle at the first hit distance"
                return next(iter(hits))
    else:
        assert direction == "right"
        for c in range(c1 + 1, rect_id.shape[1]):
            hits = {int(rect_id[r, c]) for r in band if int(rect_id[r, c]) >= 0}
            hits.discard(source_id)
            if hits:
                assert len(hits) == 1, "an arrow band must meet at most one rectangle at the first hit distance"
                return next(iter(hits))

    raise AssertionError("every arrowed rectangle must see a target rectangle")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each arrow-marked rectangle to the color of the first rectangle hit in the arrow's pointed direction.

    Input structure:
    - The background is black.
    - The foreground contains several non-touching solid monochrome rectangles, each with a distinct visible color.
    - Some rectangles have a maroon arrow attached just outside one edge; the arrow is a 3-cell straight stem plus a 1-cell turned tip.
    - The arrow stem spans three consecutive edge-aligned cells, and the tip sits one cell to the side of one stem endpoint to encode the final pointing direction.

    Rule:
    - For every rectangle with an arrow, decode the arrow's direction from the side on which the turned tip lies relative to the 3-cell stem.
    - Extend a 3-cell-wide ray band from that rectangle in the decoded direction, using the stem's three aligned positions.
    - The first other rectangle touched by that ray band is the target; recolor the whole source rectangle to the target rectangle's color.
    - Remove all maroon arrow cells to black, and leave rectangles without arrows unchanged.

    Key insight:
    - The maroon marker is not a filled triangle; its L-shaped turn implicitly tells you which way the ray continues, and that decoded direction determines which rectangle supplies the new color.
    """
    g = np.asarray(grid, dtype=int)
    objects, rect_id = _parse_scene(g)

    out = g.copy()
    for obj in objects:
        if not obj["has_arrow"]:
            continue
        target_id = _find_target(obj, rect_id)
        target_color = int(objects[target_id]["color"])
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = target_color

    out[g == MAROON] = BG
    return out


def _reserve_with_padding(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        rr0 = max(0, r - 1)
        rr1 = min(h, r + 2)
        cc0 = max(0, c - 1)
        cc1 = min(w, c + 2)
        mask[rr0:rr1, cc0:cc1] = True


def _make_pair_spec(direction: str, source_color: int, target_color: int) -> dict:
    if direction in {"up", "down"}:
        src_h = int(RNG.integers(2, 6))
        src_w = int(RNG.integers(3, 7))
        tgt_h = int(RNG.integers(2, 6))
        tgt_w = int(RNG.integers(2, 7))
        offset = int(RNG.integers(0, src_w - 2))
        tip_col = offset + (0 if int(RNG.integers(0, 2)) == 0 else 2)
        distance = int(RNG.integers(3, 8))
        tgt_left = offset + int(RNG.integers(-tgt_w + 1, 3))

        if direction == "up":
            line_row = -1
            tip = (-2, tip_col)
            tgt_bottom = -distance
            tgt_top = tgt_bottom - tgt_h + 1
            corridor = {(r, c) for r in range(tgt_bottom + 1, 0) for c in range(offset, offset + 3)}
        else:
            line_row = src_h
            tip = (src_h + 1, tip_col)
            tgt_top = src_h - 1 + distance
            corridor = {(r, c) for r in range(src_h, tgt_top) for c in range(offset, offset + 3)}

        source_cells = {(r, c): source_color for r in range(src_h) for c in range(src_w)}
        target_rows = range(tgt_top, tgt_top + tgt_h)
        target_cells = {(r, c): target_color for r in target_rows for c in range(tgt_left, tgt_left + tgt_w)}
        arrow_cells = {(line_row, offset + k): MAROON for k in range(3)}
        arrow_cells[tip] = MAROON
    else:
        src_h = int(RNG.integers(3, 7))
        src_w = int(RNG.integers(2, 6))
        tgt_h = int(RNG.integers(2, 7))
        tgt_w = int(RNG.integers(2, 6))
        offset = int(RNG.integers(0, src_h - 2))
        tip_row = offset + (0 if int(RNG.integers(0, 2)) == 0 else 2)
        distance = int(RNG.integers(3, 8))
        tgt_top = offset + int(RNG.integers(-tgt_h + 1, 3))

        if direction == "left":
            line_col = -1
            tip = (tip_row, -2)
            tgt_right = -distance
            tgt_left = tgt_right - tgt_w + 1
            corridor = {(r, c) for r in range(offset, offset + 3) for c in range(tgt_right + 1, 0)}
        else:
            line_col = src_w
            tip = (tip_row, src_w + 1)
            tgt_left = src_w - 1 + distance
            corridor = {(r, c) for r in range(offset, offset + 3) for c in range(src_w, tgt_left)}

        source_cells = {(r, c): source_color for r in range(src_h) for c in range(src_w)}
        target_cols = range(tgt_left, tgt_left + tgt_w)
        target_cells = {(r, c): target_color for r in range(tgt_top, tgt_top + tgt_h) for c in target_cols}
        arrow_cells = {(offset + k, line_col): MAROON for k in range(3)}
        arrow_cells[tip] = MAROON

    cell_colors = dict(source_cells)
    cell_colors.update(target_cells)
    cell_colors.update(arrow_cells)
    return {"cell_colors": cell_colors, "corridor": corridor}


def _place_pair(
    grid: np.ndarray,
    forbidden: np.ndarray,
    corridor_reserved: np.ndarray,
    direction: str,
    source_color: int,
    target_color: int,
) -> bool:
    h, w = grid.shape

    for _ in range(PAIR_SPEC_RETRIES):
        spec = _make_pair_spec(direction, source_color, target_color)
        occupied_local = set(spec["cell_colors"])
        footprint = occupied_local | set(spec["corridor"])
        min_r = min(r for r, _ in footprint)
        max_r = max(r for r, _ in footprint)
        min_c = min(c for _, c in footprint)
        max_c = max(c for _, c in footprint)

        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            continue

        for _ in range(PAIR_ANCHOR_RETRIES):
            anchor_r = int(RNG.integers(row_lo, row_hi + 1))
            anchor_c = int(RNG.integers(col_lo, col_hi + 1))

            actual_cells = {(anchor_r + r, anchor_c + c) for r, c in occupied_local}
            corridor_cells = {
                (anchor_r + r, anchor_c + c)
                for r, c in spec["corridor"]
                if (r, c) not in occupied_local
            }

            if any(forbidden[r, c] or corridor_reserved[r, c] for r, c in actual_cells):
                continue
            if any(forbidden[r, c] or corridor_reserved[r, c] for r, c in corridor_cells):
                continue

            for (lr, lc), color in spec["cell_colors"].items():
                grid[anchor_r + lr, anchor_c + lc] = color
            _reserve_with_padding(forbidden, actual_cells)
            for r, c in corridor_cells:
                corridor_reserved[r, c] = True
            return True

    return False


def _place_plain_rect(
    grid: np.ndarray,
    forbidden: np.ndarray,
    corridor_reserved: np.ndarray,
    color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(EXTRA_RECT_RETRIES):
        rect_h = int(RNG.integers(2, 6))
        rect_w = int(RNG.integers(2, 6))
        if rect_h > h or rect_w > w:
            continue

        r0 = int(RNG.integers(0, h - rect_h + 1))
        c0 = int(RNG.integers(0, w - rect_w + 1))
        cells = {(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)}
        if any(forbidden[r, c] or corridor_reserved[r, c] for r, c in cells):
            continue

        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
        _reserve_with_padding(forbidden, cells)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(GRID_MIN, GRID_MAX + 1))
    w = int(RNG.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((h, w), BG, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)
    corridor_reserved = np.zeros((h, w), dtype=bool)

    source_count = int(RNG.choice(np.array([1, 2, 3]), p=np.array([0.2, 0.4, 0.4])))
    max_extra = min(2, len(RECT_COLORS) - 2 * source_count)
    extra_count = 0 if max_extra == 0 else int(RNG.integers(0, max_extra + 1))

    colors = [int(v) for v in RNG.permutation(np.array(RECT_COLORS))[: 2 * source_count + extra_count]]
    directions = [str(v) for v in RNG.permutation(np.array(DIRECTIONS))[:source_count]]

    color_idx = 0
    for direction in directions:
        source_color = colors[color_idx]
        target_color = colors[color_idx + 1]
        color_idx += 2
        if not _place_pair(grid, forbidden, corridor_reserved, direction, source_color, target_color):
            return None

    while color_idx < len(colors):
        if not _place_plain_rect(grid, forbidden, corridor_reserved, colors[color_idx]):
            return None
        color_idx += 1

    try:
        objects, _rect_id = _parse_scene(grid)
        assert any(obj["has_arrow"] for obj in objects)
        out = main(grid)
        assert out.shape == grid.shape
        assert not np.any(out == MAROON)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GENERATE_RETRIES} attempts")
