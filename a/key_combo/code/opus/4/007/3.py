import random
from typing import Optional

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
SOURCE_COLORS = (BLUE, RED, GREEN, YELLOW)
MOTIF_SIZES = (3, 4)
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _ray_hit(
    grid: np.ndarray, start_r: int, start_c: int, dr: int, dc: int
) -> Optional[dict[str, object]]:
    rr = start_r + dr
    cc = start_c + dc
    dist = 1
    while _in_bounds(grid.shape, rr, cc):
        val = int(grid[rr, cc])
        if val not in (BLACK, GREY):
            return {"color": val, "dist": dist, "pos": (rr, cc)}
        rr += dr
        cc += dc
        dist += 1
    return None


def _choose_axis_color(
    neg_hit: Optional[dict[str, object]], pos_hit: Optional[dict[str, object]]
) -> Optional[int]:
    if neg_hit is None and pos_hit is None:
        return None
    if neg_hit is None:
        return int(pos_hit["color"])
    if pos_hit is None:
        return int(neg_hit["color"])
    if int(neg_hit["color"]) == int(pos_hit["color"]):
        return int(neg_hit["color"])
    if int(neg_hit["dist"]) <= int(pos_hit["dist"]):
        return int(neg_hit["color"])
    return int(pos_hit["color"])


def _analyze_blocks(
    grid: np.ndarray, matrix_bbox: tuple[int, int, int, int], tile_h: int, tile_w: int
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = matrix_bbox
    rows = (r1 - r0 + 1) // tile_h
    cols = (c1 - c0 + 1) // tile_w
    center_dr = (tile_h - 1) // 2
    center_dc = (tile_w - 1) // 2
    records: list[dict[str, object]] = []
    for br in range(rows):
        for bc in range(cols):
            sr = r0 + br * tile_h
            sc = c0 + bc * tile_w
            cr = sr + center_dr
            cc = sc + center_dc
            left = _ray_hit(grid, cr, cc, 0, -1)
            right = _ray_hit(grid, cr, cc, 0, 1)
            up = _ray_hit(grid, cr, cc, -1, 0)
            down = _ray_hit(grid, cr, cc, 1, 0)
            horizontal = _choose_axis_color(left, right)
            vertical = _choose_axis_color(up, down)
            axis_colors = tuple(x for x in (horizontal, vertical) if x is not None)
            records.append(
                {
                    "block_row": br,
                    "block_col": bc,
                    "r0": sr,
                    "c0": sc,
                    "r1": sr + tile_h - 1,
                    "c1": sc + tile_w - 1,
                    "center": (cr, cc),
                    "tile_h": tile_h,
                    "tile_w": tile_w,
                    "hits": {"left": left, "right": right, "up": up, "down": down},
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "visible_colors": tuple(sorted(set(axis_colors))),
                }
            )
    return records


def _layout_candidates(grid: np.ndarray) -> list[dict[str, object]]:
    bbox = _bbox(grid == GREY)
    if bbox is None:
        return []
    r0, c0, r1, c1 = bbox
    matrix = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, matrix.shape)
    candidates: list[dict[str, object]] = []
    for tile_h in MOTIF_SIZES:
        if h % tile_h != 0:
            continue
        rows = h // tile_h
        if not 2 <= rows <= 4:
            continue
        for tile_w in MOTIF_SIZES:
            if w % tile_w != 0:
                continue
            cols = w // tile_w
            if not 2 <= cols <= 4:
                continue
            motif = matrix[:tile_h, :tile_w]
            if int(np.sum(motif == BLACK)) < 1 or int(np.sum(motif == BLACK)) > 3:
                continue
            ok = True
            for br in range(rows):
                for bc in range(cols):
                    block = matrix[
                        br * tile_h : (br + 1) * tile_h, bc * tile_w : (bc + 1) * tile_w
                    ]
                    if not np.array_equal(block, motif):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            records = _analyze_blocks(grid, bbox, tile_h, tile_w)
            if any(len(rec["visible_colors"]) == 0 for rec in records):
                continue
            candidates.append(
                {
                    "matrix_bbox": bbox,
                    "tile_h": tile_h,
                    "tile_w": tile_w,
                    "rows": rows,
                    "cols": cols,
                    "motif": motif.copy(),
                    "records": records,
                }
            )
    return candidates


def _infer_layout(grid: np.ndarray) -> dict[str, object]:
    candidates = _layout_candidates(grid)
    if not candidates:
        raise AssertionError("failed to infer repeated motif layout")
    if len(candidates) == 1:
        return candidates[0]
    # Prefer the layout that best matches the puzzle's semantic constraints.
    scored: list[tuple[tuple[int, int, int, int], dict[str, object]]] = []
    for cand in candidates:
        records = cand["records"]
        split_blocks = sum(1 for rec in records if len(rec["visible_colors"]) == 2)
        axis_conflicts = sum(
            1
            for rec in records
            if (
                rec["hits"]["left"] is not None
                and rec["hits"]["right"] is not None
                and int(rec["hits"]["left"]["color"]) != int(rec["hits"]["right"]["color"])
            )
            or (
                rec["hits"]["up"] is not None
                and rec["hits"]["down"] is not None
                and int(rec["hits"]["up"]["color"]) != int(rec["hits"]["down"]["color"])
            )
        )
        score = (split_blocks, axis_conflicts, cand["tile_h"] * cand["tile_w"], cand["rows"] * cand["cols"])
        scored.append((score, cand))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each repeated motif block from the edge-source colors seen along its four rays.

    Input structure:
    - A black background contains one central R x C matrix of tightly packed, identical grey motif blocks.
    - Each block is a 3x3 or 4x4 grey mask with 1-3 internal black holes.
    - Colored single-pixel sources sit on the outer border and align with some block-center rows or columns.

    Rule:
    - Detect the repeated motif matrix, then treat each tile as one block and cast rays from its center up, down, left, and right.
    - A ray ignores black background and grey motif pixels; its visible source is the first non-black, non-grey pixel hit on that line.
    - Collapse left/right into one horizontal source color and up/down into one vertical source color, choosing the nearer source when opposite directions disagree.
    - If only one source color is visible for a block, recolor all of that block's grey pixels with it; if two source colors remain, recolor the left half of the block's grey pixels with the horizontal color and the right half with the vertical color, keeping black holes unchanged.

    Key insight:
    - The motif does not encode its own output colors; each block must first decode its horizontal and vertical colors from ray visibility to the border sources.
    """
    g = np.asarray(grid, dtype=int)
    info = _infer_layout(g)
    out = g.copy()

    for rec in info["records"]:
        sr = int(rec["r0"])
        sc = int(rec["c0"])
        tile_h = int(rec["tile_h"])
        tile_w = int(rec["tile_w"])
        horizontal = rec["horizontal"]
        vertical = rec["vertical"]
        colors = rec["visible_colors"]
        block = out[sr : sr + tile_h, sc : sc + tile_w]
        grey_mask = block == GREY
        if len(colors) == 1:
            block[grey_mask] = int(colors[0])
        elif len(colors) == 2:
            left_w = (tile_w + 1) // 2
            left_mask = np.zeros_like(block, dtype=bool)
            left_mask[:, :left_w] = True
            right_mask = ~left_mask
            if horizontal is not None:
                mask = grey_mask & left_mask
                block[mask] = int(horizontal)
            if vertical is not None:
                mask = grey_mask & right_mask
                block[mask] = int(vertical)
        out[sr : sr + tile_h, sc : sc + tile_w] = block

    return out


def _make_motif(size: int) -> np.ndarray:
    motif = np.full((size, size), GREY, dtype=int)
    if size == 3:
        motif[1, 1] = BLACK
        return motif

    holes = random.randint(1, 3)
    cells = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for r, c in random.sample(cells, holes):
        motif[r, c] = BLACK
    return motif


def _place_grid(
    motif: np.ndarray,
    rows: int,
    cols: int,
    mode: str,
    margins: tuple[int, int, int, int],
    edge_colors: dict[str, int],
) -> np.ndarray:
    top_margin, bottom_margin, left_margin, right_margin = margins
    tile_h, tile_w = motif.shape
    matrix_h = rows * tile_h
    matrix_w = cols * tile_w
    h = matrix_h + top_margin + bottom_margin
    w = matrix_w + left_margin + right_margin
    grid = np.zeros((h, w), dtype=int)
    matrix_r0 = top_margin
    matrix_c0 = left_margin

    for br in range(rows):
        for bc in range(cols):
            sr = matrix_r0 + br * tile_h
            sc = matrix_c0 + bc * tile_w
            grid[sr : sr + tile_h, sc : sc + tile_w] = motif

    row_centers = [matrix_r0 + br * tile_h + (tile_h - 1) // 2 for br in range(rows)]
    col_centers = [matrix_c0 + bc * tile_w + (tile_w - 1) // 2 for bc in range(cols)]

    if mode == "vertical_cover":
        selected_row = random.randrange(rows)
        top_col_idx, bottom_col_idx = random.sample(range(2), 2)
        positions = {
            "top": (0, col_centers[top_col_idx]),
            "bottom": (h - 1, col_centers[bottom_col_idx]),
            "left": (row_centers[selected_row], 0),
            "right": (row_centers[selected_row], w - 1),
        }
    else:
        selected_col = random.randrange(cols)
        left_row_idx, right_row_idx = random.sample(range(2), 2)
        positions = {
            "top": (0, col_centers[selected_col]),
            "bottom": (h - 1, col_centers[selected_col]),
            "left": (row_centers[left_row_idx], 0),
            "right": (row_centers[right_row_idx], w - 1),
        }

    for edge, (r, c) in positions.items():
        grid[r, c] = int(edge_colors[edge])

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert max(grid.shape) <= 30

    colors = [int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY)]
    assert len(colors) == 4 and len(set(colors)) == 4

    bbox = _bbox(grid == GREY)
    assert bbox is not None
    candidates = _layout_candidates(grid)
    assert len(candidates) == 1
    info = candidates[0]
    rows = int(info["rows"])
    cols = int(info["cols"])
    assert 2 <= rows <= 4 and 2 <= cols <= 4
    assert int(np.sum(info["motif"] == BLACK)) in (1, 2, 3)

    whole_blocks = [rec for rec in info["records"] if len(rec["visible_colors"]) == 1]
    split_blocks = [rec for rec in info["records"] if len(rec["visible_colors"]) == 2]
    assert whole_blocks, "generator must create whole-block recolors"
    assert split_blocks, "generator must create split-block recolors"

    horizontal_conflict_blocks = [
        rec
        for rec in info["records"]
        if rec["hits"]["left"] is not None
        and rec["hits"]["right"] is not None
        and int(rec["hits"]["left"]["color"]) != int(rec["hits"]["right"]["color"])
    ]
    vertical_conflict_blocks = [
        rec
        for rec in info["records"]
        if rec["hits"]["up"] is not None
        and rec["hits"]["down"] is not None
        and int(rec["hits"]["up"]["color"]) != int(rec["hits"]["down"]["color"])
    ]
    assert horizontal_conflict_blocks or vertical_conflict_blocks

    if horizontal_conflict_blocks:
        used = {int(rec["horizontal"]) for rec in horizontal_conflict_blocks}
        assert len(used) == 2, "nearer horizontal source must matter"
    if vertical_conflict_blocks:
        used = {int(rec["vertical"]) for rec in vertical_conflict_blocks}
        assert len(used) == 2, "nearer vertical source must matter"

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        mode = random.choice(("vertical_cover", "horizontal_cover"))
        tile_size = random.choice(MOTIF_SIZES)
        motif = _make_motif(tile_size)

        if mode == "vertical_cover":
            rows = random.randint(2, 4)
            cols = 2
            top_margin = random.randint(2, 6)
            bottom_margin = random.randint(2, 6)
            left_margin = random.randint(2, 6)
            right_margin = random.randint(2, 6)
            # Keep the conflict row meaningful by letting the two columns sit on opposite sides of center.
            if abs(left_margin - right_margin) > tile_size + 1:
                continue
        else:
            rows = 2
            cols = random.randint(2, 4)
            top_margin = random.randint(2, 6)
            bottom_margin = random.randint(2, 6)
            left_margin = random.randint(2, 6)
            right_margin = random.randint(2, 6)
            # Same idea for the top/bottom conflict in the two-row case.
            if abs(top_margin - bottom_margin) > tile_size + 1:
                continue

        grid_h = rows * tile_size + top_margin + bottom_margin
        grid_w = cols * tile_size + left_margin + right_margin
        if grid_h > 30 or grid_w > 30:
            continue

        edge_colors = dict(zip(("top", "bottom", "left", "right"), random.sample(SOURCE_COLORS, 4)))
        grid = _place_grid(
            motif,
            rows,
            cols,
            mode,
            (top_margin, bottom_margin, left_margin, right_margin),
            edge_colors,
        )

        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ray-visibility recoloring input")
