from __future__ import annotations

import random

import numpy as np


BG = 0
SPRITE_COLORS = (1, 3, 4, 7, 9)
BLOCK_COLORS = (2, 5, 6)

_SCENARIO_CYCLE = (
    {"above": 1, "left": 1},
    {"above": 2, "left": 0},
    {"above": 0, "left": 2},
    {"above": 2, "left": 1},
)
_scenario_index = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_monochrome_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    allowed = set(int(color) for color in colors)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color not in allowed:
                continue

            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": _bbox_coords(coord_arr),
                    "area": int(coord_arr.shape[0]),
                }
            )

    return components


def _extract_blocks(grid: np.ndarray) -> list[dict]:
    blocks = []
    for comp in _find_monochrome_components(grid, BLOCK_COLORS):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 != 2 or c1 - c0 != 2 or comp["area"] != 9:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(patch == int(comp["color"])):
            blocks.append(
                {
                    "row": r0 + 1,
                    "col": c0 + 1,
                    "color": int(comp["color"]),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    assert 2 <= len(blocks) <= 3, "expected 2 or 3 solid 3x3 generator blocks"
    rows = [int(block["row"]) for block in blocks]
    cols = [int(block["col"]) for block in blocks]
    assert len(set(rows)) == len(rows), "block center rows must be distinct"
    assert len(set(cols)) == len(cols), "block center cols must be distinct"
    return blocks


def _extract_partition_structure(grid: np.ndarray) -> tuple[list[int], list[int], tuple[int, int, int, int]]:
    parts = _find_monochrome_components(grid, SPRITE_COLORS)
    assert parts, "expected a central partition sprite"

    union = np.concatenate([part["coords"] for part in parts], axis=0)
    sprite_bbox = _bbox_coords(union)
    sr0, sc0, sr1, sc1 = sprite_bbox
    sprite = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
    assert np.all(np.isin(sprite, SPRITE_COLORS)), "sprite bbox must contain only partition colors"

    for part in parts:
        r0, c0, r1, c1 = part["bbox"]
        assert part["area"] == (r1 - r0 + 1) * (c1 - c0 + 1), "each partition must be a solid rectangle"

    row_starts = sorted({int(part["bbox"][0]) for part in parts})
    col_starts = sorted({int(part["bbox"][1]) for part in parts})
    assert 3 <= len(row_starts) <= 5, "expected 3..5 partition rows"
    assert 3 <= len(col_starts) <= 5, "expected 3..5 partition cols"
    assert len(parts) == len(row_starts) * len(col_starts), "partitions must form a full rectangle grid"

    row_ends = row_starts[1:] + [sr1 + 1]
    col_ends = col_starts[1:] + [sc1 + 1]
    part_at = {(int(part["bbox"][0]), int(part["bbox"][1])): part for part in parts}
    for ri, r0 in enumerate(row_starts):
        for ci, c0 in enumerate(col_starts):
            assert (r0, c0) in part_at, "every partition grid slot needs a rectangle"
            part = part_at[(r0, c0)]
            expected_bbox = (r0, c0, row_ends[ri] - 1, col_ends[ci] - 1)
            assert tuple(part["bbox"]) == expected_bbox, "partition rectangles must align to shared row/col cuts"

    return row_starts, col_starts, sprite_bbox


def _apply_rays(grid: np.ndarray, blocks: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for block in blocks:
        out[int(block["row"]), :] = int(block["color"])
        out[:, int(block["col"])] = int(block["color"])

    for row_block in blocks:
        for col_block in blocks:
            if row_block is col_block:
                continue
            out[int(row_block["row"]), int(col_block["col"])] = BG
    return out


def _sample_skeleton(grid: np.ndarray, row_starts: list[int], col_starts: list[int]) -> np.ndarray:
    out = np.full((len(row_starts), len(col_starts)), BG, dtype=int)
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            out[ri, ci] = int(grid[int(r), int(c)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress the partition sprite after outside 3x3 blocks repaint sample-point rays through it.

    Input structure:
    - Black background surrounds one central multicolor sprite made of 3..5 by 3..5 rectangular partitions.
    - Each partition is a solid rectangle in one of the sprite colors, and the partition grid fully tiles one bbox.
    - Outside that sprite sit 2 or 3 solid 3x3 blocks in distinct block colors, placed above and/or left of the sprite.

    Rule:
    - For every solid 3x3 block, extend a full row ray through its center row and a full column ray through its center col using the block color.
    - Whenever the center row of one block crosses the center col of another block, reset that crossing pixel to black.
    - Detect the original sprite partitions from the unmodified input, collect their top-left coordinates, then sample the post-ray grid at those coordinates.
    - Arrange the sampled colors into the compact partition skeleton grid.

    Key insight:
    - The output is not the sprite's original downscale; it samples the original partition top-left anchors after ray overwrites have changed those exact anchor pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blocks = _extract_blocks(g)
    row_starts, col_starts, _sprite_bbox = _extract_partition_structure(g)
    repainted = _apply_rays(g, blocks)
    return _sample_skeleton(repainted, row_starts, col_starts)


def _next_scenario() -> dict:
    global _scenario_index
    scenario = dict(_SCENARIO_CYCLE[_scenario_index % len(_SCENARIO_CYCLE)])
    _scenario_index += 1
    if scenario["above"] == 2 and scenario["left"] == 1 and random.random() < 0.5:
        scenario = {"above": 1, "left": 2}
    return scenario


def _random_lengths(count: int) -> list[int]:
    for _ in range(200):
        lengths = [random.choice((2, 3)) for _ in range(count)]
        if len(set(lengths)) > 1:
            return lengths
    raise RuntimeError("failed to sample non-uniform partition lengths")


def _color_matrix(n_rows: int, n_cols: int) -> np.ndarray:
    colors = np.zeros((n_rows, n_cols), dtype=int)
    target_unique = min(5, max(4, (n_rows * n_cols) // 3))
    for _ in range(300):
        for r in range(n_rows):
            for c in range(n_cols):
                forbidden = set()
                if r > 0:
                    forbidden.add(int(colors[r - 1, c]))
                if c > 0:
                    forbidden.add(int(colors[r, c - 1]))
                choices = [color for color in SPRITE_COLORS if color not in forbidden]
                colors[r, c] = random.choice(choices)
        if len(set(int(x) for x in colors.flat)) >= target_unique:
            return colors.copy()
    raise RuntimeError("failed to sample partition colors")


def _build_sprite(row_sizes: list[int], col_sizes: list[int], colors: np.ndarray) -> np.ndarray:
    sprite = np.zeros((sum(row_sizes), sum(col_sizes)), dtype=int)
    r0 = 0
    for ri, rh in enumerate(row_sizes):
        c0 = 0
        for ci, cw in enumerate(col_sizes):
            sprite[r0 : r0 + rh, c0 : c0 + cw] = int(colors[ri, ci])
            c0 += cw
        r0 += rh
    return sprite


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_blocks(
    grid: np.ndarray,
    sprite_bbox: tuple[int, int, int, int],
    row_starts: list[int],
    col_starts: list[int],
    scenario: dict,
) -> list[dict] | None:
    sr0, sc0, _sr1, _sc1 = sprite_bbox
    above_count = int(scenario["above"])
    left_count = int(scenario["left"])
    above_cols = random.sample(col_starts, above_count)
    left_rows = random.sample(row_starts, left_count)
    block_colors = random.sample(BLOCK_COLORS, above_count + left_count)

    blocks: list[dict] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    used_boxes: list[tuple[int, int, int, int]] = []

    for target_col, color in zip(above_cols, block_colors[:above_count]):
        candidate_rows = list(range(1, sr0 - 2))
        random.shuffle(candidate_rows)
        placed = False
        for center_row in candidate_rows:
            if center_row in used_rows or target_col in used_cols:
                continue
            bbox = (center_row - 1, target_col - 1, center_row + 1, target_col + 1)
            if any(_boxes_too_close(bbox, other) for other in used_boxes):
                continue
            used_rows.add(center_row)
            used_cols.add(target_col)
            used_boxes.append(bbox)
            blocks.append({"row": center_row, "col": target_col, "color": int(color), "bbox": bbox, "kind": "above"})
            placed = True
            break
        if not placed:
            return None

    for target_row, color in zip(left_rows, block_colors[above_count:]):
        candidate_cols = list(range(1, sc0 - 2))
        random.shuffle(candidate_cols)
        placed = False
        for center_col in candidate_cols:
            if target_row in used_rows or center_col in used_cols:
                continue
            bbox = (target_row - 1, center_col - 1, target_row + 1, center_col + 1)
            if any(_boxes_too_close(bbox, other) for other in used_boxes):
                continue
            used_rows.add(target_row)
            used_cols.add(center_col)
            used_boxes.append(bbox)
            blocks.append({"row": target_row, "col": center_col, "color": int(color), "bbox": bbox, "kind": "left"})
            placed = True
            break
        if not placed:
            return None

    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(block["color"])
    return blocks


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        blocks = _extract_blocks(grid)
        row_starts, col_starts, sprite_bbox = _extract_partition_structure(grid)
        base = _sample_skeleton(grid, row_starts, col_starts)
        out = main(grid)
    except Exception:
        return False

    sr0, sc0, _sr1, _sc1 = sprite_bbox
    above_blocks = [block for block in blocks if int(block["row"]) < sr0]
    left_blocks = [block for block in blocks if int(block["col"]) < sc0]
    if len(above_blocks) + len(left_blocks) != len(blocks):
        return False
    if any(int(block["col"]) not in col_starts for block in above_blocks):
        return False
    if any(int(block["row"]) not in row_starts for block in left_blocks):
        return False

    changed = out != base
    if not np.any(changed):
        return False
    if np.all(changed):
        return False

    row_index = {int(r): idx for idx, r in enumerate(row_starts)}
    col_index = {int(c): idx for idx, c in enumerate(col_starts)}
    target_row_ids = {row_index[int(block["row"])] for block in left_blocks}
    target_col_ids = {col_index[int(block["col"])] for block in above_blocks}

    for block in above_blocks:
        ci = col_index[int(block["col"])]
        expected = np.full(len(row_starts), int(block["color"]), dtype=int)
        for left in left_blocks:
            expected[row_index[int(left["row"])]] = BG
        if not np.array_equal(out[:, ci], expected):
            return False

    for block in left_blocks:
        ri = row_index[int(block["row"])]
        expected = np.full(len(col_starts), int(block["color"]), dtype=int)
        for above in above_blocks:
            expected[col_index[int(above["col"])]] = BG
        if not np.array_equal(out[ri, :], expected):
            return False

    for ri in range(out.shape[0]):
        for ci in range(out.shape[1]):
            if ri in target_row_ids or ci in target_col_ids:
                continue
            if int(out[ri, ci]) != int(base[ri, ci]):
                return False

    zero_count = int(np.count_nonzero(out == BG))
    if above_blocks and left_blocks:
        if zero_count != len(above_blocks) * len(left_blocks):
            return False
        for block_r in left_blocks:
            for block_c in above_blocks:
                if int(out[row_index[int(block_r["row"])], col_index[int(block_c["col"])]]) != BG:
                    return False
    elif zero_count != 0:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    scenario = _next_scenario()
    n_rows = random.randint(3, 5)
    n_cols = random.randint(3, 5)
    if scenario["left"] >= n_rows or scenario["above"] >= n_cols:
        return None

    row_sizes = _random_lengths(n_rows)
    col_sizes = _random_lengths(n_cols)
    sprite = _build_sprite(row_sizes, col_sizes, _color_matrix(n_rows, n_cols))
    sprite_h, sprite_w = map(int, sprite.shape)

    top_margin = random.randint(scenario["above"] + 4 if scenario["above"] else 2, scenario["above"] + 7 if scenario["above"] else 6)
    left_margin = random.randint(scenario["left"] + 4 if scenario["left"] else 2, scenario["left"] + 7 if scenario["left"] else 6)
    bottom_margin = random.randint(2, 5)
    right_margin = random.randint(2, 5)

    h = top_margin + sprite_h + bottom_margin
    w = left_margin + sprite_w + right_margin
    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    sr0, sc0 = top_margin, left_margin
    grid[sr0 : sr0 + sprite_h, sc0 : sc0 + sprite_w] = sprite

    row_starts = [sr0]
    for height in row_sizes[:-1]:
        row_starts.append(row_starts[-1] + int(height))
    col_starts = [sc0]
    for width in col_sizes[:-1]:
        col_starts.append(col_starts[-1] + int(width))

    sprite_bbox = (sr0, sc0, sr0 + sprite_h - 1, sc0 + sprite_w - 1)
    blocks = _place_blocks(grid, sprite_bbox, row_starts, col_starts, scenario)
    if blocks is None:
        return None

    if not _validate_semantics(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
