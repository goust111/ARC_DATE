from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = tuple(range(10))
NON_BLACK = tuple(color for color in PALETTE if color != BLACK)
GEN_ATTEMPTS = 400


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the inclusive bbox of a boolean mask, or None for an empty mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to an inclusive bbox."""
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _most_common_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    assert flat.size > 0
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    values = [int(x) for x in indices]
    assert values, "expected at least one separator line"
    runs: list[tuple[int, int]] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = value
        prev = value
    runs.append((start, prev))
    return runs


def _intervals_between_runs(runs: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    assert runs[0][0] == 0 and runs[-1][1] == limit - 1, "expected separator border lines"
    spans: list[tuple[int, int]] = []
    for (_start_a, end_a), (start_b, _end_b) in zip(runs, runs[1:]):
        lo = int(end_a + 1)
        hi = int(start_b - 1)
        assert lo <= hi, "expected non-empty block interval"
        spans.append((lo, hi))
    assert spans, "expected at least one block row/column"
    return spans


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    box = bbox(mask)
    assert box is not None, "every block must contain a non-empty pattern"
    cropped = crop_bbox(mask.astype(np.uint8), box)
    return tuple(map(int, cropped.shape)), cropped.tobytes()


def _parse_blocks(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_candidates = []
    for color in sorted(set(map(int, np.unique(g)))):
        row_hits = np.flatnonzero(np.all(g == color, axis=1))
        col_hits = np.flatnonzero(np.all(g == color, axis=0))
        if len(row_hits) >= 2 and len(col_hits) >= 2:
            sep_candidates.append(int(color))
    assert len(sep_candidates) == 1, "expected exactly one separator color"
    sep = sep_candidates[0]

    row_runs = _runs(np.flatnonzero(np.all(g == sep, axis=1)))
    col_runs = _runs(np.flatnonzero(np.all(g == sep, axis=0)))
    row_spans = _intervals_between_runs(row_runs, g.shape[0])
    col_spans = _intervals_between_runs(col_runs, g.shape[1])

    row_heights = [r1 - r0 + 1 for r0, r1 in row_spans]
    col_widths = [c1 - c0 + 1 for c0, c1 in col_spans]
    assert len(set(row_heights)) == 1 and len(set(col_widths)) == 1, "expected a regular block matrix"

    blocks: list[list[np.ndarray]] = []
    flat_blocks: list[np.ndarray] = []
    for r0, r1 in row_spans:
        row_blocks = []
        for c0, c1 in col_spans:
            block = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            row_blocks.append(block)
            flat_blocks.append(block)
        blocks.append(row_blocks)

    bg = _most_common_color(np.concatenate([block.ravel() for block in flat_blocks]))
    return {
        "grid": g,
        "sep": sep,
        "bg": bg,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "block_h": int(row_heights[0]),
        "block_w": int(col_widths[0]),
        "nrows": len(row_spans),
        "ncols": len(col_spans),
        "blocks": blocks,
    }


def _block_infos(blocks: list[list[np.ndarray]], bg: int) -> list[dict]:
    infos: list[dict] = []
    for br, row in enumerate(blocks):
        for bc, block in enumerate(row):
            mask = block != bg
            assert bool(mask.any()), "every block must contain a visible pattern"
            fg = block[mask]
            infos.append(
                {
                    "br": int(br),
                    "bc": int(bc),
                    "block": block,
                    "mask": mask,
                    "full_mask_sig": mask.astype(np.uint8).tobytes(),
                    "shape_sig": _mask_signature(mask),
                    "all_grey": bool(np.all(fg == GREY)),
                    "colors": tuple(sorted(set(map(int, fg)))),
                }
            )
    return infos


def _find_instruction_pair(blocks: list[list[np.ndarray]], bg: int) -> tuple[dict, dict, list[dict]]:
    infos = _block_infos(blocks, bg)

    by_full_mask: dict[bytes, list[dict]] = {}
    for info in infos:
        by_full_mask.setdefault(info["full_mask_sig"], []).append(info)

    duplicate_groups = [group for group in by_full_mask.values() if len(group) > 1]
    assert len(duplicate_groups) == 1, "expected exactly one repeated full-shape mask"
    pair_group = duplicate_groups[0]
    assert len(pair_group) == 2, "the repeated mask must occur in exactly two blocks"

    grey_blocks = [info for info in pair_group if info["all_grey"]]
    color_blocks = [info for info in pair_group if not info["all_grey"]]
    assert len(grey_blocks) == 1 and len(color_blocks) == 1, "expected one grey silhouette and one colored original"
    return grey_blocks[0], color_blocks[0], infos


def _apply_block_shift(parsed: dict, dr: int, dc: int) -> np.ndarray:
    g = parsed["grid"]
    row_spans = parsed["row_spans"]
    col_spans = parsed["col_spans"]
    blocks = parsed["blocks"]
    nrows = parsed["nrows"]
    ncols = parsed["ncols"]

    dr_mod = dr % nrows
    dc_mod = dc % ncols

    out = g.copy()
    for br, (r0, r1) in enumerate(row_spans):
        for bc, (c0, c1) in enumerate(col_spans):
            src_r = (br - dr_mod) % nrows
            src_c = (bc - dc_mod) % ncols
            out[r0 : r1 + 1, c0 : c1 + 1] = blocks[src_r][src_c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift the whole separator-partitioned block matrix by the vector encoded by a grey silhouette/original pair.

    Input structure:
    - Full separator rows and columns divide the canvas into a regular matrix of equal-sized blocks.
    - Each block contains one small pattern on a shared block background.
    - Exactly one block is a grey silhouette of another block's pattern: the occupied cells match block-cell by block-cell, but the silhouette uses only grey.

    Rule:
    - Find the grey silhouette block and its colored original block.
    - Compute the block-matrix offset (dr, dc) from the silhouette block to the original block.
    - Move every block by that same block offset with toroidal wraparound; block contents move intact, while separator lines stay fixed.

    Key insight:
    - The grey block is not the target to recolor; the only instruction is the relative block position of the repeated shape pair, and the shift happens in block coordinates rather than pixel coordinates.
    """
    parsed = _parse_blocks(np.asarray(grid, dtype=int))
    grey_block, original_block, _infos = _find_instruction_pair(parsed["blocks"], parsed["bg"])
    dr = int(original_block["br"] - grey_block["br"])
    dc = int(original_block["bc"] - grey_block["bc"])
    assert dr != 0 or dc != 0, "the instruction vector must be non-zero"
    return _apply_block_shift(parsed, dr, dc)


def _coords_to_mask(coords: set[tuple[int, int]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    box = bbox(mask)
    assert box is not None
    return crop_bbox(mask.astype(np.uint8), box).astype(bool)


def _random_connected_mask(max_h: int, max_w: int, min_area: int, max_area: int) -> np.ndarray:
    for _ in range(200):
        height = random.randint(1, max_h)
        width = random.randint(1, max_w)
        if height * width < min_area:
            continue

        area = random.randint(min_area, min(max_area, height * width))
        start = (random.randrange(height), random.randrange(width))
        coords = {start}
        while len(coords) < area:
            frontier = []
            for r, c in coords:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(frontier))

        if len(coords) != area:
            continue

        mask = _coords_to_mask(coords, height, width)
        if mask.sum() < min_area:
            continue
        return mask
    raise RuntimeError("failed to sample a connected block pattern")


def _embed_mask(mask: np.ndarray, block_h: int, block_w: int) -> np.ndarray:
    mh, mw = map(int, mask.shape)
    assert mh <= block_h and mw <= block_w
    top = random.randint(0, block_h - mh)
    left = random.randint(0, block_w - mw)
    full = np.zeros((block_h, block_w), dtype=bool)
    full[top : top + mh, left : left + mw] = mask
    return full


def _colorize_mask(full_mask: np.ndarray, usable_colors: list[int], *, force_multicolor: bool) -> np.ndarray:
    block = np.full(full_mask.shape, BLACK, dtype=int)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(full_mask)]
    assert cells
    min_k = 2 if force_multicolor and len(cells) >= 2 else 1
    max_k = min(3, len(cells))
    color_count = random.randint(min_k, max_k)
    colors = random.sample(usable_colors, color_count)
    values = list(colors)
    while len(values) < len(cells):
        values.append(random.choice(colors))
    random.shuffle(values)
    for (r, c), color in zip(cells, values):
        block[r, c] = int(color)
    return block


def _sample_pair_positions(nrows: int, ncols: int) -> tuple[tuple[int, int], tuple[int, int]]:
    modes = []
    if nrows >= 2:
        modes.append("row")
    if ncols >= 2:
        modes.append("col")
    if nrows >= 2 and ncols >= 2:
        modes.append("diag")
    mode = random.choice(modes)

    if mode == "row":
        col = random.randrange(ncols)
        r0, r1 = random.sample(range(nrows), 2)
        return (r0, col), (r1, col)
    if mode == "col":
        row = random.randrange(nrows)
        c0, c1 = random.sample(range(ncols), 2)
        return (row, c0), (row, c1)

    r0, r1 = random.sample(range(nrows), 2)
    c0, c1 = random.sample(range(ncols), 2)
    return (r0, c0), (r1, c1)


def _render_grid(
    nrows: int,
    ncols: int,
    block_h: int,
    block_w: int,
    sep: int,
    block_matrix: list[list[np.ndarray]],
) -> np.ndarray:
    height = nrows * block_h + nrows + 1
    width = ncols * block_w + ncols + 1
    grid = np.full((height, width), sep, dtype=int)
    for br in range(nrows):
        for bc in range(ncols):
            r0 = 1 + br * (block_h + 1)
            c0 = 1 + bc * (block_w + 1)
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = block_matrix[br][bc]
    return grid


def _generated_case_is_valid(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_blocks(grid)
        grey_block, original_block, infos = _find_instruction_pair(parsed["blocks"], parsed["bg"])
    except AssertionError:
        return False

    exact_block_sigs = {
        (tuple(map(int, info["block"].shape)), info["block"].tobytes())
        for info in infos
    }
    if len(exact_block_sigs) != len(infos):
        return False

    shape_groups: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for info in infos:
        shape_groups.setdefault(info["shape_sig"], []).append(info)
    repeated_shapes = [group for group in shape_groups.values() if len(group) > 1]
    if len(repeated_shapes) != 1 or len(repeated_shapes[0]) != 2:
        return False

    if sum(info["all_grey"] for info in infos) != 1:
        return False

    pair_group = repeated_shapes[0]
    if {id(info) for info in pair_group} != {id(grey_block), id(original_block)}:
        return False

    original_mask = original_block["mask"]
    original_colors = original_block["block"][original_mask]
    if len(set(map(int, original_colors))) < 2:
        return False

    dr = int(original_block["br"] - grey_block["br"])
    dc = int(original_block["bc"] - grey_block["bc"])
    if dr == 0 and dc == 0:
        return False

    expected = _apply_block_shift(parsed, dr, dc)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    matching_vectors = []
    for rr in range(parsed["nrows"]):
        for cc in range(parsed["ncols"]):
            if rr == 0 and cc == 0:
                continue
            candidate = _apply_block_shift(parsed, rr, cc)
            if np.array_equal(candidate, actual):
                matching_vectors.append((rr, cc))
    target_mod = (dr % parsed["nrows"], dc % parsed["ncols"])
    if matching_vectors != [target_mod]:
        return False

    return True


def _parameter_options() -> list[tuple[int, int, int, int]]:
    options = []
    for nrows in range(2, 5):
        for ncols in range(2, 6):
            if nrows * ncols < 6:
                continue
            for block_h in range(4, 7):
                for block_w in range(4, 7):
                    height = nrows * block_h + nrows + 1
                    width = ncols * block_w + ncols + 1
                    if height <= 30 and width <= 30:
                        options.append((nrows, ncols, block_h, block_w))
    assert options
    return options


PARAMETER_OPTIONS = _parameter_options()


def _gen_attempt() -> np.ndarray | None:
    nrows, ncols, block_h, block_w = random.choice(PARAMETER_OPTIONS)
    grey_pos, original_pos = _sample_pair_positions(nrows, ncols)

    sep = random.choice([color for color in NON_BLACK if color != GREY])
    usable_colors = [color for color in NON_BLACK if color not in (GREY, sep)]
    pair_mask = _embed_mask(
        _random_connected_mask(block_h, block_w, min_area=4, max_area=min(8, block_h * block_w - 1)),
        block_h,
        block_w,
    )

    pair_shape_sig = _mask_signature(pair_mask)
    original_block = _colorize_mask(pair_mask, usable_colors, force_multicolor=True)
    silhouette_block = np.full((block_h, block_w), BLACK, dtype=int)
    silhouette_block[pair_mask] = GREY

    block_matrix: list[list[np.ndarray]] = [
        [np.full((block_h, block_w), BLACK, dtype=int) for _ in range(ncols)]
        for _ in range(nrows)
    ]

    used_shape_sigs = {pair_shape_sig}
    used_block_sigs = {
        (tuple(map(int, original_block.shape)), original_block.tobytes()),
        (tuple(map(int, silhouette_block.shape)), silhouette_block.tobytes()),
    }

    for br in range(nrows):
        for bc in range(ncols):
            pos = (br, bc)
            if pos == original_pos:
                block_matrix[br][bc] = original_block.copy()
                continue
            if pos == grey_pos:
                block_matrix[br][bc] = silhouette_block.copy()
                continue

            placed = False
            for _ in range(200):
                shape = _random_connected_mask(block_h, block_w, min_area=3, max_area=min(8, block_h * block_w - 1))
                full_mask = _embed_mask(shape, block_h, block_w)
                shape_sig = _mask_signature(full_mask)
                if shape_sig in used_shape_sigs:
                    continue

                block = _colorize_mask(
                    full_mask,
                    usable_colors,
                    force_multicolor=bool(random.getrandbits(1)),
                )
                block_sig = (tuple(map(int, block.shape)), block.tobytes())
                if block_sig in used_block_sigs:
                    continue

                used_shape_sigs.add(shape_sig)
                used_block_sigs.add(block_sig)
                block_matrix[br][bc] = block
                placed = True
                break
            if not placed:
                return None

    grid = _render_grid(nrows, ncols, block_h, block_w, sep, block_matrix)
    if not _generated_case_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grey-shadow command puzzle")
