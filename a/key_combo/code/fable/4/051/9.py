from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
SEPARATOR = 5
BACKGROUND_OPTIONS = (0, 1, 3, 4, 6, 7, 8, 9)
SHAPE_COLORS = (1, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "shape mask must be nonempty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.shape[0] <= 1:
        return True

    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    h, w = map(int, mask.shape)
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(coords.shape[0])


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(np.array(rot, dtype=bool))
        variants.append(np.array(np.fliplr(rot), dtype=bool))
    return variants


def _shape_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    cropped = _crop_mask(mask)
    best: tuple[int, int, bytes] | None = None
    for variant in _all_transforms(cropped):
        key = (int(variant.shape[0]), int(variant.shape[1]), variant.astype(np.uint8).tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _spans_between_runs(limit: int, full_indices: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    runs = _group_runs(full_indices)
    spans: list[tuple[int, int]] = []
    start = 0
    for run_start, run_end in runs:
        if start < run_start:
            spans.append((start, run_start))
        start = run_end + 1
    if start < limit:
        spans.append((start, limit))
    if not spans:
        return None
    lengths = {end - begin for begin, end in spans}
    if 0 in lengths or len(lengths) != 1:
        return None
    return runs, spans


def _detect_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    h, w = map(int, grid.shape)
    best = None
    for color in map(int, np.unique(grid)):
        full_rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not full_rows or not full_cols:
            continue
        row_info = _spans_between_runs(h, full_rows)
        col_info = _spans_between_runs(w, full_cols)
        if row_info is None or col_info is None:
            continue
        row_runs, row_spans = row_info
        col_runs, col_spans = col_info
        score = (
            len(row_spans) * len(col_spans),
            len(full_rows) + len(full_cols),
            color == SEPARATOR,
        )
        if best is None or score > best[0]:
            best = (score, color, row_spans, col_spans, row_runs, col_runs)

    assert best is not None, "could not detect separator rows and columns"
    _score, sep_color, row_spans, col_spans, row_runs, col_runs = best
    row_sep = [r for start, end in row_runs for r in range(start, end + 1)]
    col_sep = [c for start, end in col_runs for c in range(start, end + 1)]
    return int(sep_color), row_spans, col_spans, row_sep, col_sep


def _background_color(grid: np.ndarray, row_sep: list[int], col_sep: list[int]) -> int:
    block_mask = np.ones(grid.shape, dtype=bool)
    if row_sep:
        block_mask[row_sep, :] = False
    if col_sep:
        block_mask[:, col_sep] = False
    vals = list(map(int, grid[block_mask]))
    assert vals, "block interiors must be nonempty"
    return int(Counter(vals).most_common(1)[0][0])


def _block_shape(block: np.ndarray, bg: int) -> tuple[int, np.ndarray] | None:
    fg = block != bg
    if not fg.any():
        return None
    colors = np.unique(block[fg])
    assert len(colors) == 1, "each block must contain at most one monochrome shape"
    color = int(colors[0])
    mask = _crop_mask(fg)
    assert _is_connected(mask), "each block shape must be connected"
    return color, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downsample the block array by keeping only shapes congruent to the unique red template.

    Input structure:
    - Grey separator rows and separator columns partition the grid into equal-sized rectangular blocks.
    - Exactly one block contains a red connected shape that serves as the template.
    - Every other block is either empty or contains one monochrome connected shape on a shared background.

    Rule:
    - Read the separator lattice to recover the block array.
    - Extract the red template shape, treating translation, rotation, and mirror reflection as equivalent.
    - Produce one output cell per block: copy the block's shape color when its shape is congruent to the template,
      otherwise write the background color. The template block itself stays red.

    Key insight:
    - The reduction is driven by shape identity under the full rotation/mirror family, not by occupancy or pixel count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep_color, row_spans, col_spans, row_sep, col_sep = _detect_layout(g)
    bg = _background_color(g, row_sep, col_sep)

    block_data: list[list[tuple[int, np.ndarray] | None]] = []
    red_blocks: list[tuple[int, int, tuple[int, np.ndarray]]] = []
    for rr, (r0, r1) in enumerate(row_spans):
        row_items: list[tuple[int, np.ndarray] | None] = []
        for cc, (c0, c1) in enumerate(col_spans):
            shape = _block_shape(g[r0:r1, c0:c1], bg)
            row_items.append(shape)
            if shape is not None and shape[0] == RED:
                red_blocks.append((rr, cc, shape))
        block_data.append(row_items)

    assert len(red_blocks) == 1, "expected exactly one red template block"
    template_sig = _shape_signature(red_blocks[0][2][1])

    out = np.full((len(row_spans), len(col_spans)), bg, dtype=int)
    for rr, row_items in enumerate(block_data):
        for cc, shape in enumerate(row_items):
            if shape is None:
                continue
            color, mask = shape
            if _shape_signature(mask) == template_sig:
                out[rr, cc] = color
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    mask[start] = True
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        assert frontier, "frontier unexpectedly empty before shape reached target area"
        pick = random.choice(tuple(frontier))
        cells.add(pick)
        mask[pick] = True
    return _crop_mask(mask)


def _random_shape(block_h: int, block_w: int, area: int, forbidden_sig: tuple[int, int, bytes] | None = None) -> np.ndarray:
    dims = [
        (h, w)
        for h in range(2, block_h + 1)
        for w in range(2, block_w + 1)
        if h * w >= area
    ]
    assert dims, "block is too small for any nontrivial shape"
    for _ in range(200):
        h, w = random.choice(dims)
        mask = _random_connected_mask(h, w, area)
        if mask.shape[0] > block_h or mask.shape[1] > block_w or min(mask.shape) < 2:
            continue
        sig = _shape_signature(mask)
        if forbidden_sig is None or sig != forbidden_sig:
            return mask
    raise RuntimeError("failed to sample a suitable shape")


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(np.rot90(mask, random.randrange(4)), dtype=bool)
    if random.choice((False, True)):
        out = np.array(np.fliplr(out), dtype=bool)
    return out


def _paint_block(block_h: int, block_w: int, bg: int, color: int, mask: np.ndarray) -> np.ndarray:
    options = [
        variant
        for variant in _all_transforms(mask)
        if variant.shape[0] <= block_h and variant.shape[1] <= block_w
    ]
    assert options, "shape has no fitting transform for the target block"
    shape = np.array(random.choice(options), dtype=bool)
    h, w = map(int, shape.shape)
    r0 = random.randint(0, block_h - h)
    c0 = random.randint(0, block_w - w)
    block = np.full((block_h, block_w), bg, dtype=int)
    patch = block[r0 : r0 + h, c0 : c0 + w]
    patch[shape] = color
    return block


def _assemble_grid(blocks: list[list[np.ndarray]], sep: int, sep_color: int) -> np.ndarray:
    rows = len(blocks)
    cols = len(blocks[0])
    block_h, block_w = blocks[0][0].shape
    height = rows * block_h + (rows - 1) * sep
    width = cols * block_w + (cols - 1) * sep
    grid = np.full((height, width), sep_color, dtype=int)
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (block_h + sep)
            c0 = cc * (block_w + sep)
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = blocks[rr][cc]
    return grid


def _sample_layout() -> tuple[int, int, int, int, int]:
    for _ in range(200):
        rows = random.randint(2, 4)
        cols = random.randint(3, 4)
        sep = random.choice((1, 1, 2))
        max_block_h = (30 - (rows - 1) * sep) // rows
        max_block_w = (30 - (cols - 1) * sep) // cols
        if max_block_h < 4 or max_block_w < 4:
            continue
        block_h = random.randint(4, min(7, max_block_h))
        block_w = random.randint(4, min(7, max_block_w))
        return rows, cols, block_h, block_w, sep
    raise RuntimeError("failed to sample a valid layout")


def _state_list(total: int) -> list[str]:
    states = ["template", "match", "same_area"]
    if total >= 4:
        states.append(random.choice(("other", "empty")))
    while len(states) < total:
        states.append(random.choices(
            ("match", "same_area", "other", "empty"),
            weights=(3, 2, 3, 1),
            k=1,
        )[0])
    random.shuffle(states)
    return states


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols, block_h, block_w, sep = _sample_layout()
    total = rows * cols
    bg = random.choice(BACKGROUND_OPTIONS)
    usable_colors = [color for color in SHAPE_COLORS if color != bg]
    if len(usable_colors) < 3:
        return None

    max_area = min(8, block_h * block_w - 1)
    if max_area < 4:
        return None
    area = random.randint(4, max_area)
    template = _random_shape(block_h, block_w, area)
    template_sig = _shape_signature(template)

    same_area_distractors = [
        _random_shape(block_h, block_w, area, forbidden_sig=template_sig)
        for _ in range(3)
    ]
    other_distractors = []
    for _ in range(4):
        other_area_choices = [a for a in range(3, max_area + 1) if a != area] or [area]
        other_area = random.choice(other_area_choices)
        forbidden = template_sig if other_area == area else None
        other_distractors.append(_random_shape(block_h, block_w, other_area, forbidden_sig=forbidden))

    states = _state_list(total)
    assert states.count("template") == 1
    expected = np.full((rows, cols), bg, dtype=int)
    blocks: list[list[np.ndarray]] = []

    color_pool = usable_colors[:]
    random.shuffle(color_pool)
    color_index = 0

    for rr in range(rows):
        row_blocks: list[np.ndarray] = []
        for cc in range(cols):
            state = states[rr * cols + cc]
            block = np.full((block_h, block_w), bg, dtype=int)
            if state == "template":
                block = _paint_block(block_h, block_w, bg, RED, template)
                expected[rr, cc] = RED
            elif state == "match":
                color = color_pool[color_index % len(color_pool)]
                color_index += 1
                block = _paint_block(block_h, block_w, bg, color, template)
                expected[rr, cc] = color
            elif state == "same_area":
                color = color_pool[color_index % len(color_pool)]
                color_index += 1
                mask = random.choice(same_area_distractors)
                block = _paint_block(block_h, block_w, bg, color, mask)
            elif state == "other":
                color = color_pool[color_index % len(color_pool)]
                color_index += 1
                mask = random.choice(other_distractors)
                block = _paint_block(block_h, block_w, bg, color, mask)
            row_blocks.append(block)
        blocks.append(row_blocks)

    inp = _assemble_grid(blocks, sep, SEPARATOR)
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid template-filter downsampling input")
