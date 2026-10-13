from typing import Iterable

import numpy as np


BG = 0
SEP = 5
BLOCK = 4
GEN_ATTEMPTS = 300
MASK_ATTEMPTS = 200
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
TOP_LAYOUTS = ((1, 2), (1, 3), (1, 4), (2, 2), (2, 3))
BOTTOM_LAYOUTS = ((1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (3, 2), (3, 3))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _extract_blocks(panel: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    row_runs = _true_runs(np.any(panel != BG, axis=1))
    col_runs = _true_runs(np.any(panel != BG, axis=0))
    assert row_runs and col_runs, "panel must contain at least one block"
    assert all(r1 - r0 == BLOCK for r0, r1 in row_runs), "each block row band must be 4 cells high"
    assert all(c1 - c0 == BLOCK for c0, c1 in col_runs), "each block col band must be 4 cells wide"

    blocks: list[tuple[int, int, np.ndarray]] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = panel[r0:r1, c0:c1]
            if np.any(block != BG):
                blocks.append((r0, c0, block.copy()))
    assert blocks, "panel must contain at least one non-empty 4x4 block"
    return blocks


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _split_halves(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    assert len(sep_rows) == 1, "expected exactly one grey separator row"
    sep = sep_rows[0]
    top = grid[:sep, :]
    bottom = grid[sep + 1 :, :]
    assert top.size > 0 and bottom.size > 0, "separator must split the grid into non-empty halves"
    return sep, top, bottom


def _template_mapping(top: np.ndarray) -> dict[tuple[int, bytes], tuple[int, np.ndarray]]:
    mapping: dict[tuple[int, bytes], tuple[int, np.ndarray]] = {}
    for _r0, _c0, block in _extract_blocks(top):
        assert block.shape == (BLOCK, BLOCK)
        assert np.all(block != BG), "reference templates must fill their full 4x4 blocks"
        colors = [int(color) for color in np.unique(block)]
        assert len(colors) == 2, "each reference template must use exactly two colors"
        for color in colors:
            current_mask = block == color
            assert np.all(np.any(current_mask, axis=1)), "template half must touch every row"
            assert np.all(np.any(current_mask, axis=0)), "template half must touch every column"
            other_color = next(other for other in colors if other != color)
            key = (color, _mask_key(current_mask))
            assert key not in mapping, "template halves must be unambiguous"
            mapping[key] = (other_color, block == other_color)
    assert mapping, "reference panel must define at least one mapping"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each lower 4x4 single-color half-pattern using the complementary
    half of its matching 4x4 two-color template from the reference panel.
    """
    g = np.asarray(grid, dtype=int)
    sep, top, bottom = _split_halves(g)
    mapping = _template_mapping(top)

    out = g.copy()
    for r0, c0, block in _extract_blocks(bottom):
        colors = [int(color) for color in np.unique(block) if int(color) != BG]
        assert len(colors) == 1, "each lower block must contain exactly one foreground color"
        color = colors[0]
        mask = block == color
        assert np.all(np.any(mask, axis=1)), "generated lower blocks must touch every row"
        assert np.all(np.any(mask, axis=0)), "generated lower blocks must touch every column"
        key = (color, _mask_key(mask))
        assert key in mapping, "each lower block must match one template half"
        other_color, other_mask = mapping[key]
        rr = slice(sep + 1 + r0, sep + 1 + r0 + BLOCK)
        cc = slice(c0, c0 + BLOCK)
        region = out[rr, cc]
        region[other_mask] = other_color

    return out


def _random_half_mask(rng: np.random.Generator, used: set[bytes]) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        mask = rng.integers(0, 2, size=(BLOCK, BLOCK), endpoint=False).astype(bool)
        row_sums = mask.sum(axis=1)
        col_sums = mask.sum(axis=0)
        total = int(mask.sum())
        if not (5 <= total <= 11):
            continue
        if not np.all((1 <= row_sums) & (row_sums <= BLOCK - 1)):
            continue
        if not np.all((1 <= col_sums) & (col_sums <= BLOCK - 1)):
            continue
        if _component_count(mask) > 3 or _component_count(~mask) > 3:
            continue
        key = _mask_key(mask)
        comp_key = _mask_key(~mask)
        if key in used or comp_key in used:
            continue
        return mask
    return None


def _panel_extent(rows: int, cols: int, gap_r: int, gap_c: int) -> tuple[int, int]:
    height = rows * BLOCK + (rows - 1) * gap_r
    width = cols * BLOCK + (cols - 1) * gap_c
    return height, width


def _place_block(grid: np.ndarray, top: int, left: int, block: np.ndarray) -> None:
    h, w = block.shape
    grid[top : top + h, left : left + w] = block


def _slots(rows: int, cols: int, top: int, left: int, gap_r: int, gap_c: int) -> Iterable[tuple[int, int]]:
    for r in range(rows):
        for c in range(cols):
            yield top + r * (BLOCK + gap_r), left + c * (BLOCK + gap_c)


def _build_template_block(mask: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    return np.where(mask, color_a, color_b).astype(int)


def _build_query_block(mask: np.ndarray, color_a: int, color_b: int, use_first: bool) -> np.ndarray:
    block = np.zeros((BLOCK, BLOCK), dtype=int)
    if use_first:
        block[mask] = color_a
    else:
        block[~mask] = color_b
    return block


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    top_rows, top_cols = TOP_LAYOUTS[rng.integers(len(TOP_LAYOUTS))]
    template_count = top_rows * top_cols

    bottom_choices = [dims for dims in BOTTOM_LAYOUTS if dims[0] * dims[1] >= template_count]
    bottom_rows, bottom_cols = bottom_choices[rng.integers(len(bottom_choices))]
    query_count = bottom_rows * bottom_cols

    used_masks: set[bytes] = set()
    templates: list[tuple[np.ndarray, int, int]] = []
    for _ in range(template_count):
        mask = _random_half_mask(rng, used_masks)
        if mask is None:
            return None
        color_a, color_b = [int(x) for x in rng.choice(TEMPLATE_COLORS, size=2, replace=False)]
        templates.append((mask, color_a, color_b))
        used_masks.add(_mask_key(mask))
        used_masks.add(_mask_key(~mask))

    query_templates = list(range(template_count))
    query_templates.extend(int(x) for x in rng.integers(0, template_count, size=query_count - template_count))
    rng.shuffle(query_templates)

    half_choices = rng.integers(0, 2, size=query_count).tolist()
    if query_count >= 2 and len(set(half_choices)) == 1:
        half_choices[0] = 0
        half_choices[1] = 1
        rng.shuffle(half_choices)

    top_gap_r = int(rng.integers(1, 4))
    top_gap_c = int(rng.integers(1, 4))
    bottom_gap_r = int(rng.integers(1, 4))
    bottom_gap_c = int(rng.integers(1, 4))
    top_h, top_w = _panel_extent(top_rows, top_cols, top_gap_r, top_gap_c)
    bottom_h, bottom_w = _panel_extent(bottom_rows, bottom_cols, bottom_gap_r, bottom_gap_c)

    left_margin = int(rng.integers(1, 3))
    right_margin = int(rng.integers(1, 3))
    extra_width = int(rng.integers(0, 4))
    width = max(top_w, bottom_w) + left_margin + right_margin + extra_width

    top_top_margin = int(rng.integers(1, 3))
    top_bottom_margin = int(rng.integers(1, 3))
    bottom_top_margin = int(rng.integers(1, 3))
    bottom_bottom_margin = int(rng.integers(1, 3))
    sep_row = top_top_margin + top_h + top_bottom_margin
    height = sep_row + 1 + bottom_top_margin + bottom_h + bottom_bottom_margin

    grid = np.zeros((height, width), dtype=int)
    grid[sep_row, :] = SEP

    top_left_max = width - right_margin - top_w
    bottom_left_max = width - right_margin - bottom_w
    top_left = int(rng.integers(left_margin, top_left_max + 1))
    bottom_left = int(rng.integers(left_margin, bottom_left_max + 1))
    top_origin_r = top_top_margin
    bottom_origin_r = sep_row + 1 + bottom_top_margin

    for (r0, c0), (mask, color_a, color_b) in zip(
        _slots(top_rows, top_cols, top_origin_r, top_left, top_gap_r, top_gap_c), templates
    ):
        _place_block(grid, r0, c0, _build_template_block(mask, color_a, color_b))

    for idx, (r0, c0) in enumerate(_slots(bottom_rows, bottom_cols, bottom_origin_r, bottom_left, bottom_gap_r, bottom_gap_c)):
        template_idx = query_templates[idx]
        mask, color_a, color_b = templates[template_idx]
        use_first = bool(half_choices[idx])
        _place_block(grid, r0, c0, _build_query_block(mask, color_a, color_b, use_first))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.array_equal(out[: sep_row + 1, :], grid[: sep_row + 1, :]):
        return None
    if not np.any(out[sep_row + 1 :, :] != grid[sep_row + 1 :, :]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
