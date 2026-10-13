from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, get_bg, paste


BG = Color.BLACK
SEPARATOR_GAP_RANGE = (1, 2)
BOTTOM_PAD_RANGE = (0, 2)
SIDE_PAD_RANGE = (0, 2)
BLOCK_ROWS_RANGE = (3, 6)
BLOCK_COLS_RANGE = (3, 6)
TEMPLATE_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400

_MASKS = {
    "full": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "x": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "tee": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    "corner": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    "zig": np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    "ring": np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    "arch": np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
}
_NON_FULL_NAMES = [name for name in _MASKS if name != "full"]
_TEMPLATE_KEYS_ALL = {tuple(mask.astype(np.uint8).ravel()) for mask in _MASKS.values()}
_NOISE_MASKS = [
    np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    for bits in range(1, 1 << 9)
    if 3 <= int(bits.bit_count()) <= 8
    and tuple(np.array([(bits >> i) & 1 for i in range(9)], dtype=np.uint8)) not in _TEMPLATE_KEYS_ALL
]


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(mask.astype(np.uint8).ravel())


def _row_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(flags.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _extract_legend_and_main(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    row_runs = _row_runs(np.any(grid != bg, axis=1))
    assert len(row_runs) >= 2, "expected legend and main region separated by blank rows"

    legend_run = row_runs[0]
    main_run = row_runs[1]
    legend = grid[legend_run[0] : legend_run[1] + 1, :]
    main_canvas = grid[main_run[0] : row_runs[-1][1] + 1, :]

    legend_box = bbox(legend != bg)
    main_box = bbox(main_canvas != bg)
    assert legend_box is not None and main_box is not None

    legend_crop = crop_bbox(legend, legend_box)
    main_crop = crop_bbox(main_canvas, main_box)
    assert legend_crop.shape[0] == 3, "legend must be 3 rows tall"
    assert main_crop.shape[0] % 3 == 0 and main_crop.shape[1] % 3 == 0, "main region must tile into 3x3 blocks"
    return legend_crop, main_crop


def _extract_templates(legend: np.ndarray, bg: int) -> list[np.ndarray]:
    colors = sorted(set(map(int, np.unique(legend))) - {int(bg)})
    assert len(colors) == 1, "legend must use one neutral template color"
    width = int(legend.shape[1])
    assert (width + 1) % 4 == 0, "legend must contain 3x3 templates with single-column gaps"
    count = (width + 1) // 4

    templates: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for i in range(count):
        c0 = 4 * i
        slot = legend[:, c0 : c0 + 3]
        assert slot.shape == (3, 3)
        if i < count - 1:
            assert np.all(legend[:, c0 + 3] == bg), "legend templates must be gap-separated"
        mask = slot != bg
        key = _mask_key(mask)
        assert mask.any(), "each legend slot must contain a template"
        assert key not in seen, "legend templates must be shape-distinct"
        seen.add(key)
        templates.append(mask)

    assert any(mask.all() for mask in templates), "legend must include a solid 3x3 template"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a block image by reverse-matching each 3x3 tile against a top legend of stencil masks.

    Input structure:
    - A top legend contains several 3x3 neutral-color templates laid out left to right, separated from the lower scene by one or more blank rows.
    - The lower main region is a rectangle whose height and width are multiples of 3, so it can be partitioned into 3x3 blocks.
    - Each main-region block is either a single-color copy of one legend template or a noisy 3x3 block that matches none of the legend shapes.

    Rule:
    - Read the legend templates as foreground masks, ignoring their neutral color.
    - For every 3x3 block in the main region, compare only its foreground-position mask against the legend masks.
    - If a block matches a legend mask and all of its foreground cells share one color, output that color in the corresponding downscaled cell; otherwise output background.

    Key insight:
    - The answer is not based on a block's dominant color: only exact shape agreement with the legend stencils certifies a real symbol, while unmatched noise collapses to background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    legend, main_region = _extract_legend_and_main(g, bg)
    template_keys = {_mask_key(mask) for mask in _extract_templates(legend, bg)}

    block_rows = int(main_region.shape[0] // 3)
    block_cols = int(main_region.shape[1] // 3)
    out = np.full((block_rows, block_cols), bg, dtype=int)

    for br in range(block_rows):
        for bc in range(block_cols):
            block = main_region[br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3]
            mask = block != bg
            if _mask_key(mask) not in template_keys:
                continue
            colors = set(map(int, np.unique(block[mask])))
            if len(colors) != 1:
                continue
            out[br, bc] = next(iter(colors))

    return out


def _choose_templates() -> list[np.ndarray]:
    count = random.randint(*TEMPLATE_COUNT_RANGE)
    chosen_names = ["full"] + random.sample(_NON_FULL_NAMES, count - 1)
    random.shuffle(chosen_names)
    return [_MASKS[name] for name in chosen_names]


def _render_template(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    return np.where(mask, int(color), int(bg)).astype(int)


def _build_noise_block(template_keys: set[tuple[int, ...]], palette: list[int], bg: int) -> np.ndarray:
    candidates = [mask for mask in _NOISE_MASKS if _mask_key(mask) not in template_keys]
    mask = random.choice(candidates)
    fg_coords = [tuple(coord) for coord in np.argwhere(mask)]

    dominant = random.choice(palette)
    block = np.full((3, 3), int(bg), dtype=int)
    if len(fg_coords) >= 2 and random.random() < 0.45:
        alt = random.choice([c for c in palette if c != dominant])
        accent_count = random.randint(1, min(2, len(fg_coords) - 1))
        accent_cells = set(random.sample(fg_coords, accent_count))
        for r, c in fg_coords:
            block[r, c] = alt if (r, c) in accent_cells else dominant
    else:
        block[mask] = dominant
    return block


def _sample_dims(template_count: int) -> tuple[int, int]:
    while True:
        block_rows = random.randint(*BLOCK_ROWS_RANGE)
        block_cols = random.randint(*BLOCK_COLS_RANGE)
        if block_rows * block_cols >= template_count + 4:
            return block_rows, block_cols


def _compose_main_region(
    templates: list[np.ndarray], palette: list[int], bg: int
) -> tuple[np.ndarray, np.ndarray]:
    block_rows, block_cols = _sample_dims(len(templates))
    main_region = np.full((block_rows * 3, block_cols * 3), int(bg), dtype=int)
    expected = np.full((block_rows, block_cols), int(bg), dtype=int)
    template_keys = {_mask_key(mask) for mask in templates}

    placements: dict[tuple[int, int], np.ndarray] = {}
    corner_positions = [(0, 0), (0, block_cols - 1), (block_rows - 1, 0), (block_rows - 1, block_cols - 1)]
    full_mask = next(mask for mask in templates if mask.all())
    for pos in corner_positions:
        placements[pos] = full_mask

    remaining = [(r, c) for r in range(block_rows) for c in range(block_cols) if (r, c) not in placements]
    random.shuffle(remaining)
    for mask in templates:
        if mask.all():
            continue
        placements[remaining.pop()] = mask

    assert remaining, "need at least one free slot for noise coverage"
    noise_slots = {remaining.pop()}
    extra_noise = random.randint(0, min(len(remaining), max(1, (block_rows * block_cols) // 4)))
    if extra_noise:
        noise_slots.update(random.sample(remaining, extra_noise))

    for r in range(block_rows):
        for c in range(block_cols):
            block_slice = (slice(r * 3, (r + 1) * 3), slice(c * 3, (c + 1) * 3))
            if (r, c) in noise_slots:
                block = _build_noise_block(template_keys, palette, bg)
            else:
                mask = placements.get((r, c), random.choice(templates))
                color = random.choice(palette)
                block = _render_template(mask, color, bg)
                expected[r, c] = color
            main_region[block_slice] = block

    return main_region, expected


def _compose_legend(templates: list[np.ndarray], neutral: int, bg: int) -> np.ndarray:
    legend = np.full((3, 4 * len(templates) - 1), int(bg), dtype=int)
    for i, mask in enumerate(templates):
        patch = _render_template(mask, neutral, bg)
        paste(legend, patch, 0, 4 * i, transparent=bg)
    return legend


def _assemble_input(legend: np.ndarray, main_region: np.ndarray, bg: int) -> np.ndarray | None:
    top_pad = 0
    gap = random.randint(*SEPARATOR_GAP_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
    legend_left = random.randint(*SIDE_PAD_RANGE)
    main_left = random.randint(*SIDE_PAD_RANGE)
    legend_right = random.randint(*SIDE_PAD_RANGE)
    main_right = random.randint(*SIDE_PAD_RANGE)

    height = top_pad + legend.shape[0] + gap + main_region.shape[0] + bottom_pad
    width = max(
        legend_left + legend.shape[1] + legend_right,
        main_left + main_region.shape[1] + main_right,
    )
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    paste(grid, legend, top_pad, legend_left, transparent=bg)
    paste(grid, main_region, top_pad + legend.shape[0] + gap, main_left, transparent=bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    neutral = random.choice([Color.GRAY, Color.TEAL])
    templates = _choose_templates()
    palette = [c for c in Color.ALL_COLORS if c not in {BG, neutral}]

    legend = _compose_legend(templates, neutral, BG)
    main_region, expected = _compose_main_region(templates, palette, BG)
    grid = _assemble_input(legend, main_region, BG)
    if grid is None:
        return None

    if grid.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid template-decoding ARC input")
