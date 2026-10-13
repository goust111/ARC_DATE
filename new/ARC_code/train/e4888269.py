import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste, random_free_location, random_sprite
import random


def _legend_bbox(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates = []
    for comp in find_components(g, bg=bg, connectivity=8, monochromatic=False):
        r0, c0, r1, c1 = comp["bbox"]
        h = bbox_h(comp["bbox"])
        if bbox_w(comp["bbox"]) != 2:
            continue
        if comp["area"] != 2 * h:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(sub != bg) and len(set(map(int, sub.ravel()))) >= 3:
            candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1
    return candidates[0]


def _full_separator_lines(g: np.ndarray, bg: int, legend_bbox: tuple[int, int, int, int]) -> list[tuple[str, int, int]]:
    r0, c0, r1, c1 = legend_bbox
    lines = []
    for r in range(g.shape[0]):
        vals = set(map(int, g[r, :]))
        if len(vals) == 1 and bg not in vals and not (r0 <= r <= r1):
            lines.append(("row", r, next(iter(vals))))
    for c in range(g.shape[1]):
        vals = set(map(int, g[:, c]))
        if len(vals) == 1 and bg not in vals and not (c0 <= c <= c1):
            lines.append(("col", c, next(iter(vals))))
    return lines


def _line_separates_targets(
    g: np.ndarray,
    bg: int,
    legend_bbox: tuple[int, int, int, int],
    line: tuple[str, int, int],
) -> bool:
    r0, c0, r1, c1 = legend_bbox
    typ, idx, _color = line
    mask = np.ones(g.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    if typ == "row":
        mask[idx, :] = False
    else:
        mask[:, idx] = False
    targets = np.argwhere((g != bg) & mask)
    if targets.size == 0:
        return False
    if typ == "row":
        legend_above = r1 < idx
        targets_above = np.any(targets[:, 0] < idx)
        targets_below = np.any(targets[:, 0] > idx)
        return (legend_above and targets_below) or ((not legend_above) and targets_above)
    legend_left = c1 < idx
    targets_left = np.any(targets[:, 1] < idx)
    targets_right = np.any(targets[:, 1] > idx)
    return (legend_left and targets_right) or ((not legend_left) and targets_left)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor every target pixel by following the ordered two-column substitution legend.

    Input structure
    - A dominant background surrounds one multicolor vertical legend: a connected,
      fully filled two-column block with one color pair per row.
    - Other non-background pixels or objects are the targets; divider or frame
      pixels are ordinary target-side structure that remain unchanged unless their
      color is rewritten by the legend.

    Rule
    - Read the legend rows from top to bottom as substitutions from the left
      column color to the right column color.
    - For each non-background pixel outside the legend, scan those rows in order;
      whenever the current color equals the row's left color, replace it with the
      row's right color and keep scanning lower rows.
    - Leave the legend itself and background pixels unchanged.

    Key insight
    - The same color may be rewritten multiple times, but only by substitutions
      that appear later in the vertical legend, so the row order is part of the
      symbol instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    r0, c0, r1, c1 = _legend_bbox(g, bg)
    separator_lines = _full_separator_lines(g, bg, (r0, c0, r1, c1))
    assert len(separator_lines) <= 1, "expected at most one full separator line"
    assert all(_line_separates_targets(g, bg, (r0, c0, r1, c1), line) for line in separator_lines)
    pairs = [(int(g[r, c0]), int(g[r, c1])) for r in range(r0, r1 + 1)]

    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    out = g.copy()
    targets = np.argwhere((g != bg) & (~legend_mask))
    for r, c in targets:
        color = int(g[r, c])
        for src, dst in pairs:
            if color == src:
                color = dst
        out[r, c] = color

    return out

# --------------- generator (c) ---------------

# Canvas and palette bounds.
GRID_HEIGHT_BOUNDS = (10, 30)
GRID_WIDTH_BOUNDS = (13, 30)
COLOR_VALUES = tuple(range(10))
PALETTE_SIZE_BOUNDS = (5, 8)

# Legend construction and chain constraints.
LEGEND_WIDTH = 2
LEGEND_ROW_CHOICES = (3, 4, 5, 6, 7)
LEGEND_ROW_WEIGHTS = (1, 2, 2, 3, 4)
MIN_LEGEND_COLORS = 3
PAIR_BUILD_RETRIES = 100
CHAIN_THREE_MIN_ROWS = 4
CHAIN_THREE_PROB = 0.75
MIN_CHAIN_STEPS = 2

# Blob shape sampling.
SINGLE_PIXEL_PROB = 0.45
BLOB_HEIGHT_BOUNDS = (1, 3)
BLOB_WIDTH_BOUNDS = (1, 4)
BLOB_DENSITY = 0.7

# Layout and placement controls.
LEGEND_BORDER = 1
LEGEND_PADDING = 1
DIVIDER_PROB = 0.85
DIVIDER_TARGET_MIN_SPAN = 3
TARGET_COUNT_BOUNDS = (5, 14)
TARGET_MULTI_STEP_PROB = 0.65
TARGET_ONE_STEP_PROB = 0.85
MIN_PLACED_TARGETS = 3

# Retry limits.
GEN_ATTEMPTS = 300


def _gen_apply_chain(color: int, pairs: list[tuple[int, int]]) -> int:
    current = color
    for src, dst in pairs:
        if current == src:
            current = dst
    return current


def _gen_chain_steps(color: int, pairs: list[tuple[int, int]]) -> int:
    current = color
    steps = 0
    for src, dst in pairs:
        if current == src:
            current = dst
            steps += 1
    return steps


def _gen_build_pairs(available: list[int], n_rows: int) -> list[tuple[int, int]] | None:
    """Build an ordered legend with at least one intentional 2-3 step chain."""
    for _ in range(PAIR_BUILD_RETRIES):
        chain_len = 3 if n_rows >= CHAIN_THREE_MIN_ROWS and random.random() < CHAIN_THREE_PROB else 2
        chain = random.sample(available, chain_len + 1)
        pairs = [(chain[i], chain[i + 1]) for i in range(chain_len)]

        while len(pairs) < n_rows:
            src = random.choice(available)
            dst = random.choice([c for c in available if c != src])
            pairs.append((src, dst))

        legend_colors = {c for pair in pairs for c in pair}
        src_colors = {src for src, _dst in pairs}
        if len(legend_colors) < MIN_LEGEND_COLORS:
            continue
        if any(_gen_apply_chain(src, pairs) == src for src in src_colors):
            continue
        if not any(_gen_chain_steps(src, pairs) >= MIN_CHAIN_STEPS for src in src_colors):
            continue
        return pairs
    return None


def _gen_blob(color: int, bg: int) -> np.ndarray:
    if random.random() < SINGLE_PIXEL_PROB:
        return np.array([[color]], dtype=int)
    bh = random.randint(*BLOB_HEIGHT_BOUNDS)
    bw = random.randint(*BLOB_WIDTH_BOUNDS)
    try:
        blob = random_sprite(bh, bw, color_palette=[color], background=bg, density=BLOB_DENSITY)
    except Exception:
        blob = np.array([[color]], dtype=int)
    if np.count_nonzero(blob != bg) == 0:
        blob[0, 0] = color
    return blob


def _gen_place_in_region(grid: np.ndarray, blob: np.ndarray, bg: int, row_range: tuple[int, int], col_range: tuple[int, int]) -> bool:
    r_min, r_max = row_range
    c_min, c_max = col_range
    ph, pw = blob.shape
    rows = list(range(r_min, r_max - ph + 2))
    cols = list(range(c_min, c_max - pw + 2))
    random.shuffle(rows)
    random.shuffle(cols)
    local = np.argwhere(blob != bg)

    for r0 in rows:
        for c0 in cols:
            if np.all(grid[r0 + local[:, 0], c0 + local[:, 1]] == bg):
                paste(grid, blob, r0, c0, transparent=bg)
                return True
    return False


def _gen_place_divider(grid, available, src_colors, legend_row, legend_col, legend_rows):
    """Optionally draw a full row/column divider and return the target region."""
    height, width = grid.shape
    has_divider = random.random() < DIVIDER_PROB
    divider_color = None
    target_region = (0, height - 1, 0, width - 1)
    if has_divider:
        non_src_colors = [c for c in available if c not in src_colors]
        if non_src_colors:
            divider_color = random.choice(non_src_colors)
            options = []
            for dc in range(1, width - 1):
                if legend_col + LEGEND_WIDTH - 1 < dc and width - dc - 1 >= DIVIDER_TARGET_MIN_SPAN:
                    options.append(("col", dc, (0, height - 1, dc + 1, width - 1)))
                if dc < legend_col and dc >= DIVIDER_TARGET_MIN_SPAN:
                    options.append(("col", dc, (0, height - 1, 0, dc - 1)))
            for dr in range(1, height - 1):
                if legend_row + legend_rows - 1 < dr and height - dr - 1 >= DIVIDER_TARGET_MIN_SPAN:
                    options.append(("row", dr, (dr + 1, height - 1, 0, width - 1)))
                if dr < legend_row and dr >= DIVIDER_TARGET_MIN_SPAN:
                    options.append(("row", dr, (0, dr - 1, 0, width - 1)))
            if options:
                typ, idx, target_region = random.choice(options)
                if typ == "row":
                    grid[idx, :] = divider_color
                else:
                    grid[:, idx] = divider_color
            else:
                has_divider = False
                divider_color = None
    return has_divider, divider_color, target_region


def _gen_target_colors(available, src_colors, divider_color, pairs):
    """Choose target colors with required changed and unchanged examples."""
    non_src_non_div = [c for c in available
                       if c not in src_colors and c != divider_color]
    if not non_src_non_div:
        return None
    multi_step_src = [c for c in src_colors if _gen_chain_steps(c, pairs) >= MIN_CHAIN_STEPS]
    one_step_src = [c for c in src_colors if _gen_chain_steps(c, pairs) == 1]
    if not multi_step_src:
        return None
    n_targets = random.randint(*TARGET_COUNT_BOUNDS)
    target_colors = [random.choice(multi_step_src), random.choice(multi_step_src)]
    target_colors.append(random.choice(non_src_non_div))
    for _ in range(n_targets - MIN_PLACED_TARGETS):
        roll = random.random()
        if roll < TARGET_MULTI_STEP_PROB:
            target_colors.append(random.choice(multi_step_src))
        elif roll < TARGET_ONE_STEP_PROB and one_step_src:
            target_colors.append(random.choice(one_step_src))
        else:
            target_colors.append(random.choice(non_src_non_div))
    random.shuffle(target_colors)
    return target_colors


def _gen_place_targets(grid, target_colors, bg, target_region, has_divider):
    r_min, r_max, c_min, c_max = target_region
    placed_targets = 0
    for target_color in target_colors:
        blob = _gen_blob(target_color, bg)
        if has_divider:
            if _gen_place_in_region(grid, blob, bg, (r_min, r_max), (c_min, c_max)):
                placed_targets += 1
        else:
            try:
                tr, tc_pos = random_free_location(grid, blob, transparent=bg, padding=LEGEND_PADDING, padding_connectivity=8)
            except ValueError:
                continue
            paste(grid, blob, tr, tc_pos, transparent=bg)
            placed_targets += 1
    return placed_targets


def _gen_is_valid(grid: np.ndarray, bg: int, legend_row: int, legend_col: int, legend_rows: int) -> bool:
    """Check legend uniqueness, minimum targets, and solver acceptance."""
    try:
        _legend_bbox(grid, bg)
    except Exception:
        return False

    legend_mask = np.zeros(grid.shape, dtype=bool)
    legend_mask[legend_row:legend_row + legend_rows, legend_col:legend_col + LEGEND_WIDTH] = True
    if np.count_nonzero((grid != bg) & (~legend_mask)) < MIN_PLACED_TARGETS:
        return False

    inp = grid.copy()
    try:
        out = main(inp)
    except Exception:
        return False
    return np.array_equal(main(grid), out)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input for e4888269."""
    height = random.randint(*GRID_HEIGHT_BOUNDS)
    width = random.randint(*GRID_WIDTH_BOUNDS)
    bg = random.randint(COLOR_VALUES[0], COLOR_VALUES[-1])
    grid = np.full((height, width), bg, dtype=int)
    available = [c for c in COLOR_VALUES if c != bg]

    legend_rows = random.choices(LEGEND_ROW_CHOICES, weights=LEGEND_ROW_WEIGHTS)[0]
    n_palette = min(random.randint(*PALETTE_SIZE_BOUNDS), len(available))
    palette = random.sample(available, n_palette)
    pairs = _gen_build_pairs(palette, legend_rows)
    if pairs is None:
        return None
    src_colors = set(s for s, _dst in pairs)
    legend_block = np.array([[s, d] for s, d in pairs], dtype=int)

    try:
        legend_row, legend_col = random_free_location(grid, legend_block, transparent=bg, border=LEGEND_BORDER, padding=LEGEND_PADDING)
    except ValueError:
        return None
    paste(grid, legend_block, legend_row, legend_col)

    has_divider, divider_color, target_region = _gen_place_divider(grid, available, src_colors, legend_row, legend_col, legend_rows)

    target_colors = _gen_target_colors(available, src_colors, divider_color, pairs)
    if target_colors is None:
        return None
    placed_targets = _gen_place_targets(grid, target_colors, bg, target_region, has_divider)
    if placed_targets < MIN_PLACED_TARGETS:
        return None

    if not _gen_is_valid(grid, bg, legend_row, legend_col, legend_rows):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_ATTEMPTS} attempts")
