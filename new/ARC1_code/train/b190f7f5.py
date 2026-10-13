import numpy as np

from arc_utils import colors_present
import random


def _panel_colors(panel: np.ndarray) -> set[int]:
    return set(colors_present(panel))


def _split_source_and_template(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[np.ndarray, np.ndarray]] = []

    if width == 2 * height:
        candidates.append((grid[:, :height], grid[:, height:]))
    if height == 2 * width:
        candidates.append((grid[:width, :], grid[width:, :]))
    assert len(candidates) == 1, "expected two equal square panels"

    first, second = candidates[0]
    first_colors = _panel_colors(first)
    second_colors = _panel_colors(second)
    shared = first_colors & second_colors
    assert len(shared) == 1, "expected one shared background color"
    bg = next(iter(shared))

    matches: list[tuple[np.ndarray, np.ndarray, int]] = []
    for source, template, source_colors, template_colors in (
        (first, second, first_colors, second_colors),
        (second, first, second_colors, first_colors),
    ):
        source_fg = source_colors - {bg}
        template_fg = template_colors - {bg}
        if len(source_fg) >= 1 and len(template_fg) == 1 and source_fg.isdisjoint(template_fg):
            marker = next(iter(template_fg))
            matches.append((source, template, marker))

    assert len(matches) == 1, "expected unique source panel and one-color template panel"
    source, template, marker = matches[0]
    return source, template, bg, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored source pixel into a recolored copy of the companion mask template.

    Input structure:
    - The input is two equal square panels, either side-by-side or stacked.
    - The two panels share exactly one background color.
    - One panel is a source grid with colored foreground pixels; the other is a mask template using one marker color.

    Rule:
    - Make an N^2 x N^2 output for N x N panels, initialized to the background.
    - For each non-background source pixel at source position (r, c), recolor the template marker mask with that pixel's color.
    - Stamp that recolored mask into the corresponding N x N output block at block position (r, c).

    Key insight:
    - The source grid supplies the block positions and colors, while the separate one-color panel supplies the repeated sprite shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source, template, bg, marker = _split_source_and_template(g)
    n = int(source.shape[0])
    assert source.shape == (n, n), "source panel must be square"
    assert template.shape == (n, n), "template panel must match source size"

    template_mask = template == marker
    assert np.any(template_mask), "template marker mask must be nonempty"

    out = np.full((n * n, n * n), bg, dtype=int)
    for r in range(n):
        for c in range(n):
            color = int(source[r, c])
            if color == bg:
                continue
            block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
            block[template_mask] = color

    return out

# --------------- generator (o5x) ---------------
# Fixed colors and palette choices.
BACKGROUND_COLOR, TEMPLATE_MARKER_COLOR = 0, 8
SOURCE_COLOR_POOL = tuple(range(1, 10))
SOURCE_PALETTE_RANGE = (2, 4)
# Panel sizes and fill bounds: (min_abs, min_ratio, max_bg, max_ratio).
PANEL_SIZE_RANGE = (3, 5)
TEMPLATE_COUNT_BOUNDS = (3, (3, 10), 2, (4, 5))
SOURCE_COUNT_BOUNDS = (2, (1, 5), 1, (3, 5))
# Template splitting and retry controls.
SPLIT_TEMPLATE_PROB = 0.35
GENERATION_RETRIES = 300
NEIGHBOR_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _count_range(total: int, min_abs: int, min_ratio: tuple[int, int], max_bg: int, max_ratio: tuple[int, int]) -> tuple[int, int]:
    """Return inclusive foreground-count bounds from absolute and ratio caps."""
    min_num, min_den = min_ratio
    max_num, max_den = max_ratio
    return max(min_abs, (min_num * total + min_den - 1) // min_den), min(total - max_bg, (max_num * total) // max_den)

def _gen_connected_shape(height: int, width: int, count: int) -> np.ndarray:
    """Grow exactly count cells by random 4-neighbor expansion."""
    mask = np.zeros((height, width), dtype=bool)
    r = random.randrange(height)
    c = random.randrange(width)
    mask[r, c] = True
    cells = {(r, c)}
    while len(cells) < count:
        frontier = set()
        for cr, cc in cells:
            for dr, dc in NEIGHBOR_STEPS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
        cells.add((nr, nc))
    return mask

def _gen_template_mask(n: int, count: int) -> np.ndarray:
    """Return one connected mask, or a sampled two-component split mask."""
    options = []
    for gap in range(1, n - 1):
        cap_a = n * gap
        cap_b = n * (n - gap - 1)
        for first_count in range(1, count):
            second_count = count - first_count
            if first_count <= cap_a and second_count <= cap_b:
                options.append(("v", gap, first_count, second_count))
                options.append(("h", gap, first_count, second_count))

    if options and random.random() < SPLIT_TEMPLATE_PROB:
        orient, gap, count_a, count_b = random.choice(options)
        mask = np.zeros((n, n), dtype=bool)
        if orient == "v":
            mask[:, :gap] = _gen_connected_shape(n, gap, count_a)
            mask[:, gap + 1 :] = _gen_connected_shape(n, n - gap - 1, count_b)
        else:
            mask[:gap, :] = _gen_connected_shape(gap, n, count_a)
            mask[gap + 1 :, :] = _gen_connected_shape(n - gap - 1, n, count_b)
        return mask

    return _gen_connected_shape(n, n, count)

def _gen_template_panel(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample the template mask and render it with the marker color."""
    total = n * n
    count_min, count_max = _count_range(total, *TEMPLATE_COUNT_BOUNDS)
    template_count = random.randint(count_min, count_max)
    template_mask = _gen_template_mask(n, template_count)
    template = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    template[template_mask] = TEMPLATE_MARKER_COLOR
    return template, template_mask

def _gen_source_panel(n: int) -> np.ndarray:
    """Sample source positions and colors, using every palette color once."""
    total = n * n
    count_min, count_max = _count_range(total, *SOURCE_COUNT_BOUNDS)
    source_count = random.randint(count_min, count_max)
    palette_min, palette_max = SOURCE_PALETTE_RANGE
    palette_size = random.randint(palette_min, min(palette_max, source_count))
    palette = random.sample([c for c in SOURCE_COLOR_POOL if c != TEMPLATE_MARKER_COLOR], palette_size)
    positions = random.sample([(r, c) for r in range(n) for c in range(n)], source_count)
    colors = palette[:] + [random.choice(palette) for _ in range(source_count - palette_size)]
    random.shuffle(colors)
    source = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    for (r, c), color in zip(positions, colors):
        source[r, c] = color
    return source

def _gen_expected_output(source: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    """Render the solver oracle for a source panel and template mask."""
    n = int(source.shape[0])
    out = np.full((n * n, n * n), BACKGROUND_COLOR, dtype=int)
    for r, c in np.argwhere(source != BACKGROUND_COLOR):
        out[r * n : (r + 1) * n, c * n : (c + 1) * n][template_mask] = int(source[r, c])
    return out

def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*PANEL_SIZE_RANGE)
    template, template_mask = _gen_template_panel(n)
    source = _gen_source_panel(n)

    source_first = random.choice([True, False])
    left_or_top = source if source_first else template
    right_or_bottom = template if source_first else source
    if random.choice([True, False]):
        grid = np.concatenate([left_or_top, right_or_bottom], axis=1)
    else:
        grid = np.concatenate([left_or_top, right_or_bottom], axis=0)

    expected = _gen_expected_output(source, template_mask)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b190f7f5 input")
