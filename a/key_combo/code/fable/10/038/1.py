from __future__ import annotations

import random

import numpy as np


SIDE_TO_ROT_K = {
    "top": 0,
    "right": 3,   # clockwise 90
    "bottom": 2,
    "left": 1,    # counterclockwise 90
}
SIDES = ("top", "right", "bottom", "left")
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

BG_COLOR = 0
N_RANGE = (3, 5)
TEMPLATE_DENSITY_RANGE = (0.32, 0.62)
EXTRA_SOURCE_PIXELS_RANGE = (0, 3)
GEN_RETRIES = 400


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid).tolist()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _rotations_are_distinct(mask: np.ndarray) -> bool:
    rotations = [np.rot90(mask, k=k) for k in range(4)]
    return all(
        not np.array_equal(rotations[i], rotations[j])
        for i in range(4)
        for j in range(i + 1, 4)
    )


def _candidate_panel_pairs(grid: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    h, w = map(int, grid.shape)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    if w == 2 * h:
        pairs.append((grid[:, :h], grid[:, h:]))
    if h == 2 * w:
        pairs.append((grid[:w, :], grid[w:, :]))
    return pairs


def _split_source_and_template(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    candidates = _candidate_panel_pairs(grid)
    assert len(candidates) == 1, "expected two equal square panels"

    first, second = candidates[0]
    first_colors = set(_colors_present(first))
    second_colors = set(_colors_present(second))
    shared = first_colors & second_colors
    assert len(shared) == 1, "panels must share exactly one background color"
    bg = next(iter(shared))

    matches: list[tuple[np.ndarray, np.ndarray, int]] = []
    for source, template, source_colors, template_colors in (
        (first, second, first_colors, second_colors),
        (second, first, second_colors, first_colors),
    ):
        source_fg = source_colors - {bg}
        template_fg = template_colors - {bg}
        if len(source_fg) < 2 or len(template_fg) != 1:
            continue
        if not source_fg.isdisjoint(template_fg):
            continue
        marker = next(iter(template_fg))
        template_mask = template == marker
        if not np.any(source != bg):
            continue
        if not _rotations_are_distinct(template_mask):
            continue
        matches.append((source, template, marker))

    assert len(matches) == 1, "expected a unique source panel and asymmetric template panel"
    source, template, marker = matches[0]
    return source, template, int(bg), int(marker)


def _unique_nearest_side(r: int, c: int, n: int) -> str:
    distances = {
        "top": int(r),
        "left": int(c),
        "bottom": int(n - 1 - r),
        "right": int(n - 1 - c),
    }
    nearest = min(distances.values())
    winners = [side for side, distance in distances.items() if distance == nearest]
    assert len(winners) == 1, "each source pixel must have a unique nearest side"
    return winners[0]


def _render_output(source: np.ndarray, template_mask: np.ndarray, bg: int) -> np.ndarray:
    n = int(source.shape[0])
    rotations = {
        side: np.rot90(template_mask, k=rot_k)
        for side, rot_k in SIDE_TO_ROT_K.items()
    }
    out = np.full((n * n, n * n), bg, dtype=int)
    for r, c in np.argwhere(source != bg):
        rr = int(r)
        cc = int(c)
        color = int(source[rr, cc])
        side = _unique_nearest_side(rr, cc, n)
        block = out[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n]
        block[rotations[side]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored source pixel into a rotated recolored copy of the template.

    Input structure:
    - The input is made of two equal N x N square panels, placed either side-by-side or stacked.
    - The two panels share one background color.
    - One panel is an asymmetric one-color template mask; the other is a sparse source panel with colored pixels whose nearest edge is always unique.

    Rule:
    - Build an N^2 x N^2 output filled with the shared background color.
    - For each non-background source pixel at (r, c), find which source-panel edge (top/right/bottom/left) is uniquely closest to that pixel.
    - Rotate the template so its original upward orientation points toward that nearest edge, recolor the rotated template to the source pixel's color, and stamp it into output block (r, c).

    Key insight:
    - Each source pixel simultaneously encodes block position, stamp color, and stamp orientation: the orientation is not drawn directly, but is recovered from the pixel's geometry inside the source panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    source, template, bg, marker = _split_source_and_template(g)
    n = int(source.shape[0])
    assert source.shape == (n, n) and template.shape == (n, n), "panels must be equal squares"

    template_mask = template == marker
    assert np.any(template_mask), "template mask must be non-empty"
    return _render_output(source, template_mask, bg)


def _grow_connected_mask(n: int, count: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    start = (random.randrange(n), random.randrange(n))
    mask[start] = True
    cells = {start}
    while len(cells) < count:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < n and 0 <= nc < n and not mask[nr, nc]:
                    frontier.add((nr, nc))
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
        cells.add((nr, nc))
    return mask


def _sample_template_mask(n: int) -> np.ndarray | None:
    total = n * n
    min_count = max(4, int(np.ceil(total * TEMPLATE_DENSITY_RANGE[0])))
    max_count = min(total - 3, int(np.floor(total * TEMPLATE_DENSITY_RANGE[1])))
    if min_count > max_count:
        return None

    for _ in range(120):
        count = random.randint(min_count, max_count)
        mask = _grow_connected_mask(n, count)
        if _rotations_are_distinct(mask):
            return mask
    return None


def _cells_by_unique_side(n: int) -> dict[str, list[tuple[int, int]]]:
    side_cells = {side: [] for side in SIDES}
    for r in range(n):
        for c in range(n):
            distances = [r, c, n - 1 - r, n - 1 - c]
            if distances.count(min(distances)) != 1:
                continue
            side = _unique_nearest_side(r, c, n)
            side_cells[side].append((r, c))
    return side_cells


def _sample_source_panel(n: int, marker: int, bg: int) -> np.ndarray:
    side_cells = _cells_by_unique_side(n)
    chosen = [random.choice(side_cells[side]) for side in SIDES]

    extra_pool = [
        cell
        for side in SIDES
        for cell in side_cells[side]
        if cell not in chosen
    ]
    extra_count = min(random.randint(*EXTRA_SOURCE_PIXELS_RANGE), len(extra_pool))
    if extra_count:
        chosen.extend(random.sample(extra_pool, extra_count))
    random.shuffle(chosen)

    palette_pool = [color for color in range(1, 10) if color not in {bg, marker}]
    palette_size = random.randint(2, min(5, len(chosen), len(palette_pool)))
    palette = random.sample(palette_pool, palette_size)
    colors = palette[:] + [random.choice(palette) for _ in range(len(chosen) - palette_size)]
    random.shuffle(colors)

    source = np.full((n, n), bg, dtype=int)
    for (r, c), color in zip(chosen, colors):
        source[r, c] = color
    return source


def _build_input_grid(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    source_first = random.choice([True, False])
    first = source if source_first else template
    second = template if source_first else source
    if random.choice([True, False]):
        return np.concatenate([first, second], axis=1)
    return np.concatenate([first, second], axis=0)


def _naive_unrotated_output(source: np.ndarray, template_mask: np.ndarray, bg: int) -> np.ndarray:
    n = int(source.shape[0])
    out = np.full((n * n, n * n), bg, dtype=int)
    for r, c in np.argwhere(source != bg):
        rr = int(r)
        cc = int(c)
        out[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n][template_mask] = int(source[rr, cc])
    return out


def _validate_instance(
    grid: np.ndarray,
    source: np.ndarray,
    template_mask: np.ndarray,
    bg: int,
    marker: int,
) -> bool:
    try:
        parsed_source, parsed_template, parsed_bg, parsed_marker = _split_source_and_template(grid)
    except AssertionError:
        return False

    if parsed_bg != bg or parsed_marker != marker:
        return False
    if not np.array_equal(parsed_source, source):
        return False
    if not np.array_equal(parsed_template == parsed_marker, template_mask):
        return False
    if not _rotations_are_distinct(template_mask):
        return False

    coords = np.argwhere(source != bg)
    if len(coords) < 4:
        return False
    used_sides = {_unique_nearest_side(int(r), int(c), int(source.shape[0])) for r, c in coords}
    if used_sides != set(SIDES):
        return False
    if len(_colors_present(source, exclude={bg})) < 2:
        return False

    expected = _render_output(source, template_mask, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    naive = _naive_unrotated_output(source, template_mask, bg)
    if np.array_equal(expected, naive):
        return False

    n = int(source.shape[0])
    for side in ("right", "bottom", "left"):
        side_positions = [
            (int(r), int(c))
            for r, c in coords
            if _unique_nearest_side(int(r), int(c), n) == side
        ]
        if not side_positions:
            return False
        rr, cc = side_positions[0]
        expected_block = expected[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n]
        naive_block = naive[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n]
        if np.array_equal(expected_block, naive_block):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    n = random.randint(*N_RANGE)
    marker = random.choice([color for color in range(1, 10) if color != BG_COLOR])
    template_mask = _sample_template_mask(n)
    if template_mask is None:
        return None

    template = np.full((n, n), BG_COLOR, dtype=int)
    template[template_mask] = marker
    source = _sample_source_panel(n, marker, BG_COLOR)
    grid = _build_input_grid(source, template)

    if not _validate_instance(grid, source, template_mask, BG_COLOR, marker):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid orientation-stamp input")
