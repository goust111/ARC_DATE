from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, components_mask, crop_bbox, get_bg, paste, random_free_location
import random


def _template_window(g: np.ndarray, fill_color: int) -> np.ndarray:
    template_bbox = bbox(g == fill_color)
    assert template_bbox is not None, "expected a fill-color template"
    template = crop_bbox(g, template_bbox)
    assert template.shape == (3, 3), "expected a 3x3 template bbox"
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each partial copy of the yellow-marked 3x3 template by adding the template's yellow cells.

    Input structure
    - A black background contains scattered blue/teal distractors and one full 3x3 sprite whose yellow pixels mark the cells to add.
    - The yellow pixels' bounding box is the full template window.
    - Partial copies elsewhere already contain the template's non-yellow pixels in the same relative positions.

    Rule
    - Use the yellow bounding box as the 3x3 template.
    - Search every 3x3 window for an exact match to the template's non-yellow pixels.
    - At each matching window, paint yellow into the positions that are yellow in the template.

    Key insight
    - Yellow is not a target marker; it identifies the missing-cell mask to transfer onto every same-layout partial copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    fill_color = int(Color.YELLOW)
    assert fill_color != bg, "fill color must differ from background"

    template = _template_window(g, fill_color)
    assert np.all(template != bg), "template window should be fully occupied"
    fill_mask = template == fill_color
    support_mask = template != fill_color
    assert np.any(fill_mask), "template must contain fill cells"
    assert np.any(support_mask), "template must contain support cells"

    height, width = map(int, g.shape)
    th, tw = map(int, template.shape)
    origins: list[tuple[int, int]] = []
    partial_count = 0
    for r in range(height - th + 1):
        for c in range(width - tw + 1):
            window = g[r : r + th, c : c + tw]
            if not np.array_equal(window[support_mask], template[support_mask]):
                continue
            if not np.all(np.isin(window[fill_mask], [bg, fill_color])):
                continue
            origins.append((r, c))
            if np.any(window[fill_mask] == bg):
                partial_count += 1

    assert origins, "expected at least the source template to match"
    assert partial_count >= 1, "expected at least one partial copy to complete"

    out = g.copy()
    for r, c in origins:
        patch = out[r : r + th, c : c + tw]
        patch[fill_mask] = fill_color
    return out

# --------------- generator (o5x) ---------------

# Canvas/template structure.
GRID_SHAPE = (22, 22)
TEMPLATE_SIZE = 3
EMPTY_COLOR, FILL_COLOR = 0, int(Color.YELLOW)
SUPPORT_COLORS = (int(Color.BLUE), int(Color.TEAL))
FILL_COUNT_RANGE, PARTIAL_COPY_RANGE = (4, 6), (1, 2)

# Distractor sampling.
DISTRACTOR_TOTAL_RANGE = (47, 85)
DISTRACTOR_CLUSTER_MAX, DISTRACTOR_MISS_LIMIT, DISTRACTOR_CONNECTIVITY = 7, 80, 8
SINGLE_CELL_CLUSTER_PROB = 0.68
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Retry policy.
GENERATION_ATTEMPTS = 300


def _gen_template() -> tuple[np.ndarray, np.ndarray]:
    fill_count = random.randint(*FILL_COUNT_RANGE)
    fill = {
        (0, random.randrange(TEMPLATE_SIZE)),
        (TEMPLATE_SIZE - 1, random.randrange(TEMPLATE_SIZE)),
        (random.randrange(TEMPLATE_SIZE), 0),
        (random.randrange(TEMPLATE_SIZE), TEMPLATE_SIZE - 1),
    }
    remaining = [(r, c) for r in range(TEMPLATE_SIZE) for c in range(TEMPLATE_SIZE) if (r, c) not in fill]
    random.shuffle(remaining)
    for coord in remaining[: fill_count - len(fill)]:
        fill.add(coord)

    template = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=int)
    fill_mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
    for r, c in fill:
        fill_mask[r, c] = True
        template[r, c] = FILL_COLOR

    support = [(r, c) for r in range(TEMPLATE_SIZE) for c in range(TEMPLATE_SIZE) if not fill_mask[r, c]]
    support_colors = list(SUPPORT_COLORS)
    support_colors += [random.choice(SUPPORT_COLORS) for _ in range(len(support) - 2)]
    random.shuffle(support_colors)
    for (r, c), color in zip(support, support_colors):
        template[r, c] = color
    return template, fill_mask


def _gen_support_match_origins(
    grid: np.ndarray, template: np.ndarray, intended_origins: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    support_mask = template != FILL_COLOR
    bad: list[tuple[int, int]] = []
    for r in range(grid.shape[0] - TEMPLATE_SIZE + 1):
        for c in range(grid.shape[1] - TEMPLATE_SIZE + 1):
            if (r, c) in intended_origins:
                continue
            window = grid[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
            if np.array_equal(window[support_mask], template[support_mask]):
                bad.append((r, c))
    return bad


def _gen_cluster_offsets(size: int) -> list[tuple[int, int]]:
    cells = [(0, 0)]
    while len(cells) < size:
        r, c = random.choice(cells)
        dr, dc = random.choice(ORTHOGONAL_STEPS)
        nxt = (r + dr, c + dc)
        if nxt not in cells:
            cells.append(nxt)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _gen_try_place_distractor_cluster(
    grid: np.ndarray,
    offsets: list[tuple[int, int]],
    intended_origins: set[tuple[int, int]],
    reserved_mask: np.ndarray,
    template: np.ndarray,
) -> int:
    colors = [random.choice(SUPPORT_COLORS) for _ in offsets]
    max_r = grid.shape[0] - max(r for r, _ in offsets)
    max_c = grid.shape[1] - max(c for _, c in offsets)
    anchors = [(r, c) for r in range(max_r) for c in range(max_c)]
    random.shuffle(anchors)

    for r0, c0 in anchors:
        coords = [(r0 + dr, c0 + dc) for dr, dc in offsets]
        if any(grid[r, c] != EMPTY_COLOR for r, c in coords):
            continue
        trial = grid.copy()
        for (r, c), color in zip(coords, colors):
            trial[r, c] = color
        if _gen_support_match_origins(trial, template, intended_origins):
            continue
        distractor_mask = (trial != EMPTY_COLOR) & ~reserved_mask
        if any(
            len(comp) > DISTRACTOR_CLUSTER_MAX
            for comp in components_mask(distractor_mask, connectivity=DISTRACTOR_CONNECTIVITY)
        ):
            continue
        grid[:, :] = trial
        return len(coords)
    return 0


def _gen_add_distractors(
    grid: np.ndarray,
    template: np.ndarray,
    intended_origins: set[tuple[int, int]],
    reserved_mask: np.ndarray,
    target: int,
) -> bool:
    placed = 0
    misses = 0
    while placed < target and misses < DISTRACTOR_MISS_LIMIT:
        remaining = target - placed
        if remaining == 1 or random.random() < SINGLE_CELL_CLUSTER_PROB:
            size = 1
        else:
            size = random.randint(2, min(DISTRACTOR_CLUSTER_MAX, remaining))
        offsets = _gen_cluster_offsets(size)
        added = _gen_try_place_distractor_cluster(grid, offsets, intended_origins, reserved_mask, template)
        if added:
            placed += added
            misses = 0
        else:
            misses += 1
    return placed == target


def _gen_place_template_copies(
    grid: np.ndarray, template: np.ndarray
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Place the source template, then output-state copies to hide later."""
    source_origin = random_free_location(grid, template, transparent=EMPTY_COLOR)
    paste(grid, template, *source_origin)

    partial_origins: list[tuple[int, int]] = []
    for _ in range(random.randint(*PARTIAL_COPY_RANGE)):
        origin = random_free_location(grid, template, transparent=EMPTY_COLOR)
        paste(grid, template, *origin)
        partial_origins.append(origin)
    return source_origin, partial_origins


def _gen_attempt() -> np.ndarray | None:
    template, fill_mask = _gen_template()
    out = np.zeros(GRID_SHAPE, dtype=int)

    try:
        source_origin, partial_origins = _gen_place_template_copies(out, template)
    except ValueError:
        return None

    intended_origins = {source_origin, *partial_origins}
    if _gen_support_match_origins(out, template, intended_origins):
        return None

    reserved_mask = np.zeros(GRID_SHAPE, dtype=bool)
    for r, c in intended_origins:
        reserved_mask[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE] = True

    target = random.randint(*DISTRACTOR_TOTAL_RANGE)
    if not _gen_add_distractors(out, template, intended_origins, reserved_mask, target):
        return None

    inp = out.copy()
    for r, c in partial_origins:
        patch = inp[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
        patch[fill_mask] = EMPTY_COLOR

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise ValueError("failed to generate a valid bb52a14b input")
