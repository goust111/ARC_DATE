from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the one large connected colored object from a noisy field and return its tight crop.

    Input structure:
    - The background is the most common color.
    - Non-background pixels are mostly isolated or tiny distractor components.
    - Exactly one 4-connected monochrome foreground object has the largest area.

    Rule:
    - Find all 4-connected monochrome foreground components.
    - Select the unique largest component.
    - Return its bounding box filled with background except for that component's pixels, preserving internal background holes.

    Key insight:
    - The signal object is distinguished by connected area; the scattered pixels are smaller noise, regardless of their colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one foreground component"

    max_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(largest) == 1, "expected a unique largest foreground component"

    comp = largest[0]
    color = int(comp["color"])
    assert color != bg, "selected component must be foreground"
    coords = comp["coords"]
    assert coords.ndim == 2 and coords.shape[1] == 2, "component coordinates must be row/col pairs"

    r0, c0, r1, c1 = map(int, comp["bbox"])
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return out

# --------------- generator (o5x) ---------------
# Canvas and palette.
GEN_CANVAS_SHAPE = (30, 30)
GEN_BACKGROUND_COLOR = 0
GEN_COLOR_VALUES = range(1, 10)
GEN_FG_COLOR_COUNT_RANGE = (2, 4)
# Signal object shape and placement.
GEN_SIGNAL_AREA_RANGE = (6, 12)
GEN_SIGNAL_MARGIN_RANGE = (1, 5)
GEN_SIGNAL_DIMS_BY_AREA = {
    6: [(3, 3)],
    7: [(3, 3)],
    8: [(3, 4), (4, 3)],
    9: [(3, 4), (4, 3)],
    10: [(5, 3)],
    11: [(4, 4), (5, 3)],
    12: [(4, 4), (5, 3)],
}
GEN_NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
# Distractor sizing, placement, density limits, and retry budget.
GEN_DISTRACTOR_COUNT_RANGE = (40, 110)
GEN_DISTRACTOR_AREA_WEIGHTS = [84, 12, 3, 1]
GEN_MAX_DISTRACTOR_AREA = 4
GEN_DISTRACTOR_PADDING = 1
GEN_DISTRACTOR_PADDING_CONNECTIVITY = 4
GEN_FG_COUNT_RANGE = (45, 144)
GEN_MAX_ATTEMPTS = 300


def _gen_choose_palette() -> tuple[int, list[int]]:
    all_colors = random.sample(GEN_COLOR_VALUES, random.randint(*GEN_FG_COLOR_COUNT_RANGE))
    signal_color = random.choice(all_colors)
    return signal_color, [color for color in all_colors if color != signal_color]


def _gen_signal_shape(color: int) -> np.ndarray:
    area = random.randint(*GEN_SIGNAL_AREA_RANGE)
    h, w = random.choice(GEN_SIGNAL_DIMS_BY_AREA[area])
    hub_r = random.randint(1, h - 2)
    hub_c = random.randint(1, w - 2)

    mask = np.zeros((h, w), dtype=bool)
    mask[:, hub_c] = True
    mask[hub_r, :] = True

    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in GEN_NEIGHBORS_4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
        r, c = random.choice(tuple(frontier))
        mask[r, c] = True

    patch = np.zeros((h, w), dtype=int)
    patch[mask] = int(color)
    return patch


def _gen_component_patch(area: int, color: int) -> np.ndarray:
    coords = {(0, 0)}
    while len(coords) < int(area):
        frontier = set()
        for r, c in coords:
            for dr, dc in GEN_NEIGHBORS_4:
                p = (r + dr, c + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(tuple(frontier)))

    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = [(r - min_r, c - min_c) for r, c in coords]
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.zeros((h, w), dtype=int)
    rows, cols = zip(*norm)
    patch[rows, cols] = int(color)

    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = patch.T
    return patch.copy()


def _gen_distractor_sizes(count: int, max_area: int, signal_area: int) -> list[int]:
    weights = GEN_DISTRACTOR_AREA_WEIGHTS[: int(max_area)]
    sizes = [random.choices(range(1, int(max_area) + 1), weights=weights, k=1)[0] for _ in range(int(count))]
    max_total = GEN_FG_COUNT_RANGE[1] - int(signal_area)
    while sum(sizes) > max_total:
        big = [i for i, size in enumerate(sizes) if size > 1]
        if not big:
            break
        sizes[random.choice(big)] -= 1
    return sizes


def _gen_color_assignments(colors: list[int], count: int) -> list[int]:
    assignments = list(colors)
    while len(assignments) < int(count):
        assignments.append(random.choice(colors))
    random.shuffle(assignments)
    return assignments


def _gen_place_signal(grid: np.ndarray, signal: np.ndarray) -> tuple[int, int, int, int]:
    canvas_h, canvas_w = GEN_CANVAS_SHAPE
    h, w = signal.shape
    margin = random.randint(*GEN_SIGNAL_MARGIN_RANGE)
    r0 = random.randint(margin, canvas_h - h - margin)
    c0 = random.randint(margin, canvas_w - w - margin)
    paste(grid, signal, r0, c0, transparent=GEN_BACKGROUND_COLOR)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _gen_place_distractors(
    grid: np.ndarray,
    signal_bbox: tuple[int, int, int, int],
    distractor_colors: list[int],
    count: int,
    max_area: int,
    signal_area: int,
) -> bool:
    occ = np.zeros_like(grid)
    r0, c0, r1, c1 = signal_bbox
    occ[r0 : r1 + 1, c0 : c1 + 1] = 1

    sizes = _gen_distractor_sizes(count, max_area, signal_area)
    colors = _gen_color_assignments(distractor_colors, count)
    pairs = list(zip(sizes, colors))
    random.shuffle(pairs)

    for area, color in pairs:
        patch = _gen_component_patch(area, color)
        occ_patch = (patch != 0).astype(int)
        try:
            rr, cc = random_free_location(occ, occ_patch, transparent=GEN_BACKGROUND_COLOR,
                                          padding=GEN_DISTRACTOR_PADDING,
                                          padding_connectivity=GEN_DISTRACTOR_PADDING_CONNECTIVITY)
        except ValueError:
            return False
        paste(grid, patch, rr, cc, transparent=GEN_BACKGROUND_COLOR)
        paste(occ, occ_patch, rr, cc, transparent=GEN_BACKGROUND_COLOR)
    return True


def _gen_passes_oracle(grid: np.ndarray, signal: np.ndarray) -> bool:
    fg = int(np.count_nonzero(grid))
    if not (GEN_FG_COUNT_RANGE[0] <= fg <= GEN_FG_COUNT_RANGE[1]):
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(out, signal))


def _gen_attempt() -> np.ndarray | None:
    signal_color, distractor_colors = _gen_choose_palette()
    signal = _gen_signal_shape(signal_color)
    signal_area = int(np.count_nonzero(signal))

    grid = np.zeros(GEN_CANVAS_SHAPE, dtype=int)
    signal_bbox = _gen_place_signal(grid, signal)

    count = random.randint(*GEN_DISTRACTOR_COUNT_RANGE)
    max_distractor_area = max(1, min(GEN_MAX_DISTRACTOR_AREA, signal_area // 3))
    if not _gen_place_distractors(
        grid,
        signal_bbox,
        distractor_colors,
        count,
        max_distractor_area,
        signal_area,
    ):
        return None

    if not _gen_passes_oracle(grid, signal):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 1f85a75f input")
