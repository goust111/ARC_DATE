from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def _ordered_objects(panel: np.ndarray, bg: int) -> list[dict]:
    objects = find_components(panel, bg=bg, connectivity=4, monochromatic=True)
    assert len(objects) == 3, "each panel must contain exactly three monochrome objects"

    objects = sorted(objects, key=lambda obj: obj["bbox"][1])
    previous_right = -1
    for obj in objects:
        _r0, c0, _r1, c1 = obj["bbox"]
        assert c0 > previous_right, "panel objects must be strictly ordered left to right"
        previous_right = c1
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine the left, middle, and right ordered objects from three separated panels.

    Input structure
    - The grid is three HxH panels separated by one-cell vertical separator bars.
    - The background is the dominant color, and each panel contains exactly three same-color 4-connected foreground objects.

    Rule
    - Build a new HxH background canvas.
    - From the left panel copy its leftmost object, from the middle panel copy its middle object, and from the right panel copy its rightmost object.

    Key insight
    - The separator bars create three aligned layers, and the panel's own left-to-right object order selects which object survives in the merged output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert W == 3 * H + 2, "expected three square panels separated by two columns"

    bg = get_bg(g)
    sep_a = int(g[0, H])
    sep_b = int(g[0, 2 * H + 1])
    assert sep_a == sep_b and sep_a != bg, "separator bars must use the same non-bg color"
    assert bool(np.all(g[:, H] == sep_a)), "first separator column must be solid"
    assert bool(np.all(g[:, 2 * H + 1] == sep_a)), "second separator column must be solid"

    panels = [
        g[:, :H],
        g[:, H + 1 : 2 * H + 1],
        g[:, 2 * H + 2 :],
    ]
    assert all(panel.shape == (H, H) for panel in panels), "panels must be HxH"

    out = np.full((H, H), bg, dtype=int)
    for panel_index, panel in enumerate(panels):
        obj = _ordered_objects(panel, bg)[panel_index]
        color = int(obj["color"])
        for r, c in obj["coords"]:
            rr, cc = int(r), int(c)
            assert out[rr, cc] == bg, "selected objects must not overlap"
            out[rr, cc] = color

    return out

# --------------- generator (att2) ---------------

# Canvas and palette values observed in the task.
_BG = 7
_SEP = 6
_COLORS = [0, 1, 2, 3, 4, 5, 8, 9]

# Panel geometry and object ordering.
_GRID_SIZES = [5, 6]
_N_PANELS = 3
_OBJECTS_PER_PANEL = 3
_DISTRACTOR_KEYS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

# Object-size, density, and color-reuse controls.
_MAX_OBJECT_AREA = 8
_AREA_BUDGET_BY_SIZE = {5: 56, 6: 72}
_MAX_COLOR_REUSES = 2

# Random-growth shape controls.
_FRONTIER_COMPACT_BIAS = 0.35
_FRONTIER_COMPACT_TOP_K = 3
_NEIGHBOR_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Retry limit for rejecting failed growth or oracle mismatches.
_MAX_ATTEMPTS = 300


def _segments(H: int) -> list[tuple[int, int]]:
    a, b = sorted(random.sample(range(1, H), 2))
    return [(0, a - 1), (a, b - 1), (b, H - 1)]


def _object_patch(H: int, cols: tuple[int, int], color: int, area: int) -> np.ndarray | None:
    c0, c1 = cols
    W = c1 - c0 + 1
    target = int(area)
    if target < 1 or target > H * W:
        return None

    start = (random.randrange(H), random.randrange(W))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in _NEIGHBOR_DELTAS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in cells:
                frontier.add((rr, cc))

    add_frontier(*start)
    while len(cells) < target:
        if not frontier:
            return None
        if random.random() < _FRONTIER_COMPACT_BIAS:
            options = sorted(
                frontier,
                key=lambda rc: sum(
                    (rc[0] + dr, rc[1] + dc) in cells for dr, dc in _NEIGHBOR_DELTAS
                ),
                reverse=True,
            )
            top_k = max(1, min(_FRONTIER_COMPACT_TOP_K, len(options)))
            cell = random.choice(options[:top_k])
        else:
            cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    patch = np.full((H, H), _BG, dtype=int)
    for r, c in cells:
        patch[r, c0 + c] = int(color)
    return patch


def _choose_area(H: int, cols: tuple[int, int], budget: int, remaining_slots: int) -> int | None:
    width = cols[1] - cols[0] + 1
    high = min(_MAX_OBJECT_AREA, H * width, budget - (remaining_slots - 1))
    if high < 1:
        return None
    return random.randint(1, high)


def _assign_colors(selected: list[int]) -> dict[tuple[int, int], int] | None:
    colors = {(i, i): int(selected[i]) for i in range(_OBJECTS_PER_PANEL)}
    used_by_panel = [set() for _ in range(_N_PANELS)]
    counts = {c: 0 for c in _COLORS}
    for i, color in enumerate(selected):
        used_by_panel[i].add(int(color))
        counts[int(color)] += 1

    distractors = list(_DISTRACTOR_KEYS)
    random.shuffle(distractors)
    for panel_i, obj_i in distractors:
        options = [
            color
            for color in _COLORS
            if counts[color] < _MAX_COLOR_REUSES and color not in used_by_panel[panel_i]
        ]
        if not options:
            return None
        color = random.choice(options)
        colors[(panel_i, obj_i)] = color
        used_by_panel[panel_i].add(color)
        counts[color] += 1
    return colors


def _sample_patch(H: int, segs: list[tuple[int, int]], obj_i: int, color: int, budget: int, remaining: int):
    area = _choose_area(H, segs[obj_i], budget, remaining)
    if area is None:
        return None
    patch = _object_patch(H, segs[obj_i], color, area)
    if patch is None:
        return None
    return patch, budget - area, remaining - 1


def _sample_objects(H: int, segs: list[tuple[int, int]], selected_colors: list[int], colors):
    """Return all panel objects plus the expected merged output."""
    budget = _AREA_BUDGET_BY_SIZE[H]
    remaining = _N_PANELS * _OBJECTS_PER_PANEL
    objects: dict[tuple[int, int], np.ndarray] = {}
    out = np.full((H, H), _BG, dtype=int)

    for i in range(_OBJECTS_PER_PANEL):
        sampled = _sample_patch(H, segs, i, selected_colors[i], budget, remaining)
        if sampled is None:
            return None
        patch, budget, remaining = sampled
        objects[(i, i)] = patch
        paste(out, patch, 0, 0, transparent=_BG)

    for panel_i, obj_i in _DISTRACTOR_KEYS:
        sampled = _sample_patch(H, segs, obj_i, colors[(panel_i, obj_i)], budget, remaining)
        if sampled is None:
            return None
        patch, budget, remaining = sampled
        objects[(panel_i, obj_i)] = patch
    return objects, out


def _render_input(H: int, objects: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Render all objects into three panels separated by solid bars."""
    inp = np.full((H, _N_PANELS * H + (_N_PANELS - 1)), _BG, dtype=int)
    for sep_i in range(1, _N_PANELS):
        inp[:, sep_i * H + (sep_i - 1)] = _SEP
    for panel_i in range(_N_PANELS):
        panel = np.full((H, H), _BG, dtype=int)
        for obj_i in range(_OBJECTS_PER_PANEL):
            paste(panel, objects[(panel_i, obj_i)], 0, 0, transparent=_BG)
        paste(inp, panel, 0, panel_i * (H + 1))
    return inp


def _passes_oracle(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_GRID_SIZES)
    segs = _segments(H)
    selected_colors = random.sample(_COLORS, _OBJECTS_PER_PANEL)
    colors = _assign_colors(selected_colors)
    if colors is None:
        return None

    sampled = _sample_objects(H, segs, selected_colors, colors)
    if sampled is None:
        return None

    objects, out = sampled
    inp = _render_input(H, objects)
    if _passes_oracle(inp, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 5ecac7f7")
