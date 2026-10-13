from __future__ import annotations

from collections import deque
import random

import numpy as np


ORTHOGONAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _same_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHOGONAL_STEPS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            objects.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return objects


def _ordered_objects(panel: np.ndarray, bg: int) -> list[dict]:
    objects = _same_color_components(panel, bg)
    assert len(objects) == 3, "each panel must contain exactly three foreground objects"

    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0], obj["bbox"][3], obj["bbox"][2]))
    previous_right = -1
    for obj in objects:
        _r0, c0, _r1, c1 = obj["bbox"]
        assert c0 > previous_right, "panel objects must have a strict left-to-right order"
        previous_right = c1
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge one ordered object from each of three separator-split panels.

    The input is divided into three equal-width panels by two full-height separator
    columns. Each panel contains exactly three 4-connected foreground objects
    arranged from left to right. The output is a single panel-sized canvas that
    keeps the leftmost object from the left panel, the middle object from the
    middle panel, and the rightmost object from the right panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert w >= 5 and (w - 2) % 3 == 0, "expected three equal-width panels and two separators"

    panel_w = (w - 2) // 3
    bg = _dominant_color(g)

    sep_left = int(g[0, panel_w])
    sep_right = int(g[0, 2 * panel_w + 1])
    assert sep_left == sep_right and sep_left != bg, "separator columns must match and differ from bg"
    assert bool(np.all(g[:, panel_w] == sep_left)), "first separator column must be solid"
    assert bool(np.all(g[:, 2 * panel_w + 1] == sep_left)), "second separator column must be solid"

    panels = [
        g[:, :panel_w],
        g[:, panel_w + 1 : 2 * panel_w + 1],
        g[:, 2 * panel_w + 2 :],
    ]
    assert all(panel.shape == (h, panel_w) for panel in panels)

    out = np.full((h, panel_w), bg, dtype=int)
    for panel_index, panel in enumerate(panels):
        obj = _ordered_objects(panel, bg)[panel_index]
        coords = obj["coords"]
        rr = coords[:, 0]
        cc = coords[:, 1]
        assert not bool((out[rr, cc] != bg).any()), "selected objects must not overlap in the output"
        out[rr, cc] = int(obj["color"])

    return out


# --------------- generator ---------------

GRID_HEIGHTS = tuple(range(5, 9))
PANEL_WIDTHS = tuple(range(7, 12))
COLOR_POOL = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
MAX_SLOT_FILL_RATIO = 0.6


def _split_nonnegative(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _slot_ranges(panel_w: int) -> list[tuple[int, int]]:
    slack = panel_w - 5
    assert slack >= 0
    left_margin, extra_w0, extra_gap0, extra_w1, extra_gap1, extra_w2, right_margin = _split_nonnegative(
        slack, 7
    )

    widths = [1 + extra_w0, 1 + extra_w1, 1 + extra_w2]
    gaps = [1 + extra_gap0, 1 + extra_gap1]

    ranges = []
    start = left_margin
    for i, width in enumerate(widths):
        end = start + width - 1
        ranges.append((start, end))
        if i < 2:
            start = end + 1 + gaps[i]

    assert ranges[-1][1] + 1 + right_margin == panel_w
    return ranges


def _random_polyomino(height: int, width: int, area: int) -> np.ndarray:
    assert 1 <= area <= height * width

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ORTHOGONAL_STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < area:
        choices = list(frontier)
        choices.sort(
            key=lambda rc: sum((rc[0] + dr, rc[1] + dc) in cells for dr, dc in ORTHOGONAL_STEPS),
            reverse=True,
        )
        pick_pool = choices[: max(1, min(3, len(choices)))] if random.random() < 0.5 else choices
        chosen = random.choice(pick_pool)
        frontier.remove(chosen)
        cells.add(chosen)
        add_frontier(*chosen)

    return np.array(sorted(cells), dtype=int)


def _sample_object_area(height: int, width: int) -> int:
    capacity = height * width
    high = max(1, min(capacity, int(np.ceil(MAX_SLOT_FILL_RATIO * capacity))))
    return random.randint(1, high)


def _render_panel(
    height: int, panel_w: int, bg: int, slot_ranges: list[tuple[int, int]], color: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    panel = np.full((height, panel_w), bg, dtype=int)
    objects: list[np.ndarray] = []

    for c0, c1 in slot_ranges:
        slot_w = c1 - c0 + 1
        area = _sample_object_area(height, slot_w)
        local_coords = _random_polyomino(height, slot_w, area)
        coords = local_coords.copy()
        coords[:, 1] += c0
        panel[coords[:, 0], coords[:, 1]] = color
        objects.append(coords)

    return panel, objects


def _build_expected(height: int, panel_w: int, bg: int, selected: list[tuple[np.ndarray, int]]) -> np.ndarray:
    out = np.full((height, panel_w), bg, dtype=int)
    for coords, color in selected:
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _build_input(
    height: int,
    panel_w: int,
    bg: int,
    separator: int,
    panel_colors: list[int],
    slot_ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    width = 3 * panel_w + 2
    grid = np.full((height, width), bg, dtype=int)
    grid[:, panel_w] = separator
    grid[:, 2 * panel_w + 1] = separator

    selected: list[tuple[np.ndarray, int]] = []
    for panel_index, color in enumerate(panel_colors):
        panel, objects = _render_panel(height, panel_w, bg, slot_ranges, color)
        start_c = panel_index * (panel_w + 1)
        grid[:, start_c : start_c + panel_w] = panel
        selected.append((objects[panel_index], color))

    return grid, _build_expected(height, panel_w, bg, selected)


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    return count_map.get(bg, 0) == max(count_map.values()) and list(count_map.values()).count(count_map.get(bg, 0)) == 1


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(GRID_HEIGHTS)
    panel_w = random.choice(PANEL_WIDTHS)
    bg, separator, color0, color1, color2 = random.sample(COLOR_POOL, 5)
    slot_ranges = _slot_ranges(panel_w)

    grid, expected = _build_input(height, panel_w, bg, separator, [color0, color1, color2], slot_ranges)
    if not _background_is_dominant(grid, bg):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
