from __future__ import annotations

from collections import deque
import random

import numpy as np


# This file stays self-contained so SourceFileLoader can import it directly, but
# its helpers follow arc_utils.py conventions: grids use (r, c) indexing and
# components use 4-connectivity unless stated otherwise.

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9

INPUT_COLORS = (GREEN, YELLOW, ORANGE, TEAL, MAROON)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)

PANEL_COUNT = 3
HEIGHT_RANGE = (10, 14)
PANEL_WIDTH_CHOICES = (8, 9)
OBJECTS_PER_PANEL = (3, 4, 5)
OBJECT_AREA_RANGE = (3, 8)
PRESENCE_COUNT_WEIGHTS = ((3, 4, 5), (4, 4, 1))
FREQ_WEIGHTS = ((1, 2, 3), (3, 4, 2))
MAX_GENERATION_ATTEMPTS = 400
SHAPE_RESTARTS = 60
SHAPE_SEED_LIMIT = 80
COMPACT_BIAS = 0.65
COMPACT_TOP_K = 4


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if _in_bounds(mask.shape, nr, nc) and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 1 and w >= 5, "grid too small for three panels"

    separator_cols = [c for c in range(w) if bool(np.all(g[:, c] == GREY))]
    assert len(separator_cols) == 2, "expected exactly two full-height grey separator columns"
    sep_a, sep_b = map(int, separator_cols)
    assert sep_a + 1 < sep_b, "separator columns must leave a middle panel"

    panels = [
        g[:, :sep_a],
        g[:, sep_a + 1 : sep_b],
        g[:, sep_b + 1 :],
    ]
    assert len(panels) == PANEL_COUNT and all(panel.shape[1] > 0 for panel in panels)
    return panels, (sep_a, sep_b)


def _panel_color_sets(panels: list[np.ndarray]) -> list[set[int]]:
    return [set(map(int, np.unique(panel))) - {BLACK, GREY} for panel in panels]


def _panel_frequencies(panels: list[np.ndarray]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for colors in _panel_color_sets(panels):
        for color in colors:
            counts[color] = counts.get(color, 0) + 1
    return counts


def _target_color(freq: int) -> int:
    assert freq in (1, 2, 3), "panel frequency must be 1, 2, or 3"
    return RED if freq == 1 else BLUE if freq == 3 else BLACK


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    panels, _separators = _split_panels(g)
    freq = _panel_frequencies(panels)
    out = g.copy()
    for color, panel_count in freq.items():
        out[g == int(color)] = _target_color(int(panel_count))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor or erase panel objects according to how many panels share their color.

    Input structure
    - The grid has black background and two full-height grey separator columns that split it into three side-by-side panels.
    - Each panel contains several 4-connected foreground objects drawn with colors from {3, 4, 7, 8, 9}.
    - The same foreground color may appear in one panel, two panels, or all three panels.

    Rule
    - Find the three panels using the grey separator columns.
    - For each non-black, non-grey color, count in how many different panels that color appears at least once.
    - Recolor every pixel of colors with panel frequency 1 to red, recolor every pixel of colors with panel frequency 3 to blue, and erase every pixel of colors with panel frequency 2 to black.
    - Keep the grey separator columns unchanged and return a grid of the same size.

    Key insight
    - The decisive feature is global color presence across panels, not any single object's shape, size, or position inside one panel.
    """
    return _expected_output(np.asarray(grid, dtype=int))


def _sample_panel_presence() -> tuple[list[list[int]], dict[int, int]] | None:
    n_colors = random.choices(*PRESENCE_COUNT_WEIGHTS)[0]
    colors = random.sample(INPUT_COLORS, n_colors)

    freq_plan = {
        colors[0]: 1,
        colors[1]: 2,
        colors[2]: 3,
    }
    for color in colors[3:]:
        freq_plan[color] = random.choices(*FREQ_WEIGHTS)[0]

    present_in_panel: list[list[int]] = [[] for _ in range(PANEL_COUNT)]
    for color, freq in freq_plan.items():
        if freq == 1:
            chosen = [random.randrange(PANEL_COUNT)]
        elif freq == 2:
            chosen = random.sample(range(PANEL_COUNT), 2)
        else:
            chosen = list(range(PANEL_COUNT))
        for panel_index in chosen:
            present_in_panel[panel_index].append(int(color))

    if not all(2 <= len(colors_here) <= 4 for colors_here in present_in_panel):
        return None
    return present_in_panel, freq_plan


def _sample_object_colors(panel_colors: list[int]) -> list[int]:
    count = random.randint(max(3, len(panel_colors)), max(5, len(panel_colors)))
    colors = list(panel_colors)
    while len(colors) < count:
        colors.append(random.choice(panel_colors))
    random.shuffle(colors)
    return colors


def _sample_sizes(count: int, capacity: int) -> list[int] | None:
    lo = OBJECT_AREA_RANGE[0] * count
    hi = min(OBJECT_AREA_RANGE[1] * count, capacity)
    if lo > hi:
        return None
    total = random.randint(lo, hi)
    sizes: list[int] = []
    remaining = total
    for index in range(count):
        left = count - index - 1
        low = max(OBJECT_AREA_RANGE[0], remaining - OBJECT_AREA_RANGE[1] * left)
        high = min(OBJECT_AREA_RANGE[1], remaining - OBJECT_AREA_RANGE[0] * left)
        if low > high:
            return None
        sizes.append(random.randint(low, high))
        remaining -= sizes[-1]
    random.shuffle(sizes)
    return sizes


def _shape_is_good(cells: set[tuple[int, int]]) -> bool:
    if len(cells) <= 4:
        return True
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    return len(rows) > 1 and len(cols) > 1


def _can_add_cell(
    cell: tuple[int, int],
    shape_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    r, c = cell
    if not _in_bounds(shape, r, c) or cell in shape_cells or cell in occupied:
        return False
    for nr, nc in _neighbors8(r, c):
        if _in_bounds(shape, nr, nc) and (nr, nc) in occupied:
            return False
    return True


def _grow_shape(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    size: int,
) -> set[tuple[int, int]] | None:
    h, w = shape
    seeds = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if _can_add_cell((r, c), set(), occupied, shape)
    ]
    random.shuffle(seeds)

    for seed in seeds[:SHAPE_SEED_LIMIT]:
        for _ in range(SHAPE_RESTARTS):
            cells = {seed}
            while len(cells) < size:
                frontier: set[tuple[int, int]] = set()
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c):
                        if _can_add_cell((nr, nc), cells, occupied, shape):
                            frontier.add((nr, nc))
                if not frontier:
                    break

                options = list(frontier)
                options.sort(
                    key=lambda rc: sum((rr, cc) in cells for rr, cc in _neighbors4(*rc)),
                    reverse=True,
                )
                if random.random() < COMPACT_BIAS:
                    top_k = max(1, min(COMPACT_TOP_K, len(options)))
                    chosen = random.choice(options[:top_k])
                else:
                    chosen = random.choice(options)
                cells.add(chosen)

            if len(cells) == size and _shape_is_good(cells):
                return cells
    return None


def _build_panel(height: int, width: int, object_colors: list[int]) -> np.ndarray | None:
    panel = np.zeros((height, width), dtype=int)
    size_budget = min((height * width) // 3, 24)
    sizes = _sample_sizes(len(object_colors), size_budget)
    if sizes is None:
        return None

    specs = list(zip(object_colors, sizes))
    random.shuffle(specs)
    specs.sort(key=lambda item: int(item[1]), reverse=True)
    occupied: set[tuple[int, int]] = set()

    for color, size in specs:
        cells = _grow_shape((height, width), occupied, int(size))
        if cells is None:
            return None
        for r, c in cells:
            panel[r, c] = int(color)
        occupied.update(cells)

    return panel


def _render_input(panels: list[np.ndarray]) -> np.ndarray:
    h = panels[0].shape[0]
    widths = [panel.shape[1] for panel in panels]
    out = np.full((h, sum(widths) + 2), BLACK, dtype=int)
    cursor = 0
    for index, panel in enumerate(panels):
        pw = panel.shape[1]
        out[:, cursor : cursor + pw] = panel
        cursor += pw
        if index < PANEL_COUNT - 1:
            out[:, cursor] = GREY
            cursor += 1
    return out


def _monochrome_objects(panel: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(set(map(int, np.unique(panel))) - {BLACK}):
        for comp in _components_mask(panel == color):
            coords = [tuple(map(int, rc)) for rc in comp]
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(set(coords)),
                }
            )
    return objects


def _assert_objects_are_separated(panel: np.ndarray, objects: list[dict]) -> None:
    owner = np.full(panel.shape, -1, dtype=int)
    for obj_index, obj in enumerate(objects):
        for r, c in obj["coords"]:
            assert owner[r, c] == -1
            owner[r, c] = obj_index

    for r in range(panel.shape[0]):
        for c in range(panel.shape[1]):
            if owner[r, c] == -1:
                continue
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(panel.shape, nr, nc) and owner[nr, nc] not in (-1, owner[r, c]):
                    raise AssertionError("objects must stay visually separated")


def _validate_case(inp: np.ndarray, out: np.ndarray) -> None:
    panels, (sep_a, sep_b) = _split_panels(inp)
    assert inp.shape == out.shape
    assert bool(np.all(inp[:, sep_a] == GREY)) and bool(np.all(inp[:, sep_b] == GREY))
    assert bool(np.array_equal(out[:, sep_a], inp[:, sep_a]))
    assert bool(np.array_equal(out[:, sep_b], inp[:, sep_b]))
    assert set(map(int, np.unique(inp))) <= {BLACK, GREY, *INPUT_COLORS}

    objects_per_panel: list[list[dict]] = []
    for panel in panels:
        assert GREY not in set(map(int, np.unique(panel)))
        objs = _monochrome_objects(panel)
        assert 3 <= len(objs) <= 5, "each panel must contain 3-5 objects"
        for obj in objs:
            assert OBJECT_AREA_RANGE[0] <= obj["area"] <= OBJECT_AREA_RANGE[1]
        _assert_objects_are_separated(panel, objs)
        objects_per_panel.append(objs)

    freq = _panel_frequencies(panels)
    values = set(freq.values())
    assert {1, 2, 3}.issubset(values), "all three frequency classes must appear"

    expected = _expected_output(inp)
    assert bool(np.array_equal(out, expected)), "solver output must match semantic rule"
    for color, panel_count in freq.items():
        mask = inp == int(color)
        assert mask.any()
        assert bool(np.all(out[mask] == _target_color(int(panel_count))))


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(*HEIGHT_RANGE)
        panel_width = random.choice(PANEL_WIDTH_CHOICES)
        sampled = _sample_panel_presence()
        if sampled is None:
            continue
        present_in_panel, _freq_plan = sampled

        panels: list[np.ndarray] = []
        failed = False
        for panel_colors in present_in_panel:
            object_colors = _sample_object_colors(panel_colors)
            panel = _build_panel(height, panel_width, object_colors)
            if panel is None:
                failed = True
                break
            panels.append(panel)
        if failed:
            continue

        inp = _render_input(panels)
        out = main(inp)
        try:
            _validate_case(inp, out)
        except AssertionError:
            continue
        return inp, out

    raise RuntimeError("failed to generate a valid ARC sample")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
