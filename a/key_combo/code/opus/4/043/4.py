from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_043_4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
get_bg = _ARC_UTILS.get_bg

BG = 7
WALL_COLORS = (2, 3)
MARKER_COLORS = (1, 4, 6)
YELLOW = 4

WIDTH_RANGE = (15, 23)
CORRIDOR_COUNT_RANGE = (5, 8)
GEN_ATTEMPTS = 400


def _extract_corridors(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 1 and h >= 11, "expected an odd-height stack of corridors"

    corridors: list[dict] = []
    for row in range(1, h, 2):
        wall_mask = np.isin(g[row], WALL_COLORS)
        assert wall_mask[0] and wall_mask[-1], "each corridor row needs walls on both sides"

        lo = 0
        while lo < w and wall_mask[lo]:
            lo += 1
        hi = w - 1
        while hi >= 0 and wall_mask[hi]:
            hi -= 1

        assert 1 <= lo < hi <= w - 2, "corridor opening must be internal"
        assert np.all(wall_mask[:lo]) and np.all(wall_mask[hi + 1 :]), "walls must be side bars"
        assert not np.any(wall_mask[lo : hi + 1]), "corridor interior must not contain wall colors"

        wall_color = int(g[row, 0])
        assert wall_color == int(g[row, -1]) and wall_color in WALL_COLORS

        top = g[row - 1]
        top_wall_mask = np.isin(top, WALL_COLORS)
        assert int(top[0]) == wall_color and int(top[-1]) == wall_color
        assert np.array_equal(top_wall_mask, wall_mask), "top bar row must match its corridor row"

        corridors.append({"row": row, "lo": lo, "hi": hi, "wall_color": wall_color})

    bottom = g[-1]
    bottom_wall_mask = np.isin(bottom, WALL_COLORS)
    last = corridors[-1]
    assert int(bottom[0]) == last["wall_color"] and int(bottom[-1]) == last["wall_color"]
    assert np.all(bottom_wall_mask[: last["lo"]]) and np.all(bottom_wall_mask[last["hi"] + 1 :])
    assert not np.any(bottom_wall_mask[last["lo"] : last["hi"] + 1])
    return corridors


def _build_ordered_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    corridors = _extract_corridors(grid)
    shared: list[int] = []
    for upper, lower in zip(corridors, corridors[1:]):
        matches = []
        if upper["lo"] == lower["lo"]:
            matches.append(upper["lo"])
        if upper["hi"] == lower["hi"]:
            matches.append(upper["hi"])
        assert len(matches) == 1, "adjacent corridors must share exactly one turn column"
        shared.append(matches[0])

    start_anchor = corridors[0]["hi"] if shared[0] == corridors[0]["lo"] else corridors[0]["lo"]
    end_anchor = corridors[-1]["hi"] if shared[-1] == corridors[-1]["lo"] else corridors[-1]["lo"]
    anchors = [start_anchor, *shared, end_anchor]

    path: list[tuple[int, int]] = []
    corridor_cells: list[list[tuple[int, int]]] = []
    for idx, corridor in enumerate(corridors):
        row = corridor["row"]
        c0 = int(anchors[idx])
        c1 = int(anchors[idx + 1])
        assert {c0, c1} == {corridor["lo"], corridor["hi"]}, "path must use corridor endpoints"

        step = 1 if c1 > c0 else -1
        cells = [(row, c) for c in range(c0, c1 + step, step)]
        corridor_cells.append(cells)
        if idx == 0:
            path.extend(cells)
        else:
            path.append((row - 1, c0))
            path.extend(cells)

    return path, corridor_cells


def _marker_pairs(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {BG, *WALL_COLORS}]
    assert 2 <= len(colors) <= 3, "expected 2 or 3 marker colors"

    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in sorted(colors):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(coords) == 2, "each marker color must appear exactly twice"
        coords.sort()
        assert coords[0][0] != coords[1][0], "paired markers must lie in different corridors"
        pairs[color] = (coords[0], coords[1])
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each paired marker through the corridor stack and alternately recolor the traversed path cells.

    Input structure:
    - The background is orange, with horizontal corridor rows on odd indices bounded by two same-color side bars.
    - Each corridor row has a matching bar row directly above it, and adjacent corridors share exactly one endpoint column for turning.
    - There are 2 or 3 marker pairs; each color from blue, yellow, or magenta appears exactly twice, and each pair lies in different corridors.

    Rule:
    - Reconstruct the unique snake-like corridor path by following the shared endpoint column between consecutive corridors.
    - For each marker pair, take the lexicographically earlier marker as the start and the later one as the end, then follow the ordered corridor path between them.
    - Keep the two marker endpoints unchanged, but recolor the intervening path cells in walking order: step 1,3,5... become yellow; step 2,4,6... become the pair's marker color.

    Key insight:
    - The side bars define one ordered turning path across the corridor rows, so the alternating pattern must be indexed along the whole bent route rather than restarting inside each straight corridor segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BG, "orange must be the dominant background color"

    path, _ = _build_ordered_path(g)
    index_of = {coord: idx for idx, coord in enumerate(path)}
    pairs = _marker_pairs(g)
    marker_cells = {coord for pair in pairs.values() for coord in pair}

    out = g.copy()
    painted: set[tuple[int, int]] = set()
    for color, (start, end) in sorted(pairs.items()):
        start_idx = index_of[start]
        end_idx = index_of[end]
        assert start_idx < end_idx, "marker ordering must agree with the corridor traversal"

        interior = path[start_idx + 1 : end_idx]
        assert interior, "paired markers must have at least one cell between them"
        for step, cell in enumerate(interior, start=1):
            assert cell not in marker_cells, "a path may not pass through a different marker"
            assert cell not in painted, "paired paths must be disjoint"
            assert int(g[cell]) == BG, "only background cells are recolored along the path"
            out[cell] = YELLOW if step % 2 == 1 else int(color)
            painted.add(cell)

    return out


def _sample_anchor_sequence(width: int, corridor_count: int) -> list[int]:
    left_band = list(range(2, width // 3 + 1))
    right_band = list(range((2 * width) // 3, width - 2))
    assert left_band and right_band

    use_left = random.random() < 0.5
    anchors: list[int] = []
    prev = None
    for _ in range(corridor_count + 1):
        band = left_band if use_left else right_band
        choices = [c for c in band if prev is None or abs(c - prev) >= 3]
        if not choices:
            choices = [c for c in band if prev is None or c != prev] or band
        anchors.append(random.choice(choices))
        prev = anchors[-1]
        use_left = not use_left
    return anchors


def _build_path_from_anchors(anchors: list[int]) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    path: list[tuple[int, int]] = []
    corridor_cells: list[list[tuple[int, int]]] = []
    for idx in range(len(anchors) - 1):
        row = 2 * idx + 1
        c0, c1 = int(anchors[idx]), int(anchors[idx + 1])
        step = 1 if c1 > c0 else -1
        cells = [(row, c) for c in range(c0, c1 + step, step)]
        corridor_cells.append(cells)
        if idx == 0:
            path.extend(cells)
        else:
            path.append((row - 1, c0))
            path.extend(cells)
    return path, corridor_cells


def _paint_corridor_bars(grid: np.ndarray, row: int, lo: int, hi: int, color: int) -> None:
    grid[row, :lo] = int(color)
    grid[row, hi + 1 :] = int(color)


def _sample_block_lengths(corridor_count: int, pair_count: int) -> list[int]:
    lengths = [2] * pair_count
    extra = corridor_count - 2 * pair_count
    for _ in range(extra):
        lengths[random.randrange(pair_count)] += 1
    random.shuffle(lengths)
    assert max(lengths) >= 3
    return lengths


def _biased_choices(cells: list[tuple[int, int]], from_start: bool) -> list[tuple[int, int]]:
    inner = cells[1:-1] if len(cells) >= 3 else cells[:]
    if len(inner) <= 2:
        return inner
    half = max(1, len(inner) // 2)
    return inner[:half] if from_start else inner[-half:]


def _sample_pair_colors(pair_count: int) -> list[int]:
    if pair_count == 3:
        return random.sample(list(MARKER_COLORS), 3)
    if random.random() < 0.65:
        return random.sample([1, 6], 2)
    return random.sample(list(MARKER_COLORS), 2)


def _build_candidate() -> np.ndarray:
    corridor_count = random.randint(*CORRIDOR_COUNT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    anchors = _sample_anchor_sequence(width, corridor_count)
    path, corridor_cells = _build_path_from_anchors(anchors)

    height = 2 * corridor_count + 1
    grid = np.full((height, width), BG, dtype=int)

    corridor_colors = [random.choice(WALL_COLORS) for _ in range(corridor_count)]
    for idx in range(corridor_count):
        lo = min(anchors[idx], anchors[idx + 1])
        hi = max(anchors[idx], anchors[idx + 1])
        color = corridor_colors[idx]
        _paint_corridor_bars(grid, 2 * idx, lo, hi, color)
        _paint_corridor_bars(grid, 2 * idx + 1, lo, hi, color)
    _paint_corridor_bars(
        grid,
        height - 1,
        min(anchors[-2], anchors[-1]),
        max(anchors[-2], anchors[-1]),
        corridor_colors[-1],
    )

    pair_count = 3 if corridor_count >= 7 and random.random() < 0.45 else 2
    lengths = _sample_block_lengths(corridor_count, pair_count)
    colors = _sample_pair_colors(pair_count)
    index_of = {coord: idx for idx, coord in enumerate(path)}

    blocks = []
    start_row = 0
    for length in lengths:
        blocks.append((start_row, start_row + length - 1))
        start_row += length

    for color, (first_row, last_row) in zip(colors, blocks):
        start_choices = _biased_choices(corridor_cells[first_row], from_start=True)
        end_choices = _biased_choices(corridor_cells[last_row], from_start=False)

        placed = False
        for _ in range(80):
            start = random.choice(start_choices)
            end = random.choice(end_choices)
            if index_of[end] - index_of[start] - 1 >= 4:
                grid[start] = int(color)
                grid[end] = int(color)
                placed = True
                break
        assert placed, "failed to place a marker pair with a nontrivial path"

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert max(grid.shape) <= 30
    assert int(get_bg(grid)) == BG

    path, _ = _build_ordered_path(grid)
    index_of = {coord: idx for idx, coord in enumerate(path)}
    pairs = _marker_pairs(grid)

    interior_cells: set[tuple[int, int]] = set()
    multi_turn = False
    marker_cells = {coord for pair in pairs.values() for coord in pair}
    for color, (start, end) in sorted(pairs.items()):
        start_idx = index_of[start]
        end_idx = index_of[end]
        assert start[0] % 2 == 1 and end[0] % 2 == 1
        assert start_idx < end_idx

        segment = path[start_idx : end_idx + 1]
        interior = path[start_idx + 1 : end_idx]
        visited_corridors = {r for r, _ in segment if r % 2 == 1}
        multi_turn |= len(visited_corridors) >= 3
        assert len(interior) >= 4
        for cell in interior:
            assert cell not in interior_cells, "pair paths must stay disjoint"
            assert int(grid[cell]) == BG
            interior_cells.add(cell)

    assert multi_turn, "at least one pair should demonstrate phase carry across multiple turns"

    out = main(grid)
    changed = 0
    for color, (start, end) in sorted(pairs.items()):
        start_idx = index_of[start]
        end_idx = index_of[end]
        interior = path[start_idx + 1 : end_idx]
        assert int(out[start]) == int(color) and int(out[end]) == int(color)
        for step, cell in enumerate(interior, start=1):
            expected = YELLOW if step % 2 == 1 else int(color)
            assert int(out[cell]) == expected
            changed += 1

    for cell in path:
        if cell not in marker_cells and cell not in interior_cells:
            assert int(out[cell]) == BG

    wall_mask = np.isin(grid, WALL_COLORS)
    assert np.array_equal(out[wall_mask], grid[wall_mask])
    assert changed > 0


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        try:
            _validate_instance(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid corridor-path instance")
