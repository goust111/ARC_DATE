from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components


MAX_GENERATION_ATTEMPTS = 400
GEN_MODES = ("full", "stop_immediate", "stop_late")


@dataclass(frozen=True)
class Pillar:
    color: int
    base_col: int
    height: int
    top_row: int


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, list[Pillar], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    height, _width = map(int, g.shape)
    assert int(np.count_nonzero(g == Color.BLACK)) < g.size

    components = find_components(g, bg=Color.BLACK, connectivity=4, monochromatic=True)
    pillars: list[Pillar] = []
    marker_candidates: list[tuple[int, int]] = []
    pillar_colors: set[int] = set()

    for component in components:
        color = int(component["color"])
        area = int(component["area"])
        coords = component["coords"]
        r0, c0, r1, c1 = component["bbox"]

        is_grounded_vertical = c0 == c1 and r1 == height - 1 and area == (r1 - r0 + 1)
        if is_grounded_vertical:
            pillars.append(Pillar(color=color, base_col=int(c0), height=area, top_row=int(r0)))
            pillar_colors.add(color)
            continue

        if area == 1:
            rr, cc = map(int, coords[0])
            assert rr < height - 1, "marker cannot sit on the ground row"
            marker_candidates.append((rr, cc))
            continue

        raise AssertionError("every foreground component must be a pillar or the marker")

    assert len(pillars) >= 2, "expected at least two pillars"
    assert len(pillar_colors) == len(pillars), "pillar colors must be distinct"
    assert len(marker_candidates) == 1, "expected exactly one marker"
    marker_r, marker_c = marker_candidates[0]

    pillars.sort(key=lambda pillar: pillar.base_col)
    matches: list[tuple[int, int]] = []
    for index, pillar in enumerate(pillars):
        if marker_r != pillar.top_row:
            continue
        if marker_c == pillar.base_col - 1:
            matches.append((index, -1))
        if marker_c == pillar.base_col + 1:
            matches.append((index, 1))
    assert len(matches) == 1, "marker must sit beside exactly one pillar top"
    return g, pillars, matches[0][0], matches[0][1]


def _chain_indices(pillars: list[Pillar], start_index: int, direction: int) -> list[int]:
    toppled = [int(start_index)]
    current_index = int(start_index)
    while True:
        next_index = current_index + int(direction)
        if not (0 <= next_index < len(pillars)):
            break
        current = pillars[current_index]
        nxt = pillars[next_index]
        gap = abs(nxt.base_col - current.base_col)
        if gap > current.height:
            break
        toppled.append(next_index)
        current_index = next_index
    return toppled


def _render_output(
    shape: tuple[int, int],
    pillars: list[Pillar],
    toppled: list[int],
    direction: int,
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.zeros((height, width), dtype=int)
    toppled_set = set(toppled)

    for index, pillar in enumerate(pillars):
        if index in toppled_set:
            continue
        out[height - pillar.height : height, pillar.base_col] = pillar.color

    for index in toppled:
        pillar = pillars[index]
        if direction == 1:
            start_col = pillar.base_col
            stop_col = pillar.base_col + pillar.height
        else:
            start_col = pillar.base_col - pillar.height + 1
            stop_col = pillar.base_col + 1
        assert 0 <= start_col < stop_col <= width
        out[height - 1, start_col:stop_col] = pillar.color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Knock the marked pillar sideways and propagate the fall to reachable pillars on that side.

    Input structure:
    - The grid has a black background, with several distinct-color width-1 pillars standing on the bottom row.
    - Exactly one single marker pixel floats immediately to the left or right of one pillar's top cell.
    - Pillar heights and horizontal spacings vary, and every pillar is a single vertical segment.

    Rule:
    - The marked pillar falls toward the marker, turning into a same-color horizontal bar on the bottom row with length equal to its height.
    - Repeatedly inspect the nearest still-standing pillar further in that same direction; if its base column is within the current fallen pillar's height, it also falls in the same direction, otherwise the chain stops there.
    - Later fallen bars overwrite earlier fallen bars where they overlap, untouched pillars stay standing, and the marker is removed.

    Key insight:
    - Only the ordered pillar heights and base-column gaps on the marked side matter: once one gap exceeds the current pillar's height, every farther pillar survives unchanged.
    """
    g, pillars, start_index, direction = _parse_scene(grid)
    toppled = _chain_indices(pillars, start_index, direction)
    return _render_output(g.shape, pillars, toppled, direction)


def _sample_chain_heights(chain_len: int) -> list[int]:
    if chain_len == 1:
        return [random.randint(2, 8)]
    return [random.randint(3, 8) for _ in range(chain_len)]


def _sample_chain_gaps(chain_heights: list[int]) -> list[int]:
    if len(chain_heights) <= 1:
        return []

    edge_count = len(chain_heights) - 1
    if edge_count == 1:
        kinds = [random.choice(["exact", "overlap"])]
    else:
        kinds = [random.choice(["exact", "overlap"]) for _ in range(edge_count)]
        if "exact" not in kinds:
            kinds[random.randrange(edge_count)] = "exact"
        if "overlap" not in kinds:
            kinds[random.randrange(edge_count)] = "overlap"

    gaps: list[int] = []
    for height, kind in zip(chain_heights[:-1], kinds):
        if kind == "exact":
            gaps.append(height)
        else:
            gaps.append(random.randint(2, height - 1))
    return gaps


def _build_canonical_scene() -> tuple[np.ndarray, dict]:
    mode = random.choice(GEN_MODES)
    left_extra = random.randint(0, 2)

    if mode == "full":
        chain_len = random.randint(2, 4)
        post_stop = 0
    elif mode == "stop_immediate":
        chain_len = 1
        post_stop = random.randint(1, 2)
    else:
        chain_len = random.randint(2, 4)
        post_stop = random.randint(1, 2)

    total_pillars = left_extra + chain_len + post_stop
    if total_pillars < 3:
        raise ValueError("need at least three pillars for a clear scene")
    assert total_pillars <= 8

    chain_heights = _sample_chain_heights(chain_len)
    chain_gaps = _sample_chain_gaps(chain_heights)

    heights: list[int] = [random.randint(2, 8) for _ in range(left_extra)]
    heights.extend(chain_heights)
    heights.extend(random.randint(2, 8) for _ in range(post_stop))

    base_cols: list[int] = [random.randint(1, 3)]
    for _ in range(left_extra - 1):
        base_cols.append(base_cols[-1] + random.randint(2, 5))

    start_index = left_extra
    while len(base_cols) <= start_index:
        gap = random.randint(2, 5)
        base_cols.append(base_cols[-1] + gap)

    for gap in chain_gaps:
        base_cols.append(base_cols[-1] + gap)

    if post_stop:
        stop_gap = random.randint(chain_heights[-1] + 1, chain_heights[-1] + 5)
        base_cols.append(base_cols[-1] + stop_gap)
        for _ in range(post_stop - 1):
            base_cols.append(base_cols[-1] + random.randint(2, 5))

    assert len(base_cols) == total_pillars

    max_height = max(heights)
    width = max(col + h for col, h in zip(base_cols, heights)) + random.randint(1, 3)
    height = max_height + random.randint(0, 3)
    if height > 30 or width > 30:
        raise ValueError("scene too large")

    palette = random.sample(list(Color.NOT_BLACK), total_pillars + 1)
    pillar_colors = palette[:total_pillars]
    marker_color = palette[-1]

    grid = np.zeros((height, width), dtype=int)
    for base_col, pillar_height, color in zip(base_cols, heights, pillar_colors):
        grid[height - pillar_height : height, base_col] = color

    marked_height = heights[start_index]
    marker_row = height - marked_height
    marker_col = base_cols[start_index] + 1
    assert marker_col < width
    assert grid[marker_row, marker_col] == Color.BLACK
    grid[marker_row, marker_col] = marker_color

    metadata = {
        "mode": mode,
        "start_index": start_index,
        "chain_len": chain_len,
        "chain_gaps": tuple(chain_gaps),
    }
    return grid, metadata


def _gen_attempt() -> np.ndarray | None:
    try:
        grid, metadata = _build_canonical_scene()
    except ValueError:
        return None

    direction = random.choice([-1, 1])
    if direction == -1:
        grid = np.fliplr(grid)

    try:
        parsed_grid, pillars, start_index, parsed_direction = _parse_scene(grid)
    except AssertionError:
        return None

    if parsed_direction != direction:
        return None
    toppled = _chain_indices(pillars, start_index, parsed_direction)

    if metadata["mode"] == "full":
        if direction == 1 and start_index + len(toppled) != len(pillars):
            return None
        if direction == -1 and start_index - len(toppled) + 1 != 0:
            return None
    if metadata["mode"] != "full" and len(toppled) != metadata["chain_len"]:
        return None

    out = main(parsed_grid)
    if out.shape != parsed_grid.shape or np.array_equal(out, parsed_grid):
        return None
    return parsed_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-chain input")
