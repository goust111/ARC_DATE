from __future__ import annotations

from collections import deque
import itertools
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import Color, bbox, label_mask


BACKGROUND = Color.BLACK
WALL = Color.RED
OCCLUDER = Color.PINK
COLOR_A = Color.GREY
COLOR_B = Color.GREEN

SIZE_RANGE = (13, 21)
MAX_GENERATION_ATTEMPTS = 400


def _marker_rectangle(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = grid == OCCLUDER
    if not np.any(mask):
        return None
    rect = bbox(mask)
    assert rect is not None, "occluder must exist when magenta pixels are present"
    r0, c0, r1, c1 = rect
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "magenta pixels must form one solid rectangle"
    return r0, c0, r1, c1


def _restore_walls(grid: np.ndarray) -> np.ndarray:
    restored = np.asarray(grid, dtype=int).copy()
    rect = _marker_rectangle(restored)
    if rect is None:
        return restored

    h, w = restored.shape
    r0, c0, r1, c1 = rect
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            candidates = []
            for rr, cc in {
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            }:
                value = int(restored[rr, cc])
                if value != OCCLUDER:
                    candidates.append(value)
            assert candidates, "every occluded pixel must have at least one visible symmetric mate"
            assert len(set(candidates)) == 1, "visible symmetric mates must agree"
            restored[r, c] = candidates[0]
    return restored


def _region_graph(restored: np.ndarray) -> tuple[np.ndarray, dict[int, set[int]]]:
    labels, n_regions = label_mask(restored == BACKGROUND, connectivity=4)
    assert n_regions >= 2, "restored walls must split the background"

    h, w = restored.shape
    graph = {region: set() for region in range(1, n_regions + 1)}
    for r, c in np.argwhere(restored == WALL):
        r = int(r)
        c = int(c)
        if 0 < r < h - 1:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)
        if 0 < c < w - 1:
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)
    return labels, graph


def _alternate_color(restored: np.ndarray) -> np.ndarray:
    labels, graph = _region_graph(restored)
    h, _w = restored.shape
    start_region = int(labels[h - 1, 0])
    assert start_region > 0, "the lower-left corner must belong to a background region"

    parity = {start_region: 0}
    queue = deque([start_region])
    while queue:
        region = queue.popleft()
        for neighbor in graph[region]:
            want = 1 - parity[region]
            if neighbor in parity:
                assert parity[neighbor] == want, "region adjacency graph must be bipartite"
            else:
                parity[neighbor] = want
                queue.append(neighbor)

    assert len(parity) == len(graph), "all background regions must participate in the wall adjacency graph"

    out = restored.copy()
    for region, bit in parity.items():
        out[labels == region] = COLOR_A if bit == 0 else COLOR_B
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the symmetric wall topology hidden by the magenta rectangle, then alternately color the wall-separated regions.

    Input structure:
    - The grid contains black background, red one-cell-thick walls, and one solid magenta rectangle that occludes part of the wall layout.
    - Apart from the occluder, the wall pattern is symmetric under both horizontal and vertical reflection through the grid center.
    - The red walls partition the black background into multiple 4-connected regions.

    Rule:
    - Detect the solid magenta rectangle and restore each covered cell by reading a non-magenta value from its horizontal mirror, vertical mirror, or 180-degree rotational mate.
    - Keep the restored red wall pixels unchanged.
    - Label the black 4-connected regions and connect two region labels whenever they sit on opposite sides of the same red wall cell.
    - Starting from the region containing the lower-left corner, color regions grey and green in alternating layers across that bipartite adjacency graph.

    Key insight:
    - The magenta rectangle does not hide final colors; it hides the wall topology that defines the correct region graph, so symmetry restoration must happen before alternating recoloring.
    """
    restored = _restore_walls(np.asarray(grid, dtype=int))
    return _alternate_color(restored)


def _spaced_choices(values: list[int], max_count: int, min_gap: int) -> list[tuple[int, ...]]:
    choices: list[tuple[int, ...]] = []
    for count in range(1, min(max_count, len(values)) + 1):
        for combo in itertools.combinations(values, count):
            if all(combo[i + 1] - combo[i] >= min_gap for i in range(len(combo) - 1)):
                choices.append(tuple(int(v) for v in combo))
    return choices


def _choose_symmetric_lines(length: int) -> tuple[list[int], bool]:
    center = length // 2
    candidates = list(range(3, center - 1))
    choices = _spaced_choices(candidates, max_count=2, min_gap=3)
    assert choices, "grid size must permit at least one mirrored wall pair"
    mirrored = list(random.choice(choices))
    include_center = bool(length % 2 == 1 and random.random() < 0.35)
    return mirrored, include_center


def _full_line_positions(length: int, mirrored_half: list[int], include_center: bool) -> list[int]:
    positions = list(mirrored_half) + [length - 1 - pos for pos in mirrored_half]
    if include_center:
        positions.append(length // 2)
    return sorted(set(int(pos) for pos in positions))


def _build_restored_grid() -> np.ndarray:
    height = random.randrange(SIZE_RANGE[0], SIZE_RANGE[1] + 1, 2)
    width = random.randrange(SIZE_RANGE[0], SIZE_RANGE[1] + 1, 2)
    top_rows, include_center_row = _choose_symmetric_lines(height)
    left_cols, include_center_col = _choose_symmetric_lines(width)

    rows = _full_line_positions(height, top_rows, include_center_row)
    cols = _full_line_positions(width, left_cols, include_center_col)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for r in rows:
        grid[r, 1:-1] = WALL
    for c in cols:
        grid[1:-1, c] = WALL
    return grid


def _is_critical_wall_cell(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != WALL:
        return False
    h, w = grid.shape
    left_right = 0 < c < w - 1 and int(grid[r, c - 1]) == BACKGROUND and int(grid[r, c + 1]) == BACKGROUND
    up_down = 0 < r < h - 1 and int(grid[r - 1, c]) == BACKGROUND and int(grid[r + 1, c]) == BACKGROUND
    return left_right or up_down


def _count_background_regions(grid: np.ndarray) -> int:
    _labels, n_regions = label_mask(grid == BACKGROUND, connectivity=4)
    return int(n_regions)


def _candidate_rectangles(restored: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = restored.shape
    base_regions = _count_background_regions(restored)
    top_limit = h // 2
    left_limit = w // 2
    candidates: list[tuple[int, int, int, int]] = []

    for rect_h in range(3, 6):
        for rect_w in range(3, 6):
            for r0 in range(0, top_limit - rect_h + 1):
                for c0 in range(0, left_limit - rect_w + 1):
                    r1 = r0 + rect_h
                    c1 = c0 + rect_w
                    window = restored[r0:r1, c0:c1]
                    if not np.any(window == WALL):
                        continue
                    if not any(
                        _is_critical_wall_cell(restored, r, c)
                        for r in range(r0, r1)
                        for c in range(c0, c1)
                    ):
                        continue

                    occluded_as_bg = restored.copy()
                    occluded_as_bg[r0:r1, c0:c1] = BACKGROUND
                    if _count_background_regions(occluded_as_bg) >= base_regions:
                        continue
                    candidates.append((r0, c0, rect_h, rect_w))
    return candidates


def _map_rect_to_quadrant(rect: tuple[int, int, int, int], height: int, width: int, quadrant: int) -> tuple[int, int, int, int]:
    r0, c0, rect_h, rect_w = rect
    if quadrant in (2, 3):
        r0 = height - rect_h - r0
    if quadrant in (1, 3):
        c0 = width - rect_w - c0
    return int(r0), int(c0), int(rect_h), int(rect_w)


def _place_occluder(restored: np.ndarray) -> np.ndarray | None:
    candidates = _candidate_rectangles(restored)
    if not candidates:
        return None
    rect = random.choice(candidates)
    rect = _map_rect_to_quadrant(rect, restored.shape[0], restored.shape[1], random.randrange(4))

    inp = restored.copy()
    r0, c0, rect_h, rect_w = rect
    inp[r0 : r0 + rect_h, c0 : c0 + rect_w] = OCCLUDER
    return inp


def _valid_instance(inp: np.ndarray) -> bool:
    if inp is None:
        return False
    rect = _marker_rectangle(inp)
    if rect is None:
        return False

    restored = _restore_walls(inp)
    if int(restored[-1, 0]) != BACKGROUND:
        return False
    if not np.any(restored == WALL):
        return False
    if _count_background_regions(restored) < 2:
        return False

    try:
        out = main(inp)
    except AssertionError:
        return False

    if out.shape != inp.shape:
        return False
    if not np.array_equal(out[restored == WALL], np.full(np.count_nonzero(restored == WALL), WALL, dtype=int)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        restored = _build_restored_grid()
        inp = _place_occluder(restored)
        if _valid_instance(inp):
            return inp
    raise RuntimeError("failed to generate a valid symmetric-wall occlusion instance")
