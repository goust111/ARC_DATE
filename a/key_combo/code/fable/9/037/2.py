from __future__ import annotations

from collections import deque
import hashlib
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    NOT_BLACK = (1, 2, 3, 4, 5, 6, 7, 8, 9)


GEN_RETRIES = 500
GRID_ROWS = (10, 18)
GRID_COLS = (10, 18)
SEEN_OUTPUTS: set[str] = set()
GEN_CALL_COUNT = 0


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _extract_walls(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    verticals = []
    horizontals = []
    for c in range(w):
        color = int(g[0, c])
        if color != Color.BLACK and int(g[h - 1, c]) == color:
            verticals.append((c, color))
    for r in range(h):
        color = int(g[r, 0])
        if color != Color.BLACK and int(g[r, w - 1]) == color:
            horizontals.append((r, color))
    return verticals, horizontals


def _paint_walls(shape: tuple[int, int], verticals, horizontals) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    wall_mask = np.zeros((h, w), dtype=bool)
    wall_colors = np.zeros((h, w), dtype=int)
    for c, color in verticals:
        wall_mask[:, c] = True
        wall_colors[:, c] = int(color)
    for r, color in horizontals:
        wall_mask[r, :] = True
        wall_colors[r, :] = int(color)
    return wall_mask, wall_colors


def _preserved_border_mask(grid: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    preserve = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if _is_border(g.shape, r, c) and g[r, c] != Color.BLACK and not wall_mask[r, c]:
                preserve[r, c] = True
    return preserve


def _expected_output_from_labels(
    grid: np.ndarray,
    wall_mask: np.ndarray,
    wall_colors: np.ndarray,
    regions: list[np.ndarray],
    labels: list[int | None],
) -> np.ndarray:
    out = np.zeros_like(grid)
    preserve = _preserved_border_mask(grid, wall_mask)
    out[preserve] = grid[preserve]
    out[wall_mask] = wall_colors[wall_mask]
    for region, label in zip(regions, labels):
        if label is None:
            continue
        for r, c in region:
            rr = int(r)
            cc = int(c)
            if not preserve[rr, cc]:
                out[rr, cc] = int(label)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Generate full wall lines from aligned same-color opposite-edge markers, then flood-fill each walled compartment by its payload color.

    Input structure:
    - The grid has a black background with scattered colored border markers on the outer edge.
    - Some border markers form valid opposite-edge pairs: top with bottom in the same column, or left with right in the same row, with matching colors.
    - The interior contains isolated single-pixel payloads; unpaired border markers are distractors and remain visible.

    Rule:
    - For every top/bottom matched pair, draw a full vertical wall in that column using the shared color; for every left/right matched pair, draw a full horizontal wall in that row.
    - Only opposite-edge, aligned, same-color pairs create walls; every other border marker stays exactly where it is and creates nothing.
    - Treat the drawn walls as 4-connected barriers. Any non-wall region containing interior payload pixels is filled solid with that payload color, while regions without payloads stay black; walls keep their wall colors.

    Key insight:
    - The decisive structure is hidden in the border endpoints: they imply invisible full-length walls that partition the canvas, and the interior payloads only name the compartments they already occupy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    verticals, horizontals = _extract_walls(g)
    assert verticals or horizontals, "expected at least one valid opposite-edge wall pair"

    wall_mask, wall_colors = _paint_walls(g.shape, verticals, horizontals)
    preserve = _preserved_border_mask(g, wall_mask)
    regions = components_mask(~wall_mask)

    out = np.zeros_like(g)
    out[preserve] = g[preserve]
    out[wall_mask] = wall_colors[wall_mask]

    h, w = g.shape
    for region in regions:
        payload_colors = {
            int(g[r, c])
            for r, c in map(tuple, region)
            if 0 < int(r) < h - 1 and 0 < int(c) < w - 1 and int(g[r, c]) != Color.BLACK
        }
        assert len(payload_colors) <= 1, "each compartment may contain at most one payload color"
        if not payload_colors:
            continue
        fill = next(iter(payload_colors))
        for r, c in region:
            rr = int(r)
            cc = int(c)
            if not preserve[rr, cc]:
                out[rr, cc] = fill

    return out


def _sample_positions(length: int, count: int, low: int, high: int, min_gap: int) -> list[int] | None:
    if high - low + 1 < count:
        return None
    for _ in range(200):
        pos = sorted(random.sample(range(low, high + 1), count))
        if all(b - a >= min_gap for a, b in zip(pos, pos[1:])):
            return pos
    return None


def _mode_for_call() -> str:
    global GEN_CALL_COUNT
    mode = ("vertical", "horizontal", "both")[GEN_CALL_COUNT % 3]
    GEN_CALL_COUNT += 1
    return mode


def _build_layout(mode: str) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    grid = np.zeros((h, w), dtype=int)

    if mode == "vertical":
        count = random.choice((2, 3))
        cols = _sample_positions(w, count, low=2, high=w - 3, min_gap=3)
        if cols is None:
            raise ValueError("failed to sample vertical walls")
        colors = random.sample(Color.NOT_BLACK, count)
        verticals = list(zip(cols, colors))
        horizontals: list[tuple[int, int]] = []
    elif mode == "horizontal":
        count = random.choice((2, 3))
        rows = _sample_positions(h, count, low=2, high=h - 3, min_gap=3)
        if rows is None:
            raise ValueError("failed to sample horizontal walls")
        colors = random.sample(Color.NOT_BLACK, count)
        horizontals = list(zip(rows, colors))
        verticals = []
    else:
        c = random.randint(2, w - 3)
        r = random.randint(2, h - 3)
        color = random.choice(Color.NOT_BLACK)
        verticals = [(c, color)]
        horizontals = [(r, color)]

    for c, color in verticals:
        grid[0, c] = color
        grid[h - 1, c] = color
    for r, color in horizontals:
        grid[r, 0] = color
        grid[r, w - 1] = color

    return grid, verticals, horizontals


def _place_distractors(
    grid: np.ndarray,
    verticals: list[tuple[int, int]],
    horizontals: list[tuple[int, int]],
) -> list[tuple[str, int]]:
    h, w = grid.shape
    used = {(int(r), int(c)) for r, c in np.argwhere(grid != Color.BLACK)}
    active_v = {c for c, _ in verticals}
    active_h = {r for r, _ in horizontals}
    wall_colors = [color for _idx, color in verticals + horizontals]
    free_colors = [c for c in Color.NOT_BLACK if c not in wall_colors]
    opposite_color = random.choice(wall_colors or list(Color.NOT_BLACK))
    adjacent_color = random.choice(free_colors or list(Color.NOT_BLACK))
    distractor_types: list[tuple[str, int]] = []

    def try_put(cells: list[tuple[int, int]], color: int) -> bool:
        if len(set(cells)) != len(cells):
            return False
        if any(cell in used for cell in cells):
            return False
        for r, c in cells:
            grid[r, c] = color
            used.add((r, c))
        return True

    for _ in range(80):
        if random.random() < 0.5:
            c0, c1 = random.sample([c for c in range(1, w - 1) if c not in active_v], 2)
            if try_put([(0, c0), (h - 1, c1)], opposite_color):
                distractor_types.append(("opposite_misaligned", opposite_color))
                break
        else:
            r0, r1 = random.sample([r for r in range(1, h - 1) if r not in active_h], 2)
            if try_put([(r0, 0), (r1, w - 1)], opposite_color):
                distractor_types.append(("opposite_misaligned", opposite_color))
                break
    else:
        raise ValueError("failed to place opposite-edge distractor")

    for _ in range(80):
        flavor = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
        if flavor == "top_left":
            c = random.randint(1, w - 2)
            r = random.randint(1, h - 2)
            cells = [(0, c), (r, 0)]
        elif flavor == "top_right":
            c = random.randint(1, w - 2)
            r = random.randint(1, h - 2)
            cells = [(0, c), (r, w - 1)]
        elif flavor == "bottom_left":
            c = random.randint(1, w - 2)
            r = random.randint(1, h - 2)
            cells = [(h - 1, c), (r, 0)]
        else:
            c = random.randint(1, w - 2)
            r = random.randint(1, h - 2)
            cells = [(h - 1, c), (r, w - 1)]
        if try_put(cells, adjacent_color):
            distractor_types.append(("adjacent_same_color", adjacent_color))
            break
    else:
        raise ValueError("failed to place adjacent-edge distractor")

    if random.random() < 0.7:
        singleton_color = random.choice(free_colors or list(Color.NOT_BLACK))
        border_pool = [
            (0, c) for c in range(1, w - 1)
        ] + [
            (h - 1, c) for c in range(1, w - 1)
        ] + [
            (r, 0) for r in range(1, h - 1)
        ] + [
            (r, w - 1) for r in range(1, h - 1)
        ]
        random.shuffle(border_pool)
        for cell in border_pool:
            if cell not in used:
                grid[cell] = singleton_color
                distractor_types.append(("singleton", singleton_color))
                break

    return distractor_types


def _ordered_regions_for_mode(
    regions: list[np.ndarray],
    mode: str,
) -> list[np.ndarray]:
    if mode == "vertical":
        return sorted(regions, key=lambda rc: int(rc[:, 1].min()))
    if mode == "horizontal":
        return sorted(regions, key=lambda rc: int(rc[:, 0].min()))
    return sorted(regions, key=lambda rc: (int(rc[:, 0].min()), int(rc[:, 1].min())))


def _region_labels(mode: str, region_count: int, blocked_colors: set[int]) -> list[int | None]:
    payload_colors = [c for c in Color.NOT_BLACK if c not in blocked_colors]
    random.shuffle(payload_colors)
    if mode == "both":
        assert region_count == 4
        a, b, c = payload_colors[:3]
        labels = [a, None, b, c]
        random.shuffle(labels)
        return labels
    if region_count == 3:
        a, b = payload_colors[:2]
        return random.choice([[a, None, b], [None, a, b], [a, b, None]])
    if region_count == 4:
        a, b, c = payload_colors[:3]
        return random.choice([[a, None, b, c], [a, b, None, c], [None, a, b, c], [a, None, b, None]])
    raise ValueError("unexpected region count")


def _choose_payload_cells(grid: np.ndarray, region: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    cells = [tuple(map(int, rc)) for rc in region]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for r, c in cells:
        if not (0 < r < h - 1 and 0 < c < w - 1):
            continue
        visible_neighbor = False
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != Color.BLACK:
                visible_neighbor = True
                break
        if visible_neighbor:
            continue
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == count:
            return chosen
    return None


def _place_payloads(
    grid: np.ndarray,
    regions: list[np.ndarray],
    labels: list[int | None],
) -> None:
    filled = [i for i, label in enumerate(labels) if label is not None]
    assert filled and len(filled) < len(labels)
    bonus_region = random.choice(filled) if random.random() < 0.45 else None
    for i in filled:
        desired = 2 if i == bonus_region else 1
        coords = _choose_payload_cells(grid, regions[i], desired)
        if coords is None:
            raise ValueError("failed to place payloads")
        for r, c in coords:
            grid[r, c] = int(labels[i])


def _validate_semantics(
    grid: np.ndarray,
    mode: str,
    intended_verticals: list[tuple[int, int]],
    intended_horizontals: list[tuple[int, int]],
    distractor_types: list[tuple[str, int]],
    regions: list[np.ndarray],
    labels: list[int | None],
) -> np.ndarray | None:
    actual_verticals, actual_horizontals = _extract_walls(grid)
    if actual_verticals != intended_verticals or actual_horizontals != intended_horizontals:
        return None
    if not any(kind == "opposite_misaligned" for kind, _ in distractor_types):
        return None
    if not any(kind == "adjacent_same_color" for kind, _ in distractor_types):
        return None

    wall_mask, wall_colors = _paint_walls(grid.shape, intended_verticals, intended_horizontals)
    actual_regions = _ordered_regions_for_mode(components_mask(~wall_mask), mode)
    if len(actual_regions) != len(regions):
        return None

    expected = _expected_output_from_labels(grid, wall_mask, wall_colors, actual_regions, labels)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    filled_labels = [label for label in labels if label is not None]
    if not filled_labels or len(filled_labels) == len(labels):
        return None

    region_targets = []
    h, w = grid.shape
    for region in actual_regions:
        payload_colors = {
            int(grid[r, c])
            for r, c in map(tuple, region)
            if 0 < int(r) < h - 1 and 0 < int(c) < w - 1 and int(grid[r, c]) != Color.BLACK
        }
        if len(payload_colors) > 1:
            return None
        region_targets.append(next(iter(payload_colors)) if payload_colors else None)

    if region_targets != labels:
        return None

    region_index = np.full(grid.shape, -1, dtype=int)
    for idx, region in enumerate(actual_regions):
        region_index[region[:, 0], region[:, 1]] = idx

    for c, _color in intended_verticals:
        crossing_pairs = set()
        for r in range(grid.shape[0]):
            if c - 1 < 0 or c + 1 >= grid.shape[1]:
                continue
            left_idx = int(region_index[r, c - 1])
            right_idx = int(region_index[r, c + 1])
            if left_idx != -1 and right_idx != -1:
                crossing_pairs.add((left_idx, right_idx))
        if not crossing_pairs:
            return None
        for left_idx, right_idx in crossing_pairs:
            if labels[left_idx] == labels[right_idx]:
                return None
    for r, _color in intended_horizontals:
        crossing_pairs = set()
        for c in range(grid.shape[1]):
            if r - 1 < 0 or r + 1 >= grid.shape[0]:
                continue
            top_idx = int(region_index[r - 1, c])
            bottom_idx = int(region_index[r + 1, c])
            if top_idx != -1 and bottom_idx != -1:
                crossing_pairs.add((top_idx, bottom_idx))
        if not crossing_pairs:
            return None
        for top_idx, bottom_idx in crossing_pairs:
            if labels[top_idx] == labels[bottom_idx]:
                return None

    if np.array_equal(actual, grid):
        return None
    return actual


def _output_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _gen_attempt() -> np.ndarray | None:
    mode = _mode_for_call()
    try:
        grid, verticals, horizontals = _build_layout(mode)
        distractor_types = _place_distractors(grid, verticals, horizontals)
    except ValueError:
        return None

    wall_mask, _wall_colors = _paint_walls(grid.shape, verticals, horizontals)
    regions = _ordered_regions_for_mode(components_mask(~wall_mask), mode)
    blocked_colors = {color for _idx, color in verticals + horizontals}
    labels = _region_labels(mode, len(regions), blocked_colors)

    try:
        _place_payloads(grid, regions, labels)
    except (AssertionError, ValueError):
        return None

    actual = _validate_semantics(grid, mode, verticals, horizontals, distractor_types, regions, labels)
    if actual is None:
        return None

    sig = _output_signature(actual)
    if sig in SEEN_OUTPUTS:
        return None
    SEEN_OUTPUTS.add(sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid boundary-wall input")
