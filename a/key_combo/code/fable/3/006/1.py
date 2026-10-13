from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = tuple(range(1, 10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    """Return 4-connected components of one color as coordinate arrays."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(grid == int(color)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []

        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for nr, nc in _neighbors4(r, c, h, w):
                if not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                    seen[nr, nc] = True
                    q.append((nr, nc))

        comps.append(np.array(cells, dtype=int))

    return comps


def _component_lists_by_color(grid: np.ndarray) -> dict[int, list[np.ndarray]]:
    return {
        int(color): _components_of_color(grid, int(color))
        for color in np.unique(grid)
        if int(color) != BACKGROUND
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each foreground color's 4-connected pieces and draw that count as the output diagonal layer.

    Input structure:
    - A black background contains several nonzero foreground colors as separated scattered objects.
    - For N foreground colors, their 4-connected component counts are exactly the numbers 1 through N.
    - No two foreground colors have the same component count; pixel totals and positions are distractors.

    Rule:
    - Count the 4-connected components of each foreground color.
    - Build an N by N output grid.
    - Cell (r, c) takes the color whose component count is max(r, c) + 1, so count 1 is the upper-left cell and count N is the outer L frame.

    Key insight:
    - The number of pieces of a color is the layer index; neither the input layout nor the amount of colored area determines the output order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert BACKGROUND in set(map(int, np.unique(g))), "expected black background"

    comps_by_color = _component_lists_by_color(g)
    assert comps_by_color, "expected at least one foreground color"

    count_to_color: dict[int, int] = {}
    for color, comps in comps_by_color.items():
        count = len(comps)
        assert count >= 1
        assert count not in count_to_color, "component counts must be unique"
        count_to_color[count] = int(color)

    n = len(comps_by_color)
    assert set(count_to_color) == set(range(1, n + 1)), "counts must be exactly 1..N"

    out = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            out[r, c] = count_to_color[max(r, c) + 1]
    return out


# --------------- generator ---------------


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    arr = np.array(sorted((r - min_r, c - min_c) for r, c in cells), dtype=int)
    return arr


def _transform_cells(cells: np.ndarray) -> np.ndarray:
    arr = np.array(cells, dtype=int)
    if random.random() < 0.5:
        arr = arr[:, [1, 0]]
    if random.random() < 0.5:
        arr[:, 0] = -arr[:, 0]
    if random.random() < 0.5:
        arr[:, 1] = -arr[:, 1]
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return np.array(sorted(map(tuple, arr)), dtype=int)


def _random_blob(area: int) -> np.ndarray:
    """Make one small 4-connected component with randomized orientation."""
    area = int(area)
    assert area >= 1
    cells: set[tuple[int, int]] = {(0, 0)}

    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                p = (r + dr, c + dc)
                if p not in cells:
                    frontier.append(p)
        cells.add(random.choice(frontier))

    return _transform_cells(_normalize_cells(cells))


def _area_targets(n: int) -> dict[int, int]:
    """Choose visible pixel totals that conflict with component-count ordering."""
    targets: dict[int, int] = {}
    for count in range(2, n + 1):
        targets[count] = count + (2 if count % 2 == 1 else 0)
    targets[1] = max(targets.values(), default=0) + random.randint(5, 9)
    return targets


def _split_component_areas(count: int, total: int) -> list[int]:
    sizes = [1] * int(count)
    extra = int(total) - int(count)
    while extra > 0:
        idx = random.randrange(len(sizes))
        add = random.randint(1, min(2, extra))
        sizes[idx] += add
        extra -= add
    random.shuffle(sizes)
    return sizes


def _patch_from_cells(cells: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    h = int(cells[:, 0].max()) + 1
    w = int(cells[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[cells[:, 0], cells[:, 1]] = True
    patch = np.full((h, w), BACKGROUND, dtype=int)
    patch[mask] = int(color)
    return patch, mask


def _can_place(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    ph, pw = mask.shape
    if r0 < 1 or c0 < 1 or r0 + ph >= h or c0 + pw >= w:
        return False

    for rr, cc in np.argwhere(mask):
        r = int(r0 + rr)
        c = int(c0 + cc)
        if int(grid[r, c]) != BACKGROUND:
            return False
        if np.any(grid[r - 1 : r + 2, c - 1 : c + 2] != BACKGROUND):
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _try_place_component(grid: np.ndarray, cells: np.ndarray, color: int) -> bool:
    patch, mask = _patch_from_cells(cells, color)
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return False

    for _ in range(250):
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        if _can_place(grid, mask, r0, c0):
            _place_patch(grid, patch, mask, r0, c0)
            return True
    return False


def _expected_from_counts(count_to_color: dict[int, int]) -> np.ndarray:
    n = len(count_to_color)
    out = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            out[r, c] = int(count_to_color[max(r, c) + 1])
    return out


def _semantic_checks(grid: np.ndarray, count_to_color: dict[int, int]) -> bool:
    comps_by_color = _component_lists_by_color(grid)
    n = len(count_to_color)
    if set(comps_by_color) != set(count_to_color.values()):
        return False

    counts = {color: len(comps) for color, comps in comps_by_color.items()}
    if set(counts.values()) != set(range(1, n + 1)):
        return False
    if any(counts[color] != count for count, color in count_to_color.items()):
        return False

    pixel_totals = {color: int(np.count_nonzero(grid == color)) for color in comps_by_color}
    count_order = [count_to_color[k] for k in range(1, n + 1)]
    total_ascending = sorted(pixel_totals, key=lambda color: (pixel_totals[color], color))
    total_descending = sorted(pixel_totals, key=lambda color: (-pixel_totals[color], color))
    if total_ascending == count_order or total_descending == count_order:
        return False

    expected = _expected_from_counts(count_to_color)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    # At least one multi-component color must include a non-singleton piece, so
    # "count pixels and divide" is not a reliable shortcut.
    return any(
        count > 1 and any(len(comp) > 1 for comp in comps_by_color[color])
        for count, color in count_to_color.items()
    )


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(3, 6)
    min_side = max(13, 4 * n)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    colors = random.sample(FOREGROUND_COLORS, n)
    random.shuffle(colors)
    count_to_color = {count: int(colors[count - 1]) for count in range(1, n + 1)}
    targets = _area_targets(n)

    records: list[tuple[int, int, np.ndarray]] = []
    for count in range(1, n + 1):
        color = count_to_color[count]
        for area in _split_component_areas(count, targets[count]):
            records.append((int(area), int(color), _random_blob(area)))

    records.sort(key=lambda item: item[0], reverse=True)
    for _area, color, cells in records:
        if not _try_place_component(grid, cells, color):
            return None

    if not _semantic_checks(grid, count_to_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid count-as-coordinate input")
