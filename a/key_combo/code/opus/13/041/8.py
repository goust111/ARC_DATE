from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
GRAY = 5
OBJECT_COLORS = (1, 2, 3, 4)
LEGEND_SHAPE = (2, 2)
GEN_ATTEMPTS = 800


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _legend_pairs(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    legend = np.asarray(grid[:2, :2], dtype=int)
    assert legend.shape == LEGEND_SHAPE
    colors = set(map(int, legend.ravel()))
    assert len(colors) == 4, "legend must contain four distinct colors"
    assert BACKGROUND not in colors and GRAY not in colors, "legend colors must be foreground colors"
    return (int(legend[0, 0]), int(legend[0, 1])), (int(legend[1, 0]), int(legend[1, 1]))


def _object_mask(grid: np.ndarray) -> np.ndarray:
    mask = np.asarray(grid != BACKGROUND, dtype=bool)
    mask[:2, :2] = False
    return mask


def _swap_object_colors(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    (a, b), (c, d) = _legend_pairs(g)
    mask = _object_mask(g)
    assert mask.any(), "expected a non-legend foreground object"
    out = g.copy()
    out[(g == a) & mask] = b
    out[(g == b) & mask] = a
    out[(g == c) & mask] = d
    out[(g == d) & mask] = c
    return out


def _unique_winner(counts: Counter[int]) -> int:
    assert counts, "expected at least one counted color"
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "expected a unique most frequent color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Swap legend-row color pairs, then keep only the new frequency champion and gray out the rest.

    Input structure:
    - The grid has black background and a 2x2 color legend in the upper-left corner.
    - The legend rows define two bidirectional swap pairs, and every non-legend
      foreground pixel belongs to one 4-connected multicolor object using those colors.

    Rule:
    - Leave the legend and background unchanged.
    - Simultaneously swap the first legend-row colors with each other and the
      second legend-row colors with each other, but only on non-legend foreground pixels.
    - Count the swapped object colors, keep the unique most frequent swapped color,
      and replace all other object pixels with gray.

    Key insight:
    - The correct survivor is determined only after applying the legend-driven
      color-name swap, because the swap changes which color label owns the largest population.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    (a, b), (c, d) = _legend_pairs(g)
    mask = _object_mask(g)
    legend_colors = {a, b, c, d}
    assert mask.any(), "expected a foreground object outside the legend"
    assert set(map(int, g[mask])) <= legend_colors, "object may use only legend colors"

    swapped = _swap_object_colors(g)
    counts = Counter(map(int, swapped[mask]))
    winner = _unique_winner(counts)

    out = g.copy()
    out[mask] = GRAY
    out[mask & (swapped == winner)] = winner
    return out


def _sample_bbox(rng: random.Random, height: int, width: int) -> tuple[int, int, int, int] | None:
    top_choices = list(range(2, height - 4))
    left_choices = list(range(4, width - 4))
    if not top_choices or not left_choices:
        return None
    top = rng.choice(top_choices)
    left = rng.choice(left_choices)
    max_h = min(8, height - top - 1)
    max_w = min(8, width - left - 1)
    if max_h < 5 or max_w < 5:
        return None
    box_h = rng.randint(5, max_h)
    box_w = rng.randint(5, max_w)
    return top, left, box_h, box_w


def _grow_blob(rng: random.Random, box_h: int, box_w: int, size: int) -> np.ndarray | None:
    area = box_h * box_w
    if not (0 < size <= area):
        return None

    start = (rng.randrange(box_h), rng.randrange(box_w))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < box_h and 0 <= nc < box_w:
            frontier.add((nr, nc))

    while len(cells) < size and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != size:
        return None

    coords = np.array(sorted(cells), dtype=int)
    rs, cs = coords[:, 0], coords[:, 1]
    span_h = int(rs.max() - rs.min() + 1)
    span_w = int(cs.max() - cs.min() + 1)
    density = size / float(span_h * span_w)
    if span_h < 4 or span_w < 4:
        return None
    if not (0.45 <= density <= 0.82):
        return None
    if span_h * span_w == size:
        return None
    return coords


def _sample_counts(rng: random.Random, total: int) -> dict[int, int] | None:
    for _ in range(200):
        a_count = rng.randint(8, min(15, total - 9))
        remaining = total - a_count
        b_count = rng.randint(3, min(8, remaining - 6))
        remaining -= b_count
        c_count = rng.randint(3, min(9, remaining - 3))
        d_count = remaining - c_count
        if d_count < 3 or d_count > 9:
            continue
        if a_count <= max(b_count, c_count, d_count):
            continue
        if c_count == d_count:
            continue
        return {"A": a_count, "B": b_count, "C": c_count, "D": d_count}
    return None


def _color_blob(
    rng: random.Random,
    coords: np.ndarray,
    quotas: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], int]]:
    alpha, beta = rng.choice(
        [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (-1, 2)]
    )
    scored = []
    for r, c in coords:
        noise = rng.random() * 0.35
        scored.append((alpha * int(r) + beta * int(c) + noise, (int(r), int(c))))
    scored.sort(key=lambda item: item[0])

    segment_order = list(range(len(quotas)))
    rng.shuffle(segment_order)
    per_segment: list[list[tuple[int, int]]] = [[] for _ in quotas]
    cursor = 0
    for seg_idx in segment_order:
        _, quota = quotas[seg_idx]
        per_segment[seg_idx] = [cell for _, cell in scored[cursor : cursor + quota]]
        cursor += quota

    colored: list[tuple[tuple[int, int], int]] = []
    for seg_idx, cells in enumerate(per_segment):
        color, _ = quotas[seg_idx]
        for cell in cells:
            colored.append((cell, color))
    return colored


def _wrong_preswap_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mask = _object_mask(g)
    counts = Counter(map(int, g[mask]))
    winner = _unique_winner(counts)
    out = g.copy()
    out[mask] = GRAY
    out[mask & (g == winner)] = winner
    return out


def _build_expected_from_swapped(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mask = _object_mask(g)
    swapped = _swap_object_colors(g)
    winner = _unique_winner(Counter(map(int, swapped[mask])))
    out = g.copy()
    out[mask] = GRAY
    out[mask & (swapped == winner)] = winner
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30 or min(g.shape) < 2:
        return False
    if np.any(g == GRAY):
        return False

    try:
        (a, b), (c, d) = _legend_pairs(g)
    except AssertionError:
        return False

    if set(map(int, g[:2, :2].ravel())) != set(OBJECT_COLORS):
        return False

    mask = _object_mask(g)
    if not mask.any():
        return False
    if len(_components(mask)) != 1:
        return False

    obj_colors = set(map(int, g[mask]))
    if obj_colors != {a, b, c, d}:
        return False

    counts_before = Counter(map(int, g[mask]))
    if counts_before[a] <= counts_before[b]:
        return False
    if counts_before[a] <= max(counts_before[c], counts_before[d]):
        return False
    if counts_before[c] == counts_before[d]:
        return False

    swapped = _swap_object_colors(g)
    counts_after = Counter(map(int, swapped[mask]))
    if _unique_winner(counts_after) != b:
        return False
    if counts_after[b] != counts_before[a]:
        return False
    if counts_after[a] != counts_before[b]:
        return False
    if counts_after[c] != counts_before[d] or counts_after[d] != counts_before[c]:
        return False

    expected = _build_expected_from_swapped(g)
    wrong = _wrong_preswap_output(g)
    if np.array_equal(expected, wrong):
        return False

    try:
        actual = main(g)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(11, 16)
    width = rng.randint(11, 16)
    bbox = _sample_bbox(rng, height, width)
    if bbox is None:
        return None
    top, left, box_h, box_w = bbox

    area = box_h * box_w
    total_min = max(20, int(np.ceil(area * 0.48)))
    total_max = min(40, int(np.floor(area * 0.76)))
    if total_min > total_max:
        return None
    total = rng.randint(total_min, total_max)

    counts = _sample_counts(rng, total)
    if counts is None:
        return None

    coords = _grow_blob(rng, box_h, box_w, total)
    if coords is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    legend_colors = rng.sample(list(OBJECT_COLORS), 4)
    a, b, c, d = legend_colors
    grid[:2, :2] = np.array([[a, b], [c, d]], dtype=int)

    color_plan = [(a, counts["A"]), (b, counts["B"]), (c, counts["C"]), (d, counts["D"])]
    for (rr, cc), color in _color_blob(rng, coords, color_plan):
        grid[top + rr, left + cc] = color

    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
