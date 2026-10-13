from __future__ import annotations

import random

import numpy as np


BG = 0
LABEL_COLOR_POOL = tuple(range(1, 10))
CONTAINER_COUNTS = (3, 4, 5)
BODY_SIZES = (3, 4)
PALETTE_BLOCK_WIDTHS = (1, 2)
PALETTE_GAPS = (1, 2)
TOP_MARGINS = (0, 1, 2)
BOTTOM_MARGINS = (0, 1, 2)
BETWEEN_GAPS = (1, 2, 3)
SIDE_EXTRA_SPACE = range(0, 7)
MAX_GENERATION_ATTEMPTS = 300


PATTERNS_3 = (
    np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
)

PATTERNS_4 = (
    np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
)


def _nonzero_runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    c = 0
    w = int(row.shape[0])
    while c < w:
        if int(row[c]) == BG:
            c += 1
            continue
        color = int(row[c])
        start = c
        while c + 1 < w and int(row[c + 1]) == color:
            c += 1
        runs.append((start, c, color))
        c += 1
    return runs


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    nonempty_rows = np.flatnonzero(np.any(g != BG, axis=1))
    assert nonempty_rows.size >= 2, "expected containers plus a palette row"
    palette_row = int(nonempty_rows[-1])

    palette_runs = _nonzero_runs(g[palette_row])
    assert 3 <= len(palette_runs) <= 5, "expected 3 to 5 palette blocks"
    palette_order = [color for _, _, color in palette_runs]
    assert len(set(palette_order)) == len(palette_order), "palette colors must be unique"

    above = nonempty_rows[nonempty_rows < palette_row]
    assert above.size > 0, "expected container rows above the palette"
    band_bottom = int(above[-1])
    band_top = band_bottom
    while band_top - 1 >= 0 and np.any(g[band_top - 1] != BG):
        band_top -= 1
    assert band_top < band_bottom, "container band must include label row and body"

    label_runs = _nonzero_runs(g[band_top])
    assert len(label_runs) == len(palette_order), "label count must match palette count"
    widths = {end - start + 1 for start, end, _ in label_runs}
    assert len(widths) == 1, "containers must share the same width"

    containers = []
    labels_seen = set()
    for start, end, label_color in label_runs:
        width = end - start + 1
        block = g[band_top : band_bottom + 1, start : end + 1]
        assert np.all(block[0] == label_color), "label row must be a solid color"
        body_mask = block[1:] != BG
        assert body_mask.shape == (band_bottom - band_top, width)
        assert np.any(body_mask), "container body must contain a visible pattern"
        assert np.all(body_mask.any(axis=1)), "every body row must contain at least one pattern cell"
        labels_seen.add(int(label_color))
        containers.append(
            {
                "col0": int(start),
                "col1": int(end),
                "label": int(label_color),
                "body_mask": body_mask.copy(),
            }
        )

    assert labels_seen == set(palette_order), "palette colors must match container labels"
    return {
        "palette_row": palette_row,
        "palette_order": palette_order,
        "band_top": band_top,
        "band_bottom": band_bottom,
        "containers": containers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder labeled containers by the visible palette order, then recolor each
    body pattern with the next container's label color in the new cyclic order.
    """
    g = np.asarray(grid, dtype=int)
    info = _parse_scene(g)
    order = info["palette_order"]
    by_label = {container["label"]: container for container in info["containers"]}
    band_top = int(info["band_top"])
    band_bottom = int(info["band_bottom"])

    out = g.copy()
    out[band_top : band_bottom + 1, :] = BG

    for index, slot in enumerate(info["containers"]):
        label = int(order[index])
        next_label = int(order[(index + 1) % len(order)])
        col0 = int(slot["col0"])
        col1 = int(slot["col1"])
        src = by_label[label]
        body_mask = src["body_mask"]

        out[band_top, col0 : col1 + 1] = label
        out[band_top + 1 : band_bottom + 1, col0 : col1 + 1] = BG
        out[band_top + 1 : band_bottom + 1, col0 : col1 + 1][body_mask] = next_label

    return out


def _pattern_library(size: int) -> tuple[np.ndarray, ...]:
    if size == 3:
        return PATTERNS_3
    if size == 4:
        return PATTERNS_4
    raise ValueError(f"unsupported body size: {size}")


def _sample_distinct_permutations(items: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    first = items[:]
    second = items[:]
    rng.shuffle(first)
    while True:
        rng.shuffle(second)
        if second != first:
            return first, second


def _choose_pattern_colors(label_colors: list[int], count: int, rng: random.Random) -> list[int]:
    preferred = [color for color in LABEL_COLOR_POOL if color not in label_colors]
    if not preferred:
        preferred = list(LABEL_COLOR_POOL)
    return [int(rng.choice(preferred)) for _ in range(count)]


def _make_grid(
    input_order: list[int],
    palette_order: list[int],
    body_size: int,
    pattern_masks: list[np.ndarray],
    pattern_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    n = len(input_order)
    container_w = body_size
    container_h = body_size + 1
    container_gap = 1

    palette_block_w = int(rng.choice(tuple(sorted(set(PALETTE_BLOCK_WIDTHS + (container_w,))))))
    palette_gap = int(rng.choice(PALETTE_GAPS))
    top_margin = int(rng.choice(TOP_MARGINS))
    between_gap = int(rng.choice(BETWEEN_GAPS))
    bottom_margin = int(rng.choice(BOTTOM_MARGINS))

    container_band_w = n * container_w + (n - 1) * container_gap
    palette_w = n * palette_block_w + (n - 1) * palette_gap
    width = max(container_band_w, palette_w) + int(rng.choice(tuple(SIDE_EXTRA_SPACE))) + 2
    container_left = rng.randint(0, width - container_band_w)
    palette_left = rng.randint(0, width - palette_w)

    palette_row = top_margin + container_h + between_gap
    height = palette_row + 1 + bottom_margin
    grid = np.zeros((height, width), dtype=int)

    mask_by_label = {label: mask for label, mask in zip(input_order, pattern_masks)}
    color_by_label = {label: color for label, color in zip(input_order, pattern_colors)}

    for index, label in enumerate(input_order):
        col0 = container_left + index * (container_w + container_gap)
        col1 = col0 + container_w
        grid[top_margin, col0:col1] = int(label)
        body = grid[top_margin + 1 : top_margin + 1 + body_size, col0:col1]
        body[mask_by_label[label]] = int(color_by_label[label])

    for index, color in enumerate(palette_order):
        col0 = palette_left + index * (palette_block_w + palette_gap)
        grid[palette_row, col0 : col0 + palette_block_w] = int(color)

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray:
    n = int(rng.choice(CONTAINER_COUNTS))
    body_size = int(rng.choice(BODY_SIZES))
    label_colors = list(rng.sample(LABEL_COLOR_POOL, n))
    input_order, palette_order = _sample_distinct_permutations(label_colors, rng)
    patterns = [rng.choice(_pattern_library(body_size)).copy() for _ in range(n)]
    pattern_colors = _choose_pattern_colors(label_colors, n, rng)

    grid = _make_grid(input_order, palette_order, body_size, patterns, pattern_colors, rng)
    info = _parse_scene(grid)
    assert info["palette_order"] == palette_order
    assert [container["label"] for container in info["containers"]] == input_order
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
