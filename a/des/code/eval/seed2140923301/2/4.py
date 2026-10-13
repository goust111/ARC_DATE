from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    height = int(grid.shape[0])
    for r in range(1, height - 1):
        row_color = int(grid[r, 0])
        if row_color == bg:
            continue
        if np.all(grid[r, :] == row_color) and np.all(grid[r + 1 :, :] == bg):
            candidates.append(r)
    assert len(candidates) == 1, "expected one full-width divider above the blank canvas"
    return int(candidates[0])


def _extract_templates(top: np.ndarray, bg: int) -> list[dict[str, int]]:
    templates: list[dict[str, int]] = []
    height, width = map(int, top.shape)
    for r in range(height):
        c = 0
        while c < width:
            if int(top[r, c]) == bg:
                c += 1
                continue

            start = c
            while c < width and int(top[r, c]) != bg:
                c += 1
            block = top[r, start:c]
            assert block.size >= 2, "each template needs a label pixel and a segment"

            tag = int(block[0])
            segment = int(block[1])
            assert tag != segment, "label color must differ from segment color"
            assert np.all(block[1:] == segment), "segment must be a single-color horizontal run"

            templates.append(
                {
                    "length": int(block.size - 1),
                    "tag": tag,
                    "segment": segment,
                }
            )

    assert templates, "expected at least one template bar"
    segment_colors = [spec["segment"] for spec in templates]
    lengths = [spec["length"] for spec in templates]
    assert len(segment_colors) == len(set(segment_colors)), "segment colors must be distinct"
    assert len(lengths) == len(set(lengths)), "segment lengths must be distinct"
    return templates


def _square_origin(shape: tuple[int, int], side: int) -> tuple[int, int]:
    height, width = map(int, shape)
    assert 1 <= side <= min(height, width), "square must fit in the lower panel"
    return (height - side) // 2, (width - side) // 2


def _paint_nested(panel: np.ndarray, ordered_templates: list[dict[str, int]], bg: int) -> np.ndarray:
    out = panel.copy()
    for idx, spec in enumerate(ordered_templates):
        side = int(spec["length"])
        color = int(spec["tag"])
        r0, c0 = _square_origin(out.shape, side)
        r1 = r0 + side - 1
        c1 = c0 + side - 1
        if idx == len(ordered_templates) - 1:
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            continue
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

    unchanged = panel == bg
    assert np.all(out[~unchanged] == panel[~unchanged]), "generator expects a blank lower panel"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build centered nested square rings in the lower panel using the upper template bars.

    The segment lengths determine the outer-to-inner order, while the left label
    pixels determine the colors. The shortest template produces a solid square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    divider = _find_separator_row(g, bg)
    templates = _extract_templates(g[:divider, :], bg)
    ordered = sorted(templates, key=lambda spec: int(spec["length"]), reverse=True)

    out = g.copy()
    out[divider + 1 :, :] = _paint_nested(out[divider + 1 :, :], ordered, bg)
    return out


def _sample_lengths(count: int) -> list[int]:
    parity = random.choice((0, 1))
    candidates = [side for side in range(1, 16) if side % 2 == parity and side >= 1 + (parity == 0)]
    return sorted(random.sample(candidates, count), reverse=True)


def _build_expected(inp: np.ndarray, divider: int, templates: list[dict[str, int]]) -> np.ndarray:
    out = inp.copy()
    ordered = sorted(templates, key=lambda spec: int(spec["length"]), reverse=True)
    out[divider + 1 :, :] = _paint_nested(out[divider + 1 :, :], ordered, BG)
    return out


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    lengths = _sample_lengths(count)
    largest = int(lengths[0])

    width = largest + 2 * random.randint(1, 5)
    bottom_height = largest + 2 * random.randint(1, 4)
    top_height = random.randint(2 * count + 1, 2 * count + 5)
    row_pool = list(range(0, top_height, 2))
    if len(row_pool) < count:
        return None

    colors = PALETTE[:]
    random.shuffle(colors)
    segment_colors = colors[:count]
    tag_colors = colors[count : 2 * count]
    divider = top_height

    templates = [
        {"length": int(length), "segment": int(segment), "tag": int(tag)}
        for length, segment, tag in zip(lengths, segment_colors, tag_colors)
    ]
    random.shuffle(templates)

    grid = np.full((top_height + 1 + bottom_height, width), BG, dtype=int)
    grid[divider, :] = SEPARATOR

    for row, spec in zip(random.sample(row_pool, count), templates):
        total = int(spec["length"]) + 1
        start = random.randint(0, width - total)
        grid[row, start] = int(spec["tag"])
        grid[row, start + 1 : start + total] = int(spec["segment"])

    expected = _build_expected(grid, divider, templates)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
