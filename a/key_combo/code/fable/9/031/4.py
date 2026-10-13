from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
SIDE_OPTIONS = (2, 3, 4, 5, 6)
DOT_COUNT_OPTIONS = (1, 2, 3, 4, 5, 6, 7)
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class ColorSpec:
    color: int
    side: int
    dot_count: int


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _solid_square_side(comp: list[tuple[int, int]]) -> int | None:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or len(comp) != h * w or h < 2:
        return None
    return h


def _parse_specs(grid: np.ndarray) -> list[ColorSpec]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND)
    specs: list[ColorSpec] = []
    for color in colors:
        comps = _components(g == color)
        square_side: int | None = None
        dot_count = 0
        for comp in comps:
            if len(comp) == 1:
                dot_count += 1
                continue
            side = _solid_square_side(comp)
            assert side is not None, f"color {color} must contain exactly one solid square plus isolated dots"
            assert square_side is None, f"color {color} has multiple non-dot components"
            square_side = side
        if square_side is None:
            continue
        specs.append(ColorSpec(color=color, side=square_side, dot_count=dot_count))

    assert len(specs) >= 2, "expected at least two colored squares"
    sides = [spec.side for spec in specs]
    assert len(set(sides)) == len(sides), "square side lengths must be unique"
    valid = [spec for spec in specs if spec.dot_count > 0]
    assert len(valid) >= 2, "expected at least two colors with both a square and dots"
    counts = [spec.dot_count for spec in valid]
    assert len(set(counts)) == len(counts), "dot counts must be unique among active colors"
    return specs


def _build_output(ordered_outer_to_inner: list[ColorSpec]) -> np.ndarray:
    inner = ordered_outer_to_inner[-1]
    canvas = np.full((inner.side, inner.side), inner.color, dtype=int)
    for spec in reversed(ordered_outer_to_inner[:-1]):
        new_side = canvas.shape[0] + 2 * spec.side
        wrapped = np.full((new_side, new_side), spec.color, dtype=int)
        wrapped[spec.side : spec.side + canvas.shape[0], spec.side : spec.side + canvas.shape[1]] = canvas
        canvas = wrapped
    return canvas


def _expected_from_specs(specs: list[ColorSpec]) -> np.ndarray:
    valid = [spec for spec in specs if spec.dot_count > 0]
    ordered = sorted(valid, key=lambda spec: (-spec.dot_count, spec.color))
    return _build_output(ordered)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw nested concentric square layers whose order comes from dot counts and whose thickness comes from square sizes.

    Input structure:
    - The background is black.
    - Several distinct colors each contain one solid square, and all square side lengths are different.
    - Some of those colors also have isolated single-pixel dots elsewhere; some colors may have only the square and no dots.

    Rule:
    - Keep only colors that have both a solid square and at least one dot.
    - Sort those kept colors by dot count from most to fewest: the most dots form the outermost frame and the fewest dots become the innermost solid square.
    - Use each color's square side length as that layer's thickness; the innermost layer is a solid square whose side length equals its source square side length.
    - Output the smallest square grid that exactly fits the full nesting.

    Key insight:
    - Layer order and layer thickness come from different visual channels: dots decide outside-to-inside order, while square size only decides how thick that color's layer is.
    """
    specs = _parse_specs(np.asarray(grid, dtype=int))
    valid = [spec for spec in specs if spec.dot_count > 0]
    ordered = sorted(valid, key=lambda spec: (-spec.dot_count, spec.color))
    return _build_output(ordered)


def _render_input(specs: list[ColorSpec], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    def clear_with_margin(r0: int, c0: int, h: int, w: int) -> bool:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(height, r0 + h + 1)
        cc1 = min(width, c0 + w + 1)
        return not bool(occupied[rr0:rr1, cc0:cc1].any())

    square_order = sorted(specs, key=lambda spec: (-spec.side, random.random()))
    for spec in square_order:
        placed = False
        row_choices = list(range(1, height - spec.side))
        col_choices = list(range(1, width - spec.side))
        random.shuffle(row_choices)
        random.shuffle(col_choices)
        for r0 in row_choices[: min(len(row_choices), 80)]:
            for c0 in col_choices[: min(len(col_choices), 80)]:
                if not clear_with_margin(r0, c0, spec.side, spec.side):
                    continue
                grid[r0 : r0 + spec.side, c0 : c0 + spec.side] = spec.color
                occupied[r0 : r0 + spec.side, c0 : c0 + spec.side] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    dot_colors = [spec.color for spec in specs for _ in range(spec.dot_count)]
    random.shuffle(dot_colors)
    free_cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(free_cells)

    for color in dot_colors:
        placed = False
        for r, c in free_cells:
            if occupied[r - 1 : r + 2, c - 1 : c + 2].any():
                continue
            grid[r, c] = color
            occupied[r, c] = True
            placed = True
            placed_r, placed_c = r, c
            break
        if not placed:
            return None
        free_cells = [
            (rr, cc)
            for rr, cc in free_cells
            if abs(rr - placed_r) > 1 or abs(cc - placed_c) > 1
        ]
    return grid


def _alternative_outputs(valid: list[ColorSpec]) -> list[np.ndarray]:
    return [
        _build_output(sorted(valid, key=lambda spec: (-spec.side, spec.color))),
        _build_output(sorted(valid, key=lambda spec: (spec.side, spec.color))),
    ]


def _choose_specs() -> list[ColorSpec] | None:
    n_valid = random.choice((2, 3, 3, 4, 4))
    max_decor = min(5 - n_valid, 2)
    decor_choices = [0]
    decor_weights = [2]
    if max_decor >= 1:
        decor_choices.append(1)
        decor_weights.append(4)
    if max_decor >= 2:
        decor_choices.append(2)
        decor_weights.append(1)
    n_decor = random.choices(decor_choices, weights=decor_weights, k=1)[0]

    colors = random.sample(PALETTE, n_valid + n_decor)
    sides = random.sample(SIDE_OPTIONS, n_valid + n_decor)
    dot_counts = random.sample(DOT_COUNT_OPTIONS, n_valid)

    valid = [ColorSpec(color=colors[i], side=sides[i], dot_count=dot_counts[i]) for i in range(n_valid)]
    decor = [
        ColorSpec(color=colors[n_valid + i], side=sides[n_valid + i], dot_count=0)
        for i in range(n_decor)
    ]

    expected = _expected_from_specs(valid + decor)
    if expected.shape[0] > 30:
        return None

    size_desc = [spec.color for spec in sorted(valid, key=lambda spec: (-spec.side, spec.color))]
    size_asc = [spec.color for spec in sorted(valid, key=lambda spec: (spec.side, spec.color))]
    count_desc = [spec.color for spec in sorted(valid, key=lambda spec: (-spec.dot_count, spec.color))]
    if count_desc == size_desc or count_desc == size_asc:
        return None

    for alt in _alternative_outputs(valid):
        if np.array_equal(expected, alt):
            return None

    return valid + decor


def _validate_instance(grid: np.ndarray, specs: list[ColorSpec]) -> bool:
    try:
        parsed = _parse_specs(grid)
    except AssertionError:
        return False

    parsed_map = {spec.color: (spec.side, spec.dot_count) for spec in parsed}
    target_map = {spec.color: (spec.side, spec.dot_count) for spec in specs}
    if parsed_map != target_map:
        return False

    valid = [spec for spec in specs if spec.dot_count > 0]
    expected = _expected_from_specs(specs)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    for alt in _alternative_outputs(valid):
        if np.array_equal(expected, alt):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _choose_specs()
    if specs is None:
        return None

    nonzero_cells = sum(spec.side * spec.side + spec.dot_count for spec in specs)
    min_side = max(16, int(np.ceil(np.sqrt(nonzero_cells * 5))))
    min_side = min(min_side, 26)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)

    grid = _render_input(specs, height, width)
    if grid is None:
        return None
    if not _validate_instance(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-square-counting input")
