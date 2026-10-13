from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


CoordSet = set[tuple[int, int]]


def _active_row_spans(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    active = np.any(g != bg, axis=1)
    spans: list[tuple[int, int]] = []
    r = 0
    h = g.shape[0]
    while r < h:
        if not active[r]:
            r += 1
            continue
        r0 = r
        while r < h and active[r]:
            r += 1
        spans.append((r0, r - 1))
    return spans


def _coords(mask: np.ndarray) -> CoordSet:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each right-side colored shape into the left slot whose template color is its exact complement.

    Input structure:
    - The grid has two equal-width slots separated by one background column.
    - Non-background rows form horizontal bands separated by all-background rows.
    - Each band has a shared template color in the left slot and one colored source shape in the right slot.

    Rule:
    - For each band, define its destination shape as the cells in the left slot not occupied by the template color.
    - Normalize every right-side source shape to right-slot coordinates and match it to the unique equal destination shape.
    - Clear the original right-side source pixels and paint the matched left-slot complement with the source color.

    Key insight:
    The right-side objects are not placed by row order; each one completes the unique left template whose missing cells have the same shape.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    assert w % 2 == 1, "canvas must contain two equal slots plus one separator column"
    slot_w = w // 2
    sep_c = slot_w

    bands = _active_row_spans(g, bg)
    assert bands, "grid must contain at least one active band"

    band_color_sets: list[set[int]] = []
    for r0, r1 in bands:
        colors = set(map(int, g[r0 : r1 + 1].ravel())) - {bg}
        assert colors, "each active band must contain foreground colors"
        band_color_sets.append(colors)

    shared_colors = set.intersection(*band_color_sets)
    assert len(shared_colors) == 1, "template color must be the only color shared by all bands"
    template_color = next(iter(shared_colors))

    dest_shapes: list[CoordSet] = []
    sources: list[tuple[int, CoordSet]] = []
    for band_i, (r0, r1) in enumerate(bands):
        block = g[r0 : r1 + 1]
        assert np.all(block[:, sep_c] == bg), "slot separator column must be background"

        left = block[:, :slot_w]
        right = block[:, sep_c + 1 :]
        assert np.all((left == bg) | (left == template_color)), "left slot must contain only template and background"

        template_mask = left == template_color
        for row in template_mask:
            cols = np.flatnonzero(row)
            assert cols.size > 0, "each template row must contain at least one pixel"
            expected = np.arange(int(cols[-1]) + 1)
            assert np.array_equal(cols, expected), "template pixels must be a left-anchored row prefix"

        dest_shapes.append(_coords(~template_mask))

        source_mask = right != bg
        assert np.any(source_mask), f"band {band_i} source shape must be non-empty"
        source_colors = set(map(int, right[source_mask].ravel()))
        assert len(source_colors) == 1, "each right slot must contain exactly one source color"
        source_color = next(iter(source_colors))
        assert source_color != template_color, "source color must differ from template color"
        source_shape = _coords(source_mask)
        sources.append((source_color, source_shape))

    out = g.copy()
    for r0, r1 in bands:
        right = out[r0 : r1 + 1, sep_c + 1 :]
        right[right != bg] = bg

    used_destinations: set[int] = set()
    for source_color, source_shape in sources:
        matches = [i for i, dest_shape in enumerate(dest_shapes) if dest_shape == source_shape]
        assert len(matches) == 1, "each source shape must match exactly one left-slot complement"
        dest_i = matches[0]
        assert dest_i not in used_destinations, "each destination complement must be filled once"
        used_destinations.add(dest_i)

        dest_r0, _ = bands[dest_i]
        for dr, dc in source_shape:
            rr = dest_r0 + dr
            assert out[rr, dc] == bg, "matched complement cells must be empty before painting"
            out[rr, dc] = source_color

    assert len(used_destinations) == len(bands), "all destination complements must be filled"
    assert out.shape == (h, w)
    return out

# --------------- generator (o5x) ---------------

# Fixed task colors.
BG = 7
TEMPLATE = 8

# Canvas and band geometry.
SLOT_W = 5
BAND_H = 3
BAND_GAP = 1
OUTER_MARGIN_ROWS = 1
SEPARATOR_W = 1
WIDTH = SLOT_W * 2 + SEPARATOR_W
RIGHT_SLOT_C0 = SLOT_W + SEPARATOR_W

# Sampling ranges and palette.
MIN_BANDS, MAX_BANDS = 2, 4
MIN_SUFFIX_LEN, MAX_SUFFIX_LEN = 1, SLOT_W - 1
SUFFIX_LENGTHS = range(MIN_SUFFIX_LEN, MAX_SUFFIX_LEN + 1)
SOURCE_COLORS = [1, 2, 3, 4, 5, 6, 9]
COMPLEMENTS = [(a, b, c) for a in SUFFIX_LENGTHS for b in SUFFIX_LENGTHS for c in SUFFIX_LENGTHS]
MAX_ATTEMPTS = 300


def _band_top(i: int) -> int:
    return OUTER_MARGIN_ROWS + i * (BAND_H + BAND_GAP)


def _sample_layout() -> tuple[int, list[tuple[int, int, int]], list[int], list[int]]:
    """Sample distinct band complements, source colors, and source-to-destination mapping."""
    num_bands = random.randint(MIN_BANDS, MAX_BANDS)
    comps = random.sample(COMPLEMENTS, num_bands)
    source_colors = random.sample(SOURCE_COLORS, num_bands)
    perm = list(range(num_bands))
    random.shuffle(perm)
    return num_bands, comps, source_colors, perm


def _paint_template(grid: np.ndarray, r0: int, comp: tuple[int, int, int]) -> None:
    for dr, suffix_len in enumerate(comp):
        grid[r0 + dr, : SLOT_W - suffix_len] = TEMPLATE


def _paint_suffix(grid: np.ndarray, r0: int, c0: int, comp: tuple[int, int, int], color: int) -> None:
    for dr, suffix_len in enumerate(comp):
        grid[r0 + dr, c0 + SLOT_W - suffix_len : c0 + SLOT_W] = color


def _build_output(rows: int, comps: list[tuple[int, int, int]], source_colors: list[int], perm: list[int]) -> np.ndarray:
    """Render the solved grid: templates plus matched left complements."""
    out = np.full((rows, WIDTH), BG, dtype=int)
    for dest_i, comp in enumerate(comps):
        _paint_template(out, _band_top(dest_i), comp)
    for src_i, dest_i in enumerate(perm):
        _paint_suffix(out, _band_top(dest_i), 0, comps[dest_i], source_colors[src_i])
    return out


def _build_input(out: np.ndarray, comps: list[tuple[int, int, int]], source_colors: list[int], perm: list[int]) -> np.ndarray:
    """Invert the solver transform by clearing left complements and placing right sources."""
    inp = out.copy()
    for dest_i, comp in enumerate(comps):
        _paint_suffix(inp, _band_top(dest_i), 0, comp, BG)
    for src_i, dest_i in enumerate(perm):
        _paint_suffix(inp, _band_top(src_i), RIGHT_SLOT_C0, comps[dest_i], source_colors[src_i])
    return inp


def _gen_attempt() -> np.ndarray | None:
    num_bands, comps, source_colors, perm = _sample_layout()
    rows = 2 * OUTER_MARGIN_ROWS + num_bands * BAND_H + (num_bands - 1) * BAND_GAP
    out = _build_output(rows, comps, source_colors, perm)
    inp = _build_input(out, comps, source_colors, perm)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
