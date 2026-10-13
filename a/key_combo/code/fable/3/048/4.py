from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 500


LayerSpec = tuple[int, bool]  # (original strip index, horizontally mirrored)


def _background_color(g: np.ndarray, sep_cols: list[int]) -> int:
    keep = np.ones(g.shape[1], dtype=bool)
    keep[sep_cols] = False
    vals = [int(x) for x in g[:, keep].ravel()]
    assert vals, "expected non-separator cells"
    counts = Counter(vals).most_common()
    bg, bg_count = counts[0]
    assert bg != GREY, "gray is reserved for separator lines"
    if len(counts) > 1:
        assert bg_count > counts[1][1], "background color must be uniquely dominant"
    return int(bg)


def _extract_strips(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must not exceed 30 in either dimension"

    sep_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(sep_cols) in (1, 3), "expected one or three full-height gray separator columns"

    bg = _background_color(g, sep_cols)
    bounds = [-1] + sep_cols + [w]
    widths = [bounds[i + 1] - bounds[i] - 1 for i in range(len(bounds) - 1)]
    assert len(widths) in (2, 4), "separator count must make two or four strips"
    assert min(widths) > 0 and len(set(widths)) == 1, "all strips must have the same width"

    strips = []
    for i in range(len(widths)):
        strip = g[:, bounds[i] + 1 : bounds[i + 1]].copy()
        assert not np.any(strip == GREY), "gray may appear only in separator columns"
        strips.append(strip)
    return strips, bg


def _fold_layer_specs(n: int) -> list[LayerSpec]:
    piles: list[list[LayerSpec]] = [[(i, False)] for i in range(n)]
    while len(piles) > 1:
        half = len(piles) // 2
        folded: list[list[LayerSpec]] = [list(pile) for pile in piles[:half]]
        for offset, pile in enumerate(piles[half:]):
            target = half - 1 - offset
            folded[target].extend((idx, not mirrored) for idx, mirrored in reversed(pile))
        piles = folded
    return piles[0]


def _layers_from_specs(strips: list[np.ndarray], specs: list[LayerSpec]) -> list[np.ndarray]:
    layers = []
    for idx, mirrored in specs:
        layer = strips[idx]
        layers.append(np.fliplr(layer).copy() if mirrored else layer.copy())
    return layers


def _fold_strips(strips: list[np.ndarray]) -> list[np.ndarray]:
    return _layers_from_specs(strips, _fold_layer_specs(len(strips)))


def _transparent_overlay(layers: list[np.ndarray], bg: int) -> np.ndarray:
    assert layers, "expected at least one layer"
    out = np.full(layers[0].shape, int(bg), dtype=int)
    for layer in layers:
        assert layer.shape == out.shape, "all folded layers must align"
        mask = layer != int(bg)
        out[mask] = layer[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold a separator-divided paper strip repeatedly leftward and view the final stack from above.

    Input structure:
    - A background grid is split by full-height gray vertical separator columns.
    - The separators divide the paper into exactly two or four equal-width strips.
    - Gray appears only as separators; each strip contains background and colored foreground pixels.

    Rule:
    - Repeatedly fold the right half of the current paper stack left across the middle separator.
    - Every folded half is mirrored horizontally, and its internal layer order is reversed before it lands on top.
    - Remove all separators and output one strip-width view, taking the topmost non-background color at each cell.

    Key insight:
    - Four strips do not collapse by one simultaneous mirror overlay: the second fold flips the already stacked half, giving final layer order strip 1, strip 4 mirrored, strip 3, strip 2 mirrored from bottom to top.
    """
    g = np.asarray(grid, dtype=int)
    strips, bg = _extract_strips(g)
    layers = _fold_strips(strips)
    return _transparent_overlay(layers, bg)


# --------------- generator ---------------


def _source_col(strip_w: int, final_c: int, mirrored: bool) -> int:
    return strip_w - 1 - final_c if mirrored else final_c


def _assemble(strips: list[np.ndarray]) -> np.ndarray:
    h, strip_w = strips[0].shape
    n = len(strips)
    out = np.full((h, n * strip_w + n - 1), BLACK, dtype=int)
    c = 0
    for i, strip in enumerate(strips):
        out[:, c : c + strip_w] = strip
        c += strip_w
        if i != n - 1:
            out[:, c] = GREY
            c += 1
    return out


def _pick_coords(h: int, w: int, count: int) -> list[tuple[int, int]]:
    center = {w // 2} if w % 2 == 1 else set()
    pool = [(r, c) for r in range(h) for c in range(w) if c not in center]
    random.shuffle(pool)
    for gap in (3, 2, 1, 0):
        chosen: list[tuple[int, int]] = []
        for r, c in pool:
            if all(max(abs(r - rr), abs(c - cc)) >= gap for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return pool[:count]


def _paint_layers_at_final(
    strips: list[np.ndarray],
    specs: list[LayerSpec],
    coord: tuple[int, int],
    layer_colors: list[tuple[int, int]],
) -> None:
    r, c = coord
    strip_w = strips[0].shape[1]
    for idx, mirrored in specs:
        strips[idx][r, _source_col(strip_w, c, mirrored)] = BLACK
    for layer_idx, color in layer_colors:
        idx, mirrored = specs[layer_idx]
        strips[idx][r, _source_col(strip_w, c, mirrored)] = int(color)


def _two_different_colors() -> tuple[int, int]:
    return tuple(random.sample(PALETTE, 2))  # type: ignore[return-value]


def _force_two_strip_clues(strips: list[np.ndarray], specs: list[LayerSpec]) -> None:
    h, w = strips[0].shape
    coords = _pick_coords(h, w, 4)

    lower, upper = _two_different_colors()
    _paint_layers_at_final(strips, specs, coords[0], [(0, lower), (1, upper)])
    _paint_layers_at_final(strips, specs, coords[1], [(0, random.choice(PALETTE))])
    _paint_layers_at_final(strips, specs, coords[2], [(1, random.choice(PALETTE))])

    # A second top-only mark at a different final column makes the right-strip mirror visible.
    _paint_layers_at_final(strips, specs, coords[3], [(1, random.choice(PALETTE))])


def _force_four_strip_clues(strips: list[np.ndarray], specs: list[LayerSpec]) -> None:
    h, w = strips[0].shape
    coords = _pick_coords(h, w, 7)

    low, high = _two_different_colors()
    _paint_layers_at_final(strips, specs, coords[0], [(2, low), (3, high)])

    low, high = _two_different_colors()
    _paint_layers_at_final(strips, specs, coords[1], [(1, low), (2, high)])

    low, high = _two_different_colors()
    _paint_layers_at_final(strips, specs, coords[2], [(0, low), (1, high)])

    for layer_idx, coord in zip(range(4), coords[3:]):
        _paint_layers_at_final(strips, specs, coord, [(layer_idx, random.choice(PALETTE))])


PATTERNS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
)


def _place_random_pattern(strip: np.ndarray, color: int) -> int:
    h, w = strip.shape
    for _ in range(40):
        pattern = random.choice(PATTERNS)
        ph = max(r for r, _c in pattern) + 1
        pw = max(c for _r, c in pattern) + 1
        if ph > h or pw > w:
            continue
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        cells = [(r0 + r, c0 + c) for r, c in pattern]
        if all(strip[r, c] == BLACK for r, c in cells):
            for r, c in cells:
                strip[r, c] = int(color)
            return len(cells)
    return 0


def _add_random_marks(strips: list[np.ndarray]) -> None:
    for strip in strips:
        area = int(strip.size)
        target = random.randint(max(3, area // 12), max(4, area // 5))
        placed = 0
        attempts = 0
        while placed < target and attempts < 100:
            attempts += 1
            placed += _place_random_pattern(strip, random.choice(PALETTE))


def _owner_map(layers: list[np.ndarray], bg: int) -> np.ndarray:
    owner = np.full(layers[0].shape, -1, dtype=int)
    for idx, layer in enumerate(layers):
        owner[layer != bg] = idx
    return owner


def _has_different_overlap(layers: list[np.ndarray], lower: int, upper: int, bg: int) -> bool:
    mask = (layers[lower] != bg) & (layers[upper] != bg)
    return bool(np.any(mask & (layers[lower] != layers[upper])))


def _render_specs(strips: list[np.ndarray], specs: list[LayerSpec], bg: int) -> np.ndarray:
    return _transparent_overlay(_layers_from_specs(strips, specs), bg)


def _semantic_checks(strips: list[np.ndarray], specs: list[LayerSpec]) -> bool:
    layers = _layers_from_specs(strips, specs)
    out = _transparent_overlay(layers, BLACK)
    owner = _owner_map(layers, BLACK)
    present = {int(x) for x in owner.ravel() if int(x) >= 0}

    if present != set(range(len(layers))):
        return False
    if len(set(map(int, out.ravel())) - {BLACK}) < min(3, len(layers)):
        return False

    if len(layers) == 2:
        if not _has_different_overlap(layers, 0, 1, BLACK):
            return False
        no_mirror = _render_specs(strips, [(0, False), (1, False)], BLACK)
        return not np.array_equal(out, no_mirror)

    if len(layers) == 4:
        needed = ((2, 3), (1, 2), (0, 1))
        if not all(_has_different_overlap(layers, lo, hi, BLACK) for lo, hi in needed):
            return False
        naive_order = [(0, False), (1, True), (2, False), (3, True)]
        wrong_stack = [(0, False), (3, True), (1, True), (2, False)]
        return (
            not np.array_equal(out, _render_specs(strips, naive_order, BLACK))
            and not np.array_equal(out, _render_specs(strips, wrong_stack, BLACK))
        )

    return False


def _gen_attempt() -> np.ndarray | None:
    n = 4 if random.random() < 0.55 else 2
    h = random.randint(8, 16)
    strip_w = random.randint(4, 6) if n == 4 else random.randint(4, 12)
    if n * strip_w + n - 1 > 30:
        return None

    strips = [np.full((h, strip_w), BLACK, dtype=int) for _ in range(n)]
    _add_random_marks(strips)

    specs = _fold_layer_specs(n)
    if n == 2:
        _force_two_strip_clues(strips, specs)
    else:
        _force_four_strip_clues(strips, specs)

    if not _semantic_checks(strips, specs):
        return None

    grid = _assemble(strips)
    expected = _transparent_overlay(_fold_strips(strips), BLACK)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid repeated-fold strip input")
