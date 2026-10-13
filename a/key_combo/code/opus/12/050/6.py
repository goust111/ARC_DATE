from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
TEAL = 8

BAND_COLORS = (BLUE, RED, GREEN)
BAND_COLOR_PALETTES = (
    (BLUE, RED, GREEN),
    (BLUE, RED, TEAL),
    (BLUE, GREEN, TEAL),
    (RED, GREEN, TEAL),
    (BLUE, 6, TEAL),
    (RED, 6, 9),
    (GREEN, TEAL, 9),
)
BAND_COUNT = 3
BAND_HEIGHT = 4
GRID_WIDTH = 12
GEN_ATTEMPTS = 400


def _overlay_bands(bands: list[np.ndarray]) -> np.ndarray:
    """Overlay same-size bands with fixed priority band0 > band1 > band2."""
    out = np.zeros_like(bands[0])
    for band in reversed(bands):
        mask = band != BLACK
        out[mask] = band[mask]
    return out


def _pair_keys(marker_coords: Iterable[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[str, int, int, int]]:
    pairs: set[tuple[str, int, int, int]] = set()
    rows, cols = shape
    coords = {(int(r), int(c)) for r, c in marker_coords}
    for r in range(rows):
        row_cols = sorted(c for rr, c in coords if rr == r)
        if len(row_cols) == 2:
            pairs.add(("row", int(r), int(row_cols[0]), int(row_cols[1])))
    for c in range(cols):
        col_rows = sorted(r for r, cc in coords if cc == c)
        if len(col_rows) == 2:
            pairs.add(("col", int(c), int(col_rows[0]), int(col_rows[1])))
    return pairs


def _draw_connections(canvas: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    rows, cols = canvas.shape

    for r in range(rows):
        orange_cols = np.flatnonzero(canvas[r] == ORANGE)
        if len(orange_cols) != 2:
            continue
        c0, c1 = map(int, orange_cols)
        for c in range(c0 + 1, c1):
            if canvas[r, c] == BLACK:
                out[r, c] = YELLOW

    for c in range(cols):
        orange_rows = np.flatnonzero(canvas[:, c] == ORANGE)
        if len(orange_rows) != 2:
            continue
        r0, r1 = map(int, orange_rows)
        for r in range(r0 + 1, r1):
            if canvas[r, c] == BLACK:
                out[r, c] = YELLOW

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse three stacked mask bands, keep only the overlay-visible orange markers, and connect their aligned pairs.

    Input structure:
    - The grid is split into three equal-height horizontal bands that share one local coordinate system.
    - Each band is a sparse mask on black background, using its own foreground color plus optional orange marker pixels.
    - Higher bands can place non-black pixels on top of lower-band markers at the same local position.

    Rule:
    - Overlay the three bands into a single canvas with fixed priority first band > second band > third band.
    - On the composite canvas, keep only the orange pixels that remain visible after overlay as marker endpoints.
    - For every row or column containing exactly two visible orange markers, fill the black cells strictly between them with yellow; leave non-black intervening cells unchanged.

    Key insight:
    - Alignment is evaluated only after overlay, so hidden orange markers vanish from the topology and can remove or create candidate connections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % BAND_COUNT == 0, "expected three equal-height bands"

    band_height = height // BAND_COUNT
    assert band_height > 0 and width > 0
    bands = [g[i * band_height : (i + 1) * band_height, :] for i in range(BAND_COUNT)]

    composite = _overlay_bands(bands)
    return _draw_connections(composite)


def _stack_bands(bands: list[np.ndarray]) -> np.ndarray:
    return np.vstack(bands).astype(int)


def _quiet_candidates(
    forbidden: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    used_by_higher: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells = []
    for r in range(BAND_HEIGHT):
        for c in range(GRID_WIDTH):
            pos = (r, c)
            if pos in forbidden or pos in blocked or pos in used_by_higher:
                continue
            cells.append(pos)
    random.shuffle(cells)
    return cells


def _build_template_two_pairs() -> dict:
    c0 = random.randint(0, 3)
    c1 = random.randint(c0 + 3, 8)
    remaining = [c for c in range(GRID_WIDTH) if c not in {c0, c1}]
    cs = random.choice([c for c in remaining if c not in range(c0 + 1, c1)])
    cd = random.choice([c for c in remaining if c != cs and c not in {c0, c1}])

    visible = {(0, c0), (0, c1), (3, c1), (2, cs)}
    hidden = {(3, c0), (1, cd)}
    obstacle = (1, c1)
    visible_pairs = {
        ("row", 0, c0, c1),
        ("col", c1, 0, 3),
    }
    return {
        "visible_oranges": visible,
        "hidden_oranges": hidden,
        "required_obstacles": {obstacle},
        "visible_pairs": visible_pairs,
        "band_oranges": {
            0: {(0, c0), (0, c1)},
            1: {(3, c1), (1, cd)},
            2: {(2, cs), (3, c0)},
        },
        "band_bases": {
            0: {obstacle, (3, c0), (1, cd)},
            1: set(),
            2: set(),
        },
    }


def _build_template_three_pairs() -> dict:
    c0 = random.randint(0, 2)
    c1 = random.randint(c0 + 3, 6)
    middle = list(range(c0 + 1, c1))
    obstacle = (0, random.choice(middle))

    pool = [c for c in range(GRID_WIDTH) if c not in {c0, c1}]
    c2 = random.choice(pool)
    pool.remove(c2)
    c3_choices = [c for c in pool if abs(c - c2) >= 3]
    c3 = random.choice(c3_choices)
    pool = [c for c in pool if c != c3]
    c4 = random.choice([c for c in pool if c not in {c2, c3}])

    visible = {(0, c0), (0, c1), (2, c2), (2, c3), (1, c4), (3, c4)}
    hidden = {(3, c2)}
    visible_pairs = {
        ("row", 0, c0, c1),
        ("row", 2, min(c2, c3), max(c2, c3)),
        ("col", c4, 1, 3),
    }
    return {
        "visible_oranges": visible,
        "hidden_oranges": hidden,
        "required_obstacles": {obstacle},
        "visible_pairs": visible_pairs,
        "band_oranges": {
            0: {(0, c0), (0, c1)},
            1: {(2, c2), (2, c3)},
            2: {(1, c4), (3, c4), (3, c2)},
        },
        "band_bases": {
            0: {obstacle},
            1: {(3, c2)},
            2: set(),
        },
    }


def _segment_interiors(pair: tuple[str, int, int, int]) -> set[tuple[int, int]]:
    kind, axis, a0, a1 = pair
    cells: set[tuple[int, int]] = set()
    if kind == "row":
        r = axis
        for c in range(a0 + 1, a1):
            cells.add((int(r), int(c)))
    else:
        c = axis
        for r in range(a0 + 1, a1):
            cells.add((int(r), int(c)))
    return cells


def _realize_bands(spec: dict, band_colors: tuple[int, int, int]) -> list[np.ndarray] | None:
    bands = [np.zeros((BAND_HEIGHT, GRID_WIDTH), dtype=int) for _ in range(BAND_COUNT)]
    blocked = set()
    for pair in spec["visible_pairs"]:
        blocked |= _segment_interiors(pair)
    allowed_obstacles = set(spec["required_obstacles"])
    blocked -= allowed_obstacles

    for band_idx in range(BAND_COUNT):
        band = bands[band_idx]
        for r, c in spec["band_oranges"][band_idx]:
            band[r, c] = ORANGE
        for r, c in spec["band_bases"][band_idx]:
            band[r, c] = band_colors[band_idx]

    # Add filler base-color pixels so every band has a varied but sparse footprint.
    higher_used = set()
    for band_idx in range(BAND_COUNT):
        band = bands[band_idx]
        count = int(np.count_nonzero(band))
        if count > 8:
            return None
        target = random.randint(max(3, count), 8)
        need = target - count
        candidates = _quiet_candidates(
            forbidden=set(spec["visible_oranges"]) | set(spec["hidden_oranges"]),
            blocked=blocked,
            used_by_higher=higher_used,
        )
        if len(candidates) < need:
            return None
        for r, c in candidates[:need]:
            band[r, c] = band_colors[band_idx]
        higher_used |= {(int(r), int(c)) for r, c in np.argwhere(band != BLACK)}

    # Final range checks: every band uses only black, its band color, and orange.
    for band_idx, band in enumerate(bands):
        palette = set(map(int, np.unique(band)))
        if not palette <= {BLACK, band_colors[band_idx], ORANGE}:
            return None
        orange_count = int(np.count_nonzero(band == ORANGE))
        fg_count = int(np.count_nonzero(band))
        if not (2 <= orange_count <= 4):
            return None
        if not (3 <= fg_count <= 8):
            return None
    return bands


def _validate_semantics(inp: np.ndarray, spec: dict) -> bool:
    band_h = inp.shape[0] // BAND_COUNT
    bands = [inp[i * band_h : (i + 1) * band_h] for i in range(BAND_COUNT)]
    composite = _overlay_bands(bands)
    out = main(inp)

    visible_oranges = {(int(r), int(c)) for r, c in np.argwhere(composite == ORANGE)}
    all_oranges = {
        (int(r), int(c))
        for band in bands
        for r, c in np.argwhere(band == ORANGE)
    }
    hidden_oranges = all_oranges - visible_oranges
    visible_pairs = _pair_keys(visible_oranges, composite.shape)
    all_pairs = _pair_keys(all_oranges, composite.shape)
    expected = _draw_connections(composite)

    if visible_oranges != set(spec["visible_oranges"]):
        return False
    if not set(spec["hidden_oranges"]) <= hidden_oranges:
        return False
    if visible_pairs != set(spec["visible_pairs"]):
        return False
    if len(visible_pairs) not in {2, 3}:
        return False
    if not (all_pairs - visible_pairs):
        return False
    if not np.array_equal(out, expected):
        return False
    if int(np.count_nonzero(out == YELLOW)) == 0:
        return False

    obstacle_ok = False
    for r, c in spec["required_obstacles"]:
        if composite[r, c] != BLACK:
            obstacle_ok = True
            break
    if not obstacle_ok:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    spec_builder = random.choice([_build_template_two_pairs, _build_template_three_pairs])
    spec = spec_builder()
    band_colors = random.choice(BAND_COLOR_PALETTES)
    bands = _realize_bands(spec, band_colors)
    if bands is None:
        return None
    inp = _stack_bands(bands)
    if not _validate_semantics(inp, spec):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid overlay-topology ARC input")
