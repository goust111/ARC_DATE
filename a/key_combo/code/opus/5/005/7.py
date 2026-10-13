from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

Coord = tuple[int, int]


# 3x3 asymmetric 4-connected sprites with a center anchor.
SHAPE_LIBRARY: tuple[tuple[Coord, ...], ...] = (
    ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 1)),
    ((-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0)),
    ((-1, -1), (-1, 0), (0, 0), (0, 1), (1, 0), (1, 1)),
    ((-1, 0), (-1, 1), (0, -1), (0, 0), (1, 0), (1, 1)),
    ((-1, -1), (0, -1), (0, 0), (1, 0), (1, 1)),
)

GENERATE_RETRIES = 600
LAYOUT_RETRIES = 300
SEGMENT_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(
    grid: np.ndarray, color: int, row_lo: int = 0, row_hi: int | None = None
) -> list[np.ndarray]:
    g = np.asarray(grid)
    h, w = g.shape
    if row_hi is None:
        row_hi = h
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(row_lo, row_hi):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (row_lo <= nr < row_hi and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _center_anchor(coords: np.ndarray) -> Coord:
    r0, c0, r1, c1 = _bbox_coords(coords)
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _sprite_offsets(coords: np.ndarray) -> frozenset[Coord]:
    ar, ac = _center_anchor(coords)
    return frozenset((int(r - ar), int(c - ac)) for r, c in coords)


def _sprite_cells(anchor: Coord, offsets: frozenset[Coord] | tuple[Coord, ...]) -> set[Coord]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in offsets}


def _mirror_cells(cells: set[Coord], sep_r: int) -> frozenset[Coord]:
    return frozenset((2 * sep_r - r, c) for r, c in cells)


def _blue_endpoints(blue_coords: np.ndarray) -> tuple[Coord, Coord]:
    rows = sorted({int(r) for r, _c in blue_coords})
    cols = sorted({int(c) for _r, c in blue_coords})
    if len(rows) == 1:
        r = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1))
        return (r, cols[0]), (r, cols[-1])
    if len(cols) == 1:
        c = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1))
        return (rows[0], c), (rows[-1], c)
    raise AssertionError("blue object must be a horizontal or vertical segment")


def _bresenham_line(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    cells: list[Coord] = []
    while True:
        cells.append((r0, c0))
        if (r0, c0) == (r1, c1):
            return cells
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


def _render_output(
    shape_offsets: frozenset[Coord],
    height: int,
    width: int,
    sep_r: int,
    top_anchors: list[Coord],
    matched_indices: tuple[int, ...],
    blue_endpoints: tuple[Coord, Coord],
) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    out[sep_r, :] = RED

    matched_set = set(matched_indices)
    for idx, anchor in enumerate(top_anchors):
        if idx in matched_set:
            continue
        for r, c in _sprite_cells(anchor, shape_offsets):
            out[r, c] = GREEN

    anchors = [top_anchors[idx] for idx in matched_indices]
    for a, b in combinations(anchors, 2):
        for r, c in _bresenham_line(a, b):
            out[r, c] = GREEN

    for endpoint in blue_endpoints:
        for r, c in _sprite_cells(endpoint, shape_offsets):
            out[r, c] = GREEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use lower-half mirror partners to gate which upper sprites swap roles with the blue segment.

    Input structure:
    - Black background fills the grid, with one solid red horizontal separator row exactly in the middle.
    - The upper half contains four identical green 4-connected sprites with odd-sized bbox centers, plus one blue horizontal or vertical segment.
    - The lower half contains green sprites only at exact horizontal-axis mirror positions of a subset of the upper sprites.

    Rule:
    - Find the separator row, the upper green sprites, the lower green sprites, and the blue segment endpoints.
    - For each upper sprite, reflect all of its green cells across the separator row; sprites whose reflected cell set exists in the lower half form the matched group, while the others remain unmatched.
    - Erase the matched upper sprites, erase the blue segment, erase the lower mirrored partners, then draw green 8-connected straight lines between every pair of matched upper anchors.
    - Finally stamp the green sprite shape, aligned by its center anchor, onto both original blue-segment endpoints; unmatched upper sprites stay in place and stamped sprite pixels override any new line overlap.

    Key insight:
    Mirror symmetry is only a membership test: it selects which upper sprites participate in the connector-endpoint role swap, instead of moving every sprite.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == RED)]
    assert len(sep_rows) == 1
    sep_r = sep_rows[0]
    assert h % 2 == 1 and sep_r == h // 2

    top_comps = _components_of_color(g, GREEN, 0, sep_r)
    bottom_comps = _components_of_color(g, GREEN, sep_r + 1, h)
    assert len(top_comps) == 4
    assert 2 <= len(bottom_comps) <= 3

    blue_coords = np.argwhere(g == BLUE)
    assert blue_coords.size > 0
    blue_endpoints = _blue_endpoints(blue_coords)

    top_infos = []
    ref_offsets: frozenset[Coord] | None = None
    for coords in top_comps:
        anchor = _center_anchor(coords)
        offsets = _sprite_offsets(coords)
        if ref_offsets is None:
            ref_offsets = offsets
        assert offsets == ref_offsets
        top_infos.append(
            {
                "anchor": anchor,
                "cells": {tuple(map(int, rc)) for rc in coords},
            }
        )
    assert ref_offsets is not None

    bottom_lookup = {
        frozenset(tuple(map(int, rc)) for rc in coords): coords for coords in bottom_comps
    }
    matched_indices: list[int] = []
    for idx, info in enumerate(top_infos):
        reflected = _mirror_cells(info["cells"], sep_r)
        if reflected in bottom_lookup:
            matched_indices.append(idx)
    assert 2 <= len(matched_indices) <= 3

    return _render_output(
        ref_offsets,
        h,
        w,
        sep_r,
        [info["anchor"] for info in top_infos],
        tuple(sorted(matched_indices)),
        blue_endpoints,
    )


def _anchors_are_separated(anchors: list[Coord]) -> bool:
    for i, (r0, c0) in enumerate(anchors):
        for r1, c1 in anchors[i + 1 :]:
            if abs(r0 - r1) <= 3 and abs(c0 - c1) <= 3:
                return False
    return True


def _nontrivial_subset(anchors: list[Coord], matched: tuple[int, ...]) -> bool:
    chosen = set(matched)
    count = len(matched)
    idxs = list(range(len(anchors)))
    key_fns = (
        lambda i: (anchors[i][0], anchors[i][1]),
        lambda i: (anchors[i][1], anchors[i][0]),
        lambda i: (anchors[i][0] + anchors[i][1], anchors[i][1]),
        lambda i: (anchors[i][0] - anchors[i][1], anchors[i][1]),
    )
    for key_fn in key_fns:
        ordered = sorted(idxs, key=key_fn)
        if chosen == set(ordered[:count]) or chosen == set(ordered[-count:]):
            return False
    return True


def _collinear(points: list[Coord]) -> bool:
    if len(points) < 3:
        return False
    (r0, c0), (r1, c1), (r2, c2) = points
    return (r1 - r0) * (c2 - c0) == (r2 - r0) * (c1 - c0)


def _sample_layout(
    half_h: int, width: int, matched_count: int, shape_offsets: frozenset[Coord]
) -> tuple[list[Coord], tuple[int, ...], frozenset[Coord], list[set[Coord]]] | None:
    candidates = [(r, c) for r in range(1, half_h - 1) for c in range(1, width - 1)]
    for _ in range(LAYOUT_RETRIES):
        anchors = random.sample(candidates, 4)
        if not _anchors_are_separated(anchors):
            continue
        if len({r for r, _c in anchors}) < 2 or len({c for _r, c in anchors}) < 3:
            continue
        matched = tuple(sorted(random.sample(range(4), matched_count)))
        if not _nontrivial_subset(anchors, matched):
            continue
        if matched_count == 3 and _collinear([anchors[i] for i in matched]):
            continue

        top_cells_by_idx = [_sprite_cells(anchor, shape_offsets) for anchor in anchors]
        unmatched_cells = set().union(
            *(top_cells_by_idx[idx] for idx in range(4) if idx not in set(matched))
        )
        connector_cells = frozenset(
            cell
            for a_idx, b_idx in combinations(matched, 2)
            for cell in _bresenham_line(anchors[a_idx], anchors[b_idx])
        )
        if connector_cells & unmatched_cells:
            continue
        return anchors, matched, connector_cells, top_cells_by_idx
    return None


def _sample_blue_segment(
    half_h: int,
    width: int,
    shape_offsets: frozenset[Coord],
    top_cells: set[Coord],
    connector_cells: frozenset[Coord],
) -> tuple[tuple[Coord, Coord], frozenset[Coord]] | None:
    for _ in range(SEGMENT_RETRIES):
        orientation = random.choice(["h", "v"])
        if orientation == "h":
            max_len = min(8, width - 2)
            if max_len < 5:
                continue
            length = random.randint(5, max_len)
            r = random.randint(1, half_h - 2)
            c0 = random.randint(1, width - length - 1)
            e1 = (r, c0)
            e2 = (r, c0 + length - 1)
        else:
            max_len = min(8, half_h - 2)
            if max_len < 5:
                continue
            length = random.randint(5, max_len)
            r0_max = half_h - length - 1
            if r0_max < 1:
                continue
            r0 = random.randint(1, r0_max)
            c = random.randint(1, width - 2)
            e1 = (r0, c)
            e2 = (r0 + length - 1, c)

        blue_cells = frozenset(_bresenham_line(e1, e2))
        stamp_cells = _sprite_cells(e1, shape_offsets) | _sprite_cells(e2, shape_offsets)
        if blue_cells & top_cells:
            continue
        if stamp_cells & top_cells:
            continue
        if stamp_cells & connector_cells:
            continue
        if _sprite_cells(e1, shape_offsets) & _sprite_cells(e2, shape_offsets):
            continue
        return (e1, e2), blue_cells
    return None


def _render_input(
    shape_offsets: frozenset[Coord],
    height: int,
    width: int,
    sep_r: int,
    top_anchors: list[Coord],
    matched_indices: tuple[int, ...],
    blue_cells: frozenset[Coord],
) -> np.ndarray:
    inp = np.full((height, width), BG, dtype=int)
    inp[sep_r, :] = RED
    for anchor in top_anchors:
        for r, c in _sprite_cells(anchor, shape_offsets):
            inp[r, c] = GREEN
    for idx in matched_indices:
        mirrored = _mirror_cells(_sprite_cells(top_anchors[idx], shape_offsets), sep_r)
        for r, c in mirrored:
            inp[r, c] = GREEN
    for r, c in blue_cells:
        inp[r, c] = BLUE
    return inp


def _semantic_checks(
    shape_offsets: frozenset[Coord],
    height: int,
    width: int,
    sep_r: int,
    top_anchors: list[Coord],
    matched_indices: tuple[int, ...],
    blue_endpoints: tuple[Coord, Coord],
    blue_cells: frozenset[Coord],
    connector_cells: frozenset[Coord],
    top_cells_by_idx: list[set[Coord]],
    expected: np.ndarray,
) -> bool:
    matched_set = set(matched_indices)
    unmatched_cells = set().union(
        *(top_cells_by_idx[idx] for idx in range(4) if idx not in matched_set)
    )
    matched_cells = set().union(*(top_cells_by_idx[idx] for idx in matched_indices))
    stamp_cells = _sprite_cells(blue_endpoints[0], shape_offsets) | _sprite_cells(
        blue_endpoints[1], shape_offsets
    )

    if not (matched_cells - connector_cells - stamp_cells):
        return False
    if not (blue_cells - stamp_cells):
        return False
    if not (stamp_cells - connector_cells - unmatched_cells):
        return False
    if not (connector_cells - stamp_cells - unmatched_cells):
        return False

    alt_same_size = []
    for subset in combinations(range(4), len(matched_indices)):
        if tuple(subset) == matched_indices:
            continue
        alt_same_size.append(
            _render_output(shape_offsets, height, width, sep_r, top_anchors, tuple(subset), blue_endpoints)
        )
    if any(np.array_equal(expected, alt) for alt in alt_same_size):
        return False

    all_swap = _render_output(
        shape_offsets, height, width, sep_r, top_anchors, tuple(range(4)), blue_endpoints
    )
    if np.array_equal(expected, all_swap):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(7, 10)
    height = 2 * half_h + 1
    width = random.randint(15, 21)
    sep_r = half_h
    matched_count = random.choice([2, 3])
    shape_offsets = frozenset(random.choice(SHAPE_LIBRARY))

    layout = _sample_layout(half_h, width, matched_count, shape_offsets)
    if layout is None:
        return None
    top_anchors, matched_indices, connector_cells, top_cells_by_idx = layout

    top_cells = set().union(*top_cells_by_idx)
    blue = _sample_blue_segment(half_h, width, shape_offsets, top_cells, connector_cells)
    if blue is None:
        return None
    blue_endpoints, blue_cells = blue

    inp = _render_input(
        shape_offsets,
        height,
        width,
        sep_r,
        top_anchors,
        matched_indices,
        blue_cells,
    )
    expected = _render_output(
        shape_offsets, height, width, sep_r, top_anchors, matched_indices, blue_endpoints
    )

    if not _semantic_checks(
        shape_offsets,
        height,
        width,
        sep_r,
        top_anchors,
        matched_indices,
        blue_endpoints,
        blue_cells,
        connector_cells,
        top_cells_by_idx,
        expected,
    ):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid mirrored-gated role-swap input")
