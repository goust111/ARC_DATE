from __future__ import annotations

import itertools
import random
from collections import defaultdict, deque

import numpy as np


BG_COLOR = 0
GREY_COLOR = 5
ENDPOINT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
LINE_MODES = ("diag", "shallow", "steep")
MODE_VECTORS = {
    "diag": ((1, 1),),
    "shallow": ((1, 2), (1, 3), (2, 3)),
    "steep": ((2, 1), (3, 1), (3, 2)),
}

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
PAIR_RANGE = (2, 5)
LEGEND_ROWS_RANGE = (1, 3)
PLAY_MARGIN = 1
MIN_ENDPOINT_GAP = 2
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _raster_line(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p0
    r1, c1 = p1
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and dc != 0

    if abs(dc) > abs(dr):
        if c0 > c1:
            r0, c0, r1, c1 = r1, c1, r0, c0
            dr = r1 - r0
            dc = c1 - c0
        cells = []
        for c in range(c0, c1 + 1):
            t = (c - c0) / dc
            r = _round_half_up(r0 + dr * t)
            cells.append((r, c))
        return cells

    if r0 > r1:
        r0, c0, r1, c1 = r1, c1, r0, c0
        dr = r1 - r0
        dc = c1 - c0
    cells = []
    for r in range(r0, r1 + 1):
        t = (r - r0) / dr
        c = _round_half_up(c0 + dc * t)
        cells.append((r, c))
    return cells


def _legend_info(grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[frozenset[int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _components4(g != bg)
    candidates = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_w != 3 or len(comp) != box_h * box_w:
            continue
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if touches_corner:
            candidates.append((comp, (r0, c0, r1, c1)))
    assert len(candidates) == 1, "expected exactly one corner legend"

    comp, (r0, c0, r1, c1) = candidates[0]
    mask = np.zeros_like(g, dtype=bool)
    for r, c in comp:
        mask[r, c] = True

    mapping: dict[frozenset[int], int] = {}
    for r in range(r0, r1 + 1):
        a = int(g[r, c0])
        b = int(g[r, c0 + 1])
        mixed = int(g[r, c0 + 2])
        assert a != bg and b != bg and mixed != bg
        assert a != b
        key = frozenset((a, b))
        assert key not in mapping
        mapping[key] = mixed
    return mask, mapping


def _pair_endpoints(
    grid: np.ndarray, bg: int, legend_mask: np.ndarray
) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    endpoint_mask = (g != bg) & (~legend_mask)
    comps = _components4(endpoint_mask)
    assert comps, "expected at least one endpoint pair"
    assert all(len(comp) == 1 for comp in comps), "non-legend foreground must be isolated pixels"

    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(endpoint_mask):
        by_color[int(g[r, c])].append((int(r), int(c)))

    assert 2 <= len(by_color) <= 5
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color, pts in by_color.items():
        assert len(pts) == 2, "each endpoint color must appear exactly twice outside the legend"
        assert pts[0][0] != pts[1][0] and pts[0][1] != pts[1][1]
        pairs[color] = (pts[0], pts[1])
    return pairs


def _solve_from_parts(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    legend_mapping: dict[frozenset[int], int],
    endpoint_pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    visits: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color, (p0, p1) in endpoint_pairs.items():
        for cell in _raster_line(p0, p1):
            visits[cell].add(int(color))

    for (r, c), colors in visits.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        elif len(colors) == 2:
            out[r, c] = legend_mapping.get(frozenset(colors), GREY_COLOR)
        else:
            out[r, c] = GREY_COLOR

    out[legend_mask] = np.asarray(grid, dtype=int)[legend_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the line between each same-colored endpoint pair, then recolor
    crossings using the corner legend's unordered color-pair lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    legend_mask, legend_mapping = _legend_info(g, bg)
    endpoint_pairs = _pair_endpoints(g, bg, legend_mask)
    return _solve_from_parts(g, legend_mask, legend_mapping, endpoint_pairs)


def _legend_bbox(corner: str, h: int, w: int, rows: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, rows - 1, 2
    if corner == "tr":
        return 0, w - 3, rows - 1, w - 1
    if corner == "bl":
        return h - rows, 0, h - 1, 2
    assert corner == "br"
    return h - rows, w - 3, h - 1, w - 1


def _play_bounds(corner: str, h: int, w: int, legend_rows: int) -> tuple[int, int, int, int]:
    r_lo, c_lo, r_hi, c_hi = 0, 0, h - 1, w - 1
    if corner in ("tl", "tr"):
        r_lo = legend_rows + PLAY_MARGIN
    else:
        r_hi = h - legend_rows - PLAY_MARGIN - 1
    if corner in ("tl", "bl"):
        c_lo = 3 + PLAY_MARGIN
    else:
        c_hi = w - 3 - PLAY_MARGIN - 1
    return r_lo, r_hi, c_lo, c_hi


def _cell_in_bounds(cell: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r_lo, r_hi, c_lo, c_hi = bounds
    return r_lo <= r <= r_hi and c_lo <= c <= c_hi


def _max_steps(
    anchor: tuple[int, int], vec: tuple[int, int], bounds: tuple[int, int, int, int]
) -> int:
    r, c = anchor
    vr, vc = vec
    r_lo, r_hi, c_lo, c_hi = bounds
    limits = []
    if vr > 0:
        limits.append((r_hi - r) // vr)
    else:
        limits.append((r - r_lo) // (-vr))
    if vc > 0:
        limits.append((c_hi - c) // vc)
    else:
        limits.append((c - c_lo) // (-vc))
    return int(min(limits))


def _endpoint_conflicts(cell: tuple[int, int], endpoints: list[tuple[int, int]]) -> bool:
    for other_r, other_c in endpoints:
        if max(abs(cell[0] - other_r), abs(cell[1] - other_c)) < MIN_ENDPOINT_GAP:
            return True
    return False


def _sample_anchor(bounds: tuple[int, int, int, int], rng: random.Random) -> tuple[int, int]:
    r_lo, r_hi, c_lo, c_hi = bounds
    return rng.randint(r_lo, r_hi), rng.randint(c_lo, c_hi)


def _sample_line_through(
    anchor: tuple[int, int],
    mode: str,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    used_endpoints: list[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(80):
        base_r, base_c = rng.choice(MODE_VECTORS[mode])
        vr = base_r * rng.choice((-1, 1))
        vc = base_c * rng.choice((-1, 1))
        back = _max_steps(anchor, (-vr, -vc), bounds)
        forward = _max_steps(anchor, (vr, vc), bounds)
        if back < 1 or forward < 1:
            continue
        k_back = rng.randint(1, back)
        k_forward = rng.randint(1, forward)
        p0 = (anchor[0] - k_back * vr, anchor[1] - k_back * vc)
        p1 = (anchor[0] + k_forward * vr, anchor[1] + k_forward * vc)
        if _endpoint_conflicts(p0, used_endpoints) or _endpoint_conflicts(p1, used_endpoints):
            continue
        cells = _raster_line(p0, p1)
        if anchor not in cells or len(cells) < 5:
            continue
        return {
            "anchor": anchor,
            "mode": mode,
            "endpoints": (p0, p1),
            "cells": tuple(cells),
            "cell_set": frozenset(cells),
        }
    return None


def _sample_crossing_line(
    anchor: tuple[int, int],
    mode: str,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    used_endpoints: list[tuple[int, int]],
    required_cells: frozenset[tuple[int, int]],
    forbidden_sets: list[frozenset[tuple[int, int]]],
) -> dict[str, object] | None:
    for _ in range(120):
        rec = _sample_line_through(anchor, mode, bounds, rng, used_endpoints)
        if rec is None:
            continue
        overlap = set(rec["cell_set"]) & set(required_cells)
        if overlap != {anchor}:
            continue
        if any(set(rec["cell_set"]) & set(other) for other in forbidden_sets):
            continue
        return rec
    return None


def _sample_non_crossing_line(
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    used_endpoints: list[tuple[int, int]],
    occupied_sets: list[frozenset[tuple[int, int]]],
) -> dict[str, object] | None:
    occupied_cells = set().union(*occupied_sets) if occupied_sets else set()
    for _ in range(200):
        anchor = _sample_anchor(bounds, rng)
        if anchor in occupied_cells:
            continue
        mode = rng.choice(LINE_MODES)
        rec = _sample_line_through(anchor, mode, bounds, rng, used_endpoints)
        if rec is None:
            continue
        if any(set(rec["cell_set"]) & set(other) for other in occupied_sets):
            continue
        return rec
    return None


def _pixel_to_colors(
    colors: list[int], records: list[dict[str, object]]
) -> dict[tuple[int, int], set[int]]:
    visits: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color, rec in zip(colors, records):
        for cell in rec["cells"]:
            visits[cell].add(int(color))
    return visits


def _choose_legend_pairs(
    mode: str,
    endpoint_colors: list[int],
    cross_pairs: list[frozenset[int]],
    rng: random.Random,
) -> list[frozenset[int]] | None:
    all_pairs = [frozenset(pair) for pair in itertools.combinations(endpoint_colors, 2)]
    non_cross_pairs = [pair for pair in all_pairs if pair not in cross_pairs]

    if mode == "mapped":
        if len(cross_pairs) > LEGEND_ROWS_RANGE[1]:
            return None
        legend_pairs = list(cross_pairs)
        extra_pool = [pair for pair in non_cross_pairs if pair not in legend_pairs]
        max_extra = min(LEGEND_ROWS_RANGE[1] - len(legend_pairs), len(extra_pool))
        if max_extra > 0 and rng.random() < 0.5:
            extra_count = rng.randint(0, max_extra)
            legend_pairs.extend(rng.sample(extra_pool, extra_count))
        return legend_pairs

    if mode == "grey":
        if not non_cross_pairs:
            return None
        count = rng.randint(1, min(LEGEND_ROWS_RANGE[1], len(non_cross_pairs)))
        return rng.sample(non_cross_pairs, count)

    assert mode == "mixed"
    if len(cross_pairs) < 2 or not non_cross_pairs:
        return None
    mapped_count = rng.randint(1, min(len(cross_pairs) - 1, LEGEND_ROWS_RANGE[1]))
    mapped = rng.sample(cross_pairs, mapped_count)
    legend_pairs = list(mapped)
    remaining_slots = LEGEND_ROWS_RANGE[1] - len(legend_pairs)
    if remaining_slots > 0 and rng.random() < 0.7:
        extra_count = rng.randint(0, min(remaining_slots, len(non_cross_pairs)))
        if extra_count:
            legend_pairs.extend(rng.sample(non_cross_pairs, extra_count))
    if set(cross_pairs).issubset(set(legend_pairs)):
        return None
    return legend_pairs


def _render_legend(
    legend_pairs: list[frozenset[int]], endpoint_colors: list[int], rng: random.Random
) -> tuple[np.ndarray, dict[frozenset[int], int]]:
    rows = len(legend_pairs)
    legend = np.zeros((rows, 3), dtype=int)
    result_colors = [color for color in ENDPOINT_COLORS if color not in endpoint_colors]
    rng.shuffle(result_colors)
    mapping: dict[frozenset[int], int] = {}
    for idx, pair in enumerate(legend_pairs):
        a, b = tuple(pair)
        mixed = result_colors[idx]
        if rng.random() < 0.5:
            legend[idx] = (a, b, mixed)
        else:
            legend[idx] = (b, a, mixed)
        mapping[pair] = mixed
    return legend, mapping


def _build_expected(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    legend_mapping: dict[frozenset[int], int],
    colors: list[int],
    records: list[dict[str, object]],
) -> np.ndarray | None:
    visits = _pixel_to_colors(colors, records)
    if any(len(v) > 2 for v in visits.values()):
        return None

    out = np.array(grid, copy=True)
    for cell, color_set in visits.items():
        if len(color_set) == 1:
            out[cell] = next(iter(color_set))
        else:
            out[cell] = legend_mapping.get(frozenset(color_set), GREY_COLOR)
    out[legend_mask] = grid[legend_mask]
    return out


def _sample_initial_pair(
    bounds: tuple[int, int, int, int],
    combo: tuple[str, str],
    rng: random.Random,
) -> tuple[dict[str, object], dict[str, object]] | None:
    for _ in range(120):
        anchor = _sample_anchor(bounds, rng)
        used_endpoints: list[tuple[int, int]] = []
        first = _sample_line_through(anchor, combo[0], bounds, rng, used_endpoints)
        if first is None:
            continue
        used_endpoints.extend(first["endpoints"])
        second = _sample_crossing_line(
            anchor,
            combo[1],
            bounds,
            rng,
            used_endpoints,
            first["cell_set"],
            [],
        )
        if second is None:
            continue
        return first, second
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    mode = rng.choice(("mapped", "grey", "mixed"))
    min_pairs = 2 if mode == "mapped" else 3
    n_pairs = rng.randint(min_pairs, PAIR_RANGE[1])

    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    legend_rows = rng.randint(*LEGEND_ROWS_RANGE)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    bounds = _play_bounds(corner, h, w, legend_rows)
    r_lo, r_hi, c_lo, c_hi = bounds
    if r_hi - r_lo < 5 or c_hi - c_lo < 5:
        return None

    endpoint_colors = list(rng.sample(ENDPOINT_COLORS, n_pairs))
    combo = rng.choice(
        (
            ("diag", "diag"),
            ("diag", "shallow"),
            ("diag", "steep"),
            ("shallow", "steep"),
            ("steep", "shallow"),
        )
    )
    initial = _sample_initial_pair(bounds, combo, rng)
    if initial is None:
        return None
    records = [initial[0], initial[1]]
    used_endpoints = list(initial[0]["endpoints"]) + list(initial[1]["endpoints"])

    if mode == "mixed":
        cross2_candidates = [
            cell
            for cell in initial[0]["cells"][1:-1]
            if cell != initial[0]["anchor"] and cell not in initial[1]["cell_set"]
        ]
        if not cross2_candidates:
            return None
        rng.shuffle(cross2_candidates)
        third = None
        for cross2 in cross2_candidates:
            mode3 = rng.choice(LINE_MODES)
            third = _sample_crossing_line(
                cross2,
                mode3,
                bounds,
                rng,
                used_endpoints,
                initial[0]["cell_set"],
                [initial[1]["cell_set"]],
            )
            if third is not None:
                break
        if third is None:
            return None
        records.append(third)
        used_endpoints.extend(third["endpoints"])

    while len(records) < n_pairs:
        extra = _sample_non_crossing_line(
            bounds,
            rng,
            used_endpoints,
            [rec["cell_set"] for rec in records],
        )
        if extra is None:
            return None
        records.append(extra)
        used_endpoints.extend(extra["endpoints"])

    visits = _pixel_to_colors(endpoint_colors, records)
    if any(len(v) > 2 for v in visits.values()):
        return None

    endpoint_cells = {cell for rec in records for cell in rec["endpoints"]}
    intersections = {cell: frozenset(colors) for cell, colors in visits.items() if len(colors) == 2}
    if not intersections:
        return None
    if any(cell in endpoint_cells for cell in intersections):
        return None

    cross_pairs = sorted(set(intersections.values()), key=lambda pair: tuple(sorted(pair)))
    legend_pairs = _choose_legend_pairs(mode, endpoint_colors, cross_pairs, rng)
    if not legend_pairs:
        return None

    legend, legend_mapping = _render_legend(legend_pairs, endpoint_colors, rng)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    lr0, lc0, lr1, lc1 = _legend_bbox(corner, h, w, legend.shape[0])
    grid[lr0 : lr1 + 1, lc0 : lc1 + 1] = legend

    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    for color, rec in zip(endpoint_colors, records):
        p0, p1 = rec["endpoints"]
        if not (_cell_in_bounds(p0, bounds) and _cell_in_bounds(p1, bounds)):
            return None
        grid[p0] = color
        grid[p1] = color

    expected = _build_expected(grid, legend_mask, legend_mapping, endpoint_colors, records)
    if expected is None:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    mapped_crosses = sum(1 for pair in intersections.values() if pair in legend_mapping)
    if mode == "mapped" and mapped_crosses != len(intersections):
        return None
    if mode == "grey" and mapped_crosses != 0:
        return None
    if mode == "mixed" and not (0 < mapped_crosses < len(intersections)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
