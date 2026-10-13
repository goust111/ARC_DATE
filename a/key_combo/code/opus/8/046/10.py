from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

GEN_RETRIES = 500


def _divider_positions(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    divider_rows = [r for r in range(h) if bool(np.all(grid[r, :] == GREY))]
    divider_cols = [c for c in range(w) if bool(np.all(grid[:, c] == GREY))]
    return divider_rows, divider_cols


def _spans(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for divider in sorted(dividers):
        if start <= divider - 1:
            spans.append((start, divider - 1))
        start = divider + 1
    if start <= size - 1:
        spans.append((start, size - 1))
    return spans


def _band_index(value: int, spans: list[tuple[int, int]]) -> int | None:
    for i, (lo, hi) in enumerate(spans):
        if lo <= value <= hi:
            return i
    return None


def _region_for_cell(r: int, c: int, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    ri = _band_index(r, row_spans)
    ci = _band_index(c, col_spans)
    if ri is None or ci is None:
        return None
    return ri, ci


def _trace_paths(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    *,
    use_deflectors: bool,
) -> list[dict]:
    h, w = grid.shape
    sources = sorted(int(c) for c in np.where(grid[0] == TEAL)[0])
    deflector_rows = {int(r) for r in np.where(grid[:, 0] == RED)[0]}
    traces: list[dict] = []

    for source_c in sources:
        c = int(source_c)
        cells: list[tuple[int, int]] = []
        visited_regions: list[tuple[int, int]] = []
        seen_regions: set[tuple[int, int]] = set()

        if 0 <= c < w and grid[0, c] != GREY:
            cells.append((0, c))
            region = _region_for_cell(0, c, row_spans, col_spans)
            if region is not None and region not in seen_regions:
                seen_regions.add(region)
                visited_regions.append(region)

        for r in range(1, h):
            if use_deflectors and r in deflector_rows:
                c += 1
            if c < 0 or c >= w:
                break
            if grid[r, c] == GREY:
                continue
            cells.append((r, c))
            region = _region_for_cell(r, c, row_spans, col_spans)
            if region is not None and region not in seen_regions:
                seen_regions.add(region)
                visited_regions.append(region)

        traces.append(
            {
                "source_c": int(source_c),
                "cells": cells,
                "regions": visited_regions,
                "column_bands": sorted({region[1] for region in visited_regions}),
            }
        )
    return traces


def _analyze(grid: np.ndarray, *, use_deflectors: bool = True, require_unique: bool = True) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    divider_rows, divider_cols = _divider_positions(g)
    assert len(divider_rows) == 1, "expected exactly one full horizontal divider row"
    assert len(divider_cols) in (1, 2), "expected one or two full vertical divider columns"

    row_spans = _spans(g.shape[0], divider_rows)
    col_spans = _spans(g.shape[1], divider_cols)
    assert len(row_spans) == 2, "expected two row bands"
    assert len(col_spans) in (2, 3), "expected two or three column bands"

    sources = sorted(int(c) for c in np.where(g[0] == TEAL)[0])
    deflector_rows = sorted(int(r) for r in np.where(g[:, 0] == RED)[0])
    assert 3 <= len(sources) <= 5, "expected 3-5 teal sources on the top edge"
    assert 2 <= len(deflector_rows) <= 4, "expected 2-4 red deflectors on the left edge"
    assert 0 not in sources, "top-left corner cannot be both source and deflector"
    assert 0 not in deflector_rows, "red deflectors must not occupy the top-left corner"

    region_ids = [(ri, ci) for ri in range(len(row_spans)) for ci in range(len(col_spans))]
    traces = _trace_paths(g, row_spans, col_spans, use_deflectors=use_deflectors)
    votes: Counter[tuple[int, int]] = Counter()
    for trace in traces:
        assert trace["regions"], "each path must visit at least one region"
        for region in trace["regions"]:
            votes[region] += 1

    ranked = votes.most_common()
    assert ranked, "expected at least one visited region"
    winner = ranked[0][0]
    unique = len(ranked) == 1 or ranked[0][1] > ranked[1][1]
    if require_unique:
        assert unique, "expected a unique winning region"
    else:
        winner = ranked[0][0] if unique else None

    return {
        "grid": g,
        "divider_rows": divider_rows,
        "divider_cols": divider_cols,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "sources": sources,
        "deflector_rows": deflector_rows,
        "region_ids": region_ids,
        "traces": traces,
        "votes": votes,
        "winner": winner,
        "winner_votes": int(ranked[0][1]),
    }


def _render_output(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for ri, (r0, r1) in enumerate(analysis["row_spans"]):
        for ci, (c0, c1) in enumerate(analysis["col_spans"]):
            fill_color = ORANGE if (ri, ci) == analysis["winner"] else BLACK
            out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    for r in analysis["divider_rows"]:
        out[r, :] = GREY
    for c in analysis["divider_cols"]:
        out[:, c] = GREY
    out[0, analysis["sources"]] = TEAL
    out[analysis["deflector_rows"], 0] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the uniquely most-crossed partition after tracing deflected downward flow paths.

    Input structure:
    - A black grid is split by full grey divider rows and columns into 2x2 or 2x3 rectangular regions.
    - Several teal source pixels sit on the top edge, and several red deflector pixels sit on the left edge.
    - Each region may contain a few orange marker pixels, but the markers are only visual distractors.

    Rule:
    - For each teal source, trace a path downward from row 0 to the bottom edge.
    - Whenever the path enters a row whose left-edge cell is red, shift that path one column to the right before considering the cell on that row.
    - Grey divider lines do not block the path, but path cells on grey lines are skipped rather than drawn.
    - Count how many distinct paths enter each rectangular region at least once; the unique highest-vote region wins.
    - Fill the winning region orange, clear every other region to black, keep the grey dividers unchanged, and preserve the original teal and red edge pixels.

    Key insight:
    - The red edge pixels do not mark the winning region directly; they globally bend every path, so the winner can only be found by tracing complete path trajectories and then voting by region crossings.
    """
    analysis = _analyze(grid, use_deflectors=True)
    return _render_output(np.asarray(grid, dtype=int), analysis)


def _sample_partition(rng: random.Random) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    top_h = rng.randint(5, 8)
    bottom_h = rng.randint(5, 8)
    n_cols = rng.choice([2, 3])
    widths = [rng.randint(4, 7) for _ in range(n_cols)]

    h = top_h + 1 + bottom_h
    w = sum(widths) + (n_cols - 1)

    divider_row = top_h
    divider_cols: list[int] = []
    cursor = widths[0]
    for width in widths[:-1]:
        divider_cols.append(cursor)
        cursor += width + 1

    row_spans = _spans(h, [divider_row])
    col_spans = _spans(w, divider_cols)
    return h, w, row_spans, col_spans, [divider_row], divider_cols


def _build_base_grid(
    h: int,
    w: int,
    divider_rows: list[int],
    divider_cols: list[int],
    sources: list[int],
    deflector_rows: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r in divider_rows:
        grid[r, :] = GREY
    for c in divider_cols:
        grid[:, c] = GREY
    grid[0, sources] = TEAL
    grid[deflector_rows, 0] = RED
    return grid


def _path_regions_from_structure(
    h: int,
    w: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sources: list[int],
    deflector_rows: list[int],
) -> tuple[list[list[tuple[int, int]]], Counter[tuple[int, int]]]:
    votes: Counter[tuple[int, int]] = Counter()
    per_path: list[list[tuple[int, int]]] = []
    deflectors = set(deflector_rows)

    for source_c in sources:
        c = int(source_c)
        seen: set[tuple[int, int]] = set()
        ordered: list[tuple[int, int]] = []
        for r in range(h):
            if r > 0 and r in deflectors:
                c += 1
            if c < 0 or c >= w:
                break
            if r in {span[0] - 1 for span in row_spans[1:]}:
                pass
            region = _region_for_cell(r, c, row_spans, col_spans)
            if region is None:
                continue
            if region not in seen:
                seen.add(region)
                ordered.append(region)
        per_path.append(ordered)
        for region in ordered:
            votes[region] += 1
    return per_path, votes


def _unique_winner(votes: Counter[tuple[int, int]]) -> tuple[int, int] | None:
    ranked = votes.most_common()
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _sample_deflectors(rng: random.Random, h: int, divider_row: int) -> list[int] | None:
    total = rng.randint(2, 4)
    above = rng.randint(1, total - 1)
    below = total - above
    top_rows = list(range(1, divider_row))
    bottom_rows = list(range(divider_row + 1, h))
    if len(top_rows) < above or len(bottom_rows) < below:
        return None
    rows = rng.sample(top_rows, above) + rng.sample(bottom_rows, below)
    rows.sort()
    return rows


def _sample_sources(
    rng: random.Random,
    w: int,
    divider_cols: list[int],
    max_shift: int,
) -> list[int] | None:
    candidates = [c for c in range(1, w - max_shift) if c not in divider_cols]
    if len(candidates) < 3:
        return None
    n_sources = rng.randint(3, min(5, len(candidates)))
    return sorted(rng.sample(candidates, n_sources))


def _marker_cells_for_region(
    region: tuple[int, int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, r1 = row_spans[region[0]]
    c0, c1 = col_spans[region[1]]
    rr0 = min(r1, r0 + 1)
    rr1 = max(rr0, r1 - 1)
    cc0 = min(c1, c0 + 1)
    cc1 = max(cc0, c1 - 1)

    candidates = [
        (r, c)
        for r in range(rr0, rr1 + 1)
        for c in range(cc0, cc1 + 1)
        if (r, c) not in occupied and r > 0 and c > 0
    ]
    return candidates


def _place_markers(
    rng: random.Random,
    grid: np.ndarray,
    analysis: dict,
) -> bool:
    winner = analysis["winner"]
    region_ids = analysis["region_ids"]
    occupied = {(0, c) for c in analysis["sources"]} | {(r, 0) for r in analysis["deflector_rows"]}

    marker_counts = {region: 1 for region in region_ids}
    decoys = [region for region in region_ids if region != winner]
    extra_count = rng.randint(1, min(2, len(decoys)))
    for region in rng.sample(decoys, extra_count):
        marker_counts[region] = 2

    if marker_counts[winner] >= max(marker_counts.values()):
        return False

    for region in region_ids:
        cells = _marker_cells_for_region(region, analysis["row_spans"], analysis["col_spans"], occupied)
        rng.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in chosen):
                chosen.append(cell)
            if len(chosen) == marker_counts[region]:
                break
        if len(chosen) != marker_counts[region]:
            return False
        for r, c in chosen:
            grid[r, c] = ORANGE
            occupied.add((r, c))
    return True


def _generator_semantics_ok(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze(grid, use_deflectors=True)
        straight = _analyze(grid, use_deflectors=False, require_unique=False)
    except AssertionError:
        return False

    active_regions = [region for region, count in analysis["votes"].items() if count > 0]
    if len(active_regions) < 3:
        return False

    if not any(len(trace["column_bands"]) >= 2 for trace in analysis["traces"]):
        return False

    changed_paths = sum(
        trace["regions"] != straight_trace["regions"]
        for trace, straight_trace in zip(analysis["traces"], straight["traces"])
    )
    if changed_paths == 0:
        return False

    straight_winner = straight["winner"]
    if straight_winner is not None and straight_winner == analysis["winner"]:
        return False

    out = main(grid)
    expected = _render_output(grid, analysis)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False

    winner = analysis["winner"]
    for region in analysis["region_ids"]:
        r0, r1 = analysis["row_spans"][region[0]]
        c0, c1 = analysis["col_spans"][region[1]]
        block = out[r0 : r1 + 1, c0 : c1 + 1]
        if region == winner:
            non_edge = block.copy()
            if region[0] == 0:
                non_edge[0, :] = np.where(non_edge[0, :] == TEAL, ORANGE, non_edge[0, :])
            if region[1] == 0:
                non_edge[:, 0] = np.where(non_edge[:, 0] == RED, ORANGE, non_edge[:, 0])
            if not np.all(non_edge == ORANGE):
                return False
        else:
            allowed = {BLACK}
            if region[0] == 0:
                allowed.add(TEAL)
            if region[1] == 0:
                allowed.add(RED)
            if not set(map(int, np.unique(block))).issubset(allowed):
                return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w, row_spans, col_spans, divider_rows, divider_cols = _sample_partition(rng)
    divider_row = divider_rows[0]

    deflector_rows = _sample_deflectors(rng, h, divider_row)
    if deflector_rows is None:
        return None

    sources = _sample_sources(rng, w, divider_cols, len(deflector_rows))
    if sources is None:
        return None

    source_bands = {_band_index(c, col_spans) for c in sources}
    if len(source_bands) < 2:
        return None

    grid = _build_base_grid(h, w, divider_rows, divider_cols, sources, deflector_rows)
    try:
        analysis = _analyze(grid, use_deflectors=True)
    except AssertionError:
        return None

    if not _place_markers(rng, grid, analysis):
        return None
    if not _generator_semantics_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
