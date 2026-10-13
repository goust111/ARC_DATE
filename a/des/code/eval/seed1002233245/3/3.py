from __future__ import annotations

from collections import deque
import random

import numpy as np


SIDE_STEPS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {
    "U": "D",
    "D": "U",
    "L": "R",
    "R": "L",
}
CORNER_TO_LEGEND_POS = {
    frozenset({"R", "D"}): (0, 0),
    frozenset({"L", "D"}): (0, 1),
    frozenset({"R", "U"}): (1, 0),
    frozenset({"L", "U"}): (1, 1),
}

GEN_RETRIES = 400
GEN_SIZE_MIN = 12
GEN_SIZE_MAX = 20
GEN_SEGMENT_COUNTS = (3, 5)
GEN_GAP = 3


def _background(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r_raw, c_raw in np.argwhere(mask):
        r = int(r_raw)
        c = int(c_raw)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in SIDE_STEPS.values():
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _legend_lookup(g: np.ndarray) -> dict[frozenset[str], int]:
    assert g.shape[0] >= 2 and g.shape[1] >= 2, "grid must fit the 2x2 legend"
    legend = np.asarray(g[:2, :2], dtype=int)
    return {
        pair: int(legend[r, c])
        for pair, (r, c) in CORNER_TO_LEGEND_POS.items()
    }


def _extract_segments(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    mask = g != bg
    mask[:2, :2] = False
    comps = _components(mask)
    assert comps, "expected at least one path segment"

    segments: list[dict[str, object]] = []
    for seg_id, comp in enumerate(comps):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each segment must be monochrome"
        assert len(comp) >= 2, "segments must have length at least 2"

        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        color = next(iter(colors))

        if len(rows) == 1:
            row = rows[0]
            ordered_cols = sorted(c for _, c in comp)
            assert ordered_cols == list(range(ordered_cols[0], ordered_cols[-1] + 1))
            endpoints = [((row, ordered_cols[0]), "L"), ((row, ordered_cols[-1]), "R")]
        elif len(cols) == 1:
            col = cols[0]
            ordered_rows = sorted(r for r, _ in comp)
            assert ordered_rows == list(range(ordered_rows[0], ordered_rows[-1] + 1))
            endpoints = [((ordered_rows[0], col), "U"), ((ordered_rows[-1], col), "D")]
        else:
            raise AssertionError("each segment must be horizontal or vertical")

        segments.append(
            {
                "id": seg_id,
                "color": color,
                "cells": tuple(comp),
                "endpoints": tuple(endpoints),
            }
        )
    return segments


def _analyze_path(g: np.ndarray, bg: int) -> dict[tuple[int, int], frozenset[str]]:
    h, w = g.shape
    segments = _extract_segments(g, bg)
    border_counts = [0] * len(segments)
    adjacency: dict[int, list[int]] = {seg["id"]: [] for seg in segments}
    junction_links: dict[tuple[int, int], list[tuple[int, str]]] = {}

    for seg in segments:
        seg_id = int(seg["id"])
        for endpoint, outward in seg["endpoints"]:
            dr, dc = SIDE_STEPS[str(outward)]
            nr = endpoint[0] + dr
            nc = endpoint[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                assert int(g[nr, nc]) == bg, "segment endpoint extension must hit background"
                junction_links.setdefault((nr, nc), []).append((seg_id, OPPOSITE[str(outward)]))
            else:
                border_counts[seg_id] += 1
                assert (
                    endpoint[0] in (0, h - 1) or endpoint[1] in (0, w - 1)
                ), "only border endpoints may exit the grid"

    assert sum(border_counts) == 2, "path must have exactly two border terminals"
    junction_pairs: dict[tuple[int, int], frozenset[str]] = {}
    for cell, links in junction_links.items():
        assert len(links) == 2, "each junction must connect exactly two segment endpoints"
        (seg_a, side_a), (seg_b, side_b) = links
        assert seg_a != seg_b, "a junction cannot connect a segment to itself"
        pair = frozenset({side_a, side_b})
        assert pair in CORNER_TO_LEGEND_POS, "junction must connect one horizontal and one vertical side"
        junction_pairs[cell] = pair
        adjacency[seg_a].append(seg_b)
        adjacency[seg_b].append(seg_a)

    for seg_id in range(len(segments)):
        assert border_counts[seg_id] + len(adjacency[seg_id]) == 2, "each segment must have two terminals"

    start = next((seg_id for seg_id, cnt in enumerate(border_counts) if cnt == 1), 0)
    seen = {start}
    stack = [start]
    while stack:
        seg_id = stack.pop()
        for nxt in adjacency[seg_id]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    assert len(seen) == len(segments), "segments must form one connected chain"

    return junction_pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the colored straight segments through blank junction cells and
    color each junction by the bend-direction legend in the top-left 2x2 block.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    lookup = _legend_lookup(g)
    junction_pairs = _analyze_path(g, bg)

    out = g.copy()
    for (r, c), pair in junction_pairs.items():
        out[r, c] = lookup[pair]
    return out


def _sample_spaced_values(
    rng: random.Random, low: int, high: int, count: int, min_gap: int
) -> list[int] | None:
    if count == 0:
        return []
    span = high - low + 1
    needed = count + (count - 1) * (min_gap - 1)
    if span < needed:
        return None
    base_high = high - (count - 1) * (min_gap - 1)
    base = sorted(rng.sample(range(low, base_high + 1), count))
    return [base[i] + i * (min_gap - 1) for i in range(count)]


def _shuffle_turn_axis(rng: random.Random, values: list[int]) -> list[int]:
    seq = list(values)
    if len(seq) <= 2:
        rng.shuffle(seq)
        return seq

    require_both = rng.random() < 0.5
    for _ in range(24):
        candidate = list(seq)
        rng.shuffle(candidate)
        diffs = [candidate[i + 1] - candidate[i] for i in range(len(candidate) - 1)]
        if not require_both or (any(d > 0 for d in diffs) and any(d < 0 for d in diffs)):
            return candidate
    rng.shuffle(seq)
    return seq


def _build_horizontal_path(
    h: int, w: int, mode: str, segment_count: int, rng: random.Random
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    turns = (segment_count - 1) // 2
    rows = _sample_spaced_values(rng, 3, h - 3, turns + 1, GEN_GAP)
    cols = _sample_spaced_values(rng, 2, w - 3, turns, GEN_GAP)
    if rows is None or cols is None:
        return None

    rows = _shuffle_turn_axis(rng, rows)
    cols = cols if mode == "LR" else list(reversed(cols))

    segments: list[list[tuple[int, int]]] = []
    junctions: list[tuple[int, int]] = []

    if mode == "LR":
        segments.append([(rows[0], c) for c in range(0, cols[0])])
    else:
        segments.append([(rows[0], c) for c in range(cols[0] + 1, w)])
    junctions.append((rows[0], cols[0]))

    for i in range(turns):
        r0 = rows[i]
        r1 = rows[i + 1]
        c = cols[i]
        segments.append([(r, c) for r in range(min(r0, r1) + 1, max(r0, r1))])
        junctions.append((rows[i + 1], cols[i]))
        if i + 1 < turns:
            c0 = cols[i]
            c1 = cols[i + 1]
            segments.append([(rows[i + 1], c) for c in range(min(c0, c1) + 1, max(c0, c1))])
            junctions.append((rows[i + 1], cols[i + 1]))

    if mode == "LR":
        segments.append([(rows[turns], c) for c in range(cols[-1] + 1, w)])
    else:
        segments.append([(rows[turns], c) for c in range(0, cols[-1])])

    return segments, junctions


def _build_vertical_path(
    h: int, w: int, mode: str, segment_count: int, rng: random.Random
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    turns = (segment_count - 1) // 2
    cols = _sample_spaced_values(rng, 3, w - 3, turns + 1, GEN_GAP)
    rows = _sample_spaced_values(rng, 2, h - 3, turns, GEN_GAP)
    if cols is None or rows is None:
        return None

    cols = _shuffle_turn_axis(rng, cols)
    rows = rows if mode == "TB" else list(reversed(rows))

    segments: list[list[tuple[int, int]]] = []
    junctions: list[tuple[int, int]] = []

    if mode == "TB":
        segments.append([(r, cols[0]) for r in range(0, rows[0])])
    else:
        segments.append([(r, cols[0]) for r in range(rows[0] + 1, h)])
    junctions.append((rows[0], cols[0]))

    for i in range(turns):
        c0 = cols[i]
        c1 = cols[i + 1]
        r = rows[i]
        segments.append([(r, c) for c in range(min(c0, c1) + 1, max(c0, c1))])
        junctions.append((rows[i], cols[i + 1]))
        if i + 1 < turns:
            r0 = rows[i]
            r1 = rows[i + 1]
            segments.append([(r, cols[i + 1]) for r in range(min(r0, r1) + 1, max(r0, r1))])
            junctions.append((rows[i + 1], cols[i + 1]))

    if mode == "TB":
        segments.append([(r, cols[-1]) for r in range(rows[-1] + 1, h)])
    else:
        segments.append([(r, cols[-1]) for r in range(0, rows[-1])])

    return segments, junctions


def _render_case(
    h: int,
    w: int,
    bg: int,
    legend: np.ndarray,
    segments: list[list[tuple[int, int]]],
    junctions: list[tuple[int, int]],
    segment_colors: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    assert len(segments) == len(segment_colors)
    occupied = set()
    for seg in segments:
        assert seg, "generated segments must be non-empty"
        for cell in seg:
            if cell in occupied:
                return None
            occupied.add(cell)

    if any(cell in occupied for cell in junctions):
        return None
    if len(set(junctions)) != len(junctions):
        return None

    inp = np.full((h, w), bg, dtype=int)
    inp[:2, :2] = legend
    for color, seg in zip(segment_colors, segments):
        for r, c in seg:
            if not (0 <= r < h and 0 <= c < w):
                return None
            if (r < 2 and c < 2) or inp[r, c] != bg:
                return None
            inp[r, c] = color

    segment_set = set(occupied)
    lookup = _legend_lookup(inp)
    out = inp.copy()
    for jr, jc in junctions:
        sides = {
            side
            for side, (dr, dc) in SIDE_STEPS.items()
            if (jr + dr, jc + dc) in segment_set
        }
        pair = frozenset(sides)
        if pair not in lookup:
            return None
        out[jr, jc] = lookup[pair]

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    w = rng.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    segment_count = rng.choice(GEN_SEGMENT_COUNTS)
    mode = rng.choice(("LR", "RL", "TB", "BT"))

    if mode in ("LR", "RL"):
        built = _build_horizontal_path(h, w, mode, segment_count, rng)
    else:
        built = _build_vertical_path(h, w, mode, segment_count, rng)
    if built is None:
        return None
    segments, junctions = built

    bg = rng.randrange(10)
    palette = [color for color in range(10) if color != bg]
    chosen = rng.sample(palette, 4 + len(segments))
    legend = np.array(chosen[:4], dtype=int).reshape(2, 2)
    segment_colors = chosen[4:]

    rendered = _render_case(h, w, bg, legend, segments, junctions, segment_colors)
    if rendered is None:
        return None
    inp, out = rendered

    if _background(inp) != bg:
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
