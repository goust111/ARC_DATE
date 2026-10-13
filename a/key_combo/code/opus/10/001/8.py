from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4)
MAX_GEN_RETRIES = 500


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_segments(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    segments: list[dict[str, int]] = []
    for r in range(g.shape[0]):
        cols = np.flatnonzero(g[r] == GREY)
        if cols.size == 0:
            continue
        start = int(cols[0])
        end = int(cols[-1])
        assert cols.size == end - start + 1, "each grey object must be one horizontal segment"
        segments.append({"row": int(r), "start": start, "end": end, "length": end - start + 1})
    assert len(segments) >= 3, "expected multiple horizontal segments"
    return sorted(segments, key=lambda seg: seg["row"])


def _build_aligned_grid(grid: np.ndarray, segments: list[dict[str, int]], ref_idx: int) -> tuple[np.ndarray, dict[str, int]]:
    ref = segments[ref_idx]
    out = np.asarray(grid, dtype=int).copy()
    out[out == GREY] = BG
    for seg in segments:
        out[seg["row"], ref["start"] : ref["end"] + 1] = GREY
    return out, ref


def _analyze_reference(
    grid: np.ndarray,
    ref_idx: int,
    *,
    fill_open: bool = False,
) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    segments = _find_segments(g)
    aligned, ref = _build_aligned_grid(g, segments, ref_idx)

    blocked = aligned == GREY
    top_row = segments[0]["row"]
    bottom_row = segments[-1]["row"]
    blocked[top_row : bottom_row + 1, ref["start"]] = True
    blocked[top_row : bottom_row + 1, ref["end"]] = True

    comps = _components4(~blocked)
    out = aligned.copy()
    comp_info: list[dict[str, object]] = []
    for comp in comps:
        touches_edge = any(
            r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in comp
        )
        seeds = [(r, c) for r, c in comp if aligned[r, c] not in (BG, GREY)]
        info = {
            "cells": comp,
            "touches_edge": touches_edge,
            "seeds": seeds,
            "seed_count": len(seeds),
        }
        comp_info.append(info)
        if (fill_open or not touches_edge) and len(seeds) == 1:
            color = int(aligned[seeds[0]])
            for r, c in comp:
                out[r, c] = color

    return {
        "output": out,
        "segments": segments,
        "ref": ref,
        "ref_idx": ref_idx,
        "aligned": aligned,
        "blocked": blocked,
        "components": comp_info,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the horizontal grey segments to the median segment and flood-fill only the uniquely seeded enclosed bands.

    Input structure:
    - The background is black.
    - Several grey horizontal line segments lie on distinct rows with different start/end columns.
    - Isolated colored seed pixels sit in the black gaps between segments or in the open area above/below them.

    Rule:
    - Detect all grey horizontal segments and sort them by row; the median-row segment is the reference.
    - Replace every segment by a grey segment on the same row whose start and end match the reference segment.
    - Treat the common start/end columns as two virtual vertical walls together with the aligned grey lines, then find the resulting 4-connected empty regions.
    - Flood-fill each enclosed region that contains exactly one seed with that seed's color; do not fill open regions or enclosed regions with zero or multiple seeds.

    Key insight:
    - The aligned segments are only an intermediate fence: the median segment determines the hidden vertical walls, and those walls decide which seeds end up inside fillable cavities.
    """
    segments = _find_segments(np.asarray(grid, dtype=int))
    ref_idx = len(segments) // 2
    return np.asarray(_analyze_reference(grid, ref_idx)["output"], dtype=int)


def _make_rows(n_segments: int, gaps: list[int], top_margin: int) -> list[int]:
    rows = [top_margin]
    for gap in gaps:
        rows.append(rows[-1] + gap + 1)
    assert len(rows) == n_segments
    return rows


def _candidate_deltas(n_segments: int) -> list[tuple[int, int]]:
    mandatory = [(-2, 0), (0, 2), (2, 0), (0, -2)]
    extras = [(-1, 1), (1, -1), (-2, 1), (1, -2), (-1, 2), (2, -1)]
    need = n_segments - 1 - len(mandatory)
    assert need >= 0
    return mandatory + random.sample(extras, need)


def _place_seed(
    grid: np.ndarray,
    pos: tuple[int, int],
    color: int,
    all_positions: set[tuple[int, int]],
) -> bool:
    r, c = pos
    if grid[r, c] != BG:
        return False
    for rr, cc in all_positions:
        if max(abs(rr - r), abs(cc - c)) <= 1:
            return False
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == GREY:
            return False
    grid[r, c] = color
    all_positions.add(pos)
    return True


def _build_attempt() -> np.ndarray | None:
    n_segments = random.choice([5, 7])
    gaps = [3] * (n_segments - 1)
    max_twos = 1 if n_segments == 5 else 2
    for idx in random.sample(range(len(gaps)), random.randint(0, max_twos)):
        gaps[idx] = 2

    top_margin = 2
    bottom_margin = 2 if n_segments == 7 else random.choice([2, 3])
    height = top_margin + 1 + sum(gap + 1 for gap in gaps) + bottom_margin
    if height > 30:
        return None

    width = random.randint(18, 24)
    ref_len = random.randint(8, 11)
    ref_start = random.randint(3, width - ref_len - 4)
    ref_end = ref_start + ref_len - 1

    rows = _make_rows(n_segments, gaps, top_margin)
    ref_idx = n_segments // 2
    spans = [(ref_start, ref_end)] * n_segments
    deltas = _candidate_deltas(n_segments)
    nonref_rows = [idx for idx in range(n_segments) if idx != ref_idx]
    random.shuffle(deltas)
    used_spans = {(ref_start, ref_end)}
    for idx, (ds, de) in zip(nonref_rows, deltas):
        start = ref_start + ds
        end = ref_end + de
        if start < 1 or end > width - 2 or end - start + 1 < 5 or end - start + 1 > 15:
            return None
        if (start, end) in used_spans:
            return None
        spans[idx] = (start, end)
        used_spans.add((start, end))

    grid = np.zeros((height, width), dtype=int)
    for row, (start, end) in zip(rows, spans):
        grid[row, start : end + 1] = GREY

    safe_gaps = [idx for idx, gap in enumerate(gaps) if gap >= 3]
    if len(safe_gaps) < 3:
        return None
    fill_left_gap, fill_right_gap, outside_gap = random.sample(safe_gaps, 3)
    remaining_safe = [idx for idx in safe_gaps if idx not in (fill_left_gap, fill_right_gap, outside_gap)]
    branch_gap = random.choice(remaining_safe) if remaining_safe else None
    branch_mode = random.choice(["multi", "empty"]) if branch_gap is not None else "empty"

    seed_positions: set[tuple[int, int]] = set()
    inside_left = ref_start + random.choice([1, 2])
    inside_right = ref_end - random.choice([1, 2])
    outside_side = random.choice(["left", "right"])
    outside_col = ref_start - 1 if outside_side == "left" else ref_end + 1

    placements: list[tuple[tuple[int, int], int]] = []
    placements.append(((rows[fill_left_gap] + 2, inside_left), random.choice(SEED_COLORS)))
    placements.append(((rows[fill_right_gap] + 2, inside_right), random.choice(SEED_COLORS)))
    placements.append(((rows[outside_gap] + 2, outside_col), random.choice(SEED_COLORS)))

    if branch_gap is not None and branch_mode == "multi":
        left_multi = ref_start + 3
        right_multi = ref_end - 3
        if right_multi - left_multi < 2:
            return None
        placements.append(((rows[branch_gap] + 2, left_multi), random.choice(SEED_COLORS)))
        placements.append(((rows[branch_gap] + 2, right_multi), random.choice(SEED_COLORS)))

    open_above = random.choice([True, False])
    open_col = random.randint(ref_start + 2, ref_end - 2)
    open_row = 0 if open_above else height - 1
    placements.append(((open_row, open_col), random.choice(SEED_COLORS)))

    for pos, color in placements:
        if not _place_seed(grid, pos, color, seed_positions):
            return None

    analysis = _analyze_reference(grid, ref_idx)
    components = analysis["components"]
    filled_closed = [comp for comp in components if not comp["touches_edge"] and comp["seed_count"] == 1]
    nonfilled_closed = [comp for comp in components if not comp["touches_edge"] and comp["seed_count"] != 1]
    open_seeded = [comp for comp in components if comp["touches_edge"] and comp["seed_count"] >= 1]
    if len(filled_closed) < 2 or not nonfilled_closed or not open_seeded:
        return None

    segs = analysis["segments"]
    ref_length = segs[ref_idx]["length"]
    lengths = [seg["length"] for seg in segs]
    if not (min(lengths) < ref_length < max(lengths)):
        return None

    aligned = np.asarray(analysis["aligned"], dtype=int)
    if np.count_nonzero((aligned == GREY) & (grid != GREY)) == 0:
        return None
    if np.count_nonzero((grid == GREY) & (aligned != GREY)) == 0:
        return None

    correct_output = np.asarray(analysis["output"], dtype=int)
    for alt_idx in range(len(segs)):
        if alt_idx == ref_idx:
            continue
        alt_output = np.asarray(_analyze_reference(grid, alt_idx)["output"], dtype=int)
        if np.array_equal(alt_output, correct_output):
            return None

    if not np.array_equal(main(grid), correct_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
