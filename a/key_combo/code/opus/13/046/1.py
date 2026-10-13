from __future__ import annotations

import numpy as np


BG = 0
SEP = 5
CTRL = 7
BAR_COLORS = (1, 2, 3)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 500


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 1, "expected exactly one full-height separator column"
    return int(cols[0])


def _extract_bars(grid: np.ndarray, sep_col: int) -> list[dict]:
    bars: list[dict] = []
    for r in range(grid.shape[0]):
        runs = []
        c = 0
        while c < sep_col:
            value = int(grid[r, c])
            if value in (BG, SEP, CTRL):
                c += 1
                continue
            c0 = c
            while c < sep_col and int(grid[r, c]) == value:
                c += 1
            runs.append((c0, c - 1, value))
        assert len(runs) <= 1, "expected at most one left-side bar per row"
        if runs:
            c0, c1, color = runs[0]
            bars.append(
                {
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "length": int(c1 - c0 + 1),
                    "color": int(color),
                    "dist": int((sep_col - 1) - c1),
                }
            )
    bars.sort(key=lambda bar: (bar["dist"], bar["row"], bar["c0"]))
    return bars


def _ghost_segments(length: int, sep_col: int, width: int) -> list[tuple[int, int]]:
    """
    Starting just right of the separator, repeat a filled block of length L followed by
    an empty gap of length L. This makes the hidden extension depend on bar length.
    """
    segments = []
    start = sep_col + 1
    while start < width:
        end = min(width, start + length)
        if end - start == length:
            segments.append((int(start), int(end)))
        start += 2 * length
    return segments


def _build_ghost_grid(bars: list[dict], shape: tuple[int, int], sep_col: int) -> np.ndarray:
    ghost = np.zeros(shape, dtype=int)
    width = int(shape[1])
    for bar in bars:
        r = int(bar["row"])
        color = int(bar["color"])
        for start, end in _ghost_segments(int(bar["length"]), sep_col, width):
            segment = ghost[r, start:end]
            mask = segment == BG
            if np.any(mask):
                segment = segment.copy()
                segment[mask] = color
                ghost[r, start:end] = segment
    return ghost


def _build_solid_shortcut_ghost(bars: list[dict], shape: tuple[int, int], sep_col: int) -> np.ndarray:
    ghost = np.zeros(shape, dtype=int)
    for bar in bars:
        ghost[int(bar["row"]), sep_col + 1 :] = int(bar["color"])
    return ghost


def _cast_ray(
    grid: np.ndarray,
    ghost: np.ndarray,
    sep_col: int,
    row: int,
    col: int,
    dr: int,
    dc: int,
) -> tuple[int, int, int] | None:
    height, width = grid.shape
    r = int(row) + int(dr)
    c = int(col) + int(dc)
    while 0 <= r < height and 0 <= c < width:
        if c == sep_col:
            return None
        if c < sep_col and int(grid[r, c]) not in (BG,):
            return None
        hit_color = int(ghost[r, c])
        if hit_color != BG:
            return int(r), int(c), hit_color
        r += int(dr)
        c += int(dc)
    return None


def _control_hits(
    grid: np.ndarray,
    ghost: np.ndarray,
    sep_col: int,
    row: int,
    col: int,
) -> list[tuple[int, int, int]]:
    hits = []
    for dr, dc in DIRECTIONS:
        hit = _cast_ray(grid, ghost, sep_col, row, col, dr, dc)
        if hit is not None:
            hits.append(hit)
    return hits


def _controls_sorted(grid: np.ndarray, sep_col: int) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == CTRL) if int(c) > sep_col]
    return sorted(coords)


def _render_from_ghost(grid: np.ndarray, ghost: np.ndarray, sep_col: int, protect_centers: bool) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    protected = {(int(r), int(c)) for r, c in np.argwhere(out == CTRL)}
    for r, c in _controls_sorted(out, sep_col):
        for hr, hc, color in _control_hits(out, ghost, sep_col, r, c):
            for rr in range(hr - 1, hr + 2):
                for cc in range(hc - 1, hc + 2):
                    if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                        if protect_centers and (rr, cc) in protected:
                            continue
                        out[rr, cc] = int(color)
            out[hr, hc] = CTRL
            if protect_centers:
                protected.add((int(hr), int(hc)))
    return out


def _render_with_rules(grid: np.ndarray, bars: list[dict], sep_col: int) -> np.ndarray:
    return _render_from_ghost(grid, _build_ghost_grid(bars, grid.shape, sep_col), sep_col, True)


def _render_with_solid_shortcut(grid: np.ndarray, bars: list[dict], sep_col: int) -> np.ndarray:
    return _render_from_ghost(grid, _build_solid_shortcut_ghost(bars, grid.shape, sep_col), sep_col, True)


def _render_without_center_protection(grid: np.ndarray, bars: list[dict], sep_col: int) -> np.ndarray:
    return _render_from_ghost(grid, _build_ghost_grid(bars, grid.shape, sep_col), sep_col, False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays from orange controls into hidden periodic ghost bars and stamp each hit with the hit bar's color.

    Input structure:
    - A black background contains one full-height grey separator column that splits the grid.
    - The left side has 2-3 single-row colored bars, each on its own row, with distinct colors and lengths.
    - The right side has isolated orange control pixels and no visible obstacle bars.

    Rule:
    - For each left-side bar of length L, imagine an invisible right-side extension on the same row:
      starting at the first cell right of the separator, repeat a filled run of length L followed by a gap of length L.
      Bars are processed from nearest to farthest from the separator and later bars cannot overwrite earlier ghost cells.
    - From every orange control, cast rays up, down, left, and right. A ray stops at the first ghost cell it meets.
      If it reaches the separator, the grid edge, or the original left-side bars before any ghost cell, that direction is invalid.
    - At each valid hit, draw a 3x3 stamp in the hit bar's color, keep the stamp center orange, and never let later stamps
      overwrite any orange center. Control pixels are processed from top to bottom, then left to right.

    Key insight:
    - The visible output depends on invisible periodic copies of the left bars: the control rays do not target the original bars,
      only the mentally reconstructed ghost pattern on the blank side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_col = _find_separator(g)
    bars = _extract_bars(g, sep_col)
    assert 2 <= len(bars) <= 3
    return _render_with_rules(g, bars, sep_col)


def _sample_rows(rng: np.random.Generator, height: int, count: int) -> list[int] | None:
    rows = list(range(1, height - 1))
    for _ in range(200):
        picked = sorted(int(x) for x in rng.choice(rows, size=count, replace=False))
        if all(b - a >= 3 for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _pick_lengths_gaps(
    rng: np.random.Generator, count: int, sep_col: int
) -> tuple[list[int], list[int]] | None:
    lengths = list(map(int, rng.choice(np.array([2, 3, 4, 5]), size=count, replace=False)))
    for _ in range(100):
        gaps = list(map(int, rng.choice(np.array([0, 1, 2, 3]), size=count, replace=False)))
        if all(gap <= sep_col - length for gap, length in zip(gaps, lengths)):
            return lengths, gaps
    return None


def _bar_setup(rng: np.random.Generator) -> tuple[np.ndarray, int, list[dict]] | None:
    height = int(rng.integers(14, 20))
    width = int(rng.integers(18, 25))
    n_bars = 2 if rng.random() < 0.45 else 3
    sep_col = int(rng.integers(6, min(9, width - 8) + 1))

    rows = _sample_rows(rng, height, n_bars)
    if rows is None:
        return None

    picked = _pick_lengths_gaps(rng, n_bars, sep_col)
    if picked is None:
        return None
    lengths, gaps = picked
    colors = list(map(int, rng.choice(np.array(BAR_COLORS), size=n_bars, replace=False)))

    grid = np.zeros((height, width), dtype=int)
    grid[:, sep_col] = SEP
    bars = []
    for row, length, gap, color in zip(rows, lengths, gaps, colors):
        c1 = sep_col - gap - 1
        c0 = c1 - length + 1
        if c0 < 0:
            return None
        grid[row, c0 : c1 + 1] = color
        bars.append(
            {
                "row": int(row),
                "c0": int(c0),
                "c1": int(c1),
                "length": int(length),
                "color": int(color),
                "dist": int(gap),
            }
        )

    if grid.shape[1] - sep_col - 1 < max(lengths):
        return None
    return grid, sep_col, sorted(bars, key=lambda bar: (bar["dist"], bar["row"], bar["c0"]))


def _full_row_model_hits(bars: list[dict], row: int, col: int) -> list[tuple[int, int, int]]:
    above = [bar for bar in bars if int(bar["row"]) < row]
    below = [bar for bar in bars if int(bar["row"]) > row]
    hits = []
    if above:
        bar = max(above, key=lambda item: int(item["row"]))
        hits.append((int(bar["row"]), int(col), int(bar["color"])))
    if below:
        bar = min(below, key=lambda item: int(item["row"]))
        hits.append((int(bar["row"]), int(col), int(bar["color"])))
    return hits


def _control_candidate(
    base: np.ndarray,
    ghost: np.ndarray,
    bars: list[dict],
    sep_col: int,
    row: int,
    col: int,
) -> dict | None:
    hits = _control_hits(base, ghost, sep_col, row, col)
    if not hits:
        return None

    centers = {(int(hr), int(hc)) for hr, hc, _color in hits}
    stamp_cells: set[tuple[int, int]] = set()
    clipped = False
    colors = {int(color) for _hr, _hc, color in hits}
    for hr, hc, _color in hits:
        for rr in range(hr - 1, hr + 2):
            for cc in range(hc - 1, hc + 2):
                if 0 <= rr < base.shape[0] and 0 <= cc < base.shape[1]:
                    stamp_cells.add((int(rr), int(cc)))
                else:
                    clipped = True

    full_hits = _full_row_model_hits(bars, row, col)
    periodic_witness = centers != {(int(hr), int(hc)) for hr, hc, _color in full_hits}
    return {
        "pos": (int(row), int(col)),
        "hits": hits,
        "hit_centers": centers,
        "stamp_cells": stamp_cells,
        "hit_colors": colors,
        "two_hit": len(hits) >= 2,
        "clipped": clipped,
        "periodic_witness": periodic_witness,
    }


def _enumerate_candidates(base: np.ndarray, bars: list[dict], sep_col: int) -> list[dict]:
    ghost = _build_ghost_grid(bars, base.shape, sep_col)
    bar_rows = {int(bar["row"]) for bar in bars}
    candidates = []
    for r in range(base.shape[0]):
        if r in bar_rows:
            continue
        for c in range(sep_col + 1, base.shape[1]):
            cand = _control_candidate(base, ghost, bars, sep_col, r, c)
            if cand is not None:
                candidates.append(cand)
    return candidates


def _controls_compatible(selected: list[dict], candidate: dict) -> bool:
    r, c = candidate["pos"]
    for other in selected:
        ro, co = other["pos"]
        if max(abs(r - ro), abs(c - co)) <= 1:
            return False
        if candidate["pos"] in other["stamp_cells"] or other["pos"] in candidate["stamp_cells"]:
            return False
        if candidate["hit_centers"] & other["hit_centers"]:
            return False
    return True


def _selection_has_overlap(selected: list[dict]) -> bool:
    for i, left in enumerate(selected):
        for right in selected[i + 1 :]:
            if left["stamp_cells"] & right["stamp_cells"]:
                return True
    return False


def _choose_controls(
    rng: np.random.Generator,
    candidates: list[dict],
    bars: list[dict],
    want_overlap: bool,
    want_clipped: bool,
) -> list[dict] | None:
    if not candidates:
        return None
    max_controls = min(4, len(candidates))
    counts = [n for n in (2, 3, 4) if n <= max_controls]
    if not counts:
        return None
    weighted_counts = [n for n in (2, 2, 2, 3, 3, 4) if n in counts]

    for _ in range(700):
        target = int(rng.choice(np.array(weighted_counts)))
        selected: list[dict] = []
        while len(selected) < target:
            choices = [cand for cand in candidates if cand not in selected and _controls_compatible(selected, cand)]
            if not choices:
                break

            have_two_hit = any(cand["two_hit"] for cand in selected)
            have_witness = any(cand["periodic_witness"] for cand in selected)
            have_clipped = any(cand["clipped"] for cand in selected)
            have_overlap = _selection_has_overlap(selected)

            def score(cand: dict) -> tuple[int, int, int]:
                score_value = 0
                if not have_two_hit and cand["two_hit"]:
                    score_value += 3
                if not have_witness and cand["periodic_witness"]:
                    score_value += 3
                if want_clipped and not have_clipped and cand["clipped"]:
                    score_value += 2
                if want_overlap and not have_overlap:
                    overlap_count = sum(bool(cand["stamp_cells"] & other["stamp_cells"]) for other in selected)
                    score_value += 2 * overlap_count
                color_gain = len(set().union(*(other["hit_colors"] for other in selected), cand["hit_colors"]))
                hit_gain = len(cand["hits"])
                return score_value, color_gain, hit_gain

            best = max(score(cand) for cand in choices)
            best_choices = [cand for cand in choices if score(cand) == best]
            selected.append(best_choices[int(rng.integers(len(best_choices)))])
        if len(selected) != target:
            continue

        hit_colors = set().union(*(cand["hit_colors"] for cand in selected))
        total_hits = sum(len(cand["hits"]) for cand in selected)
        if len(hit_colors) < min(2, len(bars)):
            continue
        if total_hits < max(3, len(selected)):
            continue
        if not any(cand["two_hit"] for cand in selected):
            continue
        if not any(cand["periodic_witness"] for cand in selected):
            continue
        if want_overlap and not _selection_has_overlap(selected):
            continue
        if want_clipped and not any(cand["clipped"] for cand in selected):
            continue
        return sorted(selected, key=lambda cand: cand["pos"])
    return None


def _install_controls(base: np.ndarray, selected: list[dict]) -> np.ndarray:
    grid = np.array(base, copy=True)
    for cand in selected:
        r, c = cand["pos"]
        grid[r, c] = CTRL
    return grid


def _semantic_checks(inp: np.ndarray, bars: list[dict], sep_col: int, selected: list[dict]) -> bool:
    actual = main(inp)
    expected = _render_with_rules(inp, bars, sep_col)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, inp):
        return False

    solid_shortcut = _render_with_solid_shortcut(inp, bars, sep_col)
    if np.array_equal(actual, solid_shortcut):
        return False

    overlap_required = _selection_has_overlap(selected)
    if overlap_required:
        no_protect = _render_without_center_protection(inp, bars, sep_col)
        if np.array_equal(actual, no_protect):
            return False

    ghost = _build_ghost_grid(bars, inp.shape, sep_col)
    all_hits = []
    for cand in selected:
        hits = _control_hits(inp, ghost, sep_col, *cand["pos"])
        if hits != cand["hits"]:
            return False
        all_hits.extend(hits)

    new_orange = sum(1 for r, c, _color in all_hits if inp[r, c] != CTRL)
    if new_orange < 2:
        return False
    non_input_changes = np.argwhere(actual != inp)
    return bool(non_input_changes.shape[0] >= 8)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    base_setup = _bar_setup(rng)
    if base_setup is None:
        return None
    base, sep_col, bars = base_setup

    candidates = _enumerate_candidates(base, bars, sep_col)
    if len(candidates) < 2:
        return None

    want_overlap = bool(rng.random() < 0.6 and any(
        candidates[i]["stamp_cells"] & candidates[j]["stamp_cells"]
        for i in range(len(candidates))
        for j in range(i + 1, len(candidates))
    ))
    want_clipped = bool(rng.random() < 0.45 and any(cand["clipped"] for cand in candidates))

    selected = _choose_controls(rng, candidates, bars, want_overlap, want_clipped)
    if selected is None:
        return None

    inp = _install_controls(base, selected)
    if not _semantic_checks(inp, bars, sep_col, selected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ghost-bar stamp input")
