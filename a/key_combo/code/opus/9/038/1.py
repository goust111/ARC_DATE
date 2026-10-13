from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
STRIPE = 1
OBSTACLE = 7
SEGMENT_COLORS = (2, 3, 4, 5, 6, 9)

MIN_SEGMENTS = 4
MAX_SEGMENTS = 5
MIN_OBSTACLES = 2
MAX_OBSTACLES = 3
MIN_HEIGHT = 14
MAX_HEIGHT = 18
MIN_WIDTH = 13
MAX_WIDTH = 18
MIN_LENGTH = 3
MAX_LENGTH = 8
MAX_GENERATION_ATTEMPTS = 500


def _neighbors8_inclusive(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            yield r + dr, c + dc


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _sample_spaced_rows(lo: int, hi: int, count: int, min_gap: int) -> list[int]:
    usable = hi - lo + 1 - (min_gap - 1) * (count - 1)
    assert usable >= count
    base = sorted(random.sample(range(usable), count))
    return [lo + base[i] + i * (min_gap - 1) for i in range(count)]


def _extract_segments(grid: np.ndarray, stripe_row: int) -> list[dict]:
    segments = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BACKGROUND, STRIPE, OBSTACLE):
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        rows = np.unique(coords[:, 0])
        assert rows.size == 1, "each moving object must be a horizontal segment"
        row = int(rows[0])
        assert row < stripe_row, "segments must be above the stripe"
        cols = np.sort(coords[:, 1])
        assert int(cols[-1] - cols[0] + 1) == int(cols.size), "segment must be contiguous"
        segments.append(
            {
                "color": color,
                "row": row,
                "c0": int(cols[0]),
                "c1": int(cols[-1]),
                "length": int(cols.size),
                "coords": coords.astype(int),
            }
        )
    segments.sort(key=lambda seg: seg["row"])
    return segments


def _build_forbidden(shape: tuple[int, int], obstacles: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for r, c in obstacles:
        for nr, nc in _neighbors8_inclusive(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                forbidden[nr, nc] = True
    return forbidden


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert max(h, w) <= 30
    assert np.all(g[-1] == STRIPE), "bottom row must be the full-width stripe"
    stripe_row = h - 1

    obstacles = [tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE)]
    assert MIN_OBSTACLES <= len(obstacles) <= MAX_OBSTACLES
    assert all(r < stripe_row for r, _c in obstacles)
    for i, a in enumerate(obstacles):
        for b in obstacles[i + 1 :]:
            assert _chebyshev(a, b) >= 3, "obstacle forbidden zones must not overlap"

    segments = _extract_segments(g, stripe_row)
    assert MIN_SEGMENTS <= len(segments) <= MAX_SEGMENTS
    lengths = [seg["length"] for seg in segments]
    assert len(set(lengths)) == len(lengths), "segment lengths must be unique"

    occupied = set()
    for seg in segments:
        coords = [tuple(map(int, rc)) for rc in seg["coords"]]
        for r, c in coords:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) in occupied and (rr, cc) not in coords:
                        raise AssertionError("segments must not touch each other")
        occupied.update(coords)

    forbidden = _build_forbidden(g.shape, obstacles)
    safe_lengths = []
    covered_counts = []
    for seg in segments:
        coords = seg["coords"]
        safe = int(np.sum(~forbidden[coords[:, 0], coords[:, 1]]))
        covered = int(seg["length"] - safe)
        seg["safe_length"] = safe
        seg["covered_count"] = covered
        safe_lengths.append(safe)
        covered_counts.append(covered)

    assert len(set(safe_lengths)) == len(safe_lengths), "safe lengths must be unique"
    assert any(count > 0 for count in covered_counts), "at least one segment must be affected"
    return {
        "grid": g,
        "stripe_row": stripe_row,
        "obstacles": obstacles,
        "segments": segments,
        "safe_lengths": safe_lengths,
        "covered_counts": covered_counts,
    }


def _render_stacked(grid: np.ndarray, ranked_segments: list[dict], *, use_safe_lengths: bool) -> np.ndarray:
    out = np.array(grid, copy=True)
    stripe_row = grid.shape[0] - 1
    for seg in ranked_segments:
        coords = seg["coords"]
        out[coords[:, 0], coords[:, 1]] = BACKGROUND
    for idx, seg in enumerate(ranked_segments):
        row = stripe_row - 1 - idx
        length = int(seg["safe_length"] if use_safe_lengths else seg["length"])
        assert length >= 0
        c0 = grid.shape[1] - length
        target = out[row, c0:]
        assert not np.any(target == OBSTACLE), "generator must keep stack rows clear of obstacles"
        out[row, c0:] = int(seg["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recompute each scattered segment's safe length under orange 8-neighborhood exclusion, then stack the full segments by that ranking above the bottom stripe.

    Input structure:
    - A black background contains a blue full-width stripe on the bottom row, 4-5 scattered one-cell-thick horizontal segments of distinct colors above it, and 2-3 isolated orange pixels.
    - Each orange pixel is a blocker center; its own cell and 8-neighborhood define a forbidden zone.
    - The colored segments are disjoint, non-touching, and each segment has a unique original length.

    Rule:
    - Mark every orange pixel and its 8-neighborhood as forbidden.
    - For each horizontal segment, count how many of its own pixels are outside all forbidden zones; this is that segment's safe length.
    - Clear the segments from their original positions, keep the orange blockers and bottom stripe unchanged, then sort the segments by safe length from small to large, breaking ties by original row from top to bottom.
    - Place the sorted segments in consecutive rows immediately above the stripe, with the smallest-safe segment closest to the stripe and every segment right-aligned to the grid edge; place each segment at its full original length, not truncated.

    Key insight:
    - The orange blockers do not cut the segments in the output; they only change the measurement used to rank otherwise intact segments before the final stack is built.
    """
    info = _analyze(grid)
    ranked = sorted(info["segments"], key=lambda seg: (seg["safe_length"], seg["row"]))
    return _render_stacked(info["grid"], ranked, use_safe_lengths=False)


def _wrong_sort_by_original_length(grid: np.ndarray) -> np.ndarray:
    info = _analyze(grid)
    ranked = sorted(info["segments"], key=lambda seg: (seg["length"], seg["row"]))
    return _render_stacked(info["grid"], ranked, use_safe_lengths=False)


def _wrong_sort_by_covered_pixels(grid: np.ndarray) -> np.ndarray:
    info = _analyze(grid)
    ranked = sorted(info["segments"], key=lambda seg: (-seg["covered_count"], seg["row"]))
    return _render_stacked(info["grid"], ranked, use_safe_lengths=False)


def _wrong_use_truncated_segments(grid: np.ndarray) -> np.ndarray:
    info = _analyze(grid)
    ranked = sorted(info["segments"], key=lambda seg: (seg["safe_length"], seg["row"]))
    return _render_stacked(info["grid"], ranked, use_safe_lengths=True)


def _validate_instance(grid: np.ndarray) -> None:
    info = _analyze(grid)
    lengths = [seg["length"] for seg in info["segments"]]
    safe = [seg["safe_length"] for seg in info["segments"]]
    covered = [seg["covered_count"] for seg in info["segments"]]

    correct = main(grid)
    assert not np.array_equal(correct, _wrong_sort_by_original_length(grid)), "obstacles must change the ranking"
    assert not np.array_equal(correct, _wrong_sort_by_covered_pixels(grid)), "original lengths must matter in the ranking"
    assert not np.array_equal(correct, _wrong_use_truncated_segments(grid)), "output must keep full segment lengths"
    assert len(set(safe)) == len(safe)
    assert len(set(lengths)) == len(lengths)
    assert sum(count > 0 for count in covered) >= 2, "at least two segments should participate in the exclusion logic"

    stripe_row = grid.shape[0] - 1
    stack_rows = {stripe_row - 1 - i for i in range(len(info["segments"]))}
    for r, _c in info["obstacles"]:
        assert r not in stack_rows, "keep blocker pixels away from output stack rows"


def _sample_segments(height: int, width: int, n_segments: int) -> list[dict]:
    stripe_row = height - 1
    stack_top = stripe_row - n_segments
    input_row_hi = stack_top - 2
    rows = _sample_spaced_rows(1, input_row_hi, n_segments, min_gap=2)

    max_len = min(MAX_LENGTH, width - 3)
    lengths = random.sample(range(MIN_LENGTH, max_len + 1), n_segments)
    colors = random.sample(SEGMENT_COLORS, n_segments)
    segments = []
    for row, length, color in zip(rows, lengths, colors):
        c0 = random.randint(0, width - length)
        coords = np.array([(row, c) for c in range(c0, c0 + length)], dtype=int)
        segments.append(
            {
                "color": int(color),
                "row": int(row),
                "c0": int(c0),
                "c1": int(c0 + length - 1),
                "length": int(length),
                "coords": coords,
            }
        )
    return segments


def _candidate_obstacles(grid: np.ndarray, segments: list[dict]) -> list[dict]:
    h, w = grid.shape
    stripe_row = h - 1
    candidates = []
    for r in range(h - 1):
        for c in range(w):
            if grid[r, c] != BACKGROUND:
                continue
            influence = []
            for seg in segments:
                coords = seg["coords"]
                touched = int(np.sum(np.max(np.abs(coords - np.array([r, c])), axis=1) <= 1))
                influence.append(touched)
            if sum(influence) == 0:
                continue
            candidates.append({"pos": (r, c), "influence": tuple(influence)})
    random.shuffle(candidates)
    assert stripe_row > 0
    return candidates


def _choose_obstacles(grid: np.ndarray, segments: list[dict], n_obstacles: int) -> list[tuple[int, int]] | None:
    candidates = _candidate_obstacles(grid, segments)
    if len(candidates) < n_obstacles:
        return None

    for _ in range(250):
        chosen = []
        for cand in candidates:
            if all(_chebyshev(cand["pos"], prev["pos"]) >= 3 for prev in chosen):
                chosen.append(cand)
                if len(chosen) == n_obstacles:
                    break
        if len(chosen) != n_obstacles:
            random.shuffle(candidates)
            continue

        coverage = [0] * len(segments)
        for cand in chosen:
            for idx, touched in enumerate(cand["influence"]):
                coverage[idx] += int(touched)
        safe = [segments[idx]["length"] - coverage[idx] for idx in range(len(segments))]

        if min(safe) < 1:
            random.shuffle(candidates)
            continue
        if len(set(safe)) != len(safe):
            random.shuffle(candidates)
            continue
        if sum(value > 0 for value in coverage) < 2:
            random.shuffle(candidates)
            continue

        safe_order = sorted(range(len(segments)), key=lambda i: (safe[i], segments[i]["row"]))
        length_order = sorted(range(len(segments)), key=lambda i: (segments[i]["length"], segments[i]["row"]))
        coverage_order = sorted(range(len(segments)), key=lambda i: (-coverage[i], segments[i]["row"]))
        if safe_order == length_order or safe_order == coverage_order:
            random.shuffle(candidates)
            continue

        return [cand["pos"] for cand in chosen]
    return None


def _render_input(height: int, width: int, segments: list[dict], obstacles: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[-1, :] = STRIPE
    for seg in segments:
        grid[seg["row"], seg["c0"] : seg["c1"] + 1] = int(seg["color"])
    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    return grid


def _generate_attempt() -> np.ndarray | None:
    n_segments = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    n_obstacles = random.randint(MIN_OBSTACLES, MAX_OBSTACLES)
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)

    stripe_row = height - 1
    stack_top = stripe_row - n_segments
    input_row_hi = stack_top - 2
    if input_row_hi - 1 + 1 < n_segments + (n_segments - 1):
        return None

    segments = _sample_segments(height, width, n_segments)
    temp = _render_input(height, width, segments, [])
    obstacles = _choose_obstacles(temp, segments, n_obstacles)
    if obstacles is None:
        return None

    grid = _render_input(height, width, segments, obstacles)
    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-safe-segment instance")
