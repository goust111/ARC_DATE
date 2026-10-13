from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500
_RNG = random.Random()


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _clockwise_perimeter(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _parse_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    non_bg = np.argwhere(g != bg)
    assert non_bg.size > 0, "expected at least one non-background pixel"

    r0, c0 = map(int, non_bg.min(axis=0))
    r1, c1 = map(int, non_bg.max(axis=0))
    assert r1 > r0 and c1 > c0, "expected a rectangular ring, not a line"

    track = _clockwise_perimeter(r0, c0, r1, c1)
    border_values = [int(g[r, c]) for r, c in track]
    assert all(v != bg for v in border_values), "ring perimeter must be fully painted"

    grey = int(Counter(border_values).most_common(1)[0][0])
    markers: list[tuple[int, int]] = []
    for idx, (r, c) in enumerate(track):
        value = int(g[r, c])
        if value != grey:
            markers.append((idx, value))
    assert markers, "expected at least one colored ring marker"

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_counts = Counter(int(v) for v in inner.flat if int(v) != bg)

    return {
        "bg": bg,
        "bbox": (r0, c0, r1, c1),
        "track": track,
        "grey": grey,
        "markers": markers,
        "inner_counts": inner_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored marker clockwise along a rectangular ring by the count of
    same-colored pixels inside the ring, then clear the ring interior.
    """
    g = np.asarray(grid, dtype=int)
    spec = _parse_puzzle(g)

    bg = int(spec["bg"])
    grey = int(spec["grey"])
    r0, c0, r1, c1 = spec["bbox"]
    track = spec["track"]
    markers = spec["markers"]
    inner_counts = spec["inner_counts"]
    cycle_len = len(track)

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = bg
    for r, c in track:
        out[r, c] = grey

    for idx, color in markers:
        dest_idx = (idx + int(inner_counts.get(int(color), 0))) % cycle_len
        dr, dc = track[dest_idx]
        out[dr, dc] = int(color)

    return out


def _pick_cycle_slots(length: int, count: int, rng: random.Random) -> list[int]:
    assert count <= length // 2, "cannot place that many non-adjacent markers on a cycle"
    for _ in range(200):
        order = list(range(length))
        rng.shuffle(order)
        chosen: list[int] = []
        chosen_set: set[int] = set()
        for idx in order:
            if idx in chosen_set:
                continue
            if ((idx - 1) % length) in chosen_set or ((idx + 1) % length) in chosen_set:
                continue
            chosen.append(idx)
            chosen_set.add(idx)
            if len(chosen) == count:
                return chosen
    raise RuntimeError("failed to place non-adjacent markers")


def _sample_counts(
    marker_indices: list[int],
    marker_colors: list[int],
    interior_size: int,
    cycle_len: int,
    rng: random.Random,
) -> dict[int, int]:
    counts = {int(color): 0 for color in marker_colors}
    modes = ["simple", "collision"]
    if interior_size > cycle_len:
        modes.append("wrap")
    mode = rng.choice(modes)

    used = 0
    extra_targets = list(marker_colors)

    if mode == "collision" and len(marker_colors) >= 2:
        order = sorted(range(len(marker_indices)), key=lambda i: marker_indices[i])
        gap_choices = []
        for i, a in enumerate(order):
            b = order[(i + 1) % len(order)]
            gap = (marker_indices[b] - marker_indices[a]) % cycle_len
            if 1 <= gap <= interior_size:
                gap_choices.append((gap, a, b))
        if gap_choices:
            best_gap = min(gap for gap, _, _ in gap_choices)
            near_best = [item for item in gap_choices if item[0] <= best_gap + 2]
            gap, a, b = rng.choice(near_best)
            counts[int(marker_colors[a])] = int(gap)
            used = int(gap)
            extra_targets = [int(marker_colors[i]) for i in range(len(marker_colors)) if i not in (a, b)]
        else:
            mode = "simple"

    if mode == "wrap":
        wrap_idx = rng.randrange(len(marker_colors))
        max_wrap = min(interior_size, cycle_len + min(8, interior_size - cycle_len))
        wrap_count = rng.randint(cycle_len + 1, max_wrap)
        counts[int(marker_colors[wrap_idx])] = int(wrap_count)
        used = int(wrap_count)
        extra_targets = list(marker_colors)

    if mode == "simple":
        total = rng.randint(1, min(interior_size, max(4, cycle_len // 2)))
    else:
        max_total = min(interior_size, used + max(0, len(extra_targets)) + cycle_len // 4)
        total = rng.randint(used, max_total)

    remaining = total - used
    if remaining > 0 and extra_targets:
        for _ in range(remaining):
            counts[int(rng.choice(extra_targets))] += 1

    assert sum(counts.values()) > 0, "generator should produce at least one interior colored pixel"
    return counts


def _validate_generated(grid: np.ndarray) -> None:
    spec = _parse_puzzle(grid)
    assert int(spec["bg"]) == BG, "background should stay black"
    assert int(spec["grey"]) == GREY, "ring should stay grey"

    track = spec["track"]
    markers = spec["markers"]
    inner_counts = spec["inner_counts"]

    marker_indices = [int(idx) for idx, _ in markers]
    marker_colors = [int(color) for _, color in markers]
    assert len(marker_colors) >= 2, "need multiple ring markers"
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be distinct"

    marker_index_set = set(marker_indices)
    cycle_len = len(track)
    for idx in marker_indices:
        assert ((idx - 1) % cycle_len) not in marker_index_set
        assert ((idx + 1) % cycle_len) not in marker_index_set

    assert set(inner_counts).issubset(set(marker_colors)), "inner colors must match ring marker colors"
    assert sum(inner_counts.values()) > 0, "keep the task visually non-trivial"
    assert not np.array_equal(main(grid), grid), "generated sample should transform"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(10, 20)
    width = rng.randint(10, 20)

    max_frame_h = height - 2
    max_frame_w = width - 2
    if max_frame_h < 5 or max_frame_w < 5:
        return None

    frame_h = rng.randint(5, max_frame_h)
    frame_w = rng.randint(5, max_frame_w)
    r0 = rng.randint(1, height - frame_h - 1)
    c0 = rng.randint(1, width - frame_w - 1)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    grid = np.full((height, width), BG, dtype=int)
    track = _clockwise_perimeter(r0, c0, r1, c1)
    for r, c in track:
        grid[r, c] = GREY

    max_markers = min(len(PALETTE), len(track) // 3)
    if max_markers < 2:
        return None
    marker_count = rng.randint(2, max_markers)
    marker_indices = _pick_cycle_slots(len(track), marker_count, rng)
    marker_colors = rng.sample(PALETTE, marker_count)

    for idx, color in zip(marker_indices, marker_colors):
        r, c = track[idx]
        grid[r, c] = int(color)

    inner_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if not inner_cells:
        return None

    counts = _sample_counts(marker_indices, marker_colors, len(inner_cells), len(track), rng)
    payload = [color for color, count in counts.items() for _ in range(int(count))]
    rng.shuffle(payload)
    chosen_cells = rng.sample(inner_cells, len(payload))
    for (r, c), color in zip(chosen_cells, payload):
        grid[r, c] = int(color)

    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
