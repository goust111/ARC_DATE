from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TRACK = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _border_cycle(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
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


def _pair_slots(n_markers: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    i = 0
    while i < n_markers:
        if i + 1 < n_markers:
            pairs.append((i, i + 1))
        else:
            pairs.append((i, 0))
        i += 2
    return pairs


def _extract_tracks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    tracks: list[dict] = []
    for pts in _components(g != BG):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        cycle = _border_cycle(bbox)
        cycle_set = set(cycle)
        point_set = set(pts)
        assert point_set == cycle_set, "each component must be a 1-cell rectangular ring"

        values = np.array([int(g[r, c]) for r, c in cycle], dtype=int)
        colors, counts = np.unique(values, return_counts=True)
        track_color = int(colors[counts.argmax()])
        marker_indices = [i for i, value in enumerate(values.tolist()) if int(value) != track_color]
        assert 2 <= len(marker_indices) <= 4, "each track must have 2 to 4 markers"
        assert all(int(values[i]) != BG for i in marker_indices)

        tracks.append(
            {
                "bbox": bbox,
                "cycle": cycle,
                "track_color": track_color,
                "marker_indices": marker_indices,
            }
        )
    return tracks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill selected clockwise arcs on rectangular ring tracks.

    Each non-background connected component is a 1-cell-thick rectangular border. Markers on a
    track are ordered clockwise from the track's top-left corner. The ordered markers are paired
    as (1,2), (3,4), ... and, for odd counts, the last marker pairs with the first. A pair fills
    the clockwise arc between its endpoints iff the marker colors are equal.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for track in _extract_tracks(g):
        cycle = track["cycle"]
        marker_indices = track["marker_indices"]
        track_color = int(track["track_color"])
        cycle_len = len(cycle)

        for left_slot, right_slot in _pair_slots(len(marker_indices)):
            start_idx = int(marker_indices[left_slot])
            end_idx = int(marker_indices[right_slot])
            color_a = int(g[cycle[start_idx]])
            color_b = int(g[cycle[end_idx]])
            if color_a != color_b:
                continue

            cursor = (start_idx + 1) % cycle_len
            while cursor != end_idx:
                r, c = cycle[cursor]
                if int(g[r, c]) == track_color:
                    out[r, c] = color_a
                cursor = (cursor + 1) % cycle_len

    return out


MIN_SIDE = 4
GRID_HEIGHT_RANGE = (14, 28)
GRID_WIDTH_RANGE = (14, 30)
TRACK_COUNT_CHOICES = (1, 2, 3, 4)
TRACK_COUNT_WEIGHTS = (2, 3, 3, 2)
MARKER_COUNT_CHOICES = (2, 3, 4)
MARKER_GAP = 2
PLACEMENT_ATTEMPTS = 250
MAX_GENERATION_ATTEMPTS = 500


def _rect_height(rect: tuple[int, int, int, int]) -> int:
    return int(rect[2] - rect[0] + 1)


def _rect_width(rect: tuple[int, int, int, int]) -> int:
    return int(rect[3] - rect[1] + 1)


def _disjoint_with_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _contains_with_margin(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    margin: int = 2,
) -> bool:
    return (
        outer[0] + margin <= inner[0]
        and inner[2] <= outer[2] - margin
        and outer[1] + margin <= inner[1]
        and inner[3] <= outer[3] - margin
    )


def _rect_compatible(candidate: tuple[int, int, int, int], rectangles: list[tuple[int, int, int, int]]) -> bool:
    return all(
        _disjoint_with_gap(candidate, other)
        or _contains_with_margin(candidate, other)
        or _contains_with_margin(other, candidate)
        for other in rectangles
    )


def _sample_interval(lo: int, hi: int, min_len: int) -> tuple[int, int] | None:
    if hi - lo + 1 < min_len:
        return None
    start = random.randint(lo, hi - min_len + 1)
    end = random.randint(start + min_len - 1, hi)
    return int(start), int(end)


def _sample_free_rect(height: int, width: int) -> tuple[int, int, int, int] | None:
    rows = _sample_interval(1, height - 2, MIN_SIDE)
    cols = _sample_interval(1, width - 2, MIN_SIDE)
    if rows is None or cols is None:
        return None
    return rows[0], cols[0], rows[1], cols[1]


def _sample_inner_rect(rectangles: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    parents = [
        rect
        for rect in rectangles
        if _rect_height(rect) >= MIN_SIDE + 4 and _rect_width(rect) >= MIN_SIDE + 4
    ]
    if not parents:
        return None
    parent = random.choice(parents)
    rows = _sample_interval(parent[0] + 2, parent[2] - 2, MIN_SIDE)
    cols = _sample_interval(parent[1] + 2, parent[3] - 2, MIN_SIDE)
    if rows is None or cols is None:
        return None
    return rows[0], cols[0], rows[1], cols[1]


def _sample_outer_rect(
    height: int,
    width: int,
    rectangles: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    children = [
        rect
        for rect in rectangles
        if rect[0] >= 3 and rect[1] >= 3 and rect[2] <= height - 4 and rect[3] <= width - 4
    ]
    if not children:
        return None
    child = random.choice(children)
    top = random.randint(1, child[0] - 2)
    left = random.randint(1, child[1] - 2)
    bottom = random.randint(child[2] + 2, height - 2)
    right = random.randint(child[3] + 2, width - 2)
    if bottom - top + 1 < MIN_SIDE or right - left + 1 < MIN_SIDE:
        return None
    return int(top), int(left), int(bottom), int(right)


def _sample_candidate_rect(
    height: int,
    width: int,
    rectangles: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(12):
        roll = random.random()
        if rectangles and roll < 0.35:
            candidate = _sample_inner_rect(rectangles)
        elif rectangles and roll < 0.55:
            candidate = _sample_outer_rect(height, width, rectangles)
        else:
            candidate = _sample_free_rect(height, width)
        if candidate is not None and _rect_compatible(candidate, rectangles):
            return candidate
    return None


def _place_rectangles(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    rectangles: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            candidate = _sample_candidate_rect(height, width, rectangles)
            if candidate is None:
                continue
            rectangles.append(candidate)
            placed = True
            break
        if not placed:
            return None
    return rectangles


def _sample_marker_indices(cycle_len: int, count: int) -> list[int] | None:
    for _ in range(200):
        picks = sorted(random.sample(range(cycle_len), count))
        gaps = [
            (picks[(i + 1) % count] - picks[i]) % cycle_len
            for i in range(count)
        ]
        if all(gap >= MARKER_GAP for gap in gaps):
            return [int(x) for x in picks]
    return None


def _assign_marker_colors(count: int, fill_flags: list[bool]) -> list[int] | None:
    colors: list[int | None] = [None] * count
    for (a, b), should_fill in zip(_pair_slots(count), fill_flags):
        if colors[a] is None and colors[b] is None:
            base = random.choice(MARKER_COLORS)
            colors[a] = int(base)
            colors[b] = int(base if should_fill else random.choice([c for c in MARKER_COLORS if c != base]))
        elif colors[a] is None:
            ref = int(colors[b])
            colors[a] = int(ref if should_fill else random.choice([c for c in MARKER_COLORS if c != ref]))
        elif colors[b] is None:
            ref = int(colors[a])
            colors[b] = int(ref if should_fill else random.choice([c for c in MARKER_COLORS if c != ref]))
        else:
            same = int(colors[a]) == int(colors[b])
            if same != bool(should_fill):
                return None
    if any(color is None for color in colors):
        return None
    return [int(color) for color in colors]


def _render_input(
    height: int,
    width: int,
    rectangles: list[tuple[int, int, int, int]],
    marker_counts: list[int],
    pair_flags: list[list[bool]],
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    for rect, marker_count, flags in zip(rectangles, marker_counts, pair_flags):
        cycle = _border_cycle(rect)
        marker_indices = _sample_marker_indices(len(cycle), marker_count)
        if marker_indices is None:
            return None
        marker_colors = _assign_marker_colors(marker_count, flags)
        if marker_colors is None:
            return None

        for r, c in cycle:
            grid[r, c] = TRACK
        for idx, color in zip(marker_indices, marker_colors):
            r, c = cycle[idx]
            grid[r, c] = int(color)
    return grid


def _balanced_pair_flags(marker_counts: list[int]) -> list[list[bool]]:
    pairs_per_track = [(count + 1) // 2 for count in marker_counts]
    total_pairs = sum(pairs_per_track)
    same_count = total_pairs // 2
    if total_pairs % 2 == 1 and random.random() < 0.5:
        same_count += 1
    flags = [True] * same_count + [False] * (total_pairs - same_count)
    random.shuffle(flags)

    out: list[list[bool]] = []
    cursor = 0
    for n_pairs in pairs_per_track:
        out.append([bool(x) for x in flags[cursor : cursor + n_pairs]])
        cursor += n_pairs
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    track_count = random.choices(TRACK_COUNT_CHOICES, weights=TRACK_COUNT_WEIGHTS, k=1)[0]

    rectangles = _place_rectangles(height, width, int(track_count))
    if rectangles is None:
        return None

    marker_counts = [random.choice(MARKER_COUNT_CHOICES) for _ in rectangles]
    pair_flags = _balanced_pair_flags(marker_counts)
    grid = _render_input(height, width, rectangles, marker_counts, pair_flags)
    if grid is None:
        return None

    try:
        parsed = _extract_tracks(grid)
        if len(parsed) != len(rectangles):
            return None
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
