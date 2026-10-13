from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


TRACK_COLOR = 5   # grey
MARKER_COLOR = 9  # maroon
BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8]
BEAD_COLORS = [c for c in range(10) if c not in (TRACK_COLOR, MARKER_COLOR)]

GRID_H_RANGE = (15, 30)
GRID_W_RANGE = (15, 30)
TRACK_COUNT_CHOICES = [1, 2, 2, 3, 3]
MARKER_LENGTHS = [2, 3, 4]  # avoid ambiguous 1-pixel markers
GEN_RETRIES = 800
PLACEMENT_RETRIES = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_path(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        path.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        path.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        path.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        path.append((r, c0))
    return path


def _is_rectangular_ring(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    path = _ring_path(r0, c0, r1, c1)
    return set(points) == set(path) and len(points) == len(path)


def _marker_orientation(points: list[tuple[int, int]]) -> str | None:
    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})
    if len(points) == 1:
        return "h"
    if len(rows) == 1 and cols == list(range(cols[0], cols[-1] + 1)):
        return "h"
    if len(cols) == 1 and rows == list(range(rows[0], rows[-1] + 1)):
        return "v"
    return None


def _analyze_tracks(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    comps = _components(g != bg)

    tracks: list[dict] = []
    markers: list[dict] = []
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        if _is_rectangular_ring(comp):
            r0, c0, r1, c1 = _bbox(comp)
            path = _ring_path(r0, c0, r1, c1)
            tracks.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "path": path,
                }
            )
            continue
        if colors != {MARKER_COLOR}:
            raise AssertionError("non-ring components must be maroon markers")
        orient = _marker_orientation(comp)
        if orient is None:
            raise AssertionError("marker must be a straight horizontal or vertical segment")
        markers.append(
            {
                "points": comp,
                "length": len(comp),
                "orient": orient,
            }
        )

    if not tracks:
        raise AssertionError("expected at least one track")

    tracks.sort(key=lambda item: item["bbox"])
    assigned_markers = 0
    for track in tracks:
        r0, c0, r1, c1 = track["bbox"]
        inside = []
        for marker in markers:
            if all(r0 < r < r1 and c0 < c < c1 for r, c in marker["points"]):
                inside.append(marker)
        if len(inside) != 1:
            raise AssertionError("each track must contain exactly one marker")
        marker = inside[0]
        assigned_markers += 1
        track["marker_points"] = marker["points"]
        track["steps"] = marker["length"]
        track["clockwise"] = marker["orient"] == "h"

    if assigned_markers != len(markers):
        raise AssertionError("every marker must belong to exactly one track")

    return bg, tracks


def _rotate(values: list[int], shift: int) -> list[int]:
    n = len(values)
    shift %= n
    if shift == 0:
        return list(values)
    return list(values[-shift:] + values[:-shift])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, tracks = _analyze_tracks(g)
    out = g.copy()

    for track in tracks:
        path = track["path"]
        steps = int(track["steps"])
        clockwise = bool(track["clockwise"])
        vals = [int(g[r, c]) for r, c in path]
        shift = steps if clockwise else -steps
        new_vals = _rotate(vals, shift)
        for (r, c), v in zip(path, new_vals):
            out[r, c] = int(v)
        for r, c in track["marker_points"]:
            out[r, c] = bg

    return out


def _choose_orientations(rng: random.Random, n: int) -> list[str]:
    orientations = [rng.choice(["h", "v"]) for _ in range(n)]
    if n >= 2 and len(set(orientations)) == 1:
        orientations[0] = "h"
        orientations[1] = "v"
    rng.shuffle(orientations)
    return orientations


def _paint_track(
    grid: np.ndarray,
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    orientation: str,
    marker_len: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    path = _ring_path(r0, c0, r1, c1)
    for r, c in path:
        grid[r, c] = TRACK_COLOR

    if orientation == "h":
        row = rng.randint(r0 + 1, r1 - 1)
        start_c = rng.randint(c0 + 1, c1 - marker_len)
        for dc in range(marker_len):
            grid[row, start_c + dc] = MARKER_COLOR
    else:
        col = rng.randint(c0 + 1, c1 - 1)
        start_r = rng.randint(r0 + 1, r1 - marker_len)
        for dr in range(marker_len):
            grid[start_r + dr, col] = MARKER_COLOR

    max_beads = min(len(path) - 1, 8)
    min_beads = 2 if len(path) >= 8 else 1
    bead_count = rng.randint(min_beads, max_beads)
    bead_positions = rng.sample(range(len(path)), bead_count)

    available_colors = [c for c in BEAD_COLORS if c != bg]
    palette_size = rng.randint(1, min(4, len(available_colors)))
    palette = rng.sample(available_colors, palette_size)
    bead_colors = [rng.choice(palette) for _ in range(bead_count)]
    if bead_count >= 2 and len(set(bead_colors)) == 1 and len(available_colors) >= 2:
        alt = rng.choice([c for c in available_colors if c != bead_colors[0]])
        bead_colors[-1] = alt

    for idx, color in zip(bead_positions, bead_colors):
        r, c = path[idx]
        grid[r, c] = color


def _track_bbox_candidates(
    rng: random.Random,
    max_h: int,
    max_w: int,
    orientation: str,
    marker_len: int,
    track_count: int,
) -> tuple[int, int] | None:
    min_h = max(5, marker_len + 2 if orientation == "v" else 5)
    min_w = max(5, marker_len + 2 if orientation == "h" else 5)
    size_cap = {1: 12, 2: 10, 3: 9}[track_count]
    max_h = min(size_cap, max_h)
    max_w = min(size_cap, max_w)
    if min_h > max_h or min_w > max_w:
        return None
    track_h = rng.randint(min_h, max_h)
    track_w = rng.randint(min_w, max_w)
    return track_h, track_w


def _split_intervals(start: int, stop: int, parts: int) -> list[tuple[int, int]]:
    span = stop - start + 1
    base = span // parts
    extra = span % parts
    out = []
    cur = start
    for i in range(parts):
        length = base + (1 if i < extra else 0)
        out.append((cur, cur + length - 1))
        cur += length
    return out


def _slot_layouts(track_count: int) -> list[tuple[int, int]]:
    if track_count == 1:
        return [(1, 1), (2, 2)]
    if track_count == 2:
        return [(1, 2), (2, 1), (2, 2)]
    return [(2, 2)]


def _gen_attempt(rng: random.Random, track_count: int) -> np.ndarray | None:
    min_h = max(GRID_H_RANGE[0], 10 + 4 * track_count)
    min_w = max(GRID_W_RANGE[0], 10 + 4 * track_count)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])
    bg = rng.choice(BG_CHOICES)
    grid = np.full((h, w), bg, dtype=int)

    orientations = _choose_orientations(rng, track_count)
    lengths = [rng.choice(MARKER_LENGTHS) for _ in range(track_count)]
    layout = rng.choice(_slot_layouts(track_count))
    row_parts, col_parts = layout
    row_spans = _split_intervals(1, h - 2, row_parts)
    col_spans = _split_intervals(1, w - 2, col_parts)
    slots = [(r0, c0, r1, c1) for r0, r1 in row_spans for c0, c1 in col_spans]
    if len(slots) < track_count:
        return None
    rng.shuffle(slots)
    chosen_slots = slots[:track_count]

    specs: list[tuple[tuple[int, int, int, int], str, int]] = []
    order = list(range(track_count))
    rng.shuffle(order)
    for idx, slot in zip(order, chosen_slots):
        orientation = orientations[idx]
        marker_len = lengths[idx]
        sr0, sc0, sr1, sc1 = slot
        area_r0, area_c0 = sr0 + 1, sc0 + 1
        area_r1, area_c1 = sr1 - 1, sc1 - 1
        area_h = area_r1 - area_r0 + 1
        area_w = area_c1 - area_c0 + 1
        if area_h < 5 or area_w < 5:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            dims = _track_bbox_candidates(rng, area_h, area_w, orientation, marker_len, track_count)
            if dims is None:
                return None
            track_h, track_w = dims
            r0 = rng.randint(area_r0, area_r1 - track_h + 1)
            c0 = rng.randint(area_c0, area_c1 - track_w + 1)
            r1 = r0 + track_h - 1
            c1 = c0 + track_w - 1
            specs.append(((r0, c0, r1, c1), orientation, marker_len))
            placed = True
            break
        if not placed:
            return None

    for bbox, orientation, marker_len in specs:
        _paint_track(grid, rng, bbox, orientation, marker_len, bg)

    try:
        analyzed_bg, tracks = _analyze_tracks(grid)
    except AssertionError:
        return None
    if analyzed_bg != bg or len(tracks) != track_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    preferred = rng.choice(TRACK_COUNT_CHOICES)
    fallback_counts = [c for c in sorted(set(TRACK_COUNT_CHOICES), reverse=True) if c != preferred]
    candidate_counts = [preferred] + fallback_counts
    for idx, track_count in enumerate(candidate_counts):
        retry_budget = GEN_RETRIES if idx == 0 else max(80, GEN_RETRIES // 4)
        for _ in range(retry_budget):
            grid = _gen_attempt(rng, track_count)
            if grid is not None:
                return grid
    raise RuntimeError("generate_input: failed to sample a valid grid")
