from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
STRUCTURE = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

FRAME_OUTER = 5
FRAME_INNER = 3
FRAME_BORDER = 1

HEIGHT_RANGE = (2, 8)
FRAME_COUNT_OPTIONS = (3, 4)

GRID_HEIGHT_RANGE = (21, 28)
GRID_WIDTH_RANGE = (20, 29)
MAX_GENERATION_ATTEMPTS = 500

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _terrain_mask_and_heights(grid: np.ndarray, structure: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    heights = np.zeros(w, dtype=int)
    mask = np.zeros_like(grid, dtype=bool)
    for c in range(w):
        height = 0
        for r in range(h - 1, -1, -1):
            if int(grid[r, c]) != structure:
                break
            height += 1
        heights[c] = height
        if height > 0:
            mask[h - height :, c] = True
    return mask, heights


def _frame_from_component(coords: np.ndarray) -> dict[str, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    patch = np.zeros((h, w), dtype=bool)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    patch[rr, cc] = True
    ring = np.zeros_like(patch)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.array_equal(patch, ring):
        return None
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "height": int(h),
        "width": int(w),
        "center_col": int(c0 + (w - 1) // 2),
    }


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    bottom_non_bg = [int(v) for v in np.unique(g[-1]) if int(v) != bg]
    assert len(bottom_non_bg) == 1, "bottom row must expose a unique terrain color"
    structure = int(bottom_non_bg[0])
    assert np.all(g[-1] == structure), "bottom row must be solid terrain"

    terrain_mask, terrain_heights = _terrain_mask_and_heights(g, structure)
    assert int(terrain_heights.min()) > 0, "every column must have a terrain suffix"

    structure_above = (g == structure) & ~terrain_mask
    frame_comps = _components(structure_above, connectivity=4)
    frames = []
    for comp in frame_comps:
        frame = _frame_from_component(comp)
        assert frame is not None, "all structure components above the terrain must be rectangular frames"
        frames.append(frame)
    assert frames, "expected at least one frame"

    center_heights = []
    for frame in frames:
        center_col = int(frame["center_col"])
        center_height = int(terrain_heights[center_col])
        center_heights.append(center_height)
        frame["terrain_height"] = center_height
        assert frame["r1"] + 1 < g.shape[0] - center_height, "frames must be separated from terrain"
        interior = g[int(frame["r0"]) + 1 : int(frame["r1"]), int(frame["c0"]) + 1 : int(frame["c1"])]
        assert np.all(interior == bg), "frame interiors must be empty in the input"
    assert len(set(center_heights)) == len(center_heights), "frame center terrain heights must be unique"

    marker_mask = (g != bg) & (g != structure)
    marker_counts: dict[int, int] = {}
    count_to_color: dict[int, int] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (bg, structure)):
        count = int(np.count_nonzero(g == color))
        if count == 0:
            continue
        marker_counts[color] = count
        assert count not in count_to_color, "marker counts must be unique"
        count_to_color[count] = color
    assert len(marker_counts) == len(frames), "frame count must equal marker color count"
    assert set(count_to_color) == set(center_heights), "frame heights must match marker counts exactly"

    return {
        "grid": g,
        "bg": bg,
        "structure": structure,
        "terrain_mask": terrain_mask,
        "terrain_heights": terrain_heights,
        "frames": sorted(frames, key=lambda f: (int(f["c0"]), int(f["r0"]))),
        "marker_mask": marker_mask,
        "marker_counts": marker_counts,
        "count_to_color": count_to_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each top frame with the scattered marker color whose pixel count matches the terrain height under that frame.

    Input structure:
    - The background is black.
    - Several one-cell-thick hollow frames share the same grey color as a bottom terrain profile.
    - The terrain occupies a contiguous suffix in every column, while scattered non-grey single-pixel markers sit between frames and terrain.

    Rule:
    - Detect each hollow frame and take its horizontal center column.
    - Measure the terrain height in that column as the number of contiguous grey cells from the bottom upward.
    - Count the scattered marker pixels of each non-grey, non-black color and match each frame to the unique color whose count equals that height.
    - Fill the entire frame interior with that matched color, erase all scattered markers, and keep the frame borders and terrain unchanged.

    Key insight:
    - The fill color is determined only after joining two distant clues: a numeric key read from the terrain beneath a frame and an equal-valued pixel count read from the scattered markers.
    """
    info = _analyze(grid)
    g = info["grid"]
    bg = int(info["bg"])
    count_to_color = info["count_to_color"]

    out = g.copy()
    out[info["marker_mask"]] = bg
    for frame in info["frames"]:
        fill = int(count_to_color[int(frame["terrain_height"])])
        out[int(frame["r0"]) + 1 : int(frame["r1"]), int(frame["c0"]) + 1 : int(frame["c1"])] = fill
    return out


def _random_frame_positions(width: int, n_frames: int) -> list[tuple[int, int]]:
    possible_cols = list(range(1, width - FRAME_OUTER))
    for _ in range(200):
        random.shuffle(possible_cols)
        cols = []
        for c0 in possible_cols:
            if all(abs(c0 - prev) >= FRAME_OUTER + 1 for prev in cols):
                cols.append(c0)
                if len(cols) == n_frames:
                    break
        if len(cols) != n_frames:
            continue
        cols.sort()
        rows = [random.randint(0, 3) for _ in range(n_frames)]
        return list(zip(rows, cols))
    raise RuntimeError("unable to place frames")


def _valid_endpoint_values(target: int, dist: int, lo: int, hi: int, max_step: int) -> list[int]:
    return [v for v in range(lo, hi + 1) if abs(v - target) <= dist * max_step]


def _random_segment(start: int, end: int, steps: int, lo: int, hi: int, max_step: int) -> list[int] | None:
    if steps == 0:
        return [start] if start == end else None
    vals = [start]
    cur = start
    for pos in range(1, steps):
        remaining = steps - pos
        candidates = []
        weights = []
        for nxt in range(lo, hi + 1):
            delta = abs(nxt - cur)
            if delta > max_step:
                continue
            if abs(end - nxt) > remaining * max_step:
                continue
            candidates.append(nxt)
            weights.append({0: 5, 1: 4, 2: 2}.get(delta, 1))
        if not candidates:
            return None
        cur = random.choices(candidates, weights=weights, k=1)[0]
        vals.append(cur)
    if abs(end - cur) > max_step:
        return None
    vals.append(end)
    return vals


def _terrain_profile(width: int, fixed: dict[int, int], lo: int = 2, hi: int = 8, max_step: int = 2) -> np.ndarray | None:
    anchors = dict(fixed)
    fixed_cols = sorted(anchors)
    left_col = fixed_cols[0]
    right_col = fixed_cols[-1]

    if 0 not in anchors:
        candidates = _valid_endpoint_values(anchors[left_col], left_col, lo, hi, max_step)
        if not candidates:
            return None
        anchors[0] = random.choice(candidates)
    if width - 1 not in anchors:
        dist = (width - 1) - right_col
        candidates = _valid_endpoint_values(anchors[right_col], dist, lo, hi, max_step)
        if not candidates:
            return None
        anchors[width - 1] = random.choice(candidates)

    cols = sorted(anchors)
    profile: list[int] = []
    for c0, c1 in zip(cols, cols[1:]):
        segment = _random_segment(anchors[c0], anchors[c1], c1 - c0, lo, hi, max_step)
        if segment is None:
            return None
        if profile:
            profile.extend(segment[1:])
        else:
            profile.extend(segment)
    arr = np.array(profile, dtype=int)
    if arr.shape != (width,):
        return None
    return arr


def _draw_frame(grid: np.ndarray, r0: int, c0: int, fill: int | None = None) -> None:
    r1 = r0 + FRAME_OUTER - 1
    c1 = c0 + FRAME_OUTER - 1
    grid[r0 : r1 + 1, c0] = STRUCTURE
    grid[r0 : r1 + 1, c1] = STRUCTURE
    grid[r0, c0 : c1 + 1] = STRUCTURE
    grid[r1, c0 : c1 + 1] = STRUCTURE
    if fill is not None:
        grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)


def _draw_terrain(grid: np.ndarray, heights: np.ndarray) -> None:
    h = grid.shape[0]
    for c, height in enumerate(heights):
        grid[h - int(height) :, c] = STRUCTURE


def _scatter_isolated_markers(grid: np.ndarray, jobs: list[int], row_lo: int, row_hi: int) -> bool:
    h, w = grid.shape
    candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(1, w - 1) if grid[r, c] == BG]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    occupied = set()
    for r, c in candidates:
        if (r, c) in occupied:
            continue
        chosen.append((r, c))
        for dr, dc in NEIGHBORS8 + ((0, 0),):
            occupied.add((r + dr, c + dc))
        if len(chosen) == len(jobs):
            break
    if len(chosen) != len(jobs):
        return False
    random.shuffle(chosen)
    for (r, c), color in zip(chosen, jobs):
        grid[r, c] = int(color)
    return True


def _validate_case(inp: np.ndarray, out: np.ndarray | None = None) -> None:
    info = _analyze(inp)
    frames = info["frames"]
    assert len(frames) in FRAME_COUNT_OPTIONS
    assert all(int(frame["height"]) == FRAME_OUTER and int(frame["width"]) == FRAME_OUTER for frame in frames)

    marker_cells = np.argwhere(info["marker_mask"])
    for r, c in marker_cells:
        for dr, dc in NEIGHBORS8:
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                assert not info["marker_mask"][nr, nc], "markers must be isolated single pixels"

    left_to_right_heights = [int(frame["terrain_height"]) for frame in frames]
    if len(left_to_right_heights) >= 3:
        assert left_to_right_heights != sorted(left_to_right_heights), "height order should not be trivially ascending"
        assert left_to_right_heights != sorted(left_to_right_heights, reverse=True), "height order should not be trivially descending"

    for frame in frames:
        interior = inp[int(frame["r0"]) + 1 : int(frame["r1"]), int(frame["c0"]) + 1 : int(frame["c1"])]
        assert np.all(interior == BG), "input frame interiors must start empty"

    if out is not None:
        solved = main(inp)
        assert np.array_equal(solved, out), "oracle mismatch"
        assert np.count_nonzero(inp != out) > len(frames), "transform must change more than a trivial number of cells"

        structure = int(info["structure"])
        assert np.array_equal(inp == structure, out == structure), "frame borders and terrain must stay unchanged"
        assert not np.any((out != BG) & info["marker_mask"]), "markers must be erased in the output"
        for frame in frames:
            expected = int(info["count_to_color"][int(frame["terrain_height"])])
            interior = out[int(frame["r0"]) + 1 : int(frame["r1"]), int(frame["c0"]) + 1 : int(frame["c1"])]
            assert interior.shape == (FRAME_INNER, FRAME_INNER)
            assert np.all(interior == expected), "frame interior must be filled by its matched marker color"


def _attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_frames = random.choice(FRAME_COUNT_OPTIONS)

    try:
        placements = _random_frame_positions(width, n_frames)
    except RuntimeError:
        return None

    counts = random.sample(range(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1), n_frames)
    ordered_counts = counts[:]
    for _ in range(50):
        random.shuffle(ordered_counts)
        if n_frames < 3:
            break
        if ordered_counts != sorted(ordered_counts) and ordered_counts != sorted(ordered_counts, reverse=True):
            break
    else:
        return None

    frame_centers = [c0 + FRAME_OUTER // 2 for _r0, c0 in placements]
    fixed_heights = {center: count for center, count in zip(frame_centers, ordered_counts)}
    terrain = _terrain_profile(width, fixed_heights)
    if terrain is None:
        return None

    max_height = int(terrain.max())
    frame_bottom = max(r0 + FRAME_OUTER - 1 for r0, _c0 in placements)
    marker_row_lo = frame_bottom + 2
    marker_row_hi = height - max_height - 2
    if marker_row_hi - marker_row_lo + 1 < 4:
        return None

    colors = random.sample(MARKER_COLORS, n_frames)
    count_to_color = {count: color for count, color in zip(random.sample(ordered_counts, len(ordered_counts)), colors)}
    if len(set(count_to_color.values())) != n_frames:
        return None

    out = np.full((height, width), BG, dtype=int)
    _draw_terrain(out, terrain)
    for r0, c0 in placements:
        center = c0 + FRAME_OUTER // 2
        fill = count_to_color[int(terrain[center])]
        _draw_frame(out, r0, c0, fill=fill)

    inp = out.copy()
    for r0, c0 in placements:
        inp[r0 + 1 : r0 + FRAME_OUTER - 1, c0 + 1 : c0 + FRAME_OUTER - 1] = BG

    jobs = []
    for count, color in count_to_color.items():
        jobs.extend([int(color)] * int(count))
    random.shuffle(jobs)
    if not _scatter_isolated_markers(inp, jobs, marker_row_lo, marker_row_hi):
        return None

    try:
        _validate_case(inp, out)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
