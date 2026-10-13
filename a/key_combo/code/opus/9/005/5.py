from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
TRIGGER = 6
BG = 7


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _segment_key(seg: dict) -> tuple[int, int, int]:
    if seg["orientation"] == "H":
        return 0, seg["line"], seg["start"]
    return 1, seg["line"], seg["start"]


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blue_cells = {tuple(x) for x in np.argwhere(g == BLUE)}
    trigger_cells = {tuple(x) for x in np.argwhere(g == TRIGGER)}

    segments: list[dict] = []
    endpoint_use: set[tuple[int, int]] = set()
    trigger_use: set[tuple[int, int]] = set()

    for r in range(h):
        cols = np.flatnonzero(g[r] == BLUE)
        if len(cols) != 2:
            continue
        c0, c1 = map(int, cols)
        triggers = [(int(r), int(c)) for c in range(c0 + 1, c1) if int(g[r, c]) == TRIGGER]
        if not (1 <= len(triggers) <= 2):
            continue
        seg = {
            "orientation": "H",
            "line": int(r),
            "start": c0,
            "end": c1,
            "endpoints": {(int(r), c0), (int(r), c1)},
            "triggers": triggers,
        }
        segments.append(seg)
        endpoint_use |= seg["endpoints"]
        trigger_use |= set(triggers)

    for c in range(w):
        rows = np.flatnonzero(g[:, c] == BLUE)
        if len(rows) != 2:
            continue
        r0, r1 = map(int, rows)
        triggers = [(int(r), int(c)) for r in range(r0 + 1, r1) if int(g[r, c]) == TRIGGER]
        if not (1 <= len(triggers) <= 2):
            continue
        seg = {
            "orientation": "V",
            "line": int(c),
            "start": r0,
            "end": r1,
            "endpoints": {(r0, int(c)), (r1, int(c))},
            "triggers": triggers,
        }
        segments.append(seg)
        endpoint_use |= seg["endpoints"]
        trigger_use |= set(triggers)

    segments.sort(key=_segment_key)
    assert 2 <= len(segments) <= 4, "expected 2-4 blue endpoint pairs"
    assert endpoint_use == blue_cells, "every blue pixel must be an endpoint of exactly one segment"
    assert trigger_use == trigger_cells, "every trigger must lie on exactly one segment"

    rows_used = [seg["line"] for seg in segments if seg["orientation"] == "H"]
    cols_used = [seg["line"] for seg in segments if seg["orientation"] == "V"]
    assert len(rows_used) == len(set(rows_used))
    assert len(cols_used) == len(set(cols_used))
    return segments


def _draw_segments(grid: np.ndarray, segments: list[dict]) -> np.ndarray:
    bg = get_bg(grid)
    out = np.array(grid, dtype=int, copy=True)
    for seg in segments:
        if seg["orientation"] == "H":
            r = seg["line"]
            for c in range(seg["start"], seg["end"] + 1):
                if int(out[r, c]) == bg:
                    out[r, c] = BLUE
        else:
            c = seg["line"]
            for r in range(seg["start"], seg["end"] + 1):
                if int(out[r, c]) == bg:
                    out[r, c] = BLUE
    return out


def _draw_boundaries(grid: np.ndarray, segments: list[dict]) -> tuple[np.ndarray, dict[str, int]]:
    out = _draw_segments(grid, segments)
    bg = get_bg(grid)
    h, w = out.shape
    stats = {"stopped": 0, "boundary": 0}

    for seg in segments:
        dirs = [(-1, 0), (1, 0)] if seg["orientation"] == "H" else [(0, -1), (0, 1)]
        for r0, c0 in seg["triggers"]:
            for dr, dc in dirs:
                r, c = r0 + dr, c0 + dc
                while 0 <= r < h and 0 <= c < w:
                    if int(out[r, c]) == BLUE:
                        stats["stopped"] += 1
                        break
                    if int(out[r, c]) == bg:
                        out[r, c] = TRIGGER
                    r += dr
                    c += dc
                else:
                    stats["boundary"] += 1

    return out, stats


def _fill_color(area: int) -> int:
    if area <= 12:
        return RED
    if area <= 35:
        return GREEN
    return YELLOW


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    segments = _extract_segments(g)
    boundary, ray_stats = _draw_boundaries(g, segments)
    open_mask = (boundary != BLUE) & (boundary != TRIGGER)
    regions = components_mask(open_mask)

    out = boundary.copy()
    region_areas = []
    fill_colors = set()
    for comp in regions:
        area = len(comp)
        region_areas.append(area)
        fill = _fill_color(area)
        fill_colors.add(fill)
        for r, c in comp:
            out[r, c] = fill

    magenta_boundary_pixels = int(np.count_nonzero(boundary == TRIGGER))
    trigger_pixels = int(np.count_nonzero(g == TRIGGER))
    meta = {
        "segments": segments,
        "segment_count": len(segments),
        "region_areas": sorted(region_areas),
        "fill_colors": fill_colors,
        "ray_stats": ray_stats,
        "magenta_boundary_pixels": magenta_boundary_pixels,
        "trigger_pixels": trigger_pixels,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect axis-aligned blue endpoint pairs, emit perpendicular rays from magenta triggers, then fill the resulting regions by area.

    Input structure:
    - The grid has orange background cells, 2-4 blue endpoint pairs, and magenta trigger pixels.
    - Each blue pair is axis-aligned: either two blue pixels share one row or two blue pixels share one column.
    - Every magenta trigger lies strictly between the endpoints of one blue pair on that pair's future segment.

    Rule:
    - Draw each blue pair's horizontal or vertical segment, keeping any trigger cell on the segment magenta.
    - From each trigger on a horizontal segment, emit magenta rays upward and downward; from each trigger on a vertical segment, emit magenta rays leftward and rightward.
    - A ray paints only background cells and stops just before the first blue segment cell it encounters; otherwise it runs to the grid boundary.
    - Treat blue segments and magenta rays as barriers, find all remaining 4-connected non-barrier regions, and fill them red for area <= 12, green for area 13-35, and yellow for area > 35.

    Key insight:
    - The trigger pixels are beam splitters: their positions on the blue segments determine where perpendicular barriers appear, and those barriers change the enclosed region areas that control the final colors.
    """
    out, _ = _solve_with_metadata(grid)
    return out


def _build_scene(num_aux: int) -> tuple[np.ndarray, int]:
    inner_w = random.randint(3, 5)
    small_h = random.randint(2, min(3, 12 // inner_w))
    medium_lo = max(3, (13 + inner_w - 1) // inner_w)
    medium_hi = min(7, 35 // inner_w)
    medium_h = random.randint(medium_lo, medium_hi)

    top_margin = random.randint(2, 5)
    bottom_margin = random.randint(3, 6)
    u = top_margin
    rh = u + small_h + 1
    d = rh + medium_h + 1
    h = d + bottom_margin + 1

    c0 = random.randint(1, 3)
    t1 = c0 + random.randint(3, 5)
    t2 = t1 + inner_w + 1
    c1 = t2 + random.randint(3, 5)
    right_margin = random.randint(2, 5)
    w = c1 + right_margin + 1

    grid = np.full((h, w), BG, dtype=int)

    segments = [
        {
            "orientation": "H",
            "line": rh,
            "start": c0,
            "end": c1,
            "triggers": [(rh, t1), (rh, t2)],
        }
    ]

    include_top = num_aux == 2 or random.random() < 0.5
    include_bottom = num_aux == 2 or not include_top

    if include_top:
        top_end = random.randint(u + 1, rh - 2)
        top_rows = [u]
        top_interior = [r for r in range(1, top_end) if r != u]
        if top_interior and random.random() < 0.5:
            top_rows.append(random.choice(top_interior))
        top_rows.sort()
        segments.append(
            {
                "orientation": "V",
                "line": t1,
                "start": 0,
                "end": top_end,
                "triggers": [(r, t1) for r in top_rows],
            }
        )

    if include_bottom:
        bottom_start = random.randint(rh + 2, d - 1)
        bottom_rows = [d]
        bottom_interior = [r for r in range(bottom_start + 1, h - 1) if r != d]
        if bottom_interior and random.random() < 0.5:
            bottom_rows.append(random.choice(bottom_interior))
        bottom_rows.sort()
        segments.append(
            {
                "orientation": "V",
                "line": t2,
                "start": bottom_start,
                "end": h - 1,
                "triggers": [(r, t2) for r in bottom_rows],
            }
        )

    for seg in segments:
        if seg["orientation"] == "H":
            r = seg["line"]
            grid[r, seg["start"]] = BLUE
            grid[r, seg["end"]] = BLUE
            for _, c in seg["triggers"]:
                grid[r, c] = TRIGGER
        else:
            c = seg["line"]
            grid[seg["start"], c] = BLUE
            grid[seg["end"], c] = BLUE
            for r, _ in seg["triggers"]:
                grid[r, c] = TRIGGER

    if random.random() < 0.5:
        grid = grid.T.copy()

    return grid, num_aux


def _validate_generated_input(grid: np.ndarray, num_aux: int) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, BLUE, TRIGGER}:
        return False

    try:
        solved, meta = _solve_with_metadata(grid)
    except AssertionError:
        return False

    if meta["segment_count"] != 1 + num_aux:
        return False
    if any(not (1 <= len(seg["triggers"]) <= 2) for seg in meta["segments"]):
        return False
    if len(meta["region_areas"]) < 4:
        return False
    if meta["magenta_boundary_pixels"] <= meta["trigger_pixels"]:
        return False
    if meta["ray_stats"]["stopped"] < 1 or meta["ray_stats"]["boundary"] < 1:
        return False

    min_fill_colors = 3 if num_aux == 2 else 2
    if len(meta["fill_colors"]) < min_fill_colors:
        return False

    segments_only = _draw_segments(grid, meta["segments"])
    if np.array_equal(solved, segments_only):
        return False

    region_areas = meta["region_areas"]
    if min(region_areas) > 12:
        return False
    if max(region_areas) <= 35:
        return False
    if num_aux == 2 and not any(13 <= area <= 35 for area in region_areas):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    num_aux = random.choices([1, 2], weights=[1, 3], k=1)[0]
    grid, num_aux = _build_scene(num_aux)
    if _validate_generated_input(grid, num_aux):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
