from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


_DIR_ORDER = ("ul", "ur", "ll", "lr")
_DIR_TO_INDEX = {name: idx for idx, name in enumerate(_DIR_ORDER)}
_DIR_SIGNS = {
    "ul": (-1, -1),
    "ur": (-1, 1),
    "ll": (1, -1),
    "lr": (1, 1),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _find_palette(
    grid: np.ndarray, bg: int
) -> tuple[list[int], np.ndarray, tuple[str, int]] | None:
    h, w = grid.shape
    candidates: list[tuple[list[int], np.ndarray, tuple[str, int]]] = []

    def add_candidate(cells: list[tuple[int, int]], edge_key: tuple[str, int]) -> None:
        vals = [int(grid[r, c]) for r, c in cells]
        if any(v == bg for v in vals):
            return
        if len(set(vals)) != 4:
            return
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        candidates.append((vals, mask, edge_key))

    top_nonbg = [c for c in range(w) if int(grid[0, c]) != bg]
    if len(top_nonbg) == 4 and top_nonbg == list(range(top_nonbg[0], top_nonbg[0] + 4)):
        add_candidate([(0, c) for c in top_nonbg], ("top", top_nonbg[0]))

    bottom_nonbg = [c for c in range(w) if int(grid[h - 1, c]) != bg]
    if len(bottom_nonbg) == 4 and bottom_nonbg == list(range(bottom_nonbg[0], bottom_nonbg[0] + 4)):
        add_candidate([(h - 1, c) for c in bottom_nonbg], ("bottom", bottom_nonbg[0]))

    left_nonbg = [r for r in range(h) if int(grid[r, 0]) != bg]
    if len(left_nonbg) == 4 and left_nonbg == list(range(left_nonbg[0], left_nonbg[0] + 4)):
        add_candidate([(r, 0) for r in left_nonbg], ("left", left_nonbg[0]))

    right_nonbg = [r for r in range(h) if int(grid[r, w - 1]) != bg]
    if len(right_nonbg) == 4 and right_nonbg == list(range(right_nonbg[0], right_nonbg[0] + 4)):
        add_candidate([(r, w - 1) for r in right_nonbg], ("right", right_nonbg[0]))

    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]
    return None


def _marker_components(
    grid: np.ndarray, marker_color: int, palette_mask: np.ndarray
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or palette_mask[r, c] or int(grid[r, c]) != marker_color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and not palette_mask[nr, nc]
                        and int(grid[nr, nc]) == marker_color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _component_direction(points: list[tuple[int, int]]) -> tuple[str, tuple[int, int, int, int]] | None:
    if len(points) != 3:
        return None
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    rel = {(r - r0, c - c0) for r, c in points}
    all_rel = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing = all_rel - rel
    if len(missing) != 1:
        return None
    miss = next(iter(missing))
    if miss == (0, 0):
        direction = "ul"
    elif miss == (0, 1):
        direction = "ur"
    elif miss == (1, 0):
        direction = "ll"
    else:
        direction = "lr"
    return direction, (r0, c0, r1, c1)


def _expand_rect(
    grid: np.ndarray,
    bg: int,
    palette_mask: np.ndarray,
    marker_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
) -> tuple[tuple[int, int, int, int], dict[str, str]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    vr, vc = _DIR_SIGNS[direction]

    def passable(r: int, c: int) -> bool:
        v = int(grid[r, c])
        return v == bg or palette_mask[r, c] or marker_mask[r, c]

    while True:
        changed = False

        next_r = r0 - 1 if vr < 0 else r1 + 1
        if 0 <= next_r < h and all(passable(next_r, c) for c in range(c0, c1 + 1)):
            if vr < 0:
                r0 -= 1
            else:
                r1 += 1
            changed = True

        next_c = c0 - 1 if vc < 0 else c1 + 1
        if 0 <= next_c < w and all(passable(r, next_c) for r in range(r0, r1 + 1)):
            if vc < 0:
                c0 -= 1
            else:
                c1 += 1
            changed = True

        if not changed:
            break

    stop: dict[str, str] = {}

    check_r = r0 - 1 if vr < 0 else r1 + 1
    if not (0 <= check_r < h):
        stop["vertical"] = "boundary"
    else:
        stop["vertical"] = (
            "marker"
            if any(int(grid[check_r, c]) != bg and not palette_mask[check_r, c] for c in range(c0, c1 + 1))
            else "open"
        )

    check_c = c0 - 1 if vc < 0 else c1 + 1
    if not (0 <= check_c < w):
        stop["horizontal"] = "boundary"
    else:
        stop["horizontal"] = (
            "marker"
            if any(int(grid[r, check_c]) != bg and not palette_mask[r, check_c] for r in range(r0, r1 + 1))
            else "open"
        )

    return (r0, c0, r1, c1), stop


def _analyze_markers(
    grid: np.ndarray,
) -> tuple[int, list[int], np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    palette_info = _find_palette(g, bg)
    if palette_info is None:
        raise AssertionError("expected one 4-cell edge palette")
    palette_colors, palette_mask, _edge = palette_info

    counts = Counter(int(v) for v in g[~palette_mask].ravel() if int(v) != bg)
    if not counts:
        raise AssertionError("expected marker color")
    marker_color = counts.most_common(1)[0][0]

    raw_comps = _marker_components(g, marker_color, palette_mask)
    markers: list[dict[str, object]] = []
    for comp in raw_comps:
        parsed = _component_direction(comp)
        if parsed is None:
            continue
        direction, bbox = parsed
        comp_mask = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True
        rect, stop = _expand_rect(g, bg, palette_mask, comp_mask, bbox, direction)
        markers.append(
            {
                "points": comp,
                "bbox": bbox,
                "direction": direction,
                "rect": rect,
                "stop": stop,
            }
        )

    if not markers:
        raise AssertionError("expected at least one L marker")
    return bg, palette_colors, palette_mask, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each 2x2 L marker's missing-corner orientation as both a palette index
    and a quadrant instruction, then fill the maximal background rectangle that
    extends from that open corner toward the matching boundary or blocker.
    """
    g = np.asarray(grid, dtype=int)
    bg, palette_colors, palette_mask, markers = _analyze_markers(g)

    out = g.copy()
    out[palette_mask] = bg

    fill_jobs: list[tuple[tuple[int, int, int, int], int]] = []
    for marker in markers:
        direction = marker["direction"]
        rect = marker["rect"]
        fill_color = int(palette_colors[_DIR_TO_INDEX[str(direction)]])
        fill_jobs.append((rect, fill_color))

    for rect, fill_color in fill_jobs:
        r0, c0, r1, c1 = rect
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == bg] = fill_color

    return out


_GRID_RANGE = (12, 19)
_MARKER_COUNT_RANGE = (4, 7)
_GEN_RETRIES = 400


def _sample_colors(rng: random.Random) -> tuple[int, int, list[int]]:
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    marker = rng.choice(colors)
    colors.remove(marker)
    palette = rng.sample(colors, 4)
    return bg, marker, palette


def _place_palette(
    grid: np.ndarray, bg: int, palette: list[int], rng: random.Random
) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    edge = rng.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        start = rng.randint(0, w - 4)
        row = 0 if edge == "top" else h - 1
        for i, color in enumerate(palette):
            grid[row, start + i] = color
            mask[row, start + i] = True
    else:
        start = rng.randint(0, h - 4)
        col = 0 if edge == "left" else w - 1
        for i, color in enumerate(palette):
            grid[start + i, col] = color
            mask[start + i, col] = True
    return mask


def _draw_marker(
    grid: np.ndarray, top: int, left: int, direction: str, marker_color: int
) -> None:
    cells = {
        "ul": ((0, 1), (1, 0), (1, 1)),
        "ur": ((0, 0), (1, 0), (1, 1)),
        "ll": ((0, 0), (0, 1), (1, 1)),
        "lr": ((0, 0), (0, 1), (1, 0)),
    }[direction]
    for dr, dc in cells:
        grid[top + dr, left + dc] = marker_color


def _marker_footprint(top: int, left: int, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, top - 1), min(h, top + 3)):
        for c in range(max(0, left - 1), min(w, left + 3)):
            cells.add((r, c))
    return cells


def _build_orientation_list(rng: random.Random, n_markers: int) -> list[str]:
    base = list(_DIR_ORDER)
    rng.shuffle(base)
    out = base[:]
    while len(out) < n_markers:
        out.append(rng.choice(_DIR_ORDER))
    rng.shuffle(out)
    return out


def _marker_fill_mask(
    rect: tuple[int, int, int, int], marker_points: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    points = set(marker_points)
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in points
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*_GRID_RANGE)
    w = rng.randint(*_GRID_RANGE)
    bg, marker_color, palette = _sample_colors(rng)
    grid = np.full((h, w), bg, dtype=int)
    palette_mask = _place_palette(grid, bg, palette, rng)

    n_markers = rng.randint(*_MARKER_COUNT_RANGE)
    directions = _build_orientation_list(rng, n_markers)

    reserved = set((r, c) for r, c in np.argwhere(palette_mask))
    placed = 0
    for direction in directions:
        for _ in range(80):
            top = rng.randint(1, h - 3)
            left = rng.randint(1, w - 3)
            footprint = _marker_footprint(top, left, h, w)
            if footprint & reserved:
                continue
            if any(palette_mask[r, c] for r, c in footprint):
                continue
            _draw_marker(grid, top, left, direction, marker_color)
            reserved |= footprint
            placed += 1
            break
        else:
            return None

    if placed != n_markers:
        return None

    try:
        bg2, palette_colors, found_palette_mask, markers = _analyze_markers(grid)
    except AssertionError:
        return None

    if bg2 != bg or palette_colors != palette or int(found_palette_mask.sum()) != 4:
        return None
    if len(markers) != n_markers:
        return None

    used_dirs = {str(marker["direction"]) for marker in markers}
    if len(used_dirs) < 4:
        return None

    fill_masks: list[set[tuple[int, int]]] = []
    total_fill = 0
    marker_blocked = False
    boundary_blocked = False
    for marker in markers:
        rect = marker["rect"]
        points = marker["points"]
        fill_mask = _marker_fill_mask(rect, points)
        fill_mask = {
            (r, c)
            for r, c in fill_mask
            if int(grid[r, c]) == bg or found_palette_mask[r, c]
        }
        if len(fill_mask) < 4:
            return None
        if len(fill_mask) > (h * w) // 3:
            return None
        total_fill += len(fill_mask)
        stop = marker["stop"]
        if stop["vertical"] == "marker" or stop["horizontal"] == "marker":
            marker_blocked = True
        if stop["vertical"] == "boundary" or stop["horizontal"] == "boundary":
            boundary_blocked = True
        fill_masks.append(fill_mask)

    if not marker_blocked or not boundary_blocked:
        return None

    for i in range(len(fill_masks)):
        for j in range(i + 1, len(fill_masks)):
            if fill_masks[i] & fill_masks[j]:
                return None

    if total_fill < 18:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if any(int(v) in palette for v in out.ravel()):
        # Palette colors should remain only as fills, never as the edge strip itself.
        pass

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
