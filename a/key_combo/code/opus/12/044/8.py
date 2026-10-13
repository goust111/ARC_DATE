from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MARKER_COLORS = (BLUE, GREEN, YELLOW)
ALLOWED_COLORS = {BLACK, BLUE, RED, GREEN, YELLOW}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _exterior_non_walls(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if grid[r, c] != RED and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                add(nr, nc)
    return exterior


def _global_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid != BLACK) & (grid != RED)):
        markers.append((int(r), int(c), int(grid[r, c])))
    return markers


def _nearest_marker_color(
    region_fill_cells: list[tuple[int, int]], markers: list[tuple[int, int, int]]
) -> int:
    assert region_fill_cells, "fallback requires at least one background cell"
    assert markers, "fallback requires at least one marker"
    best: tuple[int, int, int, int] | None = None
    for rr, cc in region_fill_cells:
        for mr, mc, color in markers:
            d2 = (rr - mr) * (rr - mr) + (cc - mc) * (cc - mc)
            cand = (int(d2), int(mr), int(mc), int(color))
            if best is None or cand < best:
                best = cand
    assert best is not None
    return int(best[3])


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(map(int, g.shape)) <= 30, "ARC grids must fit in 30x30"
    assert set(map(int, np.unique(g))).issubset(ALLOWED_COLORS), "unexpected colors present"
    assert np.any(g == RED), "expected red walls"

    markers = _global_markers(g)
    exterior = _exterior_non_walls(g)
    enclosed_mask = (g != RED) & ~exterior
    regions: list[dict[str, object]] = []

    for comp in _components(enclosed_mask):
        fill_cells = [(r, c) for r, c in comp if int(g[r, c]) == BLACK]
        region_markers = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) in MARKER_COLORS]
        assert fill_cells, "each enclosed region must contain background cells to fill"

        counts = Counter(color for _r, _c, color in region_markers)
        if not counts:
            fill_color = _nearest_marker_color(fill_cells, markers)
            branch = "empty"
        else:
            top = max(counts.values())
            leaders = [int(color) for color, count in counts.items() if count == top]
            if len(leaders) == 1:
                fill_color = leaders[0]
                branch = "majority" if len(counts) >= 2 else "single"
            else:
                fill_color = _nearest_marker_color(fill_cells, markers)
                branch = "tie"

        regions.append(
            {
                "cells": comp,
                "fill_cells": fill_cells,
                "markers": region_markers,
                "counts": dict(counts),
                "fill_color": int(fill_color),
                "branch": branch,
                "bbox": _bbox(comp),
            }
        )

    regions.sort(key=lambda info: tuple(int(v) for v in info["bbox"]))
    return {
        "regions": regions,
        "markers": markers,
        "outside_markers": [(r, c, color) for r, c, color in markers if exterior[r, c]],
        "exterior": exterior,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each red-walled enclosed region with the color chosen by its internal marker vote, using nearest-marker fallback for empty or tied regions.

    Input structure
    - The grid has a black background, one-cell-thick red wall segments, and isolated marker pixels in blue, green, or yellow.
    - Red walls partition the board into open exterior background and several enclosed regions.
    - Marker pixels may lie inside enclosed regions or on the open exterior background.

    Rule
    - Treat red pixels as walls and flood-fill from the grid boundary through every non-wall cell to mark the exterior.
    - Each remaining 4-connected non-wall pocket is one enclosed region; count the marker colors inside it and use the unique majority color as that region's fill color.
    - If a region has no internal markers, or if the internal color counts tie for first place, choose the color of the globally nearest marker to any background cell of that region, breaking equal distances by smaller marker row, then smaller marker column.
    - Fill all background cells of each enclosed region with the chosen color, keep red walls unchanged, and erase every marker pixel to black.

    Key insight
    - The red topology defines the regions, internal markers vote only within their own region, and empty or tied regions collapse to a global nearest-marker lookup.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in analysis["regions"]:
        fill_color = int(region["fill_color"])
        for r, c in region["fill_cells"]:
            out[r, c] = fill_color

    for r, c, _color in analysis["markers"]:
        out[r, c] = BLACK
    return out


SLOT_H = 7
SLOT_W = 7
GAP = 2
MAX_ATTEMPTS = 400


def _role_sequence() -> list[str]:
    region_count = random.randint(3, 6)
    include_tie = random.random() < 0.7

    roles = ["majority", "empty"]
    if include_tie:
        roles.append("tie")
    else:
        roles.append("simple")

    while len(roles) < region_count:
        roles.append(random.choice(["simple", "simple", "empty"]))
    random.shuffle(roles)
    return roles


def _grid_geometry(region_count: int) -> tuple[int, int, int, int]:
    if region_count <= 3:
        cols = random.choice([2, 3])
    elif region_count == 4:
        cols = random.choice([2, 3])
    else:
        cols = 3
    rows = (region_count + cols - 1) // cols
    height = rows * SLOT_H + (rows + 1) * GAP
    width = cols * SLOT_W + (cols + 1) * GAP
    return height, width, rows, cols


def _make_frames(roles: list[str]) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w, rows, cols = _grid_geometry(len(roles))
    grid = np.zeros((h, w), dtype=int)
    slot_ids = random.sample(range(rows * cols), len(roles))
    frames: list[dict[str, object]] = []

    for role, slot_id in zip(roles, slot_ids):
        sr = slot_id // cols
        sc = slot_id % cols
        slot_r0 = GAP + sr * (SLOT_H + GAP)
        slot_c0 = GAP + sc * (SLOT_W + GAP)

        if role in {"majority", "tie"}:
            outer_h = random.choice([5, 6])
            outer_w = random.choice([5, 6])
        elif role == "simple":
            outer_h = random.choice([5, 6])
            outer_w = random.choice([4, 5, 6])
        else:
            outer_h = random.choice([4, 5, 6])
            outer_w = random.choice([4, 5, 6])

        r0 = slot_r0 + random.randint(0, SLOT_H - outer_h)
        c0 = slot_c0 + random.randint(0, SLOT_W - outer_w)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        grid[r0, c0 : c1 + 1] = RED
        grid[r1, c0 : c1 + 1] = RED
        grid[r0 : r1 + 1, c0] = RED
        grid[r0 : r1 + 1, c1] = RED

        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        corners = [(r0 + 1, c0 + 1), (r0 + 1, c1 - 1), (r1 - 1, c0 + 1), (r1 - 1, c1 - 1)]
        unique_corners: list[tuple[int, int]] = []
        for cell in corners:
            if cell not in unique_corners:
                unique_corners.append(cell)

        frames.append(
            {
                "role": role,
                "bbox": (r0, c0, r1, c1),
                "interior": interior,
                "corners": unique_corners,
            }
        )

    return grid, frames


def _place_region_markers(grid: np.ndarray, frame: dict[str, object]) -> None:
    role = str(frame["role"])
    corners = list(frame["corners"])
    random.shuffle(corners)
    palette = random.sample(list(MARKER_COLORS), k=3)

    if role == "majority":
        winner = palette[0]
        loser = palette[1]
        if len(corners) >= 4 and random.random() < 0.5:
            assignments = [(corners[0], winner), (corners[1], winner), (corners[2], winner), (corners[3], loser)]
        else:
            assignments = [(corners[0], winner), (corners[1], winner), (corners[2], loser)]
    elif role == "tie":
        color_a = palette[0]
        color_b = palette[1]
        if len(corners) >= 4 and random.random() < 0.5:
            assignments = [(corners[0], color_a), (corners[1], color_a), (corners[2], color_b), (corners[3], color_b)]
        else:
            assignments = [(corners[0], color_a), (corners[-1], color_b)]
    elif role == "simple":
        color = palette[0]
        if len(corners) >= 2 and random.random() < 0.4:
            assignments = [(corners[0], color), (corners[1], color)]
        else:
            assignments = [(corners[0], color)]
    else:
        assignments = []

    for (r, c), color in assignments:
        grid[r, c] = int(color)


def _nearby_exterior_candidates(
    exterior: np.ndarray, bbox: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    h, w = exterior.shape
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[int, int]] = []

    for c in range(c0, c1 + 1):
        for r in (r0 - 1, r1 + 1):
            if 0 <= r < h and 0 <= c < w and exterior[r, c]:
                candidates.append((r, c))
    for r in range(r0, r1 + 1):
        for c in (c0 - 1, c1 + 1):
            if 0 <= r < h and 0 <= c < w and exterior[r, c]:
                candidates.append((r, c))

    dedup: list[tuple[int, int]] = []
    for cell in candidates:
        if cell not in dedup:
            dedup.append(cell)
    random.shuffle(dedup)
    return dedup


def _random_exterior_candidates(exterior: np.ndarray) -> list[tuple[int, int]]:
    cells = [(int(r), int(c)) for r, c in np.argwhere(exterior)]
    random.shuffle(cells)
    return cells


def _place_outside_markers(grid: np.ndarray, frames: list[dict[str, object]]) -> None:
    base_analysis = _analyze_grid(grid)
    exterior = base_analysis["exterior"]
    used = {(r, c) for r, c, _ in base_analysis["markers"]}

    empty_frames = [frame for frame in frames if str(frame["role"]) == "empty"]
    assert empty_frames, "generator requires at least one empty region"
    focus_frame = random.choice(empty_frames)

    placed = 0
    focus_candidates = _nearby_exterior_candidates(exterior, focus_frame["bbox"])
    for r, c in focus_candidates:
        if (r, c) not in used and grid[r, c] == BLACK:
            grid[r, c] = random.choice(MARKER_COLORS)
            used.add((r, c))
            placed += 1
            break
    assert placed == 1, "failed to place the mandatory exterior marker"

    extra_target = random.randint(0, 2)
    for r, c in _random_exterior_candidates(exterior):
        if placed >= 1 + extra_target:
            break
        if (r, c) in used or grid[r, c] != BLACK:
            continue
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in used):
            continue
        grid[r, c] = random.choice(MARKER_COLORS)
        used.add((r, c))
        placed += 1


def _valid_example(grid: np.ndarray, planned_tie: bool) -> bool:
    analysis = _analyze_grid(grid)
    regions = analysis["regions"]
    branches = [str(region["branch"]) for region in regions]

    if not (3 <= len(regions) <= 6):
        return False
    if not analysis["outside_markers"]:
        return False
    if any(len(region["fill_cells"]) < 4 for region in regions):
        return False

    marked_regions = [region for region in regions if region["markers"]]
    if len(marked_regions) < 2:
        return False
    if "empty" not in branches:
        return False
    if not any(len(region["counts"]) >= 2 and max(region["counts"].values()) > sorted(region["counts"].values())[-2] for region in marked_regions if len(region["counts"]) >= 2):
        return False
    if planned_tie and "tie" not in branches:
        return False
    if not planned_tie and random.random() < 0.2 and "tie" in branches:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.any((grid == RED) != (out == RED)):
        return False
    if np.any(out[(grid != RED) & (grid != BLACK) & (grid != BLUE) & (grid != GREEN) & (grid != YELLOW)]):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    roles = _role_sequence()
    planned_tie = "tie" in roles
    grid, frames = _make_frames(roles)
    for frame in frames:
        _place_region_markers(grid, frame)
    _place_outside_markers(grid, frames)

    if _valid_example(grid, planned_tie):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
