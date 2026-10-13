from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5
CUP_COLORS = (2, 3, 6, 7, 8, 9)
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_cup_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in (BLACK, BLUE):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR_VECS.values():
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords.sort()
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _extract_patch(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in comp["coords"]:
        patch[r - r0, c - c0] = int(comp["color"])
    return patch


def _detect_cup_geometry(patch: np.ndarray) -> tuple[str, list[tuple[int, int]], np.ndarray]:
    wall = patch != BLACK
    h, w = wall.shape
    cases: list[tuple[str, list[tuple[int, int]]]] = []

    if h >= 3 and w >= 3:
        if wall[-1, :].all() and wall[:, 0].all() and wall[:, -1].all() and wall[0, 0] and wall[0, -1] and not wall[0, 1:-1].any():
            cases.append(("up", [(0, c) for c in range(1, w - 1)]))
        if wall[0, :].all() and wall[:, 0].all() and wall[:, -1].all() and wall[-1, 0] and wall[-1, -1] and not wall[-1, 1:-1].any():
            cases.append(("down", [(h - 1, c) for c in range(1, w - 1)]))
        if wall[:, -1].all() and wall[0, :].all() and wall[-1, :].all() and wall[0, 0] and wall[-1, 0] and not wall[1:-1, 0].any():
            cases.append(("left", [(r, 0) for r in range(1, h - 1)]))
        if wall[:, 0].all() and wall[0, :].all() and wall[-1, :].all() and wall[0, -1] and wall[-1, -1] and not wall[1:-1, -1].any():
            cases.append(("right", [(r, w - 1) for r in range(1, h - 1)]))

    if len(cases) != 1:
        raise ValueError("component is not a supported rotated cup")
    direction, opening_cells = cases[0]
    interior = ~wall
    return direction, opening_cells, interior


def _gap_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gaps: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(1, w - 1):
            if int(g[r, c]) == BLACK and int(g[r, c - 1]) == BLUE and int(g[r, c + 1]) == BLUE:
                gaps.append((r, c))
    return gaps


def _ray_path(grid: np.ndarray, opening_cells: list[tuple[int, int]], direction: str) -> set[tuple[int, int]]:
    dr, dc = DIR_VECS[direction]
    blockers = (grid != BLACK) & (grid != BLUE)
    path: set[tuple[int, int]] = set()
    for r, c in opening_cells:
        rr, cc = r + dr, c + dc
        while _in_bounds(grid.shape, rr, cc) and not blockers[rr, cc]:
            path.add((rr, cc))
            rr += dr
            cc += dc
    return path


def _scene_features(grid: np.ndarray) -> dict:
    cups = []
    for comp in _find_cup_components(grid):
        patch = _extract_patch(comp)
        direction, opening_local, interior_local = _detect_cup_geometry(patch)
        r0, c0, _r1, _c1 = comp["bbox"]
        opening_global = [(r0 + r, c0 + c) for r, c in opening_local]
        interior_global = [(r0 + r, c0 + c) for r, c in np.argwhere(interior_local)]
        path = _ray_path(grid, opening_global, direction)
        cups.append(
            {
                "color": int(comp["color"]),
                "direction": direction,
                "opening": opening_global,
                "interior": interior_global,
                "path": path,
                "bbox": comp["bbox"],
            }
        )
    gaps = _gap_coords(grid)
    for cup in cups:
        cup["hit_gaps"] = sorted(gap for gap in gaps if gap in cup["path"])
    return {"cups": cups, "gaps": gaps}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill cup interiors yellow and recolor dashed-bar gaps according to which cup ray passes through them.

    Input structure:
    - The grid is a black background containing 2-3 horizontal blue dashed segments and 2-3 monochrome cup-shaped containers.
    - Each cup is a 4-connected rotated U shape with one open side; its interior and mouth cells are black inside the cup bbox.
    - Blue dash pixels do not block rays, while cup-wall pixels keep their original colors.

    Rule:
    - Detect each cup, determine its opening direction, and treat every black cell inside the cup bbox as the cup interior to be filled yellow.
    - Emit a straight ray outward from every opening cell, using the opening width as the ray width; rays travel through black and blue cells but stop before any other colored pixel.
    - For every black one-cell gap inside a horizontal blue dashed segment, recolor it with the wall color of the cup whose ray passes through that gap; gaps missed by all rays become grey.

    Key insight:
    - The target color of each dashed gap is not fixed: it is determined by the intersection between a cup's opening-defined ray and that specific gap location.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    features = _scene_features(g)

    for cup in features["cups"]:
        for r, c in cup["interior"]:
            out[r, c] = YELLOW

    for gap in features["gaps"]:
        owners = [cup["color"] for cup in features["cups"] if gap in cup["path"]]
        if owners:
            color = owners[0]
            if any(owner != color for owner in owners[1:]):
                raise ValueError("a dashed gap is hit by multiple differently colored rays")
            out[gap] = color
        else:
            out[gap] = GREY

    return out


def _make_cup_patch(direction: str, opening_width: int, color: int) -> np.ndarray:
    if direction in ("up", "down"):
        patch = np.zeros((3, opening_width + 2), dtype=int)
        if direction == "up":
            patch[-1, :] = color
            patch[:, 0] = color
            patch[:, -1] = color
        else:
            patch[0, :] = color
            patch[:, 0] = color
            patch[:, -1] = color
        return patch

    patch = np.zeros((opening_width + 2, 3), dtype=int)
    if direction == "left":
        patch[:, -1] = color
        patch[0, :] = color
        patch[-1, :] = color
    else:
        patch[:, 0] = color
        patch[0, :] = color
        patch[-1, :] = color
    return patch


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BLACK
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    if np.any(grid[r0 : r0 + h, c0 : c0 + w] != BLACK):
        return False

    wall_cells = [(r0 + r, c0 + c) for r, c in np.argwhere(patch != BLACK)]
    wall_set = set(wall_cells)
    for gr, gc in wall_cells:
        for nr in range(gr - 1, gr + 2):
            for nc in range(gc - 1, gc + 2):
                if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in wall_set:
                    continue
                if int(grid[nr, nc]) != BLACK:
                    return False
    return True


def _candidate_placements_for_gap(
    grid: np.ndarray,
    gap: tuple[int, int],
    direction: str,
    color: int,
    span_order: list[int],
) -> list[dict]:
    gap_r, gap_c = gap
    dr, dc = DIR_VECS[direction]
    candidates: list[dict] = []
    max_dist = max(grid.shape)
    for opening_width in span_order:
        patch = _make_cup_patch(direction, opening_width, color)
        _detected_dir, opening_local, _interior = _detect_cup_geometry(patch)
        for orow, ocol in opening_local:
            for dist in range(2, max_dist + 1):
                r0 = gap_r - dist * dr - orow
                c0 = gap_c - dist * dc - ocol
                if not _can_place_patch(grid, patch, r0, c0):
                    continue
                candidates.append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "patch": patch,
                        "direction": direction,
                        "opening_width": opening_width,
                        "target_gap": gap,
                    }
                )
    random.shuffle(candidates)
    return candidates


def _generate_dashes(height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    bar_count = random.choice((2, 3))
    row_pool = list(range(2, height - 2))
    for _ in range(200):
        rows = sorted(random.sample(row_pool, bar_count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            break
    else:
        raise RuntimeError("failed to choose dash rows")

    for row in rows:
        max_blue = min(6, (width + 1) // 2)
        blue_count = random.randint(4, max_blue)
        span = 2 * blue_count - 1
        start = random.randint(0, width - span)
        grid[row, start : start + span : 2] = BLUE
    return grid


def _partial_scene_ok(grid: np.ndarray) -> bool:
    try:
        features = _scene_features(grid)
    except Exception:
        return False
    cups = features["cups"]
    if not cups:
        return False
    directions = [cup["direction"] for cup in cups]
    if len(set(directions)) != len(directions):
        return False
    hit_sets = [set(cup["hit_gaps"]) for cup in cups]
    if any(not hit_set for hit_set in hit_sets):
        return False
    for i in range(len(hit_sets)):
        for j in range(i + 1, len(hit_sets)):
            if hit_sets[i] & hit_sets[j]:
                return False
    return True


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        features = _scene_features(grid)
        out = main(grid)
    except Exception:
        return False

    cups = features["cups"]
    gaps = features["gaps"]
    if not (2 <= len(cups) <= 3):
        return False
    if len(set(cup["direction"] for cup in cups)) != len(cups):
        return False
    if not gaps:
        return False

    hit_sets = [set(cup["hit_gaps"]) for cup in cups]
    if any(not hit_set for hit_set in hit_sets):
        return False
    for i in range(len(hit_sets)):
        for j in range(i + 1, len(hit_sets)):
            if hit_sets[i] & hit_sets[j]:
                return False

    hit_union = set().union(*hit_sets)
    if len(hit_union) >= len(gaps):
        return False

    bar_rows = sorted({r for r, c in np.argwhere(grid == BLUE)})
    if not (2 <= len(bar_rows) <= 3):
        return False
    for row in bar_rows:
        cols = np.where(grid[row] == BLUE)[0]
        if len(cols) < 4:
            return False
        if np.any(np.diff(cols) != 2):
            return False
        if cols[-1] - cols[0] + 1 < 7:
            return False

    for cup in cups:
        if not cup["interior"]:
            return False
        for r, c in cup["interior"]:
            if int(out[r, c]) != YELLOW:
                return False
    for gap in gaps:
        expected = GREY
        for cup in cups:
            if gap in cup["path"]:
                expected = cup["color"]
                break
        if int(out[gap]) != expected:
            return False
    return True


def _generate_attempt() -> np.ndarray:
    height = random.randint(15, 20)
    width = random.randint(15, 20)
    grid = _generate_dashes(height, width)
    gaps = _gap_coords(grid)

    cup_count = random.choices((2, 3), weights=(2, 3))[0]
    directions = random.sample(list(DIR_VECS), cup_count)
    colors = random.sample(CUP_COLORS, cup_count)
    remaining_gaps = gaps.copy()
    random.shuffle(remaining_gaps)

    wide_index = random.randrange(cup_count) if random.random() < 0.7 else None
    for idx, (direction, color) in enumerate(zip(directions, colors)):
        preferred = 2 if idx == wide_index else random.choice((1, 2))
        span_order = [preferred] + [span for span in (1, 2) if span != preferred]
        gap_order = remaining_gaps.copy()
        random.shuffle(gap_order)

        placed = False
        for gap in gap_order:
            candidates = _candidate_placements_for_gap(grid, gap, direction, color, span_order)
            for cand in candidates:
                trial = grid.copy()
                _paste_patch(trial, cand["patch"], cand["r0"], cand["c0"])
                if _partial_scene_ok(trial):
                    grid = trial
                    remaining_gaps.remove(gap)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            raise RuntimeError("failed to place all cups")

    if not _validate_generated_grid(grid):
        raise RuntimeError("generated scene failed semantic validation")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _generate_attempt()
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 4")
