from __future__ import annotations

import numpy as np


BG = 0
FRAME = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GAP_FROM_DIRECTION = {"up": "bottom", "down": "top", "left": "right", "right": "left"}
DIR_FROM_GAP = {v: k for k, v in GAP_FROM_DIRECTION.items()}
TARGET_RESERVE = 5
MAX_SHAPE_DIM = 4
MAX_ATTEMPTS = 400
PLACEMENT_TRIES = 80


def _components_same_color(grid: np.ndarray, forbidden: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in forbidden or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            objects.append({"color": color, "coords": coords})
    return objects


def _find_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    coords = np.argwhere(grid == FRAME)
    assert coords.size > 0, "expected a grey frame"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4, "frame too small"

    for r, c in coords:
        rr = int(r)
        cc = int(c)
        assert rr in (r0, r1) or cc in (c0, c1), "grey pixels must lie on the frame border"
    assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == FRAME), "frame interior cannot contain grey"

    sides = {
        "top": grid[r0, c0 : c1 + 1] == FRAME,
        "bottom": grid[r1, c0 : c1 + 1] == FRAME,
        "left": grid[r0 : r1 + 1, c0] == FRAME,
        "right": grid[r0 : r1 + 1, c1] == FRAME,
    }
    gap_sides = [name for name, occ in sides.items() if not bool(np.all(occ))]
    assert len(gap_sides) == 1, "expected exactly one side gap"
    gap_side = gap_sides[0]

    missing = np.flatnonzero(~sides[gap_side])
    assert len(missing) > 0, "gap cannot be empty"
    assert int(missing[-1]) - int(missing[0]) + 1 == len(missing), "gap must be contiguous"
    return (r0, c0, r1, c1), gap_side


def _objects_inside(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    objects = _components_same_color(inner, forbidden={BG, FRAME})
    assert 2 <= len(objects) <= 4, "expected 2-4 colored objects"

    out: list[dict] = []
    for obj in objects:
        color = int(obj["color"])
        coords = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in obj["coords"]]
        assert 3 <= len(coords) <= 8, "objects must have 3-8 pixels"
        out.append({"color": color, "coords": coords})
    return out


def _line_profiles(obj: dict, direction: str, frame_bbox: tuple[int, int, int, int]) -> dict[int, tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    top_inner = r0 + 1
    bottom_inner = r1 - 1
    left_inner = c0 + 1
    right_inner = c1 - 1
    coords = obj["coords"]

    if direction in ("up", "down"):
        by_col: dict[int, list[int]] = {}
        for r, c in coords:
            by_col.setdefault(int(c), []).append(int(r))
        profiles = {}
        for col, rows in by_col.items():
            count = len(rows)
            if direction == "down":
                front = bottom_inner - max(rows)
            else:
                front = min(rows) - top_inner
            profiles[int(col)] = (int(count), int(front))
        return profiles

    by_row: dict[int, list[int]] = {}
    for r, c in coords:
        by_row.setdefault(int(r), []).append(int(c))
    profiles = {}
    for row, cols in by_row.items():
        count = len(cols)
        if direction == "right":
            front = right_inner - max(cols)
        else:
            front = min(cols) - left_inner
        profiles[int(row)] = (int(count), int(front))
    return profiles


def _project_candidates(obj: dict, direction: str, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = frame_bbox
    top_inner = r0 + 1
    bottom_inner = r1 - 1
    left_inner = c0 + 1
    right_inner = c1 - 1
    color = int(obj["color"])
    profiles = _line_profiles(obj, direction, frame_bbox)

    out: list[tuple[int, int, int, int]] = []
    if direction == "down":
        for col, (count, front) in profiles.items():
            for k in range(count):
                out.append((bottom_inner - k, int(col), int(front), color))
    elif direction == "up":
        for col, (count, front) in profiles.items():
            for k in range(count):
                out.append((top_inner + k, int(col), int(front), color))
    elif direction == "right":
        for row, (count, front) in profiles.items():
            for k in range(count):
                out.append((int(row), right_inner - k, int(front), color))
    else:
        for row, (count, front) in profiles.items():
            for k in range(count):
                out.append((int(row), left_inner + k, int(front), color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the frame gap to infer the opposite projection direction, then compress each
    interior object into edge-attached strips along that direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox, gap_side = _find_frame(g)
    direction = DIR_FROM_GAP[gap_side]
    objects = _objects_inside(g, frame_bbox)

    best: dict[tuple[int, int], tuple[int, int, int]] = {}
    for obj_idx, obj in enumerate(objects):
        for rr, cc, front, color in _project_candidates(obj, direction, frame_bbox):
            key = (int(rr), int(cc))
            prev = best.get(key)
            if prev is None or int(front) < int(prev[0]):
                best[key] = (int(front), int(obj_idx), int(color))

    out = g.copy()
    for (r, c), (_front, _idx, color) in best.items():
        if int(out[r, c]) == BG:
            out[r, c] = int(color)
    return out


def _grow_polyomino(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        seen_frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in cells or (nr, nc) in seen_frontier:
                    continue
                seen_frontier.add((nr, nc))
                frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _random_shape(rng: np.random.Generator) -> dict:
    area = int(rng.integers(3, 9))
    dims = [(h, w) for h in range(1, MAX_SHAPE_DIM + 1) for w in range(1, MAX_SHAPE_DIM + 1) if h * w >= area]
    h0, w0 = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_polyomino(rng, h0, w0, area)
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    rmin, cmin = min(rs), min(cs)
    rel = sorted((int(r - rmin), int(c - cmin)) for r, c in cells)
    h = max(r for r, _c in rel) + 1
    w = max(c for _r, c in rel) + 1
    return {"cells": rel, "area": area, "h": h, "w": w}


def _placement_window(ih: int, iw: int, shape: dict, direction: str) -> tuple[int, int, int, int] | None:
    h = int(shape["h"])
    w = int(shape["w"])
    if direction == "down":
        min_r, max_r = 1, ih - h - TARGET_RESERVE
        min_c, max_c = 1, iw - w - 1
    elif direction == "up":
        min_r, max_r = TARGET_RESERVE, ih - h - 1
        min_c, max_c = 1, iw - w - 1
    elif direction == "right":
        min_r, max_r = 1, ih - h - 1
        min_c, max_c = 1, iw - w - TARGET_RESERVE
    else:
        min_r, max_r = 1, ih - h - 1
        min_c, max_c = TARGET_RESERVE, iw - w - 1
    if max_r < min_r or max_c < min_c:
        return None
    return min_r, max_r, min_c, max_c


def _enumerate_positions(shape: dict, ih: int, iw: int, direction: str) -> list[dict]:
    window = _placement_window(ih, iw, shape, direction)
    if window is None:
        return []
    min_r, max_r, min_c, max_c = window
    positions: list[dict] = []
    for top in range(min_r, max_r + 1):
        for left in range(min_c, max_c + 1):
            abs_cells = tuple((top + r, left + c) for r, c in shape["cells"])
            if direction in ("up", "down"):
                lines = frozenset(int(left + c) for _r, c in shape["cells"])
            else:
                lines = frozenset(int(top + r) for r, _c in shape["cells"])
            positions.append({"top": top, "left": left, "abs_cells": abs_cells, "lines": lines})
    return positions


def _spatially_clear(abs_cells: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    for r, c in abs_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (int(r + dr), int(c + dc)) in occupied:
                    return False
    return True


def _projection_overlap_signature(objects: list[dict], direction: str, ih: int, iw: int) -> tuple[bool, bool]:
    fake_bbox = (-1, -1, ih, iw)
    by_line: dict[int, list[tuple[int, int, int]]] = {}
    for idx, obj in enumerate(objects):
        for line, (count, front) in _line_profiles(obj, direction, fake_bbox).items():
            by_line.setdefault(int(line), []).append((int(front), int(count), int(idx)))

    has_overlap = False
    has_tie = False
    for entries in by_line.values():
        if len(entries) < 2:
            continue
        has_overlap = True
        fronts = [front for front, _count, _idx in entries]
        if len(fronts) != len(set(fronts)):
            has_tie = True
            break
    return has_overlap, has_tie


def _choose(rng: np.random.Generator, items: list[dict]) -> dict:
    return items[int(rng.integers(0, len(items)))]


def _place_objects(
    rng: np.random.Generator,
    specs: list[dict],
    ih: int,
    iw: int,
    direction: str,
    overlap_case: bool,
) -> list[dict] | None:
    candidates = [_enumerate_positions(spec["shape"], ih, iw, direction) for spec in specs]
    if any(len(cands) == 0 for cands in candidates):
        return None

    if not overlap_case:
        for _ in range(PLACEMENT_TRIES):
            occupied: set[tuple[int, int]] = set()
            used_lines: set[int] = set()
            placed: list[dict] = []
            ok = True
            for spec, cands in zip(specs, candidates):
                valid = [
                    cand
                    for cand in cands
                    if _spatially_clear(cand["abs_cells"], occupied) and cand["lines"].isdisjoint(used_lines)
                ]
                if not valid:
                    ok = False
                    break
                chosen = _choose(rng, valid)
                occupied.update((int(r), int(c)) for r, c in chosen["abs_cells"])
                used_lines.update(int(line) for line in chosen["lines"])
                placed.append(
                    {
                        "color": int(spec["color"]),
                        "coords": [(int(r), int(c)) for r, c in chosen["abs_cells"]],
                    }
                )
            if ok:
                return placed
        return None

    for _ in range(PLACEMENT_TRIES):
        occupied: set[tuple[int, int]] = set()
        first = _choose(rng, candidates[0])
        occupied.update((int(r), int(c)) for r, c in first["abs_cells"])
        valid_second = [
            cand
            for cand in candidates[1]
            if not cand["lines"].isdisjoint(first["lines"]) and _spatially_clear(cand["abs_cells"], occupied)
        ]
        if not valid_second:
            continue

        second = _choose(rng, valid_second)
        occupied.update((int(r), int(c)) for r, c in second["abs_cells"])
        placed = [
            {"color": int(specs[0]["color"]), "coords": [(int(r), int(c)) for r, c in first["abs_cells"]]},
            {"color": int(specs[1]["color"]), "coords": [(int(r), int(c)) for r, c in second["abs_cells"]]},
        ]

        ok = True
        for spec, cands in zip(specs[2:], candidates[2:]):
            valid = [cand for cand in cands if _spatially_clear(cand["abs_cells"], occupied)]
            if not valid:
                ok = False
                break
            chosen = _choose(rng, valid)
            occupied.update((int(r), int(c)) for r, c in chosen["abs_cells"])
            placed.append(
                {
                    "color": int(spec["color"]),
                    "coords": [(int(r), int(c)) for r, c in chosen["abs_cells"]],
                }
            )
        if not ok:
            continue

        has_overlap, has_tie = _projection_overlap_signature(placed, direction, ih, iw)
        if has_overlap and not has_tie:
            return placed
    return None


def _draw_frame_with_gap(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    gap_side: str,
    gap_start: int,
    gap_len: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    if gap_side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif gap_side == "bottom":
        grid[r1, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif gap_side == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_len, c0] = BG
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_len, c1] = BG


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    direction = ["up", "down", "left", "right"][int(rng.integers(0, 4))]
    overlap_case = bool(int(rng.integers(0, 2)))
    object_count = int(rng.integers(2, 5))

    shapes = [_random_shape(rng) for _ in range(object_count)]
    line_counts = [
        len({c for _r, c in shape["cells"]}) if direction in ("up", "down") else len({r for r, _c in shape["cells"]})
        for shape in shapes
    ]

    if direction in ("up", "down"):
        min_iw = max(14, sum(line_counts) + 2) if not overlap_case else 14
        min_ih = 14
    else:
        min_iw = 14
        min_ih = max(14, sum(line_counts) + 2) if not overlap_case else 14
    if min_ih > 18 or min_iw > 18:
        return None

    ih = int(rng.integers(min_ih, 19))
    iw = int(rng.integers(min_iw, 19))
    colors = [int(c) for c in rng.choice(PALETTE, size=object_count, replace=False)]

    specs = [{"shape": shape, "color": color} for shape, color in zip(shapes, colors)]
    specs.sort(key=lambda spec: (-int(spec["shape"]["area"]), -int(spec["shape"]["h"]) - int(spec["shape"]["w"])))

    placed = _place_objects(rng, specs, ih, iw, direction, overlap_case)
    if placed is None:
        return None

    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))

    frame_h = ih + 2
    frame_w = iw + 2
    grid = np.zeros((top_margin + frame_h + bottom_margin, left_margin + frame_w + right_margin), dtype=int)
    frame_bbox = (
        top_margin,
        left_margin,
        top_margin + frame_h - 1,
        left_margin + frame_w - 1,
    )

    gap_side = GAP_FROM_DIRECTION[direction]
    side_len = frame_w if gap_side in ("top", "bottom") else frame_h
    max_gap = side_len - 4
    if max_gap < 2:
        return None
    gap_len = int(rng.integers(2, max_gap + 1))
    gap_start = int(rng.integers(1, side_len - gap_len))
    _draw_frame_with_gap(grid, frame_bbox, gap_side, gap_start, gap_len)

    fr0, fc0, _fr1, _fc1 = frame_bbox
    for obj in placed:
        for r, c in obj["coords"]:
            grid[fr0 + 1 + int(r), fc0 + 1 + int(c)] = int(obj["color"])

    has_overlap, has_tie = _projection_overlap_signature(placed, direction, ih, iw)
    if overlap_case:
        if not has_overlap or has_tie:
            return None
    elif has_overlap:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
