from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
BOARD = 8
LINE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _board_and_lines(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int, int, int], np.ndarray, dict[int, dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    non_bg = g != bg
    h, w = g.shape
    best: tuple[int, int, int, int, int] | None = None
    for r0 in range(h):
        good = np.ones(w, dtype=bool)
        for r1 in range(r0, h):
            good &= non_bg[r1]
            c = 0
            while c < w:
                if not good[c]:
                    c += 1
                    continue
                c0 = c
                while c < w and good[c]:
                    c += 1
                c1 = c - 1
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                cand = (area, r0, c0, r1, c1)
                if best is None or cand > best:
                    best = cand
    assert best is not None, "expected a dense non-background board"
    _, br0, bc0, br1, bc1 = best
    br0, bc0, br1, bc1 = map(int, (br0, bc0, br1, bc1))
    board = g[br0 : br1 + 1, bc0 : bc1 + 1].copy()
    assert not np.any(board == bg), "board bbox must not contain background"
    vals, counts = np.unique(board, return_counts=True)
    board_color = int(vals[np.argmax(counts)])

    lines: dict[int, dict[str, int]] = {}
    for color in np.unique(board):
        color = int(color)
        if color == board_color:
            continue
        pts = np.argwhere(board == color)
        rows = np.unique(pts[:, 0])
        cols = np.unique(pts[:, 1])
        if len(rows) == 1 and len(cols) >= 2:
            coord = int(rows[0])
            dist_key = abs(2 * coord - (board.shape[0] - 1))
            lines[color] = {"axis": "h", "coord": coord, "dkey": int(dist_key)}
        elif len(cols) == 1 and len(rows) >= 2:
            coord = int(cols[0])
            dist_key = abs(2 * coord - (board.shape[1] - 1))
            lines[color] = {"axis": "v", "coord": coord, "dkey": int(dist_key)}
        else:
            raise AssertionError("each crease color must identify one board row or one board column")

    assert lines, "expected at least one crease line"
    return bg, board_color, (br0, bc0, br1, bc1), board, lines


def _classify_side(
    bbox: tuple[int, int, int, int], board_bbox: tuple[int, int, int, int]
) -> str:
    r0, c0, r1, c1 = bbox
    br0, bc0, br1, bc1 = board_bbox
    if r1 < br0 and bc0 <= c1 and c0 <= bc1:
        return "top"
    if r0 > br1 and bc0 <= c1 and c0 <= bc1:
        return "bottom"
    if c1 < bc0 and br0 <= r1 and r0 <= br1:
        return "left"
    if c0 > bc1 and br0 <= r1 and r0 <= br1:
        return "right"
    raise AssertionError("external fragment must lie on exactly one side of the board")


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, board_color, board_bbox, board, lines = _board_and_lines(g)
    br0, bc0, br1, bc1 = board_bbox

    outside = np.zeros_like(g, dtype=bool)
    outside[g != bg] = True
    outside[br0 : br1 + 1, bc0 : bc1 + 1] = False

    fragments = []
    for comp in _components(outside):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each fragment must be monochrome"
        color = int(next(iter(colors)))
        assert color in lines, "fragment color must match a board crease line"
        bbox = _bbox(comp)
        side = _classify_side(bbox, board_bbox)
        info = lines[color]
        if side in ("top", "bottom"):
            assert info["axis"] == "h", "top/bottom fragments must match horizontal creases"
            line_abs = br0 + int(info["coord"])
        else:
            assert info["axis"] == "v", "left/right fragments must match vertical creases"
            line_abs = bc0 + int(info["coord"])
        fragments.append(
            {
                "color": color,
                "cells": comp,
                "bbox": bbox,
                "side": side,
                "axis": info["axis"],
                "line_abs": int(line_abs),
                "dkey": int(info["dkey"]),
            }
        )

    out = board.copy()
    line_mask = board != board_color
    fragments.sort(key=lambda item: (-int(item["dkey"]), item["side"], int(item["color"])))

    for frag in fragments:
        r0, c0, r1, c1 = frag["bbox"]
        color = int(frag["color"])
        if frag["side"] == "top":
            delta_r = br0 - 1 - r1
            for r, c in frag["cells"]:
                aligned_r = r + delta_r
                rr = 2 * int(frag["line_abs"]) - aligned_r
                cc = c
                if br0 <= rr <= br1 and bc0 <= cc <= bc1:
                    out[rr - br0, cc - bc0] = color
        elif frag["side"] == "bottom":
            delta_r = br1 + 1 - r0
            for r, c in frag["cells"]:
                aligned_r = r + delta_r
                rr = 2 * int(frag["line_abs"]) - aligned_r
                cc = c
                if br0 <= rr <= br1 and bc0 <= cc <= bc1:
                    out[rr - br0, cc - bc0] = color
        elif frag["side"] == "left":
            delta_c = bc0 - 1 - c1
            for r, c in frag["cells"]:
                aligned_c = c + delta_c
                rr = r
                cc = 2 * int(frag["line_abs"]) - aligned_c
                if br0 <= rr <= br1 and bc0 <= cc <= bc1:
                    out[rr - br0, cc - bc0] = color
        else:
            delta_c = bc1 + 1 - c0
            for r, c in frag["cells"]:
                aligned_c = c + delta_c
                rr = r
                cc = 2 * int(frag["line_abs"]) - aligned_c
                if br0 <= rr <= br1 and bc0 <= cc <= bc1:
                    out[rr - br0, cc - bc0] = color

    out[line_mask] = board[line_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each outside fragment across the board crease with the same color.

    The board is the central azure rectangle.  Fragments above/below fold across horizontal
    creases; fragments left/right fold across vertical creases.  Folds farther from the
    center happen first, so creases closer to the center remain on top when overlaps occur.
    The output is the cropped board after all folds.
    """
    return _solve(grid)


def _distance_key(axis: str, coord: int, board_h: int, board_w: int) -> int:
    if axis == "h":
        return abs(2 * coord - (board_h - 1))
    return abs(2 * coord - (board_w - 1))


def _candidate_positions(length: int, side: str) -> list[int]:
    if side in ("top", "left"):
        hi = (length - 2) // 2
        return list(range(1, hi + 1))
    lo = (length + 1) // 2
    return list(range(lo, length - 1))


def _reachable_and_boundary(
    spec: dict[str, object], board_h: int, board_w: int, line_mask: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    coord = int(spec["coord"])
    side = str(spec["side"])
    allowed = ~line_mask.copy()

    if side == "top":
        boundary_row = 2 * coord + 1
        allowed[:boundary_row, :] = False
        boundary = [(boundary_row, c) for c in range(board_w) if allowed[boundary_row, c]]
    elif side == "bottom":
        boundary_row = 2 * coord - board_h
        allowed[boundary_row + 1 :, :] = False
        boundary = [(boundary_row, c) for c in range(board_w) if allowed[boundary_row, c]]
    elif side == "left":
        boundary_col = 2 * coord + 1
        allowed[:, :boundary_col] = False
        boundary = [(r, boundary_col) for r in range(board_h) if allowed[r, boundary_col]]
    else:
        boundary_col = 2 * coord - board_w
        allowed[:, boundary_col + 1 :] = False
        boundary = [(r, boundary_col) for r in range(board_h) if allowed[r, boundary_col]]

    return allowed, boundary


def _grow_region(
    allowed: np.ndarray,
    rng: random.Random,
    seed: tuple[int, int],
    target_size: int,
    max_h: int = 5,
    max_w: int = 5,
) -> set[tuple[int, int]] | None:
    if not allowed[seed]:
        return None
    cells = {seed}

    while len(cells) < target_size:
        r0, c0, r1, c1 = _bbox(cells)
        candidates: list[tuple[int, int]] = []
        for r, c in list(cells):
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1]):
                    continue
                if not allowed[nr, nc] or (nr, nc) in cells:
                    continue
                nr0, nc0 = min(r0, nr), min(c0, nc)
                nr1, nc1 = max(r1, nr), max(c1, nc)
                if nr1 - nr0 + 1 <= max_h and nc1 - nc0 + 1 <= max_w:
                    candidates.append((nr, nc))
        if not candidates:
            return None
        cells.add(rng.choice(candidates))

    return cells


def _actual_rel_cells(
    final_cells: set[tuple[int, int]],
    spec: dict[str, object],
    gap: int,
) -> set[tuple[int, int]]:
    coord = int(spec["coord"])
    side = str(spec["side"])
    actual: set[tuple[int, int]] = set()
    for r, c in final_cells:
        if side == "top":
            actual.add((2 * coord - r - gap, c))
        elif side == "bottom":
            actual.add((2 * coord - r + gap, c))
        elif side == "left":
            actual.add((r, 2 * coord - c - gap))
        else:
            actual.add((r, 2 * coord - c + gap))
    return actual


def _touches_existing(
    cells: set[tuple[int, int]], occupied: set[tuple[int, int]]
) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return True
        for dr, dc in DIR4:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _render_expected(
    board: np.ndarray,
    board_color: int,
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = board.copy()
    line_mask = board != board_color
    paint_order = sorted(specs, key=lambda item: -int(item["dkey"]))
    for spec in paint_order:
        color = int(spec["color"])
        for r, c in spec["final_cells"]:
            out[r, c] = color
    out[line_mask] = board[line_mask]
    return out


def _fragment_visibility_ok(
    board: np.ndarray,
    board_color: int,
    specs: list[dict[str, object]],
) -> bool:
    line_cells = {(int(r), int(c)) for r, c in np.argwhere(board != board_color)}
    occluders = set(line_cells)
    for spec in reversed(sorted(specs, key=lambda item: -int(item["dkey"]))):
        region = set(spec["final_cells"])
        if not (region - occluders):
            return False
        occluders |= region
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    board_h = rng.randint(9, 15)
    board_w = rng.randint(9, 15)
    mode = rng.choice(("h", "v", "hv"))
    num_h = rng.randint(1, 2) if mode in ("h", "hv") else 0
    num_v = rng.randint(1, 2) if mode in ("v", "hv") else 0

    tasks: list[tuple[str, str]] = []
    tasks.extend(("h", rng.choice(("top", "bottom"))) for _ in range(num_h))
    tasks.extend(("v", rng.choice(("left", "right"))) for _ in range(num_v))
    rng.shuffle(tasks)

    specs: list[dict[str, object]] = []
    used_dkeys: set[int] = set()
    used_rows: list[int] = []
    used_cols: list[int] = []

    for axis, side in tasks:
        candidates = _candidate_positions(board_h if axis == "h" else board_w, side)
        if axis == "h":
            candidates = [
                pos
                for pos in candidates
                if all(abs(pos - other) >= 2 for other in used_rows)
                and _distance_key(axis, pos, board_h, board_w) not in used_dkeys
            ]
        else:
            candidates = [
                pos
                for pos in candidates
                if all(abs(pos - other) >= 2 for other in used_cols)
                and _distance_key(axis, pos, board_h, board_w) not in used_dkeys
            ]
        if not candidates:
            return None
        coord = rng.choice(candidates)
        dkey = _distance_key(axis, coord, board_h, board_w)
        used_dkeys.add(dkey)
        if axis == "h":
            used_rows.append(coord)
        else:
            used_cols.append(coord)
        specs.append({"axis": axis, "side": side, "coord": coord, "dkey": dkey})

    if len(specs) > len(LINE_COLORS):
        return None

    colors = list(LINE_COLORS)
    rng.shuffle(colors)
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)

    board = np.full((board_h, board_w), BOARD, dtype=int)
    draw_order = specs[:]
    rng.shuffle(draw_order)
    for spec in draw_order:
        color = int(spec["color"])
        coord = int(spec["coord"])
        if spec["axis"] == "h":
            board[coord, :] = color
        else:
            board[:, coord] = color

    line_mask = board != BOARD
    want_overlap = len(specs) >= 2 and rng.random() < 0.7
    occupied_actual: set[tuple[int, int]] = set()
    prior_final: set[tuple[int, int]] = set()

    for spec in sorted(specs, key=lambda item: -int(item["dkey"])):
        allowed, boundary = _reachable_and_boundary(spec, board_h, board_w, line_mask)
        if not boundary:
            return None

        overlap_boundary = [cell for cell in boundary if cell in prior_final]
        seed_candidates = overlap_boundary if want_overlap and overlap_boundary else boundary
        placed = False
        for _ in range(80):
            seed = rng.choice(seed_candidates)
            max_area = min(6, int(allowed.sum()))
            if max_area < 2:
                break
            target_size = rng.randint(2, max_area)
            region = _grow_region(allowed, rng, seed, target_size)
            if region is None:
                continue
            gaps = [0, 1, 2, 3]
            rng.shuffle(gaps)
            for gap in gaps:
                actual = _actual_rel_cells(region, spec, gap)
                if _touches_existing(actual, occupied_actual):
                    continue
                spec["final_cells"] = region
                spec["actual_rel_cells"] = actual
                occupied_actual |= actual
                prior_final |= region
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    overlap_count: dict[tuple[int, int], int] = {}
    for spec in specs:
        for cell in spec["final_cells"]:
            overlap_count[cell] = overlap_count.get(cell, 0) + 1
    if want_overlap and not any(v >= 2 for v in overlap_count.values()):
        return None
    if not _fragment_visibility_ok(board, BOARD, specs):
        return None

    top_req = 0
    bottom_req = 0
    left_req = 0
    right_req = 0
    for spec in specs:
        rows = [r for r, _ in spec["actual_rel_cells"]]
        cols = [c for _, c in spec["actual_rel_cells"]]
        top_req = max(top_req, -min(rows))
        bottom_req = max(bottom_req, max(rows) - (board_h - 1))
        left_req = max(left_req, -min(cols))
        right_req = max(right_req, max(cols) - (board_w - 1))

    top_margin = top_req + rng.randint(1, 3)
    bottom_margin = bottom_req + rng.randint(1, 3)
    left_margin = left_req + rng.randint(1, 3)
    right_margin = right_req + rng.randint(1, 3)

    if abs(top_margin - bottom_margin) > 3:
        if top_margin < bottom_margin:
            top_margin = max(top_req + 1, bottom_margin - rng.randint(0, 3))
        else:
            bottom_margin = max(bottom_req + 1, top_margin - rng.randint(0, 3))
    if abs(left_margin - right_margin) > 3:
        if left_margin < right_margin:
            left_margin = max(left_req + 1, right_margin - rng.randint(0, 3))
        else:
            right_margin = max(right_req + 1, left_margin - rng.randint(0, 3))

    height = top_margin + board_h + bottom_margin
    width = left_margin + board_w + right_margin
    if not (16 <= height <= 30 and 16 <= width <= 30):
        return None

    br0 = top_margin
    bc0 = left_margin
    grid = np.full((height, width), BG, dtype=int)
    grid[br0 : br0 + board_h, bc0 : bc0 + board_w] = board

    for spec in specs:
        color = int(spec["color"])
        for rr, cc in spec["actual_rel_cells"]:
            ar = br0 + rr
            ac = bc0 + cc
            if not (0 <= ar < height and 0 <= ac < width):
                return None
            grid[ar, ac] = color

    expected = _render_expected(board, BOARD, specs)
    try:
        solved = _solve(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    if int((grid == BOARD).sum()) <= max(int((grid == color).sum()) for color in LINE_COLORS):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
