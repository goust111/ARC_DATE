from __future__ import annotations

from collections import defaultdict
import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords.sort()
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_square(comp: dict) -> bool:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    if len(coords) != 9 or r1 - r0 != 2 or c1 - c0 != 2:
        return False
    coord_set = set(coords)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if (rr, cc) not in coord_set:
                return False
    return True


def _is_line(comp: dict) -> bool:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    if not (3 <= len(coords) <= 8):
        return False
    coord_set = set(coords)
    if r0 == r1:
        return all((r0, cc) in coord_set for cc in range(c0, c1 + 1))
    if c0 == c1:
        return all((rr, c0) in coord_set for rr in range(r0, r1 + 1))
    return False


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    squares: list[dict] = []
    lines: list[dict] = []
    for comp in _components_by_color(g, bg):
        if _is_square(comp):
            r0, c0, r1, c1 = comp["bbox"]
            squares.append(
                {
                    "color": int(comp["color"]),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )
        elif _is_line(comp):
            r0, c0, r1, c1 = comp["bbox"]
            if r0 == r1:
                lines.append(
                    {
                        "color": int(comp["color"]),
                        "axis": "h",
                        "row": int(r0),
                        "start": int(c0),
                        "end": int(c1),
                        "cells": [(int(r0), cc) for cc in range(c0, c1 + 1)],
                    }
                )
            else:
                lines.append(
                    {
                        "color": int(comp["color"]),
                        "axis": "v",
                        "col": int(c0),
                        "start": int(r0),
                        "end": int(r1),
                        "cells": [(rr, int(c0)) for rr in range(r0, r1 + 1)],
                    }
                )
    return bg, squares, lines


def _line_candidates(line: dict, squares: list[dict]) -> list[dict]:
    cands: list[dict] = []
    if line["axis"] == "h":
        row = int(line["row"])
        left_end = int(line["start"])
        right_end = int(line["end"])
        for idx, sq in enumerate(squares):
            if not (sq["r0"] <= row <= sq["r1"]):
                continue
            if sq["c1"] < left_end:
                dist = left_end - sq["c1"] - 1
                if dist >= 1:
                    cands.append(
                        {
                            "square_idx": idx,
                            "distance": int(dist),
                            "pointing_end": "left",
                            "tail": (row, right_end),
                            "tail_dir": (0, 1),
                        }
                    )
            if sq["c0"] > right_end:
                dist = sq["c0"] - right_end - 1
                if dist >= 1:
                    cands.append(
                        {
                            "square_idx": idx,
                            "distance": int(dist),
                            "pointing_end": "right",
                            "tail": (row, left_end),
                            "tail_dir": (0, -1),
                        }
                    )
    else:
        col = int(line["col"])
        top_end = int(line["start"])
        bottom_end = int(line["end"])
        for idx, sq in enumerate(squares):
            if not (sq["c0"] <= col <= sq["c1"]):
                continue
            if sq["r1"] < top_end:
                dist = top_end - sq["r1"] - 1
                if dist >= 1:
                    cands.append(
                        {
                            "square_idx": idx,
                            "distance": int(dist),
                            "pointing_end": "up",
                            "tail": (bottom_end, col),
                            "tail_dir": (1, 0),
                        }
                    )
            if sq["r0"] > bottom_end:
                dist = sq["r0"] - bottom_end - 1
                if dist >= 1:
                    cands.append(
                        {
                            "square_idx": idx,
                            "distance": int(dist),
                            "pointing_end": "down",
                            "tail": (top_end, col),
                            "tail_dir": (-1, 0),
                        }
                    )
    return cands


def _analyze_line(line: dict, squares: list[dict]) -> dict | None:
    candidates = _line_candidates(line, squares)
    if not candidates:
        return None

    grouped: dict[str, dict] = {}
    for cand in candidates:
        key = cand["pointing_end"]
        best = grouped.get(key)
        if best is None or cand["distance"] < best["distance"]:
            grouped[key] = cand

    if len(grouped) == 1:
        best = next(iter(grouped.values()))
    else:
        best = min(grouped.values(), key=lambda cand: (cand["distance"], cand["square_idx"]))
    tail_r, tail_c = best["tail"]
    dr, dc = best["tail_dir"]
    plus_center = (tail_r + 2 * dr, tail_c + 2 * dc)
    return {
        "line": line,
        "square_idx": int(best["square_idx"]),
        "distance": int(best["distance"]),
        "tail": (int(tail_r), int(tail_c)),
        "tail_dir": (int(dr), int(dc)),
        "plus_center": (int(plus_center[0]), int(plus_center[1])),
    }


def _plus_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat line segments as pointers.

    - The pointed square receives a 1-pixel border in the line color.
    - The line tail receives a nearby '+' marker in the square's original color.
    - The line itself stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _bg, squares, lines = _extract_scene(g)
    out = g.copy()

    analyses = [info for line in lines if (info := _analyze_line(line, squares)) is not None]

    best_border: dict[int, tuple[int, int]] = {}
    for info in analyses:
        square_idx = int(info["square_idx"])
        line_color = int(info["line"]["color"])
        distance = int(info["distance"])
        prev = best_border.get(square_idx)
        if prev is None or distance < prev[0]:
            best_border[square_idx] = (distance, line_color)

    for square_idx, (_distance, border_color) in best_border.items():
        sq = squares[square_idx]
        for rr in range(sq["r0"] - 1, sq["r1"] + 2):
            for cc in range(sq["c0"] - 1, sq["c1"] + 2):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if rr in (sq["r0"] - 1, sq["r1"] + 1) or cc in (sq["c0"] - 1, sq["c1"] + 1):
                    out[rr, cc] = int(border_color)

    line_cells = {cell for line in lines for cell in line["cells"]}
    for info in analyses:
        square_color = int(squares[int(info["square_idx"])]["color"])
        for rr, cc in _plus_cells(info["plus_center"]):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in line_cells:
                out[rr, cc] = square_color

    return out


def _cells_around(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _square_reserve(square: dict, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rr in range(square["r0"] - 1, square["r1"] + 2):
        for cc in range(square["c0"] - 1, square["c1"] + 2):
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
    return cells


def _line_geometry(square: dict, side: str, slot: int, gap: int, length: int) -> dict | None:
    if side == "left":
        row = square["r0"] + slot
        near = square["c0"] - gap - 1
        tail = near - length + 1
        line_cells = [(row, cc) for cc in range(tail, near + 1)]
        plus_center = (row, tail - 2)
    elif side == "right":
        row = square["r0"] + slot
        near = square["c1"] + gap + 1
        tail = near + length - 1
        line_cells = [(row, cc) for cc in range(near, tail + 1)]
        plus_center = (row, tail + 2)
    elif side == "up":
        col = square["c0"] + slot
        near = square["r0"] - gap - 1
        tail = near - length + 1
        line_cells = [(rr, col) for rr in range(tail, near + 1)]
        plus_center = (tail - 2, col)
    else:
        col = square["c0"] + slot
        near = square["r1"] + gap + 1
        tail = near + length - 1
        line_cells = [(rr, col) for rr in range(near, tail + 1)]
        plus_center = (tail + 2, col)
    return {
        "side": side,
        "slot": int(slot),
        "gap": int(gap),
        "length": int(length),
        "line_cells": line_cells,
        "plus_center": plus_center,
        "plus_cells": _plus_cells(plus_center),
    }


def _in_bounds(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _enumerate_candidates(squares: list[dict], h: int, w: int) -> dict[int, list[dict]]:
    reserve_union = set()
    for square in squares:
        reserve_union |= _square_reserve(square, h, w)

    candidates: dict[int, list[dict]] = defaultdict(list)
    for sq_idx, square in enumerate(squares):
        for side in ("left", "right", "up", "down"):
            for slot in range(3):
                for gap in range(1, 5):
                    for length in range(3, 9):
                        geom = _line_geometry(square, side, slot, gap, length)
                        actual = geom["line_cells"] + geom["plus_cells"]
                        if not _in_bounds(actual, h, w):
                            continue
                        if any(cell in reserve_union for cell in actual):
                            continue
                        line = {
                            "color": 1,
                            "axis": "h" if side in ("left", "right") else "v",
                            "row": geom["line_cells"][0][0] if side in ("left", "right") else None,
                            "col": geom["line_cells"][0][1] if side in ("up", "down") else None,
                            "start": min(v for v, _ in geom["line_cells"]) if side in ("up", "down") else min(v for _, v in geom["line_cells"]),
                            "end": max(v for v, _ in geom["line_cells"]) if side in ("up", "down") else max(v for _, v in geom["line_cells"]),
                            "cells": geom["line_cells"],
                        }
                        analysis = _analyze_line(line, squares)
                        if analysis is None or int(analysis["square_idx"]) != sq_idx:
                            continue
                        point = analysis["tail_dir"]
                        expected_tail_dir = {
                            "left": (0, -1),
                            "right": (0, 1),
                            "up": (-1, 0),
                            "down": (1, 0),
                        }[side]
                        if point != expected_tail_dir:
                            continue
                        endpoint_keys = {cand["pointing_end"] for cand in _line_candidates(line, squares)}
                        if len(endpoint_keys) != 1:
                            continue
                        candidates[sq_idx].append(
                            {
                                "square_idx": sq_idx,
                                "side": side,
                                "slot": int(slot),
                                "distance": int(gap),
                                "line_cells": geom["line_cells"],
                                "plus_cells": geom["plus_cells"],
                                "buffer_cells": _cells_around(actual, h, w),
                                "actual_cells": actual,
                            }
                        )
    return candidates


def _choose_targets(rng: np.random.Generator, candidates: dict[int, list[dict]], color_budget: int) -> list[int] | None:
    eligible = [sq_idx for sq_idx, opts in candidates.items() if opts]
    if len(eligible) < 2 or color_budget < 2:
        return None

    duplicateable = [
        sq_idx
        for sq_idx, opts in candidates.items()
        if len({int(opt["distance"]) for opt in opts}) >= 2
    ]
    duplicate_mode = bool(
        duplicateable
        and len(eligible) >= 2
        and color_budget >= 3
        and rng.random() < 0.5
    )

    if duplicate_mode:
        base = int(rng.choice(np.array(duplicateable, dtype=int)))
        others = [sq_idx for sq_idx in eligible if sq_idx != base]
        max_lines = min(color_budget, 5, len(others) + 2)
        if max_lines < 3:
            return None
        line_count = int(rng.integers(3, max_lines + 1))
        extra = line_count - 2
        chosen = [base, base]
        if extra:
            picks = rng.choice(np.array(others, dtype=int), size=extra, replace=False)
            chosen.extend(int(v) for v in np.atleast_1d(picks))
        rng.shuffle(chosen)
        return chosen

    max_lines = min(color_budget, 4, len(eligible))
    if max_lines < 2:
        return None
    line_count = int(rng.integers(2, max_lines + 1))
    picks = rng.choice(np.array(eligible, dtype=int), size=line_count, replace=False)
    chosen = [int(v) for v in np.atleast_1d(picks)]
    rng.shuffle(chosen)
    return chosen


def _select_line_candidates(
    rng: np.random.Generator,
    target_squares: list[int],
    candidates: dict[int, list[dict]],
    h: int,
    w: int,
) -> list[dict] | None:
    used_buffer = np.zeros((h, w), dtype=bool)
    used_distance: dict[int, set[int]] = defaultdict(set)
    chosen: list[dict] = []

    order = list(target_squares)
    order.sort(key=lambda sq_idx: len(candidates[sq_idx]))

    shuffled_options = {
        sq_idx: [candidates[sq_idx][i] for i in rng.permutation(len(candidates[sq_idx]))]
        for sq_idx in set(order)
    }

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        sq_idx = int(order[pos])
        for opt in shuffled_options[sq_idx]:
            if int(opt["distance"]) in used_distance[sq_idx]:
                continue
            if any(used_buffer[r, c] for r, c in opt["actual_cells"]):
                continue

            changed = []
            for r, c in opt["buffer_cells"]:
                if not used_buffer[r, c]:
                    used_buffer[r, c] = True
                    changed.append((r, c))
            used_distance[sq_idx].add(int(opt["distance"]))
            chosen.append(opt)
            if rec(pos + 1):
                return True
            chosen.pop()
            used_distance[sq_idx].remove(int(opt["distance"]))
            for r, c in changed:
                used_buffer[r, c] = False
        return False

    if rec(0):
        return chosen
    return None


def _place_squares(rng: np.random.Generator, h: int, w: int, n_squares: int) -> list[dict] | None:
    squares: list[dict] = []
    reserve_union: set[tuple[int, int]] = set()
    colors = [int(v) for v in rng.choice(np.arange(1, 10), size=n_squares, replace=False)]
    for idx in range(n_squares):
        placed = False
        for _ in range(200):
            r0 = int(rng.integers(2, h - 4))
            c0 = int(rng.integers(2, w - 4))
            square = {"color": colors[idx], "r0": r0, "c0": c0, "r1": r0 + 2, "c1": c0 + 2}
            reserve = _square_reserve(square, h, w)
            if reserve & reserve_union:
                continue
            squares.append(square)
            reserve_union |= reserve
            placed = True
            break
        if not placed:
            return None
    return squares


def _render_grid(h: int, w: int, squares: list[dict], lines: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for square in squares:
        grid[square["r0"] : square["r1"] + 1, square["c0"] : square["c1"] + 1] = int(square["color"])
    for line in lines:
        for r, c in line["cells"]:
            grid[r, c] = int(line["color"])
    return grid


def _make_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(20, 31))
    w = int(rng.integers(20, 31))
    n_squares = int(rng.integers(2, 5))
    squares = _place_squares(rng, h, w, n_squares)
    if squares is None:
        return None

    candidates = _enumerate_candidates(squares, h, w)
    target_squares = _choose_targets(rng, candidates, 9 - n_squares)
    if target_squares is None:
        return None

    selected = _select_line_candidates(rng, target_squares, candidates, h, w)
    if selected is None:
        return None

    line_colors_pool = [color for color in range(1, 10) if color not in {sq["color"] for sq in squares}]
    if len(line_colors_pool) < len(selected):
        return None
    line_colors = [int(v) for v in rng.choice(np.array(line_colors_pool, dtype=int), size=len(selected), replace=False)]

    lines: list[dict] = []
    for color, opt in zip(line_colors, selected, strict=False):
        cells = list(opt["line_cells"])
        if len({r for r, _ in cells}) == 1:
            lines.append(
                {
                    "color": int(color),
                    "cells": cells,
                    "axis": "h",
                    "row": int(cells[0][0]),
                    "start": int(min(c for _, c in cells)),
                    "end": int(max(c for _, c in cells)),
                }
            )
        else:
            lines.append(
                {
                    "color": int(color),
                    "cells": cells,
                    "axis": "v",
                    "col": int(cells[0][1]),
                    "start": int(min(r for r, _ in cells)),
                    "end": int(max(r for r, _ in cells)),
                }
            )

    grid = _render_grid(h, w, squares, lines)

    bg, parsed_squares, parsed_lines = _extract_scene(grid)
    if bg != 0 or len(parsed_squares) != len(squares) or len(parsed_lines) != len(lines):
        return None

    analyses = [_analyze_line(line, parsed_squares) for line in parsed_lines]
    if any(info is None for info in analyses):
        return None

    per_square_distances: dict[int, list[int]] = defaultdict(list)
    for info in analyses:
        per_square_distances[int(info["square_idx"])].append(int(info["distance"]))
        center = info["plus_center"]
        cells = _plus_cells(center)
        if not _in_bounds(cells, h, w):
            return None
        if any(grid[r, c] != 0 for r, c in cells):
            return None
    if any(len(dists) != len(set(dists)) for dists in per_square_distances.values()):
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
