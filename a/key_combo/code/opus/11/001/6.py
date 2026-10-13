from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color


BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
GREY = int(Color.GREY)
ORANGE = int(Color.ORANGE)
BLACK = int(Color.BLACK)

FILL_BY_COMPLEXITY = {2: GREEN, 3: YELLOW, 4: ORANGE}

MAX_ATTEMPTS = 500
GRID_MIN = 20
GRID_MAX = 28


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _blue_positive_prototypes() -> dict[int, tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]]:
    return {
        2: ({(0, 1), (0, 2), (1, 2)}, {(0, 0), (1, 1)}, (2, 3)),
        4: (
            {(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)},
            {(0, 0), (1, 1), (2, 2)},
            (3, 5),
        ),
    }


def _transpose(
    coords: set[tuple[int, int]], line: set[tuple[int, int]], dims: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    h, w = dims
    return {(c, r) for r, c in coords}, {(c, r) for r, c in line}, (w, h)


def _rot_cw(
    coords: set[tuple[int, int]], line: set[tuple[int, int]], dims: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    h, w = dims
    return (
        {(c, h - 1 - r) for r, c in coords},
        {(c, h - 1 - r) for r, c in line},
        (w, h),
    )


def _rot_ccw(
    coords: set[tuple[int, int]], line: set[tuple[int, int]], dims: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    h, w = dims
    return (
        {(w - 1 - c, r) for r, c in coords},
        {(w - 1 - c, r) for r, c in line},
        (w, h),
    )


def _build_prototypes():
    out: dict[tuple[str, int, int], dict] = {}
    for offset, (coords, line_cells, dims) in _blue_positive_prototypes().items():
        variants = {
            ("blue", 1): (coords, line_cells, dims),
            ("blue", -1): _transpose(coords, line_cells, dims),
            ("red", 1): _rot_cw(coords, line_cells, dims),
            ("red", -1): _rot_ccw(coords, line_cells, dims),
        }
        for (family, sign), (shape, line_ref, (h, w)) in variants.items():
            if family == "blue":
                line_vals = {c - r for r, c in line_ref}
                assert len(line_vals) == 1
                line_anchor = next(iter(line_vals))
                distances = [(c - r) - line_anchor for r, c in shape]
            else:
                line_vals = {r + c for r, c in line_ref}
                assert len(line_vals) == 1
                line_anchor = next(iter(line_vals))
                distances = [(r + c) - line_anchor for r, c in shape]
            extreme = max(sign * d for d in distances)
            tip_coords = [
                rc for rc, d in zip(sorted(shape), distances) if sign * d == extreme
            ]
            assert len(tip_coords) == 1
            assert extreme == offset
            out[(family, sign, offset)] = {
                "shape": sorted(shape),
                "h": h,
                "w": w,
                "line_anchor": line_anchor,
                "tip": tip_coords[0],
            }
    return out


PROTOTYPES = _build_prototypes()


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    blue_cells = [(int(r), int(c)) for r, c in np.argwhere(g == BLUE)]
    red_cells = [(int(r), int(c)) for r, c in np.argwhere(g == RED)]
    assert blue_cells and red_cells

    blue_diffs = {c - r for r, c in blue_cells}
    red_sums = {r + c for r, c in red_cells}
    assert len(blue_diffs) == 1
    assert len(red_sums) == 1
    base_d = next(iter(blue_diffs))
    base_s = next(iter(red_sums))

    grey_comps = _components4(g == GREY)
    assert len(grey_comps) >= 4

    blue_consts = {base_d}
    red_consts = {base_s}
    guides = []

    for comp in grey_comps:
        blue_touch = False
        red_touch = False
        for r, c in comp:
            for nr, nc in _neighbors8(r, c):
                if not _in_bounds((h, w), nr, nc):
                    continue
                if g[nr, nc] == BLUE:
                    blue_touch = True
                elif g[nr, nc] == RED:
                    red_touch = True
        assert blue_touch ^ red_touch

        if blue_touch:
            offsets = [(c - r) - base_d for r, c in comp]
            family = "blue"
            sign = 1 if max(offsets) > 0 else -1
            extreme = max(sign * d for d in offsets)
            tips = [comp[i] for i, d in enumerate(offsets) if sign * d == extreme]
            assert len(tips) == 1
            tip = tips[0]
            new_const = tip[1] - tip[0]
            blue_consts.add(new_const)
        else:
            offsets = [(r + c) - base_s for r, c in comp]
            family = "red"
            sign = 1 if max(offsets) > 0 else -1
            extreme = max(sign * d for d in offsets)
            tips = [comp[i] for i, d in enumerate(offsets) if sign * d == extreme]
            assert len(tips) == 1
            tip = tips[0]
            new_const = tip[0] + tip[1]
            red_consts.add(new_const)

        guides.append(
            {
                "family": family,
                "sign": sign,
                "tip": tip,
                "const": new_const,
                "size": len(comp),
            }
        )

    return {
        "shape": (h, w),
        "base_d": base_d,
        "base_s": base_s,
        "blue_consts": sorted(blue_consts),
        "red_consts": sorted(red_consts),
        "guides": guides,
    }


def _intersection_of(d: int, s: int) -> tuple[int, int] | None:
    if (s - d) % 2 != 0:
        return None
    r = (s - d) // 2
    c = (s + d) // 2
    return int(r), int(c)


def _render_from_constants(
    shape: tuple[int, int], blue_consts: list[int], red_consts: list[int], base_d: int, base_s: int
) -> tuple[np.ndarray, list[dict]]:
    h, w = shape
    out = np.full((h, w), BLACK, dtype=int)
    rhombi: list[dict] = []

    for i in range(len(blue_consts) - 1):
        d_lo = int(blue_consts[i])
        d_hi = int(blue_consts[i + 1])
        if d_hi - d_lo < 2:
            continue
        blue_complexity = 1 if base_d in (d_lo, d_hi) else 2
        for j in range(len(red_consts) - 1):
            s_lo = int(red_consts[j])
            s_hi = int(red_consts[j + 1])
            if s_hi - s_lo < 2:
                continue
            red_complexity = 1 if base_s in (s_lo, s_hi) else 2

            vertices = [
                _intersection_of(d_lo, s_lo),
                _intersection_of(d_hi, s_lo),
                _intersection_of(d_hi, s_hi),
                _intersection_of(d_lo, s_hi),
            ]
            if any(v is None for v in vertices):
                continue
            verts = [v for v in vertices if v is not None]
            if any(
                not (0 < r < h - 1 and 0 < c < w - 1)
                for r, c in verts
            ):
                continue

            cells = []
            for r in range(h):
                for c in range(w):
                    if d_lo < c - r < d_hi and s_lo < r + c < s_hi:
                        cells.append((r, c))
            if not cells:
                continue

            complexity = blue_complexity + red_complexity
            color = FILL_BY_COMPLEXITY[complexity]
            for r, c in cells:
                out[r, c] = color
            rhombi.append(
                {
                    "blue_pair": (d_lo, d_hi),
                    "red_pair": (s_lo, s_hi),
                    "complexity": complexity,
                    "cells": cells,
                }
            )

    for s in red_consts:
        for r in range(h):
            c = s - r
            if 0 <= c < w:
                out[r, c] = RED
    for d in blue_consts:
        for r in range(h):
            c = r + d
            if 0 <= c < w:
                out[r, c] = BLUE

    return out, rhombi


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    info = _parse_input(grid)
    out, rhombi = _render_from_constants(
        info["shape"], info["blue_consts"], info["red_consts"], info["base_d"], info["base_s"]
    )
    return out, {**info, "rhombi": rhombi}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend two crossed diagonals through grey guide tips, erase the guides, and fill the enclosed diamonds by boundary complexity.

    Input structure:
    - The grid has a black background with one blue main diagonal (top-left to bottom-right) and one red main diagonal (top-right to bottom-left).
    - Several grey right-triangle guide objects sit against exactly one of those diagonals; each guide has one unique tip pixel farthest from the diagonal it touches.
    - No generated parallel lines are present yet: only the two main diagonals and the grey guides are visible.

    Rule:
    - For each grey guide touching the blue diagonal, draw a new full blue diagonal parallel to the blue main diagonal through that guide's tip pixel.
    - Do the analogous construction for guides touching the red diagonal, then erase every grey guide pixel.
    - The resulting blue/red line mesh encloses rhombus regions. Keep only rhombi whose four vertices lie strictly inside the grid.
    - If a rhombus uses the original blue diagonal as one of its two blue sides, that blue side-pair contributes 1 to the boundary complexity; otherwise it contributes 2. Apply the same rule to the red side-pair. Fill the rhombus interior with green for total complexity 2, yellow for 3, and orange for 4.
    - Preserve all diagonals. At blue/red crossings, use blue when the crossing is tied.

    Key insight:
    - The grey triangles are construction guides: their remote tips define the offsets of the parallel diagonals, and the presence or absence of the two original diagonals on a rhombus boundary determines its fill color.
    """
    out, _ = _solve(grid)
    return out


def _draw_base_lines(shape: tuple[int, int], base_d: int, base_s: int) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    for r in range(h):
        c = base_s - r
        if 0 <= c < w:
            grid[r, c] = RED
    for r in range(h):
        c = r + base_d
        if 0 <= c < w:
            grid[r, c] = BLUE
    return grid


def _family_offsets(count: int) -> list[int]:
    if count == 2:
        return list(random.choice([[-4, 2], [-2, 4], [-4, 4]]))
    return list(random.choice([[-4, 2, 4], [-4, -2, 4]]))


def _candidate_top_lefts(
    shape: tuple[int, int], family: str, proto: dict, base_d: int, base_s: int
) -> list[tuple[int, int]]:
    h, w = shape
    ph = int(proto["h"])
    pw = int(proto["w"])
    out = []
    if family == "blue":
        anchor = int(proto["line_anchor"])
        for r0 in range(h - ph + 1):
            c0 = r0 + base_d - anchor
            if 0 <= c0 <= w - pw:
                out.append((r0, c0))
    else:
        anchor = int(proto["line_anchor"])
        for r0 in range(h - ph + 1):
            c0 = base_s - r0 - anchor
            if 0 <= c0 <= w - pw:
                out.append((r0, c0))
    random.shuffle(out)
    return out


def _touches_color(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == int(color):
                return True
    return False


def _valid_guide_placement(
    grid: np.ndarray, cells: list[tuple[int, int]], family: str
) -> bool:
    h, w = grid.shape
    want = BLUE if family == "blue" else RED
    forbid = RED if family == "blue" else BLUE

    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False
        for nr, nc in _neighbors8(r, c):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if int(grid[nr, nc]) == GREY:
                return False
            if int(grid[nr, nc]) == forbid:
                return False

    return _touches_color(grid, cells, want)


def _place_guide(
    grid: np.ndarray, family: str, offset: int, base_d: int, base_s: int
) -> bool:
    sign = 1 if offset > 0 else -1
    proto = PROTOTYPES[(family, sign, abs(offset))]
    candidates = _candidate_top_lefts(grid.shape, family, proto, base_d, base_s)

    h, w = grid.shape
    center = ((h - 1) / 2.0, (w - 1) / 2.0)

    def score(pos: tuple[int, int]) -> float:
        r0, c0 = pos
        tip_r, tip_c = proto["tip"]
        tr = r0 + tip_r
        tc = c0 + tip_c
        return abs(tr - center[0]) + abs(tc - center[1]) + random.random() * 0.01

    candidates.sort(key=score, reverse=True)

    for r0, c0 in candidates:
        cells = [(r0 + rr, c0 + cc) for rr, cc in proto["shape"]]
        if not _valid_guide_placement(grid, cells, family):
            continue
        for r, c in cells:
            grid[r, c] = GREY
        return True
    return False


def _choose_canvas_and_center() -> tuple[tuple[int, int], int, int]:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    r = random.randint(7, h - 8)
    c = random.randint(7, w - 8)
    return (h, w), c - r, r + c


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    expected_blue: list[int],
    expected_red: list[int],
    expected_guide_counts: tuple[int, int],
) -> bool:
    try:
        solved, meta = _solve(inp)
    except Exception:
        return False

    if not np.array_equal(solved, out):
        return False
    if meta["blue_consts"] != sorted(expected_blue):
        return False
    if meta["red_consts"] != sorted(expected_red):
        return False

    blue_guides = [g for g in meta["guides"] if g["family"] == "blue"]
    red_guides = [g for g in meta["guides"] if g["family"] == "red"]
    if len(blue_guides) != expected_guide_counts[0]:
        return False
    if len(red_guides) != expected_guide_counts[1]:
        return False

    if len(meta["rhombi"]) < 4:
        return False

    complexities = {item["complexity"] for item in meta["rhombi"]}
    if expected_guide_counts == (3, 3) and complexities != {2, 3, 4}:
        return False
    if not complexities.issubset({2, 3, 4}) or not complexities:
        return False

    input_colors = set(map(int, np.unique(inp)))
    output_colors = set(map(int, np.unique(out)))
    if not input_colors.issubset({BLACK, BLUE, RED, GREY}):
        return False
    if not output_colors.issubset({BLACK, BLUE, RED, GREEN, YELLOW, ORANGE}):
        return False
    if GREY in output_colors:
        return False
    if np.array_equal(inp, out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape, base_d, base_s = _choose_canvas_and_center()
    blue_count = 3 if random.random() < 0.75 else 2
    red_count = 3 if random.random() < 0.75 else 2
    blue_offsets = _family_offsets(blue_count)
    red_offsets = _family_offsets(red_count)

    blue_consts = sorted([base_d] + [base_d + off for off in blue_offsets])
    red_consts = sorted([base_s] + [base_s + off for off in red_offsets])

    out, rhombi = _render_from_constants(shape, blue_consts, red_consts, base_d, base_s)
    complexities = {item["complexity"] for item in rhombi}
    if len(rhombi) < 4:
        return None
    if blue_count == 3 and red_count == 3 and complexities != {2, 3, 4}:
        return None

    inp = _draw_base_lines(shape, base_d, base_s)

    plans = [("blue", off) for off in sorted(blue_offsets, key=lambda x: -abs(x))]
    plans += [("red", off) for off in sorted(red_offsets, key=lambda x: -abs(x))]
    random.shuffle(plans)

    for family, off in plans:
        if not _place_guide(inp, family, off, base_d, base_s):
            return None

    if not _validate_semantics(
        inp,
        out,
        blue_consts,
        red_consts,
        (blue_count, red_count),
    ):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
