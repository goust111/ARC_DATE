# Variant of e8686506
# Dimension: D15 — 符号指令：图例或提示标记 (symbol_instructions)
# Change: 无图例或符号指令——所有信息从碎片的几何关系和对称性推断... → 边缘图例指示拼合顺序

from __future__ import annotations

import sys
from pathlib import Path
import random
from collections import Counter

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D15 variant
# -----------------------------------
# a) Concrete behavior:
#    The input still contains scattered colored fragments plus one anchored frame piece.
#    The new visible cue is a 3-pixel legend placed on an outer grid edge. Its left-to-right
#    colors encode the assembly order "outer frame -> middle fillers -> inner core".
#    Example trace:
#      - Original: all non-bg area contributed to output area; the solver inferred placement only
#        from symmetry and shape compatibility.
#      - Variant: the top-edge legend pixels are NOT puzzle pieces. The solver must strip them
#        before area factoring, then enforce that pieces of the first legend color touch the border
#        of the assembled rectangle, pieces of the second color first appear one layer inward, and
#        pieces of the third color first appear two layers inward.
#      - Difference: D15 changes both the input anatomy and the constraint system. Area inference
#        must ignore the legend; placement legality now depends on an explicit symbolic order.
# b) Complexity calibration:
#    The legend uses exactly 3 colors because width stays 5, so there are exactly three visible
#    nesting depths (0/1/2). That keeps the instruction obvious without introducing too many
#    opaque color classes. The assembled rectangle still has many scattered fragments because the
#    middle/inner rows are split into repeated small motifs.
# c) Old assumptions audited:
#    - "all non-bg pixels are pieces" is no longer true, so main() removes the edge legend first.
#    - "color alone does not encode placement depth" is no longer true; can_place() now checks the
#      legend-derived depth index for each piece.
#    - "frame anchors are optional heuristics" becomes unnecessary here; the explicit legend and the
#      preserved full-bbox frame piece provide the needed structure.
# d) Placement legality:
#    The generator reserves row 0 for the legend and places the frame bbox strictly below it, so
#    legend pixels never overlap scattered pieces. Non-frame pieces are scattered outside the frame
#    bbox with padding, preserving separability for the solver.
# e) Visual observability:
#    Humans can see the new rule directly: the edge strip always shows three colors, and outputs
#    consistently place those colors from the border inward. No ordering ever depends on numeric
#    color ids.
# f) Visual cleanliness:
#    The legend occupies its own edge strip with empty background around it. The frame remains a
#    single recognizable rectangle, and the interior motifs are row-based and symmetric, so the
#    scene stays readable.
# g) Color inference:
#    Only three non-bg colors participate in the assembly, and each has a stable spatial role:
#    border/frame, middle-layer fillers, inner-core cells. That gives enough repeated evidence
#    across 2-4 examples for humans to infer the mapping.
# h) New solver assumptions checked against generation:
#    - Exactly one isolated 3-pixel edge legend exists: generate_input() always creates it.
#    - The first legend color is the unique full-bbox frame piece color: _build_output() guarantees it.
#    - Remaining piece colors appear only at their designated first-contact depth: active row motifs
#      are constructed to preserve that invariant.


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            out.append((n // w, w))
    return out


def _edge_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, v in enumerate(map(int, line)):
        if v >= 0:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(line) - 1))
    return runs


def _find_legend(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[int]]:
    marked = np.where(g == bg, -1, g)
    H, W = g.shape
    candidates: list[tuple[list[tuple[int, int]], list[int]]] = []

    def isolated(coords: list[tuple[int, int]]) -> bool:
        coord_set = set(coords)
        for r, c in coords:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < H and 0 <= cc < W and g[rr, cc] != bg and (rr, cc) not in coord_set:
                    return False
        return True

    for r in (0, H - 1):
        for s, e in _edge_runs(marked[r]):
            coords = [(r, c) for c in range(s, e + 1)]
            cols = [int(g[r, c]) for c in range(s, e + 1)]
            if len(coords) == 3 and len(set(cols)) == 3 and isolated(coords):
                candidates.append((coords, cols))
    for c in (0, W - 1):
        for s, e in _edge_runs(marked[:, c]):
            coords = [(r, c) for r in range(s, e + 1)]
            cols = [int(g[r, c]) for r in range(s, e + 1)]
            if len(coords) == 3 and len(set(cols)) == 3 and isolated(coords):
                candidates.append((coords, cols))

    assert len(candidates) == 1, "variant input must contain exactly one isolated 3-pixel edge legend"
    return candidates[0]


def _closest_portrait_dims(area: int) -> tuple[int, int]:
    dims = [(h, w) for h, w in _factors(area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best) == 1, "closest-to-square portrait factor pair must be unique"
    return best[0]


def _depth_grid(H: int, W: int) -> np.ndarray:
    d = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            d[r, c] = min(r, H - 1 - r, c, W - 1 - c)
    return d


def _is_lr_symmetric_piece(pts: np.ndarray, H: int, W: int) -> bool:
    r0, c0, r1, c1 = bbox_coords(pts)
    if (r1 - r0 + 1, c1 - c0 + 1) != (H, W):
        return False
    bmp = np.zeros((H, W), dtype=bool)
    for r, c in pts:
        bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _find_frame_piece(
    comps_by_color: dict[int, list[np.ndarray]],
    H: int,
    W: int,
    outer_color: int,
) -> np.ndarray:
    comps = comps_by_color.get(outer_color, [])
    matches = [pts for pts in comps if _is_lr_symmetric_piece(pts, H, W)]
    assert len(matches) == 1, "legend outer color must contain exactly one full-bbox LR-symmetric frame piece"
    return matches[0]


def _remove_one_piece(
    pieces: list[tuple[int, tuple[tuple[int, int], ...]]],
    target_col: int,
    target_coords: tuple[tuple[int, int], ...],
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    removed = False
    out = []
    for col, coords in pieces:
        if not removed and col == target_col and coords == target_coords:
            removed = True
            continue
        out.append((col, coords))
    assert removed, "frame piece must be present in the piece multiset"
    return out


def _assemble(
    board: np.ndarray,
    types: list[dict],
    color_depth: dict[int, int],
    depth: np.ndarray,
    disallow_touch: set[int],
) -> np.ndarray:
    H, W = board.shape
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        want_depth = int(color_depth[col])
        seen_depths = []
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if not mirror_ok(r, c, col):
                return False
            seen_depths.append(int(depth[r, c]))
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return min(seen_depths) == want_depth

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [
            (r0, c0)
            for r0 in range(H - int(t["h"]) + 1)
            for c0 in range(W - int(t["w"]) + 1)
            if can_place(ti, r0, c0)
        ]

    def place(ti: int, r0: int, c0: int) -> list[tuple[int, int]]:
        t = types[ti]
        col = int(t["col"])
        changed = []
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    def search() -> bool:
        while True:
            best_ti = None
            best_places = None
            progress = False
            for ti in range(len(types)):
                if counts[ti] <= 0:
                    continue
                places = placements_for(ti)
                if not places:
                    return False
                if len(places) == 1:
                    counts[ti] -= 1
                    place(ti, places[0][0], places[0][1])
                    progress = True
                    break
                if len(places) == counts[ti]:
                    for r0, c0 in places:
                        counts[ti] -= 1
                        place(ti, r0, c0)
                    progress = True
                    break
                if best_places is None or len(places) < len(best_places):
                    best_ti = ti
                    best_places = places
            if progress:
                continue
            if best_ti is None:
                return np.all(board >= 0) and np.array_equal(board, board[:, ::-1])
            break

        assert best_ti is not None and best_places is not None
        for r0, c0 in best_places:
            changed = place(best_ti, r0, c0)
            counts[best_ti] -= 1
            if search():
                return True
            counts[best_ti] += 1
            unplace(changed)
        return False

    ok = search()
    assert ok, "assembly failed under legend-depth constraints"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments into a left-right symmetric rectangle using an edge legend.

    Input structure:
    - bg is the dominant color; one isolated 3-pixel legend lies on a grid edge and is not part of the puzzle pieces.
    - The remaining non-bg cells are scattered fragments, including one full-bbox outer frame piece and smaller middle/inner pieces.

    Rule:
    - Read the legend from left-to-right (or top-to-bottom if vertical): it gives the color order from outermost layer to innermost layer.
    - Ignore the legend when computing the target rectangle, then pack the fragments into the unique closest-to-square portrait rectangle.
    - Preserve exact left-right symmetry, and place each color so the first border depth it touches matches its legend position.

    Key insight:
    - The edge legend supplies the missing assembly order explicitly: the first color must define the frame, later colors can only appear farther inward, so symmetry plus layer depth pins down the reconstruction.
    """
    g = np.asarray(grid, dtype=int).copy()
    bg = get_bg(g)

    legend_coords, legend_colors = _find_legend(g, bg)
    for r, c in legend_coords:
        g[r, c] = bg

    comps_by_color: dict[int, list[np.ndarray]] = {}
    pieces: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = [np.asarray(pts, dtype=int) for pts in components_mask(g == col, connectivity=4)]
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((col, coords))

    total_area = sum(len(coords) for _, coords in pieces)
    assert total_area > 0, "grid must contain puzzle fragments after removing the legend"
    H, W = _closest_portrait_dims(total_area)

    color_depth = {int(col): i for i, col in enumerate(legend_colors)}
    assert set(color_depth) == set(comps_by_color), "legend colors must match the assembly colors exactly"

    frame_piece = _find_frame_piece(comps_by_color, H, W, legend_colors[0])
    _, _, frame_coords = _piece_sig(frame_piece)
    pieces = _remove_one_piece(pieces, int(legend_colors[0]), frame_coords)

    board = np.full((H, W), -1, dtype=int)
    fr0, fc0, _, _ = bbox_coords(frame_piece)
    for r, c in frame_piece:
        board[int(r) - fr0, int(c) - fc0] = int(legend_colors[0])

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    depth = _depth_grid(H, W)
    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(board, types, color_depth, depth, disallow_touch)
    return sol.astype(int)


# --------------- generator ---------------

_HEIGHT_CHOICES = (8, 9, 10)
_WIDTH = 5
_CANVAS_MIN = 14
_CANVAS_MAX = 16
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 400


def _build_output(H: int, colors: list[int]) -> np.ndarray:
    outer, middle, inner = map(int, colors)
    out = np.full((H, _WIDTH), outer, dtype=int)

    interior = H - 2
    active_slots = [1 + 2 * i for i in range((interior + 1) // 2)]
    n_active = random.randint(max(2, len(active_slots) - 1), len(active_slots))
    active_rows = sorted(random.sample(active_slots, n_active))
    inner_rows = set(random.sample(active_rows, random.randint(1, len(active_rows))))

    for rr in range(1, H - 1):
        if rr not in active_rows:
            continue
        if rr in inner_rows:
            out[rr] = np.array([outer, middle, inner, middle, outer], dtype=int)
        else:
            out[rr] = np.array([outer, middle, middle, middle, outer], dtype=int)
    return out


def _extract_patches(out: np.ndarray, bg: int, frame_color: int) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col == frame_color:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _place_legend(canvas: np.ndarray, bg: int, legend_colors: list[int]) -> None:
    cols = canvas.shape[1]
    start = random.randint(1, cols - 4)
    canvas[0, :] = bg
    for i, col in enumerate(legend_colors):
        canvas[0, start + i] = int(col)


def _scatter(out: np.ndarray, bg: int, legend_colors: list[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    min_r = 2
    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < min_r or max_c < 1:
        raise ValueError

    off_r = random.randint(min_r, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    for patch in _extract_patches(out, bg, legend_colors[0]):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = out == int(legend_colors[0])
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    _place_legend(canvas, bg, legend_colors)
    return canvas


def _make_output() -> tuple[np.ndarray, int, list[int]]:
    H = random.choice(_HEIGHT_CHOICES)
    bg = random.randint(0, 9)
    colors = [c for c in range(10) if c != bg]
    legend_colors = random.sample(colors, 3)
    out = _build_output(H, legend_colors)
    return out, bg, legend_colors


def _variant_metrics(inp: np.ndarray, out: np.ndarray, legend_colors: list[int]) -> dict[str, int]:
    H, W = out.shape
    depth = _depth_grid(H, W)
    first_touch = {}
    for col in legend_colors:
        vals = [int(depth[r, c]) for r, c in np.argwhere(out == int(col))]
        first_touch[int(col)] = min(vals)
    return {
        "legend_len": len(legend_colors),
        "outer_cells": int((out == legend_colors[0]).sum()),
        "middle_cells": int((out == legend_colors[1]).sum()),
        "inner_cells": int((out == legend_colors[2]).sum()),
        "first_touch_outer": first_touch[legend_colors[0]],
        "first_touch_middle": first_touch[legend_colors[1]],
        "first_touch_inner": first_touch[legend_colors[2]],
    }


def _gen_attempt() -> np.ndarray | None:
    out, bg, legend_colors = _make_output()
    try:
        inp = _scatter(out, bg, legend_colors)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D15 input")
