# Combo variant of e8686506 (index 22)
# combo_spec: D3/3+D15/0
# Name: 碎片叠放到模板上+边缘图例

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    - Source task: every non-bg component is a puzzle fragment, the solver infers a dense LR-symmetric
#      rectangle, and that rectangle is returned directly.
#    - D15 adds an isolated 3-pixel edge legend. Those pixels are not pieces; their left-to-right order
#      tells the solver which color must appear first at depth 0, then depth 1, then depth 2 in the
#      reconstructed rectangle.
#    - D3 adds a visible hollow rectangular template. That ring is also not a piece; after reconstruction,
#      the dense rectangle must be centered inside the ring and merged with it.
#    - Combined example: input shows one top-edge legend [outer, middle, inner], one empty ring elsewhere,
#      one anchored outer-color frame piece elsewhere, and scattered middle/inner fragments. The solver
#      removes legend + template, rebuilds the hidden 8x5/9x5/10x5 board under depth constraints, then
#      inserts it into the 12x9/13x9/14x9 template patch.
# b) Complexity calibration:
#    - The legend is fixed to 3 colors because width stays 5, so only depths 0/1/2 exist. That keeps the
#      symbolic cue explicit and repeated across samples.
#    - The template margin is fixed to 2 cells on every side. This makes the overlay visually obvious while
#      avoiding free alignment choices.
#    - The board generator keeps many repeated middle/inner fragments, so the task is still nontrivial and
#      does not collapse into reading the legend alone.
# c) Old assumptions audited:
#    - "all non-bg cells are puzzle pieces" breaks twice: legend pixels and template ring must both be
#      stripped before area factoring and frame detection.
#    - "output equals assembled rectangle" no longer holds: after assembly the rectangle must be centered
#      into another visible object.
#    - "geometry alone fixes color depth" no longer holds: placement legality now depends on the legend.
#    - The original corner/bar heuristics are unnecessary here; the legend plus full-bbox outer frame piece
#      provide stronger constraints.
# d) Placement legality:
#    - The generator reserves disjoint rectangles for template and frame, and keeps all other fragments away
#      from them with padding.
#    - All placed objects stay at least one row away from the edge legend, so the legend remains isolated.
# e) Visual observability:
#    - Across samples, the same three legend colors always map to border/middle/core depths, and the output
#      always keeps the empty ring while filling its centered interior with the assembled board.
#    - No rule depends on numeric color ids; only visible order and geometry matter.
# f) Visual cleanliness:
#    - The template is a single-color hollow ring, the legend is alone on the edge, and the anchored frame
#      piece stays separate from the ring. This keeps the three roles easy to distinguish.
# g) New solver assumptions guaranteed by generation:
#    - Exactly one isolated 3-pixel edge legend exists.
#    - Exactly one hollow rectangular template exists, with empty interior and fixed 2-cell margin.
#    - The first legend color is the unique full-bbox LR-symmetric outer piece of the hidden board.
#    - The second and third legend colors first appear at depths 1 and 2 respectively.


_WIDTH = 5
_HEIGHT_CHOICES = (8, 9, 10)
_TEMPLATE_MARGIN = 2
_LEGEND_LEN = 3

_CANVAS_EXTRA_MIN = 8
_CANVAS_EXTRA_MAX = 12
_SCATTER_PADDING = 1
_RESERVE_PADDING = 1
_EDGE_BORDER = 2
_GEN_MAX_RETRIES = 500


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


def _closest_portrait_dims(area: int) -> tuple[int, int]:
    dims = [(h, w) for h, w in _factors(area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best) == 1, "closest-to-square portrait factor pair must be unique"
    return best[0]


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
            if len(coords) == _LEGEND_LEN and len(set(cols)) == _LEGEND_LEN and isolated(coords):
                candidates.append((coords, cols))

    for c in (0, W - 1):
        for s, e in _edge_runs(marked[:, c]):
            coords = [(r, c) for r in range(s, e + 1)]
            cols = [int(g[r, c]) for r in range(s, e + 1)]
            if len(coords) == _LEGEND_LEN and len(set(cols)) == _LEGEND_LEN and isolated(coords):
                candidates.append((coords, cols))

    assert len(candidates) == 1, "input must contain exactly one isolated 3-pixel edge legend"
    return candidates[0]


def _find_template(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    candidates = []
    for col in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in components_mask(grid == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < min(_HEIGHT_CHOICES) + 2 * _TEMPLATE_MARGIN or w < _WIDTH + 2 * _TEMPLATE_MARGIN:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border_mask = np.zeros((h, w), dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            if not np.all(patch[border_mask] == col):
                continue
            if np.any(patch[~border_mask] != bg):
                continue
            if len(pts) != int(border_mask.sum()):
                continue
            candidates.append((h * w, int(col), int(r0), int(c0), int(r1), int(c1)))
    assert len(candidates) == 1, "input must contain exactly one hollow rectangular template"
    _, col, r0, c0, r1, c1 = candidates[0]
    return r0, c0, r1, c1, col


def _overlay_on_template(template_patch: np.ndarray, board: np.ndarray, bg: int) -> np.ndarray:
    out = template_patch.copy()
    TH, TW = out.shape
    H, W = board.shape
    assert TH - H == 2 * _TEMPLATE_MARGIN and TW - W == 2 * _TEMPLATE_MARGIN
    r0 = _TEMPLATE_MARGIN
    c0 = _TEMPLATE_MARGIN
    assert np.all(out[r0 : r0 + H, c0 : c0 + W] == bg), "template interior under centered board must be empty"
    out[r0 : r0 + H, c0 : c0 + W] = board
    return out


def _depth_grid(H: int, W: int) -> np.ndarray:
    depth = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            depth[r, c] = min(r, H - 1 - r, c, W - 1 - c)
    return depth


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
    Assemble scattered fragments into a left-right symmetric rectangle, then center that rectangle inside a visible template.

    Input structure:
    - bg is the dominant color; one isolated 3-pixel legend sits on a grid edge and one large hollow rectangle is a destination template.
    - After removing those two non-piece structures, the remaining non-bg cells are puzzle fragments: one full-bbox outer piece plus scattered middle and inner fragments.

    Rule:
    - Read the legend from left to right (or top to bottom if vertical) to get the color order from outer layer to inner layer.
    - Ignore both legend and template while reconstructing the unique closest-to-square portrait rectangle, enforcing exact LR symmetry and matching each color to the first border depth named by the legend.
    - Once the rectangle is rebuilt, center it inside the hollow template and merge it with the template border.

    Key insight:
    - The legend tells you how deeply each color is allowed to appear in the hidden rectangle, and the template tells you the returned object is not the bare rectangle but that rectangle placed into a separate visible ring.
    """
    g = np.asarray(grid, dtype=int).copy()
    bg = get_bg(g)

    legend_coords, legend_colors = _find_legend(g, bg)
    for r, c in legend_coords:
        g[r, c] = bg

    tr0, tc0, tr1, tc1, _template_col = _find_template(g, bg)
    template_patch = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    g[tr0 : tr1 + 1, tc0 : tc1 + 1] = bg

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
    assert total_area > 0, "grid must contain puzzle fragments after removing legend and template"
    H, W = _closest_portrait_dims(total_area)

    color_depth = {int(col): i for i, col in enumerate(legend_colors)}
    assert set(color_depth) == set(comps_by_color), "legend colors must match the assembly colors exactly"

    frame_piece = _find_frame_piece(comps_by_color, H, W, int(legend_colors[0]))
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
    solved = _assemble(board, types, color_depth, depth, disallow_touch)
    return _overlay_on_template(template_patch, solved, bg).astype(int)


# --------------- generator ---------------


def _build_board(H: int, colors: list[int]) -> np.ndarray:
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


def _make_template_patch(board: np.ndarray, bg: int, template_col: int) -> tuple[np.ndarray, np.ndarray]:
    H, W = board.shape
    TH = H + 2 * _TEMPLATE_MARGIN
    TW = W + 2 * _TEMPLATE_MARGIN
    template = np.full((TH, TW), bg, dtype=int)
    template[0, :] = template_col
    template[-1, :] = template_col
    template[:, 0] = template_col
    template[:, -1] = template_col
    out = template.copy()
    out[_TEMPLATE_MARGIN : _TEMPLATE_MARGIN + H, _TEMPLATE_MARGIN : _TEMPLATE_MARGIN + W] = board
    return template, out


def _make_output() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[int], int]:
    H = random.choice(_HEIGHT_CHOICES)
    bg = random.randint(0, 9)
    colors = random.sample([c for c in range(10) if c != bg], 4)
    legend_colors = colors[:3]
    template_col = colors[3]
    board = _build_board(H, legend_colors)
    template, out = _make_template_patch(board, bg, template_col)
    return board.astype(int), template.astype(int), out.astype(int), bg, legend_colors, int(template_col)


def _extract_patches(board: np.ndarray, bg: int, frame_color: int) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(board))):
        if col == frame_color:
            continue
        for pts in components_mask(board == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _frame_patch(board: np.ndarray, bg: int, frame_color: int) -> np.ndarray:
    patch = np.full(board.shape, bg, dtype=int)
    patch[board == frame_color] = frame_color
    return patch


def _reserve_rect(canvas: np.ndarray, h: int, w: int, bg: int) -> tuple[int, int]:
    block = np.full((h, w), -1, dtype=int)
    r, c = random_free_location(
        canvas,
        block,
        transparent=bg,
        border=_EDGE_BORDER,
        padding=_RESERVE_PADDING,
        padding_connectivity=8,
    )
    paste(canvas, block, r, c, transparent=bg)
    return r, c


def _place_legend(canvas: np.ndarray, bg: int, legend_colors: list[int]) -> None:
    cols = canvas.shape[1]
    start = random.randint(1, cols - _LEGEND_LEN - 1)
    for i, col in enumerate(legend_colors):
        canvas[0, start + i] = int(col)
    canvas[0, :start] = bg
    canvas[0, start + _LEGEND_LEN :] = bg


def _scatter(board: np.ndarray, template: np.ndarray, bg: int, legend_colors: list[int]) -> np.ndarray:
    bh, bw = board.shape
    th, tw = template.shape
    base = max(bh, th, bw, tw)
    rows = random.randint(base + _CANVAS_EXTRA_MIN, base + _CANVAS_EXTRA_MAX)
    cols = random.randint(base + _CANVAS_EXTRA_MIN, base + _CANVAS_EXTRA_MAX)
    cols = max(cols, _LEGEND_LEN + 3)
    canvas = np.full((rows, cols), bg, dtype=int)

    tr, tc = _reserve_rect(canvas, th, tw, bg)
    fr, fc = _reserve_rect(canvas, bh, bw, bg)

    for patch in _extract_patches(board, bg, int(legend_colors[0])):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            border=_EDGE_BORDER,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    canvas[tr : tr + th, tc : tc + tw] = bg
    paste(canvas, template, tr, tc, transparent=bg)

    canvas[fr : fr + bh, fc : fc + bw] = bg
    paste(canvas, _frame_patch(board, bg, int(legend_colors[0])), fr, fc, transparent=bg)

    _place_legend(canvas, bg, legend_colors)
    return canvas


def _first_touch_depths(board: np.ndarray, legend_colors: list[int]) -> list[int]:
    depth = _depth_grid(*board.shape)
    vals = []
    for col in legend_colors:
        hits = [int(depth[r, c]) for r, c in np.argwhere(board == int(col))]
        vals.append(min(hits))
    return vals


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, object]:
    bg = get_bg(inp)
    legend_coords, legend_colors = _find_legend(inp, bg)
    tr0, tc0, tr1, tc1, template_col = _find_template(inp, bg)
    template_h = tr1 - tr0 + 1
    template_w = tc1 - tc0 + 1
    board = out[_TEMPLATE_MARGIN : out.shape[0] - _TEMPLATE_MARGIN, _TEMPLATE_MARGIN : out.shape[1] - _TEMPLATE_MARGIN]
    depths = _first_touch_depths(board, legend_colors)
    return {
        "legend_colors": tuple(map(int, legend_colors)),
        "legend_size": len(legend_coords),
        "template_color": int(template_col),
        "template_shape": (int(template_h), int(template_w)),
        "output_shape": tuple(map(int, out.shape)),
        "board_shape": tuple(map(int, board.shape)),
        "depths": tuple(depths),
    }


def _gen_attempt() -> np.ndarray | None:
    board, template, out, bg, legend_colors, _template_col = _make_output()
    try:
        inp = _scatter(board, template, bg, legend_colors)
        got = main(inp)
        metrics = _fusion_metrics(inp, got)
    except Exception:
        return None

    if not np.array_equal(got, out):
        return None
    if metrics["legend_size"] != _LEGEND_LEN:
        return None
    if metrics["depths"] != (0, 1, 2):
        return None
    if metrics["output_shape"] == metrics["board_shape"]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
