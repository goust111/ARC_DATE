# Combo variant of 7b3084d4 (index 1)
# combo_spec: D1/1+D8/0
# Name: 凹凸互锁不规则分割拼图
#
# Design analysis:
# - Concrete example: with N=4 (output side 8), the original task uses a straight
#   2x2 split plus four one-cell-wide tabs. Here the split is driven by two
#   staircase cut profiles such as vcut=[4,5,5,4,3,3,4,4] and
#   hcut=[4,4,3,4,5,5,4,4]. A cell is "left" when c < vcut[r] and "top" when
#   r < hcut[c]. The four regions are therefore polyominoes with repeated
#   concave/convex teeth, so assembling them requires matching full silhouettes
#   rather than only aligning rectangular tabs.
# - Complexity calibration: the variant still uses exactly four pieces, only
#   90-degree rotations, and one anchor marker. To keep the geometry visible but
#   learnable, cut offsets are limited to a small band around the center, yet we
#   require multiple turns and minimum piece area so the shapes do not collapse
#   back into nearly-plain quadrants.
# - Implicit assumptions review: the original solver extracted pieces by color,
#   which is brittle once the anchor is a multi-color component and the key cue
#   is the irregular outline. This version extracts 8-connected components,
#   identifies the anchor by the singleton marker, and packs exact cell masks.
#   The remaining assumptions are explicit and generator-backed: pieces are
#   separated by at least one background cell, rotations are allowed but flips
#   are not, and the four pieces tile a full square.
# - Placement legality: the generator is output-first, rotates the four tight
#   patches independently, and scatters them on a 20x20 canvas via
#   random_free_location with padding=1, so pieces never overlap or touch.
# - Visual observability: distinct colors mark the four regions, the marker
#   still pins the anchor to output (0,0), and the staircase cuts appear on both
#   central axes in every sample. The rule is readable from a few examples
#   because no hidden color ordering or non-visual metadata is used.

from collections import deque
import math
import random

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


BG = 0
ANCHOR_COLOR = 7
MARKER_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
CANVAS_H = 20
CANVAS_W = 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 500
N_MIN = 4
N_MAX = 6
TL, TR, BL, BR = 0, 1, 2, 3


def _normalize(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    out = [(r - min_r, c - min_c, col) for r, c, col in cells]
    out.sort()
    return out


def _rotate_cells(cells: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    pts = list(cells)
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]
    return _normalize(pts)


def _variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    outs: list[list[tuple[int, int, int]]] = []
    for k in range(4):
        rot = _rotate_cells(cells, k)
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            outs.append(rot)
    return outs


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _components(g: np.ndarray) -> tuple[list[dict], int]:
    bg = _bg_color(g)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for sr, sc in np.argwhere(g != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        abs_cells: list[tuple[int, int, int]] = []
        while q:
            r, c = q.popleft()
            abs_cells.append((r, c, int(g[r, c])))
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        local = _normalize(abs_cells)
        marker_pts = [(r, c) for r, c, col in local if col == MARKER_COLOR]
        comps.append({"cells": local, "has_marker": len(marker_pts) == 1, "marker_local": marker_pts[0] if marker_pts else None})
    return comps, bg


def _square_side(area: int) -> int | None:
    side = int(math.isqrt(area))
    return side if side * side == area else None


def _dims(area: int) -> list[tuple[int, int]]:
    side = _square_side(area)
    if side is not None:
        return [(side, side)]
    out: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    out.sort(key=lambda t: (abs(t[0] - t[1]), t[0], t[1]))
    return out


def _placeable(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    H, W = board.shape
    for r, c, _ in placement:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if board[r, c] != -1:
            return False
    return True


def _apply(board: np.ndarray, placement: list[tuple[int, int, int]], fill: int | None = None) -> None:
    for r, c, col in placement:
        board[r, c] = col if fill is None else fill


def _touches_filled(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    H, W = board.shape
    for r, c, _ in placement:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W and board[nr, nc] != -1:
                return True
    return False


def _all_placements(var: list[tuple[int, int, int]], H: int, W: int) -> list[list[tuple[int, int, int]]]:
    ph = max(r for r, _, _ in var) + 1
    pw = max(c for _, c, _ in var) + 1
    out: list[list[tuple[int, int, int]]] = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            out.append([(r + tr, c + tc, col) for r, c, col in var])
    return out


def _solve_packing(comps: list[dict], H: int, W: int) -> np.ndarray | None:
    board = np.full((H, W), -1, dtype=int)
    variants = [_variants(comp["cells"]) for comp in comps]
    placement_cache: dict[tuple[int, int], list[list[tuple[int, int, int]]]] = {}
    anchor_idx = next((i for i, comp in enumerate(comps) if comp["has_marker"]), None)

    if anchor_idx is not None:
        anchor_placements = []
        for var in variants[anchor_idx]:
            marker_pts = [(r, c) for r, c, col in var if col == MARKER_COLOR]
            if len(marker_pts) != 1:
                continue
            mr, mc = marker_pts[0]
            placed = [(r - mr, c - mc, col) for r, c, col in var]
            if _placeable(board, placed):
                anchor_placements.append(placed)
        if not anchor_placements:
            return None
    else:
        anchor_placements = [[]]

    for anchor_place in anchor_placements:
        remaining = list(range(len(comps)))
        if anchor_idx is not None:
            remaining.remove(anchor_idx)
            _apply(board, anchor_place)

        def options_for(pi: int) -> list[list[tuple[int, int, int]]]:
            opts: list[list[tuple[int, int, int]]] = []
            for vi, var in enumerate(variants[pi]):
                key = (pi, vi)
                if key not in placement_cache:
                    placement_cache[key] = _all_placements(var, H, W)
                for placement in placement_cache[key]:
                    if not _placeable(board, placement):
                        continue
                    if np.any(board != -1) and not _touches_filled(board, placement):
                        continue
                    opts.append(placement)
            return opts

        def backtrack(rem: list[int]) -> bool:
            if not rem:
                return True
            best_pi = None
            best_opts = None
            for pi in rem:
                opts = options_for(pi)
                if not opts:
                    return False
                if best_opts is None or len(opts) < len(best_opts):
                    best_pi = pi
                    best_opts = opts
                    if len(best_opts) == 1:
                        break
            assert best_pi is not None and best_opts is not None
            next_rem = [x for x in rem if x != best_pi]
            for placement in best_opts:
                _apply(board, placement)
                if backtrack(next_rem):
                    return True
                _apply(board, placement, -1)
            return False

        if backtrack(remaining):
            return board.copy()
        if anchor_idx is not None:
            _apply(board, anchor_place, -1)
    return None


def _turn_count(seq: list[int]) -> int:
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])


def _cut_sequence(S: int, N: int) -> list[int]:
    span = min(2, max(1, N - 2))
    out: list[int] = []
    cur = random.randint(-span, span)
    while len(out) < S:
        seg_len = random.randint(1, 3)
        out.extend([N + cur] * min(seg_len, S - len(out)))
        if len(out) == S:
            break
        delta = random.choice([-1, 1])
        cur = max(-span, min(span, cur + delta))
    return out[:S]


def _build_irregular_ownership(N: int) -> tuple[np.ndarray, dict] | None:
    S = 2 * N
    for _ in range(200):
        vcut = _cut_sequence(S, N)
        hcut = _cut_sequence(S, N)
        vturns = _turn_count(vcut)
        hturns = _turn_count(hcut)
        if len(set(vcut)) < 2 or len(set(hcut)) < 2:
            continue
        if vturns < 2 or hturns < 2:
            continue
        if max(vcut) - min(vcut) < 2 or max(hcut) - min(hcut) < 2:
            continue
        if vturns + hturns < 6:
            continue

        own = np.zeros((S, S), dtype=int)
        for r in range(S):
            for c in range(S):
                top = r < hcut[c]
                left = c < vcut[r]
                own[r, c] = TL if top and left else TR if top else BL if left else BR

        areas = [int(np.sum(own == q)) for q in (TL, TR, BL, BR)]
        if min(areas) < max(8, 2 * N):
            continue
        if sum(abs(a - N * N) for a in areas) < 2 * N:
            continue
        if not all(is_contiguous((own == q).astype(int), background=0, connectivity=8) for q in (TL, TR, BL, BR)):
            continue

        meta = {
            "vcut": vcut,
            "hcut": hcut,
            "vturns": vturns,
            "hturns": hturns,
            "turns": vturns + hturns,
            "area_deviation": sum(abs(a - N * N) for a in areas),
            "areas": areas,
        }
        return own, meta
    return None


def _extract_piece(own: np.ndarray, q: int, color: int, marker_abs: tuple[int, int] | None = None) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == q)
    min_r = int(rows.min())
    min_c = int(cols.min())
    cells = [(int(r) - min_r, int(c) - min_c, color) for r, c in zip(rows, cols)]
    if marker_abs is not None:
        mr, mc = marker_abs
        marker_local = (mr - min_r, mc - min_c)
        cells = [(r, c, MARKER_COLOR if (r, c) == marker_local else col) for r, c, col in cells]
    return _normalize(cells)


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=0,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        region = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    N = random.randint(N_MIN, N_MAX)
    built = _build_irregular_ownership(N)
    if built is None:
        return None
    own, meta = built
    S = 2 * N

    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}

    output = np.zeros((S, S), dtype=int)
    for q, col in colors.items():
        output[own == q] = col
    output[0, 0] = MARKER_COLOR

    pieces = [
        _extract_piece(own, TL, ANCHOR_COLOR, marker_abs=(0, 0)),
        _extract_piece(own, TR, colors[TR]),
        _extract_piece(own, BL, colors[BL]),
        _extract_piece(own, BR, colors[BR]),
    ]
    pieces = [_rotate_cells(piece, random.randint(0, 3)) for piece in pieces]
    canvas = _scatter_patches([_cells_to_patch(piece) for piece in pieces])
    if canvas is None:
        return None

    try:
        solved = main(canvas)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None

    info = {"N": N, **meta}
    return canvas, info


def _behavior_score(info: dict) -> tuple[int, int]:
    return int(info["turns"]), int(info["area_deviation"])


def sample_behavior_stats(trials: int = 32) -> dict:
    scores: list[tuple[int, int]] = []
    attempts = 0
    while len(scores) < trials and attempts < trials * 20:
        attempts += 1
        made = _gen_attempt()
        if made is None:
            continue
        _grid, info = made
        scores.append(_behavior_score(info))
    assert scores, "no successful samples generated for behavior stats"
    turns = [t for t, _ in scores]
    deviations = [d for _, d in scores]
    return {
        "count": len(scores),
        "turns_min": min(turns),
        "turns_max": max(turns),
        "turns_avg": float(sum(turns) / len(turns)),
        "deviation_min": min(deviations),
        "deviation_max": max(deviations),
        "deviation_avg": float(sum(deviations) / len(deviations)),
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        made = _gen_attempt()
        if made is not None:
            return made[0]
    raise RuntimeError("generate_input: exhausted attempts")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four scattered jigsaw-like polyominoes into the square they came from, using the marker-bearing piece as the anchor.

    Input structure:
    - The background is the most common color on a 20x20 canvas.
    - Four separated 8-connected puzzle pieces are scattered on the canvas; each is mostly one color, and the anchor piece contains a single gray marker pixel.
    - The piece outlines are not straight rectangles with one-cell tabs: the internal cut lines are staircase-like, so each boundary contains visible concave/convex teeth.

    Rule:
    - Extract the four connected components, treating the marker-containing component as the anchor.
    - Rotate the anchor so its marker lands at output position (0,0).
    - Rotate the remaining pieces by 90-degree steps as needed and pack the exact cell masks into one gapless square.

    Key insight: the colors identify which cells belong to each piece, but the placement is determined by the complementary irregular outlines, so the solve step is shape matching on the full polyomino boundaries.
    """
    g = np.asarray(grid, dtype=int)
    comps, _bg = _components(g)
    assert len(comps) == 4, "input must contain exactly four separated pieces"
    assert sum(comp["has_marker"] for comp in comps) == 1, "exactly one piece must contain the singleton marker"

    area = sum(len(comp["cells"]) for comp in comps)
    for H, W in _dims(area):
        solved = _solve_packing(comps, H, W)
        if solved is not None:
            return solved.astype(int)
    raise AssertionError("packing failed for all candidate dimensions")
