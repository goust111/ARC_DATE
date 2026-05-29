# Combo variant of 7b3084d4 (index 3)
# combo_spec: D1/1+D2/1+D10/0
# Name: 非矩形块八向翻转凹凸匹配
#
# Design analysis
# a) Concrete behavior:
#    - Original: four color-separated near-quadrant pieces are recovered mostly by color, then packed
#      with only 4 rotations. Straight one-strip tabs are enough.
#    - Here: the target is still a compact square rectangle, but each quadrant acquires several local
#      seam intrusions, so every piece becomes a non-rectangular polyomino with visible notches/tabs.
#      Example for N=4: rows 0-1 on the top vertical seam may be bitten left by TR while row 3 may
#      protrude right from TL; after scattering, one piece is also reflected, so the solver must search
#      over dihedral variants instead of only 90-degree rotations.
#    - Difference: color no longer identifies a rectangle, and “can tile a rectangle” is constrained by
#      irregular complementary boundaries plus forced reflection usage in the input.
# b) Complexity calibration:
#    - Keep exactly 4 pieces and one marker so the ARC task stays readable.
#    - Each of the 4 shared half-seams gets 2-3 short transfers, making the irregularity obvious without
#      exploding into arbitrary noisy shapes.
#    - Generator enforces all 4 pieces non-rectangular and at least 1 reflected-only placement.
# c) Broken assumptions from the seed solver:
#    - “one color == one piece” is replaced by 8-connected component extraction because the anchor is
#      multi-color (marker + anchor color) and shape, not color, defines the piece.
#    - “piece is rectangular” no longer holds; bbox area can exceed piece area substantially.
#    - “4 rotations are enough” no longer holds; some generated inputs require a mirror image.
#    - “any non-overlapping packing is fine” is weakened by contact-based search heuristics because legal
#      placements should lock onto the growing assembly through complementary edges.
# d) Placement legality:
#    - Output ownership edits are local to the central seams, never touch the outer frame, and are
#      accepted only when every region remains connected and in-bounds.
#    - Input scattering uses `random_free_location` with padding=1, so pieces never overlap or touch.
# e) Visual observability:
#    - The marker is still a unique gray singleton at the anchor corner.
#    - Four distinct solid colors keep the piece identities human-visible; the new rule is inferable from
#      the repeated jagged seam profiles and occasional mirrored input orientation.
# f) Visual cleanliness:
#    - No extra overlays or distractors are added. The only multi-color object is the anchor marker.
#    - Irregularity is concentrated around the central seams, so each piece silhouette stays legible.
# g) Quantifiable fusion checks:
#    - `nonrectangular_count`: how many output pieces have area < bbox area.
#    - `flip_required_count`: how many scattered pieces are shown in a dihedral pose unreachable by
#      rotation alone.
#    - `jagged_seam_count`: how many of the 4 shared half-seams use at least 2 distinct intrusion
#      depths. The generator rejects samples unless these metrics confirm the intended fusion behavior.

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import get_bg as _bg_color, neighbors8, random_free_location, is_contiguous


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


def _reflect_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return _normalize([(r, -c, col) for r, c, col in cells])


def _variant_records(cells: list[tuple[int, int, int]]) -> list[dict]:
    rot_keys = {tuple(_rotate_cells(cells, k)) for k in range(4)}
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    records: list[dict] = []
    for base, used_flip in ((cells, False), (_reflect_cells(cells), True)):
        for k in range(4):
            v = _rotate_cells(base, k)
            key = tuple(v)
            if key in seen:
                continue
            seen.add(key)
            records.append({"cells": v, "requires_flip": used_flip and key not in rot_keys})
    return records


def _variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    return [rec["cells"] for rec in _variant_records(cells)]


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _components(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps: list[dict] = []
    for sr, sc in np.argwhere(g != bg):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        abs_cells: list[tuple[int, int, int]] = []
        while q:
            r, c = q.popleft()
            abs_cells.append((int(r), int(c), int(g[r, c])))
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append({"cells": _normalize(abs_cells), "abs": abs_cells})
    return comps, bg


def _extract_pieces(g: np.ndarray) -> tuple[list[dict], int]:
    comps, bg = _components(g)
    assert len(comps) == 4, "input must contain exactly four separated pieces"
    marker_total = int(np.sum(g == MARKER_COLOR))
    assert marker_total == 1, "input must contain exactly one marker pixel"
    anchor_idxs = [i for i, comp in enumerate(comps) if sum(col == MARKER_COLOR for _, _, col in comp["cells"]) == 1]
    assert len(anchor_idxs) == 1, "exactly one piece must contain the marker"
    anchor_idx = anchor_idxs[0]

    pieces: list[dict] = []
    for i, comp in enumerate(comps):
        piece = {"name": "anchor" if i == anchor_idx else f"p{i}", "cells": comp["cells"]}
        if i == anchor_idx:
            piece["marker_color"] = MARKER_COLOR
        pieces.append(piece)
    return pieces, bg


def _dims(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            pairs.append((h, w))
            if h != w:
                pairs.append((w, h))
    pairs.sort(key=lambda t: (abs(t[0] - t[1]), t[0], t[1]))
    return pairs


def _placeable(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    H, W = board.shape
    for r, c, _ in placement:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if board[r, c] != -1:
            return False
    return True


def _apply(board: np.ndarray, placement: list[tuple[int, int, int]], val: int | None = None) -> None:
    if val is None:
        for r, c, col in placement:
            board[r, c] = col
    else:
        for r, c, _ in placement:
            board[r, c] = val


def _all_placements(variant: list[tuple[int, int, int]], H: int, W: int) -> list[list[tuple[int, int, int]]]:
    ph = max(r for r, _, _ in variant) + 1
    pw = max(c for _, c, _ in variant) + 1
    out = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            out.append([(r + tr, c + tc, col) for r, c, col in variant])
    return out


def _edge_touch(board: np.ndarray, placement: list[tuple[int, int, int]]) -> int:
    occ = set((int(r), int(c)) for r, c in np.argwhere(board != -1))
    touch = 0
    for r, c, _ in placement:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in occ:
                touch += 1
    return touch


def _empty_component_sizes(board: np.ndarray) -> list[int]:
    H, W = board.shape
    seen = np.zeros_like(board, dtype=bool)
    sizes = []
    for sr, sc in np.argwhere(board == -1):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        size = 0
        while q:
            r, c = q.popleft()
            size += 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and board[nr, nc] == -1:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        sizes.append(size)
    return sizes


def _remaining_space_ok(board: np.ndarray, remaining_areas: list[int]) -> bool:
    if not remaining_areas:
        return not np.any(board == -1)
    comps = _empty_component_sizes(board)
    if len(comps) > len(remaining_areas):
        return False
    if comps and min(comps) < min(remaining_areas):
        return False
    return True


def _solve_packing(pieces: list[dict], H: int, W: int) -> np.ndarray | None:
    board = np.full((H, W), -1, dtype=int)
    placements_by_piece: list[list[list[tuple[int, int, int]]]] = []
    for piece in pieces:
        piece_placements = []
        for var in _variants(piece["cells"]):
            piece_placements.extend(_all_placements(var, H, W))
        placements_by_piece.append(piece_placements)

    anchor_idx = next((i for i, p in enumerate(pieces) if p["name"] == "anchor"), None)

    def backtrack(rem: list[int]) -> bool:
        if not rem:
            return True

        occ_any = bool(np.any(board != -1))
        best_pi = None
        best_options: list[tuple[int, int, int, list[tuple[int, int, int]]]] | None = None

        for pi in rem:
            opts = []
            for placement in placements_by_piece[pi]:
                if not _placeable(board, placement):
                    continue
                touch = _edge_touch(board, placement) if occ_any else 0
                if occ_any and touch == 0:
                    continue
                rows = [r for r, _, _ in placement]
                cols = [c for _, c, _ in placement]
                bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
                opts.append((-touch, bbox_area, min(rows), min(cols), placement))
            if not opts:
                continue
            opts.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            if best_options is None or len(opts) < len(best_options):
                best_pi = pi
                best_options = opts

        if best_pi is None or best_options is None:
            return False

        next_rem = [x for x in rem if x != best_pi]
        rem_areas = [len(pieces[x]["cells"]) for x in next_rem]
        for _, _, _, _, placement in best_options:
            _apply(board, placement)
            if _remaining_space_ok(board, rem_areas) and backtrack(next_rem):
                return True
            _apply(board, placement, -1)
        return False

    if anchor_idx is None:
        return board if backtrack(list(range(len(pieces)))) else None

    remaining = [i for i in range(len(pieces)) if i != anchor_idx]
    marker_col = int(pieces[anchor_idx]["marker_color"])
    for rec in _variant_records(pieces[anchor_idx]["cells"]):
        marker_pts = [(r, c) for r, c, col in rec["cells"] if col == marker_col]
        if len(marker_pts) != 1:
            continue
        mr, mc = marker_pts[0]
        placed = [(r - mr, c - mc, col) for r, c, col in rec["cells"]]
        if not _placeable(board, placed):
            continue
        _apply(board, placed)
        if _remaining_space_ok(board, [len(pieces[x]["cells"]) for x in remaining]) and backtrack(remaining):
            return board
        _apply(board, placed, -1)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four scattered interlocking polyomino pieces into one filled rectangle, using the marker piece as the fixed corner.

    Input structure:
    - Background is the most common color on a 20x20 canvas.
    - Exactly four separated puzzle pieces appear; each is a solid-color polyomino except the anchor, which includes one gray marker pixel.
    - Piece silhouettes include visible notches and tabs rather than plain rectangles, and any piece may be rotated or mirrored.

    Rule:
    - Extract the four disconnected pieces, identify the anchor as the one containing the unique marker, and search over all 8 dihedral transforms per piece.
    - Place the anchor so its gray marker lands at output coordinate (0,0).
    - Arrange the remaining transformed pieces so they edge-lock with the growing assembly and together fill one gapless rectangle.

    Key insight: The marker fixes the assembled rectangle's orientation, but the actual placement must be recovered by matching complementary irregular boundaries in the full rotate-and-flip search space.
    """
    g = np.asarray(grid, dtype=int)
    pieces, _bg = _extract_pieces(g)
    area = sum(len(piece["cells"]) for piece in pieces)
    for H, W in _dims(area):
        solved = _solve_packing(pieces, H, W)
        if solved is not None:
            return solved.astype(int)
    raise AssertionError("packing failed for all candidate dimensions")


def _base_ownership(N: int) -> np.ndarray:
    S = 2 * N
    own = np.zeros((S, S), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR
    return own


def _segment_profile_metrics(own: np.ndarray, N: int) -> dict:
    top_v: list[int] = []
    bot_v: list[int] = []
    left_h: list[int] = []
    right_h: list[int] = []
    for r in range(N):
        left_depth = int(np.sum(own[r, :N] == TR))
        right_depth = int(np.sum(own[r, N:] == TL))
        top_v.append(right_depth if right_depth else -left_depth)
    for r in range(N, 2 * N):
        left_depth = int(np.sum(own[r, :N] == BR))
        right_depth = int(np.sum(own[r, N:] == BL))
        bot_v.append(right_depth if right_depth else -left_depth)
    for c in range(N):
        up_depth = int(np.sum(own[:N, c] == BL))
        down_depth = int(np.sum(own[N:, c] == TL))
        left_h.append(down_depth if down_depth else -up_depth)
    for c in range(N, 2 * N):
        up_depth = int(np.sum(own[:N, c] == BR))
        down_depth = int(np.sum(own[N:, c] == TR))
        right_h.append(down_depth if down_depth else -up_depth)

    profiles = [top_v, bot_v, left_h, right_h]
    variation_counts = [len({abs(v) for v in prof if v != 0}) for prof in profiles]
    return {
        "profiles": profiles,
        "variation_counts": variation_counts,
        "jagged_seam_count": sum(int(v >= 2) for v in variation_counts),
    }


def _sample_intrusion_profile(length: int, max_depth: int, sign: int) -> list[int]:
    profile = [0] * length
    start = random.randint(0, length - 2)
    end = random.randint(start + 1, length - 1)
    active_len = end - start + 1
    seg_count = random.randint(2, min(3, active_len))
    cuts = sorted(random.sample(range(1, active_len), seg_count - 1))
    bounds = [0] + cuts + [active_len]
    depth_choices = list(range(1, max_depth + 1))
    if len(depth_choices) >= 2:
        used_depths = random.sample(depth_choices, 2)
    else:
        used_depths = [1, 1]
    while len(used_depths) < seg_count:
        used_depths.append(random.choice(depth_choices))
    random.shuffle(used_depths)
    if len({d for d in used_depths}) < 2 and len(depth_choices) >= 2:
        used_depths[-1] = used_depths[0] % max_depth + 1
    for seg_idx in range(seg_count):
        a = start + bounds[seg_idx]
        b = start + bounds[seg_idx + 1]
        for idx in range(a, b):
            profile[idx] = sign * used_depths[seg_idx]
    return profile


def _apply_vertical_profile(
    own: np.ndarray,
    N: int,
    row_start: int,
    left_owner: int,
    right_owner: int,
    profile: list[int],
) -> None:
    for idx, offset in enumerate(profile):
        r = row_start + idx
        if offset < 0:
            own[r, N + offset : N] = right_owner
        elif offset > 0:
            own[r, N : N + offset] = left_owner


def _apply_horizontal_profile(
    own: np.ndarray,
    N: int,
    col_start: int,
    top_owner: int,
    bottom_owner: int,
    profile: list[int],
) -> None:
    for idx, offset in enumerate(profile):
        c = col_start + idx
        if offset < 0:
            own[N + offset : N, c] = bottom_owner
        elif offset > 0:
            own[N : N + offset, c] = top_owner


def _piece_cells_from_ownership(
    own: np.ndarray,
    q: int,
    color: int,
    marker_pos: tuple[int, int] | None = None,
) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == q)
    min_r = int(rows.min())
    min_c = int(cols.min())
    out = []
    for r, c in zip(rows, cols):
        r = int(r)
        c = int(c)
        col = MARKER_COLOR if marker_pos is not None and (r, c) == marker_pos else color
        out.append((r - min_r, c - min_c, col))
    out.sort()
    return out


def _nonrectangular(cells: list[tuple[int, int, int]]) -> bool:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    return len(cells) < h * w


def _piece_has_flip_only_variant(cells: list[tuple[int, int, int]]) -> bool:
    return any(rec["requires_flip"] for rec in _variant_records(cells))


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    N = random.randint(N_MIN, N_MAX)
    own = _base_ownership(N)

    max_depth = max(1, min(3, N - 2))
    top_v = _sample_intrusion_profile(N, max_depth, 1)
    bot_v = _sample_intrusion_profile(N, max_depth, -1)
    left_h = _sample_intrusion_profile(N, max_depth, -1)
    right_h = _sample_intrusion_profile(N, max_depth, 1)

    _apply_vertical_profile(own, N, 0, TL, TR, top_v)
    _apply_vertical_profile(own, N, N, BL, BR, bot_v)
    _apply_horizontal_profile(own, N, 0, TL, BL, left_h)
    _apply_horizontal_profile(own, N, N, TR, BR, right_h)

    for q in (TL, TR, BL, BR):
        mask = (own == q).astype(int)
        if mask.sum() == 0 or not is_contiguous(mask, background=0, connectivity=4):
            return None

    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}

    output = np.zeros((2 * N, 2 * N), dtype=int)
    for q, color in colors.items():
        output[own == q] = color
    output[0, 0] = MARKER_COLOR

    pieces = [
        _piece_cells_from_ownership(own, TL, ANCHOR_COLOR, marker_pos=(0, 0)),
        _piece_cells_from_ownership(own, TR, colors[TR]),
        _piece_cells_from_ownership(own, BL, colors[BL]),
        _piece_cells_from_ownership(own, BR, colors[BR]),
    ]

    if not all(_nonrectangular(piece) for piece in pieces):
        return None

    if not any(_piece_has_flip_only_variant(piece) for piece in pieces):
        return None

    seam_metrics = _segment_profile_metrics(own, N)
    if seam_metrics["jagged_seam_count"] < 2:
        return None

    transformed: list[list[tuple[int, int, int]]] = []
    flip_required_count = 0
    forceable = [i for i, piece in enumerate(pieces) if _piece_has_flip_only_variant(piece)]
    if not forceable:
        return None
    forced_idx = random.choice(forceable)
    for i, piece in enumerate(pieces):
        records = _variant_records(piece)
        if i == forced_idx:
            chosen_pool = [rec for rec in records if rec["requires_flip"]]
        else:
            chosen_pool = records
        chosen = random.choice(chosen_pool)
        transformed.append(chosen["cells"])
        if chosen["requires_flip"]:
            flip_required_count += 1
    if flip_required_count == 0:
        return None

    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    order = list(range(len(transformed)))
    random.shuffle(order)
    for idx in order:
        patch = _cells_to_patch(transformed[idx])
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

    meta = {
        "N": N,
        "nonrectangular_count": sum(int(_nonrectangular(piece)) for piece in pieces),
        "flip_required_count": flip_required_count,
        "jagged_seam_count": seam_metrics["jagged_seam_count"],
        "variation_counts": seam_metrics["variation_counts"],
    }

    try:
        solved = main(canvas)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return canvas, output, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("generate_input: exhausted attempts")
