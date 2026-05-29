# Variant of e8686506
# Dimension: D2 — 源对象：碎片的类型与结构 (src)
# Change: 散落的单色非背景像素碎片和精灵片段，每个碎片为单一颜色的连通区域... → 碎片带锚点标记

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D2 anchor-point variant
# -----------------------------------------------
# spec:
#   visual_concretization:
#     - Every real jigsaw piece is now a 4-connected non-background component with exactly
#       one anchor pixel of a dedicated anchor color and >=2 body pixels of a single body color.
#     - The output remains a dense LR-symmetric rectangle, but each assembled piece keeps its
#       anchor pixel visible inside the rectangle.
#     - In the input, frame piece(s) stay in-place as before; scattered non-frame pieces keep
#       their embedded anchors; the target rectangle also shows one isolated single-pixel
#       anchor marker for every scattered piece, at that piece's final anchor location.
#   behavior_trace:
#     - Original: collect monochrome components by color -> infer HxW from total area ->
#       use frame + symmetry + same-color non-touch to place each pure-color fragment.
#     - Variant: collect actual pieces by binary connectivity (color alone no longer defines
#       a piece) -> treat singleton components as target anchor marks, not pieces ->
#       infer HxW from only multi-cell components -> place each anchored piece so its unique
#       anchor pixel lands on a same-colored target mark inside the frame.
#     - Difference: precise point-to-point alignment is now explicit. A shape that could fit
#       several symmetric holes is rejected unless its anchor lands on the matching marker.
#   complexity_calibration:
#     - Anchor colors come from a small dedicated palette reused across multiple pieces.
#       This keeps the rule visible across 2-3 examples without exploding the color count.
#     - Real pieces are forced to have area >= 3, so any singleton component in the input
#       is unambiguously a target anchor marker rather than a fragment.
#   implicit_assumptions_replaced:
#     - Old `components_mask(g == col, ...)` assumed every piece was monochrome. Invalid now.
#       New solver uses `components_mask(g != bg, connectivity=4)` and parses per-piece colors.
#     - Old edge/corner heuristics in `_build_requirements()` assumed special roles were encoded
#       by whole-color components. Multi-color anchored pieces would contaminate those signals,
#       so the variant uses only the fixed frame geometry plus isolated anchor targets.
#     - Old total-area logic counted every non-bg cell. Here singleton anchor markers must be
#       excluded before factorizing the output shape.
#   placement_legality_and_generator_guarantees:
#     - Non-frame anchor markers are chosen so they are 4-disconnected from the fixed frame and
#       from each other after non-frame bodies are removed from the template.
#     - Anchor colors are never used as body colors, so the marker role is visually legible.
#     - Frame generation avoids diagonal-only links so frame pieces remain 4-connected.
#   observability:
#     - Human-visible cue: every scattered patch contains one odd-colored pixel, and the target
#       rectangle contains a matching constellation of isolated odd-colored dots.
#     - These dots are not decorative: generator validation compares against the new solver on
#       anchored instances, and the old monochrome-component interpretation no longer applies.


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _piece_sig(cells: list[tuple[int, int, int]]) -> tuple[int, int, tuple[tuple[int, int, int], ...]]:
    arr = np.array([(r, c) for r, c, _ in cells], dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    norm = tuple(sorted((int(r) - r0, int(c) - c0, int(v)) for r, c, v in cells))
    return (r1 - r0 + 1), (c1 - c0 + 1), norm


def _sprite_from_component(g: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    cells: list[tuple[int, int, int]] = []
    counts: Counter[int] = Counter()
    for r, c in pts:
        rr = int(r) - r0
        cc = int(c) - c0
        v = int(g[int(r), int(c)])
        patch[rr, cc] = v
        cells.append((rr, cc, v))
        counts[v] += 1
    anchor_colors = [col for col, n in counts.items() if n == 1]
    assert len(counts) == 2 and len(anchor_colors) == 1, "each piece must contain one unique anchor color"
    anchor_color = int(anchor_colors[0])
    body_color = next(col for col, n in counts.items() if col != anchor_color and n >= 2)
    anchor = next((rr, cc) for rr, cc, v in cells if v == anchor_color)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": patch,
        "cells": tuple(sorted(cells)),
        "anchor": (int(anchor[0]), int(anchor[1])),
        "anchor_color": int(anchor_color),
        "body_color": int(body_color),
        "area": len(cells),
    }


def _is_lr_symmetric_union(parts: list[np.ndarray]) -> bool:
    all_pts = np.concatenate(parts)
    r0, c0, r1, c1 = bbox_coords(all_pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    bmp = np.zeros((h, w), dtype=bool)
    for pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _find_frame_parts(piece_comps: list[np.ndarray], H: int, W: int) -> list[np.ndarray]:
    candidates: list[list[np.ndarray]] = []
    for i, p1 in enumerate(piece_comps):
        r0, c0, r1, c1 = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1 - c0 + 1 == W and _is_lr_symmetric_union([p1]):
            candidates.append([p1])
        for j in range(i + 1, len(piece_comps)):
            p2 = piece_comps[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_lr_symmetric_union([p1, p2]):
                candidates.append([p1, p2])
    assert len(candidates) == 1, "frame pieces must be uniquely identifiable by their LR-symmetric bbox"
    return candidates[0]


def _coords_key(pts: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in pts)


def _assemble(
    pieces: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    fixed: list[tuple[int, int, dict]],
    anchor_targets: dict[int, set[tuple[int, int]]],
    disallow_touch: set[int],
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)

    def mirror_ok(proposed: dict[tuple[int, int], int], r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        if (r, mc) in proposed and proposed[(r, mc)] != col:
            return False
        v = board[r, mc]
        if v != -1 and v != col:
            return False
        rv = int(req[r, mc])
        if rv != -1 and rv != col:
            return False
        return True

    def can_place(piece: dict, r0: int, c0: int) -> bool:
        ar = r0 + int(piece["anchor"][0])
        ac = c0 + int(piece["anchor"][1])
        if (ar, ac) not in anchor_targets.get(int(piece["anchor_color"]), set()):
            return False
        proposed = {(r0 + dr, c0 + dc): int(col) for dr, dc, col in piece["cells"]}
        for (r, c), col in proposed.items():
            if not (0 <= r < H and 0 <= c < W):
                return False
            if board[r, c] != -1:
                return False
            rv = int(req[r, c])
            if rv != -1 and rv != col:
                return False
            if not mirror_ok(proposed, r, c, col):
                return False
            if col == int(piece["body_color"]) and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def placements_for(piece: dict) -> list[tuple[int, int]]:
        h = int(piece["patch"].shape[0])
        w = int(piece["patch"].shape[1])
        out = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                if can_place(piece, r0, c0):
                    out.append((r0, c0))
        return out

    def place(piece: dict, r0: int, c0: int) -> list[tuple[int, int]]:
        changed = []
        for dr, dc, col in piece["cells"]:
            r = r0 + dr
            c = c0 + dc
            board[r, c] = int(col)
            changed.append((r, c))
        if not piece.get("fixed"):
            ar = r0 + int(piece["anchor"][0])
            ac = c0 + int(piece["anchor"][1])
            anchor_targets[int(piece["anchor_color"])].remove((ar, ac))
        return changed

    def unplace(piece: dict, r0: int, c0: int, changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1
        if not piece.get("fixed"):
            ar = r0 + int(piece["anchor"][0])
            ac = c0 + int(piece["anchor"][1])
            anchor_targets[int(piece["anchor_color"])].add((ar, ac))

    for r0, c0, piece in fixed:
        changed = place(piece, r0, c0)
        assert len(changed) == int(piece["area"])

    remaining = [piece for piece in pieces if not piece.get("fixed")]

    def dfs(rem: list[dict]) -> bool:
        placed_now: list[tuple[dict, int, int, list[tuple[int, int]]]] = []
        cur = list(rem)
        while True:
            forced = None
            best_piece = None
            best_places = None
            for piece in cur:
                places = placements_for(piece)
                if not places:
                    for p, rr, cc, changed in reversed(placed_now):
                        unplace(p, rr, cc, changed)
                    return False
                places.sort()
                if best_places is None or len(places) < len(best_places):
                    best_piece = piece
                    best_places = places
                if len(places) == 1:
                    forced = (piece, places[0][0], places[0][1])
                    break
            if forced is None:
                break
            piece, r0, c0 = forced
            changed = place(piece, r0, c0)
            placed_now.append((piece, r0, c0, changed))
            cur.remove(piece)

        if not cur:
            ok = not any(v == -1 for v in board.ravel())
            ok = ok and all(len(v) == 0 for v in anchor_targets.values())
            ok = ok and np.array_equal(board, board[:, ::-1])
            if ok:
                return True
            for p, rr, cc, changed in reversed(placed_now):
                unplace(p, rr, cc, changed)
            return False

        assert best_piece is not None and best_places is not None
        cur.remove(best_piece)
        for r0, c0 in best_places:
            changed = place(best_piece, r0, c0)
            if dfs(cur):
                return True
            unplace(best_piece, r0, c0, changed)

        for p, rr, cc, changed in reversed(placed_now):
            unplace(p, rr, cc, changed)
        return False

    ok = dfs(remaining)
    assert ok, "assembly failed"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble anchored fragments into a left-right symmetric rectangle.

    Input structure:
    - bg is the most common color; every real fragment is a 4-connected non-bg component
      with one special anchor pixel and a larger single-color body.
    - One or two frame fragments already sit in their final rectangle positions.
    - Every scattered non-frame fragment has a matching isolated anchor marker inside the frame.

    Rule:
    - Ignore singleton non-bg components as target markers, and infer the output size from the
      multi-cell anchored fragments only.
    - Detect the in-place LR-symmetric frame, then place the remaining anchored fragments into
      the rectangle so the whole output is exactly mirror-symmetric.
    - A placement is legal only if the fragment's unique anchor pixel lands on a same-colored
      target marker and same-body-color fragments do not 4-touch.

    Key insight: the odd-colored pixel inside each fragment is not decorative; it is a precise
    alignment key. Shape and symmetry narrow the options, and the anchor markers finish the match.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = components_mask(g != bg, connectivity=4)
    markers = [pts for pts in comps if len(pts) == 1]
    piece_comps = [pts for pts in comps if len(pts) > 1]
    assert piece_comps, "input must contain real anchored pieces"

    total_area = sum(len(pts) for pts in piece_comps)
    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(piece_comps, H, W)
    frame_keys = {_coords_key(pts) for pts in frame_parts}
    frame_union = np.concatenate(frame_parts)
    fr0, fc0, _, _ = bbox_coords(frame_union)

    req = np.full((H, W), -1, dtype=int)
    pieces: list[dict] = []
    body_counts: Counter[int] = Counter()
    fixed: list[tuple[int, int, dict]] = []
    anchor_targets: dict[int, set[tuple[int, int]]] = {}

    for pts in piece_comps:
        piece = _sprite_from_component(g, pts, bg)
        body_counts[int(piece["body_color"])] += 1
        key = _coords_key(pts)
        if key in frame_keys:
            r0, c0, _, _ = piece["bbox"]
            or0 = int(r0) - fr0
            oc0 = int(c0) - fc0
            piece["fixed"] = True
            fixed.append((or0, oc0, piece))
            for dr, dc, col in piece["cells"]:
                req[or0 + dr, oc0 + dc] = int(col)
        pieces.append(piece)

    for pts in markers:
        r = int(pts[0, 0])
        c = int(pts[0, 1])
        or0 = r - fr0
        oc0 = c - fc0
        assert 0 <= or0 < H and 0 <= oc0 < W, "anchor marker must lie inside the frame bbox"
        col = int(g[r, c])
        rv = int(req[or0, oc0])
        assert rv == -1 or rv == col, "marker must not conflict with fixed frame"
        req[or0, oc0] = col
        anchor_targets.setdefault(col, set()).add((or0, oc0))

    for piece in pieces:
        if piece.get("fixed"):
            continue
        col = int(piece["anchor_color"])
        assert col in anchor_targets, "every scattered piece must have a target anchor marker"

    assert sum(len(v) for v in anchor_targets.values()) == sum(1 for p in pieces if not p.get("fixed"))
    disallow_touch = {col for col, n in body_counts.items() if n > 1}
    sol = _assemble(pieces, H, W, req, fixed, anchor_targets, disallow_touch)
    return sol.astype(int)


# --------------- generator (output-first anchored assembly variant) ---------------

# Output dimensions
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# Colors
_BODY_COLORS_MIN = 3
_BODY_COLORS_MAX = 4
_ANCHOR_COLORS_MIN = 2
_ANCHOR_COLORS_MAX = 3
_CENTER_CAP_PROB = 0.5

# Piece constraints
_MIN_PIECE_AREA = 3
_MAX_PIECE_AREA = 10
_PIECE_COUNT_MIN = 5
_PIECE_COUNT_MAX = 9
_MAX_CHUNK_SIZE = 6

# Canvas / scatter
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# Retry
_GEN_MAX_RETRIES = 400


def _valid_dims() -> list[tuple[int, int]]:
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h <= _HEIGHT_MAX:
                results.append((h, w))
    return results


_VALID_DIMS = [dims for dims in _valid_dims() if dims[0] % 2 == 0 and dims[0] >= 6]
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz"])
        if move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
    return cells, start_c, c


def _build_base_output(H: int, W: int) -> tuple[np.ndarray | None, int]:
    n_body = random.randint(_BODY_COLORS_MIN, _BODY_COLORS_MAX)
    palette = list(range(n_body))
    random.shuffle(palette)
    frame_color = palette[0]
    fill_pool = palette[1:]
    if not fill_pool:
        return None, -1

    grid = np.full((H, W), frame_color, dtype=int)
    prev_col = None
    for r0 in range(0, H, 2):
        opts = [f for f in fill_pool if f != prev_col] or fill_pool
        col = random.choice(opts)
        grid[r0 : r0 + 2, 1 : W - 1] = col
        prev_col = col
    return grid, frame_color


def _extract_base_pieces(out: np.ndarray, frame_color: int) -> tuple[list[dict], list[int]]:
    pieces = []
    frame_ids = []
    for col in sorted(map(int, np.unique(out))):
        for pts in components_mask(out == col, connectivity=4):
            coords = [(int(r), int(c)) for r, c in pts]
            piece = {
                "color": int(col),
                "coords": coords,
                "coord_set": frozenset(coords),
                "area": len(coords),
            }
            if col == frame_color:
                frame_ids.append(len(pieces))
            pieces.append(piece)
    return pieces, frame_ids


def _mirror_groups(pieces: list[dict], W: int) -> list[list[int]]:
    lookup = {(piece["color"], piece["coord_set"]): i for i, piece in enumerate(pieces)}
    groups = []
    seen = set()
    for i, piece in enumerate(pieces):
        if i in seen:
            continue
        mirrored = frozenset((r, W - 1 - c) for r, c in piece["coords"])
        j = lookup[(piece["color"], mirrored)]
        if j == i:
            groups.append([i])
            seen.add(i)
        else:
            groups.append(sorted([i, j]))
            seen.add(i)
            seen.add(j)
    return groups


def _frame_mask(shape: tuple[int, int], pieces: list[dict], frame_ids: list[int]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    for idx in frame_ids:
        for r, c in pieces[idx]["coords"]:
            mask[r, c] = True
    return mask


def _choose_anchor_positions(
    out: np.ndarray,
    pieces: list[dict],
    frame_ids: list[int],
    anchor_palette: list[int],
) -> tuple[np.ndarray | None, list[dict] | None, set[int]]:
    H, W = out.shape
    frame_set = set(frame_ids)
    frame_mask = _frame_mask((H, W), pieces, frame_ids)
    groups = _mirror_groups(pieces, W)
    random.shuffle(groups)

    anchored = out.copy()
    markers: list[dict] = []
    chosen_markers: set[tuple[int, int]] = set()
    real_frame_colors: set[int] = set()

    def touches_frame(r: int, c: int) -> bool:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and frame_mask[rr, cc]:
                return True
        return False

    def touches_marker(r: int, c: int) -> bool:
        return any((rr, cc) in chosen_markers for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))

    for group in groups:
        a_col = random.choice(anchor_palette)
        if len(group) == 1:
            idx = group[0]
            cells = list(pieces[idx]["coords"])
            if idx in frame_set:
                candidates = [(r, c) for r, c in cells if c == W // 2]
                if not candidates:
                    return None, None, set()
                r, c = random.choice(candidates)
                anchored[r, c] = a_col
                real_frame_colors.add(a_col)
            else:
                candidates = [
                    (r, c)
                    for r, c in cells
                    if c == W // 2 and not touches_frame(r, c) and not touches_marker(r, c)
                ]
                if not candidates:
                    return None, None, set()
                r, c = random.choice(candidates)
                anchored[r, c] = a_col
                markers.append({"pos": (r, c), "color": int(a_col)})
                chosen_markers.add((r, c))
        else:
            i, j = group
            left = i if np.mean([c for _, c in pieces[i]["coords"]]) <= np.mean([c for _, c in pieces[j]["coords"]]) else j
            right = j if left == i else i
            candidates = []
            for r, c in pieces[left]["coords"]:
                mr = r
                mc = W - 1 - c
                if (mr, mc) not in pieces[right]["coord_set"]:
                    continue
                if left not in frame_set:
                    if touches_frame(r, c) or touches_frame(mr, mc):
                        continue
                    if touches_marker(r, c) or touches_marker(mr, mc):
                        continue
                candidates.append((r, c, mr, mc))
            if not candidates:
                return None, None, set()
            r, c, mr, mc = random.choice(candidates)
            anchored[r, c] = a_col
            anchored[mr, mc] = a_col
            if left in frame_set:
                real_frame_colors.add(a_col)
            else:
                markers.append({"pos": (r, c), "color": int(a_col)})
                markers.append({"pos": (mr, mc), "color": int(a_col)})
                chosen_markers.add((r, c))
                chosen_markers.add((mr, mc))

    return anchored, markers, real_frame_colors


def _extract_piece_patch(out: np.ndarray, coords: list[tuple[int, int]], bg: int) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(out[r, c])
    return patch


def _make_output() -> tuple[np.ndarray | None, int, list[dict] | None, list[int] | None, list[dict] | None]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    base_out, frame_color = _build_base_output(H, W)
    if base_out is None:
        return None, 0, None, None, None
    if not np.array_equal(base_out, base_out[:, ::-1]):
        return None, 0, None, None, None

    base_pieces, frame_ids = _extract_base_pieces(base_out, frame_color)
    piece_areas = [int(piece["area"]) for piece in base_pieces]
    if any(a < _MIN_PIECE_AREA or a > _MAX_PIECE_AREA for a in piece_areas):
        return None, 0, None, None, None
    if not (_PIECE_COUNT_MIN <= len(base_pieces) <= _PIECE_COUNT_MAX):
        return None, 0, None, None, None
    if any(idx not in frame_ids for idx in range(len(base_pieces))) is False:
        return None, 0, None, None, None

    bg = random.randint(0, 9)
    body_colors = sorted(map(int, np.unique(base_out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(body_colors) + _ANCHOR_COLORS_MIN:
        return None, 0, None, None, None

    body_cmap = {int(a): available[i] for i, a in enumerate(body_colors)}
    body_used = set(body_cmap.values())
    anchor_available = [c for c in available if c not in body_used]
    if len(anchor_available) < _ANCHOR_COLORS_MIN:
        return None, 0, None, None, None
    n_anchor = random.randint(_ANCHOR_COLORS_MIN, min(_ANCHOR_COLORS_MAX, len(anchor_available)))
    anchor_palette = anchor_available[:n_anchor]

    recolored = np.vectorize(lambda x: body_cmap[int(x)])(base_out)
    recolored_pieces, recolored_frame_ids = _extract_base_pieces(recolored, body_cmap[frame_color])
    anchored_out, markers, _ = _choose_anchor_positions(recolored, recolored_pieces, recolored_frame_ids, anchor_palette)
    if anchored_out is None or markers is None:
        return None, 0, None, None, None
    if len(markers) != len(recolored_pieces) - len(recolored_frame_ids):
        return None, 0, None, None, None
    return anchored_out.astype(int), bg, markers, recolored_frame_ids, recolored_pieces


def _scatter(
    out: np.ndarray,
    bg: int,
    markers: list[dict],
    frame_ids: list[int],
    pieces: list[dict],
) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)
    canvas[off_r:off_r + oh, off_c:off_c + ow] = -1

    frame_set = set(frame_ids)
    patches = []
    frame_mask = np.zeros((oh, ow), dtype=bool)
    for idx, piece in enumerate(pieces):
        if idx in frame_set:
            for r, c in piece["coords"]:
                frame_mask[r, c] = True
            continue
        patch = _extract_piece_patch(out, piece["coords"], bg)
        patches.append(patch)

    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    for patch in patches:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    bbox_region = canvas[off_r:off_r + oh, off_c:off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    for marker in markers:
        mr, mc = marker["pos"]
        bbox_region[int(mr), int(mc)] = int(marker["color"])
    return canvas


def _gen_attempt() -> np.ndarray | None:
    out, bg, markers, frame_ids, pieces = _make_output()
    if out is None or markers is None or frame_ids is None or pieces is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_BODY_COLORS_MIN + _ANCHOR_COLORS_MIN <= len(np.unique(out)) <= _BODY_COLORS_MAX + _ANCHOR_COLORS_MAX):
        return None
    try:
        inp = _scatter(out, bg, markers, frame_ids, pieces)
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
    raise RuntimeError("failed to generate a valid anchored e8686506 input")
