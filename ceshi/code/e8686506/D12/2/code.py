# Variant of e8686506
# Dimension: D12 — 碎片的旋转/翻转约束 (meta)
# Change: 碎片不可旋转不可翻转，保持原始朝向直接拼合... → 部分碎片需旋转，部分不需

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D12 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original: every scattered fragment keeps the same orientation it has in the target rectangle
#     new: a visible subset of non-frame fragments is marked as rotatable; those marked pieces are
#       scattered after a non-zero 90-degree rotation, while unmarked pieces keep their output orientation
#   visual_form:
#     marker:
#       color: a fixed dedicated marker color (9) that never appears in the output
#       shape: a single pixel placed in 8-neighborhood contact with exactly one scattered fragment
#       meaning: "this fragment may be turned before assembly"
#     fixed_fragments:
#       appearance: same as the original task, no extra mark, orientation already matches output
#     rotatable_fragments:
#       appearance: same colored fragment plus one adjacent marker pixel
#       effect: the input shape is a rotated version of its output shape, so the solver must search
#         over the four 90-degree orientations after detecting the marker
#   concrete_flow_example:
#     original:
#       - suppose output contains a blue L-shape with coords ((0,0),(1,0),(1,1))
#       - input scatters the same blue L-shape unchanged
#       - main() records that exact signature and only searches translations
#     variant:
#       - the same blue L-shape is chosen as rotatable
#       - generator rotates it to ((0,1),(1,0),(1,1)) before scattering and places a marker pixel next to it
#       - solver first strips/ignores the marker, classifies the piece as rotatable, then tries all 4 rotations
#       - only one orientation fits the symmetric hole, so assembly still resolves by propagation
#     difference:
#       - translation-only matching is no longer valid for marked pieces
#       - requirement extraction and frame detection must ignore the marker color, otherwise singleton
#         markers could be mistaken for corner clues
#   complexity_calibration:
#     - only 2-4 pieces are marked in each generated input, so the new behavior is repeated and visible
#     - marked pieces are restricted to unique non-frame piece types with a genuinely non-trivial rotation,
#       preventing ambiguous many-copy matching and avoiding invisible "rotation" on symmetric shapes
#     - frame pieces remain unmarked and preserve their original placement, so the task stays close to
#       the source task instead of drifting into a full free-rotation puzzle
#   implicit_assumptions_to_fix:
#     - original piece grouping used the raw input signature as the final signature; this breaks once a
#       marked input piece is rotated, so marked pieces are stored with explicit orientation variants
#     - original comps_by_color included every non-bg color; this is unsafe because marker singletons could
#       trigger the old corner heuristic when there happen to be four of them
#     - original frame search inspected all non-bg 8-components; marker color must be excluded or a marker
#       could accidentally enter the frame-part candidate pool
#   placement_legality:
#     - every marked fragment is scattered as one composite patch: rotated piece + marker + one-cell bg border
#     - patches are placed with the same non-overlap / Chebyshev padding rule as the source generator
#     - the marker is chosen from the one-cell halo around the rotated fragment, so it is adjacent to exactly
#       one piece and remains visually tied to that piece
#   observability:
#     - humans can see that some fragments have an extra maroon dot and those are the only ones whose
#       orientation differs from the assembled rectangle
#     - the output contains no marker color, making the mark behave as an input-only instruction rather than
#       as a hidden bookkeeping channel
#   cleanliness:
#     - only one extra pixel is added per marked piece, so the scatter field stays legible
#     - the mark never overlays a piece and is not reused for any other role
#   generator_guarantees_needed_by_main:
#     - marker color 9 is absent from the output and from the background
#     - each marker touches exactly one non-frame piece
#     - each marked piece is selected from a unique (color, shape) type and is rotated by a non-zero angle
#     - at least two marked pieces and at least two unmarked non-frame pieces exist in every generated example


_MARKER_COLOR = 9
_ROTATABLE_MIN = 2
_ROTATABLE_MAX = 4


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _rotate_coords(coords: tuple[tuple[int, int], ...], k: int) -> tuple[tuple[int, int], ...]:
    return _mask_to_coords(np.rot90(_coords_to_mask(coords), int(k) % 4))


def _rotation_variants(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    out = []
    seen = set()
    for k in range(4):
        rc = _rotate_coords(coords, k)
        if rc not in seen:
            seen.add(rc)
            out.append(rc)
    return out


def _nontrivial_rotations(coords: tuple[tuple[int, int], ...]) -> list[int]:
    return [k for k in (1, 2, 3) if _rotate_coords(coords, k) != coords]


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    Marker pixels are already excluded from comps_by_color.
    """
    req = np.full((H, W), -1, dtype=int)

    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    corner_color = corner_colors[0] if corner_colors else None
    if corner_color is not None:
        req[0, 0] = corner_color
        req[0, W - 1] = corner_color
        req[H - 1, 0] = corner_color
        req[H - 1, W - 1] = corner_color

    bar_colors: list[tuple[int, int]] = []
    if W >= 3 and corner_color is not None:
        for col, comps in comps_by_color.items():
            for pts in comps:
                r0, c0, r1, c1 = bbox_coords(pts)
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh == 1 and ww == (W - 2) and len(pts) == ww:
                    bar_colors.append((r0, col))
                    break
        if bar_colors:
            assert len(bar_colors) <= 2, "horizontal edge bar colors must be top/bottom only"
            bar_colors.sort()
            top_col = bar_colors[0][1]
            bot_col = bar_colors[-1][1]
            req[0, 1 : W - 1] = top_col
            req[H - 1, 1 : W - 1] = bot_col

    vertical_line_candidates = []
    for col, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [_piece_sig(pts) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            vertical_line_candidates.append((col, h0))
    assert len(vertical_line_candidates) <= 1, "paired vertical edge-line color must be unique when present"
    if vertical_line_candidates:
        col, L = vertical_line_candidates[0]
        top = (H - L) // 2
        assert 0 <= top and top + L <= H, "paired vertical edge lines must fit in the output height"
        req[top : top + L, 0] = col
        req[top : top + L, W - 1] = col

    return req


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, vi: int, r0: int, c0: int) -> bool:
        t = types[ti]
        v = t["variants"][vi]
        col = int(t["col"])
        is_singleton = len(v["coords"]) == 1
        for dr, dc in v["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not mirror_ok(r, c, col):
                return False
            if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def place(ti: int, vi: int, r0: int, c0: int) -> list[tuple[int, int]]:
        changed = []
        for dr, dc in types[ti]["variants"][vi]["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = int(types[ti]["col"])
            changed.append((r, c))
        return changed

    def placements_for(ti: int) -> list[tuple[int, int, int]]:
        t = types[ti]
        out = []
        for vi, v in enumerate(t["variants"]):
            for r0 in range(H - int(v["h"]) + 1):
                for c0 in range(W - int(v["w"]) + 1):
                    if can_place(ti, vi, r0, c0):
                        out.append((vi, r0, c0))
        return out

    changed = True
    while changed:
        changed = False
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            assert len(places) > 0, "assembly dead end: a piece has no valid placement"
            if len(places) == 1:
                vi, r0, c0 = places[0]
                counts[ti] -= 1
                place(ti, vi, r0, c0)
                changed = True
            elif len(types[ti]["variants"]) == 1 and len(places) == counts[ti]:
                for vi, r0, c0 in places:
                    counts[ti] -= 1
                    place(ti, vi, r0, c0)
                changed = True
    assert all(c == 0 for c in counts), "assembly not purely propagation-solvable"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def _extract_marked_pieces(g: np.ndarray, bg: int) -> tuple[list[dict], dict[int, list[list[tuple[int, int]]]], dict[int, int]]:
    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    color_piece_count: dict[int, int] = {}
    pieces: list[dict] = []

    for col in sorted(set(map(int, np.unique(g)))):
        if col in {bg, _MARKER_COLOR}:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append({"col": int(col), "coords": coords, "rotatable": False})

    marker_pts = []
    if _MARKER_COLOR in set(map(int, np.unique(g))):
        marker_comps = components_mask(g == _MARKER_COLOR, connectivity=4)
        assert all(len(m) == 1 for m in marker_comps), "marker color must appear only as singleton pixels"
        marker_pts = [(int(m[0][0]), int(m[0][1])) for m in marker_comps]

    piece_cells = []
    for col, comps in comps_by_color.items():
        for pts in comps:
            piece_cells.append((int(col), {(int(r), int(c)) for r, c in pts}))

    assert len(piece_cells) == len(pieces)
    matched_piece_indices = set()
    for mr, mc in marker_pts:
        matches = []
        for i, (_col, cells) in enumerate(piece_cells):
            if any(max(abs(mr - r), abs(mc - c)) == 1 for r, c in cells):
                matches.append(i)
        assert len(matches) == 1, "each marker must touch exactly one fragment"
        pieces[matches[0]]["rotatable"] = True
        matched_piece_indices.add(matches[0])

    assert len(matched_piece_indices) == len(marker_pts), "markers must not collide onto the same piece"
    return pieces, comps_by_color, color_piece_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered colored fragments into one left-right symmetric rectangle.

    Input structure:
    - The background is the most common color; the rectangle frame still appears as 1-2 anchored pieces.
    - Non-frame fragments are scattered around the canvas.
    - A single maroon marker pixel next to a fragment means that fragment may need a 90-degree rotation before placement.

    Rule:
    - Ignore the marker pixels as output content, but use them to classify which fragments are rotatable.
    - Infer the output rectangle size from the total non-marker fragment area.
    - Pack all fragments into the rectangle so the final board is exactly left-right symmetric; fixed fragments keep their input orientation, while marked fragments may use any quarter-turn.

    Key insight:
    - The original frame lock and mirror constraints still pin down the rectangle, but the marked pieces can no longer
      be matched by translation alone. The decisive extra step is to strip the marker, expand those pieces into their
      unique 90-degree variants, and let constraint propagation choose the only orientation that fits.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    pieces, comps_by_color, color_piece_count = _extract_marked_pieces(g, bg)

    fixed_ctr = Counter((p["col"], p["coords"]) for p in pieces if not p["rotatable"])
    types = []
    for (col, coords), cnt in sorted(fixed_ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        types.append(
            {
                "col": int(col),
                "cnt": int(cnt),
                "variants": [{"coords": coords, "h": int(arr[:, 0].max() + 1), "w": int(arr[:, 1].max() + 1)}],
            }
        )

    rot_pieces = [p for p in pieces if p["rotatable"]]
    rot_pieces.sort(key=lambda p: (-len(p["coords"]), p["col"], p["coords"]))
    for p in rot_pieces:
        variants = []
        for coords in _rotation_variants(p["coords"]):
            arr = np.array(coords, dtype=int)
            variants.append({"coords": coords, "h": int(arr[:, 0].max() + 1), "w": int(arr[:, 1].max() + 1)})
        types.append({"col": int(p["col"]), "cnt": 1, "variants": variants})

    total_area = sum(int(t["cnt"]) * len(t["variants"][0]["coords"]) for t in types)
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg, _MARKER_COLOR}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_lr_symmetric(parts, h, w):
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and _is_lr_symmetric([(c1, p1)], H, W):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                frame_parts = [(c1, p1), (c2, p2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"

    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    return _assemble(types, H, W, req, disallow_touch).astype(int)


# --------------- generator (o5x) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_CENTER_CAP_PROB = 0.5

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 500


def _valid_dims():
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


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _zigzag_frame(H):
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "down":
            pass
        elif move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        else:
            c = 1 - c
    return cells, start_c, c


def _fill_holes(grid, H, W, frame1, frame2, split, fill_pool):
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        chunks = [cells[i : i + _MAX_CHUNK_SIZE] for i in range(0, len(cells), _MAX_CHUNK_SIZE)]
        for chunk in chunks:
            fc_set = {frame1 if r < split else frame2 for r, _ in chunk}
            avoid = fc_set | ({prev_col} if prev_col is not None else set())
            opts = [f for f in fill_pool if f not in avoid]
            if not opts:
                opts = [f for f in fill_pool if f not in fc_set] or fill_pool
            col = random.choice(opts)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            prev_col = col


def _build_output(H, W):
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = (
        H >= _TWO_FRAME_MIN_H
        and n_colors >= _TWO_FRAME_MIN_COLORS
        and random.random() < _TWO_FRAME_PROB
    )
    ci = 0
    frame1 = palette[ci]
    ci += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[ci]
        ci += 1
    fill_pool = palette[ci:]
    if not fill_pool:
        return None, set()

    half = W // 2
    frame_half, start_c, end_c = _zigzag_frame(H)
    split = H // 2 if frame1 != frame2 else H

    grid = np.full((H, W), -1, dtype=int)
    for r, c in frame_half:
        fc = frame1 if r < split else frame2
        grid[r, c] = fc
        grid[r, W - 1 - c] = fc

    if start_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[0, half] = frame1
    if end_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[H - 1, half] = frame2

    _fill_holes(grid, H, W, frame1, frame2, split, fill_pool)
    if np.any(grid < 0):
        return None, set()

    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_output(H, W)
    if out is None:
        return None, 0, set()

    bg_choices = [c for c in range(10) if c != _MARKER_COLOR]
    bg = random.choice(bg_choices)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c not in {bg, _MARKER_COLOR}]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg, real_fcs


def _patch_from_points(col: int, pts: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    coords = []
    for r, c in pts:
        rr = int(r) - r0
        cc = int(c) - c0
        patch[rr, cc] = int(col)
        coords.append((rr, cc))
    coords_t = tuple(sorted(coords))
    return {"col": int(col), "patch": patch, "coords": coords_t, "area": len(coords_t)}


def _extract_non_frame_entries(out: np.ndarray, bg: int, fcs: set[int]) -> list[dict]:
    entries = []
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            entries.append(_patch_from_points(int(col), pts, bg))
    random.shuffle(entries)
    entries.sort(key=lambda e: (-e["area"], random.random()))
    return entries


def _marked_patch(base_patch: np.ndarray, bg: int, k: int) -> np.ndarray:
    rot_patch = np.rot90(base_patch, int(k) % 4)
    h, w = rot_patch.shape
    padded = np.full((h + 2, w + 2), bg, dtype=int)
    padded[1 : 1 + h, 1 : 1 + w] = rot_patch
    occ = np.argwhere(rot_patch != bg) + 1
    candidates = []
    for r in range(h + 2):
        for c in range(w + 2):
            if padded[r, c] != bg:
                continue
            if any(max(abs(r - rr), abs(c - cc)) == 1 for rr, cc in occ):
                candidates.append((r, c))
    assert candidates, "marked patch needs at least one adjacent marker location"
    mr, mc = sorted(candidates)[0]
    padded[mr, mc] = _MARKER_COLOR
    return padded


def _choose_rotatable(entries: list[dict]) -> list[dict] | None:
    sig_counts = Counter((e["col"], e["coords"]) for e in entries)
    candidates = [
        e
        for e in entries
        if sig_counts[(e["col"], e["coords"])] == 1 and _nontrivial_rotations(e["coords"])
    ]
    if len(candidates) < _ROTATABLE_MIN:
        return None
    max_rot = min(_ROTATABLE_MAX, len(candidates), len(entries) - 2)
    if max_rot < _ROTATABLE_MIN:
        return None
    chosen = random.sample(candidates, random.randint(_ROTATABLE_MIN, max_rot))
    chosen_keys = {(e["col"], e["coords"]) for e in chosen}
    out = []
    for e in entries:
        e2 = dict(e)
        e2["rotatable"] = (e["col"], e["coords"]) in chosen_keys
        e2["k"] = 0
        if e2["rotatable"]:
            e2["k"] = random.choice(_nontrivial_rotations(e["coords"]))
        out.append(e2)
    return out


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> tuple[np.ndarray, dict]:
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

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    entries = _extract_non_frame_entries(out, bg, fcs)
    marked_entries = _choose_rotatable(entries)
    if marked_entries is None:
        raise ValueError

    for entry in marked_entries:
        patch = entry["patch"] if not entry["rotatable"] else _marked_patch(entry["patch"], bg, entry["k"])
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    meta = {
        "rotatable_count": sum(1 for e in marked_entries if e["rotatable"]),
        "turned_count": sum(1 for e in marked_entries if e["k"] != 0),
        "rotation_ks": sorted(int(e["k"]) for e in marked_entries if e["rotatable"]),
    }
    return canvas, meta


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    out, bg, fcs = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(out) for pts in components_mask(out == col, connectivity=4)):
        return None
    try:
        inp, meta = _scatter(out, bg, fcs)
        if meta["rotatable_count"] < _ROTATABLE_MIN or meta["turned_count"] != meta["rotatable_count"]:
            return None
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def _marked_piece_stats(inp: np.ndarray, out: np.ndarray) -> dict:
    bg = get_bg(inp)
    pieces, _, _ = _extract_marked_pieces(inp, bg)
    out_piece_sigs = Counter()
    for col in sorted(map(int, np.unique(out))):
        for pts in components_mask(out == col, connectivity=4):
            out_piece_sigs[(int(col), _piece_sig(pts)[2])] += 1

    rotatable = [p for p in pieces if p["rotatable"]]
    turned = 0
    for p in rotatable:
        if out_piece_sigs[(p["col"], p["coords"])] == 0:
            turned += 1
    return {"rotatable_count": len(rotatable), "turned_count": turned}


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D12 input")
