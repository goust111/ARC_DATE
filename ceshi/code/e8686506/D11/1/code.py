# Variant of e8686506
# Dimension: D11 — 多步处理逻辑 (multi_step)
# Change: Step 1: 识别所有非背景碎片; Step 2: 通过边缘匹配将碎片拼合成矩形; Step 3: 紧凑裁剪输出... → 拼合后进行对称修补

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D11 variant:
# - Concreteized behavior:
#   The input still contains the original task's scattered pieces plus a frame piece/group whose
#   relative position already encodes the output rectangle. The crucial difference is that the
#   assembled rectangle is no longer guaranteed to be color-symmetric. Instead, a few mirrored
#   cell pairs are intentionally damaged: one side may contain an isolated wrong-color pixel,
#   or both sides may contain conflicting isolated colors. After the rectangle is assembled,
#   symmetry is repaired as a separate pass:
#     * if one side of a mirror pair still matches its local same-color neighborhood better than
#       the other side, that supported color is copied across the axis;
#     * if neither side is locally supported, the pair is cleared to the background color.
#   Small trace example: a clean pair is (3,3). The generator changes it to (3,7) where the 7 is
#   an isolated singleton and the 3 still touches the rest of a 3-component. Assembly reconstructs
#   (3,7); repair sees support scores (>=1, 0) and outputs (3,3). Another pair is changed from
#   (4,4) to (8,2); both are isolated, so repair outputs (bg,bg).
# - Complexity calibration:
#   The repair rule uses only two visible branches and one visible cue: local same-color support.
#   That is richer than the original direct-symmetry assembly, but still learnable from 2-4 ARC
#   examples. Generator-side corruption is kept sparse and isolated so the difference is obvious
#   without turning the rectangle into noise.
# - Implicit assumption audit:
#   The original `_assemble` assumed color symmetry during placement (`mirror_ok`) and asserted the
#   assembled board already equals its mirror. That is invalid here, so assembly is separated from
#   repair and solved without color-mirror constraints. The original total-area→dimensions logic
#   remains valid because the pre-repair rectangle is still dense: damage changes colors only.
#   Frame locking still relies on the frame colors remaining untouched, so the generator never
#   corrupts frame cells.
# - Placement legality:
#   Scattering logic is preserved: non-frame patches are placed outside the blocked frame bbox with
#   padding, and the frame group is pasted back in place. Damage colors are chosen to avoid local
#   same-color adjacency, so newly created anomalies stay isolated and do not accidentally merge
#   with neighboring pieces.
# - Visual observability:
#   The examples visibly differ from the original task because the input pieces reconstruct a
#   rectangle containing one-sided color mistakes and two-sided color conflicts, while the output
#   removes or mirrors them. The rule is based on visible geometry and adjacency, not on color ids.
# - Visual cleanliness:
#   Corruptions are limited to a few non-frame cells and are kept non-adjacent, so the frame and
#   large filled regions remain readable. Output holes only appear at a small number of mirrored
#   locations.
# - Color inference:
#   Extra anomaly colors are used only as isolated mistakes. They vanish or get overwritten during
#   repair, and their role is structurally signaled by isolation, so no hidden color mapping is
#   required.


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


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
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
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
    pieces: list[dict] = []
    for t in types:
        for _ in range(int(t["cnt"])):
            pieces.append(
                {
                    "col": int(t["col"]),
                    "coords": tuple(t["coords"]),
                    "h": int(t["h"]),
                    "w": int(t["w"]),
                    "area": len(t["coords"]),
                }
            )
    pieces.sort(key=lambda p: (-p["area"], p["col"], p["coords"]))

    def can_place(piece: dict, r0: int, c0: int) -> bool:
        col = int(piece["col"])
        is_singleton = len(piece["coords"]) == 1
        for dr, dc in piece["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def placements_for(piece: dict) -> list[tuple[int, int]]:
        return [
            (r0, c0)
            for r0 in range(H - int(piece["h"]) + 1)
            for c0 in range(W - int(piece["w"]) + 1)
            if can_place(piece, r0, c0)
        ]

    def place(piece: dict, r0: int, c0: int) -> list[tuple[int, int]]:
        changed = []
        col = int(piece["col"])
        for dr, dc in piece["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    def search(remaining: list[dict]) -> bool:
        forced_changes: list[list[tuple[int, int]]] = []
        while True:
            if not remaining:
                return True
            best_idx = -1
            best_places: list[tuple[int, int]] | None = None
            for idx, piece in enumerate(remaining):
                places = placements_for(piece)
                if not places:
                    for changed in reversed(forced_changes):
                        unplace(changed)
                    return False
                if best_places is None or len(places) < len(best_places):
                    best_idx = idx
                    best_places = places
                    if len(best_places) == 1:
                        break
            assert best_places is not None
            if len(best_places) != 1:
                branch_piece = remaining.pop(best_idx)
                for r0, c0 in best_places:
                    changed = place(branch_piece, r0, c0)
                    if search(remaining):
                        return True
                    unplace(changed)
                remaining.insert(best_idx, branch_piece)
                for changed in reversed(forced_changes):
                    unplace(changed)
                return False
            forced_piece = remaining.pop(best_idx)
            forced_changes.append(place(forced_piece, best_places[0][0], best_places[0][1]))

    ok = search(pieces[:])
    assert ok, "assembly search failed"
    assert np.all(board != -1), "assembled board must be dense before repair"
    return board


def _support_score(board: np.ndarray, r: int, c: int, bg: int) -> int:
    col = int(board[r, c])
    if col == bg:
        return -1
    H, W = board.shape
    score = 0
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < H and 0 <= cc < W and int(board[rr, cc]) == col:
            score += 1
    return score


def _repair_symmetry(board: np.ndarray, bg: int) -> np.ndarray:
    H, W = board.shape
    out = board.copy()
    for r in range(H):
        for c in range(W // 2):
            mc = W - 1 - c
            a = int(board[r, c])
            b = int(board[r, mc])
            if a == b:
                out[r, c] = a
                out[r, mc] = b
                continue
            sa = _support_score(board, r, c, bg)
            sb = _support_score(board, r, mc, bg)
            if sa > sb and sa > 0:
                out[r, c] = a
                out[r, mc] = a
            elif sb > sa and sb > 0:
                out[r, c] = b
                out[r, mc] = b
            else:
                out[r, c] = bg
                out[r, mc] = bg
    return out


def _repair_event_counts(board: np.ndarray, bg: int) -> dict[str, int]:
    H, W = board.shape
    copied = 0
    cleared = 0
    unchanged_mismatch = 0
    for r in range(H):
        for c in range(W // 2):
            mc = W - 1 - c
            a = int(board[r, c])
            b = int(board[r, mc])
            if a == b:
                continue
            sa = _support_score(board, r, c, bg)
            sb = _support_score(board, r, mc, bg)
            if sa > sb and sa > 0:
                copied += 1
            elif sb > sa and sb > 0:
                copied += 1
            else:
                cleared += 1
    repaired = _repair_symmetry(board, bg)
    for r in range(H):
        for c in range(W // 2):
            mc = W - 1 - c
            if int(board[r, c]) != int(board[r, mc]) and int(repaired[r, c]) != int(repaired[r, mc]):
                unchanged_mismatch += 1
    return {"copy_pairs": copied, "clear_pairs": cleared, "bad_pairs_after_repair": unchanged_mismatch}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into a rectangle, then repair left-right symmetry.

    Input structure:
    - The most frequent color is the background; every other component is a scattered puzzle piece.
    - One or two 8-connected frame pieces still preserve the output rectangle's footprint and anchor cells.
    - A few non-frame cells may be intentionally wrong-colored on only one side of the future mirror axis.

    Rule:
    - Gather every non-background component and infer the unique portrait rectangle whose area matches the full piece set.
    - Repack the pieces into that dense rectangle using the frame/edge anchors and the no-same-color-touch constraint.
    - After assembly, scan mirrored cell pairs: copy the side with stronger local same-color support, or clear the pair to background when neither side is supported.

    Key insight:
    - The task is no longer "assemble directly into a perfectly symmetric picture". The rectangle must be reconstructed first, and only then denoised by symmetry using visible neighborhood support.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((col, coords))

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (col1, pts1) in enumerate(comps8_all):
        r0, c0, r1, c1 = bbox_coords(pts1)
        if r1 - r0 + 1 == H and c1 - c0 + 1 == W and _is_lr_symmetric([(col1, pts1)], H, W):
            frame_parts = [(col1, pts1)]
            break
        for j in range(i + 1, len(comps8_all)):
            col2, pts2 = comps8_all[j]
            union = np.concatenate([pts1, pts2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_lr_symmetric([(col1, pts1), (col2, pts2)], H, W):
                frame_parts = [(col1, pts1), (col2, pts2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric frame parts whose union fixes the output bbox"

    req = _build_requirements(H, W, comps_by_color)
    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = int(col)

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    assembled = _assemble(types, H, W, req, disallow_touch)
    repaired = _repair_symmetry(assembled, bg)
    assert np.array_equal(repaired, repaired[:, ::-1]), "repair must produce LR symmetry"
    return repaired.astype(int)


# --------------- generator (D11 variant) ---------------

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
_PIECE_COUNT_MAX = 20

# === CFG: Variant damage ===
_FILL_REPAIRS_REQUIRED = 1
_CLEAR_REPAIRS_REQUIRED = 1
_EXTRA_FILL_PROB = 0.35
_PAIR_GAP = 1

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 500


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


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        elif move == "diag":
            c = 1 - c
    return cells, start_c, c


def _fill_holes(grid: np.ndarray, H: int, W: int, frame1: int, frame2: int, split: int, fill_pool: list[int]) -> None:
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        chunks = [cells[i : i + _MAX_CHUNK_SIZE] for i in range(0, len(cells), _MAX_CHUNK_SIZE)]
        for chunk in chunks:
            frame_touch = {frame1 if r < split else frame2 for r, _ in chunk}
            avoid = frame_touch | ({prev_col} if prev_col is not None else set())
            opts = [f for f in fill_pool if f not in avoid]
            if not opts:
                opts = [f for f in fill_pool if f not in frame_touch] or fill_pool
            col = random.choice(opts)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            prev_col = col


def _build_clean_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
    half = W // 2
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = (
        H >= _TWO_FRAME_MIN_H
        and n_colors >= _TWO_FRAME_MIN_COLORS
        and random.random() < _TWO_FRAME_PROB
    )
    idx = 0
    frame1 = palette[idx]
    idx += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[idx]
        idx += 1
    fill_pool = palette[idx:]
    if not fill_pool:
        return None, set()

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


def _make_clean_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_clean_output(H, W)
    if out is None:
        return None, 0, set()
    bg = random.randint(0, 9)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    mapped = np.vectorize(lambda x: cmap[int(x)])(out)
    return mapped.astype(int), bg, real_fcs


def _local_colors(board: np.ndarray, r: int, c: int) -> set[int]:
    H, W = board.shape
    colors = {int(board[r, c])}
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < H and 0 <= cc < W:
            colors.add(int(board[rr, cc]))
    return colors


def _candidate_pairs(board: np.ndarray, frame_colors: set[int]) -> list[tuple[int, int]]:
    H, W = board.shape
    out = []
    for r in range(H):
        for c in range(W // 2):
            mc = W - 1 - c
            col = int(board[r, c])
            if col != int(board[r, mc]):
                continue
            if col in frame_colors:
                continue
            if _support_score(board, r, c, bg=-99) <= 0:
                continue
            out.append((r, c))
    random.shuffle(out)
    return out


def _far_enough(chosen: list[tuple[int, int]], cand: tuple[int, int], W: int) -> bool:
    r, c = cand
    pts = [(r, c), (r, W - 1 - c)]
    for rr, cc in chosen:
        cur = [(rr, cc), (rr, W - 1 - cc)]
        for a in pts:
            for b in cur:
                if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= _PAIR_GAP:
                    return False
    return True


def _pick_damage_color(
    board: np.ndarray,
    bg: int,
    frame_colors: set[int],
    r: int,
    c: int,
    extra_preferred: list[int],
    disallow: set[int] | None = None,
) -> int | None:
    forbid = _local_colors(board, r, c) | frame_colors | {bg}
    if disallow:
        forbid |= set(disallow)
    preferred = [col for col in extra_preferred if col not in forbid]
    if preferred:
        return random.choice(preferred)
    fallback = [col for col in range(10) if col not in forbid]
    if fallback:
        return random.choice(fallback)
    return None


def _damage_clean_board(clean: np.ndarray, bg: int, frame_colors: set[int]) -> tuple[np.ndarray | None, dict[str, int]]:
    board = clean.copy()
    present = set(map(int, np.unique(clean)))
    extra_preferred = [c for c in range(10) if c not in present and c != bg and c not in frame_colors]
    candidates = _candidate_pairs(clean, frame_colors)
    if len(candidates) < (_FILL_REPAIRS_REQUIRED + _CLEAR_REPAIRS_REQUIRED):
        return None, {}

    chosen: list[tuple[int, int]] = []
    stats = {"fill_pairs": 0, "clear_pairs": 0}

    def choose_candidate() -> tuple[int, int] | None:
        for cand in candidates:
            if cand not in chosen and _far_enough(chosen, cand, board.shape[1]):
                chosen.append(cand)
                return cand
        return None

    for _ in range(_FILL_REPAIRS_REQUIRED):
        cand = choose_candidate()
        if cand is None:
            return None, {}
        r, c = cand
        mc = board.shape[1] - 1 - c
        if random.random() < 0.5:
            target = (r, c)
        else:
            target = (r, mc)
        bad = _pick_damage_color(board, bg, frame_colors, target[0], target[1], extra_preferred)
        if bad is None:
            return None, {}
        board[target] = bad
        stats["fill_pairs"] += 1

    for _ in range(_CLEAR_REPAIRS_REQUIRED):
        cand = choose_candidate()
        if cand is None:
            return None, {}
        r, c = cand
        mc = board.shape[1] - 1 - c
        bad1 = _pick_damage_color(board, bg, frame_colors, r, c, extra_preferred)
        if bad1 is None:
            return None, {}
        bad2 = _pick_damage_color(board, bg, frame_colors, r, mc, extra_preferred, disallow={bad1})
        if bad2 is None:
            return None, {}
        board[r, c] = bad1
        board[r, mc] = bad2
        stats["clear_pairs"] += 1

    if random.random() < _EXTRA_FILL_PROB:
        cand = choose_candidate()
        if cand is not None:
            r, c = cand
            mc = board.shape[1] - 1 - c
            target = (r, c) if random.random() < 0.5 else (r, mc)
            bad = _pick_damage_color(board, bg, frame_colors, target[0], target[1], extra_preferred)
            if bad is not None:
                board[target] = bad
                stats["fill_pairs"] += 1

    repaired = _repair_symmetry(board, bg)
    counts = _repair_event_counts(board, bg)
    if counts["copy_pairs"] < _FILL_REPAIRS_REQUIRED or counts["clear_pairs"] < _CLEAR_REPAIRS_REQUIRED:
        return None, {}
    if counts["bad_pairs_after_repair"] != 0:
        return None, {}
    if np.array_equal(board, repaired):
        return None, {}
    return board, stats


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_colors: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in frame_colors:
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


def _scatter(out: np.ndarray, bg: int, frame_colors: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1
    for patch in _extract_non_frame_patches(out, bg, frame_colors):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in frame_colors:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    clean, bg, frame_colors = _make_clean_output()
    if clean is None:
        return None
    damaged, _ = _damage_clean_board(clean, bg, frame_colors)
    if damaged is None:
        return None
    repaired = _repair_symmetry(damaged, bg)
    if not np.array_equal(repaired, repaired[:, ::-1]):
        return None
    if _piece_count(damaged) < _PIECE_COUNT_MIN or _piece_count(damaged) > _PIECE_COUNT_MAX:
        return None
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(damaged) for pts in components_mask(damaged == col, connectivity=4)):
        return None
    counts = _repair_event_counts(damaged, bg)
    if counts["copy_pairs"] < _FILL_REPAIRS_REQUIRED or counts["clear_pairs"] < _CLEAR_REPAIRS_REQUIRED:
        return None
    try:
        inp = _scatter(damaged, bg, frame_colors)
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, repaired):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D11 input")
