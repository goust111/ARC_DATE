# Combo variant of e8686506 (index 40)
# combo_spec: D7/0+D17/2
# Name: 保持输入尺寸+碎片移动轨迹连线

import itertools
import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the fused variant:
# - Original behavior: solve a compact HxW rectangle, then return only that cropped rectangle.
# - Fused behavior: still solve the same compact rectangle internally, but paste it back onto the
#   input canvas at the frame group's original bbox, and add one colored motion line per 4-connected
#   fragment from its source anchor to its assembled anchor.
# - Concrete example: if a 13x13 input contains a 5x5 frame already sitting at rows 4..8 / cols 6..10,
#   the original solver would output just that 5x5 patch. Here we keep the full 13x13 canvas, place the
#   5x5 assembly back at rows 4..8 / cols 6..10, mark a scattered fragment's source anchor with one pixel,
#   and draw a straight raster line to the matching anchor inside the 5x5 assembly.
# - Complexity guardrails:
#   1. The target location is not free-floating; it is anchored by the same LR-symmetric frame bbox that
#      the original task already exposes in the input, so D7/0 is visually recoverable.
#   2. "Center" on a discrete grid is made concrete as the occupied cell nearest the piece bbox midpoint.
#      This avoids off-piece fractional centers and keeps every trajectory endpoint visible.
#   3. Identical copies would otherwise be ambiguous after D17/2, so matching is done per (color, shape)
#      group with minimum total squared anchor distance. The generator keeps the frame bbox clear, which
#      makes these shortest assignments stable and keeps line clutter manageable.
# - Assumption audit against the original main():
#   * Total piece area, factor-choice logic, frame discovery, and propagation assembly all still hold.
#   * The old implicit assumption "output origin = (0,0)" no longer holds; we replace it with
#     "output top-left = detected frame bbox top-left".
#   * The old output had one layer only; the fused output now overlays start markers + paths + final board,
#     so rendering order matters. We draw paths first and paste the final rectangle last so the destination
#     remains readable.
# - Generator legality:
#   * Non-frame pieces are still scattered outside the blocked frame bbox, so they never overlap the target.
#   * All pieces stay in-bounds and separated exactly as in the original generator.
#   * Because the input is still the original scatter puzzle, any generated sample necessarily exercises
#     the fused rule through the changed output, not through hidden metadata.


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


def _piece_anchor(pts: np.ndarray) -> tuple[int, int]:
    """Choose the occupied cell closest to the bbox midpoint as a discrete center anchor."""
    r0, c0, r1, c1 = bbox_coords(pts)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    best = None
    for r, c in pts:
        rr = int(r)
        cc2 = int(c)
        score = ((rr - cr) ** 2 + (cc2 - cc) ** 2, rr, cc2)
        if best is None or score < best[0]:
            best = (score, (rr, cc2))
    assert best is not None
    return best[1]


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
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        is_singleton = len(t["coords"]) == 1
        for dr, dc in t["coords"]:
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

    def place(ti: int, r0: int, c0: int) -> None:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            board[r0 + dr, c0 + dc] = col

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [
            (r0, c0)
            for r0 in range(H - int(t["h"]) + 1)
            for c0 in range(W - int(t["w"]) + 1)
            if can_place(ti, r0, c0)
        ]

    changed = True
    while changed:
        changed = False
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            assert len(places) > 0, "assembly dead end: a piece has no valid placement"
            if len(places) == 1:
                counts[ti] -= 1
                place(ti, places[0][0], places[0][1])
                changed = True
            elif len(places) == counts[ti]:
                for r0, c0 in places:
                    counts[ti] -= 1
                    place(ti, r0, c0)
                changed = True
    assert all(c == 0 for c in counts), "assembly not purely propagation-solvable"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> tuple[list[tuple[int, np.ndarray]], int, int]:
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
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and _is_lr_symmetric([(c1, p1)], H, W):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W:
                if _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0 = int(all_frame_pts[:, 0].min())
    fc0 = int(all_frame_pts[:, 1].min())
    return frame_parts, fr0, fc0


def _solve_compact(grid: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

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
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts, fr0, fc0 = _find_frame_parts(g, bg, H, W)

    req = _build_requirements(H, W, comps_by_color)
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int), bg, fr0, fc0


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    records = []
    for col in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in components_mask(grid == col, connectivity=4):
            arr = np.asarray(pts, dtype=int)
            records.append({
                "col": int(col),
                "sig": _piece_sig(arr),
                "anchor": _piece_anchor(arr),
            })
    return records


def _best_assignment(srcs: list[dict], dsts: list[dict]) -> list[tuple[dict, dict]]:
    assert len(srcs) == len(dsts)
    if len(srcs) <= 1:
        return list(zip(srcs, dsts))

    srcs = sorted(srcs, key=lambda rec: rec["anchor"])
    dsts = sorted(dsts, key=lambda rec: rec["anchor"])
    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(len(dsts))):
        score = 0
        for i, j in enumerate(perm):
            sr, sc = srcs[i]["anchor"]
            tr, tc = dsts[j]["anchor"]
            score += (sr - tr) ** 2 + (sc - tc) ** 2
        tie = tuple(dsts[j]["anchor"] for j in perm)
        full_score = (score, tie)
        if best_score is None or full_score < best_score:
            best_score = full_score
            best_perm = perm
    assert best_perm is not None
    return [(srcs[i], dsts[j]) for i, j in enumerate(best_perm)]


def _match_records(src_records: list[dict], dst_records: list[dict]) -> list[tuple[dict, dict]]:
    src_groups: dict[tuple[int, tuple], list[dict]] = defaultdict(list)
    dst_groups: dict[tuple[int, tuple], list[dict]] = defaultdict(list)
    for rec in src_records:
        src_groups[(rec["col"], rec["sig"])].append(rec)
    for rec in dst_records:
        dst_groups[(rec["col"], rec["sig"])].append(rec)
    assert set(src_groups) == set(dst_groups), "source/target piece types must match"

    pairs = []
    for key in sorted(src_groups):
        srcs = src_groups[key]
        dsts = dst_groups[key]
        assert len(srcs) == len(dsts), "source/target multiplicities must match per piece type"
        pairs.extend(_best_assignment(srcs, dsts))
    return pairs


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Integer Bresenham raster line including both endpoints."""
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    out = []
    r, c = r0, c0
    while True:
        out.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return out


def _render_variant(grid: np.ndarray, compact: np.ndarray, bg: int, fr0: int, fc0: int) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    H, W = compact.shape

    target_canvas = np.full_like(g, bg)
    target_canvas[fr0 : fr0 + H, fc0 : fc0 + W] = compact

    src_records = _component_records(g, bg)
    dst_records = _component_records(target_canvas, bg)
    pairs = _match_records(src_records, dst_records)

    out = np.full_like(g, bg)
    path_cells = set()
    moving = 0
    total_dist2 = 0
    for src, dst in sorted(pairs, key=lambda pair: (pair[1]["anchor"], pair[0]["anchor"])):
        if src["anchor"] != dst["anchor"]:
            moving += 1
            sr, sc = src["anchor"]
            tr, tc = dst["anchor"]
            total_dist2 += (sr - tr) ** 2 + (sc - tc) ** 2
        for r, c in _line_cells(src["anchor"], dst["anchor"]):
            out[r, c] = src["col"]
            path_cells.add((r, c))

    out[fr0 : fr0 + H, fc0 : fc0 + W] = compact
    return out, {"moving_pieces": moving, "path_cells": len(path_cells), "total_dist2": total_dist2}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered fragments into the hidden symmetric rectangle, but keep the full canvas and show each fragment's movement path.

    Input structure:
    - The background is the most frequent color.
    - Non-background 4-connected fragments are the movable pieces; one 8-connected LR-symmetric frame group already sits at the rectangle's final canvas location.
    - Optional anchors from the base task still appear: 4 singleton corners, horizontal edge bars, and paired vertical side lines.

    Rule:
    - Recover the compact left-right symmetric rectangle exactly as in the base assembly task.
    - Place that solved rectangle back onto the input-sized canvas at the frame group's original bounding box.
    - For every fragment, connect its discrete center anchor at the source location to the matching anchor in the assembled rectangle using a straight line in the fragment's color.

    Key insight: the puzzle still solves in compact rectangle coordinates first; the new work is to lift that compact solution back into the original canvas frame and pair equal-shaped fragments with their assembled copies so their motion can be drawn.
    """
    compact, bg, fr0, fc0 = _solve_compact(grid)
    out, _ = _render_variant(grid, compact, bg, fr0, fc0)
    return out.astype(int)


# --------------- generator ---------------

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

# === CFG: Variant validation ===
_MIN_MOVING_PIECES = 4
_MIN_PATH_CELLS = 12

# === CFG: Retry ===
_GEN_MAX_RETRIES = 800


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


def _fill_holes(
    grid: np.ndarray,
    H: int,
    W: int,
    frame1: int,
    frame2: int,
    split: int,
    fill_pool: list[int],
) -> None:
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


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
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
    bg = random.randint(0, 9)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg, real_fcs


def _extract_non_frame_patches(out: np.ndarray, bg: int, fcs: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
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


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> np.ndarray:
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
    for patch in _extract_non_frame_patches(out, bg, fcs):
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
    return canvas


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
        inp = _scatter(out, bg, fcs)
        compact, bg2, fr0, fc0 = _solve_compact(inp)
        if not np.array_equal(compact, out):
            return None
        rendered, stats = _render_variant(inp, compact, bg2, fr0, fc0)
    except Exception:
        return None

    if stats["moving_pieces"] < _MIN_MOVING_PIECES:
        return None
    if stats["path_cells"] < _MIN_PATH_CELLS:
        return None
    if not np.array_equal(main(inp), rendered):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
