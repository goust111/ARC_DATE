# Combo variant of 4a21e3da (index 2)
# combo_spec: D1/1+D5/3
# Name: 十字射线+大小排序分配

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, get_bg, is_contiguous

# Design analysis for this combo variant
# --------------------------------------
# source_task: 4a21e3da
#
# a) Concrete behavior and trace
#    Base task:
#    - a top/bottom control gives one vertical split, or a left/right control gives one horizontal split;
#    - only regions touching a control edge survive;
#    - survivors go to the corner matching their source side.
#    Combo here:
#    - every sample defines both one split row and one split column, so the central sprite is cut into
#      four quadrants even when there is only one border control;
#    - the split row/col are still read from visible geometry: the control gives one axis directly, and
#      the missing axis is recovered from the strongest fg line on that axis, with a center-biased tie-break;
#    - all four non-split quadrants survive, and their destination corners are determined only by size:
#      largest -> TL, 2nd -> TR, 3rd -> BL, smallest -> BR.
#    Example: with one top control at column c, the old task would only cut vertically and keep left/right
#    regions. Here the same input also contains a strong horizontal fg bar through the sprite on row r,
#    so the object is split into TL/TR/BL/BR. If those sizes are 8,5,3,6 in source-order, the output
#    corners become TL=8, TR=6, BL=5, BR=3 rather than preserving source quadrants.
#
# b) Complexity calibration
#    The generator keeps exactly four non-empty quadrants, enforces distinct piece sizes, and strongly
#    biases toward single-control instances so the D1 change appears often enough in few-shot examples.
#    Split bars are short fixed-width plus-shapes centered inside the sprite, which keeps the cross rule
#    stable across examples without introducing too many degrees of freedom.
#
# c) Review of broken old assumptions
#    - Old assumption: there may be only one split axis. Broken. The combo solver must always reason with
#      both r_split and c_split.
#    - Old assumption: non-edge-touching regions vanish. Broken. With one control that would erase half
#      the new quadrants and hide the D1 change; the combo keeps all four pieces.
#    - Old assumption: destination depends on source side. Broken. Source quadrant now matters only for
#      extracting the piece; corner assignment is global size ranking.
#    - New assumption: if only one control is present, the missing axis must still be inferable from the
#      sprite. The solver therefore uses "largest orthogonal support, then closest to center" rather than
#      assuming a unique longest row/col.
#
# d) Placement legality
#    - Grid stays square 18-20.
#    - Sprite bbox stays inside margins >= 4.
#    - The split row/col sit near the center of the bbox.
#    - Each quadrant piece is generated inside its own sub-bbox, flush toward the split, so pieces never
#      cross split lines in the input.
#    - Output uses translation only; compact pieces keep the four corner placements disjoint.
#
# e) Visual observability
#    Across samples, humans can see:
#    - one border control can still lead to four corner outputs, which forces the inference of a second
#      orthogonal cut inside the sprite;
#    - the four corner pieces are the same shapes as in the input quadrants, but corner destinations match
#      size order rather than source position;
#    - the split-line fg stays in the middle, so the "cut then move pieces" narrative remains legible.
#
# f) Visual cleanliness
#    Only the original three colors are used. The split marker is always a single central plus-shape,
#    pieces are compact, and the corner outputs are sparse. This avoids ambiguous overlaps or clutter.
#
# g) Change inferability
#    No hidden numeric property such as color id is used. The only global comparison is visible pixel count.
#    The generator avoids equal sizes and avoids the accidental identity case where source-quadrant order is
#    already TL > TR > BL > BR, because that would make sorted reassignment look like the old mapping.
#
# New solver assumptions audited against the generator
#    - There is exactly one fg color and one ctrl color on the border.
#    - There is at most one explicit vertical-edge control and at most one explicit horizontal-edge control.
#    - The missing split axis, if any, is chosen by largest orthogonal fg count with a center-biased tie-break.
#    - All four quadrants contain non-empty pieces in generated data.
#    - Piece sizes are pairwise distinct in generated data.


QUADRANTS = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
RANK_CORNERS = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]


def _slide_points_to_corner(
    pts: list[tuple[int, int]] | frozenset[tuple[int, int]],
    H: int,
    W: int,
    dh: str,
    dv: str,
) -> frozenset[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
    dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)
    return frozenset((int(r + dr), int(c + dc)) for r, c in pts)


def _choose_missing_row(g: np.ndarray, fg: int, c_split: int) -> int:
    H = int(g.shape[0])
    center = (H - 1) / 2.0
    candidates = [int(r) for r in range(H) if int(g[int(r), int(c_split)]) == int(fg)]
    assert candidates
    return min(
        candidates,
        key=lambda r: (
            -int(np.count_nonzero(g[int(r), :] == int(fg))),
            abs(float(r) - center),
            int(r),
        ),
    )


def _choose_missing_col(g: np.ndarray, fg: int, r_split: int) -> int:
    W = int(g.shape[1])
    center = (W - 1) / 2.0
    candidates = [int(c) for c in range(W) if int(g[int(r_split), int(c)]) == int(fg)]
    assert candidates
    return min(
        candidates,
        key=lambda c: (
            -int(np.count_nonzero(g[:, int(c)] == int(fg))),
            abs(float(c) - center),
            int(c),
        ),
    )


def _draw_col_segment(out: np.ndarray, g: np.ndarray, bg: int, color: int, col: int, start: int, stop: int) -> None:
    step = 1 if int(start) <= int(stop) else -1
    for rr in range(int(start), int(stop) + step, step):
        if int(g[int(rr), int(col)]) == int(bg):
            out[int(rr), int(col)] = int(color)


def _draw_row_segment(out: np.ndarray, g: np.ndarray, bg: int, color: int, row: int, start: int, stop: int) -> None:
    step = 1 if int(start) <= int(stop) else -1
    for cc in range(int(start), int(stop) + step, step):
        if int(g[int(row), int(cc)]) == int(bg):
            out[int(row), int(cc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the sprite by a cross and send all four pieces to corners by size rank.

    - Border control(s) still mark the main split axis directly.
    - The missing orthogonal axis is inferred from the unique longest fg line crossing that axis.
    - Fg on the split row/col stays in place.
    - Every non-split quadrant survives.
    - Quadrant pieces are reassigned by descending size to TL, TR, BL, BR.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]

    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if int(r) in (0, H - 1):
            c_split = int(c)
        if int(c) in (0, W - 1):
            r_split = int(r)

    assert c_split is not None or r_split is not None
    if c_split is None:
        c_split = _choose_missing_col(g, fg, int(r_split))
    if r_split is None:
        r_split = _choose_missing_row(g, fg, int(c_split))

    out = np.full((H, W), int(bg), dtype=int)

    col_rows = [int(r) for r in range(H) if int(g[int(r), int(c_split)]) == int(fg)]
    row_cols = [int(c) for c in range(W) if int(g[int(r_split), int(c)]) == int(fg)]
    assert col_rows and row_cols
    row_min, row_max = int(min(row_cols)), int(max(row_cols))
    col_min, col_max = int(min(col_rows)), int(max(col_rows))

    for r, c in ctrl_pts:
        if int(r) == 0:
            _draw_col_segment(out, g, bg, ctrl, int(c_split), 0, int(max(col_rows)))
            _draw_row_segment(out, g, bg, ctrl, int(r_split), row_min, row_max)
        elif int(r) == int(H - 1):
            _draw_col_segment(out, g, bg, ctrl, int(c_split), int(H - 1), int(min(col_rows)))
            _draw_row_segment(out, g, bg, ctrl, int(r_split), row_min, row_max)
        elif int(c) == 0:
            _draw_row_segment(out, g, bg, ctrl, int(r_split), 0, int(max(row_cols)))
            _draw_col_segment(out, g, bg, ctrl, int(c_split), col_min, col_max)
        elif int(c) == int(W - 1):
            _draw_row_segment(out, g, bg, ctrl, int(r_split), int(W - 1), int(min(row_cols)))
            _draw_col_segment(out, g, bg, ctrl, int(c_split), col_min, col_max)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    split_line_pts: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if int(rr) == int(r_split) or int(cc) == int(c_split):
            out[int(rr), int(cc)] = int(fg)
            split_line_pts.add((int(rr), int(cc)))

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if (rr, cc) in split_line_pts:
            continue
        hs = "T" if int(rr) < int(r_split) else "B"
        vs = "L" if int(cc) < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(rr), int(cc)))

    ordered_regions = sorted(
        regions.items(),
        key=lambda item: (
            -len(item[1]),
            min(r for r, _ in item[1]),
            min(c for _, c in item[1]),
            item[0],
        ),
    )

    for (_reg, pts), (dh, dv) in zip(ordered_regions, RANK_CORNERS):
        for rr, cc in _slide_points_to_corner(pts, H, W, dh, dv):
            out[int(rr), int(cc)] = int(fg)

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(target_px - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = [
            (nr, nc)
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(nr, nc)}) - min(r for r, _ in cells | {(nr, nc)}) + 1 <= max_h
            and max(c for _, c in cells | {(nr, nc)}) - min(c for _, c in cells | {(nr, nc)}) + 1 <= max_w
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return frozenset((r - min(rs), c - min(cs)) for r, c in cells)


def _sample_piece(
    target_px: int,
    sub_h: int,
    sub_w: int,
    quadrant: tuple[str, str],
) -> frozenset[tuple[int, int]] | None:
    max_h = int(min(sub_h, 4))
    max_w = int(min(sub_w, 4))
    if target_px > max_h * max_w:
        return None

    for _ in range(120):
        piece = _grow_piece(max_h, max_w, target_px)
        if len(piece) != int(target_px):
            continue
        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1
        if ph < 2 or pw < 2:
            continue
        if quadrant == ("T", "L") and (ph - 1, pw - 2) not in piece:
            continue
        if quadrant == ("T", "R") and (ph - 1, 1) not in piece:
            continue
        if quadrant == ("B", "L") and (0, pw - 2) not in piece:
            continue
        if quadrant == ("B", "R") and (0, 1) not in piece:
            continue
        return piece
    return None


def _choose_ranked_sizes(min_total: int, max_total: int) -> tuple[int, int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for a in range(5, 10):
        for b in range(4, a):
            for c in range(3, b):
                for d in range(3, c):
                    ranked = (int(a), int(b), int(c), int(d))
                    total = int(sum(ranked))
                    if not (int(min_total) <= total <= int(max_total)):
                        continue
                    if int(ranked[0]) - int(ranked[-1]) < 3:
                        continue
                    candidates.append(ranked)
    if not candidates:
        return None
    return random.choice(candidates)


def _piece_sort_key(pts: frozenset[tuple[int, int]]) -> tuple[int, int, int]:
    return (-len(pts), min(r for r, _ in pts), min(c for _, c in pts))


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[20, 1])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    sprite_h = random.choice([7, 8])
    sprite_w = random.choice([9, 10])
    sprite_r0 = random.randint(4, H - 5 - sprite_h + 1)
    sprite_c0 = random.randint(4, W - 5 - sprite_w + 1)
    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    top_rows = 3 if sprite_h == 7 else random.choice([3, 4])
    left_cols = 4 if sprite_w == 9 else random.choice([4, 5])
    r_split = sprite_r0 + top_rows
    c_split = sprite_c0 + left_cols

    ctrl_pts: list[tuple[int, int]] = []
    for edge in ctrl_edges:
        if edge == "top":
            ctrl_pts.append((0, int(c_split)))
        elif edge == "bottom":
            ctrl_pts.append((H - 1, int(c_split)))
        elif edge == "left":
            ctrl_pts.append((int(r_split), 0))
        elif edge == "right":
            ctrl_pts.append((int(r_split), W - 1))

    row_cols = list(range(int(c_split) - 2, int(c_split) + 3))
    col_rows = list(range(int(r_split) - 2, int(r_split) + 3))
    split_pixels = {(int(r_split), int(c)) for c in row_cols} | {(int(r), int(c_split)) for r in col_rows}

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs == "T" else int(r_split + 1)
        r1 = int(r_split - 1) if hs == "T" else sprite_r1
        c0 = sprite_c0 if vs == "L" else int(c_split + 1)
        c1 = int(c_split - 1) if vs == "L" else sprite_c1
        return r0, r1, c0, c1

    quad_dims = {reg: sub_bbox(*reg) for reg in QUADRANTS}
    for r0, r1, c0, c1 in quad_dims.values():
        if r1 < r0 or c1 < c0:
            return None
        if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2:
            return None

    bbox_area = sprite_h * sprite_w
    min_total = max(16, int(np.ceil(0.39 * bbox_area)) - len(split_pixels))
    max_total = int(np.floor(0.60 * bbox_area)) - len(split_pixels)
    if max_total < min_total:
        return None

    ranked_sizes = _choose_ranked_sizes(min_total, max_total)
    if ranked_sizes is None:
        return None

    permutations = [perm for perm in itertools.permutations(ranked_sizes) if tuple(perm) != ranked_sizes]
    if not permutations:
        return None
    assigned_sizes = random.choice(permutations)
    size_by_reg = {reg: int(size) for reg, size in zip(QUADRANTS, assigned_sizes)}

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    all_fg = set(split_pixels)

    for reg in QUADRANTS:
        hs, vs = reg
        target_px = int(size_by_reg[reg])
        r0, r1, c0, c1 = quad_dims[reg]
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1

        piece = _sample_piece(target_px, sub_h, sub_w, reg)
        if piece is None:
            return None

        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1
        inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
        inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if hs == "T" and r >= int(r_split):
                return None
            if hs == "B" and r <= int(r_split):
                return None
            if vs == "L" and c >= int(c_split):
                return None
            if vs == "R" and c <= int(c_split):
                return None
            if (r, c) in all_fg:
                return None

        input_pieces[reg] = inp_pts
        all_fg |= set(inp_pts)

    fg_arr = np.array(list(all_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    density = len(all_fg) / float(ah * aw)
    if density < 0.39 or density > 0.60:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_fg:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    ranked_regs = sorted(QUADRANTS, key=lambda reg: _piece_sort_key(input_pieces[reg]))
    if ranked_regs == QUADRANTS:
        return None

    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    occupied_out = set(split_pixels)
    for reg, (dh, dv) in zip(ranked_regs, RANK_CORNERS):
        out_pts = _slide_points_to_corner(input_pieces[reg], H, W, dh, dv)
        if occupied_out & set(out_pts):
            return None
        occupied_out |= set(out_pts)
        output_pieces[reg] = out_pts

    inp = np.full((H, W), bg, dtype=int)
    for r, c in all_fg:
        inp[int(r), int(c)] = int(fg)
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = int(ctrl_color)

    out = np.full((H, W), bg, dtype=int)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)

    col_rows_out = [int(r) for r in range(H) if int(out[int(r), int(c_split)]) == int(fg)]
    row_cols_out = [int(c) for c in range(W) if int(out[int(r_split), int(c)]) == int(fg)]
    if not col_rows_out or not row_cols_out:
        return None

    for cr, cc in ctrl_pts:
        if int(cr) == 0:
            for r in range(0, int(max(col_rows_out)) + 1):
                if int(out[int(r), int(c_split)]) == int(bg):
                    out[int(r), int(c_split)] = int(ctrl_color)
            for c in range(int(min(row_cols_out)), int(max(row_cols_out)) + 1):
                if int(out[int(r_split), int(c)]) == int(bg):
                    out[int(r_split), int(c)] = int(ctrl_color)
        elif int(cr) == int(H - 1):
            for r in range(int(H - 1), int(min(col_rows_out)) - 1, -1):
                if int(out[int(r), int(c_split)]) == int(bg):
                    out[int(r), int(c_split)] = int(ctrl_color)
            for c in range(int(min(row_cols_out)), int(max(row_cols_out)) + 1):
                if int(out[int(r_split), int(c)]) == int(bg):
                    out[int(r_split), int(c)] = int(ctrl_color)
        elif int(cc) == 0:
            for c in range(0, int(max(row_cols_out)) + 1):
                if int(out[int(r_split), int(c)]) == int(bg):
                    out[int(r_split), int(c)] = int(ctrl_color)
            for r in range(int(min(col_rows_out)), int(max(col_rows_out)) + 1):
                if int(out[int(r), int(c_split)]) == int(bg):
                    out[int(r), int(c_split)] = int(ctrl_color)
        elif int(cc) == int(W - 1):
            for c in range(int(W - 1), int(min(row_cols_out)) - 1, -1):
                if int(out[int(r_split), int(c)]) == int(bg):
                    out[int(r_split), int(c)] = int(ctrl_color)
            for r in range(int(min(col_rows_out)), int(max(col_rows_out)) + 1):
                if int(out[int(r), int(c_split)]) == int(bg):
                    out[int(r), int(c_split)] = int(ctrl_color)
    for cr, cc in ctrl_pts:
        out[int(cr), int(cc)] = int(ctrl_color)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
