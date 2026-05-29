# Combo variant of 446ef5d2 (index 1)
# combo_spec: D1/2+D4/3
# Name: 旋转匹配后镜像输出

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location


# Design analysis for the fused variant
# =====================================
# a) Concreteized behavior and trace
# - Original task:
#   1. detect the 3-cell L marker and infer which rectangle corner of the final placement box
#      is pinned at the marker gap;
#   2. read every non-marker object as one axis-aligned rectangular piece;
#   3. tile those pieces into one rectangle using boundary=rect_col and edge-color matching;
#   4. paste that assembled rectangle at the marker corner unchanged.
# - Fused variant:
#   1. keep marker detection and rectangle-piece decomposition unchanged;
#   2. during assembly, each non-anchor piece may be tested in any quarter-turn orientation
#      before edge matching, so a 2x4 piece seen in input can become 4x2 in the assembled box;
#   3. after the entire rectangle is reconstructed in its canonical orientation, reflect the
#      whole rectangle once:
#        TL/BL marker corner -> left-right reflection (`np.fliplr`)
#        TR/BR marker corner -> up-down reflection (`np.flipud`)
#      then paste the reflected rectangle into the same placement box.
# - Small trace:
#   suppose the true rectangle is 6x8 and the marker gap says `TL`.
#   Original solver: assemble `RECT`, output `RECT` with top-left `(mr, mc)`.
#   Fused solver: assemble `RECT` by trying piece rotations, then output `fliplr(RECT)` at the
#   same `(mr, mc)`. The placement box is unchanged, but both the internal tiling search and the
#   final visible arrangement differ from the original task.
#
# b) Complexity calibration
# - Rotation adds a bounded x4 branching factor per piece, which is still ARC-scale because the
#   pieces remain rectangles and only quarter-turns are allowed.
# - Reflection stays low-branching: exactly two axis choices keyed by marker side, not by color id.
# - The generator forces at least one genuinely rotated non-anchor piece and rejects rectangles
#   whose post-assembly reflection changes too few cells. This keeps both subrules visible.
#
# c) Implicit assumptions broken by the fusion
# - Original `pieces[i]` had one fixed `(h, w)`; after D1/2, placement/boundary/neighbor checks
#   must consume the currently chosen rotated array, not a fixed stored shape.
# - Original output placement assumed the assembled rectangle itself is final; after D4/3, that is
#   false and the whole rectangle must be reflected first.
# - Original anchor pre-placement silently assumed the input anchor piece can be dropped into the
#   placement-box corner unchanged. In the fused version this is still valid only because the
#   generator keeps the anchor unrotated in the input, guaranteeing the marker gap sits on the
#   correct physical corner of that scattered piece.
# - The reflection is intentionally applied only after assembly. Reflecting individual pieces would
#   violate the requested combo and would create an extra hidden degree of freedom humans could not
#   infer from 2-4 examples.
#
# d) Placement legality
# - Input generation is still output-first: build one canonical rectangle, partition it, keep the
#   anchor piece adjacent to the marker, rotate only scattered non-anchor pieces, and place them on
#   the background with padding >= 1 so pieces and marker never overlap.
# - Output legality is preserved because reflection keeps `(H, W)` unchanged, so the original
#   marker-derived placement box remains in-bounds after the final transform.
#
# e) Visual observability
# - Humans can see the rotation rule because at least one non-square non-anchor piece is quarter-
#   turned in the input and must be mentally reoriented to make edge colors line up.
# - Humans can see the reflection rule because the rectangle interior is made asymmetric enough that
#   `RECT` and `reflect(RECT)` differ by many cells; otherwise the sample is rejected.
#
# f) Visual cleanliness
# - The scene still has only three visual layers: uniform background, one 3-cell L marker, and
#   separated solid rectangular pieces. No overlapping annotations or extra symbols are introduced.
#
# g) Inferability of changes
# - No color-id arithmetic is used. Colors keep their original roles: bg, marker, rect_col, and
#   1-3 interior colors. The only new control signals are geometric: piece orientation during
#   matching and marker-corner choice of reflection axis after assembly.
#
# h) Anti-degeneration checks used by the generator
# - Reject if `reflect(rect, corner)` is too similar to `rect`; otherwise D4/3 would be invisible.
# - Reject if a no-rotation solver with reflection can still solve the same input; otherwise D1/2
#   would be inactive for that sample.
# - Reject if the rotated-piece count is zero.
#
# i) New solver assumptions that the generator explicitly guarantees
# - The marker is the only non-bg color appearing exactly 3 times in a 2x2 L.
# - Every scattered object remains a filled axis-aligned rectangle after any rotation.
# - The anchor piece is unrotated and touches the marker gap at the matching corner.
# - At least one non-anchor piece needs a non-zero quarter-turn to recover the canonical assembly.


def _marker_info(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
        if len(miss) != 1:
            continue
        missing = miss[0]
        if missing == (r0, c0):
            corner = "BR"
        elif missing == (r0, c1):
            corner = "BL"
        elif missing == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(col), (int(missing[0]), int(missing[1])), (r0, c0, r1, c1), corner
    raise AssertionError("marker not found")


def _decompose_rectangles(g: np.ndarray, bg: int, marker_col: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    occ = (g != int(bg)) & (g != int(marker_col))
    used = np.zeros_like(occ, dtype=bool)
    out: list[tuple[int, int, int, int, np.ndarray]] = []
    H, W = g.shape
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < W and bool(occ[rr, cc + w]) and not bool(used[rr, cc + w]):
            w += 1
        h = 0
        while rr + h < H and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        out.append((rr, cc, h, w, g[rr : rr + h, cc : cc + w].copy()))
    return out


def _top_left_from_corner(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - W + 1)
    if corner == "BL":
        return int(mr - H + 1), int(mc)
    return int(mr - H + 1), int(mc - W + 1)


def _reflect_axis_for_corner(corner: str) -> str:
    return "vertical" if corner in ("TL", "BL") else "horizontal"


def _reflect_rect(rect: np.ndarray, corner: str) -> np.ndarray:
    if _reflect_axis_for_corner(corner) == "vertical":
        return np.fliplr(rect)
    return np.flipud(rect)


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.array(v1, dtype=int)
    b = np.array(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def _unique_rotations(piece: np.ndarray, allow_rotation: bool) -> list[tuple[int, np.ndarray]]:
    ks = [0, 1, 2, 3] if allow_rotation else [0]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[tuple[int, np.ndarray]] = []
    for k in ks:
        arr = np.rot90(piece, int(k))
        key = (tuple(map(int, arr.shape)), arr.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((int(k), arr))
    return out


def _solve(
    grid: np.ndarray,
    *,
    allow_rotation: bool,
    apply_reflection: bool,
    return_meta: bool = False,
):
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    marker_col, missing, _bbox, corner = _marker_info(g, bg)
    rects = _decompose_rectangles(g, bg, marker_col)

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    mr, mc = map(int, missing)
    anchor_idx = None
    anchor_corner = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        corners = {
            (rr, cc): "TL",
            (rr, cc + w - 1): "TR",
            (rr + h - 1, cc): "BL",
            (rr + h - 1, cc + w - 1): "BR",
        }
        if (mr, mc) in corners:
            anchor_idx = i
            anchor_corner = corners[(mr, mc)]
            break

    assert anchor_idx is not None and anchor_corner is not None

    pieces = [a for _r, _c, _h, _w, a in rects]
    variants = [_unique_rotations(p, allow_rotation) for p in pieces]
    piece_sizes = [(int(p.shape[0]), int(p.shape[1])) for p in pieces]

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        ww = total_area // hh
        dims.append((int(hh), int(ww)))
        if hh != ww:
            dims.append((int(ww), int(hh)))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(H, W) for H, W in dims if fits(H, W)]
    dims.sort(key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def assemble(H: int, W: int):
        occ2 = np.full((H, W), -1, dtype=int)
        used = [False] * len(pieces)
        chosen_k: list[int | None] = [None] * len(pieces)

        def boundary_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = map(int, a.shape)
            if r == 0 and not np.all(a[0, :] == int(rect_col)):
                return False
            if r + h == H and not np.all(a[-1, :] == int(rect_col)):
                return False
            if c == 0 and not np.all(a[:, 0] == int(rect_col)):
                return False
            if c + w == W and not np.all(a[:, -1] == int(rect_col)):
                return False
            return True

        def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = map(int, a.shape)
            if c > 0:
                left = occ2[r : r + h, c - 1]
                m = left != -1
                if m.any() and not _compat(left[m], a[:, 0][m], rect_col):
                    return False
            if c + w < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and not _compat(right[m], a[:, -1][m], rect_col):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and not _compat(up[m], a[0, :][m], rect_col):
                    return False
            if r + h < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and not _compat(down[m], a[-1, :][m], rect_col):
                    return False
            return True

        def can_place(a: np.ndarray, r: int, c: int) -> bool:
            h, w = map(int, a.shape)
            if r + h > H or c + w > W:
                return False
            if (occ2[r : r + h, c : c + w] != -1).any():
                return False
            return boundary_ok(a, r, c) and neighbor_ok(a, r, c)

        def place(i: int, k: int, a: np.ndarray, r: int, c: int) -> None:
            h, w = map(int, a.shape)
            occ2[r : r + h, c : c + w] = a
            used[i] = True
            chosen_k[i] = int(k)

        def unplace(i: int, a: np.ndarray, r: int, c: int) -> None:
            h, w = map(int, a.shape)
            occ2[r : r + h, c : c + w] = -1
            used[i] = False
            chosen_k[i] = None

        anchor_placed = False
        for k, a in variants[anchor_idx]:
            if int(k) != 0:
                continue
            if anchor_corner != corner:
                continue
            ah, aw = map(int, a.shape)
            ar0, ac0 = _top_left_from_corner(0, 0, ah, aw, corner)
            if can_place(a, ar0, ac0):
                place(anchor_idx, k, a, ar0, ac0)
                anchor_placed = True
                break
        if not anchor_placed:
            return None

        order = sorted(
            range(len(pieces)),
            key=lambda i: (-piece_sizes[i][0] * piece_sizes[i][1], len(variants[i]), -piece_sizes[i][0], -piece_sizes[i][1], i),
        )

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

        solved = False

        def bt() -> None:
            nonlocal solved
            nxt = next_empty()
            if nxt is None:
                solved = True
                return
            r, c = nxt
            for i in order:
                if used[i]:
                    continue
                for k, a in variants[i]:
                    if can_place(a, r, c):
                        place(i, k, a, r, c)
                        bt()
                        if solved:
                            return
                        unplace(i, a, r, c)

        bt()
        if not solved:
            return None
        return occ2, {"piece_rotations": [0 if k is None else int(k) for k in chosen_k]}

    ass = None
    used_dim = None
    meta = None
    for H, W in dims:
        result = assemble(H, W)
        if result is not None:
            ass, meta = result
            used_dim = (int(H), int(W))
            break

    assert ass is not None and used_dim is not None

    H, W = used_dim
    final_rect = _reflect_rect(ass, corner) if apply_reflection else ass
    tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = final_rect
    if return_meta:
        assert meta is not None
        return out, {
            "piece_rotations": meta["piece_rotations"],
            "corner": corner,
            "rect_col": rect_col,
            "assembled_rect": ass.copy(),
            "output_rect": final_rect.copy(),
            "top_left": (int(tr), int(tc)),
            "dims": (int(H), int(W)),
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, allow_rotation=True, apply_reflection=True)


def _build_rectangle(RH: int, RW: int, rect_col: int, interior_cols: list[int]) -> np.ndarray | None:
    rect = np.full((RH, RW), int(rect_col), dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), int(rect_col), dtype=int)
    target_fill = random.uniform(0.35, 0.85)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, color in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        budget = max(1, int(budget))

        seed = None
        for _ in range(200):
            sr, sc = random.randint(0, iH - 1), random.randint(0, iW - 1)
            if blob_occupied[sr, sc]:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < iH and 0 <= nc < iW and blob_occupied[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                seed = (sr, sc)
                break
        if seed is None:
            continue

        blob = {seed}
        frontier = [seed]
        while len(blob) < budget and frontier:
            random.shuffle(frontier)
            new_frontier = []
            for fr, fc in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if (nr, nc) in blob or nr < 0 or nr >= iH or nc < 0 or nc >= iW or blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < iH and 0 <= ac < iW and blob_occupied[ar, ac] and (ar, ac) not in blob:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        continue
                    blob.add((nr, nc))
                    new_frontier.append((nr, nc))
                    if len(blob) >= budget:
                        break
                if len(blob) >= budget:
                    break
            frontier = new_frontier

        for br, bc in blob:
            interior[br, bc] = int(color)
            blob_occupied[br, bc] = True

    fill_ratio = float(np.mean(interior != int(rect_col)))
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None

    rect[1:-1, 1:-1] = interior
    return rect


def _rand_splits(total: int, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, random.randint(min_s, max_s)))
    segs: list[int] = []
    rem = int(total)
    while rem > 0:
        if rem <= eff_max:
            if rem < min_s:
                return None
            segs.append(int(rem))
            rem = 0
        else:
            hi = min(eff_max, rem - min_s)
            if min_s > hi:
                return None
            s = random.randint(min_s, hi)
            segs.append(int(s))
            rem -= s
    return segs


def _extract_pieces(rect: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    RH, RW = rect.shape
    row_segs = _rand_splits(RH)
    col_segs = _rand_splits(RW)
    if row_segs is None or col_segs is None:
        return None

    row_starts: list[int] = []
    s = 0
    for rs in row_segs:
        row_starts.append(int(s))
        s += rs
    col_starts: list[int] = []
    s = 0
    for cs in col_segs:
        col_starts.append(int(s))
        s += cs

    nr, nc = len(row_segs), len(col_segs)
    cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
    n_cells = nr * nc
    if n_cells < 4:
        return None
    target_pieces = random.randint(4, min(12, n_cells))

    def get_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [r for r, _ in cells]
        ci = [c for _, c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return int(h), int(w)

    merges = []
    for i in range(nr):
        for j in range(nc):
            if j + 1 < nc:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    random.shuffle(merges)

    def count_pieces() -> int:
        return len({cell_id[i][j] for i in range(nr) for j in range(nc)})

    for direction, mi, mj in merges:
        if count_pieces() <= target_pieces:
            break
        pid1 = cell_id[mi][mj]
        pid2 = cell_id[mi][mj + 1] if direction == "h" else cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = get_bbox(pid1)
        bb2 = get_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0 = min(bb1[0], bb2[0])
        mc0 = min(bb1[1], bb2[1])
        mr1 = max(bb1[2], bb2[2])
        mc1 = max(bb1[3], bb2[3])
        ph, pw = pixel_size(mr0, mc0, mr1, mc1)
        if ph > 7 or pw > 7:
            continue
        ok = True
        for ii in range(mr0, mr1 + 1):
            for jj in range(mc0, mc1 + 1):
                if cell_id[ii][jj] not in (pid1, pid2):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for ii in range(nr):
            for jj in range(nc):
                if cell_id[ii][jj] == pid2:
                    cell_id[ii][jj] = pid1

    pieces: list[tuple[int, int, int, int]] = []
    seen = set()
    for i in range(nr):
        for j in range(nc):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = get_bbox(pid)
            assert bb is not None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((int(pr), int(pc), int(ph), int(pw)))

    if not (4 <= len(pieces) <= 12):
        return None
    return pieces


def _piece_rotation_options(piece: np.ndarray) -> list[int]:
    out = []
    for k in (1, 2, 3):
        if not np.array_equal(np.rot90(piece, int(k)), piece):
            out.append(int(k))
    return out


def _choose_rotations(piece_arrays: list[np.ndarray], anchor_idx: int) -> list[int] | None:
    n = len(piece_arrays)
    ks = [0] * n
    rotatable = [i for i in range(n) if i != anchor_idx and _piece_rotation_options(piece_arrays[i])]
    odd_non_square = [i for i in rotatable if piece_arrays[i].shape[0] != piece_arrays[i].shape[1]]
    if not odd_non_square:
        return None

    forced = random.choice(odd_non_square)
    odd_opts = [k for k in (1, 3) if not np.array_equal(np.rot90(piece_arrays[forced], int(k)), piece_arrays[forced])]
    if not odd_opts:
        return None
    ks[forced] = random.choice(odd_opts)

    others = [i for i in rotatable if i != forced]
    if others:
        second = random.choice(others)
        ks[second] = random.choice(_piece_rotation_options(piece_arrays[second]))

    for i in rotatable:
        if ks[i] != 0:
            continue
        opts = _piece_rotation_options(piece_arrays[i])
        if opts and random.random() < 0.55:
            ks[i] = random.choice(opts)

    rotated = [i for i in range(n) if ks[i] != 0]
    if not rotated:
        return None
    if not any(piece_arrays[i].shape[0] != piece_arrays[i].shape[1] and ks[i] in (1, 3) for i in rotated):
        return None
    return ks


def _marker_cells(mr: int, mc: int, corner: str) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _gen_attempt() -> np.ndarray | None:
    GH = random.randint(10, 26)
    GW = random.randint(10, 26)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = random.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = random.choice(["TL", "TR", "BL", "BR"])
    min_side, max_side = 5, min(22, min(GH, GW) - 3)
    if max_side < min_side:
        return None

    RH = random.randint(min_side, max_side)
    RW = random.randint(min_side, max_side)
    if random.random() < 0.5 and max_side >= 8:
        RH = random.randint(max(6, min_side), max_side)
        RW = random.randint(max(6, min_side), max_side)

    if corner == "TL":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = 2, GW - RW
    elif corner == "TR":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = RW - 1, GW - 3
    elif corner == "BL":
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = 2, GW - RW
    else:
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = RW - 1, GW - 3
    if mr_min > mr_max or mc_min > mc_max:
        return None
    mr = random.randint(mr_min, mr_max)
    mc = random.randint(mc_min, mc_max)

    rect = _build_rectangle(RH, RW, rect_col, interior_cols)
    if rect is None:
        return None

    reflected = _reflect_rect(rect, corner)
    diff_cells = int(np.sum(rect != reflected))
    if diff_cells < max(6, int(0.08 * RH * RW)):
        return None

    pieces = _extract_pieces(rect)
    if pieces is None:
        return None

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
    for (pr, pc, ph, pw), piece in zip(pieces, piece_arrays):
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        fill = float(np.mean(piece != int(rect_col)))
        if fill < 0.10 or fill > 0.60:
            return None
        if not any(np.any(edge != int(rect_col)) for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]):
            return None
        if pr == 0 and not np.all(piece[0, :] == int(rect_col)):
            return None
        if pr + ph == RH and not np.all(piece[-1, :] == int(rect_col)):
            return None
        if pc == 0 and not np.all(piece[:, 0] == int(rect_col)):
            return None
        if pc + pw == RW and not np.all(piece[:, -1] == int(rect_col)):
            return None

    if corner == "TL":
        anchor_r, anchor_c = 0, 0
    elif corner == "TR":
        anchor_r, anchor_c = 0, RW - 1
    elif corner == "BL":
        anchor_r, anchor_c = RH - 1, 0
    else:
        anchor_r, anchor_c = RH - 1, RW - 1

    anchor_idx = None
    for i, (pr, pc, ph, pw) in enumerate(pieces):
        if pr <= anchor_r < pr + ph and pc <= anchor_c < pc + pw:
            anchor_idx = i
            break
    if anchor_idx is None:
        return None

    apr, apc, aph, apw = pieces[anchor_idx]
    if corner == "TL" and not (apr == 0 and apc == 0):
        return None
    if corner == "TR" and not (apr == 0 and apc + apw == RW):
        return None
    if corner == "BL" and not (apr + aph == RH and apc == 0):
        return None
    if corner == "BR" and not (apr + aph == RH and apc + apw == RW):
        return None

    rotations = _choose_rotations(piece_arrays, anchor_idx)
    if rotations is None:
        return None
    rotations[anchor_idx] = 0
    rotated_count = sum(1 for k in rotations if k != 0)
    if rotated_count < 1:
        return None

    marker_cells = _marker_cells(mr, mc, corner)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = int(marker_col)

    anchor_piece = piece_arrays[anchor_idx]
    ar0, ac0 = _top_left_from_corner(mr, mc, int(anchor_piece.shape[0]), int(anchor_piece.shape[1]), corner)
    if ar0 < 0 or ac0 < 0 or ar0 + anchor_piece.shape[0] > GH or ac0 + anchor_piece.shape[1] > GW:
        return None
    if np.any(canvas[ar0 : ar0 + anchor_piece.shape[0], ac0 : ac0 + anchor_piece.shape[1]] != int(bg)):
        return None
    paste(canvas, anchor_piece, ar0, ac0)

    order = [i for i in range(len(piece_arrays)) if i != anchor_idx]
    random.shuffle(order)
    for idx in order:
        pa = np.rot90(piece_arrays[idx], int(rotations[idx]))
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, pa, r0, c0)

    tr, tc = _top_left_from_corner(mr, mc, RH, RW, corner)
    expected = np.full((GH, GW), int(bg), dtype=int)
    expected[tr : tr + RH, tc : tc + RW] = reflected

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        no_rot_combo = _solve(canvas, allow_rotation=False, apply_reflection=True)
    except Exception:
        no_rot_combo = None
    if no_rot_combo is not None and np.array_equal(no_rot_combo, expected):
        return None

    try:
        rot_no_reflect = _solve(canvas, allow_rotation=True, apply_reflection=False)
    except Exception:
        rot_no_reflect = None
    if rot_no_reflect is not None and np.array_equal(rot_no_reflect, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        canvas = _gen_attempt()
        if canvas is not None:
            return canvas
    raise RuntimeError("generate_input: failed after 400 attempts")
