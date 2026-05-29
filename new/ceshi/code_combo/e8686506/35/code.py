# Combo variant of e8686506 (index 35)
# combo_spec: D4/1+D8/0+D12/0
# Name: 双轴对称+更多碎片+允许旋转

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# yaml:
#   concrete_behavior:
#     output:
#       - The solved board is a dense 7x5 portrait rectangle with exact left-right and up-down
#         mirror symmetry. Every visible color placement therefore belongs to a D4 reflection
#         orbit unless it lies on the center column.
#       - A recognizable frame still exists, but in this combo it is deliberately split into two
#         tall symmetric halves that stay grouped in the input. The frame fixes the rectangle's
#         bounding box while leaving ten other cells for small scattered fragments.
#       - Outside the frame there are exactly eleven non-frame fragments: four corner singletons,
#         four side dominoes, two center-axis cap dominoes, and one center-axis triomino. Seven
#         of those fragments have a visibly different 90-degree rotation.
#     input:
#       - The frame halves are pasted together in their output-relative positions inside one clean
#         reserved bbox, exactly so the rectangle outline is still observable at a glance.
#       - Every non-frame fragment is scattered elsewhere on the canvas with Chebyshev padding,
#         and several rotatable fragments are forcibly quarter-turned before scattering.
#     trace_example:
#       - Original task: a side fragment observed as a vertical domino had a fixed pose, and under
#         LR-only symmetry the solver only had to choose between a few left/right-compatible slots.
#       - This combo: the same observed domino may arrive horizontal, and even after rotating it
#         back, a top placement is only legal if the matching bottom orbit can carry the same color.
#       - Difference: D12 expands each fragment into up to four poses, D8 increases the number of
#         fragments to eleven small pieces, and D4 removes the resulting top-vs-bottom ambiguity by
#         requiring every placement to agree with its full four-way reflection orbit.
#   complexity_calibration:
#     - The geometry is kept at a fixed 7x5 rectangle so the task still looks like the source task
#       instead of turning into a free-form tiling puzzle.
#     - Piece count is raised through many small fragments rather than through many color families;
#       this keeps the board readable while still increasing combinatorial search pressure.
#     - Only two rotatable shape families are used repeatedly (domino and triomino), so humans can
#       infer the rotation rule from a few examples instead of memorizing many unrelated shapes.
#   implicit_assumption_review:
#     - Original grouping by exact observed coords is invalid once fragments may rotate, so the
#       solver must group types by rotation-canonical shape and try all unique quarter turns.
#     - Original LR-only mirror checks are insufficient here; candidate legality must be tested
#       against the whole D4 orbit, otherwise vertically inconsistent packings would pass.
#     - Original bar/line anchor heuristics are unsafe under rotation because a scattered bar no
#       longer reveals whether it belongs on an axis or off it. This combo keeps only the corner
#       singleton heuristic plus the locked frame cells.
#     - Every branch in the new search assumes the input contains exactly one dual-symmetric frame
#       group whose union bbox is the output rectangle. The generator enforces this by reserving one
#       bbox for the frame halves and scattering all other pieces strictly outside it.
#   placement_legality:
#     - The output template is written cell-by-cell, so all pieces are in-bounds and cover the
#         rectangle exactly once with no overlap.
#     - During scattering, each rotated patch is placed by random_free_location() with transparent
#       background and 8-neighborhood padding, so fragments never overlap and never touch.
#     - The frame bbox is blocked before scattering and restored afterwards, which guarantees that
#       scattered fragments cannot leak into the reserved frame area.
#   observability:
#     - From a few examples a human can see three independent signals: the output is mirrored both
#       horizontally and vertically, there are many more small fragments than in the source task,
#       and some input dominoes/triominoes appear sideways relative to their solved positions.
#     - None of those signals depends on raw color ids. The structure is visible from shape, count,
#       and symmetry alone.
#   cleanliness:
#     - The board has three clean layers only: frame, corners, and axis/side fillers. They do not
#       overlap, and the narrow five-column layout keeps the assembled pattern easy to parse.
#   inferability_of_change:
#     - Color usage is intentionally limited to five output colors. Repeated fragment families share
#       colors across symmetric positions, so the color pattern reinforces the geometry instead of
#       introducing a separate hidden mapping rule.


_LAST_GEN_INFO: dict | None = None


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


def _coords_to_bitmap(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)
    bmp = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        bmp[int(r), int(c)] = True
    return bmp


def _bitmap_to_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _unique_rotations(coords: tuple[tuple[int, int], ...]) -> list[dict]:
    bmp = _coords_to_bitmap(coords)
    out = []
    seen = set()
    for k in range(4):
        rb = np.rot90(bmp, k)
        rc = _bitmap_to_coords(rb)
        if rc in seen:
            continue
        seen.add(rc)
        out.append(
            {
                "k": int(k),
                "coords": rc,
                "h": int(rb.shape[0]),
                "w": int(rb.shape[1]),
            }
        )
    return out


def _canonical_rot_key(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rots = _unique_rotations(coords)
    return min((int(v["h"]), int(v["w"]), v["coords"]) for v in rots)


def _orbit_coords(H: int, W: int, r: int, c: int) -> list[tuple[int, int]]:
    coords = {
        (int(r), int(c)),
        (H - 1 - int(r), int(c)),
        (int(r), W - 1 - int(c)),
        (H - 1 - int(r), W - 1 - int(c)),
    }
    return sorted(coords)


def _is_dual_symmetric_bitmap(bmp: np.ndarray) -> bool:
    return np.array_equal(bmp, bmp[:, ::-1]) and np.array_equal(bmp, bmp[::-1, :])


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    Rotation makes pose-derived anchors unsafe, so the combo keeps only one robust cue:
      - if exactly one color appears as four singleton components, they are the corners.
    """
    req = np.full((H, W), -1, dtype=int)

    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    if corner_colors:
        col = int(corner_colors[0])
        req[0, 0] = col
        req[0, W - 1] = col
        req[H - 1, 0] = col
        req[H - 1, W - 1] = col
    return req


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def is_d4(parts: list[tuple[int, np.ndarray]]) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((H, W), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return _is_dual_symmetric_bitmap(bmp)

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and is_d4([(c1, p1)]):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and is_d4([(c1, p1), (c2, p2)]):
                frame_parts = [(c1, p1), (c2, p2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 dual-symmetric frame parts whose union bbox equals output dims"
    return frame_parts


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    prefilled: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = prefilled.copy()
    counts = [int(t["cnt"]) for t in types]
    fail_cache: set[tuple[tuple[int, ...], bytes]] = set()

    def candidate_cells(ti: int, variant: dict, r0: int, c0: int) -> list[tuple[int, int]] | None:
        col = int(types[ti]["col"])
        coords = variant["coords"]
        cells = [(r0 + dr, c0 + dc) for dr, dc in coords]
        cellset = set(cells)
        is_singleton = len(coords) == 1

        for r, c in cells:
            if board[r, c] != -1:
                return None
            if req[r, c] != -1 and int(req[r, c]) != col:
                return None
            for rr, cc in _orbit_coords(H, W, r, c):
                if req[rr, cc] != -1 and int(req[rr, cc]) != col:
                    return None
                v = board[rr, cc]
                if v != -1 and int(v) != col:
                    return None
            if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return None

        if disallow_touch and col in disallow_touch:
            for r, c in cells:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col and (rr, cc) not in cellset:
                        return None
        return cells

    def placements_for(ti: int) -> list[tuple[dict, int, int, list[tuple[int, int]]]]:
        t = types[ti]
        out = []
        for variant in t["rots"]:
            for r0 in range(H - int(variant["h"]) + 1):
                for c0 in range(W - int(variant["w"]) + 1):
                    cells = candidate_cells(ti, variant, r0, c0)
                    if cells is not None:
                        out.append((variant, r0, c0, cells))
        out.sort(key=lambda x: (x[1], x[2], x[0]["h"], x[0]["w"], x[0]["coords"]))
        return out

    def place(col: int, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        changed = []
        for r, c in cells:
            board[r, c] = col
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    def can_batch_force(cands: list[tuple[dict, int, int, list[tuple[int, int]]]], need: int) -> bool:
        if len(cands) != need:
            return False
        used: set[tuple[int, int]] = set()
        for _variant, _r0, _c0, cells in cands:
            cellset = set(cells)
            if used & cellset:
                return False
            used |= cellset
        return True

    def search() -> bool:
        forced: list[tuple[int, list[tuple[int, int]]]] = []
        while True:
            progress = False
            for ti in range(len(types)):
                if counts[ti] <= 0:
                    continue
                cands = placements_for(ti)
                if len(cands) < counts[ti]:
                    for fti, changed in reversed(forced):
                        unplace(changed)
                        counts[fti] += 1
                    return False
                if counts[ti] == 1 and len(cands) == 1:
                    counts[ti] -= 1
                    changed = place(int(types[ti]["col"]), cands[0][3])
                    forced.append((ti, changed))
                    progress = True
                    break
                if can_batch_force(cands, counts[ti]):
                    for _variant, _r0, _c0, cells in cands:
                        counts[ti] -= 1
                        changed = place(int(types[ti]["col"]), cells)
                        forced.append((ti, changed))
                    progress = True
                    break
            if not progress:
                break

        key = (tuple(counts), board.tobytes())
        if key in fail_cache:
            for ti, changed in reversed(forced):
                unplace(changed)
                counts[ti] += 1
            return False

        if all(c == 0 for c in counts):
            ok = (not np.any(board == -1)) and _is_dual_symmetric_bitmap(board)
            if not ok:
                fail_cache.add(key)
                for ti, changed in reversed(forced):
                    unplace(changed)
                    counts[ti] += 1
            return ok

        best_ti = -1
        best_cands = None
        best_score = None
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            cands = placements_for(ti)
            if len(cands) < counts[ti]:
                fail_cache.add(key)
                for fti, changed in reversed(forced):
                    unplace(changed)
                    counts[fti] += 1
                return False
            score = (len(cands), -int(types[ti]["area"]), counts[ti], int(types[ti]["col"]))
            if best_score is None or score < best_score:
                best_score = score
                best_ti = ti
                best_cands = cands
        assert best_ti >= 0 and best_cands is not None

        counts[best_ti] -= 1
        col = int(types[best_ti]["col"])
        for _variant, _r0, _c0, cells in best_cands:
            changed = place(col, cells)
            if search():
                return True
            unplace(changed)
        counts[best_ti] += 1

        fail_cache.add(key)
        for ti, changed in reversed(forced):
            unplace(changed)
            counts[ti] += 1
        return False

    assert search(), "assembly failed under dual-symmetric rotation-aware search"
    assert _is_dual_symmetric_bitmap(board), "assembled board must satisfy both mirror axes"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into a dense rectangle that is mirror-symmetric across both axes.

    Input structure:
    - The dominant color is background; every other monochrome 4-connected component is a puzzle piece.
    - One 1-2-part frame group already sits in-place as a dual-symmetric outline of the target rectangle.
    - The remaining small fragments are scattered elsewhere, and any non-frame fragment may have been quarter-turned.

    Rule:
    - Infer the portrait output dimensions from the total non-background area.
    - Lock the frame group into its output-relative cells, keep only rotation-safe anchor inference from corner singletons,
      then place all remaining pieces while trying every unique 0/90/180/270 orientation.
    - The completed board must satisfy both left-right and up-down symmetry, and separate pieces of the same color may not
      become 4-connected after assembly.

    Key insight:
    - The search is only manageable if rotation and symmetry are fused: each candidate pose must be checked against the
      whole four-way reflection orbit, so the stronger D4 constraint cancels much of the extra branching introduced by the
      larger piece count and the new quarter-turn freedom.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    total_area = int((g != bg).sum())
    assert total_area > 0, "grid must contain non-background pieces"

    dims = [(h, w) for h, w in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(g, bg, H, W)

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    color_piece_count: dict[int, int] = {}
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[int(col)] = comps
        color_piece_count[int(col)] = len(comps)

    req = _build_requirements(H, W, comps_by_color)
    board0 = np.full((H, W), -1, dtype=int)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    frame_input_cells = {(int(r), int(c)) for r, c in all_frame_pts}
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            assert 0 <= or_ < H and 0 <= oc < W, "frame output coords must stay in bounds"
            if req[or_, oc] != -1:
                assert int(req[or_, oc]) == int(col), "frame constraints must not conflict"
            req[or_, oc] = int(col)
            board0[or_, oc] = int(col)
    assert _is_dual_symmetric_bitmap(board0 != -1), "locked frame must already be dual-symmetric as a mask"
    for r in range(H):
        for c in range(W):
            if board0[r, c] == -1:
                continue
            for rr, cc in _orbit_coords(H, W, r, c):
                assert board0[rr, cc] in (-1, board0[r, c]), "frame colors must respect both symmetry axes"

    type_counter: Counter = Counter()
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in comps_by_color[int(col)]:
            if all((int(r), int(c)) in frame_input_cells for r, c in pts):
                continue
            coords = _piece_sig(pts)[2]
            type_counter[(int(col), _canonical_rot_key(coords))] += 1

    types = []
    for (col, canon_key), cnt in sorted(
        type_counter.items(),
        key=lambda kv: (-len(kv[0][1][2]), kv[0][0], kv[0][1]),
    ):
        _, _, canon_coords = canon_key
        types.append(
            {
                "col": int(col),
                "rots": _unique_rotations(canon_coords),
                "cnt": int(cnt),
                "area": int(len(canon_coords)),
            }
        )

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, board0, disallow_touch)
    return sol.astype(int)


# --------------- generator ---------------

_H = 7
_W = 5
_N_OUTPUT_COLORS = 5

_NON_FRAME_PIECE_MIN = 8
_NON_FRAME_PIECE_MAX = 12
_MAX_PIECE_AREA = 3

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_MIN_ROT90_VISIBLE = 2
_GEN_MAX_RETRIES = 400


def _frame_cells() -> set[tuple[int, int]]:
    left_rows = [1, 0, 0, 2, 0, 0, 1]
    cells = set()
    for r, kind in enumerate(left_rows):
        cols = [0] if kind == 0 else ([1] if kind == 1 else [0, 1])
        for c in cols:
            cells.add((r, c))
            cells.add((r, _W - 1 - c))
    return cells


def _build_output() -> tuple[np.ndarray, set[int]]:
    corner_col, frame_col, side_col, cap_col, center_col = range(_N_OUTPUT_COLORS)
    grid = np.full((_H, _W), -1, dtype=int)

    for r, c in _frame_cells():
        grid[r, c] = frame_col

    for r, c in ((0, 0), (0, _W - 1), (_H - 1, 0), (_H - 1, _W - 1)):
        grid[r, c] = corner_col

    for r, c in ((1, 1), (2, 1), (1, 3), (2, 3), (4, 1), (5, 1), (4, 3), (5, 3)):
        grid[r, c] = side_col

    for r, c in ((0, 2), (1, 2), (5, 2), (6, 2)):
        grid[r, c] = cap_col

    for r, c in ((2, 2), (3, 2), (4, 2)):
        grid[r, c] = center_col

    assert not np.any(grid < 0), "output template must be dense"
    assert _is_dual_symmetric_bitmap(grid), "output template must satisfy both mirror axes"
    return grid, {frame_col}


def _make_output() -> tuple[np.ndarray, int, set[int]]:
    out, abstract_fcs = _build_output()
    bg = random.randint(0, 9)
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    cmap = {int(a): available[i] for i, a in enumerate(sorted(np.unique(out)))}
    real_fcs = {cmap[a] for a in abstract_fcs}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg, real_fcs


def _shape_key_from_patch(patch: np.ndarray, bg: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = np.argwhere(patch != bg)
    assert coords.size > 0
    return _piece_sig(coords)


def _extract_non_frame_patches(out: np.ndarray, bg: int, fcs: set[int]) -> list[dict]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            base_key = _shape_key_from_patch(patch, bg)
            rot_keys = [_shape_key_from_patch(np.rot90(patch, k), bg) for k in range(4)]
            patches.append(
                {
                    "patch": patch,
                    "base_key": base_key,
                    "quarter_turn_visible": rot_keys[1] != base_key or rot_keys[3] != base_key,
                    "unique_rotations": len(set(rot_keys)),
                }
            )
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p["patch"] != bg).sum(), random.random()))
    return patches


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> tuple[np.ndarray, dict]:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for reserved bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    patches = _extract_non_frame_patches(out, bg, fcs)
    qturn_candidates = [i for i, p in enumerate(patches) if p["quarter_turn_visible"]]
    if len(qturn_candidates) < _MIN_ROT90_VISIBLE:
        raise ValueError("not enough quarter-turn-visible fragments")

    force_n = random.randint(_MIN_ROT90_VISIBLE, min(4, len(qturn_candidates)))
    forced = set(random.sample(qturn_candidates, force_n))
    rot90_visible_count = 0
    rotated_nontrivial_count = 0
    applied_rotations = []

    for i, info in enumerate(patches):
        if i in forced:
            k = random.choice([1, 3])
        else:
            k = random.randrange(4)
        rotated = np.rot90(info["patch"], k)
        rotated_key = _shape_key_from_patch(rotated, bg)
        if rotated_key != info["base_key"]:
            rotated_nontrivial_count += 1
        if k in (1, 3) and rotated_key != info["base_key"]:
            rot90_visible_count += 1
        r, c = random_free_location(
            canvas,
            rotated,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, rotated, r, c, transparent=bg)
        applied_rotations.append(int(k))

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    frame_component_count = sum(len(components_mask(out == col, connectivity=8)) for col in fcs)
    meta = {
        "non_frame_piece_count": int(len(patches)),
        "quarter_turn_candidate_count": int(len(qturn_candidates)),
        "rotatable_piece_count": int(sum(p["unique_rotations"] > 1 for p in patches)),
        "rot90_visible_count": int(rot90_visible_count),
        "rotated_nontrivial_count": int(rotated_nontrivial_count),
        "frame_component_count": int(frame_component_count),
        "applied_rotations": tuple(applied_rotations),
    }
    if meta["rot90_visible_count"] < _MIN_ROT90_VISIBLE:
        raise ValueError("rotation visibility requirement not met")
    return canvas, meta


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> tuple[np.ndarray | None, dict | None]:
    out, bg, fcs = _make_output()
    if not _is_dual_symmetric_bitmap(out):
        return None, None

    patches = _extract_non_frame_patches(out, bg, fcs)
    if not (_NON_FRAME_PIECE_MIN <= len(patches) <= _NON_FRAME_PIECE_MAX):
        return None, None
    if any(len(np.argwhere(p["patch"] != bg)) > _MAX_PIECE_AREA for p in patches):
        return None, None

    try:
        inp, meta = _scatter(out, bg, fcs)
        got = main(inp)
    except Exception:
        return None, None

    if np.array_equal(got, out):
        piece_meta = {
            "output_total_piece_count": int(_piece_count(out)),
            "output_non_frame_piece_count": int(len(patches)),
            "dual_axis_verified": True,
        }
        piece_meta.update(meta)
        return inp, piece_meta
    return None, None


def generate_input() -> np.ndarray:
    global _LAST_GEN_INFO
    for _ in range(_GEN_MAX_RETRIES):
        grid, meta = _gen_attempt()
        if grid is not None:
            _LAST_GEN_INFO = dict(meta or {})
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
