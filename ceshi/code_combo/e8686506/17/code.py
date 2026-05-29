# Combo variant of e8686506 (index 17)
# combo_spec: D4/2+D12/0
# Name: 90度旋转对称+允许旋转

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for combo D4/2 + D12/0
#
# a) Concrete behavior and trace:
#    The original task rebuilt a portrait rectangle whose colors only had to
#    match under left-right reflection, and each scattered piece kept the exact
#    pose seen in the input. This combo changes both parts together:
#    - the target is now a 7x7 square with exact 90-degree rotational symmetry;
#    - scattered non-frame pieces may be quarter-turned before assembly.
#    Small trace:
#    - Old rule: a horizontal domino observed in the input could only fill a
#      horizontal 2-cell slot, and a red cell at (0,1) only constrained (0,W-2).
#    - New rule: the same observed domino may need a 90-degree turn to fit a
#      vertical slot, and a placed cell at (r,c) participates in a full
#      4-cycle ((r,c) -> (c,n-1-r) -> ...).
#    - Difference: destination and orientation are inferred jointly, and the
#      four quarter-turn-related positions must all agree with one template.
#
# b) Complexity calibration:
#    A fully generic rotational jigsaw with free rotations would be too open.
#    To keep the task learnable and visually obvious, the generator uses one of
#    two pinwheel frame templates. The frame remains grouped in-place inside its
#    true square bounding box, while non-frame roles are limited to repeated
#    singleton / domino / tetromino families. This yields repeated evidence for
#    both the rotational symmetry and the rotation-allowed piece rule without
#    exploding the search space.
#
# c) Hidden assumptions audited:
#    - The old closest-portrait-factor inference is invalid; 90-degree symmetry
#      requires a square, so total area must be a perfect square.
#    - The old frame finder searched for LR-symmetric unions; here it must look
#      for quarter-turn-symmetric 8-connected frame parts.
#    - Grouping pieces by exact coordinates is unsafe once pieces may rotate;
#      they must be grouped by rotation-canonical shape.
#    - Old input-orientation heuristics such as "horizontal bar means top edge"
#      no longer hold, because an input bar may need to become vertical.
#    - Mirror-only placement checks are too weak; the output constraints must be
#      consistent with a full 90-degree rotational template.
#
# d) Placement legality:
#    The frame bbox is blocked before scattering other pieces, so the anchored
#    pinwheel outline never overlaps the scattered fragments. Every scattered
#    patch is placed by `random_free_location(..., padding=1, connectivity=8)`,
#    which enforces in-bounds, non-overlap, and a visible gap between pieces.
#
# e) Visual observability:
#    The output is rejected if it accidentally becomes LR-symmetric, so the new
#    symmetry is readable by eye rather than collapsing back to the old task.
#    At least two quarter-turn-distinguishable pieces are rotated by 90/270
#    degrees in the input, making the "pieces may rotate" rule directly visible.
#
# f) Visual cleanliness:
#    Only the frame occupies the locked square bbox in the input. All other
#    fragments are scattered outside that box. In the output, each color role
#    occupies a clean repeated family of components, so the template remains
#    easy to parse despite the added rotation freedom.
#
# g) Inferability of the change:
#    Colors are globally permuted but roles are structural: one frame color, a
#    color with four singletons, a color with four dominoes, and a color with
#    four tetrominoes. Because each role repeats four times in quarter-turn
#    positions, a human can infer both the target rotational symmetry and the
#    need to rotate fragments to fit their matching slots.
# ---------------------------------------------------------------------------


_LAST_GEN_INFO: dict | None = None

_OUTPUT_SIZE = 7
_N_COLORS = 4
_PIECE_COUNT_REQUIRED = 17
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 400
_MIN_ROT90_VISIBLE = 2
_MIN_QUARTER_TURN_CANDIDATES = 2

_TEMPLATES = (
    {
        "frame_seed": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 3), (3, 3)],
        "roles": {
            "singleton": [(0, 3)],
            "domino": [(0, 4), (0, 5)],
            "tetromino": [(1, 1), (1, 3), (1, 4), (2, 2)],
        },
    },
    {
        "frame_seed": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 3), (3, 3)],
        "roles": {
            "singleton": [(0, 1)],
            "domino": [(0, 2), (0, 3)],
            "tetromino": [(1, 1), (1, 2), (1, 3), (2, 2)],
        },
    },
)


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _rot90_cell(r: int, c: int, n: int) -> tuple[int, int]:
    return c, n - 1 - r


def _orbit_cells(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    rr, cc = int(r), int(c)
    for _ in range(4):
        cell = (rr, cc)
        if cell not in out:
            out.append(cell)
        rr, cc = _rot90_cell(rr, cc, n)
    return tuple(out)


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
        out.append({"k": int(k), "coords": rc, "h": int(rb.shape[0]), "w": int(rb.shape[1])})
    return out


def _canonical_rot_key(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rots = _unique_rotations(coords)
    return min((int(v["h"]), int(v["w"]), v["coords"]) for v in rots)


def _paint_orbit(board: np.ndarray, cells: list[tuple[int, int]], col: int) -> None:
    n = int(board.shape[0])
    for r, c in cells:
        for rr, cc in _orbit_cells(r, c, n):
            board[rr, cc] = col


def _template_frame_bitmap(tmpl: dict, n: int) -> np.ndarray:
    bmp = np.zeros((n, n), dtype=bool)
    for r, c in tmpl["frame_seed"]:
        for rr, cc in _orbit_cells(r, c, n):
            bmp[rr, cc] = True
    return bmp


def _match_template(frame_bmp: np.ndarray) -> dict:
    for tmpl in _TEMPLATES:
        if np.array_equal(frame_bmp, _template_frame_bitmap(tmpl, frame_bmp.shape[0])):
            return tmpl
    raise AssertionError("detected frame does not match any supported rotational template")


def _is_rot4_symmetric(parts: list[tuple[int, np.ndarray]], n: int) -> bool:
    all_pts = np.concatenate([pts for _, pts in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((n, n), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, np.rot90(bmp))


def _normalized_frame_bitmap(parts: list[tuple[int, np.ndarray]], n: int) -> np.ndarray:
    all_pts = np.concatenate([pts for _, pts in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((n, n), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return bmp


def _find_frame_parts(g: np.ndarray, bg: int, n: int) -> list[tuple[int, np.ndarray]]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == n and c1b - c0 + 1 == n and _is_rot4_symmetric([(c1, p1)], n):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == n and uc1 - uc0 + 1 == n:
                if _is_rot4_symmetric([(c1, p1), (c2, p2)], n):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 rotationally symmetric frame parts whose bbox is the output square"
    return frame_parts


def _build_requirements(
    n: int,
    frame_parts: list[tuple[int, np.ndarray]],
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> tuple[np.ndarray, np.ndarray]:
    req = np.full((n, n), -1, dtype=int)
    board0 = np.full((n, n), -1, dtype=int)

    frame_bmp = _normalized_frame_bitmap(frame_parts, n)
    tmpl = _match_template(frame_bmp)
    frame_colors = {int(col) for col, _ in frame_parts}

    role_colors: dict[str, int] = {}
    for col, comps in comps_by_color.items():
        if col in frame_colors:
            continue
        sizes = sorted(len(p) for p in comps)
        if sizes == [1, 1, 1, 1]:
            assert "singleton" not in role_colors, "singleton role color must be unique"
            role_colors["singleton"] = int(col)
            continue
        if sizes == [2, 2, 2, 2]:
            sigs = [_piece_sig(pts) for pts in comps]
            ok = all(
                (h == 1 and w == 2 and len(coords) == 2) or
                (h == 2 and w == 1 and len(coords) == 2)
                for h, w, coords in sigs
            )
            if ok:
                assert "domino" not in role_colors, "domino role color must be unique"
                role_colors["domino"] = int(col)
                continue
        if sizes == [4, 4, 4, 4]:
            assert "tetromino" not in role_colors, "tetromino role color must be unique"
            role_colors["tetromino"] = int(col)
            continue
    assert set(role_colors) == {"singleton", "domino", "tetromino"}, "could not classify all non-frame role colors"

    for role, color in role_colors.items():
        for r, c in tmpl["roles"][role]:
            for rr, cc in _orbit_cells(r, c, n):
                req[rr, cc] = int(color)

    all_frame_pts = np.concatenate([pts for _, pts in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if req[or_, oc] != -1:
                assert int(req[or_, oc]) == int(col), "frame and role constraints disagree"
            req[or_, oc] = int(col)
            board0[or_, oc] = int(col)

    assert not np.any(req < 0), "template requirements must cover the entire square"
    assert np.array_equal(req, np.rot90(req)), "requirements must be 90-degree rotationally symmetric"
    return req, board0


def _assemble(
    types: list[dict],
    n: int,
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
        local_rot_steps: set[int] = set()

        for r, c in cells:
            if board[r, c] != -1:
                return None
            if int(req[r, c]) != col:
                return None

            rr, cc = int(r), int(c)
            orbit = []
            for step in range(4):
                orbit.append((rr, cc, step))
                rr, cc = _rot90_cell(rr, cc, n)
            orbit_cols = {int(req[or_, oc]) for or_, oc, _ in orbit}
            if orbit_cols != {col}:
                return None
            for or_, oc, step in orbit:
                v = int(board[or_, oc])
                if v != -1 and v != col:
                    return None
            local_rot_steps.add(min(step for or_, oc, step in orbit if (or_, oc) in cellset))

        if disallow_touch and col in disallow_touch:
            for r, c in cells:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < n and 0 <= cc < n and board[rr, cc] == col and (rr, cc) not in cellset:
                        return None

        if len(local_rot_steps) > 1:
            return None
        return cells

    def placements_for(ti: int) -> list[tuple[dict, int, int, list[tuple[int, int]]]]:
        out = []
        for variant in types[ti]["rots"]:
            for r0 in range(n - int(variant["h"]) + 1):
                for c0 in range(n - int(variant["w"]) + 1):
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
                if len(cands) == counts[ti]:
                    used: set[tuple[int, int]] = set()
                    disjoint = True
                    for _, _, _, cells in cands:
                        for cell in cells:
                            if cell in used:
                                disjoint = False
                                break
                            used.add(cell)
                        if not disjoint:
                            break
                    if disjoint:
                        for _, _, _, cells in cands:
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
            ok = (not np.any(board < 0)) and np.array_equal(board, req) and np.array_equal(board, np.rot90(board))
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

    assert search(), "assembly failed under rotational symmetry plus rotation-aware placement"
    assert np.array_equal(board, req), "assembled board must match the rotational template"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered colored fragments into a dense square with 90-degree rotational symmetry, allowing quarter turns.

    Input structure:
    - The background is the dominant color; every other color forms detached monochrome fragments.
    - One 8-connected pinwheel frame stays grouped inside its true square bounding box and already reveals the output size.
    - The remaining fragments appear outside that box, and some of them may be rotated by 0/90/180/270 degrees relative to their output pose.

    Rule:
    - Infer the output side length from the total non-background area and locate the rotationally symmetric frame group.
    - Classify the remaining colors by their repeated component family: four singletons, four dominoes, and four tetrominoes.
    - Place those fragments back into the square, trying quarter-turn rotations for each piece, so the filled board matches a 90-degree rotational template exactly.

    Key insight:
    - The frame fixes which pinwheel template is present; after that, the four quarter-turn-related slots of each role are known.
    - Because each scattered domino or tetromino may need rotation before it fits its assigned slot, the symmetry rule and the rotation rule prune each other and make the final square unique.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    color_piece_count: dict[int, int] = {}
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[int(col)] = comps
        color_piece_count[int(col)] = len(comps)

    total_area = int((g != bg).sum())
    assert total_area > 0, "grid must contain non-background pieces"
    n = int(round(total_area ** 0.5))
    assert n * n == total_area, "90-degree rotational variant requires square output area"
    assert n == _OUTPUT_SIZE, "this combo generator and solver use a fixed 7x7 rotational template"

    frame_parts = _find_frame_parts(g, bg, n)
    req, board0 = _build_requirements(n, frame_parts, comps_by_color)

    all_frame_pts = np.concatenate([pts for _, pts in frame_parts])
    frame_input_cells = {(int(r), int(c)) for r, c in all_frame_pts}

    type_counter: Counter = Counter()
    for col, comps in comps_by_color.items():
        for pts in comps:
            if all((int(r), int(c)) in frame_input_cells for r, c in pts):
                continue
            coords = _piece_sig(pts)[2]
            type_counter[(int(col), _canonical_rot_key(coords))] += 1

    types = []
    for (col, canon_key), cnt in sorted(type_counter.items(), key=lambda kv: (-len(kv[0][1][2]), kv[0][0], kv[0][1])):
        _, _, canon_coords = canon_key
        types.append(
            {
                "col": int(col),
                "rots": _unique_rotations(canon_coords),
                "cnt": int(cnt),
                "area": int(len(canon_coords)),
            }
        )

    disallow_touch = {col for col, cnt in color_piece_count.items() if cnt > 1}
    sol = _assemble(types, n, req, board0, disallow_touch)
    return sol.astype(int)


def _make_output() -> tuple[np.ndarray, int, set[int], dict]:
    tmpl = random.choice(_TEMPLATES)
    n = _OUTPUT_SIZE

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    role_cols = palette[:_N_COLORS]

    board = np.full((n, n), -1, dtype=int)
    _paint_orbit(board, list(tmpl["frame_seed"]), int(role_cols[0]))
    _paint_orbit(board, list(tmpl["roles"]["singleton"]), int(role_cols[1]))
    _paint_orbit(board, list(tmpl["roles"]["domino"]), int(role_cols[2]))
    _paint_orbit(board, list(tmpl["roles"]["tetromino"]), int(role_cols[3]))

    assert not np.any(board < 0), "template must cover the full output square"
    assert np.array_equal(board, np.rot90(board)), "generator output must be 90-degree rotationally symmetric"
    assert not np.array_equal(board, board[:, ::-1]), "generator must not degenerate to LR symmetry"
    meta = {"template_id": int(_TEMPLATES.index(tmpl)), "output_side": int(n)}
    return board, int(bg), {int(role_cols[0])}, meta


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
                    "unique_rotations": len(set(rot_keys)),
                    "quarter_turn_visible": rot_keys[1] != base_key or rot_keys[3] != base_key,
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
        raise ValueError("canvas too small for frame bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    patches = _extract_non_frame_patches(out, bg, fcs)
    qturn_candidates = [i for i, p in enumerate(patches) if p["quarter_turn_visible"]]
    if len(qturn_candidates) < _MIN_QUARTER_TURN_CANDIDATES:
        raise ValueError("not enough quarter-turn-distinguishable patches")

    force_n = random.randint(_MIN_ROT90_VISIBLE, len(qturn_candidates))
    forced = set(random.sample(qturn_candidates, force_n))
    rotation_hist = Counter()
    rot90_visible_count = 0
    rotated_nontrivial_count = 0

    for i, info in enumerate(patches):
        patch = info["patch"]
        if i in forced:
            k = random.choice([1, 3])
        else:
            k = random.randrange(4)
        rotated = np.rot90(patch, k)
        rotated_key = _shape_key_from_patch(rotated, bg)
        if rotated_key != info["base_key"]:
            rotated_nontrivial_count += 1
        if k in (1, 3) and rotated_key != info["base_key"]:
            rot90_visible_count += 1
        rotation_hist[int(k)] += 1
        r, c = random_free_location(
            canvas,
            rotated,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, rotated, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    meta = {
        "rot90_visible_count": int(rot90_visible_count),
        "rotated_nontrivial_count": int(rotated_nontrivial_count),
        "quarter_turn_candidate_count": int(len(qturn_candidates)),
        "rotation_hist": {int(k): int(v) for k, v in rotation_hist.items()},
        "non_frame_piece_count": int(len(patches)),
        "frame_offset": (int(off_r), int(off_c)),
    }
    if meta["rot90_visible_count"] < _MIN_ROT90_VISIBLE:
        raise ValueError("rotation visibility requirement not met")
    return canvas, meta


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _component_size_multiset(out: np.ndarray) -> list[int]:
    sizes = []
    for col in np.unique(out):
        for pts in components_mask(out == col, connectivity=4):
            sizes.append(int(len(pts)))
    return sorted(sizes)


def _gen_attempt() -> tuple[np.ndarray | None, dict | None]:
    out, bg, fcs, base_meta = _make_output()
    if len(np.unique(out)) != _N_COLORS:
        return None, None
    if _piece_count(out) != _PIECE_COUNT_REQUIRED:
        return None, None
    if max(_component_size_multiset(out)) > 5:
        return None, None

    try:
        inp, scatter_meta = _scatter(out, bg, fcs)
        got = main(inp)
    except Exception:
        return None, None

    if np.array_equal(got, out):
        meta = dict(base_meta)
        meta.update(scatter_meta)
        return inp, meta
    return None, None


def generate_input() -> np.ndarray:
    global _LAST_GEN_INFO
    for _ in range(_GEN_MAX_RETRIES):
        grid, meta = _gen_attempt()
        if grid is not None:
            _LAST_GEN_INFO = dict(meta or {})
            return grid
    raise RuntimeError("failed to generate a valid e8686506 combo input")
