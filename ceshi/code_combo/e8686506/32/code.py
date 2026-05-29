# Combo variant of e8686506 (index 32)
# combo_spec: D3/2+D4/2+D12/0
# Name: 正方形目标+90度旋转对称+允许旋转

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for combo D3/2 + D4/2 + D12/0
#
# a) Concrete behavior and trace:
#    The old task assembled a portrait rectangle and only required left-right
#    agreement. Here the target must be a square and every painted cell belongs
#    to a 90-degree rotation orbit. In addition, scattered non-frame fragments
#    may arrive quarter-turned.
#
#    Small trace:
#    - Old rule: a blue 1x2 bar seen horizontally could only fill one
#      horizontal slot, and its placed cells only constrained their LR mates.
#    - New rule: the same observed bar may be horizontal or vertical in the
#      input; once placed in one quarter, its color determines the full orbit of
#      four bars around the center. The solver must infer both its pose and its
#      orbit role.
#    - Difference: shape inference is no longer "read coords and mirror them".
#      It is now "canonicalize by quarter-turn, identify the quarter-turn frame,
#      solve one quadrant, and propagate by the rotation chain".
#
# b) Complexity calibration:
#    A free-form square tiling with arbitrary rotations would search too widely
#    and would often look visually generic. To keep the rule observable, the
#    generator uses one of two fixed 7x7 pinwheel templates. The frame is a
#    single 8-connected rotational outline; the remaining roles are repeated
#    four times each (singletons, dominoes, tetrominoes). This yields enough
#    repetition for humans to see the quarter-turn structure without relying on
#    hidden color ordering.
#
# c) Hidden assumptions audited:
#    - The original factor search for a portrait rectangle is invalid. Total
#      area must now be a perfect square.
#    - Exact piece signatures are unsafe because the input may rotate pieces.
#      Non-frame piece types are grouped by rotation-canonical shape instead.
#    - Old corner/bar/line heuristics are unsafe because an input pose no longer
#      tells us which side of the output a piece belongs to.
#    - Mirror-only placement checks are invalid. A legal placement must respect
#      its full 4-cell rotation orbit.
#    - The solver now assumes the input contains one rotational frame matching a
#      supported template and exactly three non-frame role colors whose
#      component multisets are {4x1}, {4x2}, {4x4}. The generator guarantees
#      those roles explicitly.
#
# d) Placement legality:
#    The frame is pasted as one in-place group inside its real 7x7 bbox. Every
#    other patch is scattered outside that blocked square with Chebyshev padding,
#    so there are no overlaps and the frame cells remain readable anchors.
#
# e) Visual observability:
#    Samples are rejected unless at least one quarter-turn-distinguishable patch
#    is actually rotated by 90 or 270 degrees in the input. The output is also
#    rejected if it accidentally becomes left-right symmetric, so the new rule
#    is visibly "square + quarter-turn" rather than a disguised copy of the old
#    mirror task.
#
# f) Visual cleanliness:
#    Only the frame occupies the anchored central bbox. All rotatable fill
#    pieces are scattered elsewhere. The output uses four role colors with clear
#    repeated geometry, avoiding noisy color remaps or overlapping annotations.
#
# g) Change inferability:
#    Colors are assigned only to structural roles {frame, singleton, domino,
#    tetromino}. Because each role repeats in four quarter-turn-related places,
#    a human can infer the color grouping from geometry alone; there is no need
#    to memorize arbitrary color-id conventions.
# ---------------------------------------------------------------------------


_LAST_GEN_INFO: dict | None = None


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


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


def _canonical_rot_key(
    coords: tuple[tuple[int, int], ...]
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rots = _unique_rotations(coords)
    return min((int(v["h"]), int(v["w"]), v["coords"]) for v in rots)


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


def _paint_orbit(board: np.ndarray, cells: list[tuple[int, int]], col: int) -> None:
    n = int(board.shape[0])
    for r, c in cells:
        for rr, cc in _orbit_cells(r, c, n):
            board[rr, cc] = col


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


def _build_requirements(n: int) -> np.ndarray:
    return np.full((n, n), -1, dtype=int)


def _find_frame_parts(g: np.ndarray, bg: int, n: int) -> list[tuple[int, np.ndarray]]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    frame_parts: list[tuple[int, np.ndarray]] = []
    for col, pts in comps8_all:
        r0, c0, r1, c1 = bbox_coords(pts)
        if r1 - r0 + 1 == n and c1 - c0 + 1 == n and _is_rot4_symmetric([(col, pts)], n):
            frame_parts = [(int(col), pts)]
            break
    assert frame_parts, "input must contain one rotational frame whose bbox is the output square"
    return frame_parts


def _template_frame_bitmap(tmpl: dict, n: int) -> np.ndarray:
    bmp = np.zeros((n, n), dtype=bool)
    for r, c in tmpl["frame_seed"]:
        for rr, cc in _orbit_cells(r, c, n):
            bmp[rr, cc] = True
    return bmp


def _match_template(frame_bmp: np.ndarray) -> dict:
    for idx, tmpl in enumerate(_TEMPLATES):
        if np.array_equal(frame_bmp, _template_frame_bitmap(tmpl, frame_bmp.shape[0])):
            out = dict(tmpl)
            out["template_id"] = int(idx)
            return out
    raise AssertionError("detected frame does not match any supported rotational template")


def _assemble(
    types: list[dict],
    n: int,
    req: np.ndarray,
    prefilled: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = prefilled.copy()
    counts = [int(t["cnt"]) for t in types]
    orbit_cache = {(r, c): _orbit_cells(r, c, n) for r in range(n) for c in range(n)}
    fail_cache: set[tuple[tuple[int, ...], bytes]] = set()

    def orbit_ok(cells: list[tuple[int, int]], col: int) -> bool:
        for r, c in cells:
            for rr, cc in orbit_cache[(r, c)]:
                if req[rr, cc] != -1 and int(req[rr, cc]) != col:
                    return False
                v = int(board[rr, cc])
                if v != -1 and v != col:
                    return False
        return True

    def candidate_cells(ti: int, variant: dict, r0: int, c0: int) -> list[tuple[int, int]] | None:
        col = int(types[ti]["col"])
        cells = [(r0 + dr, c0 + dc) for dr, dc in variant["coords"]]
        cellset = set(cells)

        for r, c in cells:
            if board[r, c] != -1:
                return None
            if req[r, c] != -1 and int(req[r, c]) != col:
                return None

        if not orbit_ok(cells, col):
            return None

        if disallow_touch and col in disallow_touch:
            for r, c in cells:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < n and 0 <= cc < n and board[rr, cc] == col and (rr, cc) not in cellset:
                        return None
        return cells

    def placements_for(ti: int) -> list[tuple[dict, int, int, list[tuple[int, int]]]]:
        t = types[ti]
        out = []
        for variant in t["rots"]:
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
            if not progress:
                break

        key = (tuple(counts), board.tobytes())
        if key in fail_cache:
            for ti, changed in reversed(forced):
                unplace(changed)
                counts[ti] += 1
            return False

        if all(c == 0 for c in counts):
            ok = (not np.any(board < 0)) and np.array_equal(board, np.rot90(board))
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

    assert search(), "assembly failed under square rotational rotation-aware search"
    assert np.array_equal(board, np.rot90(board)), "assembled board must be 90-degree rotationally symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered color fragments into a dense square with 90-degree rotational symmetry, allowing quarter turns.

    Input structure:
    - The background is the dominant color; every other color forms detached monochrome fragments.
    - One rotationally symmetric 8-connected frame already sits in its true square bbox.
    - The remaining fragments are scattered elsewhere, and some may be rotated by 0/90/180/270 degrees.

    Rule:
    - Infer the square output size from the total non-background area.
    - Detect the rotational frame template and use the repeated component-size pattern of the other colors to recover which color plays the singleton, domino, and tetromino roles.
    - Place all non-frame fragments back into the square, trying all quarter-turns, so the final board matches the detected template under 90-degree rotation and same-color fragments never 4-touch across piece boundaries.

    Key insight:
    - The frame determines the square and the quarter-turn handedness. After that, each remaining color role repeats in four linked orbit positions, so solving one quadrant fixes the other three quadrants and also fixes which rotations the scattered pieces must use.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    total_area = int((g != bg).sum())
    assert total_area > 0, "grid must contain non-background pieces"
    n = int(round(total_area ** 0.5))
    assert n * n == total_area, "square rotational variant requires perfect-square area"

    frame_parts = _find_frame_parts(g, bg, n)
    frame_bmp = _normalized_frame_bitmap(frame_parts, n)
    tmpl = _match_template(frame_bmp)

    req = _build_requirements(n)
    board0 = np.full((n, n), -1, dtype=int)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    frame_input_cells = {(int(r), int(c)) for r, c in all_frame_pts}
    frame_colors = {int(col) for col, _ in frame_parts}

    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            assert 0 <= or_ < n and 0 <= oc < n, "frame cell must stay inside inferred square"
            req[or_, oc] = int(col)
            board0[or_, oc] = int(col)
    assert np.array_equal(board0, np.rot90(board0)), "locked frame must already be rotationally symmetric"

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    color_piece_count: dict[int, int] = {}
    type_counter: Counter = Counter()

    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[int(col)] = comps
        color_piece_count[int(col)] = len(comps)
        for pts in comps:
            if all((int(r), int(c)) in frame_input_cells for r, c in pts):
                continue
            coords = _piece_sig(pts)[2]
            type_counter[(int(col), _canonical_rot_key(coords))] += 1

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
                if req[rr, cc] != -1:
                    assert int(req[rr, cc]) == color, "role constraints must agree with frame constraints"
                req[rr, cc] = int(color)

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


# --------------- generator ---------------

_OUTPUT_SIZE = 7
_N_COLORS = 4
_PIECE_COUNT_REQUIRED = 17
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_MIN_ROT90_VISIBLE = 2
_MIN_QUARTER_TURN_CANDIDATES = 3
_GEN_MAX_RETRIES = 400

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


def _make_output() -> tuple[np.ndarray, int, set[int], dict]:
    tmpl = random.choice(_TEMPLATES)
    n = _OUTPUT_SIZE

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    frame_col, singleton_col, domino_col, tetromino_col = map(int, palette[:_N_COLORS])

    board = np.full((n, n), -1, dtype=int)
    _paint_orbit(board, list(tmpl["frame_seed"]), frame_col)
    _paint_orbit(board, list(tmpl["roles"]["singleton"]), singleton_col)
    _paint_orbit(board, list(tmpl["roles"]["domino"]), domino_col)
    _paint_orbit(board, list(tmpl["roles"]["tetromino"]), tetromino_col)

    assert not np.any(board < 0), "template must cover the full output square"
    assert np.array_equal(board, np.rot90(board)), "output must be 90-degree rotationally symmetric"
    assert not np.array_equal(board, board[:, ::-1]), "output must not collapse to LR symmetry"
    meta = {
        "template_id": int(0 if tmpl is _TEMPLATES[0] else 1),
        "frame_color": int(frame_col),
    }
    return board, int(bg), {int(frame_col)}, meta


def _shape_key_from_patch(patch: np.ndarray, bg: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = np.argwhere(patch != bg)
    assert coords.size > 0
    return _piece_sig(coords)


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_colors: set[int]) -> list[dict]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in frame_colors:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = int(col)
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


def _scatter(out: np.ndarray, bg: int, frame_colors: set[int]) -> tuple[np.ndarray, dict]:
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
    patches = _extract_non_frame_patches(out, bg, frame_colors)
    qturn_candidates = [i for i, p in enumerate(patches) if p["quarter_turn_visible"]]
    if len(qturn_candidates) < _MIN_QUARTER_TURN_CANDIDATES:
        raise ValueError("not enough quarter-turn-visible patches in output")

    force_n = random.randint(_MIN_ROT90_VISIBLE, min(4, len(qturn_candidates)))
    forced = set(random.sample(qturn_candidates, force_n))
    rot90_visible_count = 0
    rotated_nontrivial_count = 0

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

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in frame_colors:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    meta = {
        "rot90_visible_count": int(rot90_visible_count),
        "rotated_nontrivial_count": int(rotated_nontrivial_count),
        "quarter_turn_candidate_count": int(len(qturn_candidates)),
        "rotatable_piece_count": int(sum(p["unique_rotations"] > 1 for p in patches)),
        "non_frame_piece_count": int(len(patches)),
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
    out, bg, frame_colors, meta0 = _make_output()
    if len(np.unique(out)) != _N_COLORS:
        return None, None
    if _piece_count(out) != _PIECE_COUNT_REQUIRED:
        return None, None
    if max(_component_size_multiset(out)) > 5:
        return None, None

    try:
        inp, meta1 = _scatter(out, bg, frame_colors)
        got = main(inp)
    except Exception:
        return None, None

    if np.array_equal(got, out):
        meta = dict(meta0)
        meta.update(meta1)
        meta["output_size"] = int(out.shape[0])
        meta["is_square"] = bool(out.shape[0] == out.shape[1])
        meta["rot4"] = bool(np.array_equal(out, np.rot90(out)))
        meta["lr_symmetric"] = bool(np.array_equal(out, out[:, ::-1]))
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
