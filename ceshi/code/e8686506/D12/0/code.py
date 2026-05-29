# Variant of e8686506
# Dimension: D12 — 碎片的旋转/翻转约束 (meta)
# Change: 碎片不可旋转不可翻转，保持原始朝向直接拼合... → 允许90度旋转

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for D12 (rotation-only variant)
# spec_variant:
#   core_change:
#     original assembly treated every scattered piece as having a fixed observed orientation.
#     this variant keeps the same output target (a dense LR-symmetric rectangle) but allows each
#     scattered non-frame piece to be quarter-turned independently before assembly.
#   concrete_behavior:
#     frame cells still appear in-place as a recognizable LR-symmetric outline inside the input;
#     they are the stable anchor. The freely scattered fill pieces are now sometimes rotated by
#     90/180/270 degrees, so the solver must infer both destination and orientation.
#   behavior_trace:
#     original:
#       an L triomino seen as [(0,0),(1,0),(1,1)] could only be placed with that exact pose.
#     variant:
#       the same observed triomino may represent any of four rotations; if the output needs
#       [(0,0),(0,1),(1,0)], the solver must try a 90-degree turn before placement.
#     difference:
#       the old type signature and placement loop silently assumed coords were final; that is no
#       longer harmless once the input pose is decoupled from the output pose.
#   complexity_calibration:
#     only rotations are added, not reflections. That keeps the control set small (4 poses) and
#     visually learnable from a few examples while still making asymmetric pieces visibly move.
#   implicit_assumptions_audited:
#     - grouping by exact coords is invalid now, so types are grouped by rotation-canonical shape.
#     - input-orientation anchor heuristics (horizontal bars / vertical lines) are unsafe now,
#       because a rotated piece no longer reveals which output side it belongs to.
#     - frame locking remains valid because the generator keeps the frame group unrotated and in
#       output-relative position; only the scattered non-frame patches are rotated.
#   generator_guarantees:
#     - frame cells are pasted as one in-place outline and never overlap scattered pieces.
#     - scattered patches are placed with padding, no overlap, and at least one quarter-turn-visible
#       asymmetric patch is actually rotated by 90 or 270 degrees.
#     - non-frame pieces are extracted from the intended output first, then rotated in input, and
#       the new solver is used as an oracle before accepting the sample.
#   visual_observability:
#     the variant is not encoded by color ids. It is visible because some input fragments have
#     different bounding boxes / silhouettes from their output counterparts, while the frame stays
#     fixed and shows where those fragments must eventually fit.
#   cleanliness:
#     rotation is limited to scattered fill pieces, avoiding stacked cues inside the frame region.
#     this keeps the outline readable and the changed rule local to the fragments the user handles.


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


def _build_requirements(H: int, W: int) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    In the rotation variant we intentionally avoid input-pose-dependent anchor inference:
    a scattered bar or line no longer tells us whether it belongs horizontally or vertically
    in the output. The only hard requirements come from the locked frame cells.
    """
    return np.full((H, W), -1, dtype=int)


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def is_lr_symmetric(parts: list[tuple[int, np.ndarray]]) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((H, W), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and is_lr_symmetric([(c1, p1)]):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and is_lr_symmetric([(c1, p1), (c2, p2)]):
                frame_parts = [(c1, p1), (c2, p2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric frame groups whose union bbox equals output dims"
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
        t = types[ti]
        col = int(t["col"])
        coords = variant["coords"]
        is_singleton = len(coords) == 1
        cells = [(r0 + dr, c0 + dc) for dr, dc in coords]
        cellset = set(cells)

        for r, c in cells:
            if board[r, c] != -1:
                return None
            if req[r, c] != -1 and int(req[r, c]) != col:
                return None
            mc = W - 1 - c
            if req[r, mc] != -1 and int(req[r, mc]) != col:
                return None
            v = board[r, mc]
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

    def search() -> bool:
        forced: list[tuple[int, list[tuple[int, int]]]] = []
        while True:
            progress = False
            for ti in range(len(types)):
                if counts[ti] <= 0:
                    continue
                cands = placements_for(ti)
                if len(cands) < counts[ti]:
                    for col, changed in reversed(forced):
                        unplace(changed)
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
            ok = (not np.any(board == -1)) and np.array_equal(board, board[:, ::-1])
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

        col = int(types[best_ti]["col"])
        counts[best_ti] -= 1
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

    assert search(), "assembly failed under rotation-aware search"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered color fragments into a left-right symmetric rectangle, allowing quarter turns.

    Input structure:
    - The background is the dominant color; all other cells form small monochrome fragments.
    - One in-place LR-symmetric frame group already shows the output rectangle's outline.
    - The remaining fragments are scattered elsewhere, and some of them may be rotated by 0/90/180/270 degrees.

    Rule:
    - Use the total non-background area to infer the dense output rectangle dimensions.
    - Lock the frame group into its output-relative cells.
    - Place every other fragment into the rectangle, trying all quarter-turn rotations, so the final board is exactly LR-symmetric and same-color fragments do not 4-touch across piece boundaries.

    Key insight:
    - The frame fixes the rectangle and many boundary cells; after that, mirror symmetry plus no-same-color-touching prunes the search enough that each scattered fragment's location and rotation become determined together.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    total_area = int((g != bg).sum())
    assert total_area > 0, "total_area must be positive: grid must contain non-background pieces"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(g, bg, H, W)
    req = _build_requirements(H, W)
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
    assert np.array_equal(board0, board0[:, ::-1]), "locked frame must already be LR-symmetric"

    color_piece_count: dict[int, int] = {}
    type_counter: Counter = Counter()
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        color_piece_count[int(col)] = len(comps)
        for pts in comps:
            coords = _piece_sig(pts)[2]
            if all((int(r), int(c)) in frame_input_cells for r, c in pts):
                continue
            type_counter[(int(col), _canonical_rot_key(coords))] += 1

    types = []
    for (col, canon_key), cnt in sorted(type_counter.items(), key=lambda kv: (-len(kv[0][1][2]), kv[0][0], kv[0][1])):
        _, _, canon_coords = canon_key
        rots = _unique_rotations(canon_coords)
        types.append(
            {
                "col": int(col),
                "rots": rots,
                "cnt": int(cnt),
                "area": int(len(canon_coords)),
            }
        )

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, board0, disallow_touch)
    return sol.astype(int)


# --------------- generator ---------------

_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_CENTER_CAP_PROB = 0.5

_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_MIN_ROT90_VISIBLE = 1
_MIN_QUARTER_TURN_CANDIDATES = 2

_GEN_MAX_RETRIES = 400


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
    half = W // 2
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
        raise ValueError
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    patches = _extract_non_frame_patches(out, bg, fcs)
    qturn_candidates = [i for i, p in enumerate(patches) if p["quarter_turn_visible"]]
    if len(qturn_candidates) < _MIN_QUARTER_TURN_CANDIDATES:
        raise ValueError("not enough quarter-turn-distinguishable patches")

    force_n = random.randint(_MIN_ROT90_VISIBLE, min(3, len(qturn_candidates)))
    forced = set(random.sample(qturn_candidates, force_n))
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
        "rotatable_piece_count": int(sum(p["unique_rotations"] > 1 for p in patches)),
        "non_frame_piece_count": int(len(patches)),
    }
    if meta["rot90_visible_count"] < _MIN_ROT90_VISIBLE:
        raise ValueError("rotation visibility requirement not met")
    return canvas, meta


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> tuple[np.ndarray | None, dict | None]:
    out, bg, fcs = _make_output()
    if out is None:
        return None, None
    if not np.array_equal(out, out[:, ::-1]):
        return None, None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None, None

    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None, None
    if any(len(pts) > _MAX_PIECE_AREA for col in np.unique(out) for pts in components_mask(out == col, connectivity=4)):
        return None, None

    try:
        inp, meta = _scatter(out, bg, fcs)
        got = main(inp)
    except Exception:
        return None, None

    if np.array_equal(got, out):
        return inp, meta
    return None, None


def generate_input() -> np.ndarray:
    global _LAST_GEN_INFO
    for _ in range(_GEN_MAX_RETRIES):
        grid, meta = _gen_attempt()
        if grid is not None:
            _LAST_GEN_INFO = dict(meta or {})
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D12 input")
