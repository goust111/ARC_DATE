# Variant of e8686506
# Dimension: D4 — 对称约束：输出的对称性类型 (match_cond)
# Change: 输出具有严格的左右（垂直轴）镜像对称性... → 双轴对称（左右+上下）

# Design analysis
# yaml:
#   concrete_behavior:
#     output:
#       - The solved rectangle remains a dense portrait 5-column assembly, but every colored cell now
#         belongs to a 4-way reflection orbit (left/right + up/down) unless it lies on a symmetry axis.
#       - A visually recognizable 8-connected frame still exists, but its row pattern is vertically palindromic
#         and drawn from a narrow zigzag family, so the frame bitmap itself is symmetric across both axes.
#       - Four corner singletons are intentionally present as anchors. They stay on the border, which preserves
#         the original singleton heuristic while making the stronger symmetry easy to see.
#     input:
#       - The frame is kept together inside the input exactly as before; all other pieces are scattered.
#       - Non-frame pieces are produced as mirrored interior stripes plus a center-axis run, so several piece
#         types are only vertically disambiguated by the new UD constraint.
#     trace_example:
#       - Original logic: a row-pair hole chunk mirrored only left/right can legally sit near the top or near
#         the bottom as long as horizontal reflection is respected.
#       - New logic: the same chunk must also agree with its up/down counterpart, so a top candidate is only
#         legal if the matching bottom cells can carry the same color; this removes the old vertical ambiguity.
#   complexity_calibration:
#     - Width stays 5 and the frame/hole assembly style is preserved, so the task still looks like the source task.
#     - Height is restricted to 7 so a real center row exists and the four-quadrant symmetry is always visible.
#     - Corner markers plus a mirrored zigzag frame keep the search constrained; the interior is reduced to a
#       few repeated symmetric piece families so the task stays readable and nontrivial.
#   implicit_assumption_review:
#     - Old _assemble() only checked LR mirrors; in this variant every candidate cell must agree with all cells
#       in its reflection orbit, otherwise a vertically inconsistent packing could slip through.
#     - Old frame detection only tested LR symmetry; that would incorrectly accept inputs whose frame does not
#       enforce the new rule, so the frame bitmap check must become dual-axis.
#     - The original singleton-border rule is kept, which means the generator must avoid an isolated center cell;
#       the center axis is therefore filled as a vertical run instead of a singleton.
#     - _build_requirements() still uses the old corner/bar/line heuristics. To keep those heuristics safe,
#       the generator deliberately supplies only one genuine 4-singleton color (the corners) and avoids
#       accidental off-axis singleton quartets.
#   placement_legality:
#     - The frame is pasted as a fixed group inside a reserved bbox; scattered pieces are placed with the same
#       non-overlap and Chebyshev padding policy as the source task.
#     - The generator uses fixed row families whose component sizes are known in advance, so every piece stays
#       in-bounds, non-overlapping, and inside the area budget before scattering.
#   observability:
#     - From 2-3 examples a human can see the new rule directly because the entire output is now symmetric in
#       four quadrants, not merely mirrored left/right.
#     - The generator keeps several color assignments over the same dual-symmetric geometry so the change is not
#       hidden, while the stronger up/down mirror remains immediately visible.
#   cleanliness:
#     - Only three structural layers are visible in the output: corner anchors, the symmetric frame, and the
#       mirrored fill chunks. They do not overlap, so the rectangle stays readable.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


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

    In the D4 variant we keep the corner-singleton heuristic and add one
    axis-specific hint:
      - If a color forms exactly one odd-length vertical line, it must be the
        self-symmetric center-axis line and is therefore placed on the middle column.

    The source task's bar / paired-line hints are intentionally dropped here because
    the new generator uses center-axis vertical pieces that would otherwise be
    mistaken for edge markers.
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

    center_line_candidates = []
    for col, comps in comps_by_color.items():
        if len(comps) != 1:
            continue
        h0, w0, coords0 = _piece_sig(comps[0])
        want = tuple((r, 0) for r in range(h0))
        if w0 == 1 and h0 % 2 == 1 and h0 > 1 and coords0 == want:
            center_line_candidates.append((col, h0))
    assert len(center_line_candidates) <= 1, "center vertical line hint must be unique when present"
    if center_line_candidates:
        col, L = center_line_candidates[0]
        top = (H - L) // 2
        req[top : top + L, W // 2] = col

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

    def orbit_ok(r: int, c: int, col: int) -> bool:
        for rr, cc in _orbit_coords(H, W, r, c):
            v = board[rr, cc]
            if v != -1 and v != col:
                return False
        return True

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
            if not orbit_ok(r, c, col):
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
            r, c = r0 + dr, c0 + dc
            board[r, c] = col

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
    assert np.array_equal(board, board[::-1, :]), "assembled board must be UD-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a dense rectangle with both left-right and up-down symmetry.

    Input structure:
    - The background is the dominant color; every non-background component is a puzzle fragment.
    - One 8-connected frame still appears as 1-2 grouped parts, and some fragment multiplicities may act as corner or edge hints.

    Rule:
    - Collect every non-background component as an untranslated piece and infer the output rectangle from total area.
    - Use the grouped frame cells plus any corner/edge hints to anchor the rectangle, then pack all pieces so the result
      is simultaneously mirror-symmetric across the vertical and horizontal axes.
    - As in the source task, pieces of the same color must not become 4-connected to each other in the assembled board.

    Key insight:
    - The decisive new constraint is not just matching left and right, but enforcing a full 4-way reflection orbit for
      every placement; this removes the top-versus-bottom ambiguities that the original single-axis rule would leave open.
    """
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

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_d4_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return _is_dual_symmetric_bitmap(bmp)

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W:
            if _is_d4_symmetric([(c1, p1)], H, W):
                frame_parts = [(c1, p1)]
                break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W:
                if _is_d4_symmetric([(c1, p1), (c2, p2)], H, W):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 dual-symmetric pieces (8-conn) whose union bbox equals output dims"

    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int)


# --------------- generator ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHTS_REQUIRED = (7,)

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 400


def _valid_dims():
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h in _HEIGHTS_REQUIRED:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _fixed_frame_cells(H: int, W: int) -> set[tuple[int, int]]:
    assert (H, W) == (7, 5)
    left_rows = [1, 0, 0, 2, 0, 0, 1]
    cells = set()
    for r, kind in enumerate(left_rows):
        cols = [0] if kind == 0 else ([1] if kind == 1 else [0, 1])
        for c in cols:
            cells.add((r, c))
            cells.add((r, W - 1 - c))
    return cells


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
    palette = list(range(random.randint(_N_COLORS_MIN, _N_COLORS_MAX)))
    random.shuffle(palette)
    corner_col = palette[0]
    frame_col = palette[1]
    side_col = palette[2]
    center_col = palette[3]
    cap_col = side_col

    grid = np.full((H, W), -1, dtype=int)
    for r, c in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        grid[r, c] = corner_col

    for r, c in _fixed_frame_cells(H, W):
        if grid[r, c] != -1:
            return None, set()
        grid[r, c] = frame_col

    for r in (1, 2, 4, 5):
        grid[r, 1] = side_col
        grid[r, 3] = side_col

    for r in (2, 3, 4):
        grid[r, 2] = center_col

    for r in (0, 1, 5, 6):
        grid[r, 2] = cap_col

    if np.any(grid < 0):
        return None, set()
    if not np.array_equal(grid, grid[:, ::-1]):
        return None, set()
    if not np.array_equal(grid, grid[::-1, :]):
        return None, set()
    return grid, {frame_col}


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
        frame_mask |= (out == col)
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
    if not np.array_equal(out, out[::-1, :]):
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
    raise RuntimeError("failed to generate a valid e8686506 D4 input")
