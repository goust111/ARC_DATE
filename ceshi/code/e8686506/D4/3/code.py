# Variant of e8686506
# Dimension: D4 — 对称约束：输出的对称性类型 (match_cond)
# Change: 输出具有严格的左右（垂直轴）镜像对称性... → 对角线镜像对称

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D4 variant
# - Concrete behavior:
#   The old task assembled a portrait rectangle whose color pattern had to satisfy
#   board[r, c] == board[r, W-1-c]. The new variant must satisfy board[r, c] == board[c, r],
#   so the output must be square. The input still contains scattered 4-connected color pieces,
#   but the preserved frame group now forms a diagonal-symmetric skeleton rather than an LR one.
#   In generation we use a 6x6 square whose frame is the union of the main diagonal and the
#   anti-diagonal. The lower-triangle filler pieces are mirrored to the upper triangle by
#   transpose, so the transpose relation is directly visible in the output.
# - Small trace:
#   Old logic for a candidate cell (r, c) checked whether board[r, W-1-c] was empty or had
#   the same color. In the variant, placing at (4, 1) instead constrains cell (1, 4).
#   This changes both legality checking and the way frame anchors propagate: a locked diagonal
#   chain now forces its transposed mate rather than a left/right mate.
# - Complexity calibration:
#   A plain main diagonal alone would underconstrain the board, and many outputs would still
#   look symmetric under LR by accident. The generator therefore uses two diagonal chains and
#   different frame colors, which makes the diagonal rule visually obvious and usually breaks
#   LR symmetry. Hole pieces stay large enough to be visually meaningful but bounded by area 6.
# - Implicit assumptions audited:
#   1. The old solver allowed rectangular outputs; diagonal symmetry requires H == W.
#   2. The old mirror check used column reflection; now it must use transpose.
#   3. The old "singleton pieces only on the border" rule is invalid here because diagonal
#      frame cells are interior singletons by 4-connectivity.
#   4. Old horizontal/vertical edge-bar anchors were tied to LR geometry, so the variant keeps
#      only corner-singleton hints and relies primarily on the diagonal frame lock.
#   5. A single frame component can already span the full square bbox, so frame detection now
#      prefers the valid 1-2-part candidate with the largest union area, ensuring both
#      diagonals are locked when present.
# - Placement legality:
#   The generator blocks the frame bbox while scattering non-frame pieces, preserves the whole
#   frame group in-place, and leaves at least one-cell padding between scattered pieces.
#   Every produced input is oracle-checked by main(input) == output.
# - Visual observability and cleanliness:
#   The output is a square with explicit diagonal structure and non-matching frame colors on
#   the two diagonals, so the task does not collapse back into an LR-symmetric reading.
#   Only four colors are used in each sample, keeping the grid readable.


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
    n: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    The LR task had optional horizontal/vertical edge anchors. Those are not stable
    under a diagonal-symmetry interpretation, so this variant only keeps the corner
    singleton hint when it exists.
    """
    req = np.full((n, n), -1, dtype=int)
    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    if corner_colors:
        col = int(corner_colors[0])
        req[0, 0] = col
        req[0, n - 1] = col
        req[n - 1, 0] = col
        req[n - 1, n - 1] = col
    return req


def _assemble(
    types: list[dict],
    n: int,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = np.full((n, n), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def transpose_ok(r: int, c: int, col: int) -> bool:
        v = board[c, r]
        if v != -1 and v != col:
            return False
        mirror_req = int(req[c, r])
        return mirror_req in (-1, col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not transpose_ok(r, c, col):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < n and 0 <= cc < n and board[rr, cc] == col:
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
            for r0 in range(n - int(t["h"]) + 1)
            for c0 in range(n - int(t["w"]) + 1)
            if can_place(ti, r0, c0)
        ]

    changed = True
    while changed:
        changed = False
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            assert places, "assembly dead end: a piece has no valid placement"
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
    assert np.array_equal(board, board.T), "assembled board must be diagonally symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a square whose colors mirror across the main diagonal.

    Input structure:
    - The background is the most common color; every other connected component is a puzzle fragment.
    - A preserved 8-connected frame group spans the eventual square output and already sits in its
      final relative arrangement. In this variant that frame is diagonally symmetric, not left-right symmetric.

    Rule:
    - Extract all non-background 4-connected components as pieces and infer the square output size
      from their total area.
    - Lock the diagonal frame cells, then place the remaining pieces so the filled square satisfies
      board[r, c] == board[c, r], while different pieces of the same color never become 4-neighbors.

    Key insight: the fixed frame exposes the output's orientation, and transpose symmetry couples each
    cell to its coordinate-swapped partner. That transpose constraint, combined with same-color separation,
    makes the packing propagation-solvable without backtracking.
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
            pieces.append((int(col), coords))

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
    assert len(best_dims) == 1, "closest-to-square output shape must be unique"
    h, w = best_dims[0]
    assert h == w, "diagonal symmetry requires a square output"
    n = h

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_diag_symmetric(parts: list[tuple[int, np.ndarray]], size: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((size, size), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp.T)

    frame_parts: list[tuple[int, np.ndarray]] = []
    best_frame_area = -1
    for i, (c1, p1) in enumerate(comps8_all):
        candidates = [[(c1, p1)]]
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            candidates.append([(c1, p1), (c2, p2)])
        for parts in candidates:
            union = np.concatenate([p for _, p in parts])
            r0, c0, r1, c1b = bbox_coords(union)
            if r1 - r0 + 1 != n or c1b - c0 + 1 != n:
                continue
            if not _is_diag_symmetric(parts, n):
                continue
            area = sum(len(p) for _, p in parts)
            if area > best_frame_area:
                best_frame_area = area
                frame_parts = parts
    assert frame_parts, "input must contain 1-2 diagonal-symmetric frame parts spanning the output square"

    req = _build_requirements(n, comps_by_color)

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < n and 0 <= oc < n:
                req[or_, oc] = int(col)

    disallow_touch = {col for col, count in color_piece_count.items() if count > 1}
    sol = _assemble(types, n, req, disallow_touch)
    return sol.astype(int)


# --------------- generator (output-first diagonal assembly) ---------------

_N = 6

_N_COLORS_MIN = 4
_N_COLORS_MAX = 4
_PIECE_COUNT_MIN = 14
_PIECE_COUNT_MAX = 17
_MAX_PIECE_AREA = 6

_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

_GEN_MAX_RETRIES = 300


def _build_output() -> tuple[np.ndarray, set[int]]:
    """
    Build a 6x6 diagonally symmetric output.

    Frame = main diagonal + anti-diagonal. We keep the two diagonals in different colors
    so the samples do not also read as LR-symmetric.
    """
    palette = [0, 1, 2, 3]
    random.shuffle(palette)
    main_col, anti_col, fill_a, fill_b = palette

    out = np.full((_N, _N), -1, dtype=int)
    for i in range(_N):
        out[i, i] = main_col
        out[i, _N - 1 - i] = anti_col

    lower_mask = np.tril(np.ones((_N, _N), dtype=bool), k=-1) & (out == -1)
    lower_comps = components_mask(lower_mask, connectivity=4)
    assert len(lower_comps) == 2, "the 6x6 double-diagonal template must create two lower components"
    lower_comps = sorted(lower_comps, key=lambda pts: tuple(np.min(pts, axis=0)))
    fill_cols = [fill_a, fill_b]
    random.shuffle(fill_cols)

    for pts, col in zip(lower_comps, fill_cols):
        for r, c in pts:
            rr, cc = int(r), int(c)
            out[rr, cc] = col
            out[cc, rr] = col

    assert np.all(out >= 0)
    assert np.array_equal(out, out.T)
    assert not np.array_equal(out, out[:, ::-1]), "variant output must not collapse back to LR symmetry"
    return out, {main_col, anti_col}


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    out, abstract_fcs = _build_output()
    bg = random.randint(0, 9)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out)
    return remapped.astype(int), bg, real_fcs


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
        raise ValueError("canvas too small for preserved frame bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r:off_r + oh, off_c:off_c + ow] = -1
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
    bbox_region = canvas[off_r:off_r + oh, off_c:off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    out, bg, fcs = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out.T):
        return None
    if np.array_equal(out, out[:, ::-1]):
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
    raise RuntimeError("failed to generate a valid diagonal-symmetry e8686506 input")
