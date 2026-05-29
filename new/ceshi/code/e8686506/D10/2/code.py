# Variant of e8686506
# Dimension: D10 — 框架碎片的结构特征 (meta)
# Change: 最大碎片为环形/U形等非凸形状（中心有洞或凹陷），其他小碎片填入空隙... → 多个中型框架碎片

# Design analysis for the D10 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     from: "one dominant frame object (or a split into 1-2 large outline pieces)"
#     to: "3-4 medium frame fragments that interlock as peers"
#   concrete_pixel_behavior:
#     output:
#       - still a dense LR-symmetric portrait rectangle with width 5
#       - a thin zigzag scaffold is split into 3-4 colored row segments
#       - each segment contains one center-bridge row so the segment is a single 8-connected frame fragment
#       - the segments are medium-sized (roughly 5-9 cells each), so no single frame fragment dominates
#       - remaining cells are filled by smaller symmetric chunks
#     input:
#       - the 3-4 frame fragments stay together inside one clean HxW bbox, preserving output-relative positions
#       - all non-frame chunks are scattered outside that bbox with padding, so the group of frame fragments is visually recognizable
#   behavior_trace:
#     original:
#       - solver searched for 1 or 2 eight-connected components whose union bbox matched the output
#       - generator therefore kept one big outline (or a top/bottom split) inside the bbox
#     variant:
#       - solver now searches for exactly 3 or 4 eight-connected components whose union bbox matches the output
#       - it additionally checks that the bbox contains no extra colored cells beyond those components
#       - generator constructs 3-4 bridge-like frame bands and keeps that whole set inside a clean bbox
#     visible_difference:
#       - old samples could look like "one preserved outline plus loose fillers"
#       - new samples look like "several comparable preserved scaffold pieces already partially assembled"
#   complexity_calibration:
#     - width remains 5 so the task family is still recognizably e8686506
#     - height is restricted to 6-10 to make 3-4 medium frame bands visually meaningful
#     - the scaffold still follows a left-half zigzag like the source task; only the segmentation changes
#     - fill chunks stay bounded so assembly remains propagation-solvable
#   implicit_assumptions_audit:
#     - old main() assumed a valid frame witness existed among subsets of size 1 or 2; this is false here
#     - old frame detection only checked bbox + symmetry; with 3-4 parts that is too weak because a wrong subset can share the same bbox
#     - new main() therefore adds the clean-bbox test: every non-bg cell inside the witness bbox must belong to the chosen frame subset
#     - old generator allowed H=5 where 3-4 medium frame bands are impossible; new generator forbids that
#     - _build_requirements() is kept but no longer asserts on ambiguous legacy anchors; it simply ignores them
#   placement_legality:
#     - frame bands are painted directly into the output grid and are pairwise non-overlapping by row partition
#     - the frame bbox is blocked on the input canvas before scattering fillers, so fillers cannot overlap or leak into the preserved scaffold
#     - random_free_location() with padding=1 keeps scattered chunks separated
#   visual_observability:
#     - the control variable is the count of preserved frame fragments (3 or 4), which is directly visible in the input
#     - row-band templates create bridge-like U/H/C silhouettes, so the "multi-frame" change is visible without relying on color ids
#   cleanliness:
#     - only one structured region exists inside the input bbox: the frame fragments
#     - everything else is outside that rectangle, so there is no layered ambiguity
#   color_reasoning:
#     - frame colors and fill colors are disjoint
#     - total colors stay small (4-6) and there is no hidden color ordering rule
#   solver_generator_contract:
#     - main() assumes exactly one clean 3-4-part frame witness exists
#     - generate_input() guarantees that by preserving exactly those parts in the bbox and scattering all other pieces away from it

import itertools
import random
from collections import Counter

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


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    Legacy anchors are still supported, but the D10 generator intentionally avoids them.
    """
    req = np.full((H, W), -1, dtype=int)

    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    corner_color = corner_colors[0] if len(corner_colors) == 1 else None
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
            if len(bar_colors) <= 2:
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
    if len(vertical_line_candidates) == 1:
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


def _is_lr_symmetric_union(parts: list[tuple[int, np.ndarray]], H: int, W: int) -> bool:
    all_pts = np.concatenate([pts for _, pts in parts])
    r0, c0, _, _ = bbox_coords(all_pts)
    bmp = np.zeros((H, W), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, bmp[:, ::-1])


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            if len(pts) >= 3:
                comps8_all.append((int(col), pts))

    candidates: list[list[tuple[int, np.ndarray]]] = []
    for k in (3, 4):
        for idxs in itertools.combinations(range(len(comps8_all)), k):
            parts = [comps8_all[i] for i in idxs]
            union = np.concatenate([pts for _, pts in parts])
            r0, c0, r1, c1 = bbox_coords(union)
            if r1 - r0 + 1 != H or c1 - c0 + 1 != W:
                continue
            if not _is_lr_symmetric_union(parts, H, W):
                continue
            if int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1] != bg)) != int(len(union)):
                continue
            candidates.append(parts)

    assert len(candidates) == 1, "input must contain exactly one clean 3-4 part LR-symmetric frame group"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into a dense left-right symmetric rectangle whose scaffold is made of several peer frame pieces.

    Input structure:
    - The background is the dominant color; every other color forms detached puzzle fragments.
    - One preserved HxW rectangle-shaped bbox contains exactly 3-4 medium 8-connected frame fragments kept in their output-relative positions.
    - All other filler chunks are scattered elsewhere on the canvas.

    Rule:
    - Extract every non-background 4-connected component as a piece type and infer the output dimensions from total area.
    - Detect the unique clean bbox whose colored cells are exactly 3-4 LR-symmetric frame fragments, and lock those cells to their output positions.
    - Pack the remaining pieces so the final rectangle is perfectly left-right mirrored and same-color components never merge by 4-neighbor contact.

    Key insight:
    - The preserved multi-part scaffold now comes from a group of 3-4 comparable frame fragments rather than a single dominant outline, so the solver must identify that whole clean group before ordinary symmetry-based propagation finishes the assembly.
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

    frame_parts = _find_frame_parts(g, bg, H, W)
    req = _build_requirements(H, W, comps_by_color)

    all_frame_pts = np.concatenate([pts for _, pts in frame_parts])
    fr0, fc0, _, _ = bbox_coords(all_frame_pts)
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int)


# --------------- generator (o5x) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 30
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MIN = 6
_HEIGHT_MAX = 10

# === CFG: Colors ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_FRAME_PARTS_MIN = 3
_FRAME_PARTS_MAX = 4

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 5
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 18

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
            if w == _WIDTH_REQUIRED and _HEIGHT_MIN <= h <= _HEIGHT_MAX:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _row_partitions(total: int, k: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, cur: list[int]) -> None:
        if slots == 0:
            if remaining == 0:
                out.append(tuple(cur))
            return
        lo = max(2, remaining - 4 * (slots - 1))
        hi = min(4, remaining - 2 * (slots - 1))
        for v in range(lo, hi + 1):
            cur.append(v)
            rec(remaining - v, slots - 1, cur)
            cur.pop()

    rec(total, k, [])
    return out


def _zigzag_cols(H: int) -> list[int] | None:
    for _ in range(80):
        cols = [0]
        c = random.choice([0, 1])
        for r in range(1, H - 1):
            if r >= 2 and cols[-1] == cols[-2]:
                if random.random() < 0.75:
                    c = 1 - c
            else:
                if random.random() < 0.45:
                    c = 1 - c
            cols.append(c)
        cols.append(0)
        if sum(v == 1 for v in cols) >= _FRAME_PARTS_MIN:
            return cols
    return None


def _choose_segments(H: int, cols: list[int]) -> list[int] | None:
    feasible = []
    for k in range(_FRAME_PARTS_MIN, _FRAME_PARTS_MAX + 1):
        if 2 * k <= H <= 4 * k:
            feasible.append(k)
    if not feasible:
        return None

    random.shuffle(feasible)
    for k in feasible:
        parts = _row_partitions(H, k)
        random.shuffle(parts)
        for lengths in parts:
            if max(lengths) - min(lengths) > 1:
                continue
            row0 = 0
            ok = True
            for L in lengths:
                if 1 not in cols[row0 : row0 + L]:
                    ok = False
                    break
                row0 += L
            if ok:
                return list(lengths)
    return None


def _paint_segment(grid: np.ndarray, row0: int, cols: list[int], color: int) -> None:
    bridge_rows = [i for i, v in enumerate(cols) if v == 1]
    bridge = random.choice(bridge_rows)
    for dr, c in enumerate(cols):
        r = row0 + dr
        grid[r, c] = color
        grid[r, grid.shape[1] - 1 - c] = color
        if dr == bridge:
            grid[r, grid.shape[1] // 2] = color


def _adjacent_palette(grid: np.ndarray, cells: list[tuple[int, int]]) -> set[int]:
    H, W = grid.shape
    out = set()
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] >= 0:
                out.add(int(grid[rr, cc]))
    return out


def _fill_holes(grid: np.ndarray, fill_pool: list[int]) -> None:
    H, W = grid.shape
    half = W // 2
    holes = components_mask(grid == -1, connectivity=4)
    prev_col: int | None = None
    for hole in holes:
        canon = sorted((int(r), int(c)) for r, c in hole if int(c) <= half)
        chunks = [canon[i : i + _MAX_CHUNK_SIZE] for i in range(0, len(canon), _MAX_CHUNK_SIZE)]
        for chunk in chunks:
            mirrored_cells = []
            for r, c in chunk:
                mirrored_cells.append((r, c))
                if W - 1 - c != c:
                    mirrored_cells.append((r, W - 1 - c))
            avoid = _adjacent_palette(grid, mirrored_cells)
            if prev_col is not None:
                avoid.add(prev_col)
            opts = [col for col in fill_pool if col not in avoid]
            if not opts:
                opts = [col for col in fill_pool if col not in _adjacent_palette(grid, mirrored_cells)] or list(fill_pool)
            col = random.choice(opts)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            prev_col = col


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
    cols = _zigzag_cols(H)
    if cols is None:
        return None, set()
    lengths = _choose_segments(H, cols)
    if lengths is None:
        return None, set()
    k = len(lengths)

    n_colors = random.randint(max(_N_COLORS_MIN, k + 1), _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)
    frame_colors = palette[:k]
    fill_pool = palette[k:]
    if not fill_pool:
        return None, set()

    grid = np.full((H, W), -1, dtype=int)
    row = 0
    for color, L in zip(frame_colors, lengths):
        _paint_segment(grid, row, cols[row : row + L], color)
        row += L

    _fill_holes(grid, fill_pool)
    if np.any(grid < 0):
        return None, set()
    return grid, set(frame_colors)


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, frame_colors = _build_output(H, W)
    if out is None:
        return None, 0, set()

    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in frame_colors}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out)
    return remapped.astype(int), bg, real_fcs


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
        raise ValueError("output bbox does not fit the canvas")
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

    frame_mask = np.isin(out, list(frame_colors))
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _frame_component_sizes(out: np.ndarray, frame_colors: set[int]) -> list[int]:
    sizes = []
    for col in frame_colors:
        comps = components_mask(out == col, connectivity=8)
        sizes.extend(len(pts) for pts in comps)
    return sorted(int(v) for v in sizes)


def _has_legacy_anchors(out: np.ndarray) -> bool:
    for col in map(int, np.unique(out)):
        comps = components_mask(out == col, connectivity=4)
        if len(comps) == 4 and all(len(pts) == 1 for pts in comps):
            return True
        if len(comps) == 2:
            sigs = [_piece_sig(pts) for pts in comps]
            (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
            if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
                return True
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            hh, ww = r1 - r0 + 1, c1 - c0 + 1
            if hh == 1 and ww == out.shape[1] - 2 and len(pts) == ww:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    out, bg, frame_colors = _make_output()
    if out is None:
        return None
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None
    frame_sizes = _frame_component_sizes(out, frame_colors)
    if not (_FRAME_PARTS_MIN <= len(frame_sizes) <= _FRAME_PARTS_MAX):
        return None
    if min(frame_sizes) < 5:
        return None
    if max(frame_sizes) > 10:
        return None

    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(out)
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None

    try:
        inp = _scatter(out, bg, frame_colors)
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
    raise RuntimeError("failed to generate a valid e8686506 D10 variant input")
