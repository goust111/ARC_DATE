# Variant of e8686506
# Dimension: D12 — 碎片的旋转/翻转约束 (meta)
# Change: 碎片不可旋转不可翻转，保持原始朝向直接拼合... → 允许水平翻转

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D12 variant
# a) Concrete behavior:
#    - The output stays a dense left-right symmetric rectangle with the same frame/anchor logic.
#      A supporting adjustment widens the generated rectangle from width 5 to width 7 so that
#      individual non-frame pieces can visibly change under horizontal flip.
#    - The input still shows the frame in its final orientation, but some scattered non-frame
#      fragments are mirrored horizontally relative to the slot they must occupy in the output.
#    - Small example: if the output needs a 2x3 L-shape occupying {(0,1),(1,0),(1,1)}, the
#      original puzzle can only accept that exact orientation from input. This variant may show
#      the mirrored patch {(0,0),(1,0),(1,1)} in the input, so the solver must test both
#      "as-is" and "flipped" placements against the same hole.
# b) Complexity calibration:
#    - Only horizontal flip is added; rotations are still disallowed.
#    - The generator rejects outputs that do not contain at least two horizontally asymmetric
#      non-frame patches, so the new degree of freedom is visible in examples.
#    - The frame remains unflipped and preserves its original anchoring role, preventing the
#      variant from becoming a full transform-search task.
# c) Hidden assumptions from the original solver:
#    - `_piece_sig` was used as an exact type key; this relied on orientation being fixed.
#      It is replaced by a horizontal-flip canonical key.
#    - `_assemble(... len(places) == cnt ...)` assumed candidate placements for one type were
#      non-overlapping. With flip variants, two candidates can become alternative orientations
#      for the same region, so batch placement is now guarded by an explicit compatibility check.
#    - Frame locking via `req[...] = col` still assumes the frame group is already in final
#      orientation. The generator keeps that invariant.
# d) Placement legality:
#    - Non-frame patches are still scattered outside the blocked frame bbox with 8-neighbor
#      padding, so flipped patches cannot overlap each other or the visible frame.
#    - Flipping is done on the cropped patch array itself, which preserves bounds and color.
# e) Visual observability:
#    - The changed behavior is visible because only horizontally asymmetric pieces are flipped.
#      Symmetric bars/singletons are not counted toward the variant signal.
#    - Each generated sample contains at least one flipped asymmetric patch, usually several.
# f) Visual cleanliness:
#    - The frame is kept intact inside one clear bbox; all flipped fragments remain separated in
#      the surrounding background area, so there is no ambiguous overdraw.
# g) Color inference:
#    - Colors keep their original role as piece identities only. No ordering or mapping depends
#      on color ids; palette assignment remains shuffled.
# h) Reference-task baseline:
#    - `fc7cae8d` resolves orientation by trying transformed candidates against visible controls.
#      This variant uses the same idea at piece level, but the transform set is restricted to
#      `{original, horizontal-flip}` and the controls come from the frame/edge constraints.
# i) New solver assumptions introduced here:
#    - The generator must provide at least one LR-asymmetric non-frame piece that is flipped in
#      the input.
#    - The frame group must stay unflipped so that its locked cells remain a stable anchor.
#    - Adding flip candidates makes some layouts branch, so the solver now uses forced placements
#      first and a small bounded search only when the frame constraints do not fully determine the
#      next move. The generator verifies every accepted sample with `main(input)`.


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _coords_hw(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arr = np.array(coords, dtype=int)
    return int(arr[:, 0].max() + 1), int(arr[:, 1].max() + 1)


def _flip_coords_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _, w = _coords_hw(coords)
    return tuple(sorted((int(r), int(w - 1 - c)) for r, c in coords))


def _canonical_piece_key(col: int, coords: tuple[tuple[int, int], ...]) -> tuple[int, tuple[tuple[int, int], ...]]:
    flipped = _flip_coords_h(coords)
    return int(col), min(coords, flipped)


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

    Heuristics supported by training examples:
      - If there is a color with exactly 4 singleton components, those are the four corners.
      - If there are full horizontal bars of length W-2 in the input, use them as the top/bottom edges
        between the corners (ordered by their vertical position in the input).
      - If there is a color with exactly two full vertical line components (width=1) in the input,
        place them on the left/right edges centered vertically.
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
    start_board = np.full((H, W), -1, dtype=int)
    start_counts = [int(t["cnt"]) for t in types]
    node_budget = 5000
    nodes = 0

    def mirror_ok(board: np.ndarray, r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place_variant(board: np.ndarray, ti: int, variant: dict, r0: int, c0: int) -> bool:
        col = int(types[ti]["col"])
        is_singleton = len(variant["coords"]) == 1
        for dr, dc in variant["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not mirror_ok(board, r, c, col):
                return False
            if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def placements_for(board: np.ndarray, ti: int) -> list[dict]:
        seen: dict[tuple[tuple[int, int], ...], dict] = {}
        for variant in types[ti]["variants"]:
            for r0 in range(H - int(variant["h"]) + 1):
                for c0 in range(W - int(variant["w"]) + 1):
                    if not can_place_variant(board, ti, variant, r0, c0):
                        continue
                    cells = tuple(sorted((r0 + dr, c0 + dc) for dr, dc in variant["coords"]))
                    if cells not in seen:
                        seen[cells] = {"cells": cells}
        return list(seen.values())

    def place_candidate(board: np.ndarray, ti: int, cand: dict) -> np.ndarray:
        out = board.copy()
        col = int(types[ti]["col"])
        for r, c in cand["cells"]:
            out[r, c] = col
        return out

    def batch_compatible(ti: int, places: list[dict]) -> bool:
        occupied = set()
        col = int(types[ti]["col"])
        for cand in places:
            cells = cand["cells"]
            if any(cell in occupied for cell in cells):
                return False
            if disallow_touch and col in disallow_touch:
                for r, c in cells:
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if (rr, cc) in occupied:
                            return False
            occupied.update(cells)
        return True

    def solve(board: np.ndarray, counts: list[int]) -> np.ndarray | None:
        nonlocal nodes
        nodes += 1
        if nodes > node_budget:
            return None
        while True:
            progress = False
            cached: dict[int, list[dict]] = {}
            for ti in range(len(types)):
                if counts[ti] <= 0:
                    continue
                places = placements_for(board, ti)
                if not places:
                    return None
                cached[ti] = places
            if not cached:
                return board

            for ti, places in cached.items():
                if len(places) == 1:
                    counts = counts.copy()
                    counts[ti] -= 1
                    board = place_candidate(board, ti, places[0])
                    progress = True
                    break
                if len(places) == counts[ti] and batch_compatible(ti, places):
                    counts = counts.copy()
                    for cand in places:
                        counts[ti] -= 1
                        board = place_candidate(board, ti, cand)
                    progress = True
                    break
            if not progress:
                break

        if all(c == 0 for c in counts):
            return board

        choice = min(
            (
                (len(placements_for(board, ti)), ti)
                for ti in range(len(types))
                if counts[ti] > 0
            ),
            key=lambda x: (x[0], counts[x[1]], -len(types[x[1]]["variants"][0]["coords"])),
        )
        ti = choice[1]
        for cand in placements_for(board, ti):
            next_counts = counts.copy()
            next_counts[ti] -= 1
            solved = solve(place_candidate(board, ti, cand), next_counts)
            if solved is not None:
                return solved
        return None

    solved = solve(start_board, start_counts)
    assert solved is not None, "assembly dead end"
    assert np.array_equal(solved, solved[:, ::-1]), "assembled board must be LR-symmetric"
    return solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered monochrome fragments into a left-right symmetric rectangle, allowing horizontal mirroring.

    Input structure:
    - The background is the dominant color; all other 4-connected monochrome components are puzzle fragments.
    - A 1-2 piece left-right symmetric frame still appears already aligned as a visible anchor, and some optional corner/edge markers may also be present.
    - Unlike the base task, scattered non-frame fragments may already be horizontally mirrored relative to their final slot.

    Rule:
    - Collect every non-background component and infer the output rectangle size from total area.
    - Preserve the anchored frame/edge requirements, then place each piece either as-is or horizontally flipped.
    - Fill the rectangle so the result is exactly left-right symmetric and same-color fragments remain 4-disconnected.

    Key insight: The frame fixes the target silhouette, while irregular hole shapes determine which asymmetric pieces must be mirrored before they can fit.
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

    ctr = Counter(_canonical_piece_key(col, coords) for col, coords in pieces)
    types = []
    for (col, canon_coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        variant_coords = sorted({canon_coords, _flip_coords_h(canon_coords)})
        variants = []
        for coords in variant_coords:
            h, w = _coords_hw(coords)
            variants.append({"coords": coords, "h": h, "w": w})
        types.append({"col": int(col), "variants": variants, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["variants"][0]["coords"]) for t in types)
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

    def _is_lr_symmetric(parts, h, w):
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
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                frame_parts = [(c1, p1), (c2, p2)]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"

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


# --------------- generator (D12 horizontal-flip variant) ---------------

# === CFG: Output dimensions ===
_AREA_MIN = 35
_AREA_MAX = 63
_WIDTH_REQUIRED = 7
_HEIGHT_MAX = 9

# === CFG: Color and palette ===
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_CENTER_CAP_PROB = 0.5

# === CFG: Piece constraints ===
_MAX_PIECE_AREA = 12
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 10
_PIECE_COUNT_MAX = 26
_MIN_FLIPPABLE_PATCHES = 2
_MIN_FLIPPED_PATCHES = 1
_HFLIP_PROB = 0.5

# === CFG: Canvas / scatter ===
_CANVAS_MIN = 15
_CANVAS_MAX = 18
_SCATTER_PADDING = 1

# === CFG: Retry ===
_GEN_MAX_RETRIES = 300


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


def _zigzag_frame(H, half):
    cells = set()
    start_c = random.randint(0, half - 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "down":
            pass
        elif move == "horiz":
            opts = [cc for cc in (c - 1, c + 1) if 0 <= cc < half]
            if opts:
                nc = random.choice(opts)
                step = 1 if nc > c else -1
                for cc in range(c, nc + step, step):
                    cells.add((r, cc))
                c = nc
        else:
            opts = [cc for cc in (c - 1, c + 1) if 0 <= cc < half]
            if opts:
                c = random.choice(opts)
    return cells, start_c, c


def _norm_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _cells_hflip_distinct(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> bool:
    coords = _norm_cells(cells)
    return coords != _flip_coords_h(coords)


def _pick_chunk(
    remaining: set[tuple[int, int]],
    prefer_asym: bool,
) -> set[tuple[int, int]]:
    best = None
    best_score = None
    tries = 24
    rem_list = list(remaining)
    for _ in range(tries):
        seed = random.choice(rem_list)
        target_max = min(_MAX_CHUNK_SIZE, len(remaining))
        target_min = 2 if target_max >= 2 else 1
        target = random.randint(target_min, target_max)
        chunk = {seed}
        frontier = {seed}
        while frontier and len(chunk) < target:
            nbrs = []
            for r, c in frontier:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (rr, cc) in remaining and (rr, cc) not in chunk:
                        nbrs.append((rr, cc))
            if not nbrs:
                break
            nxt = random.choice(nbrs)
            chunk.add(nxt)
            frontier.add(nxt)
        asym = _cells_hflip_distinct(chunk)
        width2 = len({c for _, c in chunk}) >= 2
        score = (int(asym), int(width2), len(chunk))
        if prefer_asym and asym and width2:
            return chunk
        if best_score is None or score > best_score:
            best = chunk
            best_score = score
    return best if best is not None else {random.choice(rem_list)}


def _choose_fill_color(grid: np.ndarray, cells: list[tuple[int, int]], fill_pool: list[int]) -> int:
    avoid = set()
    H, W = grid.shape
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] >= 0:
                avoid.add(int(grid[rr, cc]))
    opts = [col for col in fill_pool if col not in avoid]
    if not opts:
        opts = fill_pool
    return random.choice(opts)


def _fill_holes(grid, H, W, half, frame1, frame2, split, fill_pool):
    left_mask = grid[:, :half] == -1
    asym_pairs = 0

    for comp in components_mask(left_mask, connectivity=4):
        remaining = {(int(r), int(c)) for r, c in comp}
        while remaining:
            prefer_asym = asym_pairs < _MIN_FLIPPABLE_PATCHES
            chunk = _pick_chunk(remaining, prefer_asym=prefer_asym)
            full_cells = []
            for r, c in chunk:
                full_cells.append((r, c))
                full_cells.append((r, W - 1 - c))
            col = _choose_fill_color(grid, full_cells, fill_pool)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            if _cells_hflip_distinct(chunk):
                asym_pairs += 1
            remaining -= chunk

    center_col = half
    center_cells = [(r, center_col) for r in range(H) if grid[r, center_col] == -1]
    if center_cells:
        current = []
        for cell in center_cells:
            if current and cell[0] != current[-1][0] + 1:
                col = _choose_fill_color(grid, current, fill_pool)
                for r, c in current:
                    grid[r, c] = col
                current = []
            current.append(cell)
            if len(current) == _MAX_CHUNK_SIZE:
                col = _choose_fill_color(grid, current, fill_pool)
                for r, c in current:
                    grid[r, c] = col
                current = []
        if current:
            col = _choose_fill_color(grid, current, fill_pool)
            for r, c in current:
                grid[r, c] = col


def _build_output(H, W):
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

    frame_half, start_c, end_c = _zigzag_frame(H, half)
    split = H // 2 if frame1 != frame2 else H

    grid = np.full((H, W), -1, dtype=int)

    for r, c in frame_half:
        fc = frame1 if r < split else frame2
        grid[r, c] = fc
        grid[r, W - 1 - c] = fc
        if c == half - 1:
            grid[r, half] = fc

    if start_c == half - 1 and random.random() < _CENTER_CAP_PROB:
        grid[0, half] = frame1
    if end_c == half - 1 and random.random() < _CENTER_CAP_PROB:
        grid[H - 1, half] = frame2

    _fill_holes(grid, H, W, half, frame1, frame2, split, fill_pool)

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


def _extract_non_frame_patches(out, bg, fcs):
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


def _is_hflip_distinct_patch(patch: np.ndarray) -> bool:
    return not np.array_equal(patch, patch[:, ::-1])


def _scatter(out: np.ndarray, bg: int, fcs: set[int], patches: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
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

    flippable_idx = [i for i, patch in enumerate(patches) if _is_hflip_distinct_patch(patch)]
    if len(flippable_idx) < _MIN_FLIPPABLE_PATCHES:
        raise ValueError

    chosen = {i for i in flippable_idx if random.random() < _HFLIP_PROB}
    while len(chosen) < _MIN_FLIPPED_PATCHES:
        chosen.add(random.choice(flippable_idx))

    for i, patch in enumerate(patches):
        placed_patch = np.fliplr(patch) if i in chosen else patch
        r, c = random_free_location(
            canvas,
            placed_patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, placed_patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == col
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas, len(chosen), len(flippable_idx)


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> dict | None:
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
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(out)
        if int(col) not in fcs
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None

    patches = _extract_non_frame_patches(out, bg, fcs)
    flippable_count = sum(_is_hflip_distinct_patch(patch) for patch in patches)
    if flippable_count < _MIN_FLIPPABLE_PATCHES:
        return None

    try:
        inp, flipped_count, _ = _scatter(out, bg, fcs, patches)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return {
            "input": inp,
            "output": out,
            "flip_count": int(flipped_count),
            "flippable_count": int(flippable_count),
        }
    return None


def _generate_case() -> dict:
    for _ in range(_GEN_MAX_RETRIES):
        case = _gen_attempt()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid e8686506 input")


def generate_input() -> np.ndarray:
    return _generate_case()["input"]
