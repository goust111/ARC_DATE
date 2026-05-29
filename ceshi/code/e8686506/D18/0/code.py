# Variant of e8686506
# Dimension: D18 — 拼合矩形的内部结构模式 (meta)
# Change: 拼合后大碎片（框架）在外围、小碎片填入凹陷和边角处，形成分层嵌套的矩形... → 棋盘格式交替排列

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D18 variant
#
# variant_spec:
#   core_change:
#     original: outer zigzag frame is preserved as the anchored large piece(s), and
#       smaller fragments fill the frame's interior holes and corners.
#     variant: roughly half of the output cells belong to a checkerboard anchor mask
#       that is already grouped in the input, while the complementary checkerboard
#       cells are scattered as detached singleton pieces. The assembled rectangle is
#       therefore a regular alternating tiling instead of a nested frame-with-holes.
#   concrete_pixel_behavior:
#     output:
#       shape: same portrait dense rectangle family as the source task, still forced
#         by total piece area and left-right symmetry.
#       pattern:
#         - choose one checkerboard parity as the anchored parity
#         - all cells on that parity are colored by one anchor color, or by two
#           anchor colors split into top/bottom row bands
#         - every complementary parity cell uses one fill color
#       visual_effect: the whole rectangle alternates cell-by-cell, so the interior
#         no longer reads as "outline + cavities"; it reads as a tiled board.
#     input:
#       - the anchored parity remains grouped inside one HxW bounding box using the
#         exact output-relative positions, which creates 1-2 large 8-connected clue
#         objects made of diagonally linked checkerboard cells
#       - the complementary parity is broken into scattered singleton fragments
#   behavior_trace:
#     old_flow:
#       1. detect 1-2 8-connected frame components whose union bbox is HxW
#       2. lock those frame cells into req
#       3. fill irregular interior holes using remaining pieces
#     new_flow:
#       1. detect 1-2 8-connected checkerboard anchor components whose union bbox is HxW
#       2. lock those alternating anchor cells into req
#       3. fill every remaining cell with the scattered complementary-color singletons
#     difference:
#       the locked clue cells now occupy the whole rectangle in an alternating way,
#       including many interior cells, rather than mostly tracing the perimeter.
#   complexity_calibration:
#     - keeping the source task's HxW factor logic and LR symmetry preserves enough
#       structure that the puzzle still feels like "assemble the rectangle".
#     - using 1-2 anchor colors gives a small number of repeated clue categories,
#       which is human-discernible from 2-3 examples.
#     - scattering only the complementary parity avoids degenerate "crop the existing
#       object" behavior while staying generator-friendly.
#   implicit_assumption_audit:
#     - _assemble.can_place() previously forbade singleton placement away from the
#       border. That relied on the old frame task, where singleton pieces only served
#       as corner/edge decorations. In the checkerboard variant, interior singleton
#       cells are the normal fill pieces, so that restriction must be removed.
#     - _build_requirements() contains old corner/bar/line heuristics. They are left
#       in place because the new generator never creates those signatures, so they do
#       not interfere; the real constraints come from the detected checkerboard anchor.
#     - the old "frame" search actually only assumes there are 1-2 LR-symmetric
#       8-connected parts whose union bbox equals the output bbox. This still holds
#       for diagonally linked checkerboard masks, so the mechanism is reusable.
#   placement_legality:
#     - the anchor mask is stamped into a reserved HxW bbox on the input canvas.
#     - all scattered singleton fill pieces are placed outside that bbox with
#       Chebyshev padding 1, so they do not overlap or accidentally merge.
#     - if two anchor colors are used, each row band has height >= 2 so each anchor
#       color forms a single 8-connected checker mask rather than splitting apart.
#   visual_observability:
#     - a human sees a spotted rectangle already occupying alternating cells and many
#       loose single pixels of one color elsewhere. The output is the obvious dense
#       completion of that alternating pattern.
#     - this is visibly different from the source task because the preserved grouped
#       clue is interior-heavy: many clue cells lie strictly inside the bbox.
#   neatness:
#     - only two visual roles exist in the input: the checker anchor bbox and the
#       scattered complementary singletons. They do not overlap, so the image stays
#       clean and readable.
#   color_inference:
#     - color mapping is purely structural: anchor-color cells stay where the grouped
#       checker clue already places them, and the one scattered fill color occupies
#       the complementary alternating positions. No hidden color-id ordering is used.
#   solver_generator_contract:
#     - main() now assumes that at least one 8-connected LR-symmetric checker anchor
#       component (or a pair of row-band components) spans the output bbox.
#     - generate_input() guarantees exactly that by constructing the output first,
#       preserving the anchor parity in-place, and scattering only the complement.
# -----------------------------------------------------------------------------


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

    Retained source-task heuristics:
      - If there is a color with exactly 4 singleton components, those are the corners.
      - If there are full horizontal bars of length W-2 in the input, use them as the
        top/bottom edges between the corners (ordered by input row).
      - If there is a color with exactly two identical full vertical lines in the input,
        place them on the left/right edges centered vertically.

    The D18 checkerboard generator does not emit those signatures, so the effective
    constraints for this variant come from the detected checkerboard anchor parts that
    are written into req later in main().
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
    board = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not mirror_ok(r, c, col):
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
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a left-right symmetric rectangle whose
    interior follows an alternating checkerboard layout.

    Input structure:
    - bg = most common color; the non-bg cells form puzzle fragments.
    - One 8-connected checkerboard anchor component, or two row-band anchor components,
      already preserves its HxW relative positions inside the input.
    - The complementary checkerboard cells appear as detached singleton fragments
      scattered elsewhere on the canvas.

    Rule:
    - Collect all non-bg components as puzzle pieces and infer the output rectangle
      dimensions from total piece area.
    - Find the 1-2 LR-symmetric 8-connected anchor parts whose union bounding box is
      exactly the output size, and lock their alternating cells into the output.
    - Fill the remaining cells with the scattered complementary pieces so the result is
      an exact left-right symmetric checkerboard-style tiling.

    Key insight: the grouped anchor mask already reveals half of the final alternating
    pattern across the whole rectangle, so once those cells are fixed, the remaining
    singleton fragments can only occupy the complementary checkerboard positions.
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

    def _is_lr_symmetric(parts, h, w):
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    anchor_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and _is_lr_symmetric([(c1, p1)], H, W):
            anchor_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                anchor_parts = [(c1, p1), (c2, p2)]
                break
        if anchor_parts:
            break
    assert anchor_parts, "input must contain 1-2 LR-symmetric checker anchor parts spanning the output bbox"

    req = _build_requirements(H, W, comps_by_color)

    all_anchor_pts = np.concatenate([p for _, p in anchor_parts])
    ar0, ac0 = int(all_anchor_pts[:, 0].min()), int(all_anchor_pts[:, 1].min())
    for col, pts in anchor_parts:
        for r, c in pts:
            or_, oc = int(r) - ar0, int(c) - ac0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = col

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int)


# --------------- generator (D18 checkerboard variant) ---------------

# Output dimensions stay in the original family.
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# Palette / pattern choices.
_TWO_ANCHOR_PROB = 0.55

# Canvas / scatter.
_CANVAS_MIN = 18
_CANVAS_MAX = 22
_SCATTER_PADDING = 1

# Retry.
_GEN_MAX_RETRIES = 300


def _valid_dims() -> list[tuple[int, int]]:
    """
    Keep the source task's observed output size family.

    D18 changes the internal tiling only; it should not silently broaden the size
    distribution to new portrait factors that never appeared in the source task.
    """
    return [(5, 5), (8, 5)]


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _choose_anchor_bands(H: int) -> list[tuple[int, int]]:
    use_two = H >= 4 and random.random() < _TWO_ANCHOR_PROB
    if not use_two:
        return [(0, H)]
    splits = [s for s in range(2, H - 1) if s >= 2 and (H - s) >= 2]
    if not splits:
        return [(0, H)]
    split = random.choice(splits)
    return [(0, split), (split, H)]


def _build_output(H: int, W: int) -> tuple[np.ndarray, set[int]]:
    """
    Build an abstract LR-symmetric checkerboard output.

    One checkerboard parity is the anchor parity and may use one or two row-band
    colors. The complementary parity uses a single fill color so the scattered
    singleton pieces have a unique destination class.
    """
    anchor_parity = random.randint(0, 1)
    bands = _choose_anchor_bands(H)
    fill_color = 0
    anchor_colors = list(range(1, 1 + len(bands)))

    grid = np.full((H, W), -1, dtype=int)
    for r in range(H):
        anchor_col = None
        for i, (r0, r1) in enumerate(bands):
            if r0 <= r < r1:
                anchor_col = anchor_colors[i]
                break
        assert anchor_col is not None
        for c in range(W):
            if (r + c) % 2 == anchor_parity:
                grid[r, c] = anchor_col
            else:
                grid[r, c] = fill_color

    assert np.array_equal(grid, grid[:, ::-1])
    return grid, set(anchor_colors)


def _make_output() -> tuple[np.ndarray, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_anchor_colors = _build_output(H, W)
    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    cmap = {a: available[i] for i, a in enumerate(abstract)}
    real_anchor_colors = {cmap[a] for a in abstract_anchor_colors}
    return np.vectorize(lambda x: cmap[int(x)])(out), bg, real_anchor_colors


def _extract_fill_patches(out: np.ndarray, bg: int, anchor_colors: set[int]) -> list[np.ndarray]:
    """Extract the complementary singleton cells that must be scattered."""
    patches = []
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            col = int(out[r, c])
            if col in anchor_colors:
                continue
            patch = np.array([[col]], dtype=int)
            patches.append(patch)
    random.shuffle(patches)
    return patches


def _scatter(out: np.ndarray, bg: int, anchor_colors: set[int]) -> np.ndarray:
    """
    Place the checker anchor as a grouped spotted bbox and scatter the complementary
    singleton pieces elsewhere on the canvas.
    """
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

    for patch in _extract_fill_patches(out, bg, anchor_colors):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    anchor_mask = np.isin(out, list(anchor_colors))
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[anchor_mask] = out[anchor_mask]
    return canvas


def _is_checkerboard_variant(out: np.ndarray) -> bool:
    """Variant-specific metric: every 4-neighbor pair must use different colors."""
    return bool(np.all(out[:, :-1] != out[:, 1:]) and np.all(out[:-1, :] != out[1:, :]))


def _anchor_interior_ratio(inp: np.ndarray, bg: int) -> float:
    """Measure how interior-heavy the grouped anchor clue is inside its own bbox."""
    comps = []
    for col in sorted(set(map(int, np.unique(inp))) - {bg}):
        comps.extend((col, pts) for pts in components_mask(inp == col, connectivity=8))
    best = None
    best_area = -1
    for col, pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area > best_area:
            best_area = area
            best = (col, pts)
    if best is None:
        return 0.0
    _, pts = best
    r0, c0, r1, c1 = bbox_coords(pts)
    cells = {(int(r), int(c)) for r, c in pts}
    interior_total = 0
    interior_anchor = 0
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            interior_total += 1
            if (r, c) in cells:
                interior_anchor += 1
    return 0.0 if interior_total == 0 else interior_anchor / interior_total


def _gen_attempt() -> np.ndarray | None:
    out, bg, anchor_colors = _make_output()
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if not _is_checkerboard_variant(out):
        return None
    try:
        inp = _scatter(out, bg, anchor_colors)
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
    raise RuntimeError("failed to generate a valid e8686506 D18 input")
