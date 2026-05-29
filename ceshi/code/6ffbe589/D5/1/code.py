# Variant of 6ffbe589
# Dimension: D5 — 目标对象：板内颜色层结构 (dst)
# Change: 3个同心颜色层（外框、中环、内核），均为方形，共享中心点... → 层数可变：2-5个颜色层

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the D5 variant
# a) Concrete behavior:
#    The board is still the unique largest square cluster that gets cropped out, and each
#    board color still owns exactly one independently rotated layer. The change is that the
#    cropped board no longer always has 3 colors/layers; it now has 2-5 nested square layers
#    that share the same center. The outer L-1 layers are one-cell-thick square rings and the
#    innermost layer is a centered square core. Every layer has a small asymmetric notch, so
#    rotating that color changes the visible orientation of its missing cells.
#    Example trace at the critical step:
#    - Original generator: always build outer/middle/inner for exactly 3 colors, then invert
#      their rotations into the input board.
#    - Variant generator: choose layer_count in [2,5], build sizes N, N-2, ..., N-2*(L-1),
#      assign one color per size, then invert each color's rotation in the same way.
#    - Difference: solver logic is still "for each crop color, inspect same-color cue and
#      rotate that mask", but the crop can now contain a variable number of layers/colors.
# b) Complexity calibration:
#    A fixed three-layer board would collapse back to the original behavior, so the generator
#    samples 2, 3, 4, and 5 layers. The board keeps a single clean nesting scheme rather than
#    mixing unrelated shapes, which avoids ambiguity while still making 4-5 layer cases visibly
#    different from the source task.
# c) Hidden-assumption audit for main():
#    - crop_components_near_largest(...) assumes the board is the unique largest component and
#      cues stay outside the margin-3 expansion. The generator enforces both.
#    - colors_present(crop, exclude={bg}) used to see exactly 3 colors; in the variant it must
#      handle 2-5 colors. This already works because the loop is generic.
#    - outside cue lookup assumes at most one monochrome cue component per board color. The
#      generator creates zero or one cue per color.
#    - rotate_color_layer(...) assumes the crop is square. The generator always builds an NxN
#      board and keeps cues outside the crop.
#    - Per-color rotation order must not create overlaps. The generator places every mask inside
#      its own rotationally invariant annulus or centered core, so rotating one color never moves
#      it into another layer's zone.
# d) Placement legality:
#    The board is pasted flush to one canvas edge like the source task. Cue bboxes must stay at
#    Chebyshev distance >= 4 from the board bbox so the crop helper never swallows them, and cue
#    bboxes stay separated from one another to remain visually distinct.
# e) Visual observability:
#    Humans can infer the variant because some examples will clearly show 2 layers, others 4 or 5.
#    Every layer is nested, colored distinctly, and has an oriented notch, so rotations remain
#    visible. No color-id ordering or invisible metadata is used.
# f) Visual cleanliness:
#    The board is a single centered nesting of square rings/core, so the crop stays tidy. Cues are
#    isolated outside the crop instead of mixing with the board region.
# g) Color inference:
#    Colors do not get remapped. Each cue uses the same color as exactly one board layer, preserving
#    the original same-color matching rule even when layer count increases.
# h) Source-task check:
#    No source_task was supplied, so there is no extra reference solver to inspect.
# i) New assumptions introduced by the variant:
#    The solver now relies on the generator to ensure every visible crop color is a true layer color
#    and not a stray extra object inside the crop. The generator satisfies this by keeping all cues
#    outside the crop bbox and by assigning each board color to exactly one nested layer.


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """
    Local helper (only used by this solver): crop the cluster near the largest non-bg component.

    Returns:
        (crop, crop_bbox, bg) where crop_bbox is in the original grid coordinates (inclusive).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=int(connectivity), monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    largest_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest foreground component"
    mr0, mc0, mr1, mc1 = map(int, largest[0]["bbox"])

    h, w = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(h - 1), int(mr1 + margin))
    ec1 = min(int(w - 1), int(mc1 + margin))

    keep_mask = np.zeros((h, w), dtype=bool)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 < er0 or r0 > er1 or c1 < ec0 or c0 > ec1:
            continue
        coords = comp["coords"]
        keep_mask[coords[:, 0], coords[:, 1]] = True

    keep_coords = np.argwhere(keep_mask)
    assert keep_coords.size > 0, "expected the expanded board crop to retain the largest component"

    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = keep_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    crop[sub] = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1][sub]
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


def rotate_color_layer(grid: np.ndarray, color: int, rot_k: int, *, bg: int | None = None) -> np.ndarray:
    """
    Local helper (only used by this solver): rotate a single color's mask inside a fixed-size grid.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    k = int(rot_k) % 4
    if k == 0:
        return grid

    col = int(color)
    mask = (grid == col).astype(int)
    rmask = np.rot90(mask, k)
    assert rmask.shape == mask.shape, "expected a square crop for layer rotation"

    out = grid.copy()
    out[out == col] = bg
    out[rmask == 1] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and restore each visible color layer using same-color outside cues.

    Input structure:
    - Background is the dominant color.
    - One largest multicolor square board contains 2-5 concentric, center-aligned color layers.
    - Small monochrome cue sprites may appear outside the board, with cue color matching a board layer.

    Rule:
    - Crop the board cluster away from the distant cue sprites.
    - For every non-background color in the crop, count the pixels in its same-color outside cue.
    - Rotate that color's full mask clockwise by 90 degrees times (cue area mod 4).
    - If a board color has no cue, leave that layer unchanged.

    Key insight:
    Cue size is not about where the cue sits; its pixel count modulo 4 encodes the quarter-turn
    needed for the matching color layer, and this works no matter whether the board has 2, 3, 4,
    or 5 nested layers.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    colors = colors_present(crop, exclude={int(bg)})
    assert colors, "expected the cropped board to contain foreground colors"
    out = crop.copy()

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    for col in colors:
        outside_comps = find_components(
            outside_grid,
            bg=int(bg),
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        assert len(outside_comps) <= 1, f"expected at most one outside cue for color {col}"
        cue_size = int(outside_comps[0]["area"]) if outside_comps else 0
        rot_k = (-cue_size) % 4
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


# --------------- generator ---------------

CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
MIN_LAYERS, MAX_LAYERS = 2, 5

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 2

GEN_RETRIES = 400


def _is_rot_asym(mask: np.ndarray) -> bool:
    """True iff mask changes under every non-trivial 90-degree rotation."""
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _embed_local(n: int, inset: int, local: np.ndarray) -> np.ndarray:
    """Embed a local square mask into an n*n board at the requested inset."""
    out = np.zeros((n, n), dtype=bool)
    s = int(local.shape[0])
    out[inset : inset + s, inset : inset + s] = local
    return out


def _ring_local(size: int) -> np.ndarray:
    """Build a one-cell-thick square ring with a small asymmetric notch."""
    local = np.zeros((size, size), dtype=bool)
    local[0, :] = True
    local[-1, :] = True
    local[:, 0] = True
    local[:, -1] = True

    side = random.randrange(4)
    span = list(range(1, size - 1))
    notch_len = 1 if len(span) == 1 else random.randint(1, min(2, len(span)))
    start = random.randint(0, len(span) - notch_len)
    positions = span[start : start + notch_len]
    for pos in positions:
        if side == 0:
            local[0, pos] = False
        elif side == 1:
            local[pos, size - 1] = False
        elif side == 2:
            local[size - 1, pos] = False
        else:
            local[pos, 0] = False

    assert _is_rot_asym(local), "ring layer should be visibly rotatable"
    return local


def _core_local(size: int) -> np.ndarray:
    """Build the centered core square with an asymmetric border notch."""
    local = np.ones((size, size), dtype=bool)
    if size == 2:
        local[random.randrange(2), random.randrange(2)] = False
    else:
        side = random.randrange(4)
        span = list(range(1, size - 1))
        notch_len = 1 if len(span) == 1 else random.randint(1, min(2, len(span)))
        start = random.randint(0, len(span) - notch_len)
        positions = span[start : start + notch_len]
        for pos in positions:
            if side == 0:
                local[0, pos] = False
            elif side == 1:
                local[pos, size - 1] = False
            elif side == 2:
                local[size - 1, pos] = False
            else:
                local[pos, 0] = False

    assert _is_rot_asym(local), "core layer should be visibly rotatable"
    return local


def _make_layer_masks(n: int, layer_count: int) -> list[np.ndarray]:
    """Create 2-5 concentric square layers that partition nested annuli plus the center core."""
    masks: list[np.ndarray] = []
    for idx in range(layer_count):
        size = n - 2 * idx
        inset = idx
        local = _core_local(size) if idx == layer_count - 1 else _ring_local(size)
        masks.append(_embed_local(n, inset, local))

    for i in range(layer_count):
        for j in range(i + 1, layer_count):
            assert not np.any(masks[i] & masks[j]), "concentric layers must stay disjoint"
    return masks


def _pick_rotations(layer_count: int) -> list[int]:
    """Pick per-layer quarter-turn counts, forcing at least one real cue."""
    rot_ks = [random.randrange(4) for _ in range(layer_count)]
    if not any(rot_ks):
        rot_ks[random.randrange(layer_count)] = random.randint(1, 3)
    return rot_ks


def _cue_patch(color: int, area: int) -> np.ndarray:
    """Grow an 8-connected monochrome blob of a requested area, then crop it."""
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < CUE_GRID and 0 <= nc < CUE_GRID and (nr, nc) not in coords:
                        frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    arr = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        arr[r, c] = int(color)
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev gap between two axis-aligned bounding boxes."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    """Place one cue sprite while respecting board/cue spacing constraints."""
    h, w = patch.shape
    mask = patch != 0
    candidates = []
    for r in range(CANVAS_SIZE - h + 1):
        for c in range(CANVAS_SIZE - w + 1):
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            bb = (r, c, r + h - 1, c + w - 1)
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            candidates.append((r, c, bb))
    if not candidates:
        return False

    r, c, bb = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    cue_bboxes.append(bb)
    return True


def _board_anchor(n: int) -> tuple[int, int]:
    """Place the board flush to one edge so distant cue space remains available."""
    spare = CANVAS_SIZE - n
    side = random.randrange(4)
    if side == 0:
        return spare, random.randint(0, spare)
    if side == 1:
        return 0, random.randint(0, spare)
    if side == 2:
        return random.randint(0, spare), spare
    return random.randint(0, spare), 0


def _cue_area_for_rotation(rot_k: int) -> int:
    """Choose a cue area in 1..5 whose modulo-4 code asks for rot_k."""
    mod = (-int(rot_k)) % 4
    assert mod != 0
    if mod == 1:
        return random.choice([1, 5])
    return mod


def _choose_board_size(layer_count: int) -> int:
    """Pick a square board size that can fit the requested number of concentric layers."""
    valid = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if n - 2 * (layer_count - 1) >= 2]
    assert valid, "expected at least one valid board size"
    return random.choice(valid)


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt."""
    layer_count = random.randint(MIN_LAYERS, MAX_LAYERS)
    n = _choose_board_size(layer_count)
    colors = random.sample(range(1, 10), layer_count)
    rot_ks = _pick_rotations(layer_count)

    masks = _make_layer_masks(n, layer_count)

    out_board = np.zeros((n, n), dtype=int)
    for color, mask in zip(colors, masks):
        out_board[mask] = int(color)

    in_board = np.zeros((n, n), dtype=int)
    for color, mask, rot_k in zip(colors, masks, rot_ks):
        inv = np.rot90(mask, (-int(rot_k)) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(color)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [(int(color), _cue_area_for_rotation(rot_k)) for color, rot_k in zip(colors, rot_ks) if rot_k != 0]
    random.shuffle(cue_items)
    for color, area in cue_items:
        if not _place_cue(canvas, _cue_patch(color, area), board_bbox, cue_bboxes):
            return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None
    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 D5 input")
