# Combo variant of 291dc1e1 (index 36)
# combo_spec: D6/4+D7/2+D12/1
# Name: 重力堆叠+方形输出+角落方向

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for the combo variant
#
# a) Concrete behavior trace
# - Source task recap:
#   1. detect the marked corner where the horizontal and vertical edge lines meet,
#   2. treat those lines as blue/red axes and sort interior solid rectangles by
#      (distance-to-blue, distance-to-red, tie-breaks along the axes),
#   3. rotate every rectangle 90 degrees CCW iff the blue axis is vertical,
#   4. stack top-to-bottom in a tight non-square canvas.
# - Combo behavior:
#   the extraction / sorting / optional rotation stay unchanged, but the corner no
#   longer only tells us where the input axes start. It now also chooses the output
#   gravity direction:
#     tl -> down   (start at the top edge, stack downward)
#     tr -> left   (start at the right edge, stack leftward)
#     br -> up     (start at the bottom edge, stack upward)
#     bl -> right  (start at the left edge, stack rightward)
#   After the direction is known, the ordered rectangles are placed into a square
#   canvas whose side is the larger dimension of the tight directed stack.
# - Small trace:
#   Suppose the sorted rotated sequence is [A 2x6, B 2x2, C 2x4].
#   Original output: tight 6x6 vertical stack A/B/C from top to bottom.
#   Combo with marker in br: direction=up, so the same vertical sequence is pasted
#   from the bottom edge upward, producing top-to-bottom rows C/B/A in a 6x6 square.
#   Combo with marker in tr: direction=left, so the same sequence becomes a
#   horizontal row C|B|A anchored to the right edge inside a square of side
#   max(max_heights=2, sum_widths=12)=12, with background padding above/below.
#   The important fusion is that the same global corner cue controls both the start
#   edge and the axis of stacking, while the square constraint applies after that
#   choice rather than before it.
#
# b) Complexity calibration
# - The input family is intentionally kept close to the source task: two guide lines,
#   one marker, two rectangle bands, solid 4-connected rectangles, occasional
#   minority pixels. This preserves the original sorting clue.
# - The generator rejects samples where the square padding disappears, or where the
#   current directed square rendering matches too many alternative directions. That
#   keeps the D7 and D12 signals visually present.
# - Four direction values are enough; adding more symbols or colors would create an
#   unnecessary second puzzle on top of the original ordering rule.
#
# c) Implicit-assumption audit against the original main()
# - Corner detection still assumes exactly one corner marker differing from its two
#   adjacent line colors. Preserved by construction.
# - Blue/red axis inference still assumes a full row and full column crossing at the
#   marker. Preserved.
# - Rectangle extraction still assumes interior components are solid (no background
#   holes inside their bbox). Preserved by generator.
# - Sorting still depends only on rectangle boxes relative to the blue/red axes, so
#   the original ordering code remains valid.
# - The original output code assumed a vertical top-origin stack and encoded tight
#   dimensions as (sum heights, max widths). That assumption is no longer valid once
#   the corner may request left/right or bottom-origin stacking. The combo therefore
#   computes tight shape from the chosen direction first, then pads to a square.
# - Cross-axis centering was originally always horizontal. In the combo, vertical
#   stacks still center horizontally, but horizontal stacks must instead center
#   vertically.
#
# d) Placement legality
# - In the output, legality is trivial once the tight directed dimensions are
#   computed exactly: cumulative cursor movement uses each rectangle's full extent,
#   and the square side is at least as large as both tight dimensions.
# - In the input, the generator keeps the source task's two separated bands and
#   fixed gaps, so rectangles never overlap or touch 4-neighbor-wise.
# - Whole-grid flips/rotations are used only after a legal canonical layout is
#   created, so transformed instances remain legal.
#
# e) Visual observability
# - Human solvers can still identify the original cue: rectangles closer to one edge
#   line end up earlier in the sequence, and blue-vertical cases still rotate them.
# - The new cue is also observable from a few examples: moving only the corner marker
#   changes which side of the square output acts like the gravity source.
# - Because the generator keeps varying widths and occasional minority pixels, order
#   reversals and horizontal-vs-vertical stacking remain visible to the eye.
#
# f) Visual cleanliness
# - The input still contains only three visually distinct structure types: edge
#   lines, the corner marker, and separated rectangles.
# - No extra instruction glyphs are added inside the crowded interior; the existing
#   corner marker is reused as the global instruction carrier.
# - The output is a single square canvas with one centered directed stack, so there
#   is no overlap between instruction and content.
#
# g) Color predictability
# - No new color remapping is introduced. Colors remain object-local texture.
# - Minority pixels are sparse and only serve to make rotations / order reversals
#   perceptible; they do not encode a separate rule.
#
# h) Quantitative anti-degeneracy check used by the generator
# - padding_area = side^2 - tight_h*tight_w must be > 0, so the square constraint is
#   actually visible.
# - current_diff_count = how many of the other three directed square renderings differ
#   from the current one; we require at least 2.
# - distinct_direction_outputs = number of distinct square renderings across the four
#   directions; we require at least 3.
# - legacy_equal checks equality to the original task's tight top-down output; it
#   must be False.
#
# i) New solver assumptions introduced by the combo
# - The corner-to-direction mapping is fixed and one-to-one; the generator must use
#   all four corners so that the mapping has repeated evidence.
# - Direction observability assumes the sequence is not perfectly symmetric under all
#   four renderings. The generator enforces this with the anti-degeneracy metrics.
# - The square renderer assumes at least one rectangle exists; generator guarantees
#   4-9 rectangles, matching the source-task family.

BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
CORNER_TO_DIRECTION = {
    "tl": "down",
    "tr": "left",
    "br": "up",
    "bl": "right",
}
DIRECTIONS = ("down", "left", "up", "right")


def _corner_key(cr: int, cc: int, H: int, W: int) -> str:
    if int(cr) == 0 and int(cc) == 0:
        return "tl"
    if int(cr) == 0 and int(cc) == int(W - 1):
        return "tr"
    if int(cr) == int(H - 1) and int(cc) == 0:
        return "bl"
    return "br"


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int, int, str]:
    H, W = g.shape
    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None
    cr, cc = map(int, corner)
    return cr, cc, int(g[cr, cc]), _corner_key(cr, cc, H, W)


def _extract_ordered_sequence(grid: np.ndarray) -> tuple[int, str, list[np.ndarray]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cr, cc, marker, corner_key = _locate_corner(g, bg)

    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": sub.copy()})

    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            col = int(edge_idx)
            if int(c0) > col:
                return int(c0 - col - 1)
            if int(c1) < col:
                return int(col - c1 - 1)
            return 0
        row = int(edge_idx)
        if int(r0) > row:
            return int(r0 - row - 1)
        if int(r1) < row:
            return int(row - r1 - 1)
        return 0

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        rect["db"] = int(dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            along_from_corner(rect, blue_orient),
            along_from_corner(rect, red_orient),
        )
    )

    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    return bg, corner_key, seq


def _tight_shape(seq: list[np.ndarray], direction: str) -> tuple[int, int]:
    if direction in {"down", "up"}:
        tight_h = int(sum(int(arr.shape[0]) for arr in seq))
        tight_w = int(max(int(arr.shape[1]) for arr in seq))
    else:
        tight_h = int(max(int(arr.shape[0]) for arr in seq))
        tight_w = int(sum(int(arr.shape[1]) for arr in seq))
    return tight_h, tight_w


def _render_sequence(bg: int, seq: list[np.ndarray], direction: str, *, square: bool) -> np.ndarray:
    tight_h, tight_w = _tight_shape(seq, direction)
    if square:
        side = int(max(tight_h, tight_w))
        out_h = side
        out_w = side
    else:
        out_h = int(tight_h)
        out_w = int(tight_w)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    if direction == "down":
        cursor = 0
        for arr in seq:
            h, w = map(int, arr.shape)
            paste(out, arr, int(cursor), int((out_w - w) // 2))
            cursor += int(h)
    elif direction == "up":
        cursor = int(out_h)
        for arr in seq:
            h, w = map(int, arr.shape)
            cursor -= int(h)
            paste(out, arr, int(cursor), int((out_w - w) // 2))
    elif direction == "right":
        cursor = 0
        for arr in seq:
            h, w = map(int, arr.shape)
            paste(out, arr, int((out_h - h) // 2), int(cursor))
            cursor += int(w)
    else:
        cursor = int(out_w)
        for arr in seq:
            h, w = map(int, arr.shape)
            cursor -= int(w)
            paste(out, arr, int((out_h - h) // 2), int(cursor))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the interior rectangles exactly as in the source task, rotate them when the
    blue axis is vertical, then use the marked corner to choose a gravity direction
    and place the sequence inside a square output canvas.
    """
    bg, corner_key, seq = _extract_ordered_sequence(np.array(grid, dtype=int))
    direction = CORNER_TO_DIRECTION[corner_key]
    return _render_sequence(bg, seq, direction, square=True)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _distribute_band_widths(count: int, used_target: int) -> list[int]:
    extra_units = (used_target - (4 * count - 2)) // 2
    order = np.random.permutation(count)
    units = np.zeros(count, dtype=int)
    remaining = int(extra_units)
    for i, idx in enumerate(order):
        slots_left = count - i - 1
        lo = max(0, remaining - 3 * slots_left)
        hi = min(3, remaining)
        take = int(np.random.randint(lo, hi + 1))
        units[int(idx)] = take
        remaining -= take
    return [2 + 2 * int(u) for u in units]


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.58:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _build_canonical() -> np.ndarray:
    long_dim = int(np.random.randint(9, 29))
    options = []
    for trailing in (1, 2):
        max_used = long_dim - 2 - trailing
        if max_used < 6 or max_used % 2:
            continue
        max_count = min(5, (max_used + 2) // 4)
        pairs = []
        for n0 in range(2, max_count + 1):
            for n1 in range(2, max_count + 1):
                if n0 + n1 > 9:
                    continue
                eligible = []
                if max_used <= 10 * n0 - 2:
                    eligible.append(0)
                if max_used <= 10 * n1 - 2:
                    eligible.append(1)
                if eligible:
                    pairs.append((n0, n1, eligible))
        if pairs:
            options.append((max_used, pairs))

    assert options
    max_used, pairs = options[int(np.random.randint(len(options)))]
    n0, n1, eligible = pairs[int(np.random.randint(len(pairs)))]
    max_band = int(np.random.choice(eligible))
    max0 = min(max_used, 10 * n0 - 2)
    max1 = min(max_used, 10 * n1 - 2)
    used0 = int(max_used if max_band == 0 else np.random.choice(np.arange(4 * n0 - 2, max0 + 1, 2)))
    used1 = int(max_used if max_band == 1 else np.random.choice(np.arange(4 * n1 - 2, max1 + 1, 2)))
    band0 = _distribute_band_widths(n0, used0)
    band1 = _distribute_band_widths(n1, used1)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, int(w)), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, int(r0), int(c))
            c += int(w) + 2

    return grid


def _grid_signature(arr: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, arr.shape)), tuple(map(int, arr.reshape(-1)))


def _behavior_metrics(grid: np.ndarray) -> dict:
    bg, corner_key, seq = _extract_ordered_sequence(np.array(grid, dtype=int))
    direction = CORNER_TO_DIRECTION[corner_key]
    outputs = {d: _render_sequence(bg, seq, d, square=True) for d in DIRECTIONS}
    current = outputs[direction]
    tight_h, tight_w = _tight_shape(seq, direction)
    side = int(max(tight_h, tight_w))
    legacy = _render_sequence(bg, seq, "down", square=False)
    return {
        "direction": direction,
        "rect_count": int(len(seq)),
        "tight_h": int(tight_h),
        "tight_w": int(tight_w),
        "side": side,
        "padding_area": int(side * side - tight_h * tight_w),
        "legacy_equal": bool(np.array_equal(current, legacy)),
        "current_diff_count": int(
            sum(not np.array_equal(current, outputs[d]) for d in DIRECTIONS if d != direction)
        ),
        "distinct_direction_outputs": int(len({_grid_signature(outputs[d]) for d in DIRECTIONS})),
    }


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    metrics = _behavior_metrics(grid)
    if metrics["rect_count"] < 4:
        raise ValueError("too few rectangles")
    if metrics["padding_area"] <= 0:
        raise ValueError("square padding not visible")
    if metrics["legacy_equal"]:
        raise ValueError("combo output collapsed to original output")
    if metrics["current_diff_count"] < 2:
        raise ValueError("corner direction too weak")
    if metrics["distinct_direction_outputs"] < 3:
        raise ValueError("too many direction collisions")

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        try:
            return _gen_attempt()
        except (AssertionError, IndexError, ValueError):
            continue
    raise RuntimeError("failed to generate input for combo variant 291dc1e1/36")
