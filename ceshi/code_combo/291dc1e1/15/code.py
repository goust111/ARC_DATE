# Combo variant of 291dc1e1 (index 15)
# combo_spec: D6/4+D12/1
# Name: 重力堆叠+角落编码方向

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for the combo variant
# ====================================
#
# a) Concretization and behavior trace
# - Base task: find the marked corner where a horizontal and a vertical edge line
#   meet; extract all solid interior rectangles; sort them by
#   (distance to blue edge, distance to red edge, tie-breakers along the axes);
#   rotate every rectangle 90 deg CCW iff the blue edge is vertical; then stack.
# - D6/4 changes only the destination cursor: instead of always starting at the
#   top, stacking uses a gravity-like anchor and starts from one edge.
# - D12/1 gives the corner a second meaning: the marker corner no longer only
#   identifies the coordinate system, it also selects which output edge acts as
#   the gravity source. We use a 4-way mapping with one direction per corner:
#     tl -> top to bottom
#     tr -> right to left
#     bl -> left to right
#     br -> bottom to top
# - Concrete trace:
#   Suppose the sorted sequence after any needed rotation is [A(2x6), B(2x2),
#   C(2x4)] and the marker sits at bottom-right. The original task would output
#   A/B/C from top to bottom. Pure D6/4 would output C/B/A from top to bottom by
#   placing A on the bottom first. This combo instead reads bottom-right as the
#   "bottom-origin" code, so A is placed flush to the bottom, then B above it,
#   then C above that. If the marker were bottom-left, the same sorted sequence
#   would be laid out horizontally from left to right instead.
#
# b) Complexity calibration
# - Only four directions exist, one per corner. That is enough to create a real
#   symbolic code without adding a large legend vocabulary.
# - The interior structure remains the original two-band rectangle scene, so the
#   reasoning burden is still focused on sorting + one global direction code.
# - Generator rejection ensures the chosen direction materially changes the
#   output; otherwise the corner code would be visually toothless.
#
# c) Implicit assumption audit against the original main()
# - Corner finding still assumes a unique non-background corner pixel distinct
#   from its adjacent line cells. The generator preserves that exact geometry.
# - Blue/red line identification still assumes full edge lines meeting at that
#   corner. Preserved.
# - Rectangle extraction in the original code masked out every cell equal to the
#   marker color. That was only safe because the marker color 0 never appeared
#   inside rectangles. The combo solver removes only the actual corner cell by
#   index, so future accent choices cannot silently delete object pixels.
# - The original output builder assumed a vertical stack with width=max(widths)
#   and height=sum(heights). That assumption breaks for left/right directions, so
#   the combo layout computes dimensions from the chosen axis instead.
# - The original blue-orientation test still controls the CCW rotation step. The
#   combo does not change that; it only changes where the rotated sequence lands.
#
# d) Placement legality
# - Vertical layouts use height=sum(h_i), width=max(w_i); horizontal layouts use
#   width=sum(w_i), height=max(h_i). Each cursor moves monotonically by the full
#   object extent along the active axis, so overlaps are impossible.
# - The generator reuses the original legal two-band placement logic, then only
#   applies whole-grid flips/rotations, which preserve non-overlap and bounds.
#
# e) Visual observability
# - Human solvers can compare examples and see that the same sorted set of
#   rectangles sometimes stacks vertically and sometimes horizontally, with the
#   start edge determined by which corner contains the marker.
# - Because the generator keeps 4-9 rectangles and rejects direction-degenerate
#   samples, the four direction codes produce visibly different outputs.
#
# f) Visual cleanliness
# - No extra glyphs are introduced. The single corner marker remains the only
#   instruction carrier, and the rest of the grid keeps the clean edge-line plus
#   separated-rectangles structure of the base task.
#
# g) Color inferability
# - There is no new color mapping. Corner position, not color identity, encodes
#   the new behavior. This keeps the symbolic burden small and spatially local.
#
# h) New solver assumptions and generator guarantees
# - Solver assumption: the marker occupies one of the four corners and uniquely
#   identifies the edge intersection. Generator guarantee: canonical scene writes
#   exactly one marker at the edge junction before any global transform.
# - Solver assumption: the chosen corner direction must lead to a distinctive
#   output often enough for induction. Generator guarantee: it rejects any sample
#   whose chosen direction output matches an alternative direction layout.
# - Solver assumption: rectangles are solid (no background holes inside bbox).
#   Generator guarantee: every object is a filled 2xW rectangle with optional
#   recolored accent pixels but no background holes.

BG = 8
BLUE = 1
RED = 2
MARKER = 0

DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

CORNER_TO_DIRECTION = {
    "tl": "down",
    "tr": "left",
    "bl": "right",
    "br": "up",
}

ALL_DIRECTIONS = ("down", "up", "right", "left")


def _corner_name(corner: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = map(int, corner)
    H, W = map(int, shape)
    if r == 0 and c == 0:
        return "tl"
    if r == 0 and c == W - 1:
        return "tr"
    if r == H - 1 and c == 0:
        return "bl"
    assert r == H - 1 and c == W - 1
    return "br"


def _find_corner(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, W = map(int, g.shape)
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    for r, c in corners:
        val = int(g[r, c])
        if val == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        else:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        else:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != val for n in nbr):
            return int(r), int(c)
    raise AssertionError("corner marker not found")


def _identify_axes(g: np.ndarray, corner: tuple[int, int]) -> tuple[str, str, int, int, int]:
    H, W = map(int, g.shape)
    cr, cc = map(int, corner)
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
    return blue_orient, red_orient, int(cr), int(cc), int(g[cr, cc])


def _extract_sorted_sequence(grid: np.ndarray) -> tuple[list[np.ndarray], int, str, str]:
    g = np.array(grid, dtype=int)
    H, W = map(int, g.shape)
    bg = int(get_bg(g))

    corner = _find_corner(g, bg)
    corner_name = _corner_name(corner, g.shape)
    blue_orient, red_orient, cr, cc, marker = _identify_axes(g, corner)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask)
    mask[cr, cc] = False
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": sub.copy()})

    assert rects and marker == MARKER

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            if int(c0) > int(edge_idx):
                return int(c0 - edge_idx - 1)
            if int(c1) < int(edge_idx):
                return int(edge_idx - c1 - 1)
            return 0
        if int(r0) > int(edge_idx):
            return int(r0 - edge_idx - 1)
        if int(r1) < int(edge_idx):
            return int(edge_idx - r1 - 1)
        return 0

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

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

    return seq, bg, corner_name, blue_orient


def _layout_sequence(seq: list[np.ndarray], bg: int, direction: str) -> np.ndarray:
    if direction in {"down", "up"}:
        out_h = int(sum(int(arr.shape[0]) for arr in seq))
        out_w = int(max(int(arr.shape[1]) for arr in seq))
        out = np.full((out_h, out_w), int(bg), dtype=int)
        if direction == "down":
            out_r = 0
            for arr in seq:
                h, w = map(int, arr.shape)
                paste(out, arr, int(out_r), int((out_w - w) // 2))
                out_r += int(h)
        else:
            out_r = int(out_h)
            for arr in seq:
                h, w = map(int, arr.shape)
                out_r -= int(h)
                paste(out, arr, int(out_r), int((out_w - w) // 2))
        return out

    out_h = int(max(int(arr.shape[0]) for arr in seq))
    out_w = int(sum(int(arr.shape[1]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    if direction == "right":
        out_c = 0
        for arr in seq:
            h, w = map(int, arr.shape)
            paste(out, arr, int((out_h - h) // 2), int(out_c))
            out_c += int(w)
    else:
        out_c = int(out_w)
        for arr in seq:
            h, w = map(int, arr.shape)
            out_c -= int(w)
            paste(out, arr, int((out_h - h) // 2), int(out_c))
    return out


def _solve_with_direction(grid: np.ndarray, direction: str | None = None) -> np.ndarray:
    seq, bg, corner_name, _blue_orient = _extract_sorted_sequence(grid)
    actual_direction = CORNER_TO_DIRECTION[corner_name] if direction is None else str(direction)
    return _layout_sequence(seq, bg, actual_direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder rectangles by proximity to the two edge axes, rotate them when the
    blue edge is vertical, then use the marker corner as a 4-way gravity code.
    """
    return _solve_with_direction(grid)


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
    if area < 8 or np.random.rand() < 0.45:
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
            patch = np.full((2, w), color, dtype=int)
            paste(grid, _sprinkle_minorities(patch, color), int(r0), int(c))
            c += int(w + 2)

    return grid


def _count_rectangles(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    H, W = map(int, g.shape)
    bg = int(get_bg(g))
    cr, cc = _find_corner(g, bg)
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True
    mask = (g != int(bg)) & (~line_mask)
    mask[cr, cc] = False
    count = 0
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        sub = crop_bbox(g, bbox_coords(coords))
        if bool((sub == int(bg)).any()):
            continue
        count += 1
    return int(count)


def _direction_uniqueness_score(grid: np.ndarray) -> int:
    chosen = main(grid)
    score = 0
    for direction in ALL_DIRECTIONS:
        alt = _solve_with_direction(grid, direction)
        if not np.array_equal(chosen, alt):
            score += 1
    return int(score)


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    assert _count_rectangles(grid) >= 4
    chosen = main(grid)
    original = _solve_with_direction(grid, "down")
    variants = {direction: _solve_with_direction(grid, direction) for direction in ALL_DIRECTIONS}

    if sum(np.array_equal(chosen, alt) for alt in variants.values()) != 1:
        raise ValueError("direction code is not unique on this sample")
    if np.array_equal(chosen, original) and CORNER_TO_DIRECTION[corner] != "down":
        raise ValueError("corner-coded direction collapsed to original top-down")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        try:
            return _gen_attempt()
        except (AssertionError, IndexError, ValueError):
            continue
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant")
