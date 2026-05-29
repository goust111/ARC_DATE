# Variant of 291dc1e1
# Dimension: D6 — 堆叠方式与对齐 (dst)
# Change: 矩形从上到下紧密堆叠，水平方向居中对齐，无间隙... → 水平排列（从左到右）替代垂直堆叠
#
# Design analysis
# - Source-task baseline:
#   The prompt cites b5ca7ac4 for its horizontal stacking feel. This workspace does
#   not contain a local solver file for b5ca7ac4, so the only available reference is
#   its official evaluation I/O JSON. Those examples are enough to confirm the
#   relevant visual baseline: objects are composed left-to-right in the output rather
#   than top-to-bottom. No other mechanism from that task is imported here.
# - a) Concrete behavior and trace:
#   Input still contains one marked corner, one horizontal edge line, one vertical
#   edge line, and several solid interior rectangles. The solver still ranks those
#   rectangles by (distance to blue edge, distance to red edge), and still rotates
#   every rectangle 90 degrees CCW when the blue edge is vertical.
#   The only behavioral change is the final packing stage.
#   Example trace:
#   Suppose the sorted rectangle sequence after any needed rotation is
#   A: 2x4, B: 2x2, C: 2x6 with background 8.
#   Original task:
#     outW = max(4, 2, 6) = 6
#     outH = 2 + 2 + 2 = 6
#     paste A at row 0, centered; B at row 2, centered; C at row 4, centered.
#   This variant:
#     outH = max(2, 2, 2) = 2
#     outW = 4 + 2 + 6 = 12
#     paste A at col 0, vertically centered; B at col 4; C at col 6.
#   The ranking and rotation are unchanged; only the output canvas dimensions and
#   paste coordinates switch from vertical accumulation to horizontal accumulation.
# - b) Complexity calibration:
#   The input generator keeps 4-8 rectangles, spread across two interior bands, with
#   varied widths and occasional minority-colored cells. That is enough repetition for
#   a human to see that ordering is unchanged while composition direction has flipped.
#   The design avoids adding extra grouping or spacing rules, so D6 remains the sole
#   conceptual change.
# - c) Implicit-assumption review of the original main():
#   The corner detection, blue/red line inference, line masking, rectangle extraction,
#   distance computation, sort key, and optional rotation all remain valid because
#   they depend only on the input geometry, not on the old output layout.
#   The only lines that relied on the old D6 were:
#     outW = max(widths), outH = sum(heights), out_r accumulation, horizontal centering.
#   Those assumptions are false in the variant, so they are replaced by:
#     outH = max(heights), outW = sum(widths), out_c accumulation, vertical centering.
#   A subtle old assumption was that "stack index" always mapped to row offsets.
#   Under the variant, index maps to column offsets instead; failing to change that
#   would silently recreate the original task.
# - d) Placement legality:
#   Generator rectangles remain disjoint because each band uses a running column with
#   a fixed 2-cell gap, and the two bands live on separate rows in the canonical
#   template. Transformations only mirror/rotate the whole grid, preserving
#   non-overlap and border validity. Output placement is also safe because the canvas
#   is sized exactly as max-height by summed-widths for the post-rotation sequence.
# - e) Visual observability:
#   With at least four rectangles, the output becomes a single horizontal strip (or a
#   short strip with varied heights after rotation) instead of a tall vertical stack.
#   This is immediately visible across 2-3 samples, even when input structure is
#   unchanged. Occasional asymmetric minority pixels ensure the blue-vertical rotation
#   branch is also human-observable.
# - f) Visual neatness:
#   The input keeps large background gaps and two clean bands, so objects do not blur
#   together. The output uses tight side-by-side placement with no overlaps and only
#   one alignment rule (vertical centering), which is easy to parse visually.
# - g) Color-change inference:
#   No new color mapping is introduced. Colors remain object-internal decoration only,
#   so the variant does not burden the solver with extra color inference.
# - Post-change assumption audit:
#   The new main() assumes there is at least one rectangle and that post-rotation
#   rectangles can be arranged left-to-right on a max-height canvas. The generator
#   guarantees this by always producing 4-8 rectangles and validating that the new
#   horizontal layout differs from the old vertical layout on the same input.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


def _ordered_sequence(grid: np.ndarray) -> tuple[int, str, list[np.ndarray]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

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
    marker = int(g[cr, cc])

    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
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
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, b)), "arr": sub.copy()})

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

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    for rct in rects:
        rct["db"] = int(dist_to_edge(rct, blue_orient, blue_edge_idx))
        rct["dr"] = int(dist_to_edge(rct, red_orient, red_edge_idx))

    rects.sort(
        key=lambda r: (
            int(r["db"]),
            int(r["dr"]),
            along_from_corner(r, blue_orient),
            along_from_corner(r, red_orient),
        )
    )

    seq = []
    for rct in rects:
        arr = rct["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    return int(bg), str(blue_orient), seq


def _assemble(seq: list[np.ndarray], bg: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        out_h = int(max(int(a.shape[0]) for a in seq))
        out_w = int(sum(int(a.shape[1]) for a in seq))
        out = np.full((out_h, out_w), int(bg), dtype=int)
        out_c = 0
        for arr in seq:
            h, w = map(int, arr.shape)
            out_r = int((out_h - h) // 2)
            paste(out, arr, int(out_r), int(out_c))
            out_c += int(w)
        return out

    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder rectangles by proximity to the two edge lines, optionally rotate them,
    then place them left-to-right with vertical centering.
    """
    bg, _blue_orient, seq = _ordered_sequence(grid)
    return _assemble(seq, bg, horizontal=True)


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


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
    if area < 8 or np.random.rand() < 0.6:
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
        for a in range(2, max_count + 1):
            for b in range(2, max_count + 1):
                if a + b > 9:
                    continue
                eligible = []
                if max_used <= 10 * a - 2:
                    eligible.append(0)
                if max_used <= 10 * b - 2:
                    eligible.append(1)
                if eligible:
                    pairs.append((a, b, eligible))
        if pairs:
            options.append((max_used, pairs))

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
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, r0, c)
            c += w + 2

    return grid


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def _variant_visible(grid: np.ndarray) -> bool:
    bg, _blue_orient, seq = _ordered_sequence(grid)
    if len(seq) < 2:
        return False
    horizontal = _assemble(seq, bg, horizontal=True)
    vertical = _assemble(seq, bg, horizontal=False)
    return horizontal.shape != vertical.shape or not np.array_equal(horizontal, vertical)


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        try:
            _ = main(grid)
        except Exception:
            continue
        if _variant_visible(grid):
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D6 variant")
