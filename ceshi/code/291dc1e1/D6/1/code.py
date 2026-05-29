# Variant of 291dc1e1
# Dimension: D6 — 堆叠方式与对齐 (dst)
# Change: 矩形从上到下紧密堆叠，水平方向居中对齐，无间隙... → 右对齐堆叠

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the D6 variant
# ----------------------------------
# 2) Source task:
# - No source_task is provided, so there is no extra reference solver to inspect.
#
# 3a) Concretization and behavior trace:
# - The input still contains one marked corner, one horizontal edge line, one vertical edge line,
#   and several solid rectangular components in the interior.
# - The solver still extracts rectangles, sorts them by distance to the blue edge and then the red
#   edge, and rotates them 90 degrees CCW if the blue edge is vertical.
# - The only intended rule change is in the final placement step of the output stack:
#   original: each rectangle is centered within the output width.
#   variant: each rectangle is pasted so its rightmost column touches the output's right boundary.
# - Small trace:
#   suppose sorted rectangles have final widths [6, 2, 4], so outW = 6.
#   original centers them at columns [0, 2, 1].
#   variant places them at columns [0, 4, 2].
#   The ordering and rotations are identical; only horizontal alignment differs.
#
# 3b) Complexity calibration:
# - A pure solver-side one-line change would be too weak if every generated rectangle ended up with
#   the same final width, because then centered and right-aligned outputs would coincide.
# - The generator therefore keeps the original concept (two distance bands near the edge lines) but
#   now varies both rectangle widths and heights. This preserves the core task while ensuring that
#   the final stacked widths differ in both blue-orientation branches.
#
# 3c) Hidden assumptions review of main():
# - Corner detection assumes exactly one non-background corner marker whose neighbors along the two
#   border lines have different colors. The generator still constructs that.
# - Line masking removes the full marker row and column. Therefore no rectangle may touch either
#   axis line. The generator leaves at least one cell gap before every lane and rectangle.
# - Rectangle extraction assumes each component fills its bbox without background holes. The
#   generator only creates filled rectangles with optional recolored cells, never holes.
# - Sorting assumes distance-to-edge remains meaningful under the transformed corner/orientation.
#   The same canonical layout is rotated/flipped, so this stays valid.
# - Final placement previously assumed centering; the variant replaces that with right alignment,
#   which only depends on outW >= w. That always holds by construction of outW = max width.
#
# 3d) Placement legality:
# - Canonical generation places rectangles in two horizontal lanes separated by at least one blank
#   row and with two blank columns between neighboring rectangles. This prevents overlap and keeps
#   component extraction unambiguous.
# - After rotating/flipping the whole grid, non-overlap and in-bounds placement remain true.
#
# 3e) Visual observability:
# - For blue horizontal, output widths equal input rectangle widths.
# - For blue vertical, rectangles rotate, so output widths equal input rectangle heights.
# - The generator rejects candidates unless the final widths under the chosen branch are not all
#   equal. Therefore at least one stacked rectangle visibly shifts relative to a centered baseline.
#
# 3f) Visual cleanliness:
# - The input still has only four visual element types: background, two axis lines, one corner
#   marker, and separated rectangles. No overlays or ambiguous touching cases are introduced.
#
# 3g) Color-change inference:
# - Colors are unchanged as a rule dimension. Minority cells remain sparse decoration inside
#   rectangles, as in the original generator style, and do not encode new logic.
#
# 8) Variant-behavior metric to validate after implementation:
# - For a generated sample, compute the widths of the rectangles after the solver's rotation branch.
# - The D6 behavior is exercised iff at least one final width is strictly smaller than max(final
#   widths), because only then right-aligned output differs from centered output on the same input.
# - We also track blue_orient in generated samples to ensure both rotation branches appear.
#
# 9) New solver assumptions introduced by the variant:
# - The solver itself adds no new structural assumptions beyond outW >= w for each rectangle.
# - The generator must guarantee D6 visibility by producing width variation after the chosen branch.
#   It does so by rejecting any candidate whose final widths are all equal.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines.

    Variant rule:
    - Sort rectangles by (distance to blue edge, distance to red edge).
    - If the blue edge is vertical, rotate each rectangle 90° CCW.
    - Stack top-to-bottom with right edges aligned.
    """
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

    outW = int(max(int(arr.shape[1]) for arr in seq))
    outH = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int(outW - w)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)

    return out


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


def _sample_lane(count: int) -> tuple[list[int], list[int]]:
    widths = [int(np.random.choice([2, 4, 6])) for _ in range(count)]
    heights = [int(np.random.choice([2, 3, 4])) for _ in range(count)]
    return widths, heights


def _final_widths(rects: list[tuple[int, int, np.ndarray]], blue_orient: str) -> list[int]:
    if blue_orient == "h":
        return [int(w) for _h, w, _patch in rects]
    return [int(h) for h, _w, _patch in rects]


def _build_canonical() -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    for _ in range(400):
        n0 = int(np.random.randint(2, 5))
        n1 = int(np.random.randint(2, 5))
        w0, h0 = _sample_lane(n0)
        w1, h1 = _sample_lane(n1)

        all_widths = w0 + w1
        all_heights = h0 + h1
        if len(set(all_widths)) < 2 or len(set(all_heights)) < 2:
            continue

        used0 = int(sum(w0) + 2 * (n0 - 1))
        used1 = int(sum(w1) + 2 * (n1 - 1))
        trailing_w = int(np.random.randint(1, 3))
        long_dim = int(max(used0, used1) + 2 + trailing_w)
        if long_dim < 9 or long_dim > 28:
            continue

        top_r0 = 2
        top_max_h = int(max(h0))
        bottom_r0 = int(top_r0 + top_max_h + 2)
        bottom_max_h = int(max(h1))
        trailing_h = int(np.random.randint(1, 3))
        grid_h = int(bottom_r0 + bottom_max_h + trailing_h)
        if grid_h > 18:
            continue

        grid = np.full((grid_h, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        rects: list[tuple[int, int, np.ndarray]] = []
        for r0, widths, heights in ((top_r0, w0, h0), (bottom_r0, w1, h1)):
            c = 2
            for w, h in zip(widths, heights):
                dominant = int(np.random.choice(DOMINANT_COLORS))
                patch = np.full((int(h), int(w)), dominant, dtype=int)
                patch = _sprinkle_minorities(patch, dominant)
                paste(grid, patch, int(r0), int(c))
                rects.append((int(h), int(w), patch.copy()))
                c += int(w) + 2

        return grid, rects

    raise RuntimeError("failed to build canonical input for 291dc1e1 D6")


def _gen_attempt() -> np.ndarray | None:
    canonical, rects = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    if len(set(_final_widths(rects, blue_orient))) < 2:
        return None
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D6")
