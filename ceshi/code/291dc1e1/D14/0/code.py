# Variant of 291dc1e1
# Dimension: D14 — 干扰项与噪声 (src)
# Change: 网格中除矩形和边缘线外无额外对象... → 增加非矩形干扰色块

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
for helper_dir in (ROOT / "solvers", ROOT / "arc2_solver_select"):
    helper_str = str(helper_dir)
    if helper_str not in sys.path:
        sys.path.append(helper_str)

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

"""
Design analysis for the D14 variant

Reference-task requirement:
- The source task `b99e7126` is present only as evaluation JSON in this workspace.
- I searched the repository for a corresponding solver `.py` and none exists, so per
  the instruction "不存在就跳过", the solver-reading step is skipped.

a) Concrete behavior and trace
- Original input: edge lines define a corner-based coordinate system; every interior
  non-background component is intended to be a valid rectangle.
- Variant input: keep the same edge lines and legal rectangles, but add 1-3 small,
  detached, 4-connected, non-rectangular noise blobs in the background corridors.
- Output stays identical to the original task: only legal rectangles are sorted,
  optionally rotated, and stacked.
- Local trace on one candidate component:
  original logic:
    1. Take a connected component from all non-background interior pixels.
    2. Crop its bbox.
    3. If the crop contains no background, treat it as a rectangle.
  variant logic:
    1. Take the same component.
    2. Build the component occupancy mask inside its bbox.
    3. Keep it only if the occupancy mask fully fills the bbox and both dimensions
       are at least 2.
  difference:
    - Legal rectangles still pass.
    - An L/T/zigzag noise blob is now explicitly rejected as "not bbox-filling",
      instead of relying on the accidental fact that its bbox contains background.

b) Complexity calibration
- Noise blobs are limited to a small template set with 3-5 cells. This makes them
  visibly irregular without adding too many possible subcases.
- Count is limited to 1-3 per input so the grid changes are obvious but not cluttered.
- Rectangles keep the original width variation and internal minority-color sprinkles,
  so the task still has enough repeated evidence for the sorting/rotation rule.

c) Implicit-assumption audit of the original `main()`
- `mask = (g != bg) & (~line_mask) & (g != marker)`:
  assumes every remaining component is rectangle-like enough that bbox checking is
  sufficient. This no longer holds with noise.
- `sub = crop_bbox(g, b)` then `(sub == bg).any()`:
  assumes "bbox has no background" is equivalent to "component is a rectangle".
  Under D14 this is the wrong predicate; the solver must validate the component's
  own occupancy pattern, not the raw crop alone.
- `rects.append({"box": ..., "arr": sub.copy()})`:
  assumes every accepted component can be pasted as-is. That is still valid once
  the stricter rectangle check is applied.
- Sorting, rotation, stacking:
  depend only on valid rectangles and remain unchanged.

d) Placement legality
- Generator creates rectangles first, then places noise only on currently empty cells.
- Noise is kept 4-neighbor-separated from rectangles and from other noise so solver
  components do not merge.
- Noise never overlaps the corner marker or the edge lines.
- After generation, `main(input)` is checked against the generator's expected output.

e) Visual observability
- The new feature is directly visible: besides the two-row rectangles, the input now
  contains extra jagged blobs that do not appear in the output.
- Because output ignores those blobs completely, a human can infer they are distractors.
- The original stacking rule remains inferable across samples because rectangles still
  dominate the structure and the noise count stays low.

f) Visual cleanliness
- Noise is placed in dedicated empty corridors above, between, or below rectangle bands.
- 4-neighbor gaps avoid object fusion and keep object boundaries legible.
- Color choices avoid using the background or marker color, so noise remains distinct
  from empty space and the corner marker.

g) Color-change inferability
- No new color mapping rule is introduced.
- Noise reuses ordinary rectangle colors, preventing trivial "ignore this one fixed
  color" shortcuts while keeping the palette small and familiar.

h) Variant-activity metric used after implementation
- Measure `rejected_noise_components`: the number of connected components seen by the
  rectangle extractor but rejected by the structural rectangle test.
- The variant is active iff this count is regularly positive in generated samples.

i) New assumptions introduced by the updated solver
- Legal rectangles must be 4-connected and fully fill their bbox.
- Legal rectangles must be at least 2x2.
- Noise must remain detached from rectangles by at least one 4-neighbor gap.
- Generator enforces all of the above.
"""


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    H, W = g.shape
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
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
            return int(r), int(c), int(g[r, c]), int(H), int(W)
    raise AssertionError("corner not found")


def _line_info(g: np.ndarray, bg: int) -> tuple[int, int, int, str, str]:
    cr, cc, marker, H, W = _locate_corner(g, bg)
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
    else:
        hlen = 0
        c = cc + hstep
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += hstep
        vlen = 0
        r = cr + vstep
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += vstep
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"
    return int(cr), int(cc), int(marker), blue_orient, red_orient


def _extract_rectangles(g: np.ndarray, bg: int, cr: int, cc: int, marker: int) -> list[dict]:
    H, W = g.shape
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []

    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 2 or w < 2:
            continue

        occ = np.zeros((h, w), dtype=bool)
        occ[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not bool(occ.all()):
            continue

        rects.append(
            {
                "box": (int(r0), int(c0), int(r1), int(c1)),
                "arr": crop_bbox(g, (int(r0), int(c0), int(r1), int(c1))),
            }
        )

    if not rects:
        raise AssertionError("no rectangles found")
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _line_info(g, bg)
    rects = _extract_rectangles(g, bg, cr, cc, marker)

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            if c0 > edge_idx:
                return int(c0 - edge_idx - 1)
            if c1 < edge_idx:
                return int(edge_idx - c1 - 1)
            return 0
        if r0 > edge_idx:
            return int(r0 - edge_idx - 1)
        if r1 < edge_idx:
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

    out_w = max(int(arr.shape[1]) for arr in seq)
    out_h = sum(int(arr.shape[0]) for arr in seq)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, out_r, out_c)
        out_r += h

    return out


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
NOISE_TEMPLATES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid.copy()
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


def _stack_expected(rects: list[np.ndarray], rotate: bool) -> np.ndarray:
    seq = [np.rot90(rect, k=1) if rotate else rect.copy() for rect in rects]
    out_w = max(int(arr.shape[1]) for arr in seq)
    out_h = sum(int(arr.shape[0]) for arr in seq)
    out = np.full((out_h, out_w), BG, dtype=int)
    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        paste(out, arr, out_r, (out_w - w) // 2)
        out_r += h
    return out


def _neighbor_block(mask: np.ndarray) -> np.ndarray:
    blocked = mask.copy()
    blocked[1:, :] |= mask[:-1, :]
    blocked[:-1, :] |= mask[1:, :]
    blocked[:, 1:] |= mask[:, :-1]
    blocked[:, :-1] |= mask[:, 1:]
    return blocked


def _paint_noise_patch(shape: np.ndarray) -> np.ndarray:
    patch = np.full(shape.shape, BG, dtype=int)
    palette = np.random.choice(DOMINANT_COLORS, size=int(np.random.randint(1, 3)), replace=False)
    coords = np.argwhere(shape)
    for idx, (r, c) in enumerate(coords):
        patch[int(r), int(c)] = int(palette[idx % len(palette)])
    return patch


def _add_noise(grid: np.ndarray, rect_mask: np.ndarray) -> int:
    H, W = grid.shape
    noise_count = 0
    occupied = rect_mask.copy()

    target = int(np.random.randint(1, 4))
    shapes = list(np.random.permutation(len(NOISE_TEMPLATES)))

    for _ in range(target):
        blocked = _neighbor_block(occupied)
        candidates: list[tuple[int, int, np.ndarray]] = []
        for idx in shapes:
            shape = NOISE_TEMPLATES[int(idx)]
            h, w = shape.shape
            for r in range(1, H - h + 1):
                for c in range(1, W - w + 1):
                    cells = np.argwhere(shape) + np.array([[r, c]])
                    if np.any(blocked[cells[:, 0], cells[:, 1]]):
                        continue
                    candidates.append((int(r), int(c), shape))
        if not candidates:
            break
        r, c, shape = candidates[int(np.random.randint(len(candidates)))]
        patch = _paint_noise_patch(shape)
        paste(grid, patch, r, c, transparent=BG)
        occ = np.argwhere(shape) + np.array([[r, c]])
        occupied[occ[:, 0], occ[:, 1]] = True
        noise_count += 1

    return noise_count


def _build_canonical() -> tuple[np.ndarray, list[np.ndarray], int]:
    long_dim = int(np.random.randint(11, 29))
    gap_rows = int(np.random.randint(2, 5))
    bottom_margin = int(np.random.randint(2, 5))
    top_r = 2
    bottom_r = top_r + 2 + gap_rows
    height = bottom_r + 2 + bottom_margin

    options = []
    for trailing in (2, 3, 4):
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

    grid = np.full((height, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    rects: list[np.ndarray] = []
    rect_mask = np.zeros_like(grid, dtype=bool)
    rect_mask[top_r : top_r + 2, 0] = False
    rect_mask[bottom_r : bottom_r + 2, 0] = False

    for r0, widths in ((top_r, band0), (bottom_r, band1)):
        c = 2
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, r0, c)
            rects.append(patch.copy())
            rect_mask[r0 : r0 + 2, c : c + w] = True
            c += w + 2

    noise_count = _add_noise(grid, rect_mask)
    return grid, rects, noise_count


def _gen_attempt() -> np.ndarray | None:
    canonical, rects, noise_count = _build_canonical()
    if noise_count < 1:
        return None

    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    expected = _stack_expected(rects, rotate=(blue_orient == "v"))
    grid = _transform(canonical, blue_orient, corner)
    actual = main(grid)
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D14")
