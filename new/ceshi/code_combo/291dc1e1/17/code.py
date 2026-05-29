# Combo variant of 291dc1e1 (index 17)
# combo_spec: D8/0+D11/1
# Name: 去少数色+渐变着色

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Fusion design analysis
======================

a) Concrete behavior trace
--------------------------
Source task:
- detect the marked corner plus one horizontal and one vertical edge line
- extract the interior filled rectangles
- sort them by (distance to blue edge, distance to red edge, tie-breakers)
- if blue is vertical, rotate each rectangle 90 degrees CCW
- stack them top-to-bottom while preserving rectangle colors

This combo changes the rectangle-content pipeline in two consecutive stages:
1. D8/0 cleanup: within each extracted rectangle, repaint every minority pixel
   to the rectangle's dominant color
2. D11/1 gradient recolor: after geometric ordering and optional rotation,
   overwrite the whole rectangle with a fixed rank-color ladder

Small trace on three sorted rectangles:
- input rectangles might be
    rect1 = [[6, 6, 1, 6],
             [6, 6, 6, 6]]
    rect2 = [[4, 4],
             [4, 4]]
    rect3 = [[7, 2, 7, 7],
             [7, 7, 7, 7]]
- source task output would keep those exact colors after stacking
- combo intermediate after D8 cleanup becomes
    rect1 -> all 6
    rect2 -> all 4
    rect3 -> all 7
- combo final after D11 recolor becomes
    rank1 -> all 3
    rank2 -> all 4
    rank3 -> all 5

The visible output therefore differs from the source task in two ways:
- noisy multicolor rectangles no longer survive
- even the cleaned dominant colors are not preserved; rank determines output

b) Complexity calibration
-------------------------
- The rank ladder is fixed to six colors `[3, 4, 5, 6, 7, 9]`, so the number
  of rectangles is restricted to 4-6. This keeps the color rule small enough
  to infer from a few ARC examples and avoids cycling.
- At least one rectangle is guaranteed to be "dirty" and wide enough to hold
  2-3 minority pixels while still having a strict majority color.
- Every rectangle's dominant input color is forced to differ from its assigned
  rank color, so the gradient recolor is always visibly active.
- Dirty rectangles are only a subset of all rectangles. That gives repeated
  evidence that minority pixels are local noise rather than a global palette.

c) Hidden-assumption audit against the source solver
----------------------------------------------------
- Corner detection only reads the four corners and adjacent edge cells.
  Minority pixels are interior-only, so this still holds.
- Blue/red axis detection depends only on the corner row/column. Still safe.
- Component extraction assumes each object bbox has no background holes.
  The combo preserves geometry; only colors vary inside a filled bbox.
- Sorting uses rectangle bbox coordinates only. Color cleanup does not affect
  order, which is important because recoloring happens by sorted rank.
- Rotation is shape-only. Cleaning before rotation is equivalent to cleaning
  after rotation; recoloring after rotation ensures final color depends only on
  rank, not on original pigment placement.
- The original `paste(out, a, ...)` block assumed `a` already carried the final
  visible colors. That assumption breaks under D11 and is replaced here by an
  explicit rank-color fill step.

d) Placement legality
---------------------
- Canonical generation keeps the original two-band layout: rows 2 and 5, with
  rectangles starting at column 2 and separated by 2 background cells.
- Rectangle widths remain even and in `{2, 4, 6, 8}` with fixed height 2.
- The chosen long dimension is `2 + max(used_band_width) + trailing`, so all
  rectangles fit without overlap or clipping.
- Minority pixels are only written inside already allocated rectangle cells, so
  they cannot create new components or break connectivity.
- Final orientation/corner randomization uses only flips/rotations of a legal
  canonical layout, preserving non-overlap.

e) Visual observability
-----------------------
- In training pairs, the input contains some rectangles with obvious stray
  pixels, while the output contains only pure rectangles. That supports the D8
  inference that minority pixels are ignored and collapsed to the main color.
- The output also follows a repeated small ladder of stack-order colors, which
  supports the D11 inference that rank, not original color, controls output.
- Generator-side validation explicitly rejects samples where either stage is
  behaviorally silent on that input.

f) Visual cleanliness
---------------------
- The scene remains sparse: two edge lines, two rectangle bands, consistent
  gaps, and no overlapping symbols.
- Marker color 0 is kept unique at the corner and never used inside rectangles.
- Future rank colors are never used as a rectangle's dominant input color for
  that same rectangle, which avoids confusing "already recolored" appearances.

g) Color-change inferability
----------------------------
- The color rewrite is not arbitrary. It is tied to sorted stack position via a
  fixed small ladder `[3, 4, 5, 6, 7, 9]`.
- The cleanup stage is also structured: minority colors collapse locally to the
  rectangle's majority color before the rank recolor.
- Together this yields a two-step mental model that is still compact enough for
  ARC: first ignore sparse noise inside each rectangle, then recolor by order.

Post-change implicit assumptions and generator guarantees
---------------------------------------------------------
1. There are at most six rectangles, because the solver uses six rank colors.
   The generator restricts layouts to 4-6 rectangles.
2. Dirty rectangles must still have a unique dominant color.
   The generator injects minority pixels only when area >= 8 and caps them at
   <= 25 percent of the rectangle area.
3. Marker/background colors must not appear inside rectangles.
   The generator forbids colors 0 and 8 for minority pixels.
4. The D8 stage should be observable on sampled tasks.
   The generator rejects any sample where `legacy_output == cleaned_output`.
5. The D11 stage should also be observable after cleanup.
   The generator rejects any sample where `cleaned_output == combo_output`.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
RANK_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
INPUT_DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_CHOICES = np.array([2, 4, 6, 8], dtype=int)
COUNT_CHOICES = [(2, 2), (2, 3), (3, 2), (3, 3)]


def _clean_to_dominant(rect: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(rect, return_counts=True)
    dominant = int(vals[int(np.argmax(counts))])
    out = np.array(rect, copy=True)
    out[out != dominant] = dominant
    return out


def _locate_layout(g: np.ndarray) -> tuple[int, int, int, int, str, str]:
    H, W = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        neighbors = []
        if r == 0:
            neighbors.append(int(g[r + 1, c]))
        if r == H - 1:
            neighbors.append(int(g[r - 1, c]))
        if c == 0:
            neighbors.append(int(g[r, c + 1]))
        if c == W - 1:
            neighbors.append(int(g[r, c - 1]))
        if neighbors and all(int(n) != int(g[r, c]) for n in neighbors):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(g[cr, cc])

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
    return int(bg), int(cr), int(cc), int(marker), blue_orient, red_orient


def _sorted_rectangles(g: np.ndarray, *, clean_rectangles: bool) -> tuple[int, str, list[dict]]:
    H, W = g.shape
    bg, cr, cc, marker, blue_orient, red_orient = _locate_layout(g)

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
        if clean_rectangles:
            sub = _clean_to_dominant(sub)
        rects.append({"box": tuple(map(int, box)), "arr": np.array(sub, copy=True)})

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
    return int(bg), blue_orient, rects


def _stack_rectangles(arrays: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(a.shape[1]) for a in arrays))
    out_h = int(sum(int(a.shape[0]) for a in arrays))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for a in arrays:
        h, w = map(int, a.shape)
        out_c = int((out_w - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _solve(grid: np.ndarray, *, clean_rectangles: bool, recolor_by_rank: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, blue_orient, rects = _sorted_rectangles(g, clean_rectangles=clean_rectangles)

    if recolor_by_rank:
        assert len(rects) <= len(RANK_COLORS)

    seq = []
    for idx, rect in enumerate(rects):
        arr = np.array(rect["arr"], copy=True)
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        if recolor_by_rank:
            arr = np.full(arr.shape, int(RANK_COLORS[idx]), dtype=int)
        seq.append(arr)

    return _stack_rectangles(seq, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - sort rectangles by the original geometric rule
    - clean minority pixels to the dominant color
    - rotate if blue is vertical
    - overwrite each rectangle by its stack-rank gradient color
    """
    return _solve(grid, clean_rectangles=True, recolor_by_rank=True)


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


def _band_used(widths: list[int]) -> int:
    return int(sum(widths) + 2 * (len(widths) - 1))


def _sprinkle_minorities(
    patch: np.ndarray,
    dominant: int,
    *,
    force: bool,
    forbidden: set[int],
) -> tuple[np.ndarray, int]:
    area = int(patch.size)
    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch, 0
    if not force and np.random.rand() < 0.7:
        return patch, 0

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    choices = [int(c) for c in MINORITY_COLORS if int(c) not in forbidden and int(c) != dominant]
    if not choices:
        return patch, 0
    colors = np.random.choice(choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch, n_minor


def _build_patch(width: int, rank_color: int, prev_color: int | None, *, force_dirty: bool) -> tuple[np.ndarray, int, int]:
    choices = [int(c) for c in INPUT_DOMINANT_COLORS if int(c) != int(rank_color) and int(c) != prev_color]
    if not choices:
        choices = [int(c) for c in INPUT_DOMINANT_COLORS if int(c) != int(rank_color)]
    dominant = int(np.random.choice(choices))
    patch = np.full((2, int(width)), dominant, dtype=int)
    patch, n_minor = _sprinkle_minorities(
        patch,
        dominant,
        force=force_dirty,
        forbidden={BG, MARKER, int(rank_color)},
    )
    return patch, dominant, int(n_minor)


def _build_canonical() -> tuple[np.ndarray, dict] | None:
    top_count, bottom_count = COUNT_CHOICES[int(np.random.randint(len(COUNT_CHOICES)))]
    widths0 = [int(np.random.choice(WIDTH_CHOICES)) for _ in range(top_count)]
    widths1 = [int(np.random.choice(WIDTH_CHOICES)) for _ in range(bottom_count)]

    all_widths = widths0 + widths1
    if all(int(w) == 2 for w in all_widths):
        idx = int(np.random.randint(len(all_widths)))
        all_widths[idx] = 4
        widths0 = all_widths[:top_count]
        widths1 = all_widths[top_count:]

    used0 = _band_used(widths0)
    used1 = _band_used(widths1)
    trailing = int(np.random.randint(1, 4))
    long_dim = int(2 + max(used0, used1) + trailing)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    specs = []
    for r0, widths in ((2, widths0), (5, widths1)):
        c = 2
        for width in widths:
            specs.append({"r": int(r0), "c": int(c), "width": int(width)})
            c += int(width) + 2

    rect_count = int(len(specs))
    if not (4 <= rect_count <= len(RANK_COLORS)):
        return None

    eligible_dirty = [idx for idx, spec in enumerate(specs) if int(spec["width"]) >= 4]
    if not eligible_dirty:
        return None

    dirty_target = int(np.random.randint(1, min(3, len(eligible_dirty)) + 1))
    dirty_indices = set(map(int, np.random.choice(eligible_dirty, size=dirty_target, replace=False)))

    dirty_pixels = 0
    prev_color = None
    for idx, spec in enumerate(specs):
        rank_color = int(RANK_COLORS[idx])
        patch, dominant, n_minor = _build_patch(
            int(spec["width"]),
            rank_color,
            prev_color,
            force_dirty=idx in dirty_indices,
        )
        paste(grid, patch, int(spec["r"]), int(spec["c"]))
        spec["dominant"] = int(dominant)
        spec["rank_color"] = int(rank_color)
        spec["dirty_pixels"] = int(n_minor)
        dirty_pixels += int(n_minor)
        prev_color = int(dominant)

    meta = {
        "rectangles": rect_count,
        "dirty_rectangles": int(sum(int(spec["dirty_pixels"]) > 0 for spec in specs)),
        "dirty_pixels": int(dirty_pixels),
        "rank_mismatches": int(sum(int(spec["dominant"]) != int(spec["rank_color"]) for spec in specs)),
        "specs": specs,
    }
    return grid, meta


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    legacy = _solve(grid, clean_rectangles=False, recolor_by_rank=False)
    cleaned = _solve(grid, clean_rectangles=True, recolor_by_rank=False)
    combo = _solve(grid, clean_rectangles=True, recolor_by_rank=True)

    _, _, raw_rects = _sorted_rectangles(np.array(grid, dtype=int), clean_rectangles=False)
    dirty_rectangles = 0
    dirty_pixels = 0
    for rect in raw_rects:
        arr = np.array(rect["arr"], copy=True)
        cleaned_arr = _clean_to_dominant(arr)
        delta = int(np.count_nonzero(arr != cleaned_arr))
        if delta > 0:
            dirty_rectangles += 1
            dirty_pixels += delta

    return {
        "rectangles": int(len(raw_rects)),
        "dirty_rectangles": int(dirty_rectangles),
        "dirty_pixels": int(dirty_pixels),
        "legacy_vs_clean_pixels": int(np.count_nonzero(legacy != cleaned)),
        "clean_vs_combo_pixels": int(np.count_nonzero(cleaned != combo)),
        "legacy_vs_combo_pixels": int(np.count_nonzero(legacy != combo)),
    }


def _gen_attempt() -> np.ndarray | None:
    built = _build_canonical()
    if built is None:
        return None
    canonical, meta = built
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    try:
        metrics = _behavior_metrics(grid)
    except Exception:
        return None

    if int(meta["dirty_rectangles"]) < 1 or int(meta["dirty_pixels"]) < 2:
        return None
    if int(meta["rank_mismatches"]) != int(meta["rectangles"]):
        return None
    if int(metrics["dirty_rectangles"]) < 1 or int(metrics["dirty_pixels"]) < 2:
        return None
    if int(metrics["legacy_vs_clean_pixels"]) <= 0:
        return None
    if int(metrics["clean_vs_combo_pixels"]) <= 0:
        return None
    if int(metrics["legacy_vs_combo_pixels"]) <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo D8/0+D11/1")
