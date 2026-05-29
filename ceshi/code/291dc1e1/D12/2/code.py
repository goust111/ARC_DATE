# Variant of 291dc1e1
# Dimension: D12 — 角落与方向编码 (symbol_instructions)
# Change: 标记像素(色0)位于蓝线与红线交叉的网格角落，蓝线颜色固定为1，方向由网格宽高比隐含... → 边缘线上的额外标记像素编码分隔符

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for this D12 variant
#
# a) Concrete behavior / trace
# - The original task uses two edge lines that meet at one corner and a special
#   corner marker. Rectangles are sorted by (distance to blue edge,
#   distance to red edge), then stacked; if blue is vertical, every rectangle
#   is rotated 90 degrees CCW before stacking.
# - In this variant the corner marker is removed as the instruction carrier.
#   Instead, the frame contains up to two extra colored pixels placed on the
#   blue or red edge lines. These are "instruction slots" at fixed distances
#   2 and 4 from the corner. Their colors become separator-row colors.
# - Rectangles are still sorted exactly as before. The only new output effect
#   is: after the first blue-distance group, insert a 1-row separator if slot 1
#   is occupied; after the second blue-distance group, insert one if slot 2 is
#   occupied. The separator row spans the full output width.
# - Small trace in canonical orientation:
#   top blue edge, left red edge, three rectangle bands at rows 2-3 / 5-6 /
#   8-9. If slot-1 contains color 4 and slot-2 is empty, the output becomes:
#   [band group 1 stacked] + [full row of 4] + [group 2 stacked] + [group 3].
#
# b) Complexity calibration
# - Only two discrete separator boundaries are used. This is enough to create
#   visible, repeated evidence without turning the task into a long symbolic
#   parsing problem.
# - Three blue-distance groups are used in the generator, so the new mechanism
#   has two potential places where it can act. This avoids the degenerate case
#   where a symbol system exists but there is only one possible boundary.
#
# c) Hidden assumptions in the original solver that break here
# - The old corner finder assumes a unique non-background pixel at the corner.
#   That is invalid now because the corner is no longer specially colored.
# - The old blue/red line detector assumes corner-adjacent cells directly reveal
#   line colors. That becomes fragile once edge-line cells may be overwritten by
#   instruction markers. This variant therefore infers blue orientation from the
#   aspect ratio, then locates the active border row/column by density of
#   non-background cells.
# - The old output height is just the sum of rectangle heights. That is no
#   longer valid because separator rows add extra height.
#
# d) Placement legality
# - Objects never touch the frame lines: the generator reserves row/col 0 for
#   the lines, row/col 1 as buffer, and uses rectangle bands at rows 2/5/8.
# - Instruction markers only replace cells on the two frame lines, so they do
#   not merge with rectangles and do not alter component extraction.
# - At most one marker may occupy each slot distance; the solver asserts this.
#
# e) Visual observability
# - Human-visible clue: isolated colored pixels sit on the frame lines, and
#   output rows of exactly those colors appear between stacked rectangle groups.
# - The inserted rows change output height, so the new behavior cannot silently
#   collapse back to the original stacking rule.
#
# f) Visual cleanliness
# - Markers are sparse and use fixed slots, so the frame remains readable.
# - Separator rows are full-width and 1 cell tall, producing a clean output.
#
# g) Color inference
# - Separator colors come from a small palette excluding bg/blue/red.
# - The same marker color is copied directly into the separator row, making the
#   color mapping explicit rather than latent.
#
# 9) New assumptions introduced by the modified solver, and generator support
# - Assumption: non-square grids; width>height means blue is horizontal,
#   height>width means blue is vertical. The generator enforces canonical height
#   11 and chooses the long side > 11 before optional rotation.
# - Assumption: there are at most two separator boundaries to read. The
#   generator always creates exactly three blue-distance groups, so the slots
#   correspond to real boundaries.
# - Assumption: objects remain solid rectangles (possibly with minority colored
#   cells) whose bounding boxes contain no background. The generator preserves
#   that structure.
# ---------------------------------------------------------------------------


BG = 8
BLUE = 1
RED = 2
SEPARATOR_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

CANON_H = 11
BAND_ROWS = (2, 5, 8)
SLOT_DISTANCES = (2, 4)


def _locate_frame(g: np.ndarray, bg: int) -> tuple[int, int, str, str]:
    H, W = map(int, g.shape)
    blue_orient = "h" if W > H else "v"

    if blue_orient == "h":
        top_score = int(np.count_nonzero(g[0, :] != int(bg)))
        bot_score = int(np.count_nonzero(g[H - 1, :] != int(bg)))
        cr = 0 if top_score >= bot_score else H - 1

        left_score = int(np.count_nonzero(g[:, 0] != int(bg)))
        right_score = int(np.count_nonzero(g[:, W - 1] != int(bg)))
        cc = 0 if left_score >= right_score else W - 1
        return int(cr), int(cc), "h", "v"

    left_score = int(np.count_nonzero(g[:, 0] != int(bg)))
    right_score = int(np.count_nonzero(g[:, W - 1] != int(bg)))
    cc = 0 if left_score >= right_score else W - 1

    top_score = int(np.count_nonzero(g[0, :] != int(bg)))
    bot_score = int(np.count_nonzero(g[H - 1, :] != int(bg)))
    cr = 0 if top_score >= bot_score else H - 1
    return int(cr), int(cc), "v", "h"


def _read_separator_slots(g: np.ndarray, bg: int, cr: int, cc: int) -> list[int | None]:
    H, W = map(int, g.shape)
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    out: list[int | None] = []

    for dist in SLOT_DISTANCES:
        colors: list[int] = []

        c = int(cc + hstep * dist)
        if 0 <= c < W:
            v = int(g[cr, c])
            if v not in (int(bg), BLUE, RED):
                colors.append(v)

        r = int(cr + vstep * dist)
        if 0 <= r < H:
            v = int(g[r, cc])
            if v not in (int(bg), BLUE, RED):
                colors.append(v)

        assert len(colors) <= 1
        out.append(int(colors[0]) if colors else None)

    return out


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
    if area < 8 or np.random.rand() < 0.55:
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


def _make_band_widths(max_used: int) -> list[int]:
    max_count = min(4, (max_used + 2) // 4)
    count = int(np.random.randint(2, max_count + 1))
    used_cap = min(max_used, 10 * count - 2)
    used_target = int(np.random.choice(np.arange(4 * count - 2, used_cap + 1, 2)))
    return _distribute_band_widths(count, used_target)


def _build_canonical() -> np.ndarray:
    long_dim = int(np.random.randint(15, 30))
    grid = np.full((CANON_H, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = BLUE

    max_used = int(long_dim - 4)
    for r0 in BAND_ROWS:
        c = 2
        for w in _make_band_widths(max_used):
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, w), color, dtype=int)
            paste(grid, _sprinkle_minorities(patch, color), int(r0), int(c))
            c += int(w + 2)

    active = np.random.rand(len(SLOT_DISTANCES)) < 0.72
    if not bool(active.any()):
        active[int(np.random.randint(len(SLOT_DISTANCES)))] = True

    chosen = np.random.choice(SEPARATOR_COLORS, size=int(active.sum()), replace=False)
    color_idx = 0
    for slot_idx, dist in enumerate(SLOT_DISTANCES):
        if not bool(active[slot_idx]):
            continue
        color = int(chosen[color_idx])
        color_idx += 1
        carrier = str(np.random.choice(["blue", "red"]))
        if carrier == "blue":
            grid[0, int(dist)] = color
        else:
            grid[int(dist), 0] = color

    return grid


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to the blue and red edge lines.

    Variant rule:
    - Blue is always color 1.
    - Blue orientation is implicit from aspect ratio.
    - Extra colored pixels on the frame lines act as separator instructions.
      Slot 1 inserts a separator after the first blue-distance group.
      Slot 2 inserts a separator after the second blue-distance group.
    """
    g = np.array(grid, dtype=int)
    H, W = map(int, g.shape)
    bg = int(get_bg(g))

    cr, cc, blue_orient, red_orient = _locate_frame(g, bg)
    separator_slots = _read_separator_slots(g, bg, cr, cc)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask)
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

    groups: list[list[np.ndarray]] = []
    current_db: int | None = None
    for rct in rects:
        a = rct["arr"]
        if blue_orient == "v":
            a = np.rot90(a, k=1)
        db = int(rct["db"])
        if current_db != db:
            groups.append([])
            current_db = db
        groups[-1].append(a)

    seq = [a for grp in groups for a in grp]
    outW = int(max(int(a.shape[1]) for a in seq))

    separator_count = 0
    for idx, color in enumerate(separator_slots):
        if color is not None and idx < len(groups) - 1:
            separator_count += 1

    outH = int(sum(int(a.shape[0]) for a in seq) + separator_count)
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for gi, grp in enumerate(groups):
        for a in grp:
            h, w = map(int, a.shape)
            out_c = int((outW - w) // 2)
            paste(out, a, int(out_r), int(out_c))
            out_r += int(h)
        if gi < len(groups) - 1 and gi < len(separator_slots):
            sep_color = separator_slots[gi]
            if sep_color is not None:
                out[int(out_r), :] = int(sep_color)
                out_r += 1

    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        try:
            out = main(grid)
        except Exception:
            continue

        bg = int(get_bg(grid))
        cr, cc, _bo, _ro = _locate_frame(grid, bg)
        slot_colors = _read_separator_slots(grid, bg, cr, cc)
        if not any(color is not None for color in slot_colors):
            continue
        if int(out.shape[0]) <= 6:
            continue
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D12 variant")
