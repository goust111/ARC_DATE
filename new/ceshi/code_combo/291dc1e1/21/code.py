# Combo variant of 291dc1e1 (index 21)
# combo_spec: D12/2+D14/1
# Name: 线上标记+装饰线

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for this combo variant
#
# a) Concrete behavior / trace
# - Base task: two functional edge lines meet at one corner; interior solid
#   rectangles are sorted by distance to the blue line first, then the red line.
#   If blue is vertical, every rectangle rotates 90 degrees CCW before stacking.
# - D12/2 adds symbolic instructions on the functional lines: extra pixels on
#   the blue or red edge at fixed distances 2 and 4 from the corner encode
#   separator-row colors. The first slot controls the boundary after blue-distance
#   group 1, the second controls the boundary after group 2.
# - D14/1 adds one non-functional decorative thin line elsewhere in the grid.
#   It is visually line-like but must not affect sorting or separators.
# - Concrete canonical trace: top blue row, left red column, corner marker at
#   (0,0), rectangle bands at rows 2-3 / 5-6 / 8-9, a marker of color 4 placed
#   at (0,2), another of color 7 at (4,0), and an unrelated vertical thin line
#   on the far-right border. Old logic would stack all solid components and might
#   even treat the thin line as an extra rectangle. New logic must:
#   1) identify the functional row+column from the corner marker,
#   2) read colors 4 and 7 from the two fixed frame slots,
#   3) ignore the far-right thin line,
#   4) stack band-1 rectangles, insert a full row of 4, stack band-2 rectangles,
#   insert a full row of 7, then stack band-3 rectangles.
#
# b) Complexity calibration
# - The symbolic part is limited to exactly two slot distances and three blue-
#   distance groups. That gives repeated evidence without creating a large code.
# - The distractor part is limited to exactly one decorative thin line, so the
#   task clearly differs from the original but does not become visually noisy.
# - Separator rows are full-width and 1 cell tall, making the symbolic effect
#   obvious in output height and color.
#
# c) Implicit-assumption audit against the original solver
# - Original assumption: every non-background component away from the functional
#   axes is a rectangle. Broken by D14/1; the decorative line would be misread as
#   a 1xN or Nx1 rectangle. Fix: keep only solid components with both dimensions
#   at least 2.
# - Original assumption: output height is just the sum of rectangle heights.
#   Broken by D12/2; separator rows add extra height.
# - Original assumption: the line colors alone define the whole rule. Broken by
#   D12/2; now isolated overwritten edge cells carry extra semantics.
# - We intentionally keep the original corner marker. In this fusion, the new
#   reasoning burden is not "find the frame without a marker" but "separate the
#   functional frame and its symbolic slots from a third unrelated line."
#
# d) Placement legality
# - Canonical height is 11: row 0 is the blue edge, col 0 the red edge, rows
#   2/5/8 host the three 2-cell-high rectangle bands, and row/col 1 remain gaps.
# - Rectangles occupy only the interior starting at col 2 and leave at least one
#   empty column before the decorative line on the far border.
# - Instruction markers overwrite only cells on the functional row/column at
#   distances 2 and 4, so they never touch rectangles.
# - The decorative line sits on the border opposite the corner and leaves gaps to
#   both adjacent corners, so after flips/rotations it still stays off the two
#   functional axes.
#
# e) Visual observability
# - Across a few examples, humans can see that colored single pixels on the
#   active frame reappear as full separator rows in the output, while the third
#   long thin line never reappears.
# - Because the decorative line is also line-shaped, the solver must use location
#   and attachment to the corner, not "anything line-like is functional".
#
# f) Visual cleanliness
# - The frame markers are sparse and fixed-slot. The decorative line is long but
#   isolated on the opposite border. Rectangles stay in three clean interior
#   stripes, so no region mixes multiple object types in a hard-to-parse way.
#
# g) Color inference
# - Marker colors come from a small palette excluding bg/blue/red and are copied
#   directly into separator rows. This makes the color mapping explicit.
# - The decorative-line color is also chosen from the same general rectangle
#   palette family, but it is excluded from dominant rectangle colors and from
#   active separator colors in that sample. That avoids clutter while preventing
#   a globally fixed "decorative color".
#
# h) Fusion behavior metrics
# - Every generated sample must have exactly one thin decorative component away
#   from the functional axes.
# - Every generated sample must activate at least one separator slot.
# - The generator rejects any sample where the fused solver output equals a
#   baseline "original-like" output that ignores separators and treats the thin
#   line as a sortable object. This blocks silent regression to the old task.
#
# i) New solver assumptions and generator support
# - Assumption: the corner marker remains unique and both cells adjacent to it on
#   the functional axes are still BLUE/RED. Generator support: separator slots
#   start only at distances 2 and 4.
# - Assumption: true sortable rectangles have thickness >= 2 in both dimensions.
#   Generator support: all bands are 2 x even-width solid rectangles.
# - Assumption: there are exactly three blue-distance groups, so slot 1 and slot
#   2 correspond to real boundaries. Generator support: three fixed bands are
#   always generated at distances 1, 4, and 7 from the blue line.
# ---------------------------------------------------------------------------

BG = 8
BLUE = 1
RED = 2
MARKER = 0
CANON_H = 11
BAND_ROWS = (2, 5, 8)
SLOT_DISTANCES = (2, 4)
SYMBOL_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
BODY_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
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


def _find_corner(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, W = map(int, g.shape)
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    for r, c in corners:
        if int(g[r, c]) != MARKER:
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
        if nbr and all(v != int(bg) for v in nbr):
            return int(r), int(c)
    raise AssertionError("corner marker not found")


def _locate_frame(g: np.ndarray, bg: int) -> tuple[int, int, str, str]:
    cr, cc = _find_corner(g, bg)
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])
    assert {hcol, vcol} == {BLUE, RED}
    blue_orient = "h" if hcol == BLUE else "v"
    red_orient = "v" if blue_orient == "h" else "h"
    return int(cr), int(cc), blue_orient, red_orient


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
            if v not in (bg, MARKER, BLUE, RED):
                colors.append(v)

        r = int(cr + vstep * dist)
        if 0 <= r < H:
            v = int(g[r, cc])
            if v not in (bg, MARKER, BLUE, RED):
                colors.append(v)

        assert len(colors) <= 1
        out.append(int(colors[0]) if colors else None)

    return out


def _line_mask(shape: tuple[int, int], cr: int, cc: int) -> np.ndarray:
    H, W = map(int, shape)
    mask = np.zeros((H, W), dtype=bool)
    mask[cr, :] = True
    mask[:, cc] = True
    return mask


def _extract_rectangles(g: np.ndarray, bg: int, cr: int, cc: int) -> list[dict]:
    mask = (g != int(bg)) & (~_line_mask(g.shape, cr, cc)) & (g != MARKER)
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        if min(map(int, sub.shape)) < 2:
            continue
        rects.append({"box": tuple(map(int, b)), "arr": sub.copy()})
    return rects


def _extract_all_solids(g: np.ndarray, bg: int, cr: int, cc: int) -> list[dict]:
    mask = (g != int(bg)) & (~_line_mask(g.shape, cr, cc)) & (g != MARKER)
    objs = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        objs.append({"box": tuple(map(int, b)), "arr": sub.copy()})
    return objs


def _dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
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


def _along_from_corner(rect: dict, orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _sort_rectangles(rects: list[dict], cr: int, cc: int, blue_orient: str, red_orient: str) -> list[dict]:
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rct in rects:
        rct["db"] = int(_dist_to_edge(rct, blue_orient, blue_edge_idx))
        rct["dr"] = int(_dist_to_edge(rct, red_orient, red_edge_idx))
    rects.sort(
        key=lambda r: (
            int(r["db"]),
            int(r["dr"]),
            _along_from_corner(r, blue_orient, cr, cc),
            _along_from_corner(r, red_orient, cr, cc),
        )
    )
    return rects


def _stack_groups(groups: list[list[np.ndarray]], separator_slots: list[int | None], bg: int) -> np.ndarray:
    seq = [a for grp in groups for a in grp]
    out_w = int(max(int(a.shape[1]) for a in seq))
    separator_count = sum(1 for i, color in enumerate(separator_slots) if color is not None and i < len(groups) - 1)
    out_h = int(sum(int(a.shape[0]) for a in seq) + separator_count)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for gi, grp in enumerate(groups):
        for a in grp:
            h, w = map(int, a.shape)
            out_c = int((out_w - w) // 2)
            paste(out, a, int(out_r), int(out_c))
            out_r += int(h)
        if gi < len(groups) - 1 and gi < len(separator_slots):
            sep_color = separator_slots[gi]
            if sep_color is not None:
                out[int(out_r), :] = int(sep_color)
                out_r += 1
    return out


def _baseline_main_like_original(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, blue_orient, red_orient = _locate_frame(g, bg)
    rects = _extract_all_solids(g, bg, cr, cc)
    assert rects
    _sort_rectangles(rects, cr, cc, blue_orient, red_orient)

    seq = []
    for rct in rects:
        a = rct["arr"]
        if blue_orient == "v":
            a = np.rot90(a, k=1)
        seq.append(a)

    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for a in seq:
        h, w = map(int, a.shape)
        out_c = int((out_w - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, blue_orient, red_orient = _locate_frame(g, bg)
    separator_slots = _read_separator_slots(g, bg, cr, cc)
    rects = _extract_rectangles(g, bg, cr, cc)
    assert rects
    _sort_rectangles(rects, cr, cc, blue_orient, red_orient)

    groups: list[list[np.ndarray]] = []
    current_db: int | None = None
    for rct in rects:
        a = rct["arr"]
        if blue_orient == "v":
            a = np.rot90(a, k=1)
        db = int(rct["db"])
        if db != current_db:
            groups.append([])
            current_db = db
        groups[-1].append(a)

    return _stack_groups(groups, separator_slots, bg)


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


def _make_band_widths(max_used: int) -> list[int]:
    max_count = min(3, (max_used + 2) // 4)
    count = int(np.random.randint(2, max_count + 1))
    used_cap = min(max_used, 10 * count - 2)
    used_floor = max(4 * count - 2, used_cap - 6)
    used_target = int(np.random.choice(np.arange(used_floor, used_cap + 1, 2)))
    return _distribute_band_widths(count, used_target)


def _sprinkle_minorities(patch: np.ndarray, dominant: int, forbidden: set[int]) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.58:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant and int(c) not in forbidden]
    if not minority_choices:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor, len(minority_choices)) + 1))
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _add_decorative_line(grid: np.ndarray, deco_color: int) -> None:
    grid[2 : CANON_H - 2, -1] = int(deco_color)


def _count_decorative_candidates(g: np.ndarray) -> int:
    bg = int(get_bg(g))
    cr, cc, _blue_orient, _red_orient = _locate_frame(g, bg)
    mask = (g != int(bg)) & (~_line_mask(g.shape, cr, cc)) & (g != MARKER)
    count = 0
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        if min(map(int, sub.shape)) == 1:
            count += 1
    return int(count)


def _build_canonical() -> np.ndarray:
    long_dim = int(np.random.randint(16, 30))

    active = np.random.rand(len(SLOT_DISTANCES)) < 0.72
    if not bool(active.any()):
        active[int(np.random.randint(len(SLOT_DISTANCES)))] = True
    active_count = int(active.sum())

    sep_colors = np.random.choice(SYMBOL_COLORS, size=active_count, replace=False)
    sep_set = {int(c) for c in sep_colors.tolist()}

    deco_choices = [int(c) for c in BODY_COLORS if int(c) not in sep_set]
    deco_color = int(np.random.choice(deco_choices))

    body_choices = np.array([int(c) for c in BODY_COLORS if int(c) not in sep_set and int(c) != deco_color], dtype=int)
    assert len(body_choices) >= 2

    grid = np.full((CANON_H, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    max_used = int(long_dim - 4)
    forbidden = set(sep_set)
    forbidden.add(int(deco_color))
    for r0 in BAND_ROWS:
        c = 2
        for w in _make_band_widths(max_used):
            color = int(np.random.choice(body_choices))
            patch = np.full((2, int(w)), int(color), dtype=int)
            paste(grid, _sprinkle_minorities(patch, color, forbidden), int(r0), int(c))
            c += int(w + 2)

    color_idx = 0
    for slot_idx, dist in enumerate(SLOT_DISTANCES):
        if not bool(active[slot_idx]):
            continue
        color = int(sep_colors[color_idx])
        color_idx += 1
        if bool(np.random.randint(2)):
            grid[0, int(dist)] = int(color)
        else:
            grid[int(dist), 0] = int(color)

    _add_decorative_line(grid, deco_color)
    return grid


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        try:
            out = main(grid)
            baseline = _baseline_main_like_original(grid)
        except Exception:
            continue

        bg = int(get_bg(grid))
        cr, cc, _blue_orient, _red_orient = _locate_frame(grid, bg)
        slots = _read_separator_slots(grid, bg, cr, cc)
        if not any(color is not None for color in slots):
            continue
        if _count_decorative_candidates(grid) != 1:
            continue
        if np.array_equal(out, baseline):
            continue
        if int(out.shape[0]) <= 8:
            continue
        return grid

    raise RuntimeError("failed to generate input for 291dc1e1 combo D12/2+D14/1")
