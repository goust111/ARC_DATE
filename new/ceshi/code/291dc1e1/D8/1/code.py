# Variant of 291dc1e1
# Dimension: D8 — 多色矩形的处理 (transform)
# Change: 矩形可能有2-3个少数色像素（<=25%面积），直接保持原样参与堆叠... → 少数色像素决定排序优先级

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste
except ImportError:
    for _parent in Path(__file__).resolve().parents:
        _cand = _parent / "solvers" / "arc_utils.py"
        if _cand.exists():
            sys.path.append(str(_cand.parent))
            break
    from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# -----------------------------------------------------------------------------
# Design analysis for the D8 variant
#
# concreteization:
# - The scene layout stays the same as the source task: one corner marker, one
#   horizontal edge line, one vertical edge line, and several filled rectangles
#   in the interior.
# - The change is inside the rectangles. Some rectangles are still pure solid
#   blocks; others contain 2-3 off-color pixels, but the bbox remains fully
#   filled, so they are still detected as single rectangles.
# - Output still stacks cropped rectangles top-to-bottom and still rotates all
#   of them 90 degrees CCW when the blue line is vertical.
#
# behavior trace:
# - Source behavior: sort by (distance to blue line, distance to red line).
#   Example in canonical top-left form:
#   top-left pure A at db=1, bottom-left minority B at db=4.
#   Old order is A before B because 1 < 4.
# - Variant behavior: first split into "contains minority pixels" vs "pure".
#   With minority-first priority, B now comes before A even though it is farther
#   from the blue line. Distances still decide order inside each priority group.
# - Visible effect: output begins with the multi-color rectangles, and only then
#   continues with the pure rectangles, while the old task would have mixed them
#   only by geometric proximity.
#
# complexity calibration:
# - The priority is a single boolean layer, not a many-way mapping, so the
#   rule stays inferable.
# - Generator always creates both kinds of rectangles and mixes them within both
#   distance bands, so the new priority changes the order in a repeated way
#   instead of appearing as a one-off corner case.
# - Dominant colors remain one per rectangle, sampled without replacement, which
#   keeps identity tracking easy after stacking and optional rotation.
#
# implicit assumptions audit:
# - Corner detection still works because only interior rectangles gain extra
#   colors; the marker and edge lines remain unchanged.
# - The original extraction used a boolean mask over all non-background cells,
#   not per-color components. That is still valid because each rectangle stays
#   completely filled, so all its cells remain 4-connected through the bbox.
# - The old solver effectively assumed "all rectangles are equal with respect to
#   color composition", because color never entered the sort key. That is the
#   exact assumption changed here: we now compute a `has_minority` flag from the
#   cropped rectangle and prepend it to the sort key.
# - The new `has_minority` predicate assumes there is one dominant color and the
#   off-color total is at most 25% of area. The generator guarantees this.
#
# placement legality:
# - Rectangles are still placed on two disjoint 2-row bands with 2 columns of
#   spacing, starting at column 2, so they never overlap the edge lines or each
#   other and always stay in bounds.
# - Minority pixels never use background or marker colors, so they cannot punch
#   holes in the rectangle mask or merge with the marker.
#
# visual observability and cleanliness:
# - Each sample contains both pure and multi-color rectangles in both bands.
#   This produces multiple "far minority jumps ahead of near pure" examples in
#   the same grid, making the new rule visible from only a few demonstrations.
# - Multi-color rectangles use only 2-3 accent pixels, so the rectangles remain
#   easy to parse as coherent blocks rather than noisy textures.
# - The layout is still sparse and banded, preserving the original task's clean
#   geometry; only the ordering cue changes.
# -----------------------------------------------------------------------------


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, W = map(int, g.shape)
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
            return int(r), int(c)
    raise AssertionError("corner marker not found")


def _infer_axes(g: np.ndarray, cr: int, cc: int) -> tuple[int, str, str]:
    H, W = map(int, g.shape)
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
    return marker, blue_orient, red_orient


def _has_minority_pixels(arr: np.ndarray) -> bool:
    vals, counts = np.unique(arr, return_counts=True)
    if len(vals) <= 1:
        return False
    area = int(arr.size)
    majority = int(counts.max())
    minority_total = int(area - majority)
    return 1 <= minority_total <= area // 4


def _extract_rectangles(g: np.ndarray, bg: int, marker: int, cr: int, cc: int) -> list[dict]:
    H, W = map(int, g.shape)
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append(
            {
                "box": box,
                "arr": sub.copy(),
                "has_minority": _has_minority_pixels(sub),
            }
        )
    return rects


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _sorted_rectangles(
    rects: list[dict], cr: int, cc: int, blue_orient: str, red_orient: str, *, use_priority: bool
) -> list[dict]:
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        box = rect["box"]
        rect["db"] = int(_dist_to_edge(box, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(box, red_orient, red_edge_idx))

    def key(rect: dict) -> tuple[int, ...]:
        base = (
            int(rect["db"]),
            int(rect["dr"]),
            _along_from_corner(rect["box"], blue_orient, cr, cc),
            _along_from_corner(rect["box"], red_orient, cr, cc),
        )
        if not use_priority:
            return base
        priority = 0 if bool(rect["has_minority"]) else 1
        return (int(priority),) + base

    return sorted(rects, key=key)


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
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


def _analyze_grid(grid: np.ndarray) -> tuple[int, int, int, str, str, list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc = _locate_corner(g, bg)
    marker, blue_orient, red_orient = _infer_axes(g, cr, cc)
    rects = _extract_rectangles(g, bg, marker, cr, cc)
    if not rects:
        raise AssertionError("no rectangles found")
    return bg, cr, cc, blue_orient, red_orient, rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    1. Detect the interior rectangles exactly as in the original task.
    2. Split them into minority-bearing vs pure rectangles.
    3. Stack all minority-bearing rectangles first, ordered by the original
       distance key; then stack all pure rectangles by the same distance key.
    4. If the blue edge is vertical, rotate every rectangle 90 degrees CCW.
    """
    bg, cr, cc, blue_orient, red_orient, rects = _analyze_grid(grid)
    rects = _sorted_rectangles(rects, cr, cc, blue_orient, red_orient, use_priority=True)

    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    return _stack_sequence(seq, bg)


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


def _mixed_status_pattern(count: int) -> list[bool]:
    patterns = {
        2: [[True, False], [False, True]],
        3: [
            [True, False, True],
            [False, True, False],
            [True, False, False],
            [False, True, True],
        ],
    }
    choices = patterns[int(count)]
    return [bool(v) for v in choices[int(np.random.randint(len(choices)))]]


def _choose_widths(statuses: list[bool]) -> list[int]:
    widths = []
    for has_minority in statuses:
        if has_minority:
            widths.append(int(np.random.choice([4, 6, 8])))
        else:
            widths.append(int(np.random.choice([2, 4, 6, 8])))
    return widths


def _minority_cells(width: int, n_minor: int) -> list[tuple[int, int]]:
    candidates = [
        (0, 0),
        (1, width - 1),
        (0, max(1, width - 2)),
        (1, 1),
        (0, width // 2),
        (1, max(0, width // 2 - 1)),
    ]
    np.random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    used = set()
    for cell in candidates:
        if cell in used:
            continue
        used.add(cell)
        chosen.append((int(cell[0]), int(cell[1])))
        if len(chosen) == int(n_minor):
            break
    return chosen


def _make_patch(width: int, dominant: int, has_minority: bool) -> np.ndarray:
    patch = np.full((2, int(width)), int(dominant), dtype=int)
    if not has_minority:
        return patch

    area = int(patch.size)
    max_minor = min(3, area // 4)
    if max_minor < 2:
        raise AssertionError("minority patch is too small")
    n_minor = 2 if max_minor == 2 else int(np.random.choice([2, 3]))
    n_colors = 1 if n_minor == 2 or np.random.rand() < 0.5 else 2
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    for idx, (r, c) in enumerate(_minority_cells(int(width), int(n_minor))):
        patch[int(r), int(c)] = int(colors[idx % n_colors])
    return patch


def _row_required_width(widths: list[int]) -> int:
    return int(2 + sum(int(w) for w in widths) + 2 * (len(widths) - 1) + 1)


def _build_canonical() -> np.ndarray:
    top_count, bottom_count = [(2, 3), (3, 2), (3, 3)][int(np.random.randint(3))]
    top_statuses = _mixed_status_pattern(top_count)
    bottom_statuses = _mixed_status_pattern(bottom_count)
    top_widths = _choose_widths(top_statuses)
    bottom_widths = _choose_widths(bottom_statuses)
    long_dim = max(_row_required_width(top_widths), _row_required_width(bottom_widths)) + int(
        np.random.randint(0, 3)
    )

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    colors = np.random.choice(DOMINANT_COLORS, size=top_count + bottom_count, replace=False)
    color_idx = 0
    for r0, statuses, widths in ((2, top_statuses, top_widths), (5, bottom_statuses, bottom_widths)):
        c = 2
        for has_minority, width in zip(statuses, widths):
            dominant = int(colors[color_idx])
            color_idx += 1
            patch = _make_patch(int(width), dominant, bool(has_minority))
            paste(grid, patch, int(r0), int(c))
            c += int(width) + 2

    return grid


def _variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    bg, cr, cc, blue_orient, red_orient, rects = _analyze_grid(grid)
    del bg
    new_order = _sorted_rectangles(list(rects), cr, cc, blue_orient, red_orient, use_priority=True)
    old_order = _sorted_rectangles(list(rects), cr, cc, blue_orient, red_orient, use_priority=False)

    new_ids = [tuple(rect["box"]) for rect in new_order]
    old_ids = [tuple(rect["box"]) for rect in old_order]
    old_pos = {box: idx for idx, box in enumerate(old_ids)}
    conflicts = 0
    minority_count = 0
    pure_count = 0
    for rect_a in rects:
        if bool(rect_a["has_minority"]):
            minority_count += 1
        else:
            pure_count += 1
        for rect_b in rects:
            if bool(rect_a["has_minority"]) and not bool(rect_b["has_minority"]):
                if old_pos[tuple(rect_a["box"])] > old_pos[tuple(rect_b["box"])]:
                    conflicts += 1

    return {
        "minority_count": int(minority_count),
        "pure_count": int(pure_count),
        "order_changed": bool(new_ids != old_ids),
        "priority_conflicts": int(conflicts),
    }


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        metrics = _variant_metrics(grid)
        if metrics["minority_count"] < 2 or metrics["pure_count"] < 2:
            continue
        if not metrics["order_changed"] or metrics["priority_conflicts"] <= 0:
            continue
        main(grid)
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D8 variant")
