# Combo variant of 291dc1e1 (index 3)
# combo_spec: D1/0+D8/1
# Name: 面积排序+少数色优先级

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
# Design analysis for combo D1/0 + D8/1
#
# a) Concrete behavior and trace:
#    The source task finds the marked corner plus the blue/red edge lines, then
#    extracts the interior rectangles and stacks them. Originally the stack order
#    is geometric: (distance to blue, distance to red, tie-break by position).
#    This combo changes the order into a fused key:
#      1. rectangles containing minority-color pixels come first
#      2. inside each minority/pure group, sort by area from small to large
#      3. if a tie remains, use stable spatial tie-breaks from the source task
#    Example in canonical top-left form:
#      A = pure 2x2 area 4 near the blue line
#      B = minority 2x5 area 10 near the blue line
#      C = minority 3x3 area 9 farther away
#      D = pure 3x5 area 15 farther away
#    Source order could be A -> B -> C -> D by geometry.
#    Area-only order would be A -> C -> B -> D.
#    Fused order is C -> B -> A -> D:
#      minority rectangles C/B jump ahead of pure A/D, and inside the minority
#      group C(area 9) precedes B(area 10). Both dimensions visibly matter.
#
# b) Complexity calibration:
#    If all rectangles had one fixed height, "area" would collapse into width and
#    become too close to the old along-line geometry. The generator therefore
#    uses two bands of different heights (2 and 3), preserving the original
#    visual layout while making area true cell counting. Minority priority is
#    only a boolean split, keeping the hypothesis space small enough for ARC.
#    Each sample includes multiple minority and multiple pure rectangles, and
#    both groups contain at least two rectangles so the inner area order is also
#    visible repeatedly rather than as a single corner case.
#
# c) Implicit-assumption audit:
#    - Corner detection still assumes one marker distinct from adjacent line
#      colors. Generator preserves that exactly.
#    - Blue-line inference still relies on color 1 or, failing that, longer
#      line length. Generator keeps blue=1, so the branch remains safe.
#    - Rectangle extraction assumes each target object fills its bounding box
#      without background holes. Generator uses full rectangles and only changes
#      a few interior cells to non-background accent colors, so connectivity and
#      bbox solidity still hold.
#    - The source solver assumed geometry alone determines ordering. That
#      assumption is deliberately broken: color purity now matters first, and
#      geometry is demoted to tie-break only.
#    - The new minority test assumes exactly one dominant color per rectangle
#      and at most 25% off-color cells. Generator guarantees 2-3 minority cells
#      only on sufficiently large rectangles so the predicate is unambiguous.
#
# d) Placement legality:
#    Rectangles are placed in two separated horizontal bands inside a larger
#    canvas, start after a one-cell gap from the red line, and keep two-cell
#    gaps from each other. Whole-grid flips/rotations are then applied, which
#    preserve non-overlap and in-bounds placement.
#
# e) Visual observability:
#    A human should see that some multi-color rectangles move ahead of simpler
#    pure rectangles even when the pure ones are smaller or closer to the edge
#    lines. Within the multi-color block and within the pure block, the stack
#    then follows increasing rectangle area. The generator rejects samples unless
#    both effects occur on the same input, so the combo does not degenerate into
#    "just area" or "just minority-first".
#
# f) Visual cleanliness:
#    The scene stays sparse: one corner marker, two frame lines, and two neat
#    interior bands. Minority pixels are only 2-3 cells per decorated rectangle,
#    so objects remain easy to recognize as rectangles rather than noisy blobs.
#
# g) Color inferability:
#    No new color remapping is introduced. Colors only serve object identity plus
#    the observable cue "this rectangle is not pure". Minority colors are drawn
#    from the normal ARC palette excluding bg and marker; the rule depends on the
#    presence of mixed color, not on any specific minority hue.
#
# h) Generator validation:
#    Each candidate sample is rendered, transformed, solved with metadata using
#    the fused ordering rule, and compared against `main(input)`. Samples are
#    accepted only if they are non-degenerate under quantitative metrics.
#
# i) New solver assumptions after the edit:
#    The solver now additionally assumes minority-bearing rectangles are still
#    full rectangles and that "minority" is detectable from a single dominant
#    color plus a small off-color remainder. The generator enforces both.
# -----------------------------------------------------------------------------


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

TOP_ROW = 2
BOTTOM_ROW = 6
TOP_H = 2
BOTTOM_H = 3
CANONICAL_H = 11


def _locate_frame(g: np.ndarray) -> tuple[int, tuple[int, int], int, str, str]:
    H, W = map(int, g.shape)
    bg = int(get_bg(g))
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
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

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
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
        blue_col = int(vcol if vlen >= hlen else hcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "v" if blue_orient == "h" else "h"
    return int(bg), (int(cr), int(cc)), int(marker), blue_orient, red_orient


def _has_minority_pixels(arr: np.ndarray) -> bool:
    vals, counts = np.unique(arr, return_counts=True)
    if len(vals) <= 1:
        return False
    area = int(arr.size)
    majority = int(counts.max())
    minority_total = int(area - majority)
    return 1 <= minority_total <= area // 4


def _extract_rectangles(g: np.ndarray) -> tuple[int, tuple[int, int], str, str, list[dict]]:
    bg, (cr, cc), marker, blue_orient, red_orient = _locate_frame(g)
    H, W = map(int, g.shape)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
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
                "area": int(sub.size),
                "has_minority": bool(_has_minority_pixels(sub)),
            }
        )

    assert rects
    return int(bg), (int(cr), int(cc)), blue_orient, red_orient, rects


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _sort_rectangles(rects: list[dict], cr: int, cc: int, blue_orient: str, red_orient: str, mode: str) -> list[dict]:
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    enriched = []
    for rect in rects:
        box = tuple(map(int, rect["box"]))
        enriched.append(
            {
                **rect,
                "db": int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
                "dr": int(_dist_to_edge(box, red_orient, red_edge_idx)),
                "ab": int(_along_from_corner(box, blue_orient, cr, cc)),
                "ar": int(_along_from_corner(box, red_orient, cr, cc)),
                "priority": 0 if bool(rect["has_minority"]) else 1,
            }
        )

    def key(rect: dict) -> tuple[int, ...]:
        tail = (int(rect["ab"]), int(rect["ar"]), *tuple(map(int, rect["box"])))
        if mode == "old":
            return (int(rect["db"]), int(rect["dr"]), *tail)
        if mode == "area":
            return (int(rect["area"]), *tail)
        if mode == "priority_distance":
            return (int(rect["priority"]), int(rect["db"]), int(rect["dr"]), *tail)
        if mode == "combo":
            return (int(rect["priority"]), int(rect["area"]), *tail)
        raise ValueError(f"unknown sort mode: {mode}")

    return sorted(enriched, key=key)


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
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
    Combo rule:
    1. Detect the interior rectangles.
    2. Put minority-bearing rectangles before pure rectangles.
    3. Within each purity group, sort by area from small to large.
    4. Break ties by stable source-task spatial order.
    5. If the blue edge is vertical, rotate every rectangle 90 degrees CCW.
    """
    g = np.array(grid, dtype=int)
    bg, (cr, cc), blue_orient, red_orient, rects = _extract_rectangles(g)
    rects = _sort_rectangles(rects, cr, cc, blue_orient, red_orient, mode="combo")

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


def _sample_width(height: int, has_minority: bool) -> int:
    if height == TOP_H:
        choices = [4, 5, 6] if has_minority else [2, 3, 4, 5, 6]
    else:
        choices = [3, 4, 5, 6] if has_minority else [2, 3, 4, 5, 6]
    return int(np.random.choice(np.array(choices, dtype=int)))


def _minority_cells(h: int, w: int, n_minor: int) -> list[tuple[int, int]]:
    candidates = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
        (h // 2, w // 2),
        (0, w // 2),
        (h - 1, w // 2),
    ]
    np.random.shuffle(candidates)
    out: list[tuple[int, int]] = []
    used = set()
    for cell in candidates:
        cell = (int(cell[0]), int(cell[1]))
        if cell in used:
            continue
        used.add(cell)
        out.append(cell)
        if len(out) == int(n_minor):
            break
    return out


def _make_patch(h: int, w: int, dominant: int, has_minority: bool) -> np.ndarray:
    patch = np.full((int(h), int(w)), int(dominant), dtype=int)
    if not has_minority:
        return patch

    area = int(h * w)
    max_minor = min(3, area // 4)
    if max_minor < 2:
        raise AssertionError("minority rectangle too small")
    n_minor = 2 if max_minor == 2 else int(np.random.choice([2, 3]))
    n_colors = 1 if n_minor == 2 or np.random.rand() < 0.5 else 2
    palette = [int(c) for c in MINORITY_COLORS if int(c) not in {int(dominant), int(BG), int(MARKER)}]
    colors = np.random.choice(np.array(palette, dtype=int), size=n_colors, replace=False)
    for idx, (r, c) in enumerate(_minority_cells(int(h), int(w), int(n_minor))):
        patch[int(r), int(c)] = int(colors[idx % n_colors])
    return patch


def _row_required_width(widths: list[int]) -> int:
    return int(2 + sum(int(w) for w in widths) + 2 * (len(widths) - 1) + int(np.random.randint(1, 4)))


def _place_band(
    grid: np.ndarray,
    row: int,
    height: int,
    widths: list[int],
    statuses: list[bool],
    colors: list[int],
) -> list[dict]:
    rects = []
    c = 2
    for width, has_minority, dominant in zip(widths, statuses, colors):
        patch = _make_patch(int(height), int(width), int(dominant), bool(has_minority))
        paste(grid, patch, int(row), int(c))
        rects.append(
            {
                "patch": patch,
                "box": (int(row), int(c), int(row + height - 1), int(c + width - 1)),
                "area": int(height * width),
                "has_minority": bool(has_minority),
            }
        )
        c += int(width) + 2
    return rects


def _render_single_rect(
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    canonical_shape: tuple[int, int],
    blue_orient: str,
    corner: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full(canonical_shape, BG, dtype=int)
    r0, c0, _r1, _c1 = map(int, box)
    paste(canvas, patch, int(r0), int(c0))
    transformed = _transform(canvas, blue_orient, corner)
    coords = np.argwhere(transformed != BG)
    bbox = tuple(map(int, bbox_coords(coords)))
    return crop_bbox(transformed, bbox), bbox


def _corner_coords(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, shape)
    mapping = {
        "tl": (0, 0),
        "tr": (0, int(w - 1)),
        "bl": (int(h - 1), 0),
        "br": (int(h - 1), int(w - 1)),
    }
    return mapping[corner]


def _order_ids(rects: list[dict], input_shape: tuple[int, int], blue_orient: str, corner: str, mode: str) -> list[int]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    ordered = _sort_rectangles(rects, cr, cc, blue_orient, red_orient, mode=mode)
    return [int(rect["id"]) for rect in ordered]


def _pairwise_inversions(rects: list[dict], order_a: list[int], order_b: list[int], pair_filter) -> int:
    pos_a = {int(rid): idx for idx, rid in enumerate(order_a)}
    pos_b = {int(rid): idx for idx, rid in enumerate(order_b)}
    by_id = {int(rect["id"]): rect for rect in rects}
    inversions = 0
    ids = [int(rect["id"]) for rect in rects]
    for i, rid_a in enumerate(ids):
        for rid_b in ids[i + 1 :]:
            rect_a = by_id[int(rid_a)]
            rect_b = by_id[int(rid_b)]
            if not pair_filter(rect_a, rect_b):
                continue
            rel_a = pos_a[int(rid_a)] < pos_a[int(rid_b)]
            rel_b = pos_b[int(rid_a)] < pos_b[int(rid_b)]
            if rel_a != rel_b:
                inversions += 1
    return int(inversions)


def _variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    bg, (cr, cc), blue_orient, red_orient, rects = _extract_rectangles(np.array(grid, dtype=int))
    del bg, red_orient
    analyzed = []
    for idx, rect in enumerate(rects):
        analyzed.append(
            {
                "id": int(idx),
                "box": tuple(map(int, rect["box"])),
                "area": int(rect["area"]),
                "has_minority": bool(rect["has_minority"]),
            }
        )

    combo_ids = [int(r["id"]) for r in _sort_rectangles(analyzed, cr, cc, blue_orient, "v" if blue_orient == "h" else "h", mode="combo")]
    area_ids = [int(r["id"]) for r in _sort_rectangles(analyzed, cr, cc, blue_orient, "v" if blue_orient == "h" else "h", mode="area")]
    priority_distance_ids = [
        int(r["id"]) for r in _sort_rectangles(analyzed, cr, cc, blue_orient, "v" if blue_orient == "h" else "h", mode="priority_distance")
    ]
    old_ids = [int(r["id"]) for r in _sort_rectangles(analyzed, cr, cc, blue_orient, "v" if blue_orient == "h" else "h", mode="old")]

    minority_count = sum(int(bool(r["has_minority"])) for r in analyzed)
    pure_count = len(analyzed) - minority_count
    cross_priority_inversions = _pairwise_inversions(
        analyzed,
        combo_ids,
        area_ids,
        lambda a, b: bool(a["has_minority"]) != bool(b["has_minority"]),
    )
    same_priority_area_inversions = _pairwise_inversions(
        analyzed,
        combo_ids,
        priority_distance_ids,
        lambda a, b: bool(a["has_minority"]) == bool(b["has_minority"]),
    )

    return {
        "minority_count": int(minority_count),
        "pure_count": int(pure_count),
        "combo_vs_area_changed": bool(combo_ids != area_ids),
        "combo_vs_priority_distance_changed": bool(combo_ids != priority_distance_ids),
        "combo_vs_old_changed": bool(combo_ids != old_ids),
        "cross_priority_inversions": int(cross_priority_inversions),
        "same_priority_area_inversions": int(same_priority_area_inversions),
    }


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    count_options = [(2, 2), (2, 3), (3, 2)]
    n_top, n_bottom = count_options[int(np.random.randint(len(count_options)))]
    top_statuses = _mixed_status_pattern(int(n_top))
    bottom_statuses = _mixed_status_pattern(int(n_bottom))
    top_widths = [_sample_width(TOP_H, status) for status in top_statuses]
    bottom_widths = [_sample_width(BOTTOM_H, status) for status in bottom_statuses]

    all_areas = [TOP_H * width for width in top_widths] + [BOTTOM_H * width for width in bottom_widths]
    if len(set(all_areas)) != len(all_areas):
        return None

    width = max(_row_required_width(top_widths), _row_required_width(bottom_widths))
    canonical = np.full((CANONICAL_H, int(width)), BG, dtype=int)
    canonical[0, :] = BLUE
    canonical[:, 0] = RED
    canonical[0, 0] = MARKER

    color_pool = list(map(int, np.random.permutation(DOMINANT_COLORS)))
    rects = []
    rects.extend(_place_band(canonical, TOP_ROW, TOP_H, top_widths, top_statuses, color_pool[:n_top]))
    rects.extend(
        _place_band(
            canonical,
            BOTTOM_ROW,
            BOTTOM_H,
            bottom_widths,
            bottom_statuses,
            color_pool[n_top : n_top + n_bottom],
        )
    )

    if sum(int(rect["patch"].shape[0] != rect["patch"].shape[1]) for rect in rects) < 2:
        return None

    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    inp = _transform(canonical, blue_orient, corner)

    transformed_rects = []
    for idx, rect in enumerate(rects):
        input_patch, input_box = _render_single_rect(rect["patch"], rect["box"], canonical.shape, blue_orient, corner)
        transformed_rects.append(
            {
                "id": int(idx),
                "input_patch": input_patch,
                "box": input_box,
                "area": int(rect["area"]),
                "has_minority": bool(rect["has_minority"]),
            }
        )

    combo_ids = _order_ids(transformed_rects, inp.shape, blue_orient, corner, mode="combo")
    area_ids = _order_ids(transformed_rects, inp.shape, blue_orient, corner, mode="area")
    priority_distance_ids = _order_ids(transformed_rects, inp.shape, blue_orient, corner, mode="priority_distance")
    old_ids = _order_ids(transformed_rects, inp.shape, blue_orient, corner, mode="old")

    metrics = {
        "minority_count": sum(int(r["has_minority"]) for r in transformed_rects),
        "pure_count": len(transformed_rects) - sum(int(r["has_minority"]) for r in transformed_rects),
        "combo_vs_area_changed": combo_ids != area_ids,
        "combo_vs_priority_distance_changed": combo_ids != priority_distance_ids,
        "combo_vs_old_changed": combo_ids != old_ids,
        "cross_priority_inversions": _pairwise_inversions(
            transformed_rects,
            combo_ids,
            area_ids,
            lambda a, b: bool(a["has_minority"]) != bool(b["has_minority"]),
        ),
        "same_priority_area_inversions": _pairwise_inversions(
            transformed_rects,
            combo_ids,
            priority_distance_ids,
            lambda a, b: bool(a["has_minority"]) == bool(b["has_minority"]),
        ),
    }
    if metrics["minority_count"] < 2 or metrics["pure_count"] < 2:
        return None
    if not metrics["combo_vs_area_changed"] or not metrics["combo_vs_priority_distance_changed"]:
        return None
    if not metrics["combo_vs_old_changed"]:
        return None
    if metrics["cross_priority_inversions"] <= 0 or metrics["same_priority_area_inversions"] <= 0:
        return None

    rect_by_id = {int(rect["id"]): rect for rect in transformed_rects}
    seq = []
    for rid in combo_ids:
        arr = rect_by_id[int(rid)]["input_patch"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    expected = _stack_sequence(seq, BG)

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    meta = {
        "blue_orient": blue_orient,
        "corner": corner,
        "combo_ids": combo_ids,
        "area_ids": area_ids,
        "priority_distance_ids": priority_distance_ids,
        "old_ids": old_ids,
        **metrics,
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(800):
        built = _build_attempt()
        if built is not None:
            inp, _expected, _meta = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 combo D1/0+D8/1")
