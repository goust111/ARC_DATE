# Combo variant of 9385bd28 (index 19)
# combo_spec: D1/0+D6/0+D13/0+D14/0
# Name: 颜色匹配+反转顺序+一对多+无保护

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg


# Design analysis
# - Concrete behavior and trace:
#   The lower-left legend remains a vertical `(marker_color, fill_color)` stack, but each legend color now owns
#   several disconnected marker pairs instead of one object. Markers are no longer fixed L-corners: they come
#   from a small shape catalog, and a valid pair is defined by "same color + same shape". Example trace:
#   original task: gather all color-6 pixels, infer one bbox, fill once, process bottom->top with protection.
#   fused task: gather all color-6 components, group by normalized shape, every shape that appears exactly twice
#   defines one bbox, fill all such bboxes, execute legend entries top->bottom, and allow every fill to repaint
#   legend cells and marker cells. The visible difference is therefore simultaneous: shape matching is necessary,
#   one legend row acts multiple times, overlap priority is reversed, and marker/legend preservation disappears.
# - Complexity calibration:
#   To keep the rule observable rather than noisy, each active legend entry has only 2 or 3 pairs, with distinct
#   shapes per entry, drawn from a small reusable catalog. At least one entry also gets a same-color wrong-shape
#   singleton, so color-only grouping is insufficient. The generator forces a top-vs-bottom overlap with distinct
#   fill colors and also forces an early top-row fill to cross lower legend cells, making both the reversed order
#   and the missing protection visible in a few examples instead of being rare edge cases.
# - Hidden assumptions audited against the original solver:
#   The original `all pixels of one marker color -> one bbox` assumption is invalid once one color can own
#   multiple pairs plus distractors. The original hard-coded L-anchor logic is also invalid because marker shape
#   is now variable. The original protected painting logic is invalid because D13 removes every protection rule.
#   What still survives is `_find_legend`, the idea that geometry must be read from the untouched input `g`, and
#   the opposite-diagonal diamond fill once one matched pair bbox has been identified.
# - Placement legality:
#   The generator keeps marker cells non-overlapping, in bounds, and outside the legend scan corridor. Same-color
#   components are kept 4-disconnected, and same-color pair bboxes are spaced apart so one shape group cannot
#   merge into another. Different colors may have overlapping fill diamonds, but marker cells themselves never
#   overlap. Focus placements are rejected unless the intended overlap and destructive repaint effects are present.
# - Visual observability:
#   Humans can see all driving cues directly: one legend row maps a color to a fill; the same row has multiple
#   repeated-shape pairs; a wrong-shape singleton of the same color shows that color alone is not enough; and
#   overlapped regions reveal that lower legend rows win despite the legend being read bottom->top. No rule depends
#   on color ids or any hidden ordering beyond visible legend row position.
# - Visual cleanliness and color inference:
#   Shape vocabulary stays small, same-color pairs do not self-overlap, and only one forced interaction cluster is
#   required, which keeps the scenes readable. Fill colors are still given explicitly by the legend, so the new
#   burden is structural reasoning rather than arbitrary color remapping.
# - New solver assumptions and generator guarantees:
#   The solver assumes every useful marker color splits into components whose normalized-shape multiplicities are
#   only 1 or 2, with at least two multiplicity-2 groups. The generator guarantees that by construction. The
#   solver also assumes each multiplicity-2 group occupies exactly one bbox diagonal; the generator places every
#   matched pair flush to opposite bbox corners and rejects any sample whose recovered metrics do not show the
#   fused D1/D6/D13/D14 behaviors.


SHAPE_LIBRARY: dict[str, list[tuple[int, int]]] = {
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "PLUS": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    "ZIG": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "CUP": [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
}


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size:
            start_r = r
            start_c = int(nz[0])
            break
    assert start_r is not None and start_c is not None and start_c < w - 1, (
        "legend must exist at bottom-left with at least 2 columns"
    )

    entries: list[tuple[int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else bg
        entries.append((r, marker, fill))
    return c, entries


def _build_legend_mask(shape: tuple[int, int], legend_c: int, entries: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    legend_mask = np.zeros((h, w), dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True
    return legend_mask


def _diamond_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path between corners a and b:
      dist(a,p) + dist(p,b) == dist(a,b)
    """
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _component_info(pts: list[tuple[int, int]]) -> dict[str, object]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    rel = tuple(sorted((r - r0, c - c0) for r, c in pts))
    return {
        "pts": pts,
        "bbox": (r0, c0, r1, c1),
        "shape_key": rel,
    }


def _find_color_components(g: np.ndarray, color: int, legend_mask: np.ndarray) -> list[dict[str, object]]:
    comps = _components_from_mask((g == color) & (~legend_mask))
    return [_component_info(pts) for pts in comps]


def _pair_diagonal(pair: list[dict[str, object]]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]]:
    b0 = pair[0]["bbox"]
    b1 = pair[1]["bbox"]
    assert isinstance(b0, tuple) and isinstance(b1, tuple)
    rmin = min(int(b0[0]), int(b1[0]))
    cmin = min(int(b0[1]), int(b1[1]))
    rmax = max(int(b0[2]), int(b1[2]))
    cmax = max(int(b0[3]), int(b1[3]))

    tl_count = sum(int(comp["bbox"][0] == rmin and comp["bbox"][1] == cmin) for comp in pair)
    br_count = sum(int(comp["bbox"][2] == rmax and comp["bbox"][3] == cmax) for comp in pair)
    tr_count = sum(int(comp["bbox"][0] == rmin and comp["bbox"][3] == cmax) for comp in pair)
    bl_count = sum(int(comp["bbox"][2] == rmax and comp["bbox"][1] == cmin) for comp in pair)

    if tl_count == 1 and br_count == 1 and tr_count == 0 and bl_count == 0:
        return (rmin, cmax), (rmax, cmin), (rmin, rmax, cmin, cmax)
    if tr_count == 1 and bl_count == 1 and tl_count == 0 and br_count == 0:
        return (rmin, cmin), (rmax, cmax), (rmin, rmax, cmin, cmax)
    raise AssertionError("matched components must occupy exactly one bbox diagonal")


def _entry_pair_infos(g: np.ndarray, marker: int, legend_mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], dict[tuple[tuple[int, int], ...], int]]:
    components = _find_color_components(g, marker, legend_mask)
    marker_mask = (g == marker) & (~legend_mask)
    if not components:
        return marker_mask, [], {}

    shape_groups: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        shape_groups[comp["shape_key"]].append(comp)

    multiplicities = {shape: len(group) for shape, group in shape_groups.items()}
    assert all(count in (1, 2) for count in multiplicities.values()), "shape multiplicities must be 1 or 2"

    pair_infos: list[dict[str, object]] = []
    for group in shape_groups.values():
        if len(group) != 2:
            continue
        a, b, (rmin, rmax, cmin, cmax) = _pair_diagonal(group)
        pair_infos.append(
            {
                "bbox": (rmin, cmin, rmax, cmax),
                "fill_ends": (a, b),
            }
        )

    pair_infos.sort(key=lambda info: tuple(info["bbox"]))
    return marker_mask, pair_infos, multiplicities


def _solve_with_options(grid: np.ndarray, process_top_to_bottom: bool, protect_markers: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = _build_legend_mask((h, w), legend_c, entries)
    out = g.copy()
    marker_order = [marker for _, marker, _ in entries]
    index_order = range(len(entries) - 1, -1, -1) if process_top_to_bottom else range(len(entries))

    for i in index_order:
        _, marker, fill = entries[i]
        marker_mask, pair_infos, _ = _entry_pair_infos(g, int(marker), legend_mask)
        if not pair_infos:
            continue

        protect = None
        if protect_markers:
            protect = legend_mask.copy()
            future_markers = marker_order[:i] if process_top_to_bottom else marker_order[i + 1 :]
            for future_marker in future_markers:
                protect |= (g == future_marker) & (~legend_mask)
            protect |= marker_mask

        for pair in pair_infos:
            rmin, cmin, rmax, cmax = pair["bbox"]
            a, b = pair["fill_ends"]
            assert isinstance(a, tuple) and isinstance(b, tuple)
            diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
            if protect is None:
                out[diamond] = int(fill)
            else:
                out[diamond & (~protect)] = int(fill)

    return out


def _legacy_order_reference(grid: np.ndarray) -> np.ndarray:
    return _solve_with_options(grid, process_top_to_bottom=False, protect_markers=False)


def _protected_topdown_reference(grid: np.ndarray) -> np.ndarray:
    return _solve_with_options(grid, process_top_to_bottom=True, protect_markers=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint several same-color, same-shape marker pairs using a lower-left legend.

    Input structure:
    - The grid has one dominant background color and a vertical 2-column legend near the bottom-left.
    - Each legend row still gives `(marker_color, fill_color)`, read bottom-to-top to recover the mapping.
    - Away from the legend, one marker color may appear in several disconnected components: 2-3 same-color pairs of
      matching shapes, plus optional same-color singleton distractors of other shapes.

    Rule:
    - For each legend entry, inspect only the original non-legend pixels of that marker color, split them into
      4-connected components, and normalize each component by translation.
    - Every shape that appears exactly twice for that color defines one matched pair; each pair occupies one diagonal
      of its combined bounding box, so fill the opposite diagonal diamond with the legend's fill color.
    - Execute legend entries from top to bottom, not bottom to top, and apply every fill destructively with no pixel
      protection: legend cells, current markers, and later markers may all be repainted.

    Key insight: Color chooses which components belong to one legend row, shape decides which components pair up, and
    because lower legend rows are executed later without protection, they can overwrite both earlier fills and earlier
    marker evidence.
    """
    return _solve_with_options(grid, process_top_to_bottom=True, protect_markers=False)


# --------------- generator ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (12, 19)
GRID_W_RANGE = (12, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 4)

# ── CFG: per-entry multiplicity ────────────────────────────────────
PAIR_COUNT_CHOICES = (2, 3)
PAIR_COUNT_WEIGHTS = (0.7, 0.3)
SHAPE_DISTRACTOR_PROB = 0.45

# ── CFG: shape-pair placement ──────────────────────────────────────
PAIR_BBOX_H_MAX = 10
PAIR_BBOX_W_MAX = 10
PAIR_RETRIES = 180
SINGLE_RETRIES = 120
FOCUS_RETRIES = 320
FOCUS_PAIR_MIN_SPAN = 8
SAME_COLOR_BBOX_MARGIN = 1

# ── CFG: forced fused behavior ─────────────────────────────────────
MIN_FUTURE_LEGEND_HITS = 1
MIN_FILL_OVERLAP_CELLS = 2

# ── CFG: retry budget ──────────────────────────────────────────────
GEN_RETRIES = 450


def _shape_size(shape_name: str) -> tuple[int, int]:
    pts = SHAPE_LIBRARY[shape_name]
    return max(r for r, _ in pts) + 1, max(c for _, c in pts) + 1


def _translated_shape(shape_name: str, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in SHAPE_LIBRARY[shape_name]]


def _pair_cells(rmin: int, cmin: int, bh: int, bw: int, shape_name: str, diag: str) -> list[tuple[int, int]]:
    sh, sw = _shape_size(shape_name)
    if diag == "TLBR":
        a_top, a_left = rmin, cmin
        b_top, b_left = rmin + bh - sh, cmin + bw - sw
    else:
        a_top, a_left = rmin, cmin + bw - sw
        b_top, b_left = rmin + bh - sh, cmin
    return _translated_shape(shape_name, a_top, a_left) + _translated_shape(shape_name, b_top, b_left)


def _single_shape_cells(top: int, left: int, shape_name: str) -> list[tuple[int, int]]:
    return _translated_shape(shape_name, top, left)


def _cells_to_mask(H: int, W: int, cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _pick_pair_bbox(H: int, W: int, shape_name: str, focus: bool) -> tuple[int, int] | None:
    sh, sw = _shape_size(shape_name)
    min_h = 2 * sh
    min_w = 2 * sw
    max_h = min(PAIR_BBOX_H_MAX, H)
    max_w = min(PAIR_BBOX_W_MAX, W)
    if min_h > max_h or min_w > max_w:
        return None
    for _ in range(20 if focus else 1):
        bh = random.randint(min_h, max_h)
        bw = random.randint(min_w, max_w)
        if not focus or bh + bw >= FOCUS_PAIR_MIN_SPAN:
            return bh, bw
    return None


def _pair_layout(H: int, W: int, rmin: int, cmin: int, bh: int, bw: int, shape_name: str, diag: str) -> dict[str, object]:
    cells = _pair_cells(rmin, cmin, bh, bw, shape_name, diag)
    marker_mask = _cells_to_mask(H, W, cells)
    if diag == "TLBR":
        a, b = (rmin, cmin + bw - 1), (rmin + bh - 1, cmin)
    else:
        a, b = (rmin, cmin), (rmin + bh - 1, cmin + bw - 1)
    fill_mask = _diamond_mask(H, W, rmin, rmin + bh - 1, cmin, cmin + bw - 1, a, b)
    return {
        "cells": cells,
        "bbox": (rmin, cmin, rmin + bh - 1, cmin + bw - 1),
        "marker_mask": marker_mask,
        "fill_mask": fill_mask,
        "shape_name": shape_name,
        "diag": diag,
    }


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _try_place_shape_pair(
    H: int,
    W: int,
    color: int,
    shape_name: str,
    retries: int,
    can_place,
    do_place,
    accept=None,
    focus: bool = False,
):
    for _ in range(retries):
        size = _pick_pair_bbox(H, W, shape_name, focus=focus)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < 0.7 else "TRBL"
        layout = _pair_layout(H, W, rmin, cmin, bh, bw, shape_name, diag)
        cells = layout["cells"]
        assert isinstance(cells, list)
        if not can_place(cells, color, layout["bbox"]):
            continue
        if accept is not None and not accept(layout):
            continue
        do_place(cells, color, layout["bbox"])
        return layout
    return None


def _try_place_single_shape(H: int, W: int, color: int, shape_name: str, retries: int, can_place, do_place) -> bool:
    sh, sw = _shape_size(shape_name)
    if sh > H or sw > W:
        return False
    for _ in range(retries):
        top = random.randint(0, H - sh)
        left = random.randint(0, W - sw)
        cells = _single_shape_cells(top, left, shape_name)
        if can_place(cells, color, None):
            do_place(cells, color, None)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    max_entries = min(LEGEND_ENTRY_RANGE[1], legend_bot + 1, 9)
    if max_entries < LEGEND_ENTRY_RANGE[0]:
        return None
    n_entries = random.randint(LEGEND_ENTRY_RANGE[0], max_entries)

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _force_distinct_focus_fills(grid: np.ndarray, legend_rows: list[tuple[int, int, int]], legend_c: int) -> list[tuple[int, int, int]]:
    if legend_rows[0][2] != legend_rows[-1][2]:
        return legend_rows
    replacements = [c for c in range(10) if c != legend_rows[0][2]]
    new_fill = random.choice(replacements)
    top_row, top_marker, _ = legend_rows[-1]
    legend_rows = legend_rows.copy()
    legend_rows[-1] = (top_row, top_marker, new_fill)
    grid[top_row, legend_c + 1] = new_fill
    return legend_rows


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]) -> np.ndarray:
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _choose_pair_counts(n_entries: int) -> list[int]:
    return list(random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=n_entries))


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    legend_mask = _build_legend_mask(g.shape, legend_c, entries)

    order_ref = _legacy_order_reference(g)
    protected_ref = _protected_topdown_reference(g)
    combo_out = main(g)

    pair_count_total = 0
    multi_pair_entries = 0
    distractor_entries = 0
    future_legend_hits = 0

    for i, (_, marker, _) in enumerate(entries):
        _, pair_infos, multiplicities = _entry_pair_infos(g, int(marker), legend_mask)
        pair_count_total += len(pair_infos)
        if len(pair_infos) >= 2:
            multi_pair_entries += 1
        if any(count == 1 for count in multiplicities.values()):
            distractor_entries += 1
        future_rows = [row for row, _, _ in entries[:i]]
        for pair in pair_infos:
            rmin, cmin, rmax, cmax = pair["bbox"]
            a, b = pair["fill_ends"]
            assert isinstance(a, tuple) and isinstance(b, tuple)
            fill_mask = _diamond_mask(g.shape[0], g.shape[1], rmin, rmax, cmin, cmax, a, b)
            for row in future_rows:
                for cc in (legend_c, legend_c + 1):
                    if 0 <= cc < g.shape[1] and fill_mask[row, cc]:
                        future_legend_hits += 1

    return {
        "order_delta": int(np.count_nonzero(combo_out != order_ref)),
        "protection_delta": int(np.count_nonzero(combo_out != protected_ref)),
        "pair_count_total": pair_count_total,
        "multi_pair_entries": multi_pair_entries,
        "distractor_entries": distractor_entries,
        "future_legend_hits": future_legend_hits,
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result
    legend_rows = _force_distinct_focus_fills(grid, legend_rows, legend_c)

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()
    color_masks: dict[int, set[tuple[int, int]]] = {color: set() for color in marker_colors}
    color_bboxes: dict[int, list[tuple[int, int, int, int]]] = {color: [] for color in marker_colors}

    pair_counts = _choose_pair_counts(len(legend_rows))
    pair_shapes: dict[int, list[str]] = {}
    distractor_shapes: dict[int, str | None] = {}
    forced_distractor_idx = random.randrange(len(legend_rows))
    shape_names = list(SHAPE_LIBRARY)
    for i in range(len(legend_rows)):
        chosen = random.sample(shape_names, pair_counts[i])
        pair_shapes[i] = chosen
        leftover = [shape for shape in shape_names if shape not in chosen]
        use_distractor = i == forced_distractor_idx or (leftover and random.random() < SHAPE_DISTRACTOR_PROB)
        distractor_shapes[i] = random.choice(leftover) if use_distractor and leftover else None

    top_idx = len(legend_rows) - 1
    bottom_idx = 0

    future_legend_mask = np.zeros((H, W), dtype=bool)
    for row, _, _ in legend_rows[:top_idx]:
        future_legend_mask[row, legend_c] = True
        future_legend_mask[row, legend_c + 1] = True

    def can_place(cells: list[tuple[int, int]], color: int, bbox) -> bool:
        cellset = set(cells)
        for r, c in cellset:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        for r, c in cellset:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in color_masks[color] and (nr, nc) not in cellset:
                    return False
        if bbox is not None:
            assert isinstance(bbox, tuple)
            if any(_boxes_too_close(bbox, prev_bbox, SAME_COLOR_BBOX_MARGIN) for prev_bbox in color_bboxes[color]):
                return False
        return True

    def do_place(cells: list[tuple[int, int]], color: int, bbox) -> None:
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
            color_masks[color].add((r, c))
        if bbox is not None:
            color_bboxes[color].append(bbox)

    top_marker = legend_rows[top_idx][1]
    top_shape = pair_shapes[top_idx][0]

    def accept_top(layout) -> bool:
        fill_mask = layout["fill_mask"]
        assert isinstance(fill_mask, np.ndarray)
        return int(np.count_nonzero(fill_mask & future_legend_mask)) >= MIN_FUTURE_LEGEND_HITS

    top_layout = _try_place_shape_pair(
        H,
        W,
        top_marker,
        top_shape,
        FOCUS_RETRIES,
        can_place,
        do_place,
        accept=accept_top,
        focus=True,
    )
    if top_layout is None:
        return None

    bottom_marker = legend_rows[bottom_idx][1]
    bottom_shape = pair_shapes[bottom_idx][0]

    def accept_bottom(layout) -> bool:
        overlap = layout["fill_mask"] & top_layout["fill_mask"]
        overlap &= ~(layout["marker_mask"] | top_layout["marker_mask"] | legend_mask)
        return int(np.count_nonzero(overlap)) >= MIN_FILL_OVERLAP_CELLS

    bottom_layout = _try_place_shape_pair(
        H,
        W,
        bottom_marker,
        bottom_shape,
        FOCUS_RETRIES,
        can_place,
        do_place,
        accept=accept_bottom,
        focus=True,
    )
    if bottom_layout is None:
        return None

    for i, (_, marker, _) in enumerate(legend_rows):
        for shape_name in pair_shapes[i]:
            if i == top_idx and shape_name == top_shape:
                continue
            if i == bottom_idx and shape_name == bottom_shape:
                continue
            if _try_place_shape_pair(
                H,
                W,
                marker,
                shape_name,
                PAIR_RETRIES,
                can_place,
                do_place,
                focus=False,
            ) is None:
                return None

    for i, (_, marker, _) in enumerate(legend_rows):
        shape_name = distractor_shapes[i]
        if shape_name is None:
            continue
        if not _try_place_single_shape(H, W, marker, shape_name, SINGLE_RETRIES, can_place, do_place):
            return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    if metrics["multi_pair_entries"] != len(legend_rows):
        return None
    if metrics["distractor_entries"] <= 0:
        return None
    if metrics["order_delta"] <= 0:
        return None
    if metrics["protection_delta"] <= 0:
        return None
    if metrics["future_legend_hits"] < MIN_FUTURE_LEGEND_HITS:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
