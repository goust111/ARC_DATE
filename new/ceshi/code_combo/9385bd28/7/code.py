# Combo variant of 9385bd28 (index 7)
# combo_spec: D1/0+D10/2+D11/2
# Name: 图例色干扰+颜色匹配+无操作条目

import random
from collections import Counter, defaultdict, deque

import numpy as np

from arc_utils import get_bg


# Design analysis
# - Concrete fused appearance:
#   The bottom-left legend is still a vertical stack of `(marker_color, fill_color)` rows, but the body of the
#   grid is no longer one clean "all pixels of a color belong to one L-pair" scene. Active legend colors appear
#   as disconnected small shapes from a tiny reusable catalog. For one active color, exactly two same-shape copies
#   sit on opposite corners of a shared bbox and define the fill diamond. That same color may also contain:
#   1) one extra copy of that very same shape placed somewhere that does not make a valid diagonal pair, and
#   2) one extra component of a different shape. Several other legend rows are no-ops with no body pixels.
# - Behavior trace on one concrete color:
#   In the source task, if legend row says marker color `3`, the solver unions all non-legend `3` pixels, gets one
#   bbox, reads fixed L-corner orientation, and fills the opposite diagonal. In this combo, color `3` may have four
#   components: three T-shapes and one plus-shape. Only two of the T-shapes occupy opposite corners of one bbox; the
#   third T-shape is a same-color same-shape decoy, while the plus-shape is a same-color wrong-shape decoy.
#   The solver must therefore: split by connectivity, restrict candidates by color, compare normalized shape keys,
#   and only then validate which same-shape pair is a true diagonal pair. Another legend color may be absent
#   entirely, so that row is skipped as a no-op.
# - Complexity calibration:
#   The shape catalog is deliberately small so repeated shapes are visible across samples. The legend is restricted
#   to 4-5 rows with 2-3 no-ops, leaving exactly 2 actionable rows so the grid is not overcrowded. Every generated
#   sample forces at least one same-color same-shape invalid singleton and one same-color wrong-shape singleton,
#   making both the D1 and D11 signals visible rather than occasional. The solver still supports a 2x2 block for
#   task fidelity, but this combo's generator keeps the active structures as pairs to reserve space for the decoys.
# - Hidden-assumption audit of the original solver:
#   `_find_legend(...)` still works because the legend layout is unchanged. The original
#   `marker_pts = all pixels of one color` assumption is false: one color can now contain several logical objects,
#   including no useful pair at all for noop rows. The original bbox-over-all-pixels step is also false because
#   same-color decoys can widen the bbox. The original hard-coded L-corner scorer is invalid because marker shapes
#   are no longer fixed L anchors. The only parts that remain conceptually intact are bottom-to-top processing,
#   legend protection, future-marker protection, and direct recoloring of a true 2x2 block.
# - Placement legality:
#   All placements use a shared `occupied` mask plus the original forbidden strip around the legend, so objects never
#   overlap the legend, one another, or the protected lower-left scan region. Same-shape distractor singletons are
#   only committed if re-running the solver-side structure analysis on the hypothetical color mask still yields
#   exactly one actionable diagonal pair, namely the original one. This removes accidental second solutions.
# - Visual observability:
#   A human can see the fused rule from examples because several legend colors visibly do nothing, some active colors
#   show extra same-color components that survive unchanged, and only one pair of equal-shape copies actually drives
#   the diamond fill. Nothing depends on numeric color ids or hidden ordering beyond the visible legend order.
# - Visual cleanliness and color inference:
#   The scene stays readable because shapes are compact, disconnected, and few in number. The legend still explicitly
#   provides fill colors, so the only new inference burden is deciding which body components are the true markers.


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
    for r, c in map(tuple, np.argwhere(mask)):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(sorted(pts))
    return comps


def _component_info(pts: list[tuple[int, int]]) -> dict[str, object]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    return {
        "pts": pts,
        "bbox": (r0, c0, r1, c1),
        "shape_key": tuple(sorted((r - r0, c - c0) for r, c in pts)),
    }


def _find_color_components(g: np.ndarray, color: int, legend_mask: np.ndarray) -> list[dict[str, object]]:
    return [_component_info(pts) for pts in _components_from_mask((g == color) & (~legend_mask))]


def _is_solid_block(comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return (r1 - r0 == 1) and (c1 - c0 == 1) and len(comp["pts"]) == 4


def _pair_diagonal(
    comp_a: dict[str, object], comp_b: dict[str, object]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]] | None:
    if comp_a["shape_key"] != comp_b["shape_key"]:
        return None
    a0, a1, a2, a3 = comp_a["bbox"]
    b0, b1, b2, b3 = comp_b["bbox"]
    rmin = min(a0, b0)
    cmin = min(a1, b1)
    rmax = max(a2, b2)
    cmax = max(a3, b3)

    tl_count = int(a0 == rmin and a1 == cmin) + int(b0 == rmin and b1 == cmin)
    br_count = int(a2 == rmax and a3 == cmax) + int(b2 == rmax and b3 == cmax)
    tr_count = int(a0 == rmin and a3 == cmax) + int(b0 == rmin and b3 == cmax)
    bl_count = int(a2 == rmax and a1 == cmin) + int(b2 == rmax and b1 == cmin)

    if tl_count == 1 and br_count == 1 and tr_count == 0 and bl_count == 0:
        return (rmin, cmax), (rmax, cmin), (rmin, rmax, cmin, cmax)
    if tr_count == 1 and bl_count == 1 and tl_count == 0 and br_count == 0:
        return (rmin, cmin), (rmax, cmax), (rmin, rmax, cmin, cmax)
    return None


def _find_valid_pairs(components: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        if _is_solid_block(comp):
            continue
        groups[comp["shape_key"]].append(comp)

    pairs: list[dict[str, object]] = []
    for shape_key, group in groups.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                diag = _pair_diagonal(group[i], group[j])
                if diag is None:
                    continue
                a, b, (rmin, rmax, cmin, cmax) = diag
                pairs.append(
                    {
                        "shape_key": shape_key,
                        "pair": (group[i], group[j]),
                        "fill_corners": (a, b),
                        "bbox": (rmin, rmax, cmin, cmax),
                    }
                )
    return pairs


def _choose_structure(components: list[dict[str, object]]) -> dict[str, object] | None:
    if not components:
        return None
    if len(components) == 1 and _is_solid_block(components[0]):
        return {"kind": "block", "cells": components[0]["pts"]}

    assert all(not _is_solid_block(comp) for comp in components), "mixed block/non-block marker color is unsupported"
    pairs = _find_valid_pairs(components)
    assert len(pairs) <= 1, "marker color has multiple valid same-shape diagonal pairs"
    if not pairs:
        return None
    pair = pairs[0]
    return {
        "kind": "pair",
        "fill_corners": pair["fill_corners"],
        "bbox": pair["bbox"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill diamonds from a bottom-left legend even when legend colors contain same-color shape decoys and empty rows.

    Input structure:
    - The grid has one dominant background and a lower-left vertical legend of `(marker_color, fill_color)` rows.
    - Outside the legend, an active marker color appears as several disconnected components from a small shape catalog;
      exactly one same-color same-shape pair occupies opposite corners of a shared bounding box.
    - The same active color may also include extra components in the legend color: some are different shapes, some are
      the same shape but placed in a non-diagonal arrangement. Several legend rows may have no body pixels at all. A
      rare actionable entry can still be a lone 2x2 block.

    Rule:
    - Read legend rows from bottom to top and inspect only non-legend pixels of the current marker color.
    - If the color has no body pixels, the row is a no-op. If it has a single 2x2 block, recolor that block directly.
    - Otherwise split the color into 4-connected components, normalize each component by translation, and search among
      same-shape component pairs for the unique pair whose component bboxes occupy one diagonal of their union bbox.
    - Fill the shortest-path diamond on the opposite diagonal with the legend fill color, while preserving legend cells,
      all later legend markers, and all current marker-colored components including ignored decoys.

    Key insight:
    - Neither color alone nor shape frequency alone is sufficient here. Color narrows the search, shape identifies which
      components may belong together, and only the valid opposite-corner placement reveals the true pair; rows with no
      qualifying body components do nothing.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    out = g.copy()
    marker_order = [marker for _, marker, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        components = _find_color_components(g, marker, legend_mask)
        if not components:
            continue

        structure = _choose_structure(components)
        assert structure is not None, "non-empty legend color must have one actionable structure"

        protect = legend_mask.copy()
        future_markers = set(marker_order[i + 1 :])
        for future_marker in future_markers:
            protect |= (g == future_marker) & (~legend_mask)

        if structure["kind"] == "block":
            block_mask = np.zeros_like(g, dtype=bool)
            for r, c in structure["cells"]:
                block_mask[r, c] = True
            out[block_mask & (~protect)] = int(fill)
            continue

        protect |= (g == marker) & (~legend_mask)
        rmin, rmax, cmin, cmax = structure["bbox"]
        a, b = structure["fill_corners"]
        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (4, 5)
NOOP_ENTRY_RANGE = (2, 3)

# ── CFG: marker types ──────────────────────────────────────────────
BLOCK_PROB = 0.0
WRONG_SHAPE_SINGLE_RETRIES = 140
SAME_SHAPE_DISTRACTOR_RETRIES = 150

# ── CFG: pair bbox shape ───────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
PAIR_RETRIES = 220
GEN_RETRIES = 450

SHAPE_LIBRARY = {
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "PLUS": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    "ZIG": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "CUP": [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
}

COMPLEX_SHAPES = ["T", "ZIG", "CUP"]


def _shape_size(shape_name: str) -> tuple[int, int]:
    pts = SHAPE_LIBRARY[shape_name]
    return max(r for r, _ in pts) + 1, max(c for _, c in pts) + 1


def _translated_shape(shape_name: str, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in SHAPE_LIBRARY[shape_name]]


def _choose_fill_colors(bg: int, marker_colors: list[int]) -> list[int] | None:
    clean_pool = [color for color in range(10) if color != bg and color not in marker_colors]
    if not clean_pool:
        return None

    fills: list[int] = []
    shuffled = clean_pool[:]
    random.shuffle(shuffled)
    for i in range(len(marker_colors)):
        if i < len(shuffled):
            fills.append(shuffled[i])
        else:
            fills.append(random.choice(clean_pool))
    return fills


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


def _pick_pair_bbox(H: int, W: int, shape_name: str) -> tuple[int, int] | None:
    sh, sw = _shape_size(shape_name)
    min_h = 2 * sh
    min_w = 2 * sw
    if min_h > H or min_w > W:
        return None
    bh = random.randint(min_h, min(BBOX_H_MAX, H))
    bw = random.randint(min_w, min(BBOX_W_MAX, W))
    return bh, bw


def _try_place_shape_pair(
    H: int,
    W: int,
    color: int,
    shape_name: str,
    retries: int,
    can_place,
    do_place,
) -> bool:
    for _ in range(retries):
        size = _pick_pair_bbox(H, W, shape_name)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < 0.7 else "TRBL"
        cells = _pair_cells(rmin, cmin, bh, bw, shape_name, diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _try_place_single_shape(
    H: int,
    W: int,
    color: int,
    shape_name: str,
    retries: int,
    can_place,
    do_place,
) -> bool:
    sh, sw = _shape_size(shape_name)
    if sh > H or sw > W:
        return False
    for _ in range(retries):
        top = random.randint(0, H - sh)
        left = random.randint(0, W - sw)
        cells = _single_shape_cells(top, left, shape_name)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [color for color in range(10) if color != bg]
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)
    fill_colors = _choose_fill_colors(bg, marker_colors)
    if fill_colors is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = fill_colors[i]
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]):
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _choose_entry_types(n_entries: int) -> list[str] | None:
    noop_count = n_entries - 2
    if noop_count < NOOP_ENTRY_RANGE[0] or noop_count > NOOP_ENTRY_RANGE[1]:
        return None

    types = ["pair"] * n_entries
    indices = list(range(n_entries))
    random.shuffle(indices)
    for idx in indices[:noop_count]:
        types[idx] = "noop"

    active_indices = [idx for idx, kind in enumerate(types) if kind == "pair"]
    if len(active_indices) >= 3 and random.random() < BLOCK_PROB:
        types[random.choice(active_indices)] = "block"
    if "pair" not in types:
        return None
    return types


def _try_place_same_shape_distractor_single(
    H: int,
    W: int,
    color: int,
    shape_name: str,
    retries: int,
    current_mask: np.ndarray,
    can_place,
    do_place,
) -> bool:
    base_components = [_component_info(pts) for pts in _components_from_mask(current_mask)]
    base_structure = _choose_structure(base_components)
    if base_structure is None or base_structure["kind"] != "pair":
        return False
    base_sig = (base_structure["fill_corners"], base_structure["bbox"])

    sh, sw = _shape_size(shape_name)
    for _ in range(retries):
        if sh > H or sw > W:
            return False
        top = random.randint(0, H - sh)
        left = random.randint(0, W - sw)
        cells = _single_shape_cells(top, left, shape_name)
        if not can_place(cells):
            continue

        test_mask = current_mask.copy()
        for r, c in cells:
            test_mask[r, c] = True
        test_components = [_component_info(pts) for pts in _components_from_mask(test_mask)]
        try:
            test_structure = _choose_structure(test_components)
        except AssertionError:
            continue
        if test_structure is None or test_structure["kind"] != "pair":
            continue
        if (test_structure["fill_corners"], test_structure["bbox"]) != base_sig:
            continue

        do_place(cells, color)
        return True
    return False


def _entry_component_summary(grid: np.ndarray, legend_mask: np.ndarray, marker: int) -> dict[str, int]:
    comps = _find_color_components(grid, marker, legend_mask)
    groups: dict[tuple[tuple[int, int], ...], int] = defaultdict(int)
    for comp in comps:
        groups[comp["shape_key"]] += 1
    pairs = _find_valid_pairs(comps)
    return {
        "components": len(comps),
        "shape_groups": len(groups),
        "same_shape_triples_plus": sum(count >= 3 for count in groups.values()),
        "same_shape_singletons": sum(count == 1 for count in groups.values()),
        "valid_pairs": len(pairs),
    }


def _active_entry_changes(
    before: np.ndarray, after: np.ndarray, legend_mask: np.ndarray, legend_rows: list[tuple[int, int, int]]
) -> int:
    changed = (before != after) & (~legend_mask)
    count = 0
    for _, _, fill in legend_rows:
        if np.any(changed & (after == fill)):
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    types = _choose_entry_types(len(legend_rows))
    if types is None:
        return None

    pair_indices = [idx for idx, kind in enumerate(types) if kind == "pair"]
    if not pair_indices:
        return None
    forced_pair_idx = random.choice(pair_indices)

    def can_place(cells):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells, color):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    pair_shapes: dict[int, str] = {}
    for i, (_, marker, _) in enumerate(legend_rows):
        kind = types[i]
        if kind == "noop":
            continue
        if kind == "block":
            placed = False
            for _ in range(120):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, marker)
                    placed = True
                    break
            if not placed:
                return None
            continue

        choices = COMPLEX_SHAPES if i == forced_pair_idx else list(SHAPE_LIBRARY)
        shape_name = random.choice(choices)
        pair_shapes[i] = shape_name
        if not _try_place_shape_pair(H, W, marker, shape_name, PAIR_RETRIES, can_place, do_place):
            return None

    forced_marker = legend_rows[forced_pair_idx][1]
    forced_shape = pair_shapes[forced_pair_idx]
    forced_mask = (grid == forced_marker) & (~legend_mask)
    if not _try_place_same_shape_distractor_single(
        H,
        W,
        forced_marker,
        forced_shape,
        SAME_SHAPE_DISTRACTOR_RETRIES,
        forced_mask,
        can_place,
        do_place,
    ):
        return None

    wrong_shapes = [name for name in SHAPE_LIBRARY if name != forced_shape]
    random.shuffle(wrong_shapes)
    placed_wrong_shape = False
    for wrong_shape in wrong_shapes:
        if _try_place_single_shape(
            H,
            W,
            forced_marker,
            wrong_shape,
            WRONG_SHAPE_SINGLE_RETRIES,
            can_place,
            do_place,
        ):
            placed_wrong_shape = True
            break
    if not placed_wrong_shape:
        return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if np.array_equal(solved, grid):
        return None
    if _active_entry_changes(grid, solved, legend_mask, legend_rows) < 2:
        return None

    noop_count = 0
    active_count = 0
    forced_summary = None
    for i, (_, marker, _) in enumerate(legend_rows):
        marker_pixels = int(np.count_nonzero((grid == marker) & (~legend_mask)))
        if types[i] == "noop":
            if marker_pixels != 0:
                return None
            noop_count += 1
        else:
            if marker_pixels == 0:
                return None
            active_count += 1
        if i == forced_pair_idx:
            forced_summary = _entry_component_summary(grid, legend_mask, marker)

    if noop_count < 2 or noop_count > 3 or active_count < 2:
        return None
    if forced_summary is None:
        return None
    if forced_summary["valid_pairs"] != 1:
        return None
    if forced_summary["same_shape_triples_plus"] < 1:
        return None
    if forced_summary["same_shape_singletons"] < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
