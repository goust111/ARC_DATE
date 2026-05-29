# Combo variant of f560132c (index 28)
# combo_spec: D3/0+D9/1+D11/0+D16/1
# Name: 非矩形轮廓+6精灵+分割线+阶梯分割

import math
import random
import sys

import numpy as np

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - Original behavior:
#      4 disconnected sprites, one of them the multicolor legend, reassemble into one solid rectangle.
#      The solver inferred `k=2` from `sqrt(len(comps))`, tiled all four pieces into a full square,
#      and recolored each piece by the 2x2 legend cell containing its centroid.
#    - Combo behavior:
#      there are now exactly 6 disconnected sprites and the output is not a solid rectangle. Instead it is
#      one tight irregular silhouette whose occupied area is split into 6 regions arranged as a 2x3 or 3x2
#      logical legend grid. Adjacent regions do not touch directly: a one-pixel divider path sits between
#      them, and those dividers are staircase polylines rather than straight cuts.
#    - Small trace:
#      take a 2x3 sample. The assembled target might be a bottom-right-notched silhouette. The top-left
#      region is the legend piece and contains a 2x3 palette patch. The solver must identify the legend,
#      infer whether the palette is 2x3 or 3x2, match all 6 scattered piece shapes against one candidate
#      irregular template, then draw the template's staircase divider cells in the fixed divider color and
#      recolor the 6 regions by row-major legend position. The original centroid-on-rectangle logic is
#      invalid here because there is no full rectangle, there are 6 regions, and divider cells are not
#      owned by any piece.
# b) Complexity calibration
#    - The variant keeps the freedom small and human-readable by using a compact catalog of non-rectangular
#      silhouettes and two staircase pattern banks. This is enough to make examples visibly varied, while
#      still letting the solver reason over a finite set of structural templates.
#    - Region count rises from 4 to 6, and the legend becomes 2x3 or 3x2, so the assembly/color rule is
#      strictly richer than the base task. Body colors are intentionally reused because ARC only has 9
#      nonzero colors and we need 6 palette colors plus a divider color.
# c) Original assumptions that break
#    - `k = round(sqrt(len(comps)))` fails because 6 is not a perfect square. Replaced with the visible
#      palette shapes {(2,3), (3,2)}.
#    - "all piece cells cover a full rectangle" fails because the silhouette has notches and the divider
#      cells are separate output structure not present in any input piece.
#    - "piece color in output comes from centroid quadrant" fails because the 6 colors are attached to the
#      template's explicit logical regions, not derived from centroid quantization.
#    - "exact-cover over free placements" is overkill here. The generator constructs a small family of
#      explicit labeled templates, so the combo solver matches piece shapes against those template regions
#      directly and rejects ambiguous template matches.
# d) Placement legality
#    - Generation remains output-first. A labeled template is chosen first, then each of the 6 region masks
#      becomes one scattered input sprite. Pieces are placed with `random_free_location`, border, and padding,
#      so they never overlap or merge.
#    - The legend palette is embedded only in an interior 2x3 or 3x2 window fully surrounded by legend-body
#      cells, so the legend remains one 4-connected component.
# e) Visual observability
#    - Humans can see all four fused changes directly:
#      there are 6 sprites in the input, the output outline has visible missing corners/side bites, the
#      output contains a dedicated divider color that traces one-pixel stair lines, and the legend visibly
#      has 6 cells rather than 4.
#    - No hidden ordering by numeric color id is used. Region colors are taken from the visible legend in
#      row-major order of the 2x3 or 3x2 layout.
# f) Visual cleanliness
#    - Divider cells are the only extra overlay and live on their own cells, so nothing is stacked on top of
#      a region color. The template family avoids tiny slivers and enforces 4-connected regions, keeping the
#      output readable despite the irregular cuts.
# g) Color inferability
#    - The palette always contains 6 distinct colors and the divider color is excluded from the palette.
#      Input body colors may repeat, but that does not encode hidden data; it merely avoids exhausting the
#      ARC palette. All output recoloring information still comes from the legend's visible 6-cell patch.
# h) New solver assumptions and generator guarantees
#    - Solver assumption: exactly one input component is multicolor and is the legend. Generator guarantee:
#      only the legend contains a multicolor patch.
#    - Solver assumption: the assembled target belongs to the template catalog below and the legend occupies
#      region 0 (the template's top-left logical cell). Generator guarantee: every sample is built from that
#      catalog.
#    - Solver assumption: exactly one candidate template matches the 6 piece shapes and the visible palette
#      orientation. Generator guarantee: ambiguous templates are rejected before the sample is accepted.


NUM_PIECES = 6
DIVIDER_COLOR = 5
BG_LABEL = -1
DIVIDER_LABEL = 99
BASE_SHAPE = (13, 18)  # 2x3 base templates; transpose gives 18x13 for 3x2
ROW_BASE = 5
COL1_BASE = 5
COL2_BASE = 11
PATTERN_BANKS = [
    {
        "name": "A",
        "row_offsets": [0, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 0, 0, 1, 0, -1, 0],
        "col1_offsets": [0, 0, 1, 1, 0, -1, 0, 1, 0, -1, 0, 0, 1],
        "col2_offsets": [0, 1, 0, -1, -1, 0, 1, 0, 1, 0, -1, 0, 0],
    },
    {
        "name": "B",
        "row_offsets": [0, 1, -1, -1, 0, 1, 0, -1, -1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        "col1_offsets": [0, 1, 0, -1, -1, 0, 1, 0, 1, 0, 1, 0, -1],
        "col2_offsets": [-1, 1, 1, 1, 0, 0, -1, 0, -1, 1, 1, 0, -1],
    },
]
TEMPLATE_SPECS = [
    {"family": "ne", "cut_h": 3, "cut_w": 4},
    {"family": "se", "cut_h": 3, "cut_w": 4},
    {"family": "sw", "cut_h": 3, "cut_w": 4},
    {"family": "tbot", "cut_h": 3, "cut_w": 3},
    {"family": "rightbite", "cut_h": 3, "cut_w": 3},
]
INPUT_SIZE_LO = 28
INPUT_SIZE_HI = 36
DENSITY_LO = 0.16
DENSITY_HI = 0.24
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 600


def _normalize_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, _ = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _, w = _normalize_coords(coords)
    flp = [(r, w - 1 - c) for r, c in norm]
    return _normalize_coords(flp)


def _dihedral_orientations(
    coords: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]] = []
    cur = coords
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, int(h), int(w), int(rot_k), False))
        cur = _rot90(cur)[0]
    cur = _flip_h(coords)[0]
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, int(h), int(w), int(rot_k), True))
        cur = _rot90(cur)[0]
    return out


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, map_shape: tuple[int, int]) -> np.ndarray | None:
    ph, pw = map_shape
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    if hh < ph or ww < pw:
        return None
    best = None
    best_rc = None
    for rr in range(hh - ph + 1):
        for cc in range(ww - pw + 1):
            window = sub[rr : rr + ph, cc : cc + pw]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best is None or key > best:
                best = key
                best_rc = (rr, cc)
    if best_rc is None or best is None or best[0] < ph * pw:
        return None
    rr, cc = best_rc
    out = np.asarray(sub[rr : rr + ph, cc : cc + pw], dtype=int).copy()
    if len(set(map(int, out.ravel()))) != ph * pw:
        return None
    return out


def _fill_holes_in_component(
    g: np.ndarray, pts: np.ndarray, bg: int
) -> tuple[tuple[tuple[int, int], ...], int]:
    r0, c0, r1, c1 = bbox_coords(pts)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    occ = np.zeros((hh, ww), dtype=bool)
    for r, c in pts:
        occ[r - r0, c - c0] = True

    bgmask = ~occ
    vis = np.zeros_like(bgmask, dtype=bool)
    st: list[tuple[int, int]] = []

    for rr in range(hh):
        for cc in range(ww):
            if rr in (0, hh - 1) or cc in (0, ww - 1):
                if bgmask[rr, cc] and not vis[rr, cc]:
                    vis[rr, cc] = True
                    st.append((rr, cc))
                    while st:
                        r, c = st.pop()
                        for r2, c2 in neighbors4(r, c):
                            if 0 <= r2 < hh and 0 <= c2 < ww and bgmask[r2, c2] and not vis[r2, c2]:
                                vis[r2, c2] = True
                                st.append((r2, c2))

    holes = bgmask & ~vis
    hole_coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(holes))]
    filled = tuple(sorted(set(map(tuple, pts)).union(hole_coords)))
    return filled, int(holes.sum())


def _mask_connected(mask: np.ndarray) -> bool:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not pts:
        return False
    todo = [pts[0]]
    seen = {pts[0]}
    ptset = set(pts)
    while todo:
        r, c = todo.pop()
        for rr, cc in neighbors4(r, c):
            nxt = (int(rr), int(cc))
            if nxt in ptset and nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return len(seen) == len(pts)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    opts = []
    seen = set()
    for k_rot in range(4):
        rot = np.rot90(arr, k_rot)
        for oriented in (rot, np.fliplr(rot)):
            pts = np.argwhere(oriented > 0)
            r0, c0, r1, c1 = bbox_coords(pts)
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return random.choice(opts)


def _palette_spots(mask: np.ndarray, map_shape: tuple[int, int]) -> list[tuple[int, int]]:
    ph, pw = map_shape
    spots = []
    h, w = mask.shape
    for r in range(1, h - ph):
        for c in range(1, w - pw):
            if mask[r : r + ph, c : c + pw].all() and mask[r - 1 : r + ph + 1, c - 1 : c + pw + 1].all():
                spots.append((int(r), int(c)))
    return spots


def _transpose_template_grid(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.full((w, h), BG_LABEL, dtype=int)
    for r in range(h):
        for c in range(w):
            v = int(grid[r, c])
            if v in (BG_LABEL, DIVIDER_LABEL):
                out[c, r] = v
            else:
                rr = v // 3
                cc = v % 3
                out[c, r] = int(cc * 2 + rr)
    return out


def _occupancy_mask(base_shape: tuple[int, int], family: str, cut_h: int, cut_w: int) -> np.ndarray:
    h, w = base_shape
    occ = np.ones((h, w), dtype=bool)
    if family == "ne":
        occ[:cut_h, w - cut_w :] = False
    elif family == "se":
        occ[h - cut_h :, w - cut_w :] = False
    elif family == "sw":
        occ[h - cut_h :, :cut_w] = False
    elif family == "tbot":
        occ[h - cut_h :, :cut_w] = False
        occ[h - cut_h :, w - cut_w :] = False
    elif family == "rightbite":
        occ[cut_h : h - cut_h, w - cut_w :] = False
    else:
        raise ValueError(f"unknown family: {family}")
    return occ


def _build_template(bank: dict, spec: dict, transpose: bool) -> dict | None:
    h, w = BASE_SHAPE
    row_split = np.array([ROW_BASE + bank["row_offsets"][i % len(bank["row_offsets"])] for i in range(w)], dtype=int)
    col1_split = np.array([COL1_BASE + bank["col1_offsets"][i % len(bank["col1_offsets"])] for i in range(h)], dtype=int)
    col2_split = np.array([COL2_BASE + bank["col2_offsets"][i % len(bank["col2_offsets"])] for i in range(h)], dtype=int)
    if np.any(col2_split - col1_split < 4):
        return None

    occ = _occupancy_mask(BASE_SHAPE, spec["family"], spec["cut_h"], spec["cut_w"])
    labels = np.full((h, w), BG_LABEL, dtype=int)
    for r in range(h):
        for c in range(w):
            if not occ[r, c]:
                continue
            if r == row_split[c] or c == col1_split[r] or c == col2_split[r]:
                labels[r, c] = DIVIDER_LABEL
            else:
                row = 0 if r < row_split[c] else 1
                col = 0 if c < col1_split[r] else 1 if c < col2_split[r] else 2
                labels[r, c] = int(row * 3 + col)

    if transpose:
        labels = _transpose_template_grid(labels)

    region_masks = [labels == rid for rid in range(NUM_PIECES)]
    if not all(mask.any() and _mask_connected(mask) for mask in region_masks):
        return None
    if not ((labels[0, :] != BG_LABEL).any() and (labels[-1, :] != BG_LABEL).any() and (labels[:, 0] != BG_LABEL).any() and (labels[:, -1] != BG_LABEL).any()):
        return None

    region_norms = []
    region_orient_sets = []
    for mask in region_masks:
        pts = np.argwhere(mask)
        rel = tuple((int(r), int(c)) for r, c in pts)
        norm, _, _ = _normalize_coords(rel)
        region_norms.append(norm)
        region_orient_sets.append({coords for coords, _, _, _, _ in _dihedral_orientations(norm)})

    return {
        "id": f"{bank['name']}_{spec['family']}_{'32' if transpose else '23'}",
        "labels": labels,
        "shape": labels.shape,
        "map_shape": (3, 2) if transpose else (2, 3),
        "region_masks": region_masks,
        "region_norms": region_norms,
        "region_orient_sets": region_orient_sets,
        "divider_mask": labels == DIVIDER_LABEL,
        "occupied_mask": labels != BG_LABEL,
        "family": spec["family"],
        "transpose": bool(transpose),
    }


def _build_template_catalog() -> list[dict]:
    out = []
    for bank in PATTERN_BANKS:
        for spec in TEMPLATE_SPECS:
            for transpose in (False, True):
                tpl = _build_template(bank, spec, transpose)
                if tpl is not None:
                    out.append(tpl)
    assert out, "template catalog must not be empty"
    return out


TEMPLATES = _build_template_catalog()


def _build_piece(g: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    filled_abs, _ = _fill_holes_in_component(g, pts, bg)
    r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
    rel = tuple((r - r0, c - c0) for r, c in filled_abs)
    rel_norm, _, _ = _normalize_coords(rel)
    oris = _dihedral_orientations(rel_norm)
    return {
        "oris": oris,
        "area": len(filled_abs),
        "shape_set": {coords for coords, _, _, _, _ in oris},
    }


def _extract_scene(grid: np.ndarray) -> tuple[np.ndarray, int, list[np.ndarray], int, list[dict], dict[tuple[int, int], np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == NUM_PIECES, "combo variant expects exactly 6 disconnected sprites"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1 and max_distinct > 1, "legend component must be uniquely multicolor"
    legend_idx = legend_candidates[0]

    legend_maps: dict[tuple[int, int], np.ndarray] = {}
    for map_shape in ((2, 3), (3, 2)):
        lm = _extract_legend_map(g, comps[legend_idx], map_shape)
        if lm is not None:
            legend_maps[map_shape] = lm
    assert legend_maps, "legend must expose a 2x3 or 3x2 palette"

    pieces = [_build_piece(g, pts, bg) for pts in comps]
    return g, bg, comps, legend_idx, pieces, legend_maps


def _template_assignments(
    pieces: list[dict], legend_idx: int, template: dict, max_solutions: int = 2
) -> list[dict[int, int]]:
    legend_shape = template["region_norms"][0]
    if legend_shape not in pieces[legend_idx]["shape_set"]:
        return []

    other_pis = [i for i in range(NUM_PIECES) if i != legend_idx]
    candidates: dict[int, list[int]] = {}
    for pi in other_pis:
        hits = [rid for rid in range(1, NUM_PIECES) if template["region_norms"][rid] in pieces[pi]["shape_set"]]
        if not hits:
            return []
        candidates[pi] = hits

    order = sorted(other_pis, key=lambda pi: (len(candidates[pi]), pi))
    solutions: list[dict[int, int]] = []
    used_regions = {0}
    current = {legend_idx: 0}

    def backtrack(idx: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if idx == len(order):
            solutions.append(current.copy())
            return
        pi = order[idx]
        for rid in candidates[pi]:
            if rid in used_regions:
                continue
            used_regions.add(rid)
            current[pi] = rid
            backtrack(idx + 1)
            current.pop(pi, None)
            used_regions.remove(rid)

    backtrack(0)
    return solutions


def _matching_templates(
    pieces: list[dict],
    legend_idx: int,
    legend_maps: dict[tuple[int, int], np.ndarray],
    max_matches: int = 3,
) -> list[tuple[dict, np.ndarray, dict[int, int]]]:
    out: list[tuple[dict, np.ndarray, dict[int, int]]] = []
    for template in TEMPLATES:
        legend_map = legend_maps.get(template["map_shape"])
        if legend_map is None:
            continue
        sols = _template_assignments(pieces, legend_idx, template, max_solutions=2)
        for sol in sols:
            out.append((template, legend_map, sol))
            if len(out) >= max_matches:
                return out
    return out


def _render_output(template: dict, legend_map: np.ndarray) -> np.ndarray:
    out = np.zeros(template["shape"], dtype=int)
    out[template["divider_mask"]] = DIVIDER_COLOR
    ph, pw = template["map_shape"]
    for rid in range(NUM_PIECES):
        rr, cc = divmod(rid, pw)
        assert 0 <= rr < ph
        col = int(legend_map[rr, cc])
        out[template["region_masks"][rid]] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble six scattered sprites into an irregular silhouette with staircase dividers, then recolor by a 2x3 or 3x2 legend.

    Input structure:
    - Background is the most common color.
    - Exactly six disconnected 4-connected sprites appear on the canvas.
    - Exactly one sprite is the legend: it is the only multicolor component and contains either an interior 2x3 or an interior 3x2 color patch.

    Rule:
    - Match the six sprite shapes, up to rotation/reflection, to one irregular template from a small family of 6-region silhouettes.
    - The template contains one-pixel staircase divider cells between adjacent regions; those divider cells are restored in the output using a fixed divider color.
    - Fill the six assembled regions by reading the legend patch in row-major order over its visible 2x3 or 3x2 layout.

    Key insight:
    The legend still acts as both a jigsaw piece and a color key, but the destination is no longer a plain rectangle. The solver must recover a non-rectangular silhouette, its staircase split topology, and the palette orientation before recoloring the six regions.
    """
    _, _, _, legend_idx, pieces, legend_maps = _extract_scene(grid)
    matches = _matching_templates(pieces, legend_idx, legend_maps, max_matches=2)
    assert len(matches) == 1, "piece set must match exactly one irregular 6-region template"
    template, legend_map, _ = matches[0]
    return _render_output(template, legend_map)


def _has_hole(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return True
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    empty = ~sub
    if not empty.any():
        return False
    vis = np.zeros_like(empty, dtype=bool)
    st: list[tuple[int, int]] = []
    h, w = empty.shape
    for rr in range(h):
        for cc in range(w):
            if rr in (0, h - 1) or cc in (0, w - 1):
                if empty[rr, cc] and not vis[rr, cc]:
                    vis[rr, cc] = True
                    st.append((rr, cc))
                    while st:
                        r, c = st.pop()
                        for r2, c2 in neighbors4(r, c):
                            if 0 <= r2 < h and 0 <= c2 < w and empty[r2, c2] and not vis[r2, c2]:
                                vis[r2, c2] = True
                                st.append((r2, c2))
    return bool((empty & ~vis).any())


def _scatter_pieces(
    masks: list[np.ndarray], body_colors: list[int], legend_map: np.ndarray, map_shape: tuple[int, int]
) -> np.ndarray | None:
    total = sum(int(mask.sum()) for mask in masks)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = int(body_colors[pi])
        if pi == 0:
            spots = _palette_spots(oriented, map_shape)
            if not spots:
                return None
            rr, cc = random.choice(spots)
            ph, pw = map_shape
            patch[rr : rr + ph, cc : cc + pw] = legend_map
        try:
            box = np.ones(oriented.shape, dtype=int)
            r0, c0 = random_free_location(
                occupied,
                box,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + oriented.shape[0], c0 : c0 + oriented.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _body_colors_for_palette(palette_colors: list[int]) -> list[int]:
    pool = [c for c in range(1, 10) if c not in palette_colors and c != DIVIDER_COLOR]
    assert len(pool) >= 2, "need at least two reusable body colors"
    colors = [pool[0], pool[1], pool[0], pool[1], pool[0], pool[1]]
    random.shuffle(colors)
    return colors


def _template_metrics(template: dict) -> dict[str, int | float | tuple[int, int]]:
    occupied = template["occupied_mask"]
    pts = np.argwhere(occupied)
    r0, c0, r1, c1 = bbox_coords(pts)
    bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    fill_area = int(occupied.sum())
    divider = template["divider_mask"]
    turns = 0
    for r, c in zip(*np.where(divider)):
        up = r > 0 and divider[r - 1, c]
        down = r + 1 < divider.shape[0] and divider[r + 1, c]
        left = c > 0 and divider[r, c - 1]
        right = c + 1 < divider.shape[1] and divider[r, c + 1]
        if (up or down) and (left or right):
            turns += 1
    return {
        "map_shape": template["map_shape"],
        "notch_cells": int(bbox_area - fill_area),
        "divider_cells": int(divider.sum()),
        "divider_turn_cells": int(turns),
    }


def _gen_attempt() -> np.ndarray | None:
    template = random.choice(TEMPLATES)
    metrics = _template_metrics(template)
    if metrics["notch_cells"] <= 0 or metrics["divider_turn_cells"] < 6:
        return None

    map_shape = template["map_shape"]
    ph, pw = map_shape
    legend_mask = _crop_mask(template["region_masks"][0])
    if not _palette_spots(legend_mask, map_shape):
        return None

    masks = template["region_masks"]
    if any(not _mask_connected(mask) or _has_hole(mask) for mask in masks):
        return None

    palette_pool = [c for c in range(1, 10) if c != DIVIDER_COLOR]
    palette_colors = random.sample(palette_pool, ph * pw)
    legend_map = np.array(palette_colors, dtype=int).reshape(ph, pw)
    body_colors = _body_colors_for_palette(palette_colors)

    expected = _render_output(template, legend_map)
    grid = _scatter_pieces(masks, body_colors, legend_map, map_shape)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    _, _, _, legend_idx, pieces, legend_maps = _extract_scene(grid)
    matches = _matching_templates(pieces, legend_idx, legend_maps, max_matches=3)
    if len(matches) != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo input")
