# Combo variant of 65b59efc (index 1)
# combo_spec: D1/0+D3/1
# Name: 形状比对对角标记

from __future__ import annotations

from itertools import combinations
from collections import Counter
import random

import numpy as np

from arc_utils import paste


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace
#    - Original task: in each middle tile column, every non-bg pixel is a single marker; its color directly
#      names one top template, and its local coordinate (lr, lc) names the output block.
#    - Fused task here: every logical marker is a same-color 2-pixel diagonal pair at
#      {(lr, lc), (lr+1, lc+1)}. A solver must first detect those pairs, then collapse each pair back to its
#      upper-left anchor (lr, lc). The full anchor set inside one tile column forms a cropped shape. That
#      anchor shape, not the color, identifies which top template to stamp by matching template silhouettes up
#      to rotation / flip. The anchor coordinates still choose output blocks.
#    - Example with N=5: suppose a middle tile column contains diagonal pairs anchored at
#      {(0, 1), (1, 1), (1, 2), (2, 1)}. After collapsing the pairs, the anchor mask is a T tetromino.
#      If the top band contains a T, an L, and a zigzag, that tile column must use the T template regardless
#      of marker color. The four anchors stamp that T into four output blocks.
#
# b) Complexity calibration
#    - Marker color is forced to be disjoint from all template colors, so the old color lookup is impossible.
#    - Shapes are chosen from unique dihedral-equivalence classes, so rotation / flip matching is necessary but
#      not ambiguous.
#    - Each tile column carries 2-5 diagonal pairs instead of a lone pixel, making the visual change obvious.
#
# c) Implicit-assumption audit against the original solver
#    - Original assumption "every non-bg middle-band pixel is a placement" is false now: each logical
#      placement is encoded by a 2-pixel diagonal pair, so main() must collapse pairs before stamping.
#    - Original assumption "marker color equals some template color" is false now: generator ensures the two
#      color sets are disjoint.
#    - Original assumption "template identity can be read per pixel" is false now: identity is only available
#      after aggregating all anchors in a tile column and matching their cropped shape to the top templates.
#    - Original assumption "all N x N local slots are potentially usable" is weakened: a diagonal pair needs a
#      2x2 footprint, so generated anchors live in rows/cols 0..N-2. The generator guarantees this.
#
# d) Placement legality
#    - Diagonal pair anchors are packed into the (N-1)x(N-1) anchor domain, so every pair fits inside the
#      middle band.
#    - Different templates are assigned disjoint anchor sets, so two stamps never compete for the same output
#      block.
#    - Middle-band anchor sets are filtered so every colored pixel belongs to exactly one detected NW-SE
#      diagonal pair, keeping the marker parse clean and visually separable.
#
# e) Visual observability
#    - Humans can visibly see that middle-band objects are diagonal dominoes, not single pixels.
#    - Within each tile column, the diagonal domino anchors trace the same silhouette as one top template.
#    - Because marker colors never match template colors, examples strongly suggest "shape match" rather than
#      "color match".
#
# f) Visual cleanliness
#    - Top templates remain monochrome and separated by dashed lines.
#    - Middle-band diagonal pairs stay inside their own tile columns and do not overlap.
#    - Palette swatches remain simple centered marks in the bottom band, keeping the display legible.
#
# g) Generator guarantees for main()
#    - Every template has a unique canonical silhouette under the 8 dihedral transforms.
#    - Every middle tile column contains exactly one marker color and at least one diagonal pair.
#    - Every colored middle-band pixel belongs to exactly one detected pair, so no stray noise or hidden extra
#      anchors remain after pair extraction.


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_CHOICES = [4, 5]
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_MIN_TOTAL_ANCHOR_COVERAGE = 0.55
_MAX_RETRIES = 500
_BRANCHES = [(4, 3), (5, 3), (5, 4)]

_OPS = ("id", "rot90", "rot180", "rot270", "hflip", "vflip", "diag", "anti")
_SHAPE_CATALOG: dict[int, list[dict]] = {}


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "mask must be non-empty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform_mask(mask: np.ndarray, op: str) -> np.ndarray:
    if op == "id":
        return mask.copy()
    if op == "rot90":
        return np.rot90(mask, 3)
    if op == "rot180":
        return np.rot90(mask, 2)
    if op == "rot270":
        return np.rot90(mask, 1)
    if op == "hflip":
        return np.fliplr(mask)
    if op == "vflip":
        return np.flipud(mask)
    if op == "diag":
        return mask.T.copy()
    if op == "anti":
        return np.fliplr(np.flipud(mask)).T.copy()
    raise ValueError(op)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_mask(mask.astype(bool))
    variants = []
    for op in _OPS:
        cur = _crop_mask(_transform_mask(cropped, op).astype(bool))
        cur_u8 = cur.astype(np.uint8)
        variants.append((tuple(cur_u8.shape), cur_u8.tobytes()))
    return min(variants)


def _mask_from_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in cells:
        out[r, c] = True
    return out


def _connected4(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _sample_anchor_shape(n: int, used_sigs: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    m = n - 1
    min_area = 2 if n == 4 else 3
    max_area = 4 if n == 4 else 5
    for _ in range(300):
        target = random.randint(min_area, max_area)
        cells = {(random.randrange(m), random.randrange(m))}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in list(cells):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    cand = (nr, nc)
                    if not (0 <= nr < m and 0 <= nc < m):
                        continue
                    if cand in cells:
                        continue
                    frontier.append(cand)
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target or not _connected4(cells):
            continue
        mask = _crop_mask(_mask_from_cells(cells, (m, m)))
        if n != 4 and (mask.shape[0] == 1 or mask.shape[1] == 1):
            continue
        sig = _mask_signature(mask)
        if sig in used_sigs:
            continue
        used_sigs.add(sig)
        return mask
    return None


def _embed_template(shape: np.ndarray, n: int) -> np.ndarray:
    h, w = shape.shape
    out = np.zeros((n, n), dtype=bool)
    r0 = random.randint(0, n - h)
    c0 = random.randint(0, n - w)
    out[r0 : r0 + h, c0 : c0 + w] = shape
    return out


def _clean_anchor_set(n: int, coords: set[tuple[int, int]]) -> bool:
    sub = np.full((n, n), _BG, dtype=int)
    for rr, cc in coords:
        sub[rr, cc] = 1
        sub[rr + 1, cc + 1] = 1

    anchors: list[tuple[int, int]] = []
    cover = np.zeros_like(sub, dtype=int)
    for rr in range(n - 1):
        for cc in range(n - 1):
            if int(sub[rr, cc]) == 1 and int(sub[rr + 1, cc + 1]) == 1:
                anchors.append((rr, cc))
                cover[rr, cc] += 1
                cover[rr + 1, cc + 1] += 1

    return set(anchors) == coords and np.array_equal(sub == 1, cover == 1)


def _shape_placements(n: int, shape: np.ndarray) -> list[dict]:
    m = n - 1
    placements: list[dict] = []
    seen_coords: set[frozenset[tuple[int, int]]] = set()
    for op in _OPS:
        cur = _crop_mask(_transform_mask(shape, op).astype(bool))
        h, w = cur.shape
        for r0 in range(m - h + 1):
            for c0 in range(m - w + 1):
                coords = frozenset((r0 + int(r), c0 + int(c)) for r, c in np.argwhere(cur))
                if coords in seen_coords or not _clean_anchor_set(n, set(coords)):
                    continue
                seen_coords.add(coords)
                placements.append({"coords": set(coords), "op": op})
    random.shuffle(placements)
    return placements


def _shape_catalog(n: int) -> list[dict]:
    if n in _SHAPE_CATALOG:
        return _SHAPE_CATALOG[n]

    m = n - 1
    min_area = 2 if n == 4 else 3
    max_area = 4 if n == 4 else 5
    entries: list[dict] = []
    seen_sigs: set[tuple[tuple[int, int], bytes]] = set()

    for bits in range(1, 1 << (m * m)):
        area = bits.bit_count()
        if not (min_area <= area <= max_area):
            continue
        cells = {(idx // m, idx % m) for idx in range(m * m) if (bits >> idx) & 1}
        if not _connected4(cells):
            continue
        mask = _crop_mask(_mask_from_cells(cells, (m, m)))
        if n != 4 and (mask.shape[0] == 1 or mask.shape[1] == 1):
            continue
        sig = _mask_signature(mask)
        if sig in seen_sigs:
            continue
        placements = _shape_placements(n, mask)
        if not placements:
            continue
        seen_sigs.add(sig)
        entries.append({"mask": mask, "placements": placements, "area": area})

    _SHAPE_CATALOG[n] = entries
    return entries


def _sample_shapes_and_placements(n: int, k: int) -> tuple[list[np.ndarray], list[dict]] | None:
    catalog = _shape_catalog(n)
    m = n - 1
    min_total = int(np.ceil(_MIN_TOTAL_ANCHOR_COVERAGE * m * m))
    combo_ids = list(combinations(range(len(catalog)), k))
    random.shuffle(combo_ids)

    for ids in combo_ids:
        area_total = sum(catalog[idx]["area"] for idx in ids)
        if area_total < min_total:
            continue

        order = sorted(ids, key=lambda idx: (-catalog[idx]["area"], len(catalog[idx]["placements"])))
        max_remaining = [0] * (len(order) + 1)
        for pos in range(len(order) - 1, -1, -1):
            max_remaining[pos] = max_remaining[pos + 1] + catalog[order[pos]]["area"]

        placed: dict[int, dict] = {}

        def backtrack(pos: int, used: set[tuple[int, int]]) -> bool:
            if pos == len(order):
                return len(used) >= min_total
            if len(used) + max_remaining[pos] < min_total:
                return False

            shape_idx = order[pos]
            candidates = list(catalog[shape_idx]["placements"])
            random.shuffle(candidates)
            for cand in candidates:
                coords = cand["coords"]
                if coords & used:
                    continue
                placed[shape_idx] = cand
                if backtrack(pos + 1, used | coords):
                    return True
                del placed[shape_idx]
            return False

        if backtrack(0, set()):
            base_shapes = [catalog[idx]["mask"].copy() for idx in ids]
            placements = [placed[idx] for idx in ids]
            return base_shapes, placements

    return None


def _build_expected(n: int, templates: list[np.ndarray], pcols: list[int],
                    placements: list[tuple[int, int, int]]) -> np.ndarray:
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], pcols[j], _BG)
        paste(expected, tile, rr * n, cc * n)
    return expected


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _extract_anchor_pairs(sub: np.ndarray, bg: int, sep: int) -> tuple[int, list[tuple[int, int]]]:
    vals = {int(v) for v in sub.flat} - {bg, sep}
    assert len(vals) == 1, "each marker tile column must use one marker color"
    color = vals.pop()
    anchors: list[tuple[int, int]] = []
    cover = np.zeros_like(sub, dtype=int)
    for rr in range(sub.shape[0] - 1):
        for cc in range(sub.shape[1] - 1):
            if int(sub[rr, cc]) == color and int(sub[rr + 1, cc + 1]) == color:
                anchors.append((rr, cc))
                cover[rr, cc] += 1
                cover[rr + 1, cc + 1] += 1
    assert anchors, "each marker tile column must contain at least one diagonal pair"
    colored = (sub == color)
    assert np.array_equal(colored, cover > 0), "colored middle-band pixels must be explainable by diagonal pairs"
    return color, anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend where marker identity comes from diagonal-pair silhouettes instead of colors.

    Input structure:
    - The grid is split into K tile columns of width N, separated by 1-column dashed separators.
    - Top band (rows 0..N-1): K monochrome template masks, one per tile column.
    - Middle band (rows N+1..2N): each tile column contains several same-color 2-pixel diagonal markers; the
      set of their upper-left anchor cells forms a small shape.
    - Bottom band (rows 2N+2..): K palette swatches that give the repaint color for each top template.
    - Dashed separator rows sit at rows N and 2N+1.

    Rule:
    - Detect every diagonal marker pair and collapse it to its upper-left anchor cell.
    - Crop the anchor set in each middle tile column and match that silhouette to one cropped top template,
      allowing rotation and reflection.
    - Each anchor coordinate (lr, lc) selects an output N x N block, and the matched template is stamped there
      after recoloring it with that template's palette color.

    Key insight:
    - Marker color is intentionally useless here. The real code is the aggregate shape made by diagonal-pair
      anchors: that shape identifies the template, while the individual anchor positions identify the stamp
      destinations.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if (g[:, c] == sep).sum() >= h // 2]
    sep_rows = [r for r in range(h) if (g[r, :] == sep).sum() >= w // 2]
    assert sep_cols and sep_rows

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)

    templates: list[np.ndarray] = []
    palette_cols: list[int] = []
    sig_to_j: dict[tuple[tuple[int, int], bytes], int] = {}

    for j, st in enumerate(starts):
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        mask = (sub == col)
        templates.append(mask)
        sig = _mask_signature(mask)
        assert sig not in sig_to_j, "template silhouettes must be unique up to symmetry"
        sig_to_j[sig] = j

    pal_r0 = sep_rows[-1] + 1
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        palette_cols.append(vals.pop())

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}
    mid_r0 = n + 1

    for st in starts:
        sub = g[mid_r0 : mid_r0 + n, st : st + n]
        _, anchors = _extract_anchor_pairs(sub, bg, sep)
        anchor_mask = np.zeros((n - 1, n - 1), dtype=bool)
        for rr, cc in anchors:
            anchor_mask[rr, cc] = True
        j = sig_to_j[_mask_signature(anchor_mask)]
        tile = np.where(templates[j], palette_cols[j], bg)
        for rr, cc in anchors:
            assert (rr, cc) not in slot_owner or slot_owner[(rr, cc)] == j, "different templates cannot share a slot"
            slot_owner[(rr, cc)] = j
            out[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n] = tile

    return out


def _gen_attempt(n: int, k: int) -> np.ndarray | None:
    tcols = random.sample(_AVAILABLE_COLORS, k)
    remaining = [c for c in _AVAILABLE_COLORS if c not in tcols]
    mcols = random.sample(remaining, k)
    pcols = [random.choice(_AVAILABLE_COLORS) for _ in range(k)]

    sampled = _sample_shapes_and_placements(n, k)
    if sampled is None:
        return None
    base_shapes, placements_meta = sampled

    templates = [_embed_template(base_shapes[j], n) for j in range(k)]
    placements: list[tuple[int, int, int]] = []
    for j, info in enumerate(placements_meta):
        for rr, cc in sorted(info["coords"]):
            placements.append((j, rr, cc))

    expected = _build_expected(n, templates, pcols, placements)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, info in enumerate(placements_meta):
        st = tile_assign[j]
        col = mcols[j]
        for rr, cc in info["coords"]:
            grid[n + 1 + rr, _raw_col(st, cc, n)] = col
            grid[n + 2 + rr, _raw_col(st, cc + 1, n)] = col

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < 0.35 else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    n, k = random.choice(_BRANCHES)
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(n, k)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused 65b59efc input")
