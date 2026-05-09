# Combo variant of 3a25b0d8 (index 28)
# combo_spec: D8/0+D9/2+D13/0
# Name: 上下对称+颜色按行+仅行秩序

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior
#    Original 3a25b0d8 uses two outline objects. The source object shows one colored subsprite per
#    cavity, the destination object shows the same cavity arrangement as empty holes, and the solver
#    aligns cavities by full 2D centroid rank (row_rank, col_rank).
#    This combo changes three coupled pieces at once:
#    - D8/0: the contour skeleton is no longer left-right symmetric; it is top-bottom symmetric.
#    - D9/2: colors are no longer per-cavity. Each source row exposes exactly one colored clue
#      cavity; that clue color applies to every cavity in the same row.
#    - D13/0: matching is no longer 2D. We only care about top-to-bottom row order; inside each row
#      the cavities are interpreted left-to-right, but the final recoloring is row-wise.
#    Small trace:
#      src row counts = [2, 1, 2], dst row counts = [2, 1, 2].
#      src top row has one visible orange clue and one empty sibling cavity, middle row has one
#      visible blue clue, bottom row has one visible green clue. dst has all holes empty.
#      Solver recovers the full src row structure from {colored clue cavities + empty source holes},
#      learns {row 0 -> orange, row 1 -> blue, row 2 -> green}, then paints every dst hole in those
#      rows with that row color and tight-crops.
#    The important difference from the original is that the solver never needs column ranks across
#    different rows, and never needs LR symmetry. The puzzle becomes a row-order puzzle.
# b) Complexity calibration
#    A degenerate version would use only singleton rows or keep src/dst in identical horizontal slot
#    positions, which would make the row-only rule hard to notice and let the original 2D matcher
#    succeed accidentally. The generator prevents that by:
#    - using curated palindromic row plans with 3-5 rows and 5-9 total cavities,
#    - requiring at least two multi-cavity rows in most plans,
#    - picking src and dst horizontal templates independently inside each row so full 2D rank keys
#      usually differ even though row counts match.
# c) Original assumptions that break
#    - Original `_rank_keys()` + transform search assumes exact 2D cavity arrangement matters.
#      That is false here: only row order matters, and source/dst may disagree on within-row spacing.
#    - Original final zip assumes every colored src component maps to one dst hole. That is false
#      here because the source intentionally shows only one colored clue cavity per row.
#    - Original symmetry intuition is LR-based. Here the visible contour symmetry is UD-based, so
#      keeping LR-centered row templates would weaken the D8 change. The generator therefore uses
#      asymmetric row layouts mirrored only across the horizontal axis.
# d) Placement legality
#    Placement still uses two separately cropped 8-connected objects on a 22-30 canvas, with
#    `random_free_location(..., padding=2)` for non-overlap and border safety.
# e) Visual observability
#    Humans can see three stable cues within 2-4 examples:
#    - the outline is mirrored top-to-bottom,
#    - one clue color appears per source row,
#    - dst output repeats that clue across the whole row.
#    No invisible ordering such as numeric color ids is used.
# f) Visual cleanliness
#    There are no extra markers beyond the original outline/holes vocabulary. The clue is just one
#    filled cavity per row, so the image stays readable.
# g) Color inferability
#    Distinct colors equal the number of rows (3-5). Each row has one clue and at least one row in
#    most samples has multiple cavities, so the "color belongs to the row, not the cavity" pattern
#    is repeated visibly.
# h) New solver assumptions
#    - src and dst are used in the same upright orientation; the solver does not search flips or
#      rotations. This is deliberate: with UD-symmetric outlines and distinct row colors, allowing a
#      vertical flip would make top-vs-bottom ambiguous to humans.
#    - Components belonging to one row occupy the same vertical band, so row grouping can be done by
#      overlapping bbox intervals rather than exact centroid equality.
#    - Each source row contains exactly one colored clue cavity, and every destination row has a
#      matching source row. The generator enforces this directly.
# i) Generator-side anti-degeneration checks
#    The generator rejects layouts unless:
#    - both objects are UD-symmetric,
#    - src has exactly one colored component per row,
#    - row counts match between src and dst,
#    - src/dst full 2D rank-key multisets differ on most accepted samples, proving D13/0 is active.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill destination holes row-by-row using source row clue colors, then tight-crop.

    Input:
    - Two 8-connected outline objects share one outline color.
    - Source has one colored clue cavity per row plus other empty holes.
    - Destination has only empty holes.

    Combo rule:
    - Ignore full 2D cavity matching.
    - Recover the source's complete row structure from colored cavities + empty holes.
    - Match rows only by top-to-bottom order.
    - Fill every destination hole in a row with that row's clue color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _components(mask: np.ndarray, connectivity: int) -> list[dict]:
        lab, n = label_mask(mask, connectivity=connectivity)
        comps = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "pts": pts,
                    "size": int(len(pts)),
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        comps.sort(key=lambda d: int(d["size"]), reverse=True)
        return comps

    def _extract_two_objects() -> list[list[dict]]:
        comps = _components(g != int(bg), connectivity=8)
        if len(comps) < 2:
            return [comps]
        return [[comps[0]], [comps[1]]]

    def _bbox_of(group: list[dict]) -> tuple[int, int, int, int]:
        r0 = min(int(d["bbox"][0]) for d in group)
        c0 = min(int(d["bbox"][1]) for d in group)
        r1 = max(int(d["bbox"][2]) for d in group)
        c1 = max(int(d["bbox"][3]) for d in group)
        return int(r0), int(c0), int(r1), int(c1)

    def _extract_group_subgrid(group: list[dict]) -> np.ndarray:
        r0, c0, r1, c1 = _bbox_of(group)
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for d in group:
            for rr, cc in d["pts"]:
                sub[int(rr) - int(r0), int(cc) - int(c0)] = int(g[int(rr), int(cc)])
        return sub

    def _tight_crop_non_bg(sub: np.ndarray) -> np.ndarray:
        pts = np.argwhere(sub != int(bg))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    def _enclosed_bg_components(sub: np.ndarray) -> list[dict]:
        mask = sub == int(bg)
        lab, n = label_mask(mask, connectivity=4)
        h, w = sub.shape
        holes = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            holes.append(
                {
                    "pts": pts,
                    "size": int(len(pts)),
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        return holes

    def _outline_col(sub: np.ndarray) -> int:
        cnt = Counter(map(int, sub.flatten()))
        opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
        return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)

    def _colored_components(sub: np.ndarray) -> list[dict]:
        outc = _outline_col(sub)
        mask = (sub != int(bg)) & (sub != int(outc))
        lab, n = label_mask(mask, connectivity=4)
        comps = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
            comps.append(
                {
                    "pts": pts,
                    "size": int(len(pts)),
                    "color": int(col),
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        return comps

    def _group_rows(items: list[dict]) -> list[list[dict]]:
        if not items:
            return []
        ordered = sorted(items, key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1])))
        rows: list[list[dict]] = []
        current: list[dict] = []
        cur_r1 = -1
        for item in ordered:
            r0, _c0, r1, _c1 = item["bbox"]
            if not current or int(r0) <= int(cur_r1):
                current.append(item)
                cur_r1 = max(int(cur_r1), int(r1))
            else:
                current.sort(key=lambda d: (float(d["cent"][1]), float(d["cent"][0])))
                rows.append(current)
                current = [item]
                cur_r1 = int(r1)
        if current:
            current.sort(key=lambda d: (float(d["cent"][1]), float(d["cent"][0])))
            rows.append(current)
        rows.sort(key=lambda row: min(int(d["bbox"][0]) for d in row))
        return rows

    groups = _extract_two_objects()
    assert len(groups) >= 2
    objs = [_extract_group_subgrid(grp) for grp in groups]

    holes_list = [_enclosed_bg_components(obj) for obj in objs]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest)
    src_empty_holes = _enclosed_bg_components(src)
    src_subs = _colored_components(src)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest)

    dest_rows = _group_rows(dest_holes)
    src_rows = _group_rows(src_empty_holes + src_subs)
    if not dest_rows or not src_rows:
        return _tight_crop_non_bg(dest)

    rowwise_ok = len(src_rows) == len(dest_rows) and all(
        len(src_row) == len(dest_row) for src_row, dest_row in zip(src_rows, dest_rows)
    )
    row_color_map: dict[int, int] = {}
    if rowwise_ok:
        for row_idx, src_row in enumerate(src_rows):
            colors = [int(item["color"]) for item in src_row if "color" in item]
            if len(colors) != 1:
                rowwise_ok = False
                break
            row_color_map[int(row_idx)] = int(colors[0])

    if rowwise_ok:
        for row_idx, holes in enumerate(dest_rows):
            fill_color = int(row_color_map[int(row_idx)])
            for hole in holes:
                for rr, cc in hole["pts"]:
                    dest[int(rr), int(cc)] = fill_color
        return _tight_crop_non_bg(dest)

    # Fallback for off-spec inputs: sort colored clues and destination holes by row, then by x.
    src_subs_sorted = sorted(src_subs, key=lambda d: (int(d["bbox"][0]), float(d["cent"][1])))
    dest_holes_sorted = sorted(dest_holes, key=lambda d: (int(d["bbox"][0]), float(d["cent"][1])))
    for src_item, hole in zip(src_subs_sorted, dest_holes_sorted):
        fill_color = int(src_item["color"])
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = fill_color
    return _tight_crop_non_bg(dest)


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (5, 9)
FILL_COLOR_COUNT_RANGE = (3, 5)

ROW_PLAN_CHOICES = [
    [2, 1, 2],
    [2, 2, 2],
    [3, 1, 3],
    [2, 3, 2],
    [2, 1, 1, 2],
    [2, 2, 1, 2, 2],
    [2, 1, 2, 1, 2],
    [3, 1, 2, 1, 3],
]
PAIR_ROW_HEIGHT_CHOICES = [1, 1, 2, 2]
CENTER_ROW_HEIGHT = 1
ROW_GAP_CHOICES = [1, 1, 2]

ROW_X_TEMPLATES = {
    1: [(0,), (1,), (2,)],
    2: [(0, 4), (1, 5), (0, 5), (2, 6)],
    3: [(0, 3, 6), (1, 4, 7), (0, 4, 7)],
}
ROW_WIDTH_CHOICES = {
    1: [2, 3, 4],
    2: [2, 3],
    3: [1, 2],
}

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 400


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _shape_uses_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask.any()) and bool(mask.any(axis=0).all()) and bool(mask.any(axis=1).all())


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape
    n_remove = random.randint(1, max(1, n_remove_max))
    tried = 0
    removed = 0
    while removed < n_remove and tried < MAX_MUTATE_ATTEMPTS:
        tried += 1
        cand = []
        for r in range(h):
            for c in range(w):
                if not m[r, c]:
                    continue
                nb = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and m[nr, nc]:
                        nb += 1
                if nb <= 2:
                    cand.append((r, c))
        if not cand:
            break
        r, c = random.choice(cand)
        m2 = m.copy()
        m2[r, c] = False
        if not m2.any():
            continue
        if not _shape_uses_full_bbox(m2):
            continue
        _, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += 1
    return m


def _gen_shape(h: int, w: int, *, force_nonrect: bool | None = None) -> np.ndarray:
    nonrect = (random.random() < NONRECT_PROBABILITY) if force_nonrect is None else bool(force_nonrect)
    base = _random_rect_shape(h, w)
    if not nonrect or h * w < 3:
        return base
    best = base
    limit = max(1, (h * w) // 2)
    for _ in range(6):
        cand = _mutate_shape(base, n_remove_max=limit)
        if _shape_uses_full_bbox(cand):
            best = cand
            if not np.array_equal(cand, base):
                return cand
    return best


def _flipud_shape(mask: np.ndarray) -> np.ndarray:
    return mask[::-1, :].copy()


def _build_row_tops(row_heights: list[int], gaps: list[int]) -> list[int]:
    tops = [0]
    for idx in range(1, len(row_heights)):
        tops.append(int(tops[-1] + row_heights[idx - 1] + gaps[idx - 1]))
    return tops


def _sample_row_widths(k: int) -> list[int]:
    return [int(random.choice(ROW_WIDTH_CHOICES[k])) for _ in range(k)]


def _sample_row_offsets(k: int) -> list[int]:
    return list(map(int, random.choice(ROW_X_TEMPLATES[k])))


def _make_row_shapes(k: int, row_h: int, *, for_center: bool) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    widths_dst = _sample_row_widths(k)
    widths_src = _sample_row_widths(k)
    shapes_dst = []
    shapes_src = []
    for w_dst, w_src in zip(widths_dst, widths_src):
        force_nonrect = None if not for_center else False
        shapes_dst.append(_gen_shape(row_h, int(w_dst), force_nonrect=force_nonrect))
        shapes_src.append(_gen_shape(row_h, int(w_src), force_nonrect=force_nonrect))
    return shapes_dst, shapes_src, widths_dst, widths_src


def _design_layout() -> list[dict] | None:
    for _ in range(300):
        plan = list(random.choice(ROW_PLAN_CHOICES))
        n_rows = len(plan)
        if not (FILL_COLOR_COUNT_RANGE[0] <= n_rows <= FILL_COLOR_COUNT_RANGE[1]):
            continue
        if not (CAVITY_COUNT_RANGE[0] <= sum(plan) <= CAVITY_COUNT_RANGE[1]):
            continue
        if sum(1 for k in plan if k >= 2) < 2:
            continue

        row_heights = [0] * n_rows
        for idx in range(n_rows // 2):
            h = int(random.choice(PAIR_ROW_HEIGHT_CHOICES))
            row_heights[idx] = h
            row_heights[n_rows - 1 - idx] = h
        if n_rows % 2 == 1:
            row_heights[n_rows // 2] = int(CENTER_ROW_HEIGHT)

        gaps = [0] * max(0, n_rows - 1)
        for idx in range(len(gaps) // 2):
            g = int(random.choice(ROW_GAP_CHOICES))
            gaps[idx] = g
            gaps[len(gaps) - 1 - idx] = g
        if len(gaps) % 2 == 1:
            gaps[len(gaps) // 2] = int(random.choice(ROW_GAP_CHOICES))

        row_tops = _build_row_tops(row_heights, gaps)
        fill_height = int(row_tops[-1] + row_heights[-1])
        if fill_height + 2 > OBJECT_H_RANGE[1]:
            continue

        slots: list[dict] = []
        for row_id, k in enumerate(plan):
            if row_id > n_rows - 1 - row_id:
                continue
            mirror_row = n_rows - 1 - row_id
            row_h = int(row_heights[row_id])
            offsets_dst = _sample_row_offsets(k)
            offsets_src = _sample_row_offsets(k)
            center_row = row_id == mirror_row
            shapes_dst, shapes_src, _widths_dst, _widths_src = _make_row_shapes(
                k, row_h, for_center=center_row
            )
            top_r0 = int(row_tops[row_id])
            bottom_r0 = int(row_tops[mirror_row])
            for pos_idx in range(k):
                slot = {
                    "row_id": int(row_id),
                    "r0": top_r0,
                    "col_offset_dst": int(offsets_dst[pos_idx]),
                    "col_offset_src": int(offsets_src[pos_idx]),
                    "shape_dst": shapes_dst[pos_idx],
                    "shape_src": shapes_src[pos_idx],
                }
                slots.append(slot)
                if not center_row:
                    slots.append(
                        {
                            "row_id": int(mirror_row),
                            "r0": bottom_r0,
                            "col_offset_dst": int(offsets_dst[pos_idx]),
                            "col_offset_src": int(offsets_src[pos_idx]),
                            "shape_dst": _flipud_shape(shapes_dst[pos_idx]),
                            "shape_src": _flipud_shape(shapes_src[pos_idx]),
                        }
                    )
        slots.sort(key=lambda d: (int(d["row_id"]), int(d["col_offset_dst"]), int(d["col_offset_src"])))
        return slots
    return None


def _render_object(
    layout: list[dict],
    *,
    shape_key: str,
    offset_key: str,
    colors: list[int],
    visible_mask: list[bool],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed = []
    for idx, slot in enumerate(layout):
        sh = slot[shape_key]
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed.append((idx, slot["r0"] + rr, int(slot[offset_key]) + cc))
    if not placed:
        return None, None

    all_rows = np.concatenate([rows for _idx, rows, _cols in placed])
    all_cols = np.concatenate([cols for _idx, _rows, cols in placed])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    h = int(r_max - r_min + 1 + 2 * pad)
    w = int(c_max - c_min + 1 + 2 * pad)
    canvas = np.full((h, w), int(bg), dtype=int)
    full_canvas = np.full((h, w), int(bg), dtype=int)
    structure_mask = np.zeros((h, w), dtype=bool)

    for idx, rows, cols in placed:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        structure_mask[rr, cc] = True
        full_canvas[rr, cc] = int(colors[idx])
        if visible_mask[idx]:
            canvas[rr, cc] = int(colors[idx])

    outline_mask = np.zeros_like(structure_mask)
    outline_mask[1:, :] |= structure_mask[:-1, :]
    outline_mask[:-1, :] |= structure_mask[1:, :]
    outline_mask[:, 1:] |= structure_mask[:, :-1]
    outline_mask[:, :-1] |= structure_mask[:, 1:]
    outline_mask &= ~structure_mask
    canvas[outline_mask] = int(outline_col)
    full_canvas[outline_mask] = int(outline_col)

    b = bbox(canvas != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0, c0, r1, c1 = b
    filled_cropped = full_canvas[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return cropped, filled_cropped


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != int(bg)) & (filled_grid != int(outline_col))
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1:
            count += 1
    return count


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


def _outline_col_local(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _hole_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = sub.shape
    out = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        out.append(
            {
                "pts": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
            }
        )
    return out


def _colored_components_local(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col_local(sub, bg)
    mask = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(mask, connectivity=4)
    out = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        out.append(
            {
                "pts": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "color": int(col),
            }
        )
    return out


def _group_rows_local(items: list[dict]) -> list[list[dict]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1])))
    rows: list[list[dict]] = []
    current: list[dict] = []
    cur_r1 = -1
    for item in ordered:
        r0, _c0, r1, _c1 = item["bbox"]
        if not current or int(r0) <= int(cur_r1):
            current.append(item)
            cur_r1 = max(int(cur_r1), int(r1))
        else:
            current.sort(key=lambda d: (float(d["cent"][1]), float(d["cent"][0])))
            rows.append(current)
            current = [item]
            cur_r1 = int(r1)
    if current:
        current.sort(key=lambda d: (float(d["cent"][1]), float(d["cent"][0])))
        rows.append(current)
    rows.sort(key=lambda row: min(int(d["bbox"][0]) for d in row))
    return rows


def _is_ud_symmetric_object(sub: np.ndarray, bg: int) -> bool:
    mask = sub != int(bg)
    return bool(np.array_equal(mask, mask[::-1, :]))


def _rank_keys_from_rows(rows: list[list[dict]]) -> list[tuple[int, int]]:
    keys = []
    for row_idx, row in enumerate(rows):
        for col_idx, _item in enumerate(sorted(row, key=lambda d: (float(d["cent"][1]), float(d["cent"][0])))):
            keys.append((int(row_idx), int(col_idx)))
    return keys


def _variant_signal_active(src: np.ndarray, dst: np.ndarray, bg: int) -> bool:
    src_holes = _hole_components(src, bg)
    src_cols = _colored_components_local(src, bg)
    dst_holes = _hole_components(dst, bg)
    src_rows = _group_rows_local(src_holes + src_cols)
    dst_rows = _group_rows_local(dst_holes)
    if len(src_rows) != len(dst_rows):
        return False
    if not all(len(srow) == len(drow) for srow, drow in zip(src_rows, dst_rows)):
        return False
    if sum(len(row) >= 2 for row in dst_rows) < 2:
        return False
    if any(sum(1 for item in row if "color" in item) != 1 for row in src_rows):
        return False
    if not (_is_ud_symmetric_object(src, bg) and _is_ud_symmetric_object(dst, bg)):
        return False
    src_color_keys = _rank_keys_from_rows(_group_rows_local(src_cols))
    dst_keys = _rank_keys_from_rows(dst_rows)
    if len(src_color_keys) >= len(dst_keys):
        return False
    src_all_keys = _rank_keys_from_rows(src_rows)
    dst_all_keys = _rank_keys_from_rows(dst_rows)
    same_2d_shape = src_all_keys == dst_all_keys
    same_centroid_cols = True
    for srow, drow in zip(src_rows, dst_rows):
        s_cols = [int(round(item["cent"][1])) for item in srow]
        d_cols = [int(round(item["cent"][1])) for item in drow]
        if s_cols != d_cols:
            same_centroid_cols = False
            break
    return (not same_2d_shape) or (not same_centroid_cols)


def _make_src_visible_mask(layout: list[dict]) -> list[bool]:
    visible = [False] * len(layout)
    by_row: dict[int, list[int]] = {}
    for idx, slot in enumerate(layout):
        by_row.setdefault(int(slot["row_id"]), []).append(idx)
    for indices in by_row.values():
        visible[random.choice(indices)] = True
    return visible


def _assign_row_colors(layout: list[dict], fill_pool: list[int]) -> list[int]:
    row_ids = sorted({int(slot["row_id"]) for slot in layout})
    chosen = random.sample(fill_pool, len(row_ids))
    random.shuffle(chosen)
    row_color_map = {int(row_id): int(col) for row_id, col in zip(row_ids, chosen)}
    return [int(row_color_map[int(slot["row_id"])]) for slot in layout]


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _build_object_pair(
    layout: list[dict], colors: list[int], bg: int, outline_col: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    n_rows = len({int(slot["row_id"]) for slot in layout})
    src_visible_mask = _make_src_visible_mask(layout)
    dst_visible_mask = [False] * n_cavities

    dst_cropped, dst_filled = _render_object(
        layout,
        shape_key="shape_dst",
        offset_key="col_offset_dst",
        colors=colors,
        visible_mask=dst_visible_mask,
        bg=bg,
        outline_col=outline_col,
    )
    if dst_cropped is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(
        layout,
        shape_key="shape_src",
        offset_key="col_offset_src",
        colors=colors,
        visible_mask=src_visible_mask,
        bg=bg,
        outline_col=outline_col,
    )
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_rows:
        return None
    if not _size_in_spec(src_cropped):
        return None
    if not _variant_signal_active(src_cropped, dst_cropped, bg):
        return None

    return src_cropped, dst_cropped, dst_filled


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    try:
        r, c = random_free_location(
            canvas, src, transparent=bg, border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING
        )
    except ValueError:
        return None
    paste(canvas, src, r, c, transparent=bg)
    try:
        r, c = random_free_location(
            canvas, dst, transparent=bg, border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING
        )
    except ValueError:
        return None
    paste(canvas, dst, r, c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])

    layout = _design_layout()
    if layout is None:
        return None
    n_rows = len({int(slot["row_id"]) for slot in layout})
    if len(palette) - 1 < n_rows:
        return None
    fill_pool = palette[1 : 1 + n_rows]
    colors = _assign_row_colors(layout, fill_pool)

    built = _build_object_pair(layout, colors, bg, outline_col)
    if built is None:
        return None
    src, dst, expected = built

    canvas = _place_on_canvas(src, dst, bg)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
