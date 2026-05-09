# Combo variant of 5dbc8537 (index 10)
# combo_spec: D1/2+D2/2
# Name: 对角线标记精灵连接器匹配

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, get_bg, paste, random_free_location

# Design analysis for the fused D1/2 + D2/2 variant
# a) Concrete behavior and trace:
# - The template stays a full-width or full-height base-colored strip, and the answer is still its tight crop.
# - Each hole is an isolated rectangle cut out of that strip. The hole is mostly background-colored, except for a
#   single connector pixel placed at one corner.
# - Outside the template, a source is no longer a solid rectangle. It is encoded by exactly two diagonal corner
#   pixels inside a reserved bbox: one corner is the same connector color as the target hole marker, and the
#   opposite corner is the fill color of the recovered rectangle.
# - Example: two different 2x4 pieces may both have the same red body color. One has an orange connector at the
#   top-left corner, the other has a teal connector at the top-left corner. In the input, each source shows only
#   two diagonal dots; in the hole region, each matching 2x4 cavity shows only the corresponding orange/teal dot.
#   The solver must infer the hidden rectangle from the diagonal pair, then use the connector color and corner to
#   choose the correct same-sized hole.
#
# b) Complexity calibration:
# - At least one non-2x2 shape group is duplicated, and duplicated members share the same body color. Without the
#   connector mark, same-sized sources would remain ambiguous.
# - Connector colors come from a small reusable palette, and some connector colors are reused across different
#   shapes. That prevents the task from degenerating into "global unique color -> hole".
# - Recovered pieces are solid rectangles plus one preserved connector pixel. This keeps the diagonal encoding
#   compatible with human inference; two sparse source dots would not be enough to reconstruct a hidden texture.
#
# c) Hidden-assumption audit against the original main():
# - `base_col` as the largest non-bg component still holds because the strip dominates every sparse source marker.
# - `dst = bbox(base pixels)` still identifies the template region.
# - The original source extraction by connected components fails here: each source consists of two disconnected
#   pixels. The solver must instead recover a source bbox from a connector pixel plus the expected diagonal offset.
# - The original arbitrary same-size packing DFS is no longer appropriate once duplicate sizes exist. The generator
#   therefore makes holes disconnected rectangles, and the solver reads each hole directly as a keyed target.
#
# d) Placement legality:
# - Holes are placed with a base-colored border and at least a one-cell base gap between bboxes, so distinct holes
#   never merge under 8-connectivity.
# - Outside markers are scattered by reserving each full hidden bbox before drawing only its two corners. This
#   guarantees no overlap, no accidental third pixel inside a bbox, and at least one background cell of separation.
#
# e) Visual observability:
# - In a few examples, a human can see two simultaneous changes: source rectangles are replaced by sparse diagonal
#   markers, and repeated same-size rectangles are disambiguated by connector-colored corner dots mirrored inside
#   the holes.
# - No invisible ordering by color id is used. The only cues are visible geometry, visible corner position, and
#   visible connector-color correspondence.
#
# f) Visual cleanliness:
# - Each source bbox contains only two corner pixels, and each hole contains only one corner marker. There are no
#   stacked annotations or overlapping objects.
# - Solid output rectangles keep the completed template easy to read.
#
# g) Color inferability:
# - Connector colors are copied directly from source to hole. Body colors are copied directly from the opposite
#   source corner into the filled rectangle. There is no hidden color remapping.
#
# h) Post-change solver assumptions guaranteed by the generator:
# - Every hole component is a full rectangle containing exactly one non-{bg, base} connector pixel at a corner.
# - Every source bbox contains exactly two non-bg pixels: the connector at the matching corner and the body color
#   at the opposite corner.
# - Body colors and connector colors come from disjoint palettes.
# - The tuple `(h, w, connector_color, connector_corner)` is unique across holes, while connector colors may still
#   repeat across different shapes.


def _corner_set(h: int, w: int) -> set[tuple[int, int]]:
    return {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def _opposite_corner(h: int, w: int, pos: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, pos)
    return int(h - 1 - r), int(w - 1 - c)


def _extract_single_connector(patch: np.ndarray, *, bg: int, base: int) -> tuple[int, tuple[int, int]]:
    coords = np.argwhere((patch != int(bg)) & (patch != int(base)))
    assert int(coords.shape[0]) == 1
    rr, cc = map(int, coords[0])
    h, w = map(int, patch.shape)
    assert (rr, cc) in _corner_set(h, w)
    return int(patch[rr, cc]), (int(rr), int(cc))


def _recover_source_patch(
    masked: np.ndarray,
    *,
    h: int,
    w: int,
    conn_color: int,
    conn_pos: tuple[int, int],
    bg: int,
    base: int,
    marker_colors: set[int],
    used_connectors: set[tuple[int, int]],
) -> tuple[tuple[int, int], np.ndarray]:
    opp = _opposite_corner(h, w, conn_pos)
    candidates: list[tuple[tuple[int, int], np.ndarray]] = []
    for rr, cc in np.argwhere(masked == int(conn_color)):
        rr = int(rr)
        cc = int(cc)
        if (rr, cc) in used_connectors:
            continue
        sr0 = rr - int(conn_pos[0])
        sc0 = cc - int(conn_pos[1])
        sr1 = sr0 + h - 1
        sc1 = sc0 + w - 1
        if sr0 < 0 or sc0 < 0 or sr1 >= masked.shape[0] or sc1 >= masked.shape[1]:
            continue
        sub = masked[int(sr0) : int(sr1) + 1, int(sc0) : int(sc1) + 1]
        nz = np.argwhere(sub != int(bg))
        if int(nz.shape[0]) != 2:
            continue
        nz_set = {tuple(map(int, p)) for p in nz}
        if nz_set != {tuple(map(int, conn_pos)), tuple(map(int, opp))}:
            continue
        if int(sub[int(conn_pos[0]), int(conn_pos[1])]) != int(conn_color):
            continue
        body_color = int(sub[int(opp[0]), int(opp[1])])
        if body_color in (int(bg), int(base)) or body_color in marker_colors:
            continue
        patch = np.full((h, w), int(body_color), dtype=int)
        patch[int(conn_pos[0]), int(conn_pos[1])] = int(conn_color)
        candidates.append(((rr, cc), patch))

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct diagonal-corner sprites and place them into connector-marked cavities inside the template strip.

    Input structure:
    - The background is the most common color, and one dominant base color forms a large strip-shaped template.
    - Inside that strip, each disconnected rectangular hole is mostly background but contains one colored connector
      pixel at a corner.
    - Outside the template, each source sprite is sparse: only two diagonal corners are shown, one in the connector
      color and the opposite one in the rectangle's body color.

    Rule:
    - Detect the template bbox from the largest non-background connected component.
    - Read each hole as a full rectangle, recording its size, its connector color, and which corner contains it.
    - For each such signature, find the matching outside connector pixel, use the diagonal opposite corner to infer
      the hidden rectangle and its fill color, then paste the reconstructed rectangle back into the hole.
    - Output the tight crop of the filled template.

    Key insight: The outside sources are implicit diagonal markers, not connected sprites; the connector-colored
    corner identifies which hole to use, while the diagonal opposite corner reveals the rectangle's fill color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))
    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)

    hole_comps = components_mask(dst != int(base_col), connectivity=8)
    assert hole_comps

    holes: dict[tuple[int, int, int, tuple[int, int]], tuple[int, int]] = {}
    marker_colors: set[int] = set()
    for coords in hole_comps:
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        h = int(hr1 - hr0 + 1)
        w = int(hc1 - hc0 + 1)
        assert int(coords.shape[0]) == h * w
        patch = dst[int(hr0) : int(hr1) + 1, int(hc0) : int(hc1) + 1]
        conn_color, conn_pos = _extract_single_connector(patch, bg=bg, base=base_col)
        key = (int(h), int(w), int(conn_color), (int(conn_pos[0]), int(conn_pos[1])))
        assert key not in holes
        holes[key] = (int(hr0), int(hc0))
        marker_colors.add(int(conn_color))

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    used_connectors: set[tuple[int, int]] = set()
    out = np.full((dh, dw), int(base_col), dtype=int)
    for key, (rr0, cc0) in holes.items():
        h, w, conn_color, conn_pos = key
        connector_coord, patch = _recover_source_patch(
            masked,
            h=int(h),
            w=int(w),
            conn_color=int(conn_color),
            conn_pos=(int(conn_pos[0]), int(conn_pos[1])),
            bg=int(bg),
            base=int(base_col),
            marker_colors=marker_colors,
            used_connectors=used_connectors,
        )
        used_connectors.add(connector_coord)
        paste(out, patch, int(rr0), int(cc0), overwrite=True)

    return out


# --------------- generator (fused diagonal + connector variant) ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 5, 7
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.20, 0.34
MAX_LONG_DIM = 5
MAX_SHORT_DIM = 4
MIN_GROUPS, MAX_GROUPS = 3, 4
MAX_GROUP_COUNT = 3
MIN_LONG_COVERAGE = 0.58
MAX_RETRIES = 500


def _sample_group_counts(total: int) -> list[int]:
    num_groups = random.randint(MIN_GROUPS, min(MAX_GROUPS, total - 1))
    counts = [2] + [1] * (num_groups - 1)
    remaining = total - sum(counts)
    while remaining > 0:
        idxs = [i for i, v in enumerate(counts) if v < MAX_GROUP_COUNT]
        assert idxs
        counts[random.choice(idxs)] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _shape_candidates(dh: int, dw: int) -> list[tuple[int, int]]:
    long_is_rows = dh >= dw
    long_dim = dh if long_is_rows else dw
    short_dim = dw if long_is_rows else dh
    max_len = min(MAX_LONG_DIM, long_dim - 2)
    max_cross = min(MAX_SHORT_DIM, short_dim - 2)
    out: list[tuple[int, int]] = []
    for ln in range(2, max_len + 1):
        for cr in range(2, max_cross + 1):
            h, w = (ln, cr) if long_is_rows else (cr, ln)
            if (h, w) == (2, 2):
                continue
            if h * w <= max(4, (dh * dw) // 5):
                out.append((int(h), int(w)))
    return out


def _sample_shapes(dh: int, dw: int, counts: list[int]) -> list[tuple[int, int]] | None:
    shapes = _shape_candidates(dh, dw)
    if len(shapes) < len(counts):
        return None
    for _ in range(100):
        pool = shapes[:]
        random.shuffle(pool)
        chosen = pool[: len(counts)]
        long_is_rows = dh >= dw
        total_long = sum(int((h if long_is_rows else w) * cnt) for (h, w), cnt in zip(chosen, counts))
        area = sum(int(h * w) * int(cnt) for (h, w), cnt in zip(chosen, counts))
        lo = math.ceil(HOLE_FRAC_LO * dh * dw)
        hi = math.floor(HOLE_FRAC_HI * dh * dw)
        if total_long + len(counts) - 1 <= (dh if long_is_rows else dw) - 2 and lo <= area <= hi:
            return [(int(h), int(w)) for h, w in chosen]
    return None


def _place_rects(dh: int, dw: int, items: list[dict]) -> list[dict] | None:
    long_is_rows = dh >= dw
    long_dim = dh if long_is_rows else dw
    short_dim = dw if long_is_rows else dh

    work = items[:]
    random.shuffle(work)

    lengths = [int(item["h"] if long_is_rows else item["w"]) for item in work]
    needed = sum(lengths) + (len(work) - 1)
    free = (long_dim - 2) - needed
    if free < 0:
        return None

    gaps = [0] * (len(work) + 1)
    for _ in range(free):
        gaps[random.randrange(len(gaps))] += 1

    cursor = 1 + gaps[0]
    placed: list[dict] = []
    for i, item in enumerate(work):
        h, w = int(item["h"]), int(item["w"])
        cross = w if long_is_rows else h
        cross_max = short_dim - cross - 1
        if cross_max < 1:
            return None
        cross0 = random.randint(1, cross_max)
        new_item = dict(item)
        if long_is_rows:
            new_item["r"] = int(cursor)
            new_item["c"] = int(cross0)
            cursor += h
        else:
            new_item["r"] = int(cross0)
            new_item["c"] = int(cursor)
            cursor += w
        placed.append(new_item)
        cursor += 1 + gaps[i + 1]

    rows = [it["r"] for it in placed] + [it["r"] + it["h"] - 1 for it in placed]
    cols = [it["c"] for it in placed] + [it["c"] + it["w"] - 1 for it in placed]
    row_cov = (max(rows) - min(rows) + 1) / dh
    col_cov = (max(cols) - min(cols) + 1) / dw
    if max(row_cov, col_cov) < MIN_LONG_COVERAGE:
        return None
    return placed


def _choose_connector_pos(h: int, w: int) -> tuple[int, int]:
    return tuple(map(int, random.choice(sorted(_corner_set(h, w)))))


def _build_items(
    items: list[dict], *, bg: int, base: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    available = [c for c in range(10) if c not in (bg, base)]
    group_counts: dict[int, int] = {}
    for item in items:
        group_counts[item["group"]] = group_counts.get(item["group"], 0) + 1

    connector_pool_size = max(3, max(group_counts.values()))
    if len(available) < connector_pool_size + 2:
        return None
    connector_palette = random.sample(available, connector_pool_size)
    body_pool = [c for c in available if c not in connector_palette]
    if len(body_pool) < 2:
        return None

    groups = sorted(group_counts)
    group_meta: dict[int, dict] = {}
    for gi in groups:
        members = [item for item in items if item["group"] == gi]
        h, w = int(members[0]["h"]), int(members[0]["w"])
        conn_pos = _choose_connector_pos(h, w)
        connectors = random.sample(connector_palette, len(members))
        body_color = int(random.choice(body_pool))
        group_meta[gi] = {
            "body_color": int(body_color),
            "conn_pos": (int(conn_pos[0]), int(conn_pos[1])),
            "connectors": connectors,
        }

    output_pieces: dict[int, np.ndarray] = {}
    source_pieces: dict[int, np.ndarray] = {}
    used_per_group = {gi: 0 for gi in groups}
    for idx, item in enumerate(items):
        gi = int(item["group"])
        meta = group_meta[gi]
        conn_color = int(meta["connectors"][used_per_group[gi]])
        used_per_group[gi] += 1
        h, w = int(item["h"]), int(item["w"])
        body_color = int(meta["body_color"])
        conn_pos = meta["conn_pos"]
        opp = _opposite_corner(h, w, conn_pos)

        out_patch = np.full((h, w), int(body_color), dtype=int)
        out_patch[int(conn_pos[0]), int(conn_pos[1])] = int(conn_color)

        src_patch = np.full((h, w), int(bg), dtype=int)
        src_patch[int(conn_pos[0]), int(conn_pos[1])] = int(conn_color)
        src_patch[int(opp[0]), int(opp[1])] = int(body_color)

        output_pieces[idx] = out_patch
        source_pieces[idx] = src_patch
        item["conn_color"] = int(conn_color)
        item["conn_pos"] = (int(conn_pos[0]), int(conn_pos[1]))
        item["body_color"] = int(body_color)

    return output_pieces, source_pieces


def _render_output(dh: int, dw: int, base: int, items: list[dict], output_pieces: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    for idx, item in enumerate(items):
        paste(out, output_pieces[idx], int(item["r"]), int(item["c"]), overwrite=True)
    return out


def _scatter_markers(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    items: list[dict],
    source_pieces: dict[int, np.ndarray],
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for item in items:
        r = int(item["r"])
        c = int(item["c"])
        h = int(item["h"])
        w = int(item["w"])
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = int(bg)
        cr, cc = item["conn_pos"]
        inp[tr0 + r + cr, tc0 + c + cc] = int(item["conn_color"])

    placement = np.full((H, W), int(bg), dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    order = list(range(len(items)))
    random.shuffle(order)
    order.sort(key=lambda i: source_pieces[i].size, reverse=True)
    for idx in order:
        reserve = np.full(source_pieces[idx].shape, int(base), dtype=int)
        rr, cc = random_free_location(placement, reserve, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, source_pieces[idx], int(rr), int(cc), transparent=bg)
        paste(placement, reserve, int(rr), int(cc), overwrite=True)
    return inp


def _has_duplicate_shape_with_distinct_connectors(items: list[dict]) -> bool:
    groups: dict[tuple[int, int, int], set[int]] = {}
    counts: dict[tuple[int, int, int], int] = {}
    for item in items:
        key = (int(item["h"]), int(item["w"]), int(item["body_color"]))
        groups.setdefault(key, set()).add(int(item["conn_color"]))
        counts[key] = counts.get(key, 0) + 1
    return any(counts[key] >= 2 and len(groups[key]) >= 2 for key in counts)


def _has_connector_reuse_across_shapes(items: list[dict]) -> bool:
    seen: dict[int, set[tuple[int, int]]] = {}
    for item in items:
        seen.setdefault(int(item["conn_color"]), set()).add((int(item["h"]), int(item["w"])))
    return any(len(shapes) >= 2 for shapes in seen.values())


def _unique_hole_keys(items: list[dict]) -> bool:
    keys = {
        (int(item["h"]), int(item["w"]), int(item["conn_color"]), tuple(map(int, item["conn_pos"])))
        for item in items
    }
    return len(keys) == len(items)


def _gen_attempt() -> np.ndarray | None:
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0

    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    counts = _sample_group_counts(total)
    shapes = _sample_shapes(dh, dw, counts)
    if shapes is None:
        return None

    items: list[dict] = []
    for gi, ((h, w), cnt) in enumerate(zip(shapes, counts)):
        for _ in range(cnt):
            items.append({"group": int(gi), "h": int(h), "w": int(w)})

    items = _place_rects(dh, dw, items)
    if items is None:
        return None

    bg, base = random.sample(range(10), 2)
    built = _build_items(items, bg=bg, base=base)
    if built is None:
        return None
    output_pieces, source_pieces = built

    if not _unique_hole_keys(items):
        return None
    if not _has_duplicate_shape_with_distinct_connectors(items):
        return None
    if not _has_connector_reuse_across_shapes(items):
        return None

    out = _render_output(dh, dw, base, items, output_pieces)
    try:
        inp = _scatter_markers(H, W, bg, base, tr0, tc0, dh, dw, items, source_pieces)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
