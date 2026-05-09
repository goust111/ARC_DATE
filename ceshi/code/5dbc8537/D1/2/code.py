# Variant of 5dbc8537
# Dimension: D1 — 匹配条件：精灵与洞的对应方式 (match_cond)
# Change: s.bbox == d.rect — 精灵外接矩形尺寸与洞的矩形尺寸完全相等（shape_match）... → 连接器颜色匹配

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# Design analysis for the D1 connector-match variant
# a) Concrete behavior and execution trace:
# - The template is still a full-width or full-height base-colored strip, and the answer is still its tight crop.
# - Each hole is now an isolated rectangle inside that strip. The rectangle is mostly bg, except for exactly one
#   visible connector pixel. The matching sprite carries the same connector color at the same offset in its bbox.
# - Small example: suppose there are two 2x3 sprites with the same yellow body pattern. In the original task they
#   would be interchangeable if both fit 2x3 holes. In this variant one sprite has an orange connector and the
#   other has a teal connector; the two 2x3 holes show the same orange/teal dots, so the solver must match by
#   (shape + connector), not by shape alone.
# - The affected pipeline stage is the destination matching stage. Original logic enumerated any same-sized
#   rectangle inside one connected hole region. Here that would become underconstrained once duplicate sizes
#   are introduced, so the generator makes holes mutually separated and the solver extracts each hole directly.
#
# b) Complexity calibration:
# - At least one non-1x1 shape group is duplicated (2-4 copies).
# - Duplicated shapes reuse the same body patch; only connector colors differ, so the D1 change is visually
#   necessary rather than ornamental.
# - Connector colors come from a small reusable palette. Colors can repeat across different shapes, so the key is
#   intentionally "shape + connector" instead of "globally unique color".
#
# c) Hidden assumptions audit against the original main():
# - `base_col` as the largest non-bg connected component still holds; the strip is much larger than any sprite or
#   connector marker.
# - `dst = bbox(base pixels)` still identifies the template region.
# - The old `hole_mask = dst != base_col` remains useful, but the old DFS over arbitrary same-sized rectangles
#   relied on touching cavities plus effectively unique sizes. That is no longer valid once duplicate sizes exist.
# - The old source extraction assumed sprites are full rectangles outside the template; that still holds.
# - New solver assumptions are explicit: every hole is a disconnected full rectangle; every hole has exactly one
#   connector pixel; every sprite contains exactly one connector pixel whose color belongs to the hole-marker set;
#   and `(h, w, connector_color, connector_offset)` is unique across all pieces.
#
# d) Placement legality:
# - The generator places holes with a 1-cell base border around the strip and a 1-cell base gap between holes, so
#   holes never touch and never merge under 8-connectivity.
# - Sprites are still scattered outside the template using `random_free_location(..., padding=1)`, so they remain
#   separate components and never touch the template.
#
# e) Visual observability:
# - Duplicate same-size pieces are guaranteed, and their interiors are intentionally reused. The only difference is
#   the connector color, mirrored by the hole markers, so the new matching rule is visible in 2-3 examples.
#
# f) Visual cleanliness:
# - Each hole has only one colored marker. There are no overlapping annotations, no multi-layer tags, and the base
#   strip still reads as a clean frame with rectangular cut-outs.
#
# g) Color inferability:
# - Connector colors are few and directly copied from sprite to hole. No hidden color remapping or id-based sorting
#   is used.
#
# h) Post-change solver assumptions audit:
# - Loop over hole components assumes each component bbox is completely filled by the hole mask; generator enforces
#   disconnected rectangles.
# - Loop over sprites assumes exactly one connector-colored cell per sprite; generator reserves connector colors and
#   never uses them for body fills.
# - Matching assumes one-to-one keys; generator validates this before oracle acceptance.


def _extract_single_connector(
    patch: np.ndarray, *, marker_colors: set[int], forbidden: set[int]
) -> tuple[int, tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    cols: list[int] = []
    h, w = map(int, patch.shape)
    for r in range(h):
        for c in range(w):
            val = int(patch[r, c])
            if val in forbidden:
                continue
            if val in marker_colors:
                coords.append((int(r), int(c)))
                cols.append(int(val))
    assert len(coords) == 1
    return int(cols[0]), (int(coords[0][0]), int(coords[0][1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill marked rectangular cavities in a strip template by matching sprites through connector pixels.

    Input structure:
    - The background is the most common color. A single base color forms a large strip-shaped template.
    - Inside that strip are isolated rectangular holes; each hole contains one visible connector pixel.
    - Outside the template sit full rectangular sprites, each carrying one connector pixel of its own.

    Rule:
    - Detect the template bbox from the largest non-background connected component.
    - Read every hole as a disconnected rectangle, recording its size plus the connector color/offset inside it.
    - Read every outside sprite the same way, then place each sprite into the hole with the same signature.
    - Output the tight crop of the filled template.

    Key insight: Shape alone is no longer sufficient when multiple sprites share a bbox; the connector color at the
    corresponding relative position resolves which same-sized sprite belongs in which hole.
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

    hole_marker_colors = {
        int(dst[r, c])
        for comp in hole_comps
        for r, c in comp
        if int(dst[r, c]) not in (int(base_col), int(bg))
    }
    assert hole_marker_colors

    holes: dict[tuple[int, int, int, tuple[int, int]], tuple[int, int]] = {}
    for coords in hole_comps:
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        h = int(hr1 - hr0 + 1)
        w = int(hc1 - hc0 + 1)
        assert int(coords.shape[0]) == h * w
        patch = dst[int(hr0) : int(hr1) + 1, int(hc0) : int(hc1) + 1]
        conn_col, conn_pos = _extract_single_connector(
            patch, marker_colors=hole_marker_colors, forbidden={bg, base_col}
        )
        key = (int(h), int(w), int(conn_col), (int(conn_pos[0]), int(conn_pos[1])))
        assert key not in holes
        holes[key] = (int(hr0), int(hc0))

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert sprite_comps

    sprites: dict[tuple[int, int, int, tuple[int, int]], np.ndarray] = {}
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (int(pr0), int(pc0), int(pr1), int(pc1)))
        ph, pw = map(int, patch.shape)
        assert int(comp["coords"].shape[0]) == ph * pw
        conn_col, conn_pos = _extract_single_connector(
            patch, marker_colors=hole_marker_colors, forbidden={bg, base_col}
        )
        key = (int(ph), int(pw), int(conn_col), (int(conn_pos[0]), int(conn_pos[1])))
        assert key in holes
        assert key not in sprites
        sprites[key] = patch

    assert set(sprites) == set(holes)

    out = np.full((dh, dw), int(base_col), dtype=int)
    for key, patch in sprites.items():
        rr0, cc0 = holes[key]
        paste(out, patch, int(rr0), int(cc0), overwrite=True)
    return out


# --------------- generator (connector-match variant) ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 11
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.20, 0.34
MAX_LONG_DIM = 5
MAX_SHORT_DIM = 4
MIN_GROUPS, MAX_GROUPS = 3, 5
MAX_GROUP_COUNT = 4
MIN_LONG_COVERAGE = 0.62
MAX_RETRIES = 400


def _canonical_body_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    out = np.full((h, w), cols[0], dtype=int)
    if len(cols) == 1 or (h == 1 and w == 1):
        return out

    alt = cols[1]
    rr, cc = np.indices((h, w))
    mode = random.choice(["checker", "dots", "split", "diag"])
    if mode == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = alt
    elif mode == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not mask.any() or mask.all():
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = alt
    elif mode == "diag":
        out[(rr - cc) >= random.randint(-(w - 1), h - 1)] = alt
    elif h > 1 and (w == 1 or random.random() < 0.5):
        cut = random.randint(1, h - 1)
        out[cut:, :] = alt
    else:
        cut = random.randint(1, w - 1)
        out[:, cut:] = alt
    return out


def _sample_group_counts(total: int) -> list[int]:
    num_groups = random.randint(MIN_GROUPS, min(MAX_GROUPS, total - 1))
    counts = [2] + [1] * (num_groups - 1)
    remaining = total - sum(counts)
    while remaining > 0:
        idxs = [i for i, v in enumerate(counts) if v < MAX_GROUP_COUNT]
        assert idxs
        i = random.choice(idxs)
        counts[i] += 1
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
    for ln in range(1, max_len + 1):
        for cr in range(1, max_cross + 1):
            h, w = (ln, cr) if long_is_rows else (cr, ln)
            if h * w <= max(2, (dh * dw) // 6):
                out.append((int(h), int(w)))
    return out


def _sample_shapes(dh: int, dw: int, counts: list[int]) -> list[tuple[int, int]] | None:
    shapes = _shape_candidates(dh, dw)
    if len(shapes) < len(counts):
        return None
    dup_idxs = [i for i, cnt in enumerate(counts) if cnt >= 2]
    assert dup_idxs
    for _ in range(80):
        pool = shapes[:]
        random.shuffle(pool)
        chosen: list[tuple[int, int]] = []
        ok = True
        dup_i = random.choice(dup_idxs)
        for gi, cnt in enumerate(counts):
            found = None
            for shape in pool:
                if gi == dup_i and shape[0] * shape[1] == 1:
                    continue
                found = shape
                break
            if found is None:
                ok = False
                break
            pool.remove(found)
            chosen.append((int(found[0]), int(found[1])))
        if not ok:
            continue
        long_is_rows = dh >= dw
        total_long = sum(int((h if long_is_rows else w) * cnt) for (h, w), cnt in zip(chosen, counts))
        area = sum(int(h * w) * int(cnt) for (h, w), cnt in zip(chosen, counts))
        lo = math.ceil(HOLE_FRAC_LO * dh * dw)
        hi = math.floor(HOLE_FRAC_HI * dh * dw)
        if total_long + sum(counts) - 1 <= (dh if long_is_rows else dw) - 2 and lo <= area <= hi:
            return chosen
    return None


def _place_rects(
    dh: int,
    dw: int,
    items: list[dict],
) -> list[dict] | None:
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
    if h == 1 and w == 1:
        return 0, 0
    opts = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    if h >= 3 and w >= 3:
        opts.add((h // 2, w // 2))
    return tuple(map(int, random.choice(sorted(opts))))


def _build_pieces(
    items: list[dict], *, bg: int, base: int
) -> tuple[dict[int, np.ndarray], set[int]] | None:
    available = [c for c in range(10) if c not in (bg, base)]
    group_counts: dict[int, int] = {}
    for item in items:
        group_counts[item["group"]] = group_counts.get(item["group"], 0) + 1

    needed_connector = max(3, max(group_counts.values()))
    if len(available) < needed_connector + 2:
        return None

    connector_palette = random.sample(available, needed_connector)
    body_pool = [c for c in available if c not in connector_palette]
    if len(body_pool) < 2:
        return None

    groups = sorted(group_counts)
    group_meta: dict[int, dict] = {}
    for gi in groups:
        members = [item for item in items if item["group"] == gi]
        h, w = int(members[0]["h"]), int(members[0]["w"])
        conn_pos = _choose_connector_pos(h, w)
        assigned = random.sample(connector_palette, len(members))
        if h == 1 and w == 1:
            body = np.zeros((1, 1), dtype=int)
        else:
            body_cols = [random.choice(body_pool)]
            if h * w >= 4 and random.random() < 0.45:
                second_choices = [c for c in body_pool if c != body_cols[0]]
                if second_choices:
                    body_cols.append(random.choice(second_choices))
            body = _canonical_body_patch(h, w, body_cols)
        group_meta[gi] = {
            "conn_pos": (int(conn_pos[0]), int(conn_pos[1])),
            "body": body,
            "connectors": assigned,
        }

    pieces: dict[int, np.ndarray] = {}
    marker_colors: set[int] = set()
    per_group_used: dict[int, int] = {gi: 0 for gi in groups}
    for idx, item in enumerate(items):
        gi = int(item["group"])
        meta = group_meta[gi]
        conn_col = int(meta["connectors"][per_group_used[gi]])
        per_group_used[gi] += 1
        marker_colors.add(conn_col)
        patch = meta["body"].copy()
        if patch.shape == (1, 1):
            patch[0, 0] = int(conn_col)
        else:
            rr, cc = meta["conn_pos"]
            patch[int(rr), int(cc)] = int(conn_col)
        pieces[idx] = patch
        item["conn_color"] = int(conn_col)
        item["conn_pos"] = (int(meta["conn_pos"][0]), int(meta["conn_pos"][1]))

    return pieces, marker_colors


def _render_output(dh: int, dw: int, base: int, items: list[dict], pieces: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    for idx, item in enumerate(items):
        paste(out, pieces[idx], int(item["r"]), int(item["c"]))
    return out


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    items: list[dict],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for idx, item in enumerate(items):
        r, c = int(item["r"]), int(item["c"])
        h, w = map(int, pieces[idx].shape)
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = int(bg)
        cr, cc = item["conn_pos"]
        inp[tr0 + r + cr, tc0 + c + cc] = int(item["conn_color"])

    placement = np.full((H, W), int(bg), dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    order = list(range(len(items)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for idx in order:
        patch = pieces[idx]
        rr, cc = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, int(rr), int(cc))
        paste(placement, patch, int(rr), int(cc))
    return inp


def _observable_duplicate_support(items: list[dict]) -> bool:
    groups: dict[tuple[int, int], set[int]] = {}
    counts: dict[tuple[int, int], int] = {}
    for item in items:
        key = (int(item["h"]), int(item["w"]))
        counts[key] = counts.get(key, 0) + 1
        groups.setdefault(key, set()).add(int(item["conn_color"]))
    return any(counts[key] >= 2 and len(groups[key]) >= 2 and key != (1, 1) for key in counts)


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

    placed = _place_rects(dh, dw, items)
    if placed is None:
        return None

    bg, base = random.sample(range(10), 2)
    built = _build_pieces(placed, bg=bg, base=base)
    if built is None:
        return None
    pieces, _marker_colors = built

    if not _observable_duplicate_support(placed):
        return None

    key_set: set[tuple[int, int, int, tuple[int, int]]] = set()
    for idx, item in enumerate(placed):
        h, w = map(int, pieces[idx].shape)
        key = (int(h), int(w), int(item["conn_color"]), (int(item["conn_pos"][0]), int(item["conn_pos"][1])))
        if key in key_set:
            return None
        key_set.add(key)

    out = _render_output(dh, dw, base, placed, pieces)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, placed, pieces)
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
    raise RuntimeError("failed to generate valid 5dbc8537 connector-match input")
