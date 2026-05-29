# Combo variant of 5dbc8537 (index 5)
# combo_spec: D1/2+D3/2
# Name: 连接器+洞内引导标记双重对齐

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

# -----------------------------------------------------------------------------
# Design analysis for the D1/2 + D3/2 combo
#
# a) Concrete behavior and execution trace
#    - The global scene still has a large base-colored strip and several full
#      rectangular sprites outside it.
#    - The destination side changes in two coupled ways:
#      1. the strip no longer contains one connected cavity to pack by search;
#         instead it contains isolated rectangular slots;
#      2. each slot shows exactly one colored guide pixel inside it.
#    - That guide pixel is not decorative. It is the slot's connector anchor:
#      the matching sprite contains the same connector color at the same
#      relative coordinate inside its own rectangle, and the sprite must be
#      pasted so that those two pixels coincide.
#    - Small example: three 2x3 sprites share the same yellow/green body patch.
#      Their connector signatures are:
#        A = red at top-left
#        B = red at bottom-right
#        C = blue at top-left
#      The three empty 2x3 slots show the same three markers. Shape alone
#      cannot distinguish any of them; color alone still confuses A vs B; guide
#      position alone still confuses A vs C. The correct key is therefore
#      `(h, w, connector_color, connector_offset)`.
#
# b) Complexity calibration
#    - The generator forces repeated size groups, not merely occasional ones.
#    - At least one duplicated size group has both:
#      1. a repeated connector color across different offsets, and
#      2. a repeated offset across different connector colors.
#      This prevents the task from collapsing to shape-only, color-only, or
#      position-only matching.
#    - Connector colors come from a small reserved palette and are never reused
#      in sprite body fills, so the marker remains visually salient and easy to
#      parse from a few examples.
#
# c) Hidden-assumption audit against the original main()
#    - The original `base_col = largest non-bg component` assumption still
#      holds, because the strip is much larger than any sprite.
#    - `dst_bbox = bbox(base pixels)` still identifies the template crop.
#    - The original DFS over arbitrary same-sized windows no longer reflects the
#      intended rule. Once duplicate sizes exist, packing by size alone is
#      underconstrained; the solver must read isolated slots directly.
#    - The solver now assumes every slot component is a filled rectangle and
#      contains exactly one non-bg, non-base marker pixel. The generator makes
#      those conditions explicit.
#    - The solver also assumes connector colors never appear in sprite body
#      pixels. The generator reserves a dedicated connector palette to guarantee
#      that single-pixel extraction is reliable.
#
# d) Placement legality
#    - Slots are placed with a 1-cell base border around the strip and at least
#      a 1-cell base gap along the strip's long axis, so slot components never
#      merge.
#    - Each scattered sprite is placed with `padding=1`, keeping outside
#      components separate from one another and from the template.
#    - All connector anchors are chosen from valid coordinates inside the
#      sprite/slot rectangle, so marker alignment never points outside bounds.
#
# e) Visual observability
#    - A human can literally match colored anchor dots inside the strip to the
#      same dots on outside rectangles.
#    - Because the generator creates duplicate sizes and deliberately reuses
#      colors and positions inside one size group, multiple examples visibly
#      demonstrate that both the color and the anchor location matter.
#
# f) Visual cleanliness
#    - Each slot contains only one visible marker pixel, so the strip remains
#      readable and there is no multi-layer annotation clutter.
#    - Sprite bodies keep simple 1-2 color patterns; connector pixels are the
#      only reserved colors, so the markers stand out clearly.
#
# g) Change inferability
#    - No hidden use of numeric color ids is involved. The relation is purely
#      geometric and visual: copy the rectangle into the slot whose internal dot
#      has the same color and the same relative placement.
#
# h) New solver assumptions and generator guarantees
#    - Hole extraction loops assume disconnected non-base components; generator
#      lays out isolated rectangular slots.
#    - Connector extraction assumes one connector per slot and per sprite;
#      generator creates exactly one.
#    - Dictionary matching assumes `(h, w, color, offset)` is unique; generator
#      validates uniqueness before accepting a sample.
# -----------------------------------------------------------------------------


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
    Fill guide-marked rectangular slots in a template strip by aligning sprite connector pixels.

    Input structure:
    - The background is the most common color, while one larger non-background color forms a strip-shaped template.
    - Inside that strip are isolated rectangular slots. Each slot is mostly background but contains one colored guide pixel.
    - Outside the strip are full rectangular sprites, each carrying one connector pixel of a matching marker color.

    Rule:
    - Detect the template from the largest non-background connected component and crop to its bbox.
    - Read every slot as a disconnected rectangle, recording its size, connector color, and connector offset.
    - Read every outside sprite the same way, then place each sprite into the slot with the same full signature.
    - Return the filled template crop.

    Key insight:
    - Duplicate rectangle sizes are intentional. The correct match is not determined by size alone, nor by marker color alone:
      the sprite's connector pixel must match both the slot's marker color and its exact relative position inside the rectangle.
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
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
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


# --------------- generator ---------------

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
MAX_RETRIES = 500


def _canonical_body_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    out = np.full((h, w), int(cols[0]), dtype=int)
    if len(cols) == 1 or (h == 1 and w == 1):
        return out

    alt = int(cols[1])
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
    num_groups = random.randint(MIN_GROUPS, min(MAX_GROUPS, total - 2))
    counts = [3] + [1] * (num_groups - 1)
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


def _anchor_positions(h: int, w: int) -> list[tuple[int, int]]:
    pts = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
        (0, w // 2),
        (h - 1, w // 2),
        (h // 2, 0),
        (h // 2, w - 1),
    ]
    uniq: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for p in pts:
        if p not in seen:
            seen.add(p)
            uniq.append((int(p[0]), int(p[1])))
    return uniq


def _sample_shapes(dh: int, dw: int, counts: list[int]) -> list[tuple[int, int]] | None:
    shapes = _shape_candidates(dh, dw)
    if len(shapes) < len(counts):
        return None

    fusion_idxs = [i for i, cnt in enumerate(counts) if cnt >= 3]
    assert fusion_idxs
    for _ in range(120):
        pool = shapes[:]
        random.shuffle(pool)
        chosen: list[tuple[int, int]] = []
        ok = True
        fusion_i = random.choice(fusion_idxs)
        for gi, cnt in enumerate(counts):
            found = None
            for shape in pool:
                h, w = map(int, shape)
                if gi == fusion_i and (h * w == 1 or len(_anchor_positions(h, w)) < 2):
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


def _choose_connector_palette(bg: int, base: int) -> tuple[list[int], list[int]] | None:
    available = [c for c in range(10) if c not in (bg, base)]
    if len(available) < 5:
        return None
    random.shuffle(available)
    connector_count = random.randint(2, min(3, len(available) - 2))
    connectors = available[:connector_count]
    body_pool = available[connector_count:]
    if len(body_pool) < 2:
        return None
    return connectors, body_pool


def _build_group_assignments(
    members: list[dict],
    connector_palette: list[int],
    *,
    require_fusion: bool,
) -> list[tuple[int, tuple[int, int]]] | None:
    h, w = int(members[0]["h"]), int(members[0]["w"])
    anchors = _anchor_positions(h, w)
    all_pairs = [(int(col), (int(r), int(c))) for col in connector_palette for r, c in anchors]
    if len(all_pairs) < len(members):
        return None

    if require_fusion:
        if len(members) < 3 or len(connector_palette) < 2 or len(anchors) < 2:
            return None
        col_a, col_b = map(int, random.sample(connector_palette, 2))
        pos_a, pos_b = random.sample(anchors, 2)
        chosen = [
            (int(col_a), (int(pos_a[0]), int(pos_a[1]))),
            (int(col_a), (int(pos_b[0]), int(pos_b[1]))),
            (int(col_b), (int(pos_a[0]), int(pos_a[1]))),
        ]
        used = set(chosen)
        extras = [pair for pair in all_pairs if pair not in used]
        random.shuffle(extras)
        while len(chosen) < len(members):
            assert extras
            chosen.append(extras.pop())
        return chosen

    random.shuffle(all_pairs)
    return all_pairs[: len(members)]


def _build_pieces(items: list[dict], *, bg: int, base: int) -> dict[int, np.ndarray] | None:
    chosen = _choose_connector_palette(bg, base)
    if chosen is None:
        return None
    connector_palette, body_pool = chosen

    groups: dict[int, list[dict]] = {}
    for item in items:
        groups.setdefault(int(item["group"]), []).append(item)

    fusion_candidates = [
        gi
        for gi, members in groups.items()
        if len(members) >= 3 and members[0]["h"] * members[0]["w"] > 1 and len(_anchor_positions(members[0]["h"], members[0]["w"])) >= 2
    ]
    if not fusion_candidates:
        return None
    fusion_group = int(random.choice(fusion_candidates))

    group_bodies: dict[int, np.ndarray] = {}
    group_pairs: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for gi, members in groups.items():
        h, w = int(members[0]["h"]), int(members[0]["w"])
        if h == 1 and w == 1:
            body = np.zeros((1, 1), dtype=int)
        else:
            body_cols = [int(random.choice(body_pool))]
            if h * w >= 4 and random.random() < 0.45:
                secondary_choices = [c for c in body_pool if c != body_cols[0]]
                if secondary_choices:
                    body_cols.append(int(random.choice(secondary_choices)))
            body = _canonical_body_patch(h, w, body_cols)
        pairs = _build_group_assignments(
            members,
            connector_palette,
            require_fusion=(int(gi) == fusion_group),
        )
        if pairs is None:
            return None
        random.shuffle(pairs)
        group_bodies[int(gi)] = body
        group_pairs[int(gi)] = pairs

    pieces: dict[int, np.ndarray] = {}
    per_group_used = {int(gi): 0 for gi in groups}
    for idx, item in enumerate(items):
        gi = int(item["group"])
        patch = group_bodies[gi].copy()
        conn_color, conn_pos = group_pairs[gi][per_group_used[gi]]
        per_group_used[gi] += 1
        rr, cc = conn_pos
        patch[int(rr), int(cc)] = int(conn_color)
        pieces[idx] = patch
        item["conn_color"] = int(conn_color)
        item["conn_pos"] = (int(rr), int(cc))
    return pieces


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


def _observable_fusion_support(items: list[dict]) -> bool:
    by_shape: dict[tuple[int, int], list[dict]] = {}
    for item in items:
        by_shape.setdefault((int(item["h"]), int(item["w"])), []).append(item)
    for members in by_shape.values():
        if len(members) < 3:
            continue
        color_counts: dict[int, int] = {}
        pos_counts: dict[tuple[int, int], int] = {}
        for item in members:
            color_counts[int(item["conn_color"])] = color_counts.get(int(item["conn_color"]), 0) + 1
            pos = (int(item["conn_pos"][0]), int(item["conn_pos"][1]))
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        if (
            len(color_counts) >= 2
            and len(pos_counts) >= 2
            and any(v >= 2 for v in color_counts.values())
            and any(v >= 2 for v in pos_counts.values())
        ):
            return True
    return False


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
    pieces = _build_pieces(placed, bg=bg, base=base)
    if pieces is None:
        return None

    if not _observable_fusion_support(placed):
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
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
