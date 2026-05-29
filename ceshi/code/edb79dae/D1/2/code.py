# Variant of edb79dae
# Dimension: D1 — 第一步匹配条件：颜色匹配 (match_cond)
# Change: 精灵颜色 == 框架内目标像素颜色（s.color == d.color），通过颜色一致性将外部精灵与内部单色像素配对... → 尺寸匹配

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location

# Design analysis:
#   a) Concrete behavior:
#      - Frame interior still contains colored placeholder rectangles, because the legend still needs a visible
#        source color to determine the final recoloring.
#      - The changed step is shape selection: outside sprites no longer share those placeholder colors. Instead,
#        each placeholder color owns one unique bbox size, and every outside sprite is chosen by matching that size.
#      - Example trace:
#        original: a red 4x4 placeholder looks for the red sprite outside;
#        variant: a red 4x4 placeholder looks for the only outside sprite whose bbox is 4x4, even if that sprite
#        is teal; the red legend still says which final color to repaint it to.
#   b) Complexity calibration:
#      - Each sample uses 2-3 distinct placeholder colors and 2-3 distinct rectangle sizes chosen from a small
#        visible catalog. Every chosen size repeats at least twice inside the frame, so the size rule has support.
#      - Placeholder-color -> size is randomized per sample, so color is not a stable proxy across examples.
#   c) Old assumptions that break under D1:
#      - Old `main()` filtered sprite candidates with `c["col"] == src_col`; that hard-wires color matching.
#      - Old generator used one global N for all placeholders/sprites, which makes size useless as a discriminator.
#      - Both are replaced: `main()` now indexes sprites by bbox dims, and the generator creates multiple sizes.
#   d) Placement legality:
#      - Interior rectangles are shelf-packed with 1-cell bg gaps and at least 1-cell margin to the frame border.
#      - Outside patches still use `random_free_location`, so sprites/legends do not overlap the frame or each other.
#   e) Visual observability:
#      - Same-sized holes appear multiple times; the same outside sprite shape reappears in those holes after recolor.
#      - Sprite colors are chosen disjoint from placeholder colors, so the old color-matching story is visibly wrong.
#   f) Visual cleanliness:
#      - Only three element types remain: frame, interior placeholder rectangles, and separated outside patches.
#      - Legends keep the original compact 3x4 form; sprites are monochrome and isolated by bg gaps.
#   g) Color inference:
#      - Colors still matter only in the legend. Since each legend touches exactly one placeholder color plus one
#        target color, the recoloring cue stays structured and learnable while the size cue controls shape choice.


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color.
    Return (frame_color, (r0,c0,r1,c1)).
    """
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    frame_col, bbox = candidates[0]
    return int(frame_col), bbox


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a framed layout with recolored sprites chosen by placeholder size rather than sprite color.

    Input structure:
    - A bg canvas contains one gray rectangular frame whose interior has several solid monochrome rectangles of
      different sizes.
    - Outside the frame are monochrome sprite templates and small two-color legend blocks; legend source colors
      match the interior placeholder colors, but sprite colors do not have to.

    Rule:
    - Detect the frame, its interior placeholder components, and the outside legend mappings.
    - For each placeholder color, use its rectangle size to locate the unique outside sprite with the same bbox size.
    - Stamp that sprite into every same-color placeholder location, then recolor the stamped pixels using the legend.

    Key insight: The placeholder color no longer identifies which outside sprite to use; it only keys the final
    recoloring. The sprite identity is determined by measuring bbox dimensions and matching those sizes.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            placeholders.append((col, bbox_coords(pts)))

    phw: dict[int, tuple[int, int]] = {}
    for col in sorted(placeholder_colors):
        dims = sorted({_bbox_dims(b) for c, b in placeholders if c == col})
        assert len(dims) == 1, f"ambiguous placeholder sizes for color {col}: {dims}"
        phw[col] = dims[0]
    size_to_src = {dims: col for col, dims in phw.items()}
    assert len(size_to_src) == len(phw), f"placeholder sizes are not unique: {sorted(phw.items())}"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    cid = 0
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": [(int(r), int(c)) for r, c in pts],
                    "bbox": bbox_coords(pts),
                    "area": len(pts),
                }
            )
            cid += 1

    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"] or not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue

            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]

            assert src in placeholder_colors, f"legend source color {src} is not a placeholder"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])
    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )

    non_legend = [c for c in comps if c["id"] not in legend_ids]
    sprite_coords: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for want_dims, src_col in sorted(size_to_src.items()):
        sized = [c for c in non_legend if _bbox_dims(c["bbox"]) == want_dims]
        assert len(sized) == 1, f"expected one sprite with size {want_dims} for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        coords = tuple(sorted((r - sr0, c - sc0) for r, c in base["pts"]))
        sprite_coords[want_dims] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, bbox in placeholders:
        pr0, pc0, pr1, pc1 = bbox
        want_dims = _bbox_dims(bbox)
        assert src_col in color_map, f"missing legend mapping for placeholder color {src_col}"
        assert want_dims in sprite_coords, f"missing sprite for size {want_dims}"
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[want_dims]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col

    return out


# --------------- generator ---------------

FRAME_COL = 5
SIZE_OPTIONS = [(3, 3), (3, 4), (4, 3), (4, 4), (5, 5)]
NUM_COLORS_RANGE = (2, 3)
TOTAL_BLOCKS_RANGE = (6, 9)
SPRITE_FILL_MIN = 0.45
SPRITE_FILL_MAX = 0.82
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 400


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _manhattan_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while r != er or c != ec:
        if r != er and c != ec:
            if random.random() < 0.5:
                r += 1 if er > r else -1
            else:
                c += 1 if ec > c else -1
        elif r != er:
            r += 1 if er > r else -1
        else:
            c += 1 if ec > c else -1
        mask[r, c] = True


def _sprite_mask(h: int, w: int) -> np.ndarray:
    total = h * w
    mask = np.zeros((h, w), dtype=bool)

    center_r = random.randrange(1, h - 1) if h >= 3 else random.randrange(h)
    center_c = random.randrange(1, w - 1) if w >= 3 else random.randrange(w)
    center = (center_r, center_c)
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    for anchor in anchors:
        _manhattan_path(mask, anchor, center)

    lo = max(int(np.ceil(SPRITE_FILL_MIN * total)), int(mask.sum()))
    hi = min(int(np.floor(SPRITE_FILL_MAX * total)), total - 1)
    if hi < lo:
        hi = lo
    target = random.randint(lo, hi)

    while int(mask.sum()) < target:
        dilated = np.zeros_like(mask)
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        frontier = list(zip(*np.where(dilated & ~mask)))
        if not frontier:
            break
        r, c = random.choice(frontier)
        mask[r, c] = True

    assert mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()
    assert not mask.all()
    return mask


def _choose_counts(k: int, total: int) -> list[int]:
    counts = [2] * k
    extra = total - 2 * k
    for _ in range(extra):
        counts[random.randrange(k)] += 1
    random.shuffle(counts)
    return counts


def _pack_blocks(
    items: list[tuple[int, int, int]],
    *,
    max_h: int,
    max_w: int,
    tries: int = 200,
) -> tuple[list[tuple[int, int, int, int, int]], int, int] | None:
    best = None
    for _ in range(tries):
        order = items[:]
        random.shuffle(order)
        order.sort(key=lambda x: (-(x[1] * x[2]), -x[1], -x[2], random.random()))
        placements = []
        y = 0
        x = 0
        row_h = 0
        content_w = 0
        ok = True
        for src, h, w in order:
            need_x = x if x == 0 else x + 1
            if need_x + w > max_w:
                y += row_h + 1
                x = 0
                row_h = 0
                need_x = 0
            if y + h > max_h:
                ok = False
                break
            placements.append((src, y, need_x, h, w))
            x = need_x + w
            row_h = max(row_h, h)
            content_w = max(content_w, x)
        if not ok:
            continue
        content_h = y + row_h
        score = (content_h, content_w, len({p[1] for p in placements}))
        if best is None or score < best[0]:
            best = (score, placements, content_h, content_w)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    total = random.randint(*TOTAL_BLOCKS_RANGE)
    if total < 2 * k:
        return None

    src_cols = random.sample(usable, k)
    size_choices = random.sample(SIZE_OPTIONS, k)
    sprite_pool = [c for c in usable if c not in src_cols]
    if len(sprite_pool) < k:
        return None
    sprite_cols = random.sample(sprite_pool, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]

    counts = _choose_counts(k, total)
    size_by_src = dict(zip(src_cols, size_choices))
    sprite_color_by_src = dict(zip(src_cols, sprite_cols))
    tgt_by_src = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(*size_by_src[src]) for src in src_cols}

    items = []
    for src, count in zip(src_cols, counts):
        h, w = size_by_src[src]
        items.extend([(src, h, w)] * count)

    packed = _pack_blocks(items, max_h=MAX_FRAME_DIM - 4, max_w=MAX_FRAME_DIM - 4)
    if packed is None:
        return None
    placements, content_h, content_w = packed
    if len({size_by_src[src] for src in src_cols}) < 2:
        return None

    min_fh = max(MIN_FRAME_H, content_h + 4)
    min_fw = max(MIN_FRAME_W, content_w + 4)
    max_fh = min(MAX_FRAME_DIM, max(min_fh, content_h + 2 + FRAME_MARGIN_SLACK))
    max_fw = min(MAX_FRAME_DIM, max(min_fw, content_w + 2 + FRAME_MARGIN_SLACK))
    if min_fh > max_fh or min_fw > max_fw:
        return None
    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    y_margin = frame_h - 2 - content_h
    x_margin = frame_w - 2 - content_w
    if y_margin < 2 or x_margin < 2:
        return None
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for src, y, x, h, w in placements:
        r0 = 1 + top_margin + y
        c0 = 1 + left_margin + x
        paste(input_frame, np.full((h, w), src, dtype=int), r0, c0)
        paste(
            output_frame,
            _colored_sprite(masks[src], tgt_by_src[src], bg),
            r0,
            c0,
            transparent=bg,
        )

    ext_area = sum(h * w + 12 + 8 for _, h, w in items[:k]) + 12 * k
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    H, W = random.choice(possible_dims)
    grid = np.full((H, W), bg, dtype=int)
    reserve = np.full((H, W), bg, dtype=int)

    fr0 = random.choice([0, H - frame_h])
    fc0 = random.choice([0, W - frame_w])
    if H - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, H - frame_h + 1)
    if W - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, W - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src in src_cols:
        sprite = _colored_sprite(masks[src], sprite_color_by_src[src], bg)
        patches.append((sprite, True))
        patches.append((_legend(src, tgt_by_src[src]), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae size-match input")
