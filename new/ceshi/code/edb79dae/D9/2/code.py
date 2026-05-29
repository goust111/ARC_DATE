# Variant of edb79dae
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: 输出为灰色框架区域（grey frame region），可能裁剪到框架范围... → 按图例条目数分割输出

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
#
# a) Concrete behavior and trace
#    - The input keeps the original ingredients: one grey rectangular frame, solid
#      placeholder blocks inside it, same-color sprite templates outside, and
#      two-color legend glyphs outside.
#    - The original task builds one output frame crop containing every recolored
#      sprite together. This variant instead builds one frame-shaped subgrid per
#      legend-target color, then concatenates those subgrids horizontally with
#      full-height grey divider columns.
#    - Small trace:
#      If the frame contains source colors {red, blue, green} and legends say
#      red->yellow, blue->orange, green->teal, the original output is one frame
#      containing yellow/orange/teal sprites together. The new output becomes
#      [yellow-only frame] |5| [orange-only frame] |5| [teal-only frame].
#      Every panel preserves the same placeholder layout, but only the sprites of
#      that recolor-group are shown; other interior cells stay bg.
#
# b) Complexity calibration
#    - ARC outputs must stay <= 30 columns, so the single source frame must be
#      compact if we are going to duplicate it 2-3 times. I therefore shrink the
#      generated frame to a 2-column grid of 3x3 placeholders, giving frame width
#      9 and output widths 19 or 29 for 2 or 3 panels.
#    - The legend count is restricted to 2-3 and target colors are unique. This
#      avoids degenerate "group by color" cases where multiple legend entries
#      collapse into one output panel and makes panel-count visibly controlled by
#      the legend strip.
#
# c) Hidden assumptions audit against the original solver
#    - `_find_frame(...)`, placeholder extraction, legend parsing, and sprite
#      extraction all remain valid because the input-side objects are unchanged.
#    - The old `out = crop.copy(); out[1:-1, 1:-1] = bg` assumed there is exactly
#      one destination grid. That no longer holds; the variant must allocate a
#      separate destination panel per legend target before stamping sprites.
#    - Any ordering based on numeric color ids would be invisible to a human
#      solver, so panel order is derived from legend positions (reading order of
#      legend bboxes), which the generator makes explicit via a visible legend
#      strip.
#
# d) Placement legality
#    - The frame, legend strip, and sprite patches are all reserved on the input
#      canvas with background padding, so they never overlap.
#    - The legend strip has 1-cell bg gaps between entries; this keeps each entry
#      independently detectable by the existing bbox-overlap logic.
#    - Output panel width is bounded by construction: `k * frame_w + (k - 1) <=
#      29`, so the variant stays within ARC limits.
#
# e) Visual observability
#    - The legend entries are placed as one ordered strip rather than scattered,
#      so a human can see that the output panels follow legend order instead of
#      any color-id order.
#    - Every example has at least 2 panels, so the D9 change is always visible.
#
# f) Visual cleanliness
#    - Each output panel retains the original grey border, and panels are split by
#      full-height grey divider columns. This creates clear frame separation
#      without overlaying unrelated marks in the same area.
#
# g) Color-rule inferability
#    - Recoloring is still explained by the same source->target legends. The only
#      new inference is that equal target-color results are isolated into separate
#      frame panels by legend order; with unique targets, one legend entry yields
#      one visible panel, which is easy to read from 2-4 examples.
# ---------------------------------------------------------------------------


FRAME_COL = 5
SPRITE_N = 3
GRID_ROWS_CHOICES = (2, 3)
GRID_COLS = 2
NUM_COLORS_RANGE = (2, 3)  # inclusive
MAX_EMPTY_SLOTS = 1
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
PATCH_PADDING = 1
GEN_RETRIES = 400


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


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _collect_components(
    g: np.ndarray, outside: np.ndarray, bg: int, frame_col: int
) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts_tuples),
                }
            )
            cid += 1
    return comps


def _extract_legends(
    g: np.ndarray,
    comps: list[dict[str, object]],
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
) -> tuple[dict[int, int], set[int], list[dict[str, object]]]:
    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    legends: list[dict[str, object]] = []

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
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
            cset = set(sub_colors)
            if ci["col"] not in cset or cj["col"] not in cset:
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
            legends.append({"src": int(src), "tgt": int(tgt), "bbox": (r0, c0, r1, c1)})

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )
    return color_map, legend_ids, legends


def _extract_sprites(
    comps: list[dict[str, object]],
    legend_ids: set[int],
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> dict[int, tuple[tuple[tuple[int, int], ...], int, int]]:
    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1) == want_h and (c["bbox"][3] - c["bbox"][1] + 1) == want_w
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]

        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)
    return sprite_coords


def _panel_order(legends: list[dict[str, object]]) -> list[int]:
    # New assumption: panel order follows visible legend order, never color id.
    rep_bbox: dict[int, tuple[int, int, int, int]] = {}
    for item in legends:
        tgt = int(item["tgt"])
        bbox = item["bbox"]
        if tgt not in rep_bbox or (bbox[0], bbox[1]) < (rep_bbox[tgt][0], rep_bbox[tgt][1]):
            rep_bbox[tgt] = bbox
    order = sorted(rep_bbox, key=lambda tgt: (rep_bbox[tgt][0], rep_bbox[tgt][1], rep_bbox[tgt][2], rep_bbox[tgt][3]))
    return [int(tgt) for tgt in order]


def _concat_panels(panels: list[np.ndarray], divider_color: int) -> np.ndarray:
    assert panels, "no panels to concatenate"
    h = panels[0].shape[0]
    divider = np.full((h, 1), divider_color, dtype=int)
    pieces = [panels[0]]
    for panel in panels[1:]:
        assert panel.shape[0] == h, "panel heights must match"
        pieces.extend([divider, panel])
    return np.concatenate(pieces, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the solved frame into multiple frame panels ordered by the visible legend entries.

    Input structure:
    - A background canvas contains one grey rectangular frame with solid placeholder blocks inside.
    - Outside the frame are sprite templates in placeholder colors and a strip of two-color legend glyphs.

    Rule:
    - Recover the same placeholder->sprite and placeholder->target-color matches as in the original task.
    - Instead of drawing every recolored sprite into one output frame, build one frame panel per legend target color.
    - In each panel, keep only the sprites whose legend recolors them to that panel's color, then concatenate the panels left-to-right in legend order.

    Key insight: The legend does double duty now: it still provides recoloring, and its spatial order determines how the separate output panels are arranged.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            placeholders.append((col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps = _collect_components(g, outside, bg, frame_col)
    color_map, legend_ids, legends = _extract_legends(g, comps, bg, frame_col, placeholder_colors)
    sprite_coords = _extract_sprites(comps, legend_ids, placeholder_colors, phw)
    ordered_tgts = _panel_order(legends)

    panels = {tgt: crop.copy() for tgt in ordered_tgts}
    for tgt in panels:
        panels[tgt][1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_coords and src_col in color_map, f"placeholder color {src_col} missing sprite or color mapping"
        tgt_col = int(color_map[src_col])
        assert tgt_col in panels, f"target color {tgt_col} has no output panel"
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        assert pr0 + sh <= h and pc0 + sw <= w, f"sprite for color {src_col} does not fit at placeholder"
        for dr, dc in coords:
            rr, cc = pr0 + dr, pc0 + dc
            panels[tgt_col][rr, cc] = tgt_col

    return _concat_panels([panels[tgt] for tgt in ordered_tgts], frame_col)


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _legend_strip(order: list[int], color_to_tgt: dict[int, int], bg: int) -> np.ndarray:
    parts = []
    gap = np.full((3, 1), bg, dtype=int)
    for i, src in enumerate(order):
        if i:
            parts.append(gap)
        parts.append(_legend(src, color_to_tgt[src]))
    return np.concatenate(parts, axis=1)


def _sprite_mask() -> np.ndarray:
    target = random.randint(5, 7)
    mask = np.zeros((SPRITE_N, SPRITE_N), dtype=bool)

    c = random.randrange(SPRITE_N)
    for r in range(SPRITE_N):
        c = max(0, min(SPRITE_N - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True
    mask[random.randrange(SPRITE_N), :] = True

    if random.random() < 0.5:
        mask = mask.T.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    while int(mask.sum()) < target:
        dilated = np.zeros_like(mask)
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        frontier = list(zip(*np.where(dilated & ~mask)))
        r, c = random.choice(frontier)
        mask[r, c] = True

    assert mask.sum() < SPRITE_N * SPRITE_N
    return mask


def _distinct_masks(src_cols: list[int]) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for src in src_cols:
        for _ in range(100):
            mask = _sprite_mask()
            key = tuple(tuple(int(v) for v in row) for row in mask.astype(int))
            if key not in seen:
                seen.add(key)
                masks[src] = mask
                break
        else:
            return {}
    return masks


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _choose_empty_slots(rows: int, cols: int, empty_count: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> tuple[int, int] | None:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return None
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return int(r), int(c)


def _frame_dims(rows: int, cols: int) -> tuple[int, int]:
    inner_h = rows * SPRITE_N + (rows - 1)
    inner_w = cols * SPRITE_N + (cols - 1)
    return inner_h + 2, inner_w + 2


def _build_expected_panels(
    rows: int,
    cols: int,
    bg: int,
    masks: dict[int, np.ndarray],
    color_to_tgt: dict[int, int],
    legend_order: list[int],
    cell_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    frame_h, frame_w = _frame_dims(rows, cols)
    panels = {color_to_tgt[src]: _make_bordered_frame(frame_h, frame_w, bg) for src in legend_order}
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + rr * (SPRITE_N + 1)
        c0 = 1 + cc * (SPRITE_N + 1)
        paste(
            panels[color_to_tgt[src]],
            _colored_sprite(masks[src], color_to_tgt[src], bg),
            r0,
            c0,
            transparent=bg,
        )
    ordered_panels = [panels[color_to_tgt[src]] for src in legend_order]
    out = _concat_panels(ordered_panels, FRAME_COL)
    assert out.shape[1] <= 30 and out.shape[0] <= 30, f"output too large: {out.shape}"
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.choice(GRID_ROWS_CHOICES)
    cols = GRID_COLS
    slots = rows * cols
    k = random.randint(*NUM_COLORS_RANGE)
    assert k <= slots

    empty_count = random.randint(0, min(MAX_EMPTY_SLOTS, slots - k))
    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    src_cols = random.sample(usable, k)
    remaining_for_tgt = [c for c in usable if c not in src_cols]
    if len(remaining_for_tgt) < k:
        return None
    tgt_cols = random.sample(remaining_for_tgt, k)
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    masks = _distinct_masks(src_cols)
    if len(masks) != k:
        return None

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors: dict[tuple[int, int], int] = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    legend_order = src_cols[:]
    random.shuffle(legend_order)
    expected = _build_expected_panels(rows, cols, bg, masks, color_to_tgt, legend_order, cell_colors)

    frame_h, frame_w = _frame_dims(rows, cols)
    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + rr * (SPRITE_N + 1)
        c0 = 1 + cc * (SPRITE_N + 1)
        paste(input_frame, np.full((SPRITE_N, SPRITE_N), src, dtype=int), r0, c0)

    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= 40
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.randrange(0, h - frame_h + 1)
    fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    legend_patch = _legend_strip(legend_order, color_to_tgt, bg)
    if _place_patch(grid, reserve, legend_patch, bg, transparent=False) is None:
        return None

    sprite_patches = [(_colored_sprite(masks[src], src, bg), True) for src in src_cols]
    random.shuffle(sprite_patches)
    for patch, transparent in sprite_patches:
        if _place_patch(grid, reserve, patch, bg, transparent=transparent) is None:
            return None

    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae D9 input")
