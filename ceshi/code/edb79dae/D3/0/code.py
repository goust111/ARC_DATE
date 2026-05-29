# Variant of edb79dae
# Dimension: D3 — 源对象：框架外精灵的属性 (src)
# Change: 灰色框架外的小型彩色精灵，每个精灵有独特的颜色和形状，是可移动的源对象... → 多色精灵

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D3 variant
# a) Concrete behavior and trace
#    - Inside the frame, placeholders remain solid NxN blocks whose color is the lookup key.
#    - Outside the frame, each source sprite is now a single 4-connected two-color patch with at least one bg hole.
#      The larger-area color is the primary color; it matches the placeholder color. The smaller-area color is a
#      visible accent that stays with the sprite when it is copied into the frame.
#    - The legend still maps placeholder/source color -> target color. The adapted recoloring step changes only the
#      primary-color cells to the legend target, while keeping the accent color unchanged. This is the minimal
#      compatibility adjustment needed to make the D3 change visible in the final output instead of being erased.
#    - Small trace:
#      original: placeholder 2 -> find outside monochrome-2 sprite -> paste shape -> recolor whole shape to 7.
#      variant:  placeholder 2 -> find outside sprite {2,8} where 2 occupies more cells than 8 -> paste full {2,8}
#      shape -> recolor only the 2-cells to 7, leaving the 8-cells in place.
# b) Complexity calibration
#    - The two-color sprite adds exactly one new inference: dominant-color analysis.
#    - Sprites are limited to exactly two colors, and the primary color is strictly larger than the accent color.
#    - Placeholder colors are capped at 2-4 distinct values so repeated examples exist and the extra accent colors
#      do not make the frame unreadably noisy.
# c) Implicit-assumption audit against the original solver
#    - Original sprite lookup assumed "all cells of a source sprite share the placeholder color" and searched
#      monochrome components by that color. That breaks immediately for two-color sprites, because the sprite would
#      be split into per-color components and the full pattern would be lost.
#    - Original output painting wrote a single target color into every sprite cell. That would silently destroy the
#      new accent color. The variant therefore pastes the whole sprite patch and recolors only cells equal to the
#      primary placeholder color.
#    - Original legend detection remains valid because legends are still compact no-bg two-color blocks, while
#      variant sprites always keep at least one bg hole in their bbox, so they cannot masquerade as legends.
# d) Placement legality
#    - The generator places the frame first, reserves its full bbox, then places legends and sprite bboxes outside
#      it with a 1-cell bg gap. Reserve uses full patch rectangles, so even transparent sprite holes cannot overlap.
#    - Every sprite bbox is exactly NxN, matching the placeholder bbox for its primary color.
# e) Visual observability
#    - The output always preserves a non-primary accent color inside each copied sprite, so the D3 change is visible
#      in both input and output. A solver that ignores dominant-color analysis cannot recover the full output.
# f) Visual cleanliness
#    - Each sprite is a single connected patch inside one NxN box, legends remain fixed 3x4 blocks, and bg gaps
#      separate all outside objects. This avoids clutter and keeps frame/interior/exterior roles legible.
# g) Color-change inferability
#    - The legend still provides the only source->target recoloring cue. Because accent colors are not part of the
#      legend mapping and stay unchanged, the human-visible rule becomes: "use the dominant source color for the map,
#      keep the accent paint as-is."
# New solver assumptions and generator guarantees
#    - Each outside non-legend sprite is exactly one 4-connected two-color object.
#    - Its primary color is unique by area and equals a placeholder color.
#    - Each placeholder color has exactly one size-matched sprite outside the frame.
#    - The generator enforces all four conditions directly.


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
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


def _extract_placeholders(
    crop: np.ndarray, bg: int, frame_col: int
) -> tuple[list[tuple[int, tuple[int, int, int, int]]], dict[int, tuple[int, int]], set[int]]:
    h, w = crop.shape
    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            assert np.all(crop[r0 : r1 + 1, c0 : c1 + 1] == col), f"placeholder color {col} is not a solid block"
            placeholders.append((col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]
    return placeholders, phw, placeholder_colors


def _collect_monochrome_components(
    g: np.ndarray, outside: np.ndarray, bg: int, frame_col: int
) -> list[dict]:
    comps = []
    cid = 0
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
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


def _detect_legends(
    g: np.ndarray,
    comps: list[dict],
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
) -> tuple[dict[int, int], np.ndarray]:
    color_map: dict[int, int] = {}
    legend_mask = np.zeros(g.shape, dtype=bool)

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
            if bg in sub_colors:
                continue
            if len(sub_colors) != 2:
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
            legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )
    return color_map, legend_mask


def _extract_sprite_patches(
    g: np.ndarray,
    outside: np.ndarray,
    legend_mask: np.ndarray,
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    sprite_mask = outside & ~legend_mask & (g != bg) & (g != frame_col)
    sprite_candidates = []
    for pts in components_mask(sprite_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = [int(c) for c in np.unique(patch) if int(c) not in (bg, frame_col)]
        assert len(colors) == 2, f"expected two colors in each sprite, found {colors}"
        counts = {col: int(np.sum(patch == col)) for col in colors}
        primary, secondary = sorted(colors, key=lambda col: counts[col], reverse=True)
        assert counts[primary] > counts[secondary], f"sprite colors must have a unique dominant color: {counts}"
        sprite_candidates.append(
            {
                "primary": int(primary),
                "secondary": int(secondary),
                "bbox": (r0, c0, r1, c1),
                "dims": (r1 - r0 + 1, c1 - c0 + 1),
                "patch": patch,
            }
        )

    sprite_patches: dict[int, np.ndarray] = {}
    for src_col in sorted(placeholder_colors):
        want_h, want_w = phw[src_col]
        matched = [
            c for c in sprite_candidates if c["primary"] == src_col and c["dims"] == (want_h, want_w)
        ]
        assert len(matched) == 1, f"expected one two-color sprite for placeholder color {src_col}, found {len(matched)}"
        sprite_patches[src_col] = matched[0]["patch"]

    assert set(sprite_patches) == placeholder_colors, "not all placeholder colors received a sprite"
    return sprite_patches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy two-color sprites into a frame by using their dominant source color as the lookup key.

    Input structure:
    - A background grid contains one rectangular gray frame whose interior holds solid placeholder blocks.
    - Outside the frame there is one two-color sprite per placeholder color, plus compact two-color legend glyphs
      that encode placeholder-color -> target-color mappings.

    Rule:
    - Read the placeholder colors and legend mappings as in the original task family.
    - For each outside sprite, determine which of its two colors is the primary one by area; that dominant color
      identifies the placeholder color it should match.
    - Paste the entire two-color sprite into each matching placeholder position, then recolor only the primary-color
      cells to the legend target while leaving the minority accent color unchanged.

    Key insight: The legend still controls recoloring, but shape retrieval now depends on isolating a multicolor
    sprite and inferring which of its two colors is the true source label.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    placeholders, phw, placeholder_colors = _extract_placeholders(crop, bg, frame_col)

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps = _collect_monochrome_components(g, outside, bg, frame_col)
    color_map, legend_mask = _detect_legends(g, comps, bg, frame_col, placeholder_colors)
    sprite_patches = _extract_sprite_patches(g, outside, legend_mask, bg, frame_col, placeholder_colors, phw)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        patch = sprite_patches[src_col].copy()
        patch[patch == src_col] = int(color_map[src_col])
        sh, sw = patch.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        paste(out, patch, pr0, pc0, transparent=bg)

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 4)          # slightly reduced to keep two-color outputs readable
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    target = random.randint(lo, hi)

    mask = np.zeros((n, n), dtype=bool)
    c = random.randrange(n)
    for r in range(n):
        c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True
    mask[random.randrange(n), :] = True

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
    return mask


def _secondary_mask(mask: np.ndarray) -> np.ndarray:
    filled = list(zip(*np.where(mask)))
    area = len(filled)
    assert area >= 5, f"sprite mask too small for two colors: {area}"

    max_secondary = max(2, min((area - 1) // 2, int(np.floor(0.4 * area))))
    target = random.randint(2, max_secondary)
    chosen = {random.choice(filled)}

    while len(chosen) < target:
        frontier = []
        for r, c in chosen:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in chosen:
                    frontier.append((rr, cc))
        if not frontier:
            frontier = [cell for cell in filled if cell not in chosen]
        chosen.add(random.choice(frontier))

    out = np.zeros_like(mask, dtype=bool)
    for r, c in chosen:
        out[r, c] = True
    return out


def _multicolor_sprite(mask: np.ndarray, primary: int, accent: int, bg: int) -> np.ndarray:
    secondary = _secondary_mask(mask)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(primary)
    out[secondary] = int(accent)
    assert int(np.sum(out == primary)) > int(np.sum(out == accent)), "primary color must dominate"
    return out


def _recolor_primary(patch: np.ndarray, primary: int, target: int) -> np.ndarray:
    out = patch.copy()
    out[out == primary] = int(target)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    total = random.randint(*TOTAL_BLOCKS_RANGE)
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + 4
                base_fw = cw + 4
                slots = rows * cols
                empty = slots - total
                if base_fh <= MAX_FRAME_DIM and base_fw <= MAX_FRAME_DIM and 0 <= empty <= MAX_EMPTY_SLOTS:
                    options.append((n, rows, cols, empty, total))
    return random.choice(options) if options else None


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
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
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    accent_pool = [c for c in usable if c not in src_cols]
    accent_cols = {src: random.choice(accent_pool) for src in src_cols}
    tgt_cols = {
        src: random.choice([c for c in usable if c not in (src, accent_cols[src])])
        for src in src_cols
    }

    masks = {src: _sprite_mask(n) for src in src_cols}
    sprites = {
        src: _multicolor_sprite(masks[src], src, accent_cols[src], bg)
        for src in src_cols
    }

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    frame_h = random.randint(min_fh, min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK)))
    frame_w = random.randint(min_fw, min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK)))
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        paste(output_frame, _recolor_primary(sprites[src], src, tgt_cols[src]), r0, c0, transparent=bg)

    ext_area = k * (12 + n * n + 10)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.choice([0, h - frame_h])
    fc0 = random.choice([0, w - frame_w])
    if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, h - frame_h + 1)
    if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src in src_cols:
        patches.append((sprites[src], True))
        patches.append((_legend(src, tgt_cols[src]), False))
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
    raise RuntimeError("failed to generate valid edb79dae input")
