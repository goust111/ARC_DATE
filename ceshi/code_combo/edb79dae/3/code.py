# Combo variant of edb79dae (index 3)
# combo_spec: D2/1+D7/1
# Name: 分散图例+旋转匹配

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# a) Concrete behavior and trace:
#    - Base task:
#      placeholder color inside frame -> same-color sprite outside frame -> compact local legend maps source
#      color to target color -> recolor stamped sprite.
#    - Combo task:
#      placeholder color inside frame -> same-color sprite outside frame -> search all four outer edges for
#      split legend fragments -> pair top with bottom spans and left with right spans to recover complete
#      legend entries -> compare each recovered legend shape against the sprite up to rotation/flip -> use
#      the paired opposite-edge color strip as the recolor target.
#    - Small trace:
#      suppose source color 3 has an outside 4x4 sprite mask S. The top edge may show a common-ink bar plus
#      rot90(S); the bottom edge at the same column span shows color 8. Another color 6 may instead use a
#      left-edge fragment containing flipud(S2) and a right-edge strip color 2. The solver must first gather
#      both edge fragments, then mentally undo the transform to match each fragment to the correct sprite.
#    - Difference from the base rule:
#      no single local blob contains both source cue and target color, and the legend fragment no longer
#      exposes source color directly. Both information aggregation and geometric matching are required.
#
# b) Complexity calibration:
#    - Keep 2-5 source colors, as in the original spec, so each sample still has repeated support inside the
#      frame and a small enough mapping table for ARC-style inference.
#    - Force all sprite masks to be unique up to dihedral symmetry. This prevents two different source colors
#      from matching the same transformed legend fragment.
#    - Force every legend shape to be a non-identity rotation or reflection of its sprite mask, so exact
#      equality is insufficient.
#    - Force both vertical entries (top+bottom) and horizontal entries (left+right) to appear in each sample,
#      so legend information is visibly dispersed rather than accidentally clustered.
#
# c) Implicit-assumption audit of the original solver:
#    - The old bbox-overlap legend parser assumed each legend entry was locally complete and color keyed.
#      That is invalid here; legend parsing must instead read boundary spans and extract source fragments from
#      reserved edge bands.
#    - The old recolor lookup assumed source color directly indexed the legend. That also breaks: source
#      color only chooses the outside sprite, and shape identity under dihedral symmetry chooses the legend.
#    - Sprite extraction still assumes one bbox size per placeholder color and may need to merge multiple
#      disconnected components inside one sprite bbox. The generator preserves that.
#    - Boundary scanning assumes that every non-background object touching the outermost row/column belongs to
#      the legend. The generator enforces this by reserving the boundary before placing free sprites.
#
# d) Placement legality:
#    - The frame sits strictly inside an edge band of thickness n+1, leaving room for top/left source
#      fragments and bottom/right target strips.
#    - Top and left legend fragments are given one-cell gaps and kept away from corners, so their spans stay
#      distinct and do not overlap.
#    - External source-colored sprites are placed only in the remaining free background via
#      random_free_location, with padding and with the whole boundary already reserved.
#
# e) Visual observability:
#    - A human can see that source-shape clues live on the top/left edges while output colors live on the
#      opposite edges, so the mapping must be assembled globally rather than read locally.
#    - The legend source fragments use one shared ink color instead of the source colors themselves, making
#      shape the only stable cue and making the rotation/flip visibly meaningful.
#
# f) Visual cleanliness:
#    - The central frame remains the dominant object.
#    - Edge fragments occupy clean boundary bands, and external sprites float in the remaining interior
#      background instead of mixing with the legend.
#    - Each legend entry uses exactly one source fragment and one opposite-edge color strip, avoiding crowded
#      multi-layer symbols in one place.
#
# g) Change inferability:
#    - Target colors are still few, and each one is attached to a concrete opposite-edge strip.
#    - The recolor rule is therefore inferable from paired structure: source shape fragment on one edge,
#      target color strip on the opposite edge, repeated across a handful of entries.
#
# New solver assumptions introduced here and guaranteed by the generator:
#    - Every top edge run of legend ink has exactly one bottom strip with the same column span; every left
#      edge run has exactly one right strip with the same row span.
#    - Every legend source fragment occupies an (n+1)xn or nx(n+1) reserved patch whose inner nxn mask is a
#      non-identity dihedral transform of exactly one outside sprite mask.
#    - All outside source sprites have distinct dihedral signatures.
#    - Both vertical and horizontal legend entry families are present in every generated instance.


FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 5)
SPRITE_FILL_MIN = 0.45
SPRITE_FILL_MAX = 0.82
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (28, 40)
CANVAS_W_RANGE = (28, 40)
PATCH_PADDING = 1
GEN_RETRIES = 500
MASK_RETRIES = 160


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


def _dihedral_variants(mask: np.ndarray) -> list[tuple[np.ndarray, int, bool]]:
    base = []
    for k in range(4):
        rot = np.rot90(mask, k)
        base.append((rot.copy(), k, False))
        base.append((np.fliplr(rot).copy(), k, True))
    out = []
    seen = set()
    for arr, k, flip in base:
        key = arr.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((arr, int(k), bool(flip)))
    return out


def _mask_signature(mask: np.ndarray) -> bytes:
    return min(arr.tobytes() for arr, _k, _flip in _dihedral_variants(mask))


def _same_up_to_dihedral(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and any(np.array_equal(arr, b) for arr, _k, _flip in _dihedral_variants(a))


def _legend_match_transform(sprite_mask: np.ndarray, legend_mask: np.ndarray) -> tuple[int, bool] | None:
    if sprite_mask.shape != legend_mask.shape:
        return None
    for arr, k, flip in _dihedral_variants(sprite_mask):
        if np.array_equal(arr, legend_mask):
            return int(k), bool(flip)
    return None


def _is_rectangular_border_mask(mask: np.ndarray) -> bool:
    if mask.ndim != 2 or min(mask.shape) < 3:
        return False
    border = np.zeros(mask.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(mask[border]) and not np.all(mask[~border]))


def _collect_placeholders(
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
            placeholders.append((col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]
    return placeholders, phw, placeholder_colors


def _edge_runs(arr: np.ndarray, blocked: set[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    vals = [int(x) for x in np.asarray(arr).tolist()]
    i = 0
    while i < len(vals):
        col = vals[i]
        if col in blocked:
            i += 1
            continue
        j = i
        while j + 1 < len(vals) and vals[j + 1] == col:
            j += 1
        runs.append((i, j, col))
        i = j + 1
    return runs


def _extract_edge_legends(
    g: np.ndarray,
    bg: int,
    frame_col: int,
    placeholder_size: tuple[int, int],
) -> tuple[list[dict], np.ndarray, int]:
    n_h, n_w = placeholder_size
    assert n_h == n_w, "combo generator expects square placeholders"
    n = int(n_h)
    blocked = {bg, frame_col}

    top_runs = _edge_runs(g[0, :], blocked)
    bottom_runs = _edge_runs(g[-1, :], blocked)
    left_runs = _edge_runs(g[:, 0], blocked)
    right_runs = _edge_runs(g[:, -1], blocked)

    assert top_runs and left_runs, "combo requires source fragments on top and left edges"
    bottom_by_span = {(c0, c1): col for c0, c1, col in bottom_runs}
    right_by_span = {(r0, r1): col for r0, r1, col in right_runs}
    assert len(bottom_by_span) == len(bottom_runs), "duplicate bottom legend spans"
    assert len(right_by_span) == len(right_runs), "duplicate right legend spans"

    legend_entries = []
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_inks = set()

    for c0, c1, ink in top_runs:
        assert c1 - c0 + 1 == n, f"top legend width mismatch: {(c0, c1)} vs {n}"
        assert (c0, c1) in bottom_by_span, f"missing bottom mate for top span {(c0, c1)}"
        sub = g[0 : n + 1, c0 : c1 + 1]
        assert sub.shape == (n + 1, n), "top legend patch shape mismatch"
        assert np.all(sub[0, :] == ink), "top legend bar must be monochrome"
        shape_mask = sub[1:, :] == ink
        assert np.any(shape_mask) and np.any(~shape_mask), "top legend must contain both ink and bg"
        legend_entries.append(
            {
                "kind": "vertical",
                "shape_mask": shape_mask.copy(),
                "tgt": int(bottom_by_span[(c0, c1)]),
                "span": (int(c0), int(c1)),
            }
        )
        legend_inks.add(int(ink))
        legend_mask[0 : n + 1, c0 : c1 + 1] = True
        legend_mask[g.shape[0] - 1 : g.shape[0], c0 : c1 + 1] = True

    for r0, r1, ink in left_runs:
        assert r1 - r0 + 1 == n, f"left legend height mismatch: {(r0, r1)} vs {n}"
        assert (r0, r1) in right_by_span, f"missing right mate for left span {(r0, r1)}"
        sub = g[r0 : r1 + 1, 0 : n + 1]
        assert sub.shape == (n, n + 1), "left legend patch shape mismatch"
        assert np.all(sub[:, 0] == ink), "left legend bar must be monochrome"
        shape_mask = sub[:, 1:] == ink
        assert np.any(shape_mask) and np.any(~shape_mask), "left legend must contain both ink and bg"
        legend_entries.append(
            {
                "kind": "horizontal",
                "shape_mask": shape_mask.copy(),
                "tgt": int(right_by_span[(r0, r1)]),
                "span": (int(r0), int(r1)),
            }
        )
        legend_inks.add(int(ink))
        legend_mask[r0 : r1 + 1, 0 : n + 1] = True
        legend_mask[r0 : r1 + 1, g.shape[1] - 1 : g.shape[1]] = True

    assert len(legend_inks) == 1, f"expected one shared legend ink, found {sorted(legend_inks)}"
    return legend_entries, legend_mask, int(next(iter(legend_inks)))


def _extract_sprite_masks(
    g: np.ndarray,
    outside: np.ndarray,
    legend_mask: np.ndarray,
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    sprite_masks: dict[int, np.ndarray] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = []
        for pts in components_mask((g == src_col) & outside & (~legend_mask), connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            all_candidates.append(
                {
                    "pts": [(int(r), int(c)) for r, c in pts],
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        assert all_candidates, f"no source sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1, c["bbox"][3] - c["bbox"][1] + 1) == (want_h, want_w)
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        pts_set = set(base["pts"])
        for cand in all_candidates:
            if cand is base:
                continue
            r0, c0, r1, c1 = cand["bbox"]
            assert sr0 <= r0 and r1 <= sr1 and sc0 <= c0 and c1 <= sc1, (
                f"extra sprite component for color {src_col} lies outside the base bbox"
            )
            pts_set.update(cand["pts"])
        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        for r, c in pts_set:
            mask[r - sr0, c - sc0] = True
        sprite_masks[src_col] = mask
    return sprite_masks


def _extract_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    placeholders, phw, placeholder_colors = _collect_placeholders(crop, bg, frame_col)
    placeholder_sizes = set(phw.values())
    assert len(placeholder_sizes) == 1, "combo expects one common placeholder size"
    legend_entries, legend_mask, legend_ink = _extract_edge_legends(
        g, bg, frame_col, next(iter(placeholder_sizes))
    )
    assert len(legend_entries) == len(placeholder_colors), "legend entry count must match source color count"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    sprite_masks = _extract_sprite_masks(g, outside, legend_mask, placeholder_colors, phw)
    assert set(sprite_masks) == placeholder_colors

    return {
        "grid": g,
        "bg": int(bg),
        "frame_col": int(frame_col),
        "frame_bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
        "crop": crop,
        "placeholders": placeholders,
        "phw": phw,
        "placeholder_colors": placeholder_colors,
        "legend_entries": legend_entries,
        "legend_mask": legend_mask,
        "legend_ink": int(legend_ink),
        "sprite_masks": sprite_masks,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the framed placeholders with matched sprites, then recolor them through split edge legends.

    Input structure:
    - A single gray rectangular frame encloses solid source-color placeholder blocks on a background field.
    - Outside the frame there is one source-colored sprite template per placeholder color.
    - Legend entries are split across the grid boundary: top or left fragments show transformed shape clues in
      one shared ink color, while the paired bottom or right fragments show only the target recolor.

    Rule:
    - Find the frame and collect its placeholder blocks.
    - Reconstruct the distributed legend by scanning all four edges and pairing top with bottom spans and left
      with right spans.
    - For each placeholder color, take the same-color outside sprite mask, find the unique legend fragment
      whose ink shape matches that mask up to rotation or reflection, and recolor the stamped sprite to the
      paired opposite-edge target color.

    Key insight: Solving now requires two coupled steps: gather incomplete legend pieces from the whole outer
    boundary, then match each recovered shape clue to a sprite only after considering all dihedral transforms.
    """
    info = _extract_instance(np.asarray(grid, dtype=int))
    bg = int(info["bg"])
    crop = info["crop"]

    color_map: dict[int, int] = {}
    used_legend_ids: set[int] = set()
    for src_col, sprite_mask in info["sprite_masks"].items():
        hits = [
            idx
            for idx, entry in enumerate(info["legend_entries"])
            if _same_up_to_dihedral(sprite_mask, entry["shape_mask"])
        ]
        assert len(hits) == 1, f"expected one legend match for source color {src_col}, found {len(hits)}"
        idx = hits[0]
        assert idx not in used_legend_ids, "legend entry reused across source colors"
        color_map[src_col] = int(info["legend_entries"][idx]["tgt"])
        used_legend_ids.add(idx)
    assert set(color_map) == info["placeholder_colors"]

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in info["placeholders"]:
        sprite_mask = info["sprite_masks"][src_col]
        tgt_col = int(color_map[src_col])
        sh, sw = sprite_mask.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), "placeholder size does not match source sprite"
        for dr, dc in np.argwhere(sprite_mask):
            out[int(pr0 + dr), int(pc0 + dc)] = tgt_col

    return out


def _sprite_mask(n: int) -> np.ndarray:
    lo = max(int(np.ceil(SPRITE_FILL_MIN * n * n)), n + 1)
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

    if _is_rectangular_border_mask(mask):
        return _sprite_mask(n)
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _top_source_patch(mask: np.ndarray, ink: int, bg: int) -> np.ndarray:
    n = mask.shape[0]
    patch = np.full((n + 1, n), bg, dtype=int)
    patch[0, :] = int(ink)
    patch[1:, :][mask] = int(ink)
    return patch


def _left_source_patch(mask: np.ndarray, ink: int, bg: int) -> np.ndarray:
    n = mask.shape[0]
    patch = np.full((n, n + 1), bg, dtype=int)
    patch[:, 0] = int(ink)
    patch[:, 1:][mask] = int(ink)
    return patch


def _target_bar(length: int, color: int, *, vertical: bool) -> np.ndarray:
    if vertical:
        return np.full((1, length), int(color), dtype=int)
    return np.full((length, 1), int(color), dtype=int)


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1


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


def _slot_starts(limit: int, span: int, count: int, margin: int) -> list[int] | None:
    if count == 0:
        return []
    need = count * span + (count - 1)
    extra = limit - 2 * margin - need
    if extra < 0:
        return None
    offset = margin + random.randint(0, extra)
    return [offset + i * (span + 1) for i in range(count)]


def _place_patch(grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg)
    return True


def _choose_masks(src_cols: list[int], n: int) -> dict[int, np.ndarray] | None:
    masks: dict[int, np.ndarray] = {}
    seen = set()
    for src in src_cols:
        chosen = None
        for _ in range(MASK_RETRIES):
            mask = _sprite_mask(n)
            if len(_dihedral_variants(mask)) < 2:
                continue
            sig = _mask_signature(mask)
            if sig in seen:
                continue
            chosen = mask
            seen.add(sig)
            break
        if chosen is None:
            return None
        masks[src] = chosen
    return masks


def _choose_legend_variants(masks: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, bool]]] | None:
    legend_masks: dict[int, np.ndarray] = {}
    legend_tfms: dict[int, tuple[int, bool]] = {}
    for src, mask in masks.items():
        options = [(arr, k, flip) for arr, k, flip in _dihedral_variants(mask) if not np.array_equal(arr, mask)]
        if not options:
            return None
        arr, k, flip = random.choice(options)
        legend_masks[src] = arr.copy()
        legend_tfms[src] = (int(k), bool(flip))
    return legend_masks, legend_tfms


def _legend_entry_layout(
    src_cols: list[int], n: int, h: int, w: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if len(src_cols) < 2:
        return None
    corner_clear = n + 2
    vcount = random.randint(1, len(src_cols) - 1)
    hcount = len(src_cols) - vcount
    top_starts = _slot_starts(w, n, vcount, corner_clear)
    left_starts = _slot_starts(h, n, hcount, corner_clear)
    if top_starts is None or left_starts is None:
        return None
    order = src_cols[:]
    random.shuffle(order)
    vertical = [(order[i], top_starts[i]) for i in range(vcount)]
    horizontal = [(order[vcount + i], left_starts[i]) for i in range(hcount)]
    return vertical, horizontal


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    ink_choices = [c for c in usable if c not in src_cols]
    if not ink_choices:
        return None
    legend_ink = random.choice(ink_choices)
    tgt_pool = [c for c in usable if c != legend_ink]
    tgt_cols = [random.choice(tgt_pool) for _ in range(k)]
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    masks = _choose_masks(src_cols, n)
    if masks is None:
        return None
    legend_pick = _choose_legend_variants(masks)
    if legend_pick is None:
        return None
    legend_masks, legend_tfms = legend_pick

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    max_fh = min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK))
    max_fw = min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK))
    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    _empty_slots, filled_slots = _choose_empty_slots(rows, cols, empty_count)
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
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    edge_band = n + 1
    min_canvas_h = max(CANVAS_H_RANGE[0], frame_h + 2 * edge_band, 2 * (n + 2) + n)
    min_canvas_w = max(CANVAS_W_RANGE[0], frame_w + 2 * edge_band, 2 * (n + 2) + n)
    ext_area = k * (2 * n * n + 8 * n + 8)
    possible_dims = [
        (h, w)
        for h in range(min_canvas_h, CANVAS_H_RANGE[1])
        for w in range(min_canvas_w, CANVAS_W_RANGE[1])
        if h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)

    layout = _legend_entry_layout(src_cols, n, h, w)
    if layout is None:
        return None
    vertical_entries, horizontal_entries = layout
    assert vertical_entries and horizontal_entries, "layout must use both edge families"

    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)
    reserve[0, :] = FRAME_COL
    reserve[-1, :] = FRAME_COL
    reserve[:, 0] = FRAME_COL
    reserve[:, -1] = FRAME_COL

    fr_lo = edge_band
    fr_hi = h - frame_h - edge_band
    fc_lo = edge_band
    fc_hi = w - frame_w - edge_band
    if fr_lo > fr_hi or fc_lo > fc_hi:
        return None
    fr0 = random.randint(fr_lo, fr_hi)
    fc0 = random.randint(fc_lo, fc_hi)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    for src, c0 in vertical_entries:
        top_patch = _top_source_patch(legend_masks[src], legend_ink, bg)
        bottom_patch = _target_bar(n, color_to_tgt[src], vertical=True)
        paste(grid, top_patch, 0, c0, transparent=bg)
        paste(grid, bottom_patch, h - 1, c0)
        paste(reserve, np.full(top_patch.shape, FRAME_COL, dtype=int), 0, c0)
        paste(reserve, np.full(bottom_patch.shape, FRAME_COL, dtype=int), h - 1, c0)

    for src, r0 in horizontal_entries:
        left_patch = _left_source_patch(legend_masks[src], legend_ink, bg)
        right_patch = _target_bar(n, color_to_tgt[src], vertical=False)
        paste(grid, left_patch, r0, 0, transparent=bg)
        paste(grid, right_patch, r0, w - 1)
        paste(reserve, np.full(left_patch.shape, FRAME_COL, dtype=int), r0, 0)
        paste(reserve, np.full(right_patch.shape, FRAME_COL, dtype=int), r0, w - 1)

    sprite_patches = [_colored_sprite(masks[src], src, bg) for src in src_cols]
    random.shuffle(sprite_patches)
    for patch in sprite_patches:
        if not _place_patch(grid, reserve, patch, bg):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
    except Exception:
        return None

    metrics = variant_metrics(grid)
    if metrics["vertical_entries"] < 1 or metrics["horizontal_entries"] < 1:
        return None
    if metrics["non_identity_matches"] != k:
        return None
    if metrics["exact_match_failures"] != k:
        return None
    if set(metrics["matched_targets"]) != set(tgt_cols):
        return None
    assert all((rot != 0 or flip) for rot, flip in legend_tfms.values())
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae combo input")


def variant_metrics(grid: np.ndarray) -> dict[str, object]:
    info = _extract_instance(np.asarray(grid, dtype=int))
    used = set()
    non_identity = 0
    flip_matches = 0
    rotation_only = 0
    exact_match_failures = 0
    matched_targets = []

    for src_col, sprite_mask in info["sprite_masks"].items():
        hits = []
        for idx, entry in enumerate(info["legend_entries"]):
            tfm = _legend_match_transform(sprite_mask, entry["shape_mask"])
            if tfm is not None:
                hits.append((idx, tfm, int(entry["tgt"])))
        assert len(hits) == 1, f"ambiguous combo metrics match for color {src_col}"
        idx, (rot, flip), tgt = hits[0]
        assert idx not in used, "legend entry reused in metrics"
        used.add(idx)
        matched_targets.append(int(tgt))
        if rot != 0 or flip:
            non_identity += 1
        if flip:
            flip_matches += 1
        elif rot != 0:
            rotation_only += 1
        if not np.array_equal(sprite_mask, info["legend_entries"][idx]["shape_mask"]):
            exact_match_failures += 1

    vertical_entries = sum(1 for entry in info["legend_entries"] if entry["kind"] == "vertical")
    horizontal_entries = sum(1 for entry in info["legend_entries"] if entry["kind"] == "horizontal")
    return {
        "num_shapes": len(info["sprite_masks"]),
        "vertical_entries": int(vertical_entries),
        "horizontal_entries": int(horizontal_entries),
        "non_identity_matches": int(non_identity),
        "flip_matches": int(flip_matches),
        "rotation_only_matches": int(rotation_only),
        "exact_match_failures": int(exact_match_failures),
        "matched_targets": tuple(matched_targets),
        "legend_ink": int(info["legend_ink"]),
    }
