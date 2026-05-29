# Combo variant of edb79dae (index 9)
# combo_spec: D7/2+D13/0
# Name: 数量编码图例+干扰条目

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace:
#    - Keep the original gray rectangular frame and the regular grid of solid placeholder
#      blocks inside it. Keep one external sprite template per placeholder color.
#    - Replace the original two-color overlapping legend with a top-left strip of compact
#      single-color count codes. Each code is a 2x3 nested stencil whose colored area is the
#      lookup key; its color is the recolor target.
#    - Add extra legend entries whose counts are not used by any real sprite.
#    - Example: suppose a blue 4x4 sprite has exactly 5 colored pixels and a red 4x4 sprite
#      has exactly 3 colored pixels. The legend may show orange=5, yellow=3, plus teal=1 as
#      a distractor. Old logic would pair blue/red by source-color or shape legends. The new
#      logic must instead count the blue sprite's 5 pixels, choose orange from the count-5
#      legend entry, count the red sprite's 3 pixels, choose yellow from the count-3 entry,
#      and ignore the teal count-1 entry because no sprite has area 1.
# b) Complexity calibration:
#    - Real sprite counts are unique and chosen from 2..6, keeping the count vocabulary
#      small enough for people to compare by eye while still forcing counting.
#    - Distractor counts are chosen from the remaining 1..6 values, so they look exactly
#      like legitimate legend entries and cannot be rejected by color or shape style alone.
#    - Placeholder colors repeat inside the frame, so each valid count/color rule has
#      multiple placements supporting inference from a few examples.
# c) Implicit-assumption audit of the original solver:
#    - Original legend parsing assumed an overlapping two-color glyph. That no longer exists;
#      solver must read separate monochrome components on the top band.
#    - Original recoloring keyed legend lookup by placeholder color. That is invalid now:
#      recoloring is keyed only by sprite pixel count.
#    - Original solver treated every legend mapping as meaningful. With distractors present,
#      that assumption breaks; solver must recover sprite areas first and then keep only
#      legend counts that are actually used by some sprite.
#    - Sparse counted sprites may be disconnected, so sprite extraction must union all
#      non-legend components of the same placeholder color and verify their union bbox.
# d) Placement legality:
#    - Generator reserves a dedicated top band for the legend, so every component touching
#      row 0 is guaranteed to be a legend entry and never overlaps the frame or sprites.
#    - External patches are placed with `random_free_location` plus padding, keeping legend,
#      frame, and sprites separated by background gaps.
#    - Sparse sprite masks always include opposite corners, guaranteeing that the sprite union
#      bbox matches the placeholder NxN block even when the mask is disconnected.
# e) Visual observability:
#    - Humans can see the change from shape legend to count legend because the legend entries
#      are tiny nested dot groups, not sprite silhouettes or color pairs.
#    - The distractor effect is visible because the legend has more entries than there are
#      distinct sprite areas in use; one must explicitly count real sprites to decide which
#      legend entries matter.
# f) Visual cleanliness:
#    - The top band groups all legend entries together, while free sprites remain elsewhere.
#      This reduces clutter compared with mixing multiple legend styles around the frame.
#    - All objects keep a 1-cell background gap, so the frame, legend strip, and sprites stay
#      visually separable.
# g) Color-change inferability:
#    - Output colors still come directly from the legend entry colors, but the condition for
#      choosing a color is the sprite area rather than the placeholder color.
#    - Because counts come from only six possible nested codes and each real count has repeated
#      evidence inside the frame, 2-4 I/O examples are enough to infer both the counting rule
#      and the need to ignore unused legend counts.
# h) Fusion metric used after implementation:
#    - For each sample, record `legend_total_count - used_sprite_count` and require it to be
#      at least 1. This quantifies that distractor entries are really present.
#    - Also record the real sprite areas. They must be a strict subset of the legend counts,
#      proving the legend is count-based and filtered, not fully consumed.
# i) New solver assumptions checked against generation:
#    - Every placeholder color has exactly one external sprite union with the same bbox size
#      as its placeholders.
#    - Every real sprite area is unique and has exactly one matching legend count.
#    - Every distractor legend count differs from every real sprite area.
#    - Only legend entries occupy the reserved top band, so the top-row component test is safe.


FRAME_COL = 5
SPRITE_N_RANGE = (3, 5)             # N in {3, 4}
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 5)
SPRITE_AREA_CHOICES = (2, 3, 4, 5, 6)
LEGEND_COUNT_CHOICES = (1, 2, 3, 4, 5, 6)
DISTRACTOR_RANGE = (1, 2)
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
PATCH_PADDING = 1
GEN_RETRIES = 400

LEGEND_H = 2
LEGEND_ENTRY_W = 3
LEGEND_GAP = 1
TOP_BAND_H = LEGEND_H + 1
LEGEND_ORDER = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))


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


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1


def _normalized_coords(pts: set[tuple[int, int]] | list[tuple[int, int]], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))


def _collect_outside_components(
    g: np.ndarray, outside: np.ndarray, *, bg: int, frame_col: int
) -> list[dict]:
    comps = []
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


def _extract_legend_entries(comps: list[dict]) -> list[dict]:
    """Read top-band count entries and return them sorted by x-position."""
    entries = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 != 0:
            continue
        h, w = _bbox_hw(comp["bbox"])
        if h > LEGEND_H or w > LEGEND_ENTRY_W:
            continue
        coords = _normalized_coords(comp["pts"], r0, c0)
        if coords != LEGEND_ORDER[: len(coords)]:
            continue
        entries.append(
            {
                "x": int(c0),
                "color": int(comp["col"]),
                "count": int(comp["area"]),
                "id": int(comp["id"]),
            }
        )
    entries.sort(key=lambda entry: entry["x"])
    assert entries, "count legend must occupy the top row"
    return entries


def _analyze_scene(grid: np.ndarray) -> dict:
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
        dims = [_bbox_hw(b) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps = _collect_outside_components(g, outside, bg=bg, frame_col=frame_col)

    legend_entries = _extract_legend_entries(comps)
    legend_ids = {entry["id"] for entry in legend_entries}

    legend_map: dict[int, int] = {}
    for entry in legend_entries:
        count = entry["count"]
        assert count not in legend_map, f"duplicate legend count {count}"
        legend_map[count] = int(entry["color"])

    sprite_data: dict[int, tuple[tuple[tuple[int, int], ...], int, int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no sprite found for placeholder color {src_col}"

        pts_set = set()
        r0 = min(c["bbox"][0] for c in all_candidates)
        c0 = min(c["bbox"][1] for c in all_candidates)
        r1 = max(c["bbox"][2] for c in all_candidates)
        c1 = max(c["bbox"][3] for c in all_candidates)
        for comp in all_candidates:
            pts_set.update(comp["pts"])

        want_h, want_w = phw[src_col]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (want_h, want_w), (
            f"sprite union bbox does not match placeholder size for color {src_col}"
        )
        coords = _normalized_coords(pts_set, r0, c0)
        sprite_data[src_col] = (coords, want_h, want_w, len(coords))

    real_counts = [area for _coords, _sh, _sw, area in sprite_data.values()]
    assert len(set(real_counts)) == len(real_counts), "sprite counts must be unique across placeholder colors"

    real_count_set = set(real_counts)
    assert real_count_set.issubset(set(legend_map)), (
        f"legend counts {sorted(legend_map)} do not cover sprite counts {sorted(real_count_set)}"
    )

    color_map = {src_col: int(legend_map[area]) for src_col, (_coords, _sh, _sw, area) in sprite_data.items()}
    assert set(color_map) == placeholder_colors, "missing count-to-color legend mapping"

    return {
        "bg": bg,
        "frame_col": frame_col,
        "crop": crop,
        "placeholders": placeholders,
        "phw": phw,
        "sprite_data": sprite_data,
        "color_map": color_map,
        "legend_entries": legend_entries,
        "legend_counts": sorted(legend_map),
        "real_counts": sorted(real_count_set),
        "distractor_counts": sorted(set(legend_map) - real_count_set),
    }


def fusion_metrics(grid: np.ndarray) -> dict:
    """Expose combo-specific observability metrics for validation scripts."""
    info = _analyze_scene(grid)
    return {
        "legend_entry_count": len(info["legend_entries"]),
        "used_count_count": len(info["real_counts"]),
        "distractor_count": len(info["distractor_counts"]),
        "real_counts": tuple(info["real_counts"]),
        "distractor_counts": tuple(info["distractor_counts"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace framed placeholders with counted sprites and recolor them via a filtered count legend.

    Input structure:
    - A gray rectangular frame encloses solid-color placeholder blocks on a background field.
    - Outside the frame there is one source-colored sprite template for each placeholder color.
    - The top-left band contains compact monochrome legend entries whose colored pixel counts encode lookup keys,
      and some of those legend counts are distractors unused by any real sprite.

    Rule:
    - Recover each external sprite by placeholder color and measure how many colored pixels it contains.
    - Read the top-band legend as candidate mappings from pixel count to output color.
    - Keep only the legend entries whose counts match some recovered sprite area.
    - Replace every placeholder block with its sprite shape and recolor it using the output color of the matching count entry.

    Key insight: The legend is no longer about matching shapes or source colors; the only valid key is sprite area, and extra count entries must be rejected by counting the real sprites.
    """
    info = _analyze_scene(grid)
    crop = info["crop"]
    bg = info["bg"]
    placeholders = info["placeholders"]
    sprite_data = info["sprite_data"]
    color_map = info["color_map"]

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_data and src_col in color_map, (
            f"placeholder color {src_col} missing sprite or count mapping"
        )
        coords, sh, sw, _area = sprite_data[src_col]
        tgt_col = int(color_map[src_col])
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col

    return out


# --------------- generator (o5x) ---------------


def _legend_patch(entries: list[tuple[int, int]], bg: int) -> np.ndarray:
    """Build the top-left nested legend from (pixel_count, target_color) pairs."""
    width = len(entries) * LEGEND_ENTRY_W + max(0, len(entries) - 1) * LEGEND_GAP
    patch = np.full((LEGEND_H, width), bg, dtype=int)
    for idx, (pixel_count, color) in enumerate(entries):
        c0 = idx * (LEGEND_ENTRY_W + LEGEND_GAP)
        for rr, cc in LEGEND_ORDER[:pixel_count]:
            patch[rr, c0 + cc] = int(color)
    return patch


def _sprite_mask(n: int, area: int) -> np.ndarray:
    """Generate an NxN sparse mask with exact area and full NxN union bbox."""
    assert 2 <= area <= min(len(LEGEND_ORDER), n * n), f"unsupported area {area} for n={n}"
    mask = np.zeros((n, n), dtype=bool)

    corner_pairs = [
        ((0, 0), (n - 1, n - 1)),
        ((0, n - 1), (n - 1, 0)),
    ]
    a, b = random.choice(corner_pairs)
    chosen = {a, b}
    candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) not in chosen]
    random.shuffle(candidates)
    while len(chosen) < area:
        chosen.add(candidates.pop())

    for r, c in chosen:
        mask[r, c] = True
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
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
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int, *, transparent: bool
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _sample_target_colors(src_cols: list[int], usable: list[int]) -> list[int] | None:
    for _ in range(200):
        tgt_cols = random.sample(usable, len(src_cols))
        if all(tgt != src for src, tgt in zip(src_cols, tgt_cols)):
            return tgt_cols
    return None


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    tgt_cols = _sample_target_colors(src_cols, usable)
    if tgt_cols is None:
        return None

    real_counts = random.sample(SPRITE_AREA_CHOICES, k)
    available_distractor_counts = [count for count in LEGEND_COUNT_CHOICES if count not in real_counts]
    max_distractors = min(DISTRACTOR_RANGE[1], len(available_distractor_counts))
    if max_distractors < DISTRACTOR_RANGE[0]:
        return None
    distractor_count = random.randint(DISTRACTOR_RANGE[0], max_distractors)
    distractor_counts = random.sample(available_distractor_counts, distractor_count)

    remaining_legend_colors = [c for c in usable if c not in tgt_cols]
    if len(remaining_legend_colors) < distractor_count:
        return None
    distractor_colors = random.sample(remaining_legend_colors, distractor_count)

    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n, area) for src, area in zip(src_cols, real_counts)}

    legend_items = list(zip(real_counts, tgt_cols)) + list(zip(distractor_counts, distractor_colors))
    random.shuffle(legend_items)
    legend_patch = _legend_patch(legend_items, bg)
    legend_w = legend_patch.shape[1]

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
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    sprite_budget = k * (n * n + 6)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h + TOP_BAND_H
        and w >= max(frame_w, legend_w)
        and h * w - frame_h * frame_w - TOP_BAND_H * w >= sprite_budget
    ]
    if not possible_dims:
        return None

    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)
    reserve[:TOP_BAND_H, :] = FRAME_COL

    paste(grid, legend_patch, 0, 0, transparent=bg)

    frame_blocker = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    try:
        fr0, fc0 = random_free_location(reserve, frame_blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return None
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, frame_blocker, fr0, fc0)

    patches = [_colored_sprite(masks[src], src, bg) for src in src_cols]
    random.shuffle(patches)
    patches.sort(key=lambda patch: patch.size, reverse=True)
    for patch in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=True):
            return None

    try:
        info = _analyze_scene(grid)
        if len(info["distractor_counts"]) < 1:
            return None
        if sorted(real_counts) != info["real_counts"]:
            return None
        if sorted(distractor_counts) != info["distractor_counts"]:
            return None
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
    raise RuntimeError("failed to generate valid edb79dae combo input")
