# Combo variant of 8698868d (index 21)
# combo_spec: D9/1+D16/0
# Name: 图例消歧重复洞数

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - Preserved scaffold:
#      input still contains one tight grid of large square frames plus the same number of smaller square sprites.
#      Output is still only the compact frame grid with a sprite nested inside each frame.
#    - New fused rule:
#      hole count remains the first-stage key, but counts may repeat. A 2-row legend on the top edge contains
#      only the ambiguous repeated-count colors. Each legend entry is a 2x2 vertical band:
#        top row  = sprite color
#        bottom row = frame color
#      Example:
#        frame holes by grid order  = [1, 1, 2, 3]
#        sprite holes by input pos  = [1, 3, 1, 2]
#        top legend says blue -> red and green -> yellow
#      Then:
#        - counts 2 and 3 are unique, so they still pair by hole count alone;
#        - count 1 has two candidate sprites and two candidate frames, so the legend chooses which 1-hole sprite
#          goes into the red frame and which 1-hole sprite goes into the yellow frame.
#      Difference from original:
#        original solver could stop at `same hole count`; here that is insufficient once a bucket has size > 1.
#      Difference from pure legend mapping:
#        legend is intentionally partial; it does not mention unique-count pairs, so color lookup alone is also
#        insufficient. The reasoning path is genuinely count bucket -> legend disambiguation inside that bucket.
# b) Complexity calibration
#    - Object count is narrowed to exactly 4 (a 2x2 frame grid). This keeps the legend short and lets frame colors
#      and sprite colors be disjoint, which makes the color coding readable.
#    - Every sample uses the multiset [k, k, k+1, k+2] up to permutation, with k > 0 for the repeated value.
#      So every sample has:
#        - one repeated bucket that forces legend use;
#        - two unique buckets that still force hole counting.
#    - The generator rejects degenerate layouts unless:
#        - the combo output differs from a D9-only baseline (count + reading-order tiebreak);
#        - the combo output also differs from a legend-only residual-position baseline.
# c) Hidden assumption audit
#    - Original nested matching loop assumed every hole count appears once. That assumption breaks immediately with
#      duplicates, so matching must be bucketed explicitly.
#    - Original D16 legend variant allowed a full global color map. That would make hole counts irrelevant here, so
#      this combo uses a partial legend that covers only repeated buckets.
#    - Square detection by size still works because legend pieces are 1x2 monochromatic strips, too small to be
#      mistaken for sprites or frames.
#    - Frame color is still recoverable from the border majority color, which the generator preserves.
# d) Placement legality
#    - The frame grid is placed as one contiguous 2x2 block below the reserved legend rows.
#    - A temporary placeholder occupies the future legend band while sprites are placed, so no sprite can intrude
#      into the top-edge legend region.
#    - Sprites are placed with `random_free_location(..., padding=1)`, so they stay separated from one another,
#      from the frame grid, and from the legend band.
# e) Visual observability
#    - The repeated bucket is visible because two outside sprites have the same number of holes and two frames also
#      share that count, creating an obvious ambiguity.
#    - The legend is also visually obvious: a short 2-row color band on the top edge, and its colors are reused by
#      the repeated sprites and repeated frame borders.
#    - Unique-count pairs remain unlisted, so examples show that the legend is not the whole story.
# f) Visual cleanliness
#    - The legend occupies only 2 rows and is separated by background gaps.
#    - The frame grid stays compact; sprites remain outside it; there is no multi-layer overlap in the same region.
# g) Change inferability
#    - Colors are limited to 8 non-background colors total: 4 frame colors and 4 sprite colors, all distinct.
#      This is small enough for humans to track across 2-4 examples.
#    - Because the legend covers only the ambiguous repeated bucket, solvers can infer:
#        unique count  -> direct count match
#        repeated count -> consult the color legend
#    - No invisible ordering rule depends on numeric color ids; all tie-breaking uses visible geometry or legend
#      entries.
# h) New solver assumptions after modification
#    - `_read_legend_map()` assumes the only colored cells in rows 0-1 belong to the legend. The generator enforces
#      this by reserving those rows before sprite placement.
#    - `_match_by_combo_rule()` assumes repeated buckets are fully covered by legend entries and singleton buckets do
#      not need them. The generator always constructs exactly one repeated bucket and emits legend entries for both
#      members of that bucket.
#    - Sprite colors are unique and frame colors are unique. The generator samples disjoint color sets to guarantee
#      that `color -> object` lookup is unambiguous.


def _read_legend_map(grid: np.ndarray, bg: int) -> dict[int, int]:
    """Read the 2-row top-edge legend as sprite_color -> frame_color."""
    assert grid.shape[0] >= 2
    band = (grid[:2, :] != int(bg)).any(axis=0)
    assert bool(band.any())

    mapping: dict[int, int] = {}
    c = 0
    while c < grid.shape[1]:
        if not bool(band[int(c)]):
            c += 1
            continue
        c0 = int(c)
        while c + 1 < grid.shape[1] and bool(band[int(c + 1)]):
            c += 1
        c1 = int(c)
        sub = grid[:2, c0 : c1 + 1]
        assert sub.shape[0] == 2
        assert bool((sub[0, :] != int(bg)).all())
        assert bool((sub[1, :] != int(bg)).all())
        top_vals = {int(v) for v in sub[0, :].tolist()}
        bottom_vals = {int(v) for v in sub[1, :].tolist()}
        assert len(top_vals) == 1 and len(bottom_vals) == 1
        sprite_color = next(iter(top_vals))
        frame_color = next(iter(bottom_vals))
        assert sprite_color not in mapping
        mapping[int(sprite_color)] = int(frame_color)
        c += 1

    assert mapping
    return mapping


def _match_by_combo_rule(frames: list[dict], sprites: list[dict], legend_map: dict[int, int]) -> dict[int, dict]:
    """Pair singleton hole buckets directly, and use the legend inside repeated buckets."""
    frame_to_sprite = {int(frame_color): int(sprite_color) for sprite_color, frame_color in legend_map.items()}

    frames_by_holes: dict[int, list[dict]] = {}
    sprites_by_holes: dict[int, list[dict]] = {}
    for frame in frames:
        frames_by_holes.setdefault(int(frame["holes"]), []).append(frame)
    for sprite in sprites:
        sprites_by_holes.setdefault(int(sprite["holes"]), []).append(sprite)

    matches: dict[int, dict] = {}
    for hole_count, fgroup in frames_by_holes.items():
        sgroup = sprites_by_holes.get(int(hole_count), [])
        assert len(fgroup) == len(sgroup) and len(fgroup) >= 1
        if len(fgroup) == 1:
            matches[int(fgroup[0]["id"])] = sgroup[0]
            continue

        sprites_by_color = {int(sprite["color"]): sprite for sprite in sgroup}
        assert len(sprites_by_color) == len(sgroup)
        for frame in fgroup:
            frame_color = int(frame["color"])
            assert frame_color in frame_to_sprite
            sprite_color = int(frame_to_sprite[frame_color])
            sprite = sprites_by_color.get(int(sprite_color))
            assert sprite is not None
            matches[int(frame["id"])] = sprite

    return matches


def _build_output_from_matches(
    frames: list[dict],
    matches: dict[int, dict],
    frame_size: int,
    rows_grid: int,
    cols_grid: int,
    bg: int,
) -> np.ndarray:
    out = np.full((rows_grid * frame_size, cols_grid * frame_size), int(bg), dtype=int)
    for frame in frames:
        rr = int(frame["out_r"])
        cc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)

        sprite = matches.get(int(frame["id"]))
        assert sprite is not None
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)
        sh, sw = content.shape
        out[rr + 1 : rr + 1 + sh, cc + 1 : cc + 1 + sw] = content
    return out


def _build_count_only_matches(frames: list[dict], sprites: list[dict]) -> dict[int, dict]:
    """A D9-only baseline: equal holes, reading-order tie-break inside duplicate buckets."""
    frames_by_holes: dict[int, list[dict]] = {}
    sprites_by_holes: dict[int, list[dict]] = {}
    for frame in frames:
        frames_by_holes.setdefault(int(frame["holes"]), []).append(frame)
    for sprite in sprites:
        sprites_by_holes.setdefault(int(sprite["holes"]), []).append(sprite)

    matches: dict[int, dict] = {}
    for hole_count, fgroup in frames_by_holes.items():
        sgroup = sprites_by_holes[int(hole_count)]
        ordered_frames = sorted(fgroup, key=lambda item: (int(item["out_r"]), int(item["out_c"])))
        ordered_sprites = sorted(sgroup, key=lambda item: (int(item["in_r"]), int(item["in_c"])))
        for frame, sprite in zip(ordered_frames, ordered_sprites):
            matches[int(frame["id"])] = sprite
    return matches


def _build_residual_legend_only_matches(frames: list[dict], sprites: list[dict], legend_map: dict[int, int]) -> dict[int, dict]:
    """
    A legend-heavy ablation used only for generator checks:
    use the legend where available, then pair leftover items by position without hole counting.
    """
    frame_to_sprite = {int(frame_color): int(sprite_color) for sprite_color, frame_color in legend_map.items()}
    sprites_by_color = {int(sprite["color"]): sprite for sprite in sprites}
    matches: dict[int, dict] = {}
    used_sprite_ids: set[int] = set()

    for frame in frames:
        frame_color = int(frame["color"])
        sprite_color = frame_to_sprite.get(int(frame_color))
        if sprite_color is None:
            continue
        sprite = sprites_by_color.get(int(sprite_color))
        assert sprite is not None
        matches[int(frame["id"])] = sprite
        used_sprite_ids.add(int(sprite["id"]))

    leftover_frames = sorted(
        [frame for frame in frames if int(frame["id"]) not in matches],
        key=lambda item: (int(item["out_r"]), int(item["out_c"])),
    )
    leftover_sprites = sorted(
        [sprite for sprite in sprites if int(sprite["id"]) not in used_sprite_ids],
        key=lambda item: (int(item["in_r"]), int(item["in_c"])),
    )
    assert len(leftover_frames) == len(leftover_sprites)
    for frame, sprite in zip(leftover_frames, leftover_sprites):
        matches[int(frame["id"])] = sprite
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use connected-hole counts to form candidate buckets, then use the edge legend to break duplicate-count ties.

    Input structure:
    - Background is azure (8). A compact grid of large square frames sits below a 2-row legend band on the top edge.
    - Smaller square sprites are scattered elsewhere on the canvas. Several frames and sprites may share the same
      number of 4-connected holes.
    - Each legend entry is a tiny vertical color band: the top row shows a sprite color and the bottom row shows the
      frame color that it should match when that hole count is ambiguous.

    Rule:
    - Read the legend as sprite_color -> frame_color links.
    - Detect the large frame grid and the loose smaller sprites, and count hole components in each object.
    - If a hole count appears only once, match that frame and sprite directly by hole count.
    - If a hole count appears multiple times, stay inside that hole-count bucket and use the legend colors to decide
      which sprite goes into which frame.
    - Place each matched sprite inside the frame border and recolor the sprite's hole pixels to the frame color.

    Key insight:
    Hole count still chooses the right bucket; the legend does not replace it, but resolves only the duplicate-count
    buckets where count alone stops being unique.
    """
    g = np.array(grid, dtype=int)
    bg = 8

    legend_map = _read_legend_map(g, bg)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
            square_regions.append((int(h), comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    assert len(size_list) >= 2
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1])

    frames_raw = sizes.get(int(frame_size), [])
    sprites_raw = sizes.get(int(sprite_size), [])
    assert frames_raw and sprites_raw

    frame_positions = [(int(frame["bbox"][0]), int(frame["bbox"][1])) for frame in frames_raw]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    rows_grid = int((max_r - min_r) // frame_size + 1)
    cols_grid = int((max_c - min_c) // frame_size + 1)

    frames: list[dict] = []
    for ri in range(rows_grid):
        for ci in range(cols_grid):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            inner = region[1:-1, 1:-1]
            hole_mask = inner != int(frame_color)
            _lab, num_holes = label_mask(hole_mask, connectivity=4)
            frames.append(
                {
                    "id": len(frames),
                    "color": int(frame_color),
                    "holes": int(num_holes),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "id": len(sprites),
                "color": int(sprite_color),
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
                "in_r": int(b[0]),
                "in_c": int(b[1]),
            }
        )

    matches = _match_by_combo_rule(frames, sprites, legend_map)
    return _build_output_from_matches(frames, matches, frame_size, rows_grid, cols_grid, bg)


# --------------- generator ---------------

BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 800

ROWS_GRID = 2
COLS_GRID = 2
N = 4

LEGEND_H = 2
LEGEND_CELL_W = 2
LEGEND_GAP = 1
LEGEND_RESERVED_ROWS = 3
LEGEND_TMP_COLOR = 0


def _legend_width(count: int) -> int:
    return int(count * LEGEND_CELL_W + max(0, count - 1) * LEGEND_GAP)


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    inner_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(SEED_RETRIES):
        random.shuffle(inner_positions)
        seeds: list[tuple[int, int]] = []
        for r, c in inner_positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite

    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_idx, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_holes)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_holes:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for other_idx, other in enumerate(comps)
                        if other_idx != comp_idx
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_holes.add(cell)
            sprite[cell[0], cell[1]] = BG

    _, num_holes = label_mask(sprite != int(sprite_color), connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame

    inner_cells = (frame_size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _, num_holes = label_mask(frame != int(frame_color), connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return frame


def _generate_hole_multiset(frame_size: int) -> list[int] | None:
    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    if max_hole < 3:
        return None

    max_start = min(HOLE_START_MAX, max_hole - 3)
    if max_start < 0:
        return None
    start = random.randint(0, max_start)
    base = [int(start + offset) for offset in range(3)]
    positive_repeat = [value for value in base if value > 0]
    repeat_value = int(random.choice(positive_repeat if positive_repeat else base[1:]))
    counts = base + [repeat_value]
    assert len(counts) == N
    return counts


def _make_legend(entries: list[tuple[int, int]]) -> np.ndarray:
    width = _legend_width(len(entries))
    legend = np.full((LEGEND_H, width), BG, dtype=int)
    for idx, (sprite_color, frame_color) in enumerate(entries):
        c0 = int(idx * (LEGEND_CELL_W + LEGEND_GAP))
        legend[0, c0 : c0 + LEGEND_CELL_W] = int(sprite_color)
        legend[1, c0 : c0 + LEGEND_CELL_W] = int(frame_color)
    return legend


def _combo_metrics(frames: list[dict], sprites: list[dict], legend_map: dict[int, int], frame_size: int) -> dict[str, int]:
    combo_matches = _match_by_combo_rule(frames, sprites, legend_map)
    combo_out = _build_output_from_matches(frames, combo_matches, frame_size, ROWS_GRID, COLS_GRID, BG)

    count_only_matches = _build_count_only_matches(frames, sprites)
    count_only_out = _build_output_from_matches(frames, count_only_matches, frame_size, ROWS_GRID, COLS_GRID, BG)

    residual_legend_matches = _build_residual_legend_only_matches(frames, sprites, legend_map)
    residual_legend_out = _build_output_from_matches(
        frames,
        residual_legend_matches,
        frame_size,
        ROWS_GRID,
        COLS_GRID,
        BG,
    )

    repeated_groups = 0
    repeated_slots = 0
    unique_unlisted = 0
    frame_to_sprite = {int(frame_color): int(sprite_color) for sprite_color, frame_color in legend_map.items()}
    frames_by_holes: dict[int, list[dict]] = {}
    for frame in frames:
        frames_by_holes.setdefault(int(frame["holes"]), []).append(frame)
    for group in frames_by_holes.values():
        if len(group) > 1:
            repeated_groups += 1
            repeated_slots += len(group)
        else:
            frame_color = int(group[0]["color"])
            if frame_color not in frame_to_sprite:
                unique_unlisted += 1

    return {
        "repeated_groups": int(repeated_groups),
        "repeated_slots": int(repeated_slots),
        "legend_entries": int(len(legend_map)),
        "unique_unlisted": int(unique_unlisted),
        "count_only_diff": int(not np.array_equal(combo_out, count_only_out)),
        "legend_only_diff": int(not np.array_equal(combo_out, residual_legend_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    legend_entries_n = 2
    legend_w = _legend_width(legend_entries_n)

    grid_pixel_h = ROWS_GRID * frame_size
    grid_pixel_w = COLS_GRID * frame_size
    min_h = max(CANVAS_MIN, LEGEND_RESERVED_ROWS + grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, max(grid_pixel_w, legend_w) + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None

    h = random.randint(min_h, CANVAS_MAX)
    w = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    picked = random.sample(available, 2 * N)
    frame_colors = [int(color) for color in picked[:N]]
    sprite_colors = [int(color) for color in picked[N:]]

    hole_multiset = _generate_hole_multiset(frame_size)
    if hole_multiset is None:
        return None
    frame_holes = hole_multiset[:]
    sprite_holes = hole_multiset[:]
    random.shuffle(frame_holes)
    random.shuffle(sprite_holes)

    frames: list[dict] = []
    for idx, hole_count in enumerate(frame_holes):
        patch = _make_frame(frame_size, int(hole_count), int(frame_colors[idx]))
        if patch is None:
            return None
        ri, ci = divmod(idx, COLS_GRID)
        frames.append(
            {
                "id": idx,
                "holes": int(hole_count),
                "color": int(frame_colors[idx]),
                "patch": patch,
                "out_r": int(ri * frame_size),
                "out_c": int(ci * frame_size),
            }
        )

    sprite_templates: list[dict] = []
    for idx, hole_count in enumerate(sprite_holes):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(sprite_colors[idx]))
        if patch is None:
            return None
        sprite_templates.append(
            {
                "id": idx,
                "holes": int(hole_count),
                "color": int(sprite_colors[idx]),
                "content": patch,
            }
        )

    canvas = np.full((h, w), BG, dtype=int)

    legend_c = random.randint(0, w - legend_w)
    canvas[0:LEGEND_RESERVED_ROWS, :] = LEGEND_TMP_COLOR

    r0 = random.randint(LEGEND_RESERVED_ROWS, h - grid_pixel_h)
    c0 = random.randint(0, w - grid_pixel_w)
    for frame in frames:
        rr = int(r0 + frame["out_r"])
        cc = int(c0 + frame["out_c"])
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

    sprites: list[dict] = []
    for template in sprite_templates:
        patch = template["content"]
        try:
            rr, cc = random_free_location(canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, rr, cc, transparent=BG)
        sprites.append(
            {
                **template,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "in_r": int(rr),
                "in_c": int(cc),
            }
        )

    repeated_values = [value for value, count in Counter(frame_holes).items() if int(count) > 1]
    if len(repeated_values) != 1:
        return None
    repeated_holes = int(repeated_values[0])
    repeated_frames = sorted(
        [frame for frame in frames if int(frame["holes"]) == int(repeated_holes)],
        key=lambda item: (int(item["out_r"]), int(item["out_c"])),
    )
    repeated_sprites = sorted(
        [sprite for sprite in sprites if int(sprite["holes"]) == int(repeated_holes)],
        key=lambda item: (int(item["in_r"]), int(item["in_c"])),
    )
    if len(repeated_frames) != 2 or len(repeated_sprites) != 2:
        return None

    legend_pairs = [
        (int(repeated_sprites[1]["color"]), int(repeated_frames[0]["color"])),
        (int(repeated_sprites[0]["color"]), int(repeated_frames[1]["color"])),
    ]
    random.shuffle(legend_pairs)
    legend_map = {int(sprite_color): int(frame_color) for sprite_color, frame_color in legend_pairs}

    legend = _make_legend(legend_pairs)
    canvas[0:LEGEND_RESERVED_ROWS, :] = BG
    canvas[0:LEGEND_H, legend_c : legend_c + legend_w] = legend

    metrics = _combo_metrics(frames, sprites, legend_map, frame_size)
    if metrics["repeated_groups"] != 1 or metrics["repeated_slots"] != 2:
        return None
    if metrics["legend_entries"] != 2 or metrics["unique_unlisted"] != 2:
        return None
    if metrics["count_only_diff"] != 1 or metrics["legend_only_diff"] != 1:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_output_from_matches(
        frames,
        _match_by_combo_rule(frames, sprites, legend_map),
        frame_size,
        ROWS_GRID,
        COLS_GRID,
        BG,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
