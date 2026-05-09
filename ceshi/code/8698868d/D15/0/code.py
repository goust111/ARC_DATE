# Variant of 8698868d
# Dimension: D15 — src-dst 基数关系 (multi_step)
# Change: 1:1 双射 — 每个精灵恰好匹配一个框架，反之亦然
#   → 精灵多于框架（many:1 筛选）
#
# Design analysis
# ---------------
# spec:
#   baseline_objects:
#     frames:
#       count: N in 2..6
#       arrangement: tight rows x cols grid
#       size: 6..8 square
#       signal: interior hole-component count
#     sprites:
#       baseline_count: N
#       variant_count: N + K, K >= 1
#       size: frame_size - 2
#       style: same as original (solid outer ring, bg-colored holes inside)
#   variant_core:
#     visible_change:
#       - the input now contains extra sprite-like squares with the same size as the
#         useful sprites, so humans must count holes and discard the unmatched ones
#       - the output still contains only the frame grid populated by the matched
#         sprites, so the extra sprites disappear entirely
#     matching_rule:
#       - each frame still has one unique hole count
#       - exactly one sprite per frame shares that count
#       - distractor sprites use hole counts outside the frame set
#   behavior_trace:
#       original:
#         - collect all sprites
#         - pair each frame with the unique sprite of equal hole count
#       variant:
#         - collect all sprites
#         - first filter to the subset whose hole counts appear in the frame grid
#         - ignore every other sprite as a distractor
#         - then pair the surviving sprites to frames by equal hole count
#       difference:
#         - the input side now has an explicit reject step before the original
#           matching step; this is the only intended semantic change
#   complexity_calibration:
#       - distractors are same-sized squares, not tiny markers, so the change is
#         visually obvious and cannot be ignored by shape alone
#       - distractor hole counts are chosen from a small leftover set of feasible
#         counts, keeping the rule countable by eye instead of introducing a second
#         arbitrary cue
#       - valid hole counts remain consecutive as in the source task, preserving the
#         original concept while adding one extra reasoning step
#   implicit_assumption_audit:
#       - square-region size split still works because frames remain the largest
#         squares and all sprites, valid or distractor, remain the smaller size
#       - the old greedy matcher implicitly assumed sprite_count == frame_count; in
#         this variant that no longer holds, so main() now builds frame-count demand
#         first and filters sprites against that demand before matching
#       - output assembly still assumes at most one chosen sprite per frame; the
#         generator guarantees this by giving each valid frame hole count exactly one
#         matching sprite
#   placement_legality:
#       - generator chooses N only when the sprite size can realize at least N valid
#         counts plus one distractor count
#       - frame grid is pasted as one block; sprites are placed afterward with
#         random_free_location(..., padding=1) so squares do not overlap or touch
#       - canvas side lengths are biased upward using an area lower bound so the
#         extra distractors fit reliably
#   visual_observability:
#       - across examples, users can see more sprite candidates than frame slots
#       - only sprites whose hole count appears inside a frame are used in output
#       - distractors vanish in output, making the selection step visible
#   visual_tidiness:
#       - no extra annotations or overlays are added; the scene is still just a
#         frame block plus scattered same-scale sprites
#       - padding keeps components separated, so each candidate square is legible
#   color_inference:
#       - color behavior is unchanged from the source task: matched sprite holes are
#         recolored to the destination frame color
#       - distractor rejection never depends on numeric color ids, only on the
#         visible hole-count structure

from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PLACEMENT_PAD = 1
MAX_DISTRACTORS = 2
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 400


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        _n = _r * _c
        if N_RANGE[0] <= _n <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_n, []).append((_r, _c))


def _count_holes(region: np.ndarray, solid_color: int) -> int:
    hole_mask = np.array(region, dtype=int) != int(solid_color)
    _lab, num_holes = label_mask(hole_mask, connectivity=4)
    return int(num_holes)


def _max_isolated_count(size: int) -> int:
    inner_side = int(size) - 2
    if inner_side <= 0:
        return 0
    return int((inner_side * inner_side + 1) // 2)


def _find_square_regions(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)
    frames = sizes.get(frame_size, [])
    sprites = sizes.get(sprite_size, [])
    assert frames
    return frame_size, sprite_size, frames, sprites


def _extract_scene_signature(grid: np.ndarray) -> dict:
    frame_size, sprite_size, frames_raw, sprites_raw = _find_square_regions(grid)

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames_raw]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = grid[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            holes = _count_holes(region[1:-1, 1:-1], frame_color)
            frame_grid.append(
                {
                    "color": frame_color,
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": holes,
                }
            )

    sprites = []
    for idx, comp in enumerate(sprites_raw):
        sprite_region = crop_bbox(grid, comp["bbox"])
        sprite_color = int(comp["color"])
        sprites.append(
            {
                "index": int(idx),
                "color": sprite_color,
                "holes": _count_holes(sprite_region, sprite_color),
                "content": np.array(sprite_region, dtype=int).copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    frame_need = Counter(int(f["holes"]) for f in frame_grid)
    matched_sprites = [s for s in sprites if frame_need[int(s["holes"])] > 0]
    return {
        "frame_size": frame_size,
        "sprite_size": sprite_size,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "frames": frame_grid,
        "sprites": sprites,
        "frame_need": frame_need,
        "matched_sprites": matched_sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, but first discard extra
    sprite-like distractors whose hole counts never appear in the frame grid.
    """
    g = np.array(grid, dtype=int)
    scene = _extract_scene_signature(g)
    frame_size = int(scene["frame_size"])
    frame_grid = scene["frames"]
    frame_need: Counter = scene["frame_need"]

    # D15 change: sprite count can exceed frame count, so filter first.
    sprite_buckets: dict[int, list[dict]] = {}
    for sprite in scene["sprites"]:
        holes = int(sprite["holes"])
        if frame_need[holes] > 0:
            sprite_buckets.setdefault(holes, []).append(sprite)

    for hole_count, need in frame_need.items():
        assert len(sprite_buckets.get(int(hole_count), [])) >= int(need)

    matches: dict[tuple[int, int], dict] = {}
    used_per_hole = Counter()
    for frame in frame_grid:
        holes = int(frame["holes"])
        sprite_idx = int(used_per_hole[holes])
        matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprite_buckets[holes][sprite_idx]
        used_per_hole[holes] += 1

    out_h = int(scene["n_rows"] * frame_size)
    out_w = int(scene["n_cols"] * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = matches[(fr, fc)]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


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
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_hole:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(comps)
                        if j != ci
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    if _count_holes(sprite, sprite_color) != hole_count:
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

    if _count_holes(frame, frame_color) != hole_count:
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    valid_sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {int(k): np.array(spatch, dtype=int).copy() for k, spatch in valid_sprites}
    for idx, (k, _fpatch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(ri * frame_size)
        cc = int(ci * frame_size)
        fc = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = fc
        content = sprite_by_k[int(k)].copy()
        content[content == BG] = fc
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _choose_instance_params() -> tuple[int, int, int, int, int, int]:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    max_sprite_holes = _max_isolated_count(sprite_size)
    n_choices = [n for n in sorted(_LAYOUTS_BY_N) if n <= max_sprite_holes]
    assert n_choices
    n_frames = random.choice(n_choices)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])

    start_limit = max_sprite_holes - n_frames + 1
    start_choices = [s for s in range(start_limit + 1) if s <= HOLE_START_MAX and (s > 0 or s + n_frames - 1 < max_sprite_holes)]
    assert start_choices
    start = random.choice(start_choices)

    valid_counts = list(range(start, start + n_frames))
    distractor_pool = [k for k in range(max_sprite_holes + 1) if k not in set(valid_counts)]
    assert distractor_pool
    k_extra = random.randint(1, min(MAX_DISTRACTORS, len(distractor_pool)))
    return frame_size, sprite_size, rows_grid, cols_grid, start, k_extra


def _gen_attempt() -> np.ndarray | None:
    frame_size, sprite_size, rows_grid, cols_grid, start, k_extra = _choose_instance_params()
    n_frames = rows_grid * cols_grid

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    total_sprites = n_frames + k_extra
    effective_sprite_span = sprite_size + 2 * SPRITE_PLACEMENT_PAD
    needed_area = grid_pixel_h * grid_pixel_w + total_sprites * effective_sprite_span * effective_sprite_span + 12
    side_lb = max(CANVAS_MIN, grid_pixel_h, grid_pixel_w, int(math.ceil(math.sqrt(needed_area))))
    if side_lb > CANVAS_MAX:
        return None
    roomy_lb = max(side_lb, CANVAS_MAX - 5)
    H = random.randint(roomy_lb, CANVAS_MAX)
    W = random.randint(roomy_lb, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, total_sprites)

    valid_counts = list(range(start, start + n_frames))
    distractor_pool = [k for k in range(_max_isolated_count(sprite_size) + 1) if k not in set(valid_counts)]
    distractor_counts = random.sample(distractor_pool, k_extra)

    random.shuffle(valid_counts)
    sprite_valid_counts = valid_counts[:]
    random.shuffle(sprite_valid_counts)

    valid_sprites: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(sprite_valid_counts):
        patch = _make_sprite_interior(sprite_size, hole_count, int(sprite_colors[idx]))
        if patch is None:
            return None
        valid_sprites.append((hole_count, patch))

    distractor_sprites: list[tuple[int, np.ndarray]] = []
    for j, hole_count in enumerate(distractor_counts):
        color = int(sprite_colors[n_frames + j])
        patch = _make_sprite_interior(sprite_size, hole_count, color)
        if patch is None:
            return None
        distractor_sprites.append((hole_count, patch))

    frame_hole_counts = list(range(start, start + n_frames))
    random.shuffle(frame_hole_counts)
    frames: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(frame_hole_counts):
        patch = _make_frame(frame_size, hole_count, int(frame_colors[idx]))
        if patch is None:
            return None
        frames.append((hole_count, patch))

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_hole_count, fpatch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = fpatch

    all_sprites = valid_sprites + distractor_sprites
    random.shuffle(all_sprites)
    for _hole_count, spatch in all_sprites:
        try:
            r, c = random_free_location(canvas, spatch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, spatch, r, c, transparent=BG)

    expected = _build_expected_output(
        frames,
        valid_sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
    )

    try:
        out = main(canvas)
    except Exception:
        return None

    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    scene = _extract_scene_signature(canvas)
    if len(scene["sprites"]) <= len(scene["frames"]):
        return None
    if not any(int(sprite["holes"]) not in scene["frame_need"] for sprite in scene["sprites"]):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
