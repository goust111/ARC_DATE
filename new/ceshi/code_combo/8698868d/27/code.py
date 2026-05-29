# Combo variant of 8698868d (index 27)
# combo_spec: D9/1+D15/0+D16/0
# Name: 重复洞数+过剩精灵+图例消歧

from __future__ import annotations

# Design analysis for combo D9/1 + D15/0 + D16/0
# ------------------------------------------------
# a) Concrete behavior and trace:
#    The source task paired each frame with the unique sprite that had the same
#    4-connected hole count. That relied on two hidden simplifications:
#      - every hole count appeared exactly once;
#      - there were no extra sprites.
#    This combo deliberately removes both simplifications and then adds a visible
#    top-edge legend that resolves the ambiguity.
#    Concrete trace:
#      - Suppose the frame grid contains three frames with hole counts [1, 1, 2].
#      - Outside the grid there are four sprites with hole counts [1, 1, 1, 2].
#        Two of the "1-hole" sprites are valid and one is an extra distractor.
#      - The top legend says:
#            blue sprite -> red frame
#            green sprite -> yellow frame
#            maroon sprite -> grey frame
#      - Original logic would only look at hole counts, so among the three 1-hole
#        sprites it would have to guess by encounter order which two survive and
#        which one goes to which 1-hole frame.
#      - New logic first reads the legend, keeps only sprite colors mentioned in
#        it, and sends each listed sprite to the frame with the listed frame color.
#        Hole counts remain visible and still agree inside each valid pair, but
#        they are no longer sufficient to decide either selection or tie-breaking.
#    This makes the three dimensions cooperate instead of acting independently:
#      - D9 creates same-hole ambiguity;
#      - D15 injects extra same-scale candidates;
#      - D16 provides the explicit resolution mechanism.
# b) Complexity calibration:
#    The generator keeps only 3-4 frames. That is enough to show repeated counts,
#    at least one extra sprite, and a short readable legend, while staying within
#    the 9 non-background ARC colors.
#    Hole counts are drawn from a short consecutive range and forced to include a
#    repeated positive value. Extra sprites also reuse frame hole counts rather
#    than inventing new ones, so "filter by unmatched count" is impossible.
#    The legend uses only N entries, one per valid sprite, with 2x2 vertical color
#    bands along the top edge. This keeps the symbolic cue obvious without making
#    the canvas visually busy.
# c) Implicit assumption audit against the source solver:
#    - `sizes[max_size] -> frames` still works because frames remain the larger
#      squares and sprites remain the smaller squares; the legend pieces are only
#      2x2 and never pass the 4x4 square filter.
#    - The original greedy loop assumed `sprite_count == frame_count` and "equal
#      hole count identifies exactly one sprite". Both are false here, so matching
#      must stop depending on the hole statistic.
#    - The original repeated-count variant used sprite reading order as a second
#      key. That is intentionally broken here: the generator rejects any sample
#      whose legend-guided output equals the old hole-order baseline output.
#    - The solver now assumes every frame color appears exactly once in the legend
#      target row and every valid sprite color appears exactly once in the legend
#      source row. The generator guarantees this by using distinct frame colors and
#      distinct valid sprite colors.
#    - The solver also assumes the legend-selected sprite for a frame has the same
#      hole count as that frame. The generator enforces this so the legend really
#      resolves ambiguity within the intended hole-count buckets instead of
#      introducing an unrelated pairing rule.
# d) Placement legality:
#    The legend is reserved to rows 0-1, the frame grid begins at row >= 3, and
#    sprites are placed afterward with `random_free_location(..., padding=1)`.
#    That prevents overlap, touching, and accidental component merging.
#    Canvas size is biased upward using an area lower bound so the extra sprites
#    still fit comfortably.
# e) Visual observability:
#    A human sees three recurring clues in each sample:
#      - a compact frame grid with repeated hole counts;
#      - more scattered sprite squares than frame slots;
#      - a top legend whose top colors appear only on the useful sprites and whose
#        bottom colors appear on the frame borders.
#    Because extra sprites can share the same hole counts as valid sprites, the
#    legend is visibly necessary. Because repeated hole groups exist among the
#    valid pairs, the legend also resolves which same-hole sprite belongs to which
#    same-hole frame.
# f) Visual cleanliness:
#    The canvas still contains only three clean regions: the 2-row legend, one
#    tight frame grid, and scattered sprites. There are no overlapping markers or
#    mixed encodings inside the same square. Extra sprites are full-size peers of
#    the valid sprites, so the added difficulty is semantic rather than clutter.
# g) Change inferability:
#    No color-id arithmetic is used. The legend is spatial and symbolic: top row
#    colors point to sprite bodies, bottom row colors point to frame borders.
#    The repeated hole counts provide supporting evidence that hole-count matching
#    is relevant but incomplete; the legend explains the missing piece.
# h) New solver assumptions introduced here and how generation guarantees them:
#    - Assumption: legend entries are the only reliable match keys.
#      Guarantee: the generator includes at least one sample-specific disagreement
#      with the old "match by hole count, then by sprite position" baseline.
#    - Assumption: extra sprites are precisely those whose colors are absent from
#      the legend.
#      Guarantee: every extra sprite has a unique non-legend color and at least
#      one of them shares a repeated frame hole count.
#    - Assumption: frame colors are unique.
#      Guarantee: frame colors are sampled without replacement from the palette.

from collections import Counter
from pathlib import Path
import math
import random
import sys

import numpy as np


def _find_solvers_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "solvers" / "arc_utils.py"
        if candidate.exists():
            return candidate.parent
    raise RuntimeError("could not locate solvers/arc_utils.py")


_SOLVERS_DIR = _find_solvers_dir()
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (3, 4)
CANVAS_MIN = 18
CANVAS_MAX = 30
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 50
GEN_RETRIES = 800
MIN_FILL_RATIO = 0.5
MAX_DISTINCT_HOLE_VALUES = 3
MAX_EXTRA_SPRITES = 2
HOLE_START_MAX = 1
LEGEND_H = 2
LEGEND_CELL_W = 2
LEGEND_GAP = 1
LEGEND_RESERVED_ROWS = 3


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _rows in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _cols in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        n_items = _rows * _cols
        if N_RANGE[0] <= n_items <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(n_items, []).append((_rows, _cols))
_VALID_NS = sorted(_LAYOUTS_BY_N)


def _legend_width(count: int) -> int:
    return int(count * LEGEND_CELL_W + max(0, count - 1) * LEGEND_GAP)


def _count_holes(region: np.ndarray, solid_color: int) -> int:
    hole_mask = np.array(region, dtype=int) != int(solid_color)
    _lab, num_holes = label_mask(hole_mask, connectivity=4)
    return int(num_holes)


def _read_legend_map(grid: np.ndarray, bg: int) -> dict[int, int]:
    """Read the 2-row top legend as sprite_color -> frame_color."""
    band = (grid[:LEGEND_H, :] != int(bg)).any(axis=0)
    assert bool(band.any())

    mapping: dict[int, int] = {}
    c = 0
    width = int(grid.shape[1])
    while c < width:
        if not bool(band[int(c)]):
            c += 1
            continue

        c0 = int(c)
        while c + 1 < width and bool(band[int(c + 1)]):
            c += 1
        c1 = int(c)

        block = grid[:LEGEND_H, c0 : c1 + 1]
        assert block.shape[0] == LEGEND_H
        assert bool((block[0, :] != int(bg)).all())
        assert bool((block[1, :] != int(bg)).all())

        sprite_vals = {int(v) for v in block[0, :].tolist()}
        frame_vals = {int(v) for v in block[1, :].tolist()}
        assert len(sprite_vals) == 1 and len(frame_vals) == 1

        sprite_color = next(iter(sprite_vals))
        frame_color = next(iter(frame_vals))
        assert int(sprite_color) not in mapping
        mapping[int(sprite_color)] = int(frame_color)
        c += 1

    assert mapping
    return mapping


def _find_square_regions(grid: np.ndarray) -> dict[int, list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        box = comp["bbox"]
        height = int(bbox_h(box))
        width = int(bbox_w(box))
        area = int(comp["area"])
        if height == width and 4 <= height <= 10 and area >= int(height * height * MIN_FILL_RATIO):
            square_regions.append((height, comp))

    assert square_regions

    by_size: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        by_size.setdefault(int(size), []).append(comp)
    return by_size


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    legend_map = _read_legend_map(g, BG)
    by_size = _find_square_regions(g)

    size_list = sorted(by_size, reverse=True)
    assert len(size_list) >= 2
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1])

    frames_raw = by_size.get(frame_size, [])
    sprites_raw = by_size.get(sprite_size, [])
    assert frames_raw and sprites_raw

    frame_positions = [(int(frame["bbox"][0]), int(frame["bbox"][1])) for frame in frames_raw]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frames: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            assert region.shape == (frame_size, frame_size)

            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            holes = _count_holes(region[1:-1, 1:-1], frame_color)
            frames.append(
                {
                    "id": len(frames),
                    "color": int(frame_color),
                    "holes": int(holes),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        box = comp["bbox"]
        sprite_region = crop_bbox(g, box)
        sprite_color = int(comp["color"])
        sprites.append(
            {
                "id": len(sprites),
                "color": int(sprite_color),
                "holes": _count_holes(sprite_region, sprite_color),
                "content": np.array(sprite_region, dtype=int).copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
                "in_r": int(box[0]),
                "in_c": int(box[1]),
            }
        )

    return {
        "legend_map": legend_map,
        "frame_size": int(frame_size),
        "sprite_size": int(sprite_size),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "frames": frames,
        "sprites": sprites,
    }


def _pair_by_legend(scene: dict) -> dict[int, dict]:
    legend_map = {int(k): int(v) for k, v in scene["legend_map"].items()}
    frame_to_sprite = {int(frame_color): int(sprite_color) for sprite_color, frame_color in legend_map.items()}
    sprites_by_color = {int(sprite["color"]): sprite for sprite in scene["sprites"]}
    frame_colors = {int(frame["color"]) for frame in scene["frames"]}
    assert frame_colors == set(frame_to_sprite)

    matches: dict[int, dict] = {}
    for frame in scene["frames"]:
        sprite_color = frame_to_sprite[int(frame["color"])]
        sprite = sprites_by_color[int(sprite_color)]
        assert int(sprite["holes"]) == int(frame["holes"])
        matches[int(frame["id"])] = sprite
    return matches


def _pair_by_hole_baseline(scene: dict) -> dict[int, dict]:
    """
    Old-style fallback:
    - ignore the legend;
    - keep every sprite whose hole count appears in the frame grid;
    - break repeats by sprite reading order.
    """
    frames_by_holes: dict[int, list[dict]] = {}
    sprites_by_holes: dict[int, list[dict]] = {}
    for frame in scene["frames"]:
        frames_by_holes.setdefault(int(frame["holes"]), []).append(frame)
    for sprite in scene["sprites"]:
        holes = int(sprite["holes"])
        if holes in frames_by_holes:
            sprites_by_holes.setdefault(holes, []).append(sprite)

    matches: dict[int, dict] = {}
    for hole_count, frame_group in frames_by_holes.items():
        frames_sorted = sorted(frame_group, key=lambda item: (int(item["out_r"]), int(item["out_c"])))
        sprites_sorted = sorted(
            sprites_by_holes.get(int(hole_count), []),
            key=lambda item: (int(item["in_r"]), int(item["in_c"])),
        )
        assert len(sprites_sorted) >= len(frames_sorted)
        for frame, sprite in zip(frames_sorted, sprites_sorted):
            matches[int(frame["id"])] = sprite
    return matches


def _pair_by_valid_hole_baseline(scene: dict) -> dict[int, dict]:
    """
    Old repeated-count tie-break after perfect extra filtering:
    - keep only legend-listed sprite colors;
    - still ignore legend targets and pair by hole count + sprite reading order.
    This isolates whether D9-style same-hole ambiguity remains active even after
    D15-style distractor filtering has been solved.
    """
    valid_colors = set(map(int, scene["legend_map"].keys()))
    filtered_scene = {
        **scene,
        "sprites": [sprite for sprite in scene["sprites"] if int(sprite["color"]) in valid_colors],
    }
    return _pair_by_hole_baseline(filtered_scene)


def _render_output(scene: dict, matches: dict[int, dict]) -> np.ndarray:
    frame_size = int(scene["frame_size"])
    out_h = int(scene["n_rows"] * frame_size)
    out_w = int(scene["n_cols"] * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in scene["frames"]:
        rr = int(frame["out_r"])
        cc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)

        sprite = matches[int(frame["id"])]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)
        sh, sw = content.shape
        out[rr + 1 : rr + 1 + sh, cc + 1 : cc + 1 + sw] = content

    return out


def fusion_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    """
    Quantify whether a generated scene actually activates the combo logic.
    """
    scene = _extract_scene(np.array(grid, dtype=int))
    legend_matches = _pair_by_legend(scene)
    baseline_matches = _pair_by_hole_baseline(scene)
    valid_only_matches = _pair_by_valid_hole_baseline(scene)
    legend_out = _render_output(scene, legend_matches)
    baseline_out = _render_output(scene, baseline_matches)
    valid_only_out = _render_output(scene, valid_only_matches)

    frame_hole_counts = Counter(int(frame["holes"]) for frame in scene["frames"])
    valid_sprite_colors = set(map(int, scene["legend_map"].keys()))
    extra_sprites = [sprite for sprite in scene["sprites"] if int(sprite["color"]) not in valid_sprite_colors]

    return {
        "frame_count": int(len(scene["frames"])),
        "sprite_count": int(len(scene["sprites"])),
        "extra_sprite_count": int(len(extra_sprites)),
        "repeated_hole_groups": int(sum(1 for cnt in frame_hole_counts.values() if int(cnt) > 1)),
        "extra_shared_count": int(sum(1 for sprite in extra_sprites if frame_hole_counts[int(sprite["holes"])] > 0)),
        "baseline_uses_extra": int(
            sum(1 for sprite in baseline_matches.values() if int(sprite["color"]) not in valid_sprite_colors)
        ),
        "baseline_mismatch_frames": int(
            sum(
                1
                for frame_id, sprite in legend_matches.items()
                if int(sprite["color"]) != int(baseline_matches[int(frame_id)]["color"])
            )
        ),
        "baseline_output_differs": bool(not np.array_equal(legend_out, baseline_out)),
        "valid_only_mismatch_frames": int(
            sum(
                1
                for frame_id, sprite in legend_matches.items()
                if int(sprite["color"]) != int(valid_only_matches[int(frame_id)]["color"])
            )
        ),
        "valid_only_output_differs": bool(not np.array_equal(legend_out, valid_only_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the compact frame grid by using the legend to choose which scattered sprites are valid and where they go.

    Input structure:
    - Background is fixed to azure (8). Along the top edge there is a 2-row legend made of short vertical color bands.
    - Elsewhere there is a tight grid of larger square frames and a larger set of smaller square sprites scattered on the background.
    - Hole counts are still visible on both sides, but several frames and sprites may share the same count, and some extra sprites are not used at all.

    Rule:
    - Read the top legend as `sprite_color -> frame_color`.
    - Detect the larger square frame grid and the smaller sprite squares, recording each sprite's content and each frame's border color.
    - For every frame, select the sprite whose color is mapped to that frame color by the legend; ignore all non-legend sprite colors.
    - Nest the chosen sprite inside the frame border and recolor the sprite's hole pixels to the destination frame color.

    Key insight:
    - Hole counts alone no longer identify the correct source sprite: repeats create tie groups and extra sprites reuse those same counts.
      The legend is the disambiguating instruction that both filters the useful sprites and resolves same-hole pairings.
    """
    scene = _extract_scene(np.array(grid, dtype=int))
    matches = _pair_by_legend(scene)
    return _render_output(scene, matches)


def _max_isolated_count(size: int) -> int:
    inner_side = int(size) - 2
    if inner_side <= 0:
        return 0
    return int((inner_side * inner_side + 1) // 2)


def _rotated_copy(values: list[int]) -> list[int]:
    if len(values) <= 1:
        return values[:]
    shift = random.randint(1, len(values) - 1)
    return values[shift:] + values[:shift]


def _generate_repeated_hole_multiset(n: int, max_hole: int) -> list[int] | None:
    if n < 3:
        return None

    max_distinct = min(MAX_DISTINCT_HOLE_VALUES, max_hole + 1, n - 1)
    if max_distinct < 2:
        return None

    distinct_n = random.randint(2, max_distinct)
    max_start = max_hole - (distinct_n - 1)
    if max_start < 0:
        return None

    start = random.randint(0, min(HOLE_START_MAX, max_start))
    base = list(range(start, start + distinct_n))
    counts = base[:]
    extra_pool = [value for value in base if value > 0] or base[:]
    while len(counts) < n:
        counts.append(random.choice(extra_pool))

    random.shuffle(counts)
    freq = Counter(int(v) for v in counts)
    if len(freq) == n:
        return None
    if not any(int(value) > 0 and int(cnt) > 1 for value, cnt in freq.items()):
        return None
    return [int(v) for v in counts]


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
    sprite = np.full((size, size), int(sprite_color), dtype=int)
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
        sprite[r, c] = int(BG)

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_idx, comp in enumerate(comps):
        grow_steps = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow_steps):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_hole:
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
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = int(BG)

    if _count_holes(sprite, int(sprite_color)) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), int(frame_color), dtype=int)
    if hole_count == 0:
        return frame

    inner_cells = (frame_size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = int(BG)

    if _count_holes(frame, int(frame_color)) != int(hole_count):
        return None
    return frame


def _make_legend(entries: list[tuple[int, int]]) -> np.ndarray:
    legend = np.full((LEGEND_H, _legend_width(len(entries))), int(BG), dtype=int)
    shuffled = entries[:]
    random.shuffle(shuffled)
    for idx, (sprite_color, frame_color) in enumerate(shuffled):
        c0 = int(idx * (LEGEND_CELL_W + LEGEND_GAP))
        legend[0, c0 : c0 + LEGEND_CELL_W] = int(sprite_color)
        legend[1, c0 : c0 + LEGEND_CELL_W] = int(frame_color)
    return legend


def _build_internal_scene(
    frames: list[dict],
    sprites: list[dict],
    frame_size: int,
    sprite_size: int,
    rows_grid: int,
    cols_grid: int,
    legend_map: dict[int, int],
) -> dict:
    return {
        "legend_map": {int(k): int(v) for k, v in legend_map.items()},
        "frame_size": int(frame_size),
        "sprite_size": int(sprite_size),
        "n_rows": int(rows_grid),
        "n_cols": int(cols_grid),
        "frames": [
            {
                "id": int(frame["id"]),
                "color": int(frame["color"]),
                "holes": int(frame["holes"]),
                "out_r": int(frame["out_r"]),
                "out_c": int(frame["out_c"]),
            }
            for frame in frames
        ],
        "sprites": [
            {
                "id": int(sprite["id"]),
                "color": int(sprite["color"]),
                "holes": int(sprite["holes"]),
                "content": np.array(sprite["content"], dtype=int).copy(),
                "h": int(sprite["h"]),
                "w": int(sprite["w"]),
                "in_r": int(sprite["in_r"]),
                "in_c": int(sprite["in_c"]),
            }
            for sprite in sprites
        ],
    }


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    max_hole = _max_isolated_count(sprite_size)
    frame_holes = _generate_repeated_hole_multiset(n_frames, max_hole)
    if frame_holes is None:
        return None

    repeated_values = [value for value, cnt in Counter(frame_holes).items() if int(value) > 0 and int(cnt) > 1]
    if not repeated_values:
        return None

    max_extra = min(MAX_EXTRA_SPRITES, 9 - 2 * n_frames)
    if max_extra < 1:
        return None
    extra_count = random.randint(1, max_extra)
    extra_holes = [int(random.choice(repeated_values)) for _ in range(extra_count)]

    legend_w = _legend_width(n_frames)
    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    total_sprites = n_frames + extra_count
    effective_sprite_span = sprite_size + 2 * SPRITE_PLACEMENT_PAD + 1
    area_lb = int(
        math.ceil(
            math.sqrt(grid_pixel_h * grid_pixel_w + total_sprites * effective_sprite_span * effective_sprite_span + legend_w * LEGEND_H + 20)
        )
    )
    min_h = max(CANVAS_MIN, LEGEND_RESERVED_ROWS + grid_pixel_h, area_lb)
    min_w = max(CANVAS_MIN, max(grid_pixel_w, legend_w), area_lb)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None

    h = random.randint(min_h, CANVAS_MAX)
    w = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    picked = random.sample(available, 2 * n_frames + extra_count)
    frame_colors = [int(color) for color in picked[:n_frames]]
    valid_sprite_colors = [int(color) for color in picked[n_frames : 2 * n_frames]]
    extra_colors = [int(color) for color in picked[2 * n_frames :]]

    frames: list[dict] = []
    for idx, hole_count in enumerate(frame_holes):
        patch = _make_frame(frame_size, int(hole_count), int(frame_colors[idx]))
        if patch is None:
            return None
        ri, ci = divmod(idx, cols_grid)
        frames.append(
            {
                "id": int(idx),
                "holes": int(hole_count),
                "color": int(frame_colors[idx]),
                "patch": patch,
                "out_r": int(ri * frame_size),
                "out_c": int(ci * frame_size),
            }
        )

    valid_sprite_holes = frame_holes[:]
    random.shuffle(valid_sprite_holes)
    valid_templates: list[dict] = []
    for idx, hole_count in enumerate(valid_sprite_holes):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(valid_sprite_colors[idx]))
        if patch is None:
            return None
        valid_templates.append(
            {
                "id": int(idx),
                "holes": int(hole_count),
                "color": int(valid_sprite_colors[idx]),
                "content": patch,
                "role": "valid",
            }
        )

    extra_templates: list[dict] = []
    for j, hole_count in enumerate(extra_holes):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(extra_colors[j]))
        if patch is None:
            return None
        extra_templates.append(
            {
                "id": int(n_frames + j),
                "holes": int(hole_count),
                "color": int(extra_colors[j]),
                "content": patch,
                "role": "extra",
            }
        )

    frame_groups: dict[int, list[int]] = {}
    sprite_groups: dict[int, list[int]] = {}
    for frame in frames:
        frame_groups.setdefault(int(frame["holes"]), []).append(int(frame["id"]))
    for sprite in valid_templates:
        sprite_groups.setdefault(int(sprite["holes"]), []).append(int(sprite["id"]))

    frame_to_valid_sprite: dict[int, int] = {}
    used_nontrivial_group = False
    for hole_count, frame_ids in frame_groups.items():
        sprite_ids = sorted(sprite_groups[int(hole_count)])
        if len(frame_ids) > 1:
            sprite_ids = _rotated_copy(sprite_ids)
            used_nontrivial_group = True
        for frame_id, sprite_id in zip(frame_ids, sprite_ids):
            frame_to_valid_sprite[int(frame_id)] = int(sprite_id)

    if not used_nontrivial_group:
        return None

    legend_map: dict[int, int] = {}
    for frame in frames:
        sprite_id = int(frame_to_valid_sprite[int(frame["id"])])
        sprite_color = int(valid_templates[sprite_id]["color"])
        legend_map[int(sprite_color)] = int(frame["color"])

    legend_entries = [(int(sprite_color), int(frame_color)) for sprite_color, frame_color in legend_map.items()]
    legend = _make_legend(legend_entries)

    canvas = np.full((h, w), int(BG), dtype=int)
    legend_c = random.randint(0, w - legend_w)
    paste(canvas, legend, 0, legend_c, transparent=BG)

    frame_r0 = random.randint(LEGEND_RESERVED_ROWS, h - grid_pixel_h)
    frame_c0 = random.randint(0, w - grid_pixel_w)
    for frame in frames:
        rr = int(frame_r0 + frame["out_r"])
        cc = int(frame_c0 + frame["out_c"])
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

    placed_sprites: list[dict] = []
    placed_valid_by_template_id: dict[int, dict] = {}
    sprite_templates = valid_templates + extra_templates
    random.shuffle(sprite_templates)
    for template in sprite_templates:
        patch = np.array(template["content"], dtype=int)
        placement_canvas = np.array(canvas, dtype=int).copy()
        placement_canvas[:LEGEND_RESERVED_ROWS, :] = -1
        try:
            rr, cc = random_free_location(placement_canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, rr, cc, transparent=BG)
        placed = {
            "id": int(template["id"]),
            "holes": int(template["holes"]),
            "color": int(template["color"]),
            "content": patch.copy(),
            "h": int(patch.shape[0]),
            "w": int(patch.shape[1]),
            "in_r": int(rr),
            "in_c": int(cc),
            "role": template["role"],
        }
        placed_sprites.append(placed)
        if template["role"] == "valid":
            placed_valid_by_template_id[int(template["id"])] = placed

    scene = _build_internal_scene(
        frames,
        placed_sprites,
        frame_size,
        sprite_size,
        rows_grid,
        cols_grid,
        legend_map,
    )
    legend_matches = {
        int(frame["id"]): placed_valid_by_template_id[int(frame_to_valid_sprite[int(frame["id"])])]
        for frame in frames
    }
    expected = _render_output(scene, legend_matches)

    baseline_matches = _pair_by_hole_baseline(scene)
    baseline_output = _render_output(scene, baseline_matches)
    if np.array_equal(expected, baseline_output):
        return None

    metrics = fusion_metrics(canvas)
    if not bool(metrics["baseline_output_differs"]):
        return None
    if int(metrics["baseline_mismatch_frames"]) < 1:
        return None
    if int(metrics["baseline_uses_extra"]) < 1:
        return None
    if int(metrics["extra_shared_count"]) < 1:
        return None
    if int(metrics["extra_sprite_count"]) < 1:
        return None
    if int(metrics["repeated_hole_groups"]) < 1:
        return None
    if not bool(metrics["valid_only_output_differs"]):
        return None
    if int(metrics["valid_only_mismatch_frames"]) < 1:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
