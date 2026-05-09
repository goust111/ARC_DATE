# Variant of 8698868d
# Dimension: D15 — src-dst 基数关系 (multi_step)
# Change: 1:1 双射 — 每个精灵恰好匹配一个框架，反之亦然... → 框架多于精灵（部分框架空置）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for the D15 variant
===================================

source_task:
- None. There is no extra reference task to inspect for this dimension change.

a) Concretization and behavior trace
- The visual language stays the same: a tight grid of large colored square frames sits on the canvas, and smaller colored square sprites are scattered elsewhere.
- The original task had exactly one sprite for every frame hole-count. Here the frame grid contains more frames than there are sprites.
- Each frame still advertises a unique hole-count by putting that many 4-disconnected bg holes in its interior.
- Each sprite still advertises its own hole-count by having that many 4-connected bg-hole components.
- Matching still uses equality of hole-count. The only new behavior is that some frame counts do not appear among sprites, so those frames cannot be filled.
- Output is still the cropped frame grid, but unmatched frames now stay as hollow outlines: border in frame color, interior left bg.

Concrete trace:
- Suppose three frames have hole-counts [0, 1, 2], but only two sprites exist with counts [0, 2].
- Original rule path:
  every frame gets a sprite; the output builder can safely fill the whole square with frame color first because the matched sprite will overwrite the interior.
- Variant rule path:
  frame 0 gets sprite 0, frame 2 gets sprite 2, frame 1 has no partner and must remain only a border.
- Difference:
  the output constructor must no longer assume every frame will be overwritten inside. It must draw borders first and only paste sprite content for matched frames.

b) Complexity calibration
- To keep the new rule visible but not add unrelated ambiguity, frame hole-counts remain unique.
- Sprite hole-counts are a strict subset of frame hole-counts.
- This yields a simple partial injection: each sprite matches one frame, some frames match nothing.
- We avoid duplicated hole-counts across frames because that would introduce a second cardinality change (many-to-one matching) not requested by D15.

c) Implicit assumption audit of the original solver
- Square detection by size:
  assumes frames are the largest square components and sprites are the next-largest square components. Still valid; generator preserves that.
- Grid extent from frame components:
  assumes the frame grid is tight and gapless. Still valid; generator preserves that.
- Matching loop:
  already tolerates missing matches because it only records a match when one exists.
- Output fill:
  `out[frame_square] = frame_color` relied on the old bijection, because every interior would later be replaced by a sprite.
  In this variant that write would incorrectly fill unmatched frames solid.
  This is the critical line that must change.
- Hole recoloring:
  recoloring sprite holes to frame color is still valid for matched frames.

d) Placement legality
- Frames remain in a contiguous rows x cols grid with no gaps.
- Sprites are scattered only onto bg cells outside the grid, using `random_free_location` with padding.
- Frame hole-counts are chosen so both frame patches and sprite patches can realize the needed counts exactly.
- Generator continues to oracle-check every produced canvas via `main(canvas)`.

e) Visual observability
- Every generated instance has at least one matched frame and at least one unmatched frame.
- The number of sprites on the input canvas is visibly smaller than the number of frames.
- In output, some frames are filled and some remain hollow, making the partial-fill rule immediately visible.
- The omitted hole-counts are sampled randomly rather than always taking the largest/smallest counts, so the rule is not reducible to a position heuristic.

f) Visual cleanliness
- Empty frames are shown as single clean borders with bg interiors, not with extra overlays.
- Filled frames still use the original nesting presentation.
- No extra markers are added; the scene remains close to the base task.

g) Color inference
- Colors are not used for matching order or ranking.
- Frame colors only determine border color and recoloring of sprite holes after placement, exactly as in the base task.

New assumptions introduced by the modified solver
-------------------------------------------------
- The generator must guarantee that frame hole-counts are unique; otherwise the first-match policy would be ambiguous.
- The generator must guarantee that sprite hole-counts are unique and are a proper subset of the frame hole-counts.
- The generator must guarantee at least one unmatched frame, or the variant would collapse back to the original bijection.
"""


def _draw_frame_border(out: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    out[r0, c0 : c0 + size] = int(color)
    out[r0 + size - 1, c0 : c0 + size] = int(color)
    out[r0 : r0 + size, c0] = int(color)
    out[r0 : r0 + size, c0 + size - 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

    Variant rule:
    - There are more frames than sprites.
    - Frames whose hole-count has no corresponding sprite remain hollow in the output.
    """
    g = np.array(grid, dtype=int)
    BG = 8

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # Find square regions (lower fill threshold to catch sprites with holes).
    MIN_FILL_RATIO = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((int(h), comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict[str, int]] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != frame_color
            _lab, num_holes = label_mask(hole_mask, connectivity=4)
            frame_grid.append(
                {
                    "color": frame_color,
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": int(num_holes),
                }
            )

    sprites: list[dict[str, np.ndarray | int]] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], tuple[dict[str, int], dict[str, np.ndarray | int]]] = {}
    used_sprites: set[int] = set()
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if i not in used_sprites and int(f["holes"]) == int(s["holes"]):
                matches[(int(f["out_r"]), int(f["out_c"]))] = (f, s)
                used_sprites.add(i)
                break

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for f in frame_grid:
        fr = int(f["out_r"])
        fc = int(f["out_c"])
        frame_color = int(f["color"])
        _draw_frame_border(out, fr, fc, frame_size, frame_color)

        if (fr, fc) in matches:
            _frame, sprite = matches[(fr, fc)]
            content = np.array(sprite["content"], dtype=int).copy()
            sprite_color = int(sprite["color"])
            content[content != sprite_color] = frame_color
            sh = int(sprite["h"])
            sw = int(sprite["w"])
            out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


# --------------- generator ---------------

BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
TOTAL_FRAME_RANGE = (3, 6)
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300
CANVAS_MIN = 15
CANVAS_MAX = 30


_LAYOUTS_BY_FRAME_COUNT: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        _count = _r * _c
        if TOTAL_FRAME_RANGE[0] <= _count <= TOTAL_FRAME_RANGE[1]:
            _LAYOUTS_BY_FRAME_COUNT.setdefault(_count, []).append((_r, _c))


def _max_isolated_holes(square_size: int) -> int:
    inner_side = max(0, int(square_size) - 2)
    inner_cells = inner_side * inner_side
    return (inner_cells + 1) // 2


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

    if hole_count > _max_isolated_holes(size):
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    inner_cells = max(0, (size - 2) ** 2)
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

    _, num = label_mask(sprite != sprite_color, connectivity=4)
    if num != hole_count:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame

    if hole_count > _max_isolated_holes(frame_size):
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _, num = label_mask(frame != frame_color, connectivity=4)
    if num != hole_count:
        return None
    return frame


def _build_expected_output(
    frames: list[dict[str, int | np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_holes = {int(k): patch for k, patch in sprites}

    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = ri * frame_size
        cc = ci * frame_size
        frame_color = int(frame["color"])
        hole_count = int(frame["holes"])
        _draw_frame_border(expected, rr, cc, frame_size, frame_color)

        if hole_count in sprite_by_holes:
            content = sprite_by_holes[hole_count].copy()
            content[content == BG] = frame_color
            expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    return expected


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    sprite_capacity = _max_isolated_holes(sprite_size)
    feasible_frame_counts = [
        count
        for count in sorted(_LAYOUTS_BY_FRAME_COUNT)
        if count <= sprite_capacity + 1
    ]
    if not feasible_frame_counts:
        return None

    total_frames = random.choice(feasible_frame_counts)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_FRAME_COUNT[total_frames])
    sprite_count = random.randint(2, total_frames - 1)

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None
    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, total_frames)
    sprite_colors = random.sample(available, sprite_count)

    max_hole = min(_max_isolated_holes(frame_size), _max_isolated_holes(sprite_size))
    max_start = min(HOLE_START_MAX, max_hole - total_frames + 1)
    if max_start < 0:
        return None
    start = random.randint(0, max_start)
    frame_hole_counts = list(range(start, start + total_frames))
    random.shuffle(frame_hole_counts)

    sprite_hole_counts = random.sample(frame_hole_counts, sprite_count)
    random.shuffle(sprite_hole_counts)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(sprite_hole_counts):
        patch = _make_sprite_interior(sprite_size, hole_count, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((hole_count, patch))

    frames: list[dict[str, int | np.ndarray]] = []
    for i, hole_count in enumerate(frame_hole_counts):
        patch = _make_frame(frame_size, hole_count, frame_colors[i])
        if patch is None:
            return None
        frames.append({"holes": hole_count, "patch": patch, "color": frame_colors[i]})

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

    for _holes, spatch in sprites:
        try:
            r, c = random_free_location(canvas, spatch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, spatch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
