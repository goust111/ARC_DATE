# Combo variant of 8698868d (index 39)
# combo_spec: D11/2+D15/1
# Name: 大网格+部分空框架

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste


"""
Design analysis for the combo variant
=====================================

source_task:
- Base task 8698868d plus the requested combo changes D11/2 (3x3 large frame grid)
  and D15/1 (more frames than sprites, so some frames stay empty).

a) Concretization and behavior trace
- The scene still has two object families:
  large square frames packed into a grid, and smaller square sprites scattered
  elsewhere on the canvas.
- Base behavior:
  every frame has one unique hole-count, every sprite has the same set of
  unique hole-counts, so every frame is filled in the output.
- Combo behavior:
  the frame side becomes a fixed 3x3 grid of 9 frames, but only a subset of
  those 9 hole-counts appears among the sprites.
- Concrete trace:
  suppose the 9 frame slots carry hole-counts 0..8 in shuffled positions.
  If sprites exist only for counts {0, 2, 4, 7, 8}, then exactly those 5 frame
  positions are filled in the output. The remaining 4 slots stay as hollow
  borders. The solver must infer both the count-based pairing and the spatial
  filled/empty pattern induced by the missing counts.
- Difference from the original:
  the output can no longer be built by painting each frame square solid and
  assuming a sprite will overwrite the interior. Unmatched slots must preserve
  a BG interior.

b) Complexity calibration
- The layout difficulty is made visible by fixing a full 3x3 frame grid. This
  gives 9 candidate destinations, which is much larger than the base 2-6 range.
- To keep the matching itself interpretable, frame hole-counts remain unique:
  exactly the 9 values 0..8 are used once each.
- Sprite counts are a strict subset of those 9 values, with 4-6 sprites total.
  This guarantees both several filled slots and several hollow slots in every
  sample, so the new occupancy pattern is obvious.
- The frame size is fixed to 8. This is the smallest size that still allows a
  sprite of size 6 whose own interior can realize 9 distinct hole-component
  counts 0..8 under the original "solid border + bg holes" encoding.

c) Implicit assumption audit of the original solver
- Largest-square / next-largest-square detection still works:
  frames remain the largest square components and sprites the next-largest.
- Tight-grid assumption still works:
  all 9 frame slots are present, arranged as a gapless 3x3 lattice.
- The matching criterion still works:
  equality of connected hole-component counts is still the only key.
- The failing original line is the output fill policy:
  writing the full frame square in frame_color makes unmatched frames come out
  solid instead of hollow. The combo solver must draw only the border first and
  then paste sprite content only for matched counts.
- New solver assumption:
  frame hole-counts must be unique, and sprite hole-counts must be unique subset
  of frame hole-counts. The generator enforces this directly.

d) Placement legality
- The 3x3 frame grid occupies a 24x24 rectangle because frame_size is fixed to 8.
- A 30x30 canvas is used so there is always room for the scattered 6x6 sprites.
- The frame grid is anchored to one of four corners, leaving a horizontal strip
  and a vertical strip outside the grid.
- Sprite bounding boxes are placed entirely outside the frame-grid rectangle and
  are kept non-overlapping with one-cell padding between sprites.

e) Visual observability
- Human solvers can see 9 frame slots in the input but fewer sprites elsewhere.
- In the output, only some frame slots become filled, and the exact filled/empty
  pattern changes from sample to sample because the sprite subset changes.
- This avoids degeneration into a trivial heuristic like "top row empty" or
  "always fill the first k frames".

f) Visual cleanliness
- No extra markers are introduced. The only visible difference from the base task
  is that the frame grid is larger and some output interiors stay blank.
- Sprites are kept outside the frame rectangle, so the 3x3 structure stays easy
  to parse by eye.

g) Change inferability
- Colors do not drive the rule. Matching depends only on hole-component count.
- Sprite colors are chosen to differ from their matched frame colors so filled
  slots remain visually distinct from hollow ones and the nesting action is easy
  to notice across a few examples.
"""


BG = 8
FRAME_SIZE = 8
SPRITE_SIZE = FRAME_SIZE - 2
GRID_ROWS = 3
GRID_COLS = 3
TOTAL_FRAMES = GRID_ROWS * GRID_COLS
CANVAS_SIZE = 30
SPRITE_COUNT_RANGE = (4, 6)
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 60
SUBSET_RETRIES = 80
GEN_RETRIES = 400
AVAILABLE_COLORS = [c for c in range(10) if c != BG]


def _draw_frame_border(out: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    out[r0, c0 : c0 + size] = int(color)
    out[r0 + size - 1, c0 : c0 + size] = int(color)
    out[r0 : r0 + size, c0] = int(color)
    out[r0 : r0 + size, c0 + size - 1] = int(color)


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
                seeds.append((int(r), int(c)))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), int(sprite_color), dtype=int)
    if hole_count == 0:
        return sprite
    if hole_count > _max_isolated_holes(size):
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    inner_cells = max(0, (size - 2) ** 2)
    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_idx, comp in enumerate(comps):
        grow_steps = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow_steps):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = int(r + dr)
                    nc = int(c + dc)
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
            sprite[cell[0], cell[1]] = BG

    _, num = label_mask(sprite != int(sprite_color), connectivity=4)
    if int(num) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), int(frame_color), dtype=int)
    if hole_count == 0:
        return frame
    if hole_count > _max_isolated_holes(frame_size):
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _, num = label_mask(frame != int(frame_color), connectivity=4)
    if int(num) != int(hole_count):
        return None
    return frame


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad <= br0
        or br1 + pad <= ar0
        or ac1 + pad <= bc0
        or bc1 + pad <= ac0
    )


def _pattern_mask(slot_hole_counts: list[int], sprite_hole_counts: set[int]) -> np.ndarray:
    mask = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    for idx, hole_count in enumerate(slot_hole_counts):
        if int(hole_count) in sprite_hole_counts:
            mask[idx // GRID_COLS, idx % GRID_COLS] = 1
    return mask


def _pattern_is_good(mask: np.ndarray) -> bool:
    filled = np.argwhere(mask == 1)
    empty = np.argwhere(mask == 0)
    if filled.size == 0 or empty.size == 0:
        return False
    if len({int(r) for r, _c in filled}) < 2 or len({int(c) for _r, c in filled}) < 2:
        return False
    if len({int(r) for r, _c in empty}) < 2 or len({int(c) for _r, c in empty}) < 2:
        return False
    if not any(0 < int(mask[r].sum()) < GRID_COLS for r in range(GRID_ROWS)):
        return False
    if not any(0 < int(mask[:, c].sum()) < GRID_ROWS for c in range(GRID_COLS)):
        return False
    return True


def _choose_sprite_hole_counts(slot_hole_counts: list[int], sprite_count: int) -> list[int] | None:
    all_counts = list(slot_hole_counts)
    for _ in range(SUBSET_RETRIES):
        picked = random.sample(all_counts, sprite_count)
        if _pattern_is_good(_pattern_mask(slot_hole_counts, set(picked))):
            random.shuffle(picked)
            return [int(v) for v in picked]
    return None


def _assign_sprite_colors(
    sprite_hole_counts: list[int],
    frame_color_by_holes: dict[int, int],
) -> list[int] | None:
    for _ in range(SUBSET_RETRIES):
        colors = random.sample(AVAILABLE_COLORS, len(sprite_hole_counts))
        if all(int(colors[i]) != int(frame_color_by_holes[int(hole_count)]) for i, hole_count in enumerate(sprite_hole_counts)):
            return [int(c) for c in colors]
    return None


def _place_sprites_outside_layout(
    canvas: np.ndarray,
    sprite_patches: list[np.ndarray],
    layout_rect: tuple[int, int, int, int],
) -> bool:
    placed_rects: list[tuple[int, int, int, int]] = []
    H, W = map(int, canvas.shape)
    for spatch in sprite_patches:
        h, w = map(int, spatch.shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                rect = (int(r0), int(c0), int(r0 + h), int(c0 + w))
                if _rects_overlap(rect, layout_rect, pad=0):
                    continue
                if any(_rects_overlap(rect, other, pad=SPRITE_PLACEMENT_PAD) for other in placed_rects):
                    continue
                if np.any(canvas[r0 : r0 + h, c0 : c0 + w] != BG):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        paste(canvas, spatch, r0, c0, transparent=BG)
        placed_rects.append((int(r0), int(c0), int(r0 + h), int(c0 + w)))
    return True


def _build_expected_output(
    frames: list[dict[str, object]],
    sprites: list[dict[str, object]],
) -> np.ndarray:
    expected = np.full((GRID_ROWS * FRAME_SIZE, GRID_COLS * FRAME_SIZE), BG, dtype=int)
    sprite_by_holes = {int(sprite["holes"]): np.array(sprite["patch"], dtype=int) for sprite in sprites}

    for frame in frames:
        rr = int(frame["slot_r"]) * FRAME_SIZE
        cc = int(frame["slot_c"]) * FRAME_SIZE
        frame_color = int(frame["color"])
        _draw_frame_border(expected, rr, cc, FRAME_SIZE, frame_color)

        hole_count = int(frame["holes"])
        if hole_count in sprite_by_holes:
            content = sprite_by_holes[hole_count].copy()
            content[content == BG] = frame_color
            expected[rr + 1 : rr + 1 + SPRITE_SIZE, cc + 1 : cc + 1 + SPRITE_SIZE] = content

    return expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place scattered sprites into a large 3x3 frame grid by matching connected-hole counts.

    Input structure:
    - The background is azure (8). Nine larger square frames form a tight 3x3 grid somewhere on the canvas.
    - Smaller square sprites are scattered elsewhere. There are fewer sprites than frames.
    - Each frame and sprite has one solid object color, while bg-colored holes inside them encode a count.

    Rule:
    - Count the number of 4-connected hole components in each frame interior and in each sprite.
    - A sprite belongs inside the frame whose hole-component count matches its own.
    - Build the output as the compact 3x3 frame grid only: matched frames receive their sprite, with sprite-hole pixels recolored to the frame color; unmatched frames stay hollow borders.

    Key insight:
    - The decisive feature is the connected-hole count, and because some frame counts are missing on the sprite side, the output must preserve a partial filled-vs-hollow pattern across the 3x3 grid.
    """
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        bbox = comp["bbox"]
        h = int(bbox_h(bbox))
        w = int(bbox_w(bbox))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * 0.5):
            square_regions.append((int(h), comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames_raw = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert len(frames_raw) == TOTAL_FRAMES
    assert sprites_raw

    row_vals = sorted({int(comp["bbox"][0]) for comp in frames_raw})
    col_vals = sorted({int(comp["bbox"][1]) for comp in frames_raw})
    assert len(row_vals) == GRID_ROWS
    assert len(col_vals) == GRID_COLS
    row_to_idx = {int(r): idx for idx, r in enumerate(row_vals)}
    col_to_idx = {int(c): idx for idx, c in enumerate(col_vals)}

    frames: list[dict[str, object]] = []
    seen_frame_holes: set[int] = set()
    for comp in frames_raw:
        region = crop_bbox(g, comp["bbox"])
        frame_color = int(comp["color"])
        inner = region[1:-1, 1:-1]
        hole_mask = inner != frame_color
        _, num_holes = label_mask(hole_mask, connectivity=4)
        hole_count = int(num_holes)
        seen_frame_holes.add(hole_count)
        frames.append(
            {
                "color": frame_color,
                "holes": hole_count,
                "out_r": int(row_to_idx[int(comp["bbox"][0])] * frame_size),
                "out_c": int(col_to_idx[int(comp["bbox"][1])] * frame_size),
            }
        )
    assert len(seen_frame_holes) == TOTAL_FRAMES

    sprite_by_holes: dict[int, dict[str, object]] = {}
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _, num_holes = label_mask(hole_mask, connectivity=4)
        hole_count = int(num_holes)
        assert hole_count not in sprite_by_holes
        sprite_by_holes[hole_count] = {
            "color": sprite_color,
            "holes": hole_count,
            "content": sprite_region.copy(),
            "h": int(sprite_region.shape[0]),
            "w": int(sprite_region.shape[1]),
        }

    out = np.full((GRID_ROWS * frame_size, GRID_COLS * frame_size), BG, dtype=int)
    for frame in sorted(frames, key=lambda item: (int(item["out_r"]), int(item["out_c"]))):
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        _draw_frame_border(out, fr, fc, frame_size, frame_color)

        hole_count = int(frame["holes"])
        if hole_count not in sprite_by_holes:
            continue

        sprite = sprite_by_holes[hole_count]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _gen_attempt() -> np.ndarray | None:
    assert _max_isolated_holes(SPRITE_SIZE) >= TOTAL_FRAMES - 1

    frame_colors = random.sample(AVAILABLE_COLORS, TOTAL_FRAMES)
    slot_hole_counts = list(range(TOTAL_FRAMES))
    random.shuffle(slot_hole_counts)
    sprite_count = random.randint(SPRITE_COUNT_RANGE[0], SPRITE_COUNT_RANGE[1])
    sprite_hole_counts = _choose_sprite_hole_counts(slot_hole_counts, sprite_count)
    if sprite_hole_counts is None:
        return None

    frame_color_by_holes = {int(slot_hole_counts[idx]): int(frame_colors[idx]) for idx in range(TOTAL_FRAMES)}
    sprite_colors = _assign_sprite_colors(sprite_hole_counts, frame_color_by_holes)
    if sprite_colors is None:
        return None

    frames: list[dict[str, object]] = []
    for idx, hole_count in enumerate(slot_hole_counts):
        patch = _make_frame(FRAME_SIZE, int(hole_count), int(frame_colors[idx]))
        if patch is None:
            return None
        frames.append(
            {
                "holes": int(hole_count),
                "patch": patch,
                "color": int(frame_colors[idx]),
                "slot_r": idx // GRID_COLS,
                "slot_c": idx % GRID_COLS,
            }
        )

    sprites: list[dict[str, object]] = []
    for hole_count, sprite_color in zip(sprite_hole_counts, sprite_colors):
        patch = _make_sprite_interior(SPRITE_SIZE, int(hole_count), int(sprite_color))
        if patch is None:
            return None
        sprites.append({"holes": int(hole_count), "patch": patch, "color": int(sprite_color)})

    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    r0, c0 = random.choice([(0, 0), (0, CANVAS_SIZE - GRID_ROWS * FRAME_SIZE), (CANVAS_SIZE - GRID_ROWS * FRAME_SIZE, 0), (CANVAS_SIZE - GRID_ROWS * FRAME_SIZE, CANVAS_SIZE - GRID_COLS * FRAME_SIZE)])
    for frame in frames:
        rr = int(r0 + int(frame["slot_r"]) * FRAME_SIZE)
        cc = int(c0 + int(frame["slot_c"]) * FRAME_SIZE)
        canvas[rr : rr + FRAME_SIZE, cc : cc + FRAME_SIZE] = np.array(frame["patch"], dtype=int)

    layout_rect = (int(r0), int(c0), int(r0 + GRID_ROWS * FRAME_SIZE), int(c0 + GRID_COLS * FRAME_SIZE))
    sprite_patches = [np.array(sprite["patch"], dtype=int) for sprite in sprites]
    random.shuffle(sprite_patches)
    if not _place_sprites_outside_layout(canvas, sprite_patches, layout_rect):
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
