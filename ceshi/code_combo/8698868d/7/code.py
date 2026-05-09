# Combo variant of 8698868d (index 7)
# combo_spec: D2/1+D4/0
# Name: 无边框精灵反转着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste


"""
Design analysis for the D2/1 + D4/0 combo variant
=================================================

a) Concrete behavior and execution trace
----------------------------------------
- Original family:
  - Large square frames form a tight grid.
  - Smaller square sprites sit elsewhere.
  - Both are matched by hole-count, where the original sprite had a solid outer border.
  - In output, the sprite holes were recolored to the frame color, so no bg remained.
- Combo family:
  - Frames stay unchanged.
  - Sprites lose the solid border and become monochromatic connected blobs whose
    square bbox still has side `frame_size - 2`, but at least one bg notch opens
    to the bbox boundary.
  - Matching still uses hole-count, but for sprites we now count only enclosed bg
    components; edge-touching bg is exterior cutout, not a hole.
  - After nesting, every sprite-colored cell turns into the frame color, while every
    non-sprite cell inside the sprite bbox stays bg. The output therefore shows a bg
    negative of the sprite shape carved out of the frame block.

  Small trace on a 4x4 sprite bbox:
    sprite input patch =
      2 2 2 8
      2 8 2 8
      2 2 2 8
      2 2 2 2
    Here the right-edge run of 8s is an exterior notch because it touches the bbox edge,
    while the center 8 is an enclosed hole.
  - Old sprite hole counter would wrongly count both bg regions.
  - New sprite hole counter ignores the right-edge notch and counts only the center hole.
  - Old output logic for the borderless-only case would paste only the 2-cells, so both
    bg regions would disappear into frame color.
  - New combo output writes
      4 4 4 8
      4 8 4 8
      4 4 4 8
      4 4 4 4
    inside a frame of color 4, so both the enclosed hole and the border notch remain visible.

b) Complexity calibration
-------------------------
- The matching key stays one-dimensional and human-readable: number of enclosed holes.
- The D2 change is visually obvious because every generated sprite is forced to break its
  boundary somewhere; this prevents regression to miniature framed squares.
- The D4 inversion is also visually obvious because the output now contains bg cutouts
  inside frame blocks instead of becoming fully solid.
- To avoid ambiguity, each sample still uses a bijection of unique hole counts, and each
  sprite remains a single monochromatic connected component with a square bbox.
- The generator keeps the original small value ranges: 2-6 objects, frame size 6-8,
  consecutive hole counts. That preserves learnability.

c) Implicit assumption audit against the original main()
--------------------------------------------------------
- Original square-region detection assumed "all non-sprite-color cells inside the sprite
  bbox are holes". This breaks once bg can touch the bbox edge through notches.
- Original `hole_mask = sprite_region != sprite_color` overcounts border notches as holes.
  The combo solver replaces this with enclosed-hole counting.
- Original output recoloring assumed the sprite border existed and all bg inside the sprite
  bbox should disappear. That is false here: bg cells, whether enclosed holes or exterior
  cutouts, must stay bg in output.
- Frame extraction by solid square border is still valid because D2 changes only sprites.
- Matching by hole-count remains valid only if hole counts are unique across sprites and
  frames. The generator preserves that guarantee.

d) Placement legality
---------------------
- Frames still occupy one tight rectangular grid with no gaps.
- Sprites are placed in a reserved non-overlapping packing zone to the right or below the
  frame grid. This avoids random placement failures caused by transparent notches.
- Each sprite is generated first as a full square with enclosed holes, then boundary notches
  are carved. Validation keeps only shapes that are still one connected component, still use
  the full target bbox, and still preserve the requested enclosed-hole count.

e) Visual observability
-----------------------
- From 2-4 examples, a human can see two independent but cooperating changes:
  1. input sprites are no longer little bordered squares, because their boundaries have bites;
  2. output blocks contain bg cutouts that reproduce the sprite's negative space.
- The hole-count pairing remains inferable because frame interiors still encode the same count
  with isolated bg cells, while sprite enclosed holes keep that count despite the boundary bites.

f) Visual cleanliness
---------------------
- Each output frame block contains only two visual layers: frame color and bg cutouts.
- There are no extra markers or overlapping annotations.
- The reserved sprite packing zone keeps input objects separated, so the odd borderless shapes
  remain easy to inspect.

g) Change inferability
----------------------
- The color change is structural, not palette-dependent:
  - sprite-colored cells fuse into the frame color,
  - every non-sprite cell in the sprite bbox stays bg.
- Because frame colors vary independently from sprite colors, a solver can see that the output
  depends on frame location plus sprite shape, not on any fixed color table.

Post-change assumptions the generator must guarantee
----------------------------------------------------
- Every sprite bbox is exactly `(frame_size - 2) x (frame_size - 2)`.
- Every sprite is a single monochromatic 4-connected component with at least one boundary gap.
- Sprite enclosed-hole counts are unique and equal to the frame-hole multiset.
- At least one sprite has a nonzero enclosed-hole count, and every sprite has a boundary gap,
  so both halves of the combo are visible in generated samples.
"""


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
SPRITE_RETRIES = 80
GEN_RETRIES = 300


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))


def _count_enclosed_holes(region: np.ndarray, fg_color: int) -> int:
    """Count connected non-`fg_color` components that do not touch the bbox edge."""
    empty_mask = region != int(fg_color)
    lab, n = label_mask(empty_mask, connectivity=4)
    holes = 0
    h, w = region.shape
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        if coords.size == 0:
            continue
        if np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        ):
            continue
        holes += 1
    return int(holes)


def _has_solid_border(region: np.ndarray, color: int) -> bool:
    color = int(color)
    return bool(
        np.all(region[0, :] == color)
        and np.all(region[-1, :] == color)
        and np.all(region[:, 0] == color)
        and np.all(region[:, -1] == color)
    )


def _component_is_connected(region: np.ndarray, color: int) -> bool:
    _lab, n = label_mask(region == int(color), connectivity=4)
    return int(n) == 1


def _touches_all_bbox_sides(region: np.ndarray, color: int) -> bool:
    color = int(color)
    return bool(
        np.any(region[0, :] == color)
        and np.any(region[-1, :] == color)
        and np.any(region[:, 0] == color)
        and np.any(region[:, -1] == color)
    )


def _boundary_gap_count(region: np.ndarray, fg_color: int) -> int:
    fg_color = int(fg_color)
    boundary = np.concatenate([region[0, :], region[-1, :], region[1:-1, 0], region[1:-1, -1]])
    return int(np.count_nonzero(boundary != fg_color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match framed squares to borderless sprite blobs by enclosed-hole count, then
    render each matched blob as a background cutout inside its frame-colored block.

    Input structure:
    - Background is fixed at azure `8`.
    - Large square frame components have a solid 1-cell border and form a tight grid.
    - Smaller monochromatic sprites live elsewhere. Their bbox size is `frame_size - 2`,
      but their outer edge is not solid: some bg cells open to the bbox boundary.
    - Sprites may also contain enclosed bg holes.

    Rule:
    - Detect the true frames by their solid square borders and recover the grid layout.
    - Count frame holes from the frame interior as usual.
    - For each sprite, count only enclosed bg components; bg touching the sprite bbox edge
      is exterior cutout and must not contribute to the matching key.
    - Match each frame to the unique sprite with the same enclosed-hole count.
    - Output only the compact frame grid. Fill each frame block with the frame color, then
      place the sprite bbox inside so sprite-colored pixels become frame-colored and every
      non-sprite pixel stays background.

    Key insight:
    - The hard part is separating a borderless sprite's exterior bites from its true holes.
      Once that distinction is made, the output is the inverse recolor of the sprite mask:
      the sprite survives only as bg negative space carved into the frame color.
    """
    g = np.array(grid, dtype=int)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    frame_candidates: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        if h != w or not (6 <= h <= 8):
            continue
        region = crop_bbox(g, b)
        color = int(comp["color"])
        if _has_solid_border(region, color):
            frame_candidates.append((h, comp))

    assert frame_candidates

    frame_size = max(size for size, _comp in frame_candidates)
    sprite_size = frame_size - 2
    frames = [comp for size, comp in frame_candidates if int(size) == int(frame_size)]
    assert frames

    frame_lookup: dict[tuple[int, int], dict] = {}
    frame_bboxes: set[tuple[int, int, int, int]] = set()
    frame_positions: list[tuple[int, int]] = []
    for comp in frames:
        r0, c0, _r1, _c1 = comp["bbox"]
        key = (int(r0), int(c0))
        frame_lookup[key] = comp
        frame_bboxes.add(tuple(map(int, comp["bbox"])))
        frame_positions.append(key)

    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)
    assert n_rows * n_cols == len(frames)

    frame_grid: list[dict] = []
    frame_holes: set[int] = set()
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            comp = frame_lookup[(inp_r, inp_c)]
            region = crop_bbox(g, comp["bbox"])
            color = int(comp["color"])
            holes = int(_count_enclosed_holes(region, color))
            frame_holes.add(holes)
            frame_grid.append(
                {
                    "color": color,
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": holes,
                }
            )

    sprites: list[dict] = []
    for comp in comps:
        if tuple(map(int, comp["bbox"])) in frame_bboxes:
            continue
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        if h != sprite_size or w != sprite_size:
            continue
        region = crop_bbox(g, b)
        color = int(comp["color"])
        if _has_solid_border(region, color):
            continue
        if not _component_is_connected(region, color):
            continue
        holes = int(_count_enclosed_holes(region, color))
        sprites.append(
            {
                "color": color,
                "holes": holes,
                "content": region.copy(),
                "h": h,
                "w": w,
            }
        )

    sprite_by_holes = {int(sprite["holes"]): sprite for sprite in sprites}
    assert len(sprite_by_holes) == len(sprites)
    assert frame_holes == set(sprite_by_holes.keys())

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = sprite_by_holes[int(frame["holes"])]
        content = np.where(np.array(sprite["content"], dtype=int) == int(sprite["color"]), frame_color, BG)
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


def _make_full_square_with_holes(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
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

    if _count_enclosed_holes(sprite, sprite_color) != int(hole_count):
        return None
    return sprite


def _carve_random_notches(sprite: np.ndarray, sprite_color: int) -> np.ndarray:
    out = np.array(sprite, dtype=int).copy()
    size = int(out.shape[0])
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    notch_count = random.randint(1, min(3, len(sides)))
    for side in sides[:notch_count]:
        span = random.randint(1, min(2, size - 2))
        depth = random.randint(1, min(2, size - 2))
        if side == "top":
            c0 = random.randint(0, size - span)
            out[0:depth, c0 : c0 + span] = BG
        elif side == "bottom":
            c0 = random.randint(0, size - span)
            out[size - depth : size, c0 : c0 + span] = BG
        elif side == "left":
            r0 = random.randint(0, size - span)
            out[r0 : r0 + span, 0:depth] = BG
        else:
            r0 = random.randint(0, size - span)
            out[r0 : r0 + span, size - depth : size] = BG
    if np.all(out == BG):
        return sprite
    return out


def _make_borderless_sprite(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    for _ in range(SPRITE_RETRIES):
        base = _make_full_square_with_holes(size, hole_count, sprite_color)
        if base is None:
            continue
        candidate = _carve_random_notches(base, sprite_color)
        if _has_solid_border(candidate, sprite_color):
            continue
        if not _touches_all_bbox_sides(candidate, sprite_color):
            continue
        if not _component_is_connected(candidate, sprite_color):
            continue
        if _count_enclosed_holes(candidate, sprite_color) != int(hole_count):
            continue
        if _boundary_gap_count(candidate, sprite_color) == 0:
            continue
        return candidate
    return None


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

    if _count_enclosed_holes(frame, frame_color) != int(hole_count):
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_holes = {k: patch for k, patch in sprites}
    sprite_size = frame_size - 2

    for idx, (hole_count, _patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr = ri * frame_size
        cc = ci * frame_size
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color

        content = np.array(sprite_by_holes[int(hole_count)], dtype=int).copy()
        content[content != BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    return expected


def _packing_positions(count: int, item_size: int, pack_cols: int, pad: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    step = item_size + pad
    for idx in range(count):
        rr = (idx // pack_cols) * step
        cc = (idx % pack_cols) * step
        positions.append((int(rr), int(cc)))
    return positions


def _gen_attempt() -> np.ndarray | None:
    feasible_by_n: dict[int, list[dict]] = {}
    for frame_size in FRAME_SIZE_RANGE:
        sprite_size = frame_size - 2
        sprite_inner = sprite_size - 2
        max_sprite_holes = (sprite_inner * sprite_inner + 1) // 2
        for n_items, layouts in _LAYOUTS_BY_N.items():
            if n_items > max_sprite_holes + 1:
                continue
            for rows_grid, cols_grid in layouts:
                grid_pixel_h = rows_grid * frame_size
                grid_pixel_w = cols_grid * frame_size
                for side in ["bottom", "right"]:
                    for pack_cols in range(1, n_items + 1):
                        pack_rows = (n_items + pack_cols - 1) // pack_cols
                        zone_h = pack_rows * sprite_size + (pack_rows - 1) * SPRITE_PLACEMENT_PAD
                        zone_w = pack_cols * sprite_size + (pack_cols - 1) * SPRITE_PLACEMENT_PAD
                        if side == "bottom":
                            min_h = max(CANVAS_MIN, grid_pixel_h + SPRITE_PLACEMENT_PAD + zone_h)
                            min_w = max(CANVAS_MIN, max(grid_pixel_w, zone_w))
                        else:
                            min_h = max(CANVAS_MIN, max(grid_pixel_h, zone_h))
                            min_w = max(CANVAS_MIN, grid_pixel_w + SPRITE_PLACEMENT_PAD + zone_w)
                        if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
                            continue
                        feasible_by_n.setdefault(n_items, []).append(
                            {
                                "frame_size": frame_size,
                                "sprite_size": sprite_size,
                                "rows_grid": rows_grid,
                                "cols_grid": cols_grid,
                                "grid_pixel_h": grid_pixel_h,
                                "grid_pixel_w": grid_pixel_w,
                                "side": side,
                                "pack_cols": pack_cols,
                                "zone_h": zone_h,
                                "zone_w": zone_w,
                                "min_h": min_h,
                                "min_w": min_w,
                                "max_sprite_holes": max_sprite_holes,
                            }
                        )

    if not feasible_by_n:
        return None

    weighted_ns = [n for n in feasible_by_n for _ in range(n)]
    n_items = random.choice(weighted_ns)
    config = random.choice(feasible_by_n[n_items])
    frame_size = int(config["frame_size"])
    sprite_size = int(config["sprite_size"])
    rows_grid = int(config["rows_grid"])
    cols_grid = int(config["cols_grid"])
    grid_pixel_h = int(config["grid_pixel_h"])
    grid_pixel_w = int(config["grid_pixel_w"])
    zone_h = int(config["zone_h"])
    zone_w = int(config["zone_w"])
    canvas_h = random.randint(int(config["min_h"]), CANVAS_MAX)
    canvas_w = random.randint(int(config["min_w"]), CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n_items)
    sprite_colors = random.sample(available, n_items)

    max_sprite_holes = int(config["max_sprite_holes"])
    max_start = max(0, min(HOLE_START_MAX, max_sprite_holes - (n_items - 1)))
    start = random.randint(0, max_start)
    hole_counts = list(range(start, start + n_items))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(sprite_hole_perm):
        patch = _make_borderless_sprite(sprite_size, hole_count, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((hole_count, patch))

    if not any(_count_enclosed_holes(patch, sprite_colors[i]) > 0 for i, (_k, patch) in enumerate(sprites)):
        return None

    frames: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_size, hole_count, frame_colors[i])
        if patch is None:
            return None
        frames.append((hole_count, patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    if config["side"] == "bottom":
        r0 = random.randint(0, canvas_h - (grid_pixel_h + SPRITE_PLACEMENT_PAD + zone_h))
        c0 = random.randint(0, canvas_w - max(grid_pixel_w, zone_w))
        sprite_zone_r0 = r0 + grid_pixel_h + SPRITE_PLACEMENT_PAD
        sprite_zone_c0 = c0
    else:
        r0 = random.randint(0, canvas_h - max(grid_pixel_h, zone_h))
        c0 = random.randint(0, canvas_w - (grid_pixel_w + SPRITE_PLACEMENT_PAD + zone_w))
        sprite_zone_r0 = r0
        sprite_zone_c0 = c0 + grid_pixel_w + SPRITE_PLACEMENT_PAD

    for idx, (_k, frame_patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    slot_positions = _packing_positions(n_items, sprite_size, int(config["pack_cols"]), SPRITE_PLACEMENT_PAD)
    random.shuffle(slot_positions)
    boundary_gap_total = 0
    for (hole_count, sprite_patch), (dr, dc) in zip(sprites, slot_positions):
        r = sprite_zone_r0 + dr
        c = sprite_zone_c0 + dc
        paste(canvas, sprite_patch, r, c, transparent=BG)
        boundary_gap_total += _boundary_gap_count(sprite_patch, int(sprite_patch[sprite_patch != BG][0]))
    if boundary_gap_total <= 0:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    interior = out[1:-1, 1:-1] if min(out.shape) > 2 else out
    if int(np.count_nonzero(interior == BG)) <= 0:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
