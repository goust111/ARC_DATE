# Combo variant of 8698868d (index 10)
# combo_spec: D2/1+D5/3
# Name: 无边框精灵纯图案输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Base task:
#      1) large square frames with solid 1-cell borders form a tight grid;
#      2) smaller square sprites are scattered elsewhere;
#      3) match by hole-count; output keeps the whole frame square.
#    - D2/1 changes the sprite source object. A sprite is no longer a mini framed
#      square. It is now a single monochrome connected pattern whose bbox is still
#      `frame_size - 2`, but at least one boundary cell is missing, so the outer
#      rim is not solid. Some empty regions may be enclosed holes; some may be
#      boundary notches open to the outside.
#    - D5/3 changes the output composition. After matching, we do not emit the
#      frame border at all. We only emit the `sprite_size x sprite_size` content
#      that would sit inside the frame.
#    - Small trace with `frame_size = 6`, `sprite_size = 4`:
#      original:
#        frame slot output is 6x6, border visible, interior sprite offset by (1,1).
#      D2/1 alone:
#        still 6x6, but the nested interior keeps the sprite's missing edge bites
#        as frame-colored cells.
#      fused combo:
#        first nest the borderless sprite into the frame interior, which turns all
#        empty sprite cells into frame color; then strip the frame border and keep
#        only the 4x4 interior. The final tile has no border ring at all, but its
#        edge may contain frame-colored cells where the sprite had boundary notches.
#
# b) Complexity calibration:
#    - Hole-count matching stays as the single combinatorial key. That keeps the
#      task coherent and avoids introducing multiple independent correspondences.
#    - Every sprite is forced to have at least one boundary gap, so the D2 change
#      is visually present in every generated sample instead of occasionally
#      collapsing back to a solid-bordered mini-square.
#    - Frame colors and sprite colors are sampled from disjoint non-bg sets and
#      `N` is limited to 2-4. This sacrifices the original maximum count, but it
#      guarantees that recolored gap/hole cells remain visible and do not blend
#      back into the sprite color, which would make the fused behavior ambiguous.
#
# c) Hidden assumptions in the original solver:
#    - Original sprite-hole counting treated every non-sprite-color cell inside the
#      sprite bbox as a hole. That only works when sprites have solid borders.
#      Here, boundary notches are exterior background and must be excluded.
#    - Original output filling assumed each slot size is `frame_size` and began by
#      painting the whole frame square. That would incorrectly reintroduce the
#      removed border in this combo.
#    - Original size grouping for sprites assumed "smaller square regions" were
#      still dense enough to be caught by a fill-ratio heuristic. Borderless
#      sprites are cleaner to detect by bbox size plus connectedness and lack of a
#      solid border, while frames remain identifiable by a perfectly solid border.
#    - New combined assumption: sprites must touch all four bbox sides so their
#      bbox stays exactly `sprite_size x sprite_size` even after notches are carved.
#      The generator enforces this explicitly.
#
# d) Placement legality:
#    - Frames occupy one tight rectangular grid with no gaps.
#    - Sprites are packed in a dedicated zone either below or to the right of that
#      grid, with at least one background cell between sprite bboxes.
#    - This ensures no overlaps, no out-of-bounds writes, and no accidental merges
#      between sprites, frames, or repeated colors.
#
# e) Visual observability:
#    - A human can compare input and output and see two changes at once:
#      1) the output tiles are smaller because the border ring vanished;
#      2) the tile edges can contain frame-colored cells, proving the source sprite
#         itself was borderless rather than a solid mini-frame.
#    - Repeating this over 2-4 examples is enough to infer "match by holes, place
#      inside frame, then keep only the inner pattern block".
#
# f) Visual cleanliness:
#    - The output is a regular tiling of compact square patches with no extra
#      markers, legends, or overlapping decorations.
#    - The input keeps a clean separation between the frame grid and the sprite
#      packing zone, so object families remain easy to segment visually.
#
# g) Change inferability:
#    - The only recoloring rule is still structural: empty cells inside the matched
#      sprite bbox become the matched frame color. Because frame and sprite colors
#      are disjoint in generated examples, this recoloring remains visible.
#    - The fused signature is not just "cropped output": boundary gaps on the
#      output tile edges quantify the borderless-source behavior and distinguish
#      the combo from a plain D5/3 crop of solid-bordered sprites.
#
# 9) New solver assumptions checked against the generator:
#    - Largest solid-border square components are exactly the frames.
#    - Every sprite is one monochrome connected component with bbox
#      `sprite_size x sprite_size`, touches all four bbox sides, and has at least
#      one boundary gap.
#    - Hole counts are unique and shared bijectively between frames and sprites.
#    - Matched frame and sprite colors differ, so empty-cell recoloring stays
#      visible on the cropped pure-pattern output.


BG = 8

FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 4)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
SPRITE_RETRIES = 80
GEN_RETRIES = 400

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))


def _count_enclosed_holes(region: np.ndarray, fg_color: int) -> int:
    """Count empty 4-components that do not touch the region boundary."""
    empty_mask = region != int(fg_color)
    labels, num = label_mask(empty_mask, connectivity=4)
    holes = 0
    h, w = region.shape
    for k in range(1, num + 1):
        coords = np.argwhere(labels == k)
        touches_edge = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        )
        if not touches_edge:
            holes += 1
    return int(holes)


def _has_solid_border(region: np.ndarray, color: int) -> bool:
    return bool(
        np.all(region[0, :] == int(color))
        and np.all(region[-1, :] == int(color))
        and np.all(region[:, 0] == int(color))
        and np.all(region[:, -1] == int(color))
    )


def _component_is_connected(region: np.ndarray, color: int) -> bool:
    _labels, num = label_mask(region == int(color), connectivity=4)
    return int(num) == 1


def _touches_all_bbox_sides(region: np.ndarray, color: int) -> bool:
    return bool(
        np.any(region[0, :] == int(color))
        and np.any(region[-1, :] == int(color))
        and np.any(region[:, 0] == int(color))
        and np.any(region[:, -1] == int(color))
    )


def _boundary_gap_count(region: np.ndarray, color: int) -> int:
    boundary = np.concatenate([region[0, :], region[-1, :], region[1:-1, 0], region[1:-1, -1]])
    return int(np.count_nonzero(boundary != int(color)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the frame grid, match each borderless sprite to a frame by enclosed-hole count,
    and output only the recolored inner pattern blocks with the frame borders removed.

    Input structure:
    - Background is fixed azure `8`.
    - Large monochrome square frames with solid 1-cell borders are arranged in a tight grid.
    - Smaller scattered sprites are single-color connected components of size `frame_size-2`.
      They touch all four sides of their own bbox but do not have a solid outer border, and
      they may contain both enclosed holes and boundary notches.

    Rule:
    - Detect the large solid-border squares as frames and infer the frame-grid layout.
    - Count the number of enclosed empty 4-components in each frame interior.
    - For each sprite, crop its bbox and count only empty 4-components that do not touch the
      bbox edge.
    - Match frames to sprites by equal enclosed-hole count.
    - Recolor every empty sprite cell to the matched frame color, then output only these
      `sprite_size x sprite_size` patches in frame-grid order.

    Key insight:
    Empty cells on a borderless sprite's boundary are exterior notches rather than holes, so
    they must be ignored for matching but still become frame-colored after nesting. Stripping
    the frame border afterwards leaves a pure pattern-block grid with no visible frame ring.
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
        if _has_solid_border(region, int(comp["color"])):
            frame_candidates.append((h, comp))

    assert frame_candidates
    frame_size = max(size for size, _ in frame_candidates)
    sprite_size = int(frame_size - 2)
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

    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)
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
                    "slot_r": int(ri),
                    "slot_c": int(ci),
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
        if not _component_is_connected(region, color):
            continue
        if not _touches_all_bbox_sides(region, color):
            continue
        if _has_solid_border(region, color):
            continue
        holes = int(_count_enclosed_holes(region, color))
        sprites.append(
            {
                "color": color,
                "holes": holes,
                "content": region.copy(),
            }
        )

    sprite_by_holes = {int(sprite["holes"]): sprite for sprite in sprites}
    assert len(sprite_by_holes) == len(sprites)
    assert frame_holes == set(sprite_by_holes.keys())
    assert len(sprites) == len(frame_grid)

    out_h = int(n_rows * sprite_size)
    out_w = int(n_cols * sprite_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        sprite = sprite_by_holes[int(frame["holes"])]
        content = np.array(sprite["content"], dtype=int).copy()
        content[content == BG] = int(frame["color"])
        rr = int(frame["slot_r"] * sprite_size)
        cc = int(frame["slot_c"] * sprite_size)
        out[rr : rr + sprite_size, cc : cc + sprite_size] = content

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
        if _boundary_gap_count(candidate, sprite_color) <= 0:
            continue
        return candidate
    return None


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
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * sprite_size, cols_grid * sprite_size), BG, dtype=int)
    sprite_by_holes = {holes: patch for holes, patch in sprites}
    for idx, (holes, _frame_patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr = int(ri * sprite_size)
        cc = int(ci * sprite_size)
        content = np.array(sprite_by_holes[int(holes)], dtype=int).copy()
        content[content == BG] = int(frame_colors[idx])
        expected[rr : rr + sprite_size, cc : cc + sprite_size] = content
    return expected


def _packing_positions(count: int, item_size: int, pack_cols: int, pad: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    step = int(item_size + pad)
    for idx in range(count):
        rr = (idx // pack_cols) * step
        cc = (idx % pack_cols) * step
        positions.append((int(rr), int(cc)))
    return positions


def _gen_attempt() -> np.ndarray | None:
    feasible_by_n: dict[int, list[dict]] = {}
    for frame_size in FRAME_SIZE_RANGE:
        sprite_size = int(frame_size - 2)
        sprite_inner = int(sprite_size - 2)
        max_sprite_holes = int((sprite_inner * sprite_inner + 1) // 2)
        for n_items, layouts in _LAYOUTS_BY_N.items():
            if n_items > max_sprite_holes + 1:
                continue
            if 2 * n_items > 9:
                continue
            for rows_grid, cols_grid in layouts:
                grid_pixel_h = int(rows_grid * frame_size)
                grid_pixel_w = int(cols_grid * frame_size)
                for side in ["bottom", "right"]:
                    for pack_cols in range(1, n_items + 1):
                        pack_rows = (n_items + pack_cols - 1) // pack_cols
                        zone_h = int(pack_rows * sprite_size + (pack_rows - 1) * SPRITE_PLACEMENT_PAD)
                        zone_w = int(pack_cols * sprite_size + (pack_cols - 1) * SPRITE_PLACEMENT_PAD)
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
                                "frame_size": int(frame_size),
                                "sprite_size": int(sprite_size),
                                "rows_grid": int(rows_grid),
                                "cols_grid": int(cols_grid),
                                "grid_pixel_h": int(grid_pixel_h),
                                "grid_pixel_w": int(grid_pixel_w),
                                "side": side,
                                "pack_cols": int(pack_cols),
                                "zone_h": int(zone_h),
                                "zone_w": int(zone_w),
                                "min_h": int(min_h),
                                "min_w": int(min_w),
                                "max_sprite_holes": int(max_sprite_holes),
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

    colors = random.sample([c for c in range(10) if c != BG], 2 * n_items)
    frame_colors = colors[:n_items]
    sprite_colors = colors[n_items:]

    max_sprite_holes = int(config["max_sprite_holes"])
    max_start = max(0, min(HOLE_START_MAX, max_sprite_holes - (n_items - 1)))
    start = random.randint(0, max_start)
    hole_counts = list(range(start, start + n_items))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, holes in enumerate(sprite_hole_perm):
        patch = _make_borderless_sprite(sprite_size, holes, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((int(holes), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, holes in enumerate(hole_counts):
        patch = _make_frame(frame_size, holes, frame_colors[i])
        if patch is None:
            return None
        frames.append((int(holes), patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    if config["side"] == "bottom":
        r0 = random.randint(0, canvas_h - (grid_pixel_h + SPRITE_PLACEMENT_PAD + zone_h))
        c0 = random.randint(0, canvas_w - max(grid_pixel_w, zone_w))
        sprite_zone_r0 = int(r0 + grid_pixel_h + SPRITE_PLACEMENT_PAD)
        sprite_zone_c0 = int(c0)
    else:
        r0 = random.randint(0, canvas_h - max(grid_pixel_h, zone_h))
        c0 = random.randint(0, canvas_w - (grid_pixel_w + SPRITE_PLACEMENT_PAD + zone_w))
        sprite_zone_r0 = int(r0)
        sprite_zone_c0 = int(c0 + grid_pixel_w + SPRITE_PLACEMENT_PAD)

    for idx, (_holes, frame_patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    slot_positions = _packing_positions(n_items, sprite_size, int(config["pack_cols"]), SPRITE_PLACEMENT_PAD)
    random.shuffle(slot_positions)
    for (_holes, sprite_patch), (dr, dc) in zip(sprites, slot_positions):
        rr = int(sprite_zone_r0 + dr)
        cc = int(sprite_zone_c0 + dc)
        canvas[rr : rr + sprite_size, cc : cc + sprite_size] = sprite_patch

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
        sprite_size,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    # Guard against degeneration to plain cropped solid-border sprites.
    boundary_gaps = [_boundary_gap_count(patch, sprite_colors[i]) for i, (_h, patch) in enumerate(sprites)]
    if min(boundary_gaps) <= 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
