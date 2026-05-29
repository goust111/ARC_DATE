# Variant of 8698868d
# Dimension: D2 — 源对象：精灵的结构 (src)
# Change: 正方形精灵（frame_size-2 边长），实心边框为精灵色，内部含若干 4-连通背景色洞分量，散落在框架网格之外的背景区域
#   → 精灵无边框（纯图案块）

from __future__ import annotations

import random
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste
import numpy as np


"""
Design analysis for the D2 variant
==================================

spec_variant:
  unchanged:
    background: fixed bg color 8
    frames:
      count: 2-6
      layout: tight 1-3 by 1-3 grid with no gaps
      size: 6-8 square
      structure: solid 1-pixel frame-color border, interior contains k isolated bg cells
      pairing_key: connected hole count k
    output:
      crop_to_frame_grid: true
      frame_grid_compacted: true
  changed:
    sprites:
      bbox_size: still frame_size-2 by frame_size-2
      old_form: nearly full square with a guaranteed solid 1-pixel border
      new_form: a single monochromatic connected component whose bbox is still square,
        but whose outer border is not solid because at least one edge notch opens to the bbox boundary
      holes: exactly k enclosed bg components inside the sprite component
      nesting: paste only sprite-color cells into the frame interior; bg cells stay frame-colored

  concrete_behavior_trace:
    - Original logic on a 5x5 sprite bbox counted every non-sprite-color cell as a hole.
      With a borderless sprite that has a notch touching the top edge, that notch would be
      incorrectly counted as a hole and later recolored as if it were enclosed.
    - New logic crops the sprite bbox, labels bg cells, and counts only bg components that
      do not touch the bbox boundary. The top-edge notch is exterior background, so it is
      ignored; only fully enclosed voids are matched to the frame's hole count.
    - Output construction also changes: instead of rewriting every non-sprite-color cell,
      the solver fills the frame square with frame color first and then transparently pastes
      only the sprite-color cells. This preserves both enclosed holes and exterior notches as
      frame-colored background inside the frame.

  complexity_calibration:
    - Sprites remain monochromatic and one-component, so object identity is still human-trackable.
    - Hole counts remain the only matching key and stay unique per sample.
    - Every sprite is forced to have at least one missing boundary cell; this makes the D2
      change visually obvious and prevents regression to the original border-locked sprite family.
    - The generator restricts N based on sprite hole capacity, because removing the border
      reduces the safe maximum number of distinct hole counts for small frame sizes.

  implicit_assumption_audit_against_original_main:
    - The old square-region heuristic assumed sprite bbox background cells were all holes.
      That is no longer true once notches touch the bbox boundary.
    - The old line `hole_mask = sprite_region != sprite_color` was harmless before because
      the solid border guaranteed all non-sprite-color cells were enclosed. In this variant
      it would overcount exterior notches.
    - The old recoloring step `content[content != sprite_color] = frame_color` also relied
      on the same assumption and would destroy the visible borderless geometry by treating
      exterior bg as if it were an interior hole.
    - Frame detection still safely relies on a perfectly solid square border; that part is
      intentionally unchanged.

  placement_legality:
    - Frames occupy a contiguous grid region.
    - Sprites are placed only in background cells outside that grid region, with padding.
    - Each sprite keeps a full-size square bbox by leaving at least one colored cell on every edge.
    - Validation checks enforce: one sprite component, exact hole count, non-solid boundary,
      and oracle equality with main().

  visual_observability:
    - Humans can still match by counting holes, but the input sprites now look like irregular
      blobs with bites taken out of their edges rather than miniature framed squares.
    - The output makes the rule visible because those same bites remain frame-colored after nesting.

  new_main_assumptions_checked_by_generator:
    - Every sprite is a single monochromatic connected component.
    - Every sprite bbox is exactly frame_size-2 square.
    - Hole counts are unique and match the frame hole counts bijectively.
    - Sprites do not overlap frames or each other.
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
    """Count bg components inside `region` that do not touch the bbox edge."""
    empty_mask = region != int(fg_color)
    lab, n = label_mask(empty_mask, connectivity=4)
    holes = 0
    h, w = region.shape
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        if not np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        ):
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
    _lab, n = label_mask(region == int(color), connectivity=4)
    return int(n) == 1


def _touches_all_bbox_sides(region: np.ndarray, color: int) -> bool:
    return bool(
        np.any(region[0, :] == int(color))
        and np.any(region[-1, :] == int(color))
        and np.any(region[:, 0] == int(color))
        and np.any(region[:, -1] == int(color))
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match frames to borderless sprite components by enclosed-hole count, then
    paste the matched sprite pattern into each frame interior.
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

    sprite_by_holes = {int(s["holes"]): s for s in sprites}
    assert len(sprite_by_holes) == len(sprites)
    assert frame_holes == set(sprite_by_holes.keys())

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for f in frame_grid:
        fr = int(f["out_r"])
        fc = int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        sprite = sprite_by_holes[int(f["holes"])]
        paste(out, np.array(sprite["content"], dtype=int), fr + 1, fc + 1, transparent=BG)

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

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
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
    sprite_by_k = {k: patch for k, patch in sprites}
    sprite_size = frame_size - 2
    for idx, (k, _patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * frame_size, ci * frame_size
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_colors[idx])
        paste(expected, sprite_by_k[int(k)], rr + 1, cc + 1, transparent=BG)
        assert sprite_by_k[int(k)].shape == (sprite_size, sprite_size)
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
                                "pack_rows": pack_rows,
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
    for i, k in enumerate(sprite_hole_perm):
        patch = _make_borderless_sprite(sprite_size, k, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((k, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(hole_counts):
        patch = _make_frame(frame_size, k, frame_colors[i])
        if patch is None:
            return None
        frames.append((k, patch))

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

    slot_positions = _packing_positions(
        n_items,
        sprite_size,
        int(config["pack_cols"]),
        SPRITE_PLACEMENT_PAD,
    )
    random.shuffle(slot_positions)
    for (_k, sprite_patch), (dr, dc) in zip(sprites, slot_positions):
        r = sprite_zone_r0 + dr
        c = sprite_zone_c0 + dc
        paste(canvas, sprite_patch, r, c, transparent=BG)

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
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
