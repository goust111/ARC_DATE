# Combo variant of 8698868d (index 34)
# combo_spec: D2/1+D13/0
# Name: 变化背景色+无边框精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, label_mask, paste


"""
Design analysis for combo D2/1 + D13/0
======================================

a) Concrete behavior and trace
   The original family assumed two things at once:
   - background is always color 8
   - each sprite is a nearly solid square, so every non-sprite-color pixel inside its bbox is a true hole

   The fused variant breaks both assumptions together. A typical input now has:
   - a canvas whose dominant background may be any color, say 3
   - large square frames with solid borders, still arranged as a tight grid
   - smaller monochrome sprites elsewhere, but each sprite is borderless: some bg-colored gaps open
     directly to the sprite bbox edge, while other bg-colored regions may be fully enclosed holes

   Example trace on one sprite bbox:
   - original logic on a 5x5 sprite with bg=8 would do `hole_mask = region != sprite_color`
     and count every bg region as a hole
   - in the fused variant the true bg might be 3, and a top-edge notch touching the bbox boundary
     is exterior empty space, not a hole
   - therefore the solver must first infer bg=3 from color frequency, then count only those bg
     components that do not touch the bbox border
   - output construction changes accordingly: fill each frame square with the frame color first, then
     transparently paste only sprite-colored pixels; both enclosed holes and boundary notches remain
     frame-colored in the nested result

b) Complexity calibration
   The pairing key stays single-dimensional: enclosed-hole count. This keeps the task inferable.
   The difficulty increase comes from the front end of perception:
   - human/solver must identify the background by frequency instead of by fixed color id
   - human/solver must identify sprites by connected monochrome shapes rather than by solid borders
   - human/solver must distinguish enclosed holes from open notches on the sprite boundary
   To prevent accidental regression to the old family, the generator forces every sprite to have a
   visibly non-solid boundary and always chooses bg != 8.

c) Implicit assumption audit against the original main()
   - `BG = 8` is invalid now; all bg uses must become dynamic
   - `find_components(..., bg=8)` would wrongly treat the true background as foreground when bg != 8
   - `hole_mask = sprite_region != sprite_color` used to work because the solid border guaranteed every
     non-sprite pixel in the bbox was enclosed; borderless sprites break that assumption
   - `content[content != sprite_color] = frame_color` would wrongly erase exterior notches by treating
     them like holes; transparent pasting fixes this
   - frame detection still safely depends on the unchanged invariant that frames are the only solid-border
     monochrome squares of the largest square size in the sample

d) Placement legality
   Frames are placed as a tight grid with no gaps. Sprites are packed in a separate bottom or right zone,
   so they never overlap frames or each other and always stay in-bounds. Every sprite is generated as a
   single 4-connected monochrome component whose bbox remains exactly `frame_size - 2` square.

e) Visual observability
   Across a few examples a human can still see the main rule: count how many holes each loose sprite has
   and insert it into the matching frame. The fused changes are also visible:
   - the canvas background color varies across samples
   - sprites are irregular borderless blobs instead of miniature framed squares
   - once inserted, their bites/notches remain visible as frame-colored gaps

f) Visual cleanliness
   No extra markers are introduced. Frames stay in one compact block; sprites live in one separate block.
   This avoids clutter and makes the object classes visually separable despite the added detection burden.

g) Change inferability and anti-degeneracy
   The only color rule added is background inference, and it is supported structurally by large-area fill
   plus same-colored holes/notches. For non-8 backgrounds, the generator reserves color 8 for at least one
   sprite and forbids it on frames; this makes the old hardcoded-bg solver fail rather than accidentally pass.
   A useful degeneracy metric is:
   - bg_not_8_rate: fraction of generated samples with inferred background not equal to 8
   - borderless_rate: fraction of sprites whose boundary is not solid (target: all)
   - old_solver_break_rate: fraction of generated samples where the original solver errors or disagrees

h) New main() assumptions guaranteed by the generator
   - bg is the unique most common color in the input
   - each sprite is monochromatic, 4-connected, borderless, and touches all four sides of its bbox
   - each sprite bbox size is exactly `frame_size - 2`
   - each frame is a solid-border square of size 6-8
   - frame hole counts and sprite enclosed-hole counts form the same unique set
"""


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
    """Count non-fg components that do not touch the cropped bbox boundary."""
    empty_mask = region != int(fg_color)
    lab, n = label_mask(empty_mask, connectivity=4)
    holes = 0
    h, w = map(int, region.shape)
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        touches_edge = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        )
        if not bool(touches_edge):
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
    Pair borderless sprites with frames by enclosed-hole count, then nest them in a compact frame grid.

    Input structure:
    - The background color is not fixed; it must be inferred as the most common color in the whole input.
    - Large monochrome square frames keep a solid one-pixel border and form a tight rectangular grid.
    - Smaller monochrome sprites are scattered outside that grid; each sprite is a connected borderless
      pattern whose bbox is square and may contain both open edge notches and enclosed bg holes.

    Rule:
    - Infer the current background first, then detect the solid-border frame squares.
    - Count hole components inside each frame and count only enclosed bg components inside each sprite bbox.
    - Treat sprite bg regions touching the bbox edge as exterior gaps, not as match-defining holes.
    - Match frame and sprite by equal hole count, then paste the sprite-color pixels into the frame interior.

    Key insight:
    - The correspondence is still determined by hole count, but solving now depends on two extra perception
      steps: discovering the true background color and ignoring bbox-touching sprite gaps that are not holes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

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

    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)
    assert n_rows * n_cols == len(frames)

    frame_grid: list[dict[str, int]] = []
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

    sprites: list[dict[str, int | np.ndarray]] = []
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
        if not _touches_all_bbox_sides(region, color):
            continue
        if not _component_is_connected(region, color):
            continue
        holes = int(_count_enclosed_holes(region, color))
        sprites.append(
            {
                "color": color,
                "holes": holes,
                "content": region.copy(),
            }
        )

    sprite_by_holes = {int(s["holes"]): s for s in sprites}
    assert len(sprite_by_holes) == len(sprites)
    assert frame_holes == set(sprite_by_holes.keys())

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)
        sprite = sprite_by_holes[int(frame["holes"])]
        paste(out, np.array(sprite["content"], dtype=int), fr + 1, fc + 1, transparent=bg)

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


def _make_full_square_with_holes(size: int, hole_count: int, sprite_color: int, bg: int) -> np.ndarray | None:
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
        sprite[r, c] = int(bg)

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
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
            sprite[cell[0], cell[1]] = int(bg)

    if _count_enclosed_holes(sprite, sprite_color) != int(hole_count):
        return None
    return sprite


def _carve_random_notches(sprite: np.ndarray, sprite_color: int, bg: int) -> np.ndarray:
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
            out[0:depth, c0 : c0 + span] = int(bg)
        elif side == "bottom":
            c0 = random.randint(0, size - span)
            out[size - depth : size, c0 : c0 + span] = int(bg)
        elif side == "left":
            r0 = random.randint(0, size - span)
            out[r0 : r0 + span, 0:depth] = int(bg)
        else:
            r0 = random.randint(0, size - span)
            out[r0 : r0 + span, size - depth : size] = int(bg)
    return out


def _make_borderless_sprite(size: int, hole_count: int, sprite_color: int, bg: int) -> np.ndarray | None:
    for _ in range(SPRITE_RETRIES):
        base = _make_full_square_with_holes(size, hole_count, sprite_color, bg)
        if base is None:
            continue
        candidate = _carve_random_notches(base, sprite_color, bg)
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


def _make_frame(frame_size: int, hole_count: int, frame_color: int, bg: int) -> np.ndarray | None:
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
        frame[r, c] = int(bg)

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
    bg: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), bg, dtype=int)
    sprite_by_k = {k: patch for k, patch in sprites}
    for idx, (k, _patch) in enumerate(frames):
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = int(ri * frame_size)
        cc = int(ci * frame_size)
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_colors[idx])
        paste(expected, sprite_by_k[int(k)], rr + 1, cc + 1, transparent=bg)
    return expected


def _packing_positions(count: int, item_size: int, pack_cols: int, pad: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    step = int(item_size + pad)
    for idx in range(count):
        rr = (idx // pack_cols) * step
        cc = (idx % pack_cols) * step
        positions.append((int(rr), int(cc)))
    return positions


def _bg_is_most_common(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flatten()))
    return int(counts.most_common(1)[0][0]) == int(bg)


def _gen_attempt() -> np.ndarray | None:
    feasible_by_n: dict[int, list[dict[str, int | str]]] = {}
    for frame_size in FRAME_SIZE_RANGE:
        sprite_size = int(frame_size - 2)
        sprite_inner = int(sprite_size - 2)
        max_sprite_holes = (sprite_inner * sprite_inner + 1) // 2
        for n_items, layouts in _LAYOUTS_BY_N.items():
            if n_items > max_sprite_holes + 1:
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
                                "side": str(side),
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
    n_items = int(random.choice(weighted_ns))
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
    # Keep the dynamic-background variant explicit in every sample instead of occasionally
    # degenerating back to the original azure background family.
    bg = random.choice([c for c in range(10) if c != 8])

    available = [c for c in range(10) if c != bg]
    if bg != 8:
        frame_colors = random.sample([c for c in available if c != 8], n_items)
        sprite_colors = [8, *random.sample([c for c in available if c != 8], n_items - 1)]
        random.shuffle(sprite_colors)
    else:
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
    for i, holes in enumerate(sprite_hole_perm):
        patch = _make_borderless_sprite(sprite_size, int(holes), int(sprite_colors[i]), bg)
        if patch is None:
            return None
        sprites.append((int(holes), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, holes in enumerate(hole_counts):
        patch = _make_frame(frame_size, int(holes), int(frame_colors[i]), bg)
        if patch is None:
            return None
        frames.append((int(holes), patch))

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    if str(config["side"]) == "bottom":
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
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    slot_positions = _packing_positions(n_items, sprite_size, int(config["pack_cols"]), SPRITE_PLACEMENT_PAD)
    random.shuffle(slot_positions)
    for (_holes, sprite_patch), (dr, dc) in zip(sprites, slot_positions):
        paste(canvas, sprite_patch, sprite_zone_r0 + dr, sprite_zone_c0 + dc, transparent=bg)

    if not _bg_is_most_common(canvas, bg):
        return None

    for _holes, sprite_patch in sprites:
        color = int(next(iter(set(map(int, sprite_patch.flatten())) - {bg})))
        if _has_solid_border(sprite_patch, color):
            return None
        if not _touches_all_bbox_sides(sprite_patch, color):
            return None

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, frame_colors, rows_grid, cols_grid, frame_size, bg)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
