# Combo variant of 8698868d (index 30)
# combo_spec: D2/1+D4/0+D5/3
# Name: 无边框精灵+反转着色+纯内容输出

from __future__ import annotations

# Design analysis for combo D2/1 + D4/0 + D5/3
# a) Concrete behavior and trace:
#    The source family still has two object classes:
#      - large square frames with solid 1-pixel borders and k isolated bg holes inside;
#      - smaller scattered sprites whose bbox is `frame_size-2`.
#    The fused change affects the sprite interpretation, the recolor direction, and the output crop.
#    Small trace with `frame_size=6`, `sprite_size=4`, frame color 4, sprite color 2:
#      Borderless input sprite bbox:
#        8 2 2 2
#        2 2 8 2
#        2 2 2 2
#        2 8 2 2
#      The top-left 8 is a boundary notch, the center-right 8 is an enclosed hole, and the
#      bottom-middle 8 is another boundary notch. Original logic would wrongly count all three
#      non-2 pixels as holes, recolor them, and then keep the 6x6 frame border in output.
#      The fused logic must:
#        1. count only enclosed bg components inside the sprite bbox, so the key here is 1;
#        2. match that sprite to the frame whose interior has 1 bg component;
#        3. recolor only sprite-colored pixels to frame color 4, leaving bg cells as bg;
#        4. output only the 4x4 nested content, not the surrounding frame border.
#      Final slot becomes:
#        8 4 4 4
#        4 4 8 4
#        4 4 4 4
#        4 8 4 4
#    The visible result is therefore a frame-colored negative of the borderless sprite shape.
# b) Complexity calibration:
#    The generator keeps the original single matching key family: consecutive hole counts with one
#    unique value per frame and per sprite. That avoids ambiguity.
#    To keep the D2 change visible, every sprite is forced to have at least one boundary notch and
#    to stay connected while still touching all four bbox sides. To keep D4 visible, N>=2 ensures
#    at least one nonzero hole-count sprite in every sample. To keep D5 visible, output dimensions
#    always shrink from frame-grid size to content-grid size.
# c) Implicit assumptions audit against the original solver:
#    - The old sprite line `hole_mask = sprite_region != sprite_color` assumed every non-sprite
#      pixel was an enclosed hole. D2 breaks that because boundary notches are exterior bg.
#    - The old recolor step `content[content != sprite_color] = frame_color` assumed holes should
#      disappear. D4 breaks that because bg holes and boundary notches must remain bg.
#    - The old output builder assumed slot size `frame_size` and a visible border. D5 breaks that
#      because the final grid consists only of `sprite_size x sprite_size` content blocks.
#    - Frame extraction from solid-border squares still holds; the combo intentionally leaves frame
#      geometry unchanged so the solver can still infer the tight grid from the large squares.
# d) Placement legality:
#    Frames are placed as one tight rows-by-cols block. Sprites are packed in a disjoint zone either
#    below or to the right of the frame grid, with at least one cell of padding between sprite bboxes.
#    Transparent pasting uses bg for the sprite voids, so sprites never overlap each other or the
#    frame grid, and all objects stay inside the 15..30 canvas.
# e) Visual observability:
#    Across 2-4 examples, a human sees three repeated cues:
#      - sprites are no longer miniature framed squares but connected shapes with bites on edges;
#      - only enclosed voids matter for matching, because edge bites do not line up with frame hole counts;
#      - output is smaller and shows only frame-colored mass with bg cutouts.
#    Those cues are structural and do not depend on numeric color ids.
# f) Visual cleanliness:
#    The scene remains sparse: one compact frame grid and one compact sprite zone. Output is also
#    clean because each slot is just one recolored content patch with no border overdraw.
# g) Change inferability:
#    Color change is local and repeated: sprite solid cells take the matched frame color, while every
#    sprite bg cell stays background. Because the output is cropped, the only remaining geometry is
#    the negative mask itself, which makes the inversion visually explicit.
# h) New solver assumptions and generator guarantees:
#    - Sprite candidates must be connected monochrome components with square bbox `frame_size-2`,
#      must touch all four bbox sides, and must not have a solid border. The generator enforces this.
#    - Matching assumes enclosed-hole counts are unique on both sides. The generator uses one copy of
#      each consecutive count.
#    - Output construction assumes sprites use only sprite color and bg. The generator never inserts a
#      third color into a sprite patch.

from collections import Counter
from pathlib import Path
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

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste


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
for _rows in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _cols in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        n_items = _rows * _cols
        if N_RANGE[0] <= n_items <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(n_items, []).append((_rows, _cols))


def _count_enclosed_holes(region: np.ndarray, fg_color: int) -> int:
    empty_mask = region != int(fg_color)
    labels, n_comp = label_mask(empty_mask, connectivity=4)
    holes = 0
    height, width = region.shape
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labels == comp_id)
        if coords.size == 0:
            continue
        touches_edge = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == height - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == width - 1)
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
    _labels, n_comp = label_mask(region == int(color), connectivity=4)
    return int(n_comp) == 1


def _touches_all_bbox_sides(region: np.ndarray, color: int) -> bool:
    return bool(
        np.any(region[0, :] == int(color))
        and np.any(region[-1, :] == int(color))
        and np.any(region[:, 0] == int(color))
        and np.any(region[:, -1] == int(color))
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match borderless sprites to frames by enclosed-hole count, then emit only the
    recolored nested content as a compact grid of negative-mask patches.

    Input structure:
    - Background is fixed at azure (8).
    - Large hollow squares with solid borders form one tight frame grid.
    - Smaller same-size square bboxes elsewhere contain single-color connected sprites
      without solid borders; their bg gaps include both boundary notches and enclosed holes.

    Rule:
    - Recover the frame grid from the large solid-border squares.
    - Count frame holes as 4-connected bg components enclosed by each frame border.
    - Count sprite holes as 4-connected bg components inside the sprite bbox that do not
      touch the bbox boundary.
    - Pair each frame with the sprite having the same enclosed-hole count.
    - Recolor only the sprite's solid pixels to the matched frame color, keep sprite bg
      cells as bg, and output only these recolored `sprite_size x sprite_size` contents.

    Key insight:
    Boundary gaps in the borderless sprites are not matchable holes. The real invariant is
    the number of enclosed voids, and the final picture is the cropped negative of each
    matched sprite drawn in the corresponding frame color.
    """
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    frame_candidates: list[tuple[int, dict]] = []
    for comp in comps:
        box = comp["bbox"]
        height = int(bbox_h(box))
        width = int(bbox_w(box))
        if height != width or not (6 <= height <= 8):
            continue
        region = crop_bbox(g, box)
        color = int(comp["color"])
        if _has_solid_border(region, color):
            frame_candidates.append((height, comp))

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

    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)
    assert n_rows * n_cols == len(frames)

    frame_grid: list[dict] = []
    frame_holes: set[int] = set()
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            inp_r = int(min_r + row_idx * frame_size)
            inp_c = int(min_c + col_idx * frame_size)
            comp = frame_lookup[(inp_r, inp_c)]
            region = crop_bbox(g, comp["bbox"])
            color = int(comp["color"])
            holes = int(_count_enclosed_holes(region, color))
            frame_holes.add(holes)
            frame_grid.append(
                {
                    "color": color,
                    "slot_r": int(row_idx),
                    "slot_c": int(col_idx),
                    "holes": holes,
                }
            )

    sprites: list[dict] = []
    for comp in comps:
        if tuple(map(int, comp["bbox"])) in frame_bboxes:
            continue
        box = comp["bbox"]
        height = int(bbox_h(box))
        width = int(bbox_w(box))
        if height != sprite_size or width != sprite_size:
            continue
        region = crop_bbox(g, box)
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

    sprite_by_holes = {int(sprite["holes"]): sprite for sprite in sprites}
    assert len(sprite_by_holes) == len(sprites)
    assert frame_holes == set(sprite_by_holes.keys())

    out_h = int(n_rows * sprite_size)
    out_w = int(n_cols * sprite_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        sprite = sprite_by_holes[int(frame["holes"])]
        sprite_color = int(sprite["color"])
        frame_color = int(frame["color"])
        content = np.full((sprite_size, sprite_size), int(BG), dtype=int)
        sprite_patch = np.array(sprite["content"], dtype=int)
        content[sprite_patch == int(sprite_color)] = int(frame_color)

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

    components: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = int(BG)

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_idx, comp in enumerate(components):
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
                        (nr + ddr, nc + ddc) in other
                        for other_idx, other in enumerate(components)
                        if other_idx != comp_idx
                        for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = int(BG)

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
            out[0:depth, c0 : c0 + span] = int(BG)
        elif side == "bottom":
            c0 = random.randint(0, size - span)
            out[size - depth : size, c0 : c0 + span] = int(BG)
        elif side == "left":
            r0 = random.randint(0, size - span)
            out[r0 : r0 + span, 0:depth] = int(BG)
        else:
            r0 = random.randint(0, size - span)
            out[r0 : r0 + span, size - depth : size] = int(BG)
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
    expected = np.full((rows_grid * sprite_size, cols_grid * sprite_size), int(BG), dtype=int)
    sprite_by_holes = {int(hole_count): patch for hole_count, patch in sprites}
    for idx, (hole_count, _frame_patch) in enumerate(frames):
        rr = int((idx // cols_grid) * sprite_size)
        cc = int((idx % cols_grid) * sprite_size)
        frame_color = int(frame_colors[idx])
        sprite_patch = sprite_by_holes[int(hole_count)]
        content = np.full((sprite_size, sprite_size), int(BG), dtype=int)
        content[sprite_patch != int(BG)] = int(frame_color)
        expected[rr : rr + sprite_size, cc : cc + sprite_size] = content
    return expected


def _packing_positions(count: int, item_size: int, pack_cols: int, pad: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    step = item_size + pad
    for idx in range(count):
        rr = (idx // pack_cols) * step
        cc = (idx % pack_cols) * step
        positions.append((int(rr), int(cc)))
    return positions


def _sample_sprite_colors(
    available: list[int],
    sprite_hole_order: list[int],
    frame_color_by_holes: dict[int, int],
) -> list[int] | None:
    for _ in range(100):
        sprite_colors = random.sample(available, len(sprite_hole_order))
        if all(
            int(sprite_colors[idx]) != int(frame_color_by_holes[int(hole_count)])
            for idx, hole_count in enumerate(sprite_hole_order)
        ):
            return sprite_colors
    return None


_FEASIBLE_BY_N: dict[int, list[dict]] = {}
for _frame_size in FRAME_SIZE_RANGE:
    _sprite_size = _frame_size - 2
    _sprite_inner = _sprite_size - 2
    _max_sprite_holes = (_sprite_inner * _sprite_inner + 1) // 2
    for _n_items, _layouts in _LAYOUTS_BY_N.items():
        if _n_items > _max_sprite_holes + 1:
            continue
        for _rows_grid, _cols_grid in _layouts:
            _grid_pixel_h = _rows_grid * _frame_size
            _grid_pixel_w = _cols_grid * _frame_size
            for _side in ["bottom", "right"]:
                for _pack_cols in range(1, _n_items + 1):
                    _pack_rows = (_n_items + _pack_cols - 1) // _pack_cols
                    _zone_h = _pack_rows * _sprite_size + (_pack_rows - 1) * SPRITE_PLACEMENT_PAD
                    _zone_w = _pack_cols * _sprite_size + (_pack_cols - 1) * SPRITE_PLACEMENT_PAD
                    if _side == "bottom":
                        _min_h = max(CANVAS_MIN, _grid_pixel_h + SPRITE_PLACEMENT_PAD + _zone_h)
                        _min_w = max(CANVAS_MIN, max(_grid_pixel_w, _zone_w))
                    else:
                        _min_h = max(CANVAS_MIN, max(_grid_pixel_h, _zone_h))
                        _min_w = max(CANVAS_MIN, _grid_pixel_w + SPRITE_PLACEMENT_PAD + _zone_w)
                    if _min_h > CANVAS_MAX or _min_w > CANVAS_MAX:
                        continue
                    _FEASIBLE_BY_N.setdefault(_n_items, []).append(
                        {
                            "frame_size": _frame_size,
                            "sprite_size": _sprite_size,
                            "rows_grid": _rows_grid,
                            "cols_grid": _cols_grid,
                            "grid_pixel_h": _grid_pixel_h,
                            "grid_pixel_w": _grid_pixel_w,
                            "side": _side,
                            "pack_cols": _pack_cols,
                            "zone_h": _zone_h,
                            "zone_w": _zone_w,
                            "min_h": _min_h,
                            "min_w": _min_w,
                            "max_sprite_holes": _max_sprite_holes,
                        }
                    )


def _gen_attempt() -> np.ndarray | None:
    if not _FEASIBLE_BY_N:
        return None

    weighted_ns = [n_items for n_items in _FEASIBLE_BY_N for _ in range(n_items)]
    n_items = random.choice(weighted_ns)
    config = random.choice(_FEASIBLE_BY_N[n_items])

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

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_items)

    max_sprite_holes = int(config["max_sprite_holes"])
    max_start = max(0, min(HOLE_START_MAX, max_sprite_holes - (n_items - 1)))
    start = random.randint(0, max_start)
    hole_counts = list(range(start, start + n_items))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)
    frame_color_by_holes = {
        int(hole_count): int(frame_colors[idx]) for idx, hole_count in enumerate(hole_counts)
    }
    sprite_colors = _sample_sprite_colors(available, sprite_hole_perm, frame_color_by_holes)
    if sprite_colors is None:
        return None

    sprites: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(sprite_hole_perm):
        patch = _make_borderless_sprite(sprite_size, hole_count, sprite_colors[idx])
        if patch is None:
            return None
        sprites.append((hole_count, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_size, hole_count, frame_colors[idx])
        if patch is None:
            return None
        frames.append((hole_count, patch))

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
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

    for idx, (_hole_count, frame_patch) in enumerate(frames):
        rr = int(r0 + (idx // cols_grid) * frame_size)
        cc = int(c0 + (idx % cols_grid) * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    slot_positions = _packing_positions(n_items, sprite_size, int(config["pack_cols"]), SPRITE_PLACEMENT_PAD)
    random.shuffle(slot_positions)
    for (_hole_count, sprite_patch), (dr, dc) in zip(sprites, slot_positions):
        paste(canvas, sprite_patch, sprite_zone_r0 + dr, sprite_zone_c0 + dc, transparent=BG)

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
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
