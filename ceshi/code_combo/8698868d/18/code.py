# Combo variant of 8698868d (index 18)
# combo_spec: D8/0+D12/0
# Name: 多像素洞使用八连通计数

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for combo D8/0 + D12/0
======================================

a) Concrete behavior and execution trace
----------------------------------------
- Original family:
  - Large monochrome square frames form a tight grid.
  - Smaller monochrome square sprites are scattered elsewhere.
  - Frame holes were isolated single pixels.
  - Matching used the number of 4-connected hole components.
- Fusion change:
  - Frame holes may now be multi-pixel regions, not just isolated dots.
  - Hole counting for both frames and sprites switches to 8-connectivity.
- Small trace:
  - Suppose a frame interior contains bg cells at
      . # .
      # . .
      . . .
    where the two `#` cells touch only diagonally.
  - Original logic counts 2 holes because diagonal contact does not merge under
    4-connectivity, and the generator would never have allowed this in frames.
  - New logic counts 1 hole because the two cells are 8-connected, and the
    generator is allowed to use that same diagonal pair as one frame hole
    component. A matched sprite can use the same type of diagonal hole cluster.
  - The visible difference is therefore real in two places at once:
    1. frames can show larger hole blobs,
    2. diagonal hole pixels merge during matching.

b) Complexity calibration
-------------------------
- If every hole were arbitrary, the visual evidence would become noisy. The
  generator therefore keeps the original square/frame-grid layout and still
  uses the simple key "hole component count".
- The only added freedom is local hole shape. Counts remain consecutive
  integers, so the bijection is still easy to infer from a few examples.
- To keep the 8-connectivity change observable, the generator forces at least
  one matched frame/sprite pair to contain a diagonal-only merge where
  `count4 > count8`.

c) Implicit assumption review for main()
----------------------------------------
- Original assumption: frame holes are single pixels. This is no longer valid,
  so the solver must never infer hole count from raw hole area or isolated-dot
  placement. It must label connected components directly.
- Original code used `label_mask(..., connectivity=4)` for both frames and
  sprites. That now miscounts diagonal clusters, so both calls must use 8.
- Original size-based object detection is still valid because a square with a
  solid border remains one monochrome 4-connected component even when its holes
  become larger and more complex.
- New solver assumption introduced here:
  - each frame border is still perfectly solid and single-colored,
  - each sprite border is still perfectly solid and single-colored,
  - frame size is exactly sprite size + 2,
  - frame hole counts are unique across the grid.
  The generator constructs inputs to satisfy all four conditions.

d) Placement legality
---------------------
- Frames are placed as one tight non-overlapping grid of 1-3 rows by 1-3 cols.
- Sprites are placed on remaining bg cells using `random_free_location`, so they
  do not overlap each other or the frame grid and remain in-bounds.
- Hole shapes are generated entirely inside the interior area, so borders stay
  intact and object bounding boxes remain square.

e) Visual observability
-----------------------
- A human can see that some frame holes are no longer single dots because
  several bg cells touch each other inside a frame.
- Some of those clusters touch only diagonally, so different samples give
  repeated evidence that the pairing key is not plain hole area and not 4-way
  isolation.
- Output still uses the same structural cue as the original task: matched
  sprites are nested into their frames and hole pixels take the frame color.

f) Visual cleanliness
---------------------
- No extra markers are added. The scene still has only three clean layers:
  frame grid, scattered sprites, and bg holes.
- Because the layout stays tight and square-based, multi-pixel holes increase
  complexity without making the scene cluttered.

g) Change inferability and anti-degeneracy checks
-------------------------------------------------
- Passing `main(generate_input())` is not enough because samples could still
  accidentally behave like the original task.
- The generator therefore enforces two measurable properties:
  - at least one frame has a multi-pixel hole component,
  - at least one matched frame/sprite pair has `4-connected count >
    8-connected count`.
- The post-edit validation script below uses exactly those properties as the
  anti-degeneracy metric.
"""


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _count_hole_components(region: np.ndarray, solid_color: int, connectivity: int = 8) -> int:
    hole_mask = np.array(region, dtype=int) != int(solid_color)
    _lab, n = label_mask(hole_mask, connectivity=connectivity)
    return int(n)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered sprites with the square frames that have the same number of
    8-connected hole regions, then pack the matched sprites into the frame grid.

    Input structure:
    - Background is azure (8). Large monochrome square frames appear in a tight
      grid, while smaller monochrome square sprites are placed elsewhere.
    - Both frames and sprites may contain bg-colored holes inside their solid
      borders. Frame holes may be single pixels or larger connected blobs.

    Rule:
    - Detect the larger squares as frames and the smaller squares as sprites.
    - For each frame interior and each sprite, count bg-hole components using
      8-connectivity, so diagonal touching hole pixels belong to one hole.
    - Match each frame with the unique sprite that has the same 8-connected hole
      count, place that sprite inside the frame border, and recolor the sprite's
      hole pixels to the frame color.

    Key insight: The decisive feature is the number of 8-connected hole
    components. Hole shape may be multi-pixel and diagonal links merge holes.
    """
    g = np.array(grid, dtype=int)
    BG = 8

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * 0.4):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames and sprites_raw

    frame_positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = (max_r - min_r) // frame_size + 1
    n_cols = (max_c - min_c) // frame_size + 1

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = min_r + ri * frame_size
            inp_c = min_c + ci * frame_size
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.tolist())).most_common(1)[0][0])
            inner = region[1:-1, 1:-1]
            num_holes = _count_hole_components(inner, frame_color, connectivity=8)
            frame_grid.append(
                {
                    "color": frame_color,
                    "out_r": ri * frame_size,
                    "out_c": ci * frame_size,
                    "holes": num_holes,
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        sprites.append(
            {
                "color": sprite_color,
                "holes": _count_hole_components(sprite_region, sprite_color, connectivity=8),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for idx, sprite in enumerate(sprites):
            if idx not in used_sprites and int(frame["holes"]) == int(sprite["holes"]):
                matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprite
                used_sprites.add(idx)
                break

    assert len(matches) == len(frame_grid)

    out_h = n_rows * frame_size
    out_w = n_cols * frame_size
    out = np.full((out_h, out_w), BG, dtype=int)

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


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


BG = 8
FRAME_SIZE_OPTIONS = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
GEN_RETRIES = 400
MASK_RETRIES = 250

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        _n = _r * _c
        if N_RANGE[0] <= _n <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_n, []).append((_r, _c))

_COMPONENT_TEMPLATES = [
    {"cells": ((0, 0),), "multi": False, "diag": False},
    {"cells": ((0, 0), (0, 1)), "multi": True, "diag": False},
    {"cells": ((0, 0), (1, 0)), "multi": True, "diag": False},
    {"cells": ((0, 0), (1, 1)), "multi": True, "diag": True},
    {"cells": ((0, 1), (1, 0)), "multi": True, "diag": True},
    {"cells": ((0, 0), (0, 1), (1, 0)), "multi": True, "diag": False},
    {"cells": ((0, 0), (1, 1), (2, 2)), "multi": True, "diag": True},
]


def _max_8_components(inner_side: int) -> int:
    return ((inner_side + 1) // 2) ** 2


def _component_counts(mask: np.ndarray) -> tuple[int, int]:
    _lab4, n4 = label_mask(mask, connectivity=4)
    _lab8, n8 = label_mask(mask, connectivity=8)
    return int(n4), int(n8)


def _touches_existing(occupied: np.ndarray, r: int, c: int) -> bool:
    side = occupied.shape[0]
    r0 = max(0, r - 1)
    r1 = min(side, r + 2)
    c0 = max(0, c - 1)
    c1 = min(side, c + 2)
    return bool(occupied[r0:r1, c0:c1].any())


def _place_template(mask: np.ndarray, occupied: np.ndarray, template: dict) -> bool:
    side = mask.shape[0]
    cells = [tuple(map(int, cell)) for cell in template["cells"]]
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    positions = [(r0, c0) for r0 in range(side - max_r) for c0 in range(side - max_c)]
    random.shuffle(positions)

    for r0, c0 in positions:
        coords = [(r0 + dr, c0 + dc) for dr, dc in cells]
        if any(mask[r, c] for r, c in coords):
            continue
        if any(_touches_existing(occupied, r, c) for r, c in coords):
            continue
        for r, c in coords:
            mask[r, c] = True
            occupied[r, c] = True
        return True
    return False


def _build_hole_mask(
    side: int,
    hole_count: int,
    *,
    require_multi: bool = False,
    require_diag: bool = False,
) -> np.ndarray | None:
    if hole_count == 0:
        if require_multi or require_diag:
            return None
        return np.zeros((side, side), dtype=bool)

    fitting_templates = [
        tpl
        for tpl in _COMPONENT_TEMPLATES
        if max(r for r, _ in tpl["cells"]) < side and max(c for _, c in tpl["cells"]) < side
    ]
    if not fitting_templates:
        return None

    diag_templates = [tpl for tpl in fitting_templates if tpl["diag"]]
    multi_templates = [tpl for tpl in fitting_templates if tpl["multi"]]
    if require_diag and not diag_templates:
        return None
    if require_multi and not multi_templates:
        return None

    for _ in range(MASK_RETRIES):
        mask = np.zeros((side, side), dtype=bool)
        occupied = np.zeros((side, side), dtype=bool)
        special_diag_idx = random.randrange(hole_count) if require_diag else None
        special_multi_idx = None
        if require_multi and not require_diag:
            special_multi_idx = random.randrange(hole_count)

        ok = True
        for idx in range(hole_count):
            if idx == special_diag_idx:
                candidates = diag_templates[:]
            elif idx == special_multi_idx:
                candidates = multi_templates[:]
            else:
                candidates = fitting_templates[:]
                random.shuffle(candidates)
                if random.random() < 0.7:
                    candidates = sorted(candidates, key=lambda tpl: (not tpl["multi"], not tpl["diag"]))

            random.shuffle(candidates)
            placed = False
            for tpl in candidates:
                trial_mask = mask.copy()
                trial_occ = occupied.copy()
                if _place_template(trial_mask, trial_occ, tpl):
                    mask = trial_mask
                    occupied = trial_occ
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        n4, n8 = _component_counts(mask)
        if n8 != hole_count:
            continue
        if require_multi and int(mask.sum()) <= hole_count:
            continue
        if require_diag and n4 <= n8:
            continue
        return mask

    return None


def _embed_holes(size: int, hole_mask: np.ndarray, solid_color: int) -> np.ndarray:
    patch = np.full((size, size), solid_color, dtype=int)
    inner_side = size - 2
    patch[1 : 1 + inner_side, 1 : 1 + inner_side][hole_mask] = BG
    return patch


def _build_sprite(size: int, hole_count: int, sprite_color: int, *, require_diag: bool = False) -> np.ndarray | None:
    inner_side = size - 2
    if hole_count > _max_8_components(inner_side):
        return None
    hole_mask = _build_hole_mask(inner_side, hole_count, require_multi=(hole_count > 0), require_diag=require_diag)
    if hole_mask is None:
        return None
    sprite = _embed_holes(size, hole_mask, sprite_color)
    _, n8 = _component_counts(hole_mask)
    if n8 != hole_count:
        return None
    return sprite


def _build_frame(frame_size: int, hole_count: int, frame_color: int, *, require_diag: bool = False) -> np.ndarray | None:
    inner_side = frame_size - 2
    if hole_count > _max_8_components(inner_side):
        return None
    hole_mask = _build_hole_mask(inner_side, hole_count, require_multi=(hole_count > 0), require_diag=require_diag)
    if hole_mask is None:
        return None
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    frame[1:-1, 1:-1][hole_mask] = BG
    _, n8 = _component_counts(hole_mask)
    if n8 != hole_count:
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_count = {count: patch for count, patch in sprites}
    for idx, (count, _frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = ri * frame_size
        cc = ci * frame_size
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = np.array(sprite_by_count[count], dtype=int).copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _sample_dimensions() -> tuple[int, int, int]:
    frame_size = random.choice(FRAME_SIZE_OPTIONS)
    sprite_inner_side = frame_size - 4
    max_components = _max_8_components(sprite_inner_side)
    valid_ns = [n for n in _LAYOUTS_BY_N if n <= max_components + 1]
    if not valid_ns:
        raise RuntimeError("No valid N for selected frame size")
    n = random.choice(valid_ns)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    return frame_size, rows_grid, cols_grid


def _gen_attempt() -> np.ndarray | None:
    frame_size, rows_grid, cols_grid = _sample_dimensions()
    n = rows_grid * cols_grid
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None

    h = random.randint(min_h, CANVAS_MAX)
    w = random.randint(min_w, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n)
    sprite_colors = random.sample(available, n)

    hole_counts = list(range(n))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    positive_counts = [count for count in hole_counts if count > 0]
    special_count = random.choice(positive_counts) if positive_counts else None

    sprites: list[tuple[int, np.ndarray]] = []
    for idx, count in enumerate(sprite_hole_perm):
        patch = _build_sprite(
            sprite_size,
            count,
            sprite_colors[idx],
            require_diag=(count == special_count),
        )
        if patch is None:
            return None
        sprites.append((count, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for idx, count in enumerate(hole_counts):
        patch = _build_frame(
            frame_size,
            count,
            frame_colors[idx],
            require_diag=(count == special_count),
        )
        if patch is None:
            return None
        frames.append((count, patch))

    frame_with_multi = any(np.count_nonzero(patch == BG) > count for count, patch in frames if count > 0)
    if not frame_with_multi:
        return None

    if special_count is not None:
        frame_patch = next(patch for count, patch in frames if count == special_count)
        sprite_patch = next(patch for count, patch in sprites if count == special_count)
        frame_mask = frame_patch[1:-1, 1:-1] == BG
        sprite_mask = sprite_patch[1:-1, 1:-1] == BG
        frame_n4, frame_n8 = _component_counts(frame_mask)
        sprite_n4, sprite_n8 = _component_counts(sprite_mask)
        if not (frame_n4 > frame_n8 and sprite_n4 > sprite_n8):
            return None

    canvas = np.full((h, w), BG, dtype=int)
    r0 = random.randint(0, h - grid_pixel_h)
    c0 = random.randint(0, w - grid_pixel_w)
    for idx, (_count, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _count, sprite_patch in sprites:
        try:
            sr, sc = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, sr, sc, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, frame_colors, rows_grid, cols_grid, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
