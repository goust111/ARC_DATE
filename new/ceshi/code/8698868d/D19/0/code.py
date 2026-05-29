# Variant of 8698868d
# Dimension: D19 — 输出中框架的排列顺序 (meta)
# Change: 输出框架按输入中的网格位置排列（行优先顺序保持不变）... → 按洞计数排序输出

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location

# Design analysis for the D19 variant
# -----------------------------------
# a) Concrete behavior and trace
#    - Input still contains one tight grid of large square frames plus the same number of smaller
#      square sprites scattered elsewhere on azure background 8.
#    - Each frame interior contains k isolated azure holes; each sprite contains k connected hole
#      components in its own color patch. Matching is still bijective by k.
#    - Original task: if input frame-grid row-major hole counts are [2, 0, 1], output slots stay
#      in that same row-major order, so the left/middle/right output frames use counts [2, 0, 1].
#    - Variant: after matching sprites, reorder the output slots by frame hole count, so the same
#      example becomes [0, 1, 2] left-to-right. Frame colors move with their matched sprites.
#    - Only the "which frame goes into which output slot" stage changes; hole counting, matching,
#      recoloring, and tight output cropping stay unchanged.
# b) Complexity calibration
#    - Hole counts remain unique consecutive integers. This keeps matching unambiguous while making
#      the sorting key human-visible.
#    - Generator forces the input frame order to differ from sorted order, preventing degeneration
#      back to the original task where output order equals input order.
# c) Hidden-assumption audit of the original main()
#    - Original frame_grid append order is row-major over the detected grid. That silently encoded
#      the old D19 behavior. In this variant, that order is valid for reading input, but not for
#      writing output.
#    - Original matches dict used output coordinates derived from input positions. That is harmless
#      only when output slots equal input slots. Here we must match by hole count instead, because
#      frames move to new output slots.
#    - Frame-size detection, hole counting, and sprite extraction do not depend on slot order and
#      remain valid.
# d) Placement legality
#    - Frames are still placed as a gapless grid on the input canvas, sprites remain outside the
#      grid via random_free_location, and all colors are non-background. Solver assumptions about
#      component detection and grid reconstruction therefore still hold.
# e) Visual observability
#    - Humans can count frame holes directly in the input grid, then see that the output frames are
#      reordered from few holes to many holes. Because frame colors are distinct, movement is easy
#      to see even when sprites have similar silhouettes.
#    - The generator rejects samples where the input row-major hole sequence is already sorted, so
#      the new ordering rule is exercised in the vast majority of examples.
# f) Visual cleanliness
#    - No extra markers are added. Output remains the same clean tight frame grid as the original
#      task; only the slot assignment changes.
# g) Color-change inferability
#    - No new color rule is introduced. Sprite holes are still recolored to the containing frame's
#      color exactly as before, so D19 is isolated.
# 9) New assumptions introduced by the modified solver
#    - Sorting assumes hole counts are unique or at least stably ordered. The generator guarantees
#      uniqueness by using consecutive integers without repetition.
#    - Matching-by-hole-count assumes every frame hole count appears in exactly one sprite. The
#      generator constructs a bijection and verifies it through main(input) == expected.


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _extract_scene(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        box = comp["bbox"]
        h = int(bbox_h(box))
        w = int(bbox_w(box))
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
    frames_raw = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames_raw
    assert sprites_raw

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames_raw]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict] = []
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
                    "inp_r": inp_r,
                    "inp_c": inp_c,
                    "color": frame_color,
                    "holes": int(num_holes),
                }
            )

    sprites: list[dict] = []
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

    return frame_size, sprite_size, frame_grid, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then sort the output frames by hole count.

    Relative to the original task, only the final slot order changes:
    after matching, output slots are filled by increasing frame hole count rather than by the
    frames' original row-major input positions.
    """
    frame_size, _sprite_size, frame_grid, sprites = _extract_scene(grid)
    row_positions = sorted({int(f["inp_r"]) for f in frame_grid})
    col_positions = sorted({int(f["inp_c"]) for f in frame_grid})
    n_rows = len(row_positions)
    n_cols = len(col_positions)
    assert n_rows * n_cols == len(frame_grid)

    sprite_by_holes = {int(s["holes"]): s for s in sprites}
    sorted_frames = sorted(frame_grid, key=lambda f: (int(f["holes"]), int(f["inp_r"]), int(f["inp_c"])))

    out = np.full((n_rows * frame_size, n_cols * frame_size), BG, dtype=int)
    for out_idx, frame in enumerate(sorted_frames):
        ri, ci = divmod(out_idx, n_cols)
        out_r = int(ri * frame_size)
        out_c = int(ci * frame_size)
        frame_color = int(frame["color"])
        out[out_r : out_r + frame_size, out_c : out_c + frame_size] = frame_color

        sprite = sprite_by_holes.get(int(frame["holes"]))
        if sprite is None:
            continue
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[out_r + 1 : out_r + 1 + sh, out_c + 1 : out_c + 1 + sw] = content

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
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_idx, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_holes)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_holes:
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
            all_holes.add(cell)
            sprite[cell[0], cell[1]] = BG

    _lab, num_holes = label_mask(sprite != sprite_color, connectivity=4)
    if int(num_holes) != int(hole_count):
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

    _lab, num_holes = label_mask(frame != frame_color, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return frame


def _build_expected_output(
    frames: list[dict],
    sprites: list[dict],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
    *,
    sort_by_holes: bool,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_holes = {int(s["holes"]): np.array(s["patch"], dtype=int) for s in sprites}
    ordered_frames = (
        sorted(frames, key=lambda f: int(f["holes"]))
        if sort_by_holes
        else list(frames)
    )
    for idx, frame in enumerate(ordered_frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(ri * frame_size)
        cc = int(ci * frame_size)
        frame_color = int(frame["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_holes[int(frame["holes"])].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _frame_hole_sequence(frames: list[dict]) -> list[int]:
    return [int(frame["holes"]) for frame in frames]


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    height = random.randint(min_h, CANVAS_MAX)
    width = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n)
    sprite_colors = random.sample(available, n)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    sorted_holes = list(range(start, start + n))

    input_frame_holes = sorted_holes[:]
    random.shuffle(input_frame_holes)
    if input_frame_holes == sorted_holes:
        return None

    sprite_hole_order = sorted_holes[:]
    random.shuffle(sprite_hole_order)

    sprites: list[dict] = []
    for idx, holes in enumerate(sprite_hole_order):
        patch = _make_sprite_interior(sprite_size, holes, sprite_colors[idx])
        if patch is None:
            return None
        sprites.append({"holes": int(holes), "color": int(sprite_colors[idx]), "patch": patch})

    frames: list[dict] = []
    for idx, holes in enumerate(input_frame_holes):
        patch = _make_frame(frame_size, holes, frame_colors[idx])
        if patch is None:
            return None
        frames.append({"holes": int(holes), "color": int(frame_colors[idx]), "patch": patch})

    expected = _build_expected_output(
        frames,
        sprites,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
        sort_by_holes=True,
    )
    legacy_expected = _build_expected_output(
        frames,
        sprites,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
        sort_by_holes=False,
    )
    if np.array_equal(expected, legacy_expected):
        return None

    canvas = np.full((height, width), BG, dtype=int)
    r0 = random.randint(0, height - grid_pixel_h)
    c0 = random.randint(0, width - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

    for sprite in sprites:
        try:
            r, c = random_free_location(
                canvas,
                np.array(sprite["patch"], dtype=int),
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, np.array(sprite["patch"], dtype=int), r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    detected_frame_holes = [int(f["holes"]) for f in sorted(_extract_scene(canvas)[2], key=lambda f: (int(f["inp_r"]), int(f["inp_c"])))]
    if detected_frame_holes == sorted(detected_frame_holes):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
