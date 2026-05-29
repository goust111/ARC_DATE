# Variant of 8698868d
# Dimension: D1 — 匹配条件：连通洞计数 (match_cond)
# Change: hole_counting — 精灵与框架各自内部的 4-连通背景色洞分量数目相等即配对... → 颜色匹配替代洞计数

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location

# Design analysis for this D1 variant
# -----------------------------------
# a) Concrete behavior
# - Keep the original macro-structure: large square frames are packed in a tight grid,
#   smaller square sprites are scattered elsewhere, and the output is the cropped frame
#   grid with one sprite nested inside each frame.
# - The core change is only the matching cue. Each frame and each sprite now contains
#   exactly one 1-pixel marker at the same fixed relative position: the top-left corner
#   of that object's bounding square. Matching is by equality of that marker color.
# - Holes still exist as visible structure, but they become decoys. Frame holes remain
#   bg-colored cells in the interior; sprite holes remain bg-colored cells/components in
#   the sprite body. Output behavior stays close to the original: sprite bg holes are
#   recolored to the surrounding frame color after nesting.
# - Small trace:
#   original logic:
#     frame interior has 2 disconnected bg holes, sprite has 2 disconnected bg holes,
#     so they are paired even if their corner colors differ.
#   new logic:
#     frame top-left corner marker is red, sprite top-left corner marker is red, so they
#     are paired even if their hole counts differ or hole multiplicities are duplicated.
#   critical difference:
#     hole counts are no longer a usable bijective key; fixed-location color comparison is.
#
# b) Complexity calibration
# - A single marker pixel is visually simple, but it is anchored at a fixed semantic
#   location (top-left corner) on both object types, so the cue is easy to spot and
#   compare across 2-4 examples.
# - Hole counts are intentionally allowed to repeat and the frame/sprite hole-count
#   multisets are forced to differ, so the old counting strategy is unreliable rather
#   than silently still solving the task.
# - Marker colors are unique per pair inside each sample, which keeps matching bijective
#   without introducing too many free variables.
#
# c) Hidden assumptions in the original main()
# - Original sprite hole detection used `sprite_region != sprite_color`. That was safe
#   before because the only non-sprite-color cells were bg holes. In this variant the
#   marker pixel is also non-sprite-color, so that code would falsely count the marker
#   as a hole.
# - Original output recoloring used `content[content != sprite_color] = frame_color`.
#   That would overwrite the sprite marker. Here only bg cells are recolored.
# - Original matching loop assumed the per-object hole count was the discriminative key.
#   That assumption is removed; the new key is the marker color at a fixed location.
#
# d) Placement legality
# - Frames still form a gapless grid and sprites are still placed by `random_free_location`
#   on the remaining bg area, so objects do not overlap or leave the canvas.
# - The marker sits on a corner cell already belonging to the object bbox, so adding it
#   does not enlarge the object or change placement geometry.
# - Frame body colors remain distinct within the frame grid so adjacent frames never merge
#   into one monochromatic component.
#
# e) Visual observability
# - A solver can now infer "same corner color => same pair" directly from the input.
# - Repeated hole counts and non-matching hole-count multisets make the old cue visibly
#   untrustworthy across examples, while the repeated marker convention is stable.
#
# f) Visual cleanliness
# - Only one extra pixel per object is added, at a consistent corner location. This avoids
#   clutter and keeps frames/sprites easy to parse.
# - The output preserves the frame marker on the outer top-left corner and the sprite
#   marker inside the nested sprite, so the pairing cue remains visible after the move.
#
# g) Color-change inferability
# - No hidden numeric ordering of colors is used. The only color-based rule is direct
#   equality of two visibly marked pixels.
# - Sprite bg holes still recolor to the hosting frame color exactly as before, which is
#   structurally signaled by the nesting relation.
#
# New main() assumptions checked against the generator
# - Every valid frame region has its marker at `region[0, 0]`, distinct from the dominant
#   frame border color. Generator guarantees this.
# - Every sprite bbox also has its marker at `[0, 0]`. Generator guarantees this.
# - Marker colors are unique across pairs within a sample, so dictionary matching is safe.
# - Sprite holes are bg-colored only. Generator guarantees this, allowing bg-only recolor.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by a fixed-position marker color, then nest each sprite inside
    its matching frame and recolor sprite bg holes to the frame color.
    """
    g = np.array(grid, dtype=int)
    BG = 8

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # As in the original task, identify candidate square objects by monochrome main-body
    # components. The 1-pixel marker is a separate tiny component and is ignored here.
    MIN_FILL_RATIO = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames and sprites_raw

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
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
            frame_grid.append(
                {
                    "color": frame_color,
                    "marker": int(region[0, 0]),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
        sprites.append(
            {
                "color": int(comp["color"]),
                "marker": int(sprite_region[0, 0]),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    sprites_by_marker = {int(s["marker"]): s for s in sprites}
    assert len(sprites_by_marker) == len(sprites)

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        marker_color = int(frame["marker"])

        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color
        out[fr, fc] = marker_color

        sprite = sprites_by_marker.get(marker_color)
        assert sprite is not None

        content = np.array(sprite["content"], dtype=int).copy()
        content[content == BG] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


# --------------- generator ---------------

BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_RETRIES = 60
SEED_RETRIES = 50
GEN_RETRIES = 400
MAX_VISIBLE_HOLES = 3

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _max_isolated_holes(size: int) -> int:
    inner = size - 2
    return (inner * inner + 1) // 2


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


def _make_sprite(size: int, hole_count: int, sprite_color: int, marker_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count > 0:
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

        _, num = label_mask(sprite == BG, connectivity=4)
        if num != hole_count:
            return None

    sprite[0, 0] = marker_color
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int, marker_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count > 0:
        inner_cells = (frame_size - 2) ** 2
        if inner_cells < hole_count:
            return None

        seeds = _place_isolated_seeds(frame_size, hole_count)
        if seeds is None:
            return None

        for r, c in seeds:
            frame[r, c] = BG

        _, num = label_mask(frame == BG, connectivity=4)
        if num != hole_count:
            return None

    frame[0, 0] = marker_color
    return frame


def _pick_distinct_colors(n: int, forbidden: list[int]) -> list[int]:
    available = [c for c in range(10) if c != BG]
    for _ in range(200):
        colors = random.sample(available, n)
        if all(int(colors[i]) != int(forbidden[i]) for i in range(n)):
            return colors
    raise RuntimeError("_pick_distinct_colors: failed to avoid marker collisions")


def _hole_lists_are_decoupled(frame_holes: list[int], sprite_holes: list[int]) -> bool:
    if sorted(frame_holes) == sorted(sprite_holes):
        return False
    return len(set(frame_holes)) < len(frame_holes) or len(set(sprite_holes)) < len(sprite_holes)


def _build_expected_output(
    frames: list[np.ndarray],
    sprites: list[np.ndarray],
    frame_colors: list[int],
    marker_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    for idx, frame_patch in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = ri * frame_size
        cc = ci * frame_size
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        expected[rr, cc] = int(marker_colors[idx])

        content = sprites[idx].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    return expected


def _gen_attempt() -> np.ndarray | None:
    N = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[N])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None
    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    marker_colors = random.sample(available, N)
    frame_colors = _pick_distinct_colors(N, marker_colors)
    sprite_colors = _pick_distinct_colors(N, marker_colors)

    frame_hole_max = min(MAX_VISIBLE_HOLES, _max_isolated_holes(frame_size))
    sprite_hole_max = min(MAX_VISIBLE_HOLES, _max_isolated_holes(sprite_size))
    if frame_hole_max < 0 or sprite_hole_max < 0:
        return None

    frame_hole_choices = list(range(frame_hole_max + 1))
    sprite_hole_choices = list(range(sprite_hole_max + 1))
    for _ in range(HOLE_RETRIES):
        frame_holes = [random.choice(frame_hole_choices) for _ in range(N)]
        sprite_holes = [random.choice(sprite_hole_choices) for _ in range(N)]
        if _hole_lists_are_decoupled(frame_holes, sprite_holes):
            break
    else:
        return None

    frames: list[np.ndarray] = []
    for i in range(N):
        patch = _make_frame(frame_size, frame_holes[i], frame_colors[i], marker_colors[i])
        if patch is None:
            return None
        frames.append(patch)

    sprites: list[np.ndarray] = []
    for i in range(N):
        patch = _make_sprite(sprite_size, sprite_holes[i], sprite_colors[i], marker_colors[i])
        if patch is None:
            return None
        sprites.append(patch)

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, fpatch in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = fpatch

    sprite_order = list(range(N))
    random.shuffle(sprite_order)
    for idx in sprite_order:
        spatch = sprites[idx]
        try:
            r, c = random_free_location(canvas, spatch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, spatch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames=frames,
        sprites=sprites,
        frame_colors=frame_colors,
        marker_colors=marker_colors,
        rows_grid=rows_grid,
        cols_grid=cols_grid,
        frame_size=frame_size,
        sprite_size=sprite_size,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
