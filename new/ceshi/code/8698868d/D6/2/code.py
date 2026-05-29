# Variant of 8698868d
# Dimension: D6 — 轨迹：精灵源位置的处理 (trajectory)
# Change: 默认消失 — 精灵不出现在输出中（输出仅为框架网格）
#   -> 精灵原位置填充为框架色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the D6 variant
#
# a) Concrete behavior and trace
#    - Input stays structurally identical to the original task:
#      a tight grid of larger square frames plus the same number of scattered smaller square sprites.
#      Frames and sprites are still paired only by the number of 4-connected hole components.
#    - Original output discarded the whole input canvas and rebuilt only a compact frame grid.
#    - New output keeps the full input canvas size. Two effects happen at once:
#      1. Each frame is transformed in place exactly as before: its interior is replaced by the matched sprite,
#         and the sprite's holes are recolored to the frame color.
#      2. The matched sprite's original square location is not removed. Instead, that full square bbox becomes a
#         solid block of the matched frame color.
#    - Small trace:
#      frame holes = 2, frame color = 4; matching sprite color = 3 with two hole components.
#      Original: output only showed the 4-colored frame containing the recolored sprite.
#      Variant: that same transformed frame remains, and the sprite's old 4x4 source location becomes a solid 4 block.
#      The changed stage is therefore output construction, not detection or matching.
#
# b) Complexity calibration
#    - The hole-count bijection remains the sole matching key. This preserves the original reasoning burden.
#    - To keep the D6 change visually observable, the generator forces every sprite's own color to differ from the
#      matched frame color. Otherwise a 0-hole sprite could remain visually unchanged after recoloring, hiding D6.
#    - No extra axes such as rotation, reflection, or positional sorting are introduced.
#
# c) Implicit-assumption audit of the original main()
#    - Finding square components by size/fill ratio still works, because the input object geometry is unchanged.
#    - Grouping largest squares as frames and smaller squares as sprites still works, because D6 does not alter input.
#    - Computing frame-grid rows/cols from frame top-left coordinates still works, because frames remain a tight grid.
#    - The broken old assumption is output size:
#      original code used a new canvas of shape (n_rows * frame_size, n_cols * frame_size).
#      That silently relied on sprites disappearing. Under D6 this must become the full input canvas shape.
#    - The old `out_r/out_c` coordinates were compact output coordinates, not input coordinates.
#      Under D6 we must retain frame input coordinates and sprite input coordinates, otherwise source repaint lands wrong.
#
# d) Placement legality
#    - Frames are placed as a tight non-overlapping grid.
#    - Sprites are placed with `random_free_location(..., padding=1)`, so their square bboxes do not touch or overlap
#      other non-background content. This guarantees that turning the sprite bbox into a solid block stays legal.
#    - Output writes only to frame squares and sprite squares, both already known in-bounds.
#
# e) Visual observability
#    - Humans can see that the output now preserves the whole canvas and that the old sprite locations become solid
#      recolored blocks matching frame colors.
#    - The repeated evidence comes from several sprites per sample, not from a single exceptional object.
#
# f) Visual cleanliness
#    - The generator keeps one-cell sprite padding, so recolored source blocks do not merge with other objects.
#    - The output is layered simply: background, transformed frame grid, and separate solid source blocks.
#
# g) Color-change inference
#    - The recolored source block color is determined by the matched frame, not by the sprite itself.
#    - The matching signal is structural (hole-count), not numeric color identity. The generator avoids same-color matched
#      pairs so this recoloring remains visible in small training sets.
#
# h) New assumptions introduced by this variant and generator support
#    - main() now assumes each matched sprite bbox can be safely overwritten as a solid square; generate_input() enforces
#      sprite placement separation so this is valid.
#    - main() now assumes there is a unique frame color for each hole count; generate_input() keeps consecutive unique
#      hole counts exactly once across frames and sprites.


BG = 8


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count.

    Original behavior:
    - Nest matched sprites into a compact output frame grid.

    D6 variant behavior:
    - Keep the full canvas.
    - Transform each frame in place as before.
    - Repaint each matched sprite's original location as a solid block of the matched frame color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # Detect candidate square objects: larger ones are frames, smaller ones are sprites.
    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
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
                    "color": frame_color,
                    "inp_r": inp_r,
                    "inp_c": inp_c,
                    "holes": int(num_holes),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
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
                "inp_r": int(b[0]),
                "inp_c": int(b[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if i not in used_sprites and int(f["holes"]) == int(s["holes"]):
                matches[(int(f["inp_r"]), int(f["inp_c"]))] = s
                used_sprites.add(i)
                break

    out = np.full((H, W), BG, dtype=int)

    for f in frame_grid:
        fr, fc = int(f["inp_r"]), int(f["inp_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = matches.get((fr, fc))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh, sw = int(sprite["h"]), int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

        sr, sc = int(sprite["inp_r"]), int(sprite["inp_c"])
        out[sr : sr + sh, sc : sc + sw] = frame_color

    return out


# --------------- generator (D6 variant) ---------------

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


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place count cells in the inner area so no pair is 4-adjacent."""
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
    """Build a sprite with a solid border and exactly hole_count 4-connected hole components."""
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

    _, num = label_mask(sprite != sprite_color, connectivity=4)
    if num != hole_count:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build an input frame with hole_count isolated interior background cells."""
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

    _, num = label_mask(frame != frame_color, connectivity=4)
    if num != hole_count:
        return None
    return frame


def _choose_sprite_colors(frame_color_by_k: dict[int, int]) -> dict[int, int]:
    """Choose distinct sprite colors, each different from its matched frame color."""
    available = [c for c in range(10) if c != BG]
    sprite_color_by_k: dict[int, int] = {}
    used: set[int] = set()
    hole_keys = list(frame_color_by_k.keys())
    random.shuffle(hole_keys)
    for k in hole_keys:
        choices = [c for c in available if c not in used and c != int(frame_color_by_k[k])]
        assert choices
        chosen = random.choice(choices)
        sprite_color_by_k[int(k)] = int(chosen)
        used.add(int(chosen))
    return sprite_color_by_k


def _build_expected_output(
    H: int,
    W: int,
    frames: list[dict],
    sprites: list[dict],
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    """Build the D6 output: full canvas, transformed frames, and recolored source blocks."""
    expected = np.full((H, W), BG, dtype=int)
    sprite_by_k = {int(s["holes"]): s for s in sprites}
    frame_color_by_k = {int(f["holes"]): int(f["color"]) for f in frames}

    for f in frames:
        rr, cc = int(f["r"]), int(f["c"])
        fc = int(f["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = fc
        sprite = sprite_by_k[int(f["holes"])]
        content = np.array(sprite["patch"], dtype=int).copy()
        content[content == BG] = fc
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    for sprite in sprites:
        rr, cc = int(sprite["r"]), int(sprite["c"])
        fc = int(frame_color_by_k[int(sprite["holes"])])
        expected[rr : rr + sprite_size, cc : cc + sprite_size] = fc

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
    hole_counts = list(range(N))
    max_hole = ((frame_size - 2) ** 2 + 1) // 2
    if hole_counts[-1] > max_hole:
        return None

    start = random.randint(0, min(max(0, max_hole - N), HOLE_START_MAX))
    hole_counts = list(range(start, start + N))
    random.shuffle(hole_counts)

    frame_colors = random.sample(available, N)
    frame_color_by_k = {int(k): int(c) for k, c in zip(hole_counts, frame_colors)}
    sprite_color_by_k = _choose_sprite_colors(frame_color_by_k)

    sprite_order = hole_counts[:]
    random.shuffle(sprite_order)
    sprites: list[dict] = []
    for k in sprite_order:
        patch = _make_sprite_interior(sprite_size, int(k), int(sprite_color_by_k[int(k)]))
        if patch is None:
            return None
        sprites.append({"holes": int(k), "color": int(sprite_color_by_k[int(k)]), "patch": patch})

    frames: list[dict] = []
    for k in hole_counts:
        patch = _make_frame(frame_size, int(k), int(frame_color_by_k[int(k)]))
        if patch is None:
            return None
        frames.append({"holes": int(k), "color": int(frame_color_by_k[int(k)]), "patch": patch})

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        frame["r"] = rr
        frame["c"] = cc
        canvas[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

    for sprite in sprites:
        try:
            rr, cc = random_free_location(
                canvas,
                np.array(sprite["patch"], dtype=int),
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        sprite["r"] = int(rr)
        sprite["c"] = int(cc)
        paste(canvas, np.array(sprite["patch"], dtype=int), int(rr), int(cc), transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(H, W, frames, sprites, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
