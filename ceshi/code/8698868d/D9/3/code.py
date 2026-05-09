# Variant of 8698868d
# Dimension: D9 — 洞计数范围与分配 (meta)
# Change: 洞计数为连续整数 0..N-1，每个值恰好出现一次（双射配对）... → 洞计数逆序匹配

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Input objects stay the same as the base task: a tight grid of larger square frames,
#      plus the same number of smaller square sprites scattered elsewhere on azure bg=8.
#    - Every frame interior contains k isolated bg cells; every sprite contains k connected
#      bg-hole components. Counts are the visible feature. Colors remain unrelated to ranking.
#    - Original pairing: frame(k) <- sprite(k).
#    - Variant pairing: sort frame counts ascending and sprite counts descending, then pair
#      by rank. Small example with counts {0,1,2}:
#        original: frame0<-sprite0, frame1<-sprite1, frame2<-sprite2
#        variant : frame0<-sprite2, frame1<-sprite1, frame2<-sprite0
#      The middle item is unchanged only when N is odd; the extremes always swap.
#    - Output geometry stays identical: each frame becomes a solid square in the output and
#      its matched sprite is nested inside; sprite hole pixels are recolored to the frame color.
#
# b) Complexity calibration
#    - Keep counts unique and consecutive 0..N-1 so the ranking is unambiguous.
#    - Keep N in 2..6 and frame sizes in 6..8 as in the base task; this preserves original
#      density and avoids inventing new cues unrelated to D9.
#    - Sprites still have different colors and hole layouts, so reversing the pairing is
#      visible rather than collapsing into the same-looking result.
#
# c) Implicit-assumption review of the original main()
#    - Component finding and square-size grouping do not depend on equal matching and remain valid.
#    - Frame hole counting from the inner region also remains valid because frames still encode
#      counts by isolated bg cells.
#    - Sprite hole counting also remains valid because sprites still encode counts by connected
#      bg-hole components.
#    - The only invalid base-task assumption is the nested loop
#          if f["holes"] == s["holes"]
#      which hardcodes equality. In the variant this must become an order-based match.
#    - Output painting logic is unchanged after the pair is chosen; it depends only on having a
#      unique sprite per frame.
#
# d) Placement legality
#    - The frame grid is still pasted as one tight rectangle with no gaps.
#    - Sprites are still placed only on background using random_free_location(..., padding=1),
#      so they cannot overlap frames or each other.
#    - Counts 0..N-1 fit physically because N<=6 and the maximum isolated interior holes in the
#      smallest frame size (6 => inner 4x4) is 8, which is safely above 5.
#
# e) Visual observability
#    - The reversal is human-visible because the most-holed sprite gets inserted into the least-
#      holed frame and vice versa. This changes which sprite color/shape appears in each frame.
#    - The generator fixes the count set to 0..N-1, so across examples the solver must infer
#      an ordered reversal, not an arbitrary per-instance permutation.
#
# f) Visual cleanliness
#    - No new marks are added. The output stays a compact frame grid exactly like the original,
#      which keeps the scene legible.
#
# g) Color inference
#    - Colors still identify objects only; they do not determine the reversal. Human solvers can
#      infer the rule from hole counts and see colors merely as carriers of which sprite moved.
#
# New assumptions introduced by this variant
# ------------------------------------------
# - Matching is now rank-based, so we assume the frame counts and sprite counts form two sets of
#   equal size with a unique total ordering. The generator guarantees this by using counts 0..N-1
#   exactly once in each family.
# - Ties would make the reverse matching ambiguous; the generator forbids them.


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 50
GEN_RETRIES = 300


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _reverse_count_map(counts: list[int]) -> dict[int, int]:
    ordered = sorted(int(v) for v in counts)
    return {int(lo): int(hi) for lo, hi in zip(ordered, reversed(ordered))}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by reversed hole-count rank, then nest each sprite inside its frame.

    Frames are ranked by hole count from low to high. Sprites are ranked by hole count from high
    to low. The kth frame in ascending order receives the kth sprite in descending order.
    """
    g = np.array(grid, dtype=int)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

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
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

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
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
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

    assert len(frame_grid) == len(sprites)
    frame_holes = sorted(int(f["holes"]) for f in frame_grid)
    sprite_holes = sorted((int(s["holes"]) for s in sprites), reverse=True)
    assert len(set(frame_holes)) == len(frame_grid)
    assert len(set(sprite_holes)) == len(sprites)

    sorted_frames = sorted(frame_grid, key=lambda f: int(f["holes"]))
    sorted_sprites = sorted(sprites, key=lambda s: int(s["holes"]), reverse=True)
    matches: dict[tuple[int, int], tuple[dict, dict]] = {}
    for f, s in zip(sorted_frames, sorted_sprites):
        matches[(int(f["out_r"]), int(f["out_c"]))] = (f, s)

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for f in frame_grid:
        fr, fc = int(f["out_r"]), int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        if (fr, fc) in matches:
            _f, sprite = matches[(fr, fc)]
            content = np.array(sprite["content"], dtype=int).copy()
            sprite_color = int(sprite["color"])
            content[content != sprite_color] = frame_color
            sh, sw = int(sprite["h"]), int(sprite["w"])
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

    _lab, num = label_mask(sprite != sprite_color, connectivity=4)
    if int(num) != int(hole_count):
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

    _lab, num = label_mask(frame != frame_color, connectivity=4)
    if int(num) != int(hole_count):
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
    sprite_by_k = {int(k): spatch for k, spatch in sprites}
    reverse_map = _reverse_count_map([int(k) for k, _ in frames])
    for idx, (frame_holes, _patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * frame_size, ci * frame_size
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_k[int(reverse_map[int(frame_holes)])].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _build_equal_match_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {int(k): spatch for k, spatch in sprites}
    for idx, (frame_holes, _patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * frame_size, ci * frame_size
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_k[int(frame_holes)].copy()
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
    frame_colors = random.sample(available, N)
    sprite_colors = random.sample(available, N)

    hole_counts = list(range(N))
    frame_hole_order = hole_counts[:]
    random.shuffle(frame_hole_order)
    sprite_hole_order = hole_counts[:]
    random.shuffle(sprite_hole_order)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(sprite_hole_order):
        patch = _make_sprite_interior(sprite_size, int(k), int(sprite_colors[i]))
        if patch is None:
            return None
        sprites.append((int(k), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(frame_hole_order):
        patch = _make_frame(frame_size, int(k), int(frame_colors[i]))
        if patch is None:
            return None
        frames.append((int(k), patch))

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_holes, fpatch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = fpatch

    for _holes, spatch in sprites:
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
        frames, sprites, frame_colors, rows_grid, cols_grid, frame_size, sprite_size
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    equal_output = _build_equal_match_output(
        frames, sprites, frame_colors, rows_grid, cols_grid, frame_size, sprite_size
    )
    if np.array_equal(expected, equal_output):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
