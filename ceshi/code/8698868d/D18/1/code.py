# Variant of 8698868d
# Dimension: D18 — 框架边框厚度 (meta)
# Change: 框架边框为 1 像素厚度，内部空间为 (frame_size - 2)^2... → 不等厚边框

from __future__ import annotations

from collections import Counter
import os
import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
design_analysis:
  source_task: none
  variant_core:
    original:
      - frame outer bbox is a square of side 6-8
      - border thickness is fixed at 1 on all four sides
      - sprite size is implicitly (frame_size - 2) x (frame_size - 2), so it is also square
      - solver can safely hardcode region[1:-1, 1:-1] and paste at +1,+1
    variant:
      - frame outer bbox stays square so the global "tight square grid" remains visually familiar
      - top/bottom border thickness = tb, left/right border thickness = lr, with tb != lr
      - sprite size becomes (frame_size - 2*tb) x (frame_size - 2*lr), hence non-square
      - matching rule still uses connected hole-count only

  concrete_trace:
    example:
      - frame_size = 9, tb = 1, lr = 2
      - original solver would read inner = region[1:-1, 1:-1], a 7x7 square
      - variant inner is actually region[1:8, 2:7], a 7x5 rectangle
      - original +1,+1 paste would shift the sprite too far left/right logic-wise and produce the wrong border thickness
      - variant solver must first infer sprite bbox 7x5 from scattered sprites, then backsolve tb=(9-7)//2=1, lr=(9-5)//2=2

  complexity_calibration:
    - To keep D18 visible, the generator only uses the asymmetric pairs (1,2) or (2,1).
    - The asymmetry is repeated across every frame in one sample, so the eye can detect a consistent rectangular cavity.
    - The sprite remains bordered and hole-bearing, preserving the original "match by hole-count" logic.
    - Frame outer shape remains square to avoid introducing a second independent variation (rectangular frame footprints).

  implicit_assumption_review:
    - original "if h == w" sprite/frame detection no longer works for sprites because sprites are now rectangular
    - original "sizes[largest] = frames, sizes[second] = sprites" depends on both object classes being size-grouped by one scalar; broken here
    - original "inner = region[1:-1, 1:-1]" depends on uniform 1-pixel border; broken here
    - original "out[fr+1:..., fc+1:...]" depends on uniform 1-pixel inset; broken here
    - original fallback "sprite_size = frame_size - 2" is invalid because height/width now need separate offsets

  placement_legality:
    - Frames are still placed as a tight rows x cols grid with no gaps and no overlaps.
    - Sprites are placed only on background cells outside that grid via random_free_location(..., padding=1).
    - The generator validates every sample by checking main(input) == expected_output.

  visual_observability:
    - Human-solvers can see that the nested object in the output is a rectangle whose empty margin is thinner vertically than horizontally, or vice versa.
    - Across samples, the generator varies between (tb,lr)=(1,2) and (2,1), so the rule cannot be reduced to the old fixed +1 inset.
    - No color ordering is used; geometry alone determines the inset.

  cleanliness:
    - Only two object classes remain: square frame blocks and repeated rectangular sprites.
    - No extra markers or overlays are added, so the variant stays visually close to the base task.

  color_inference:
    - Color behavior is unchanged from the original task: sprite holes are recolored to the matched frame color in the output.
    - This remains easy to infer because the recoloring occurs exactly inside the inserted sprite region.

  new_main_assumptions_and_generator_guarantees:
    - assumption: all frames share one common outer square size
      guarantee: generator uses one frame_size per sample
    - assumption: all sprites share one common rectangular bbox
      guarantee: generator uses one (sprite_h, sprite_w) per sample
    - assumption: sprites are non-square, so the most common non-square component size identifies them
      guarantee: generator always enforces tb != lr
    - assumption: frame grid is tight and axis-aligned with stride = frame_size
      guarantee: generator places frames in a contiguous rows x cols block
"""


# ── Configuration ──────────────────────────────────────────────────────
BG = 8
FRAME_SIZE_RANGE = [8, 9, 10]
THICKNESS_OPTIONS = [(1, 2), (2, 1)]  # (top/bottom, left/right)
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 60
GEN_RETRIES = 400
MIN_FILL_RATIO = 0.45

_LAYOUTS: list[tuple[int, int]] = []
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS.append((_r, _c))


def _max_isolated_cells(h: int, w: int) -> int:
    h = int(h)
    w = int(w)
    if h <= 0 or w <= 0:
        return 0
    return int((h * w + 1) // 2)


def _place_isolated_rect_seeds(r0: int, r1: int, c0: int, c1: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` mutually 4-isolated cells in the inclusive rectangle [r0:r1] x [c0:c1]."""
    if count == 0:
        return []
    if r1 < r0 or c1 < c0:
        return None
    positions = [(r, c) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)]
    if _max_isolated_cells(r1 - r0 + 1, c1 - c0 + 1) < int(count):
        return None
    for _ in range(SEED_RETRIES):
        random.shuffle(positions)
        seeds: list[tuple[int, int]] = []
        for r, c in positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((int(r), int(c)))
                if len(seeds) == int(count):
                    return seeds
    return None


def _make_sprite(h: int, w: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    """Build a rectangular sprite with a solid 1-pixel border and `hole_count` hole-components."""
    h = int(h)
    w = int(w)
    sprite = np.full((h, w), int(sprite_color), dtype=int)
    if hole_count == 0:
        return sprite
    if h < 3 or w < 3:
        return None

    inner_h = h - 2
    inner_w = w - 2
    if _max_isolated_cells(inner_h, inner_w) < int(hole_count):
        return None

    seeds = _place_isolated_rect_seeds(1, h - 2, 1, w - 2, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_h * inner_w // max(int(hole_count), 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_h * inner_w - len(all_hole)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= h - 2 and 1 <= nc <= w - 2):
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
                    frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    _, num = label_mask(sprite != int(sprite_color), connectivity=4)
    if int(num) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, tb: int, lr: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build a square frame with asymmetric border thickness and single-cell interior holes."""
    frame_size = int(frame_size)
    tb = int(tb)
    lr = int(lr)
    frame = np.full((frame_size, frame_size), int(frame_color), dtype=int)
    if hole_count == 0:
        return frame

    inner_h = frame_size - 2 * tb
    inner_w = frame_size - 2 * lr
    seeds = _place_isolated_rect_seeds(tb, frame_size - tb - 1, lr, frame_size - lr - 1, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    inner = frame[tb : tb + inner_h, lr : lr + inner_w]
    _, num = label_mask(inner != int(frame_color), connectivity=4)
    if int(num) != int(hole_count):
        return None
    return frame


def _dense_enough(comp: dict) -> bool:
    b = comp["bbox"]
    h = int(bbox_h(b))
    w = int(bbox_w(b))
    if h <= 0 or w <= 0:
        return False
    return float(comp["area"]) / float(h * w) >= float(MIN_FILL_RATIO)


def _infer_geometry(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    frame_candidates = []
    sprite_candidates = []
    for comp in comps:
        if not _dense_enough(comp):
            continue
        h = int(bbox_h(comp["bbox"]))
        w = int(bbox_w(comp["bbox"]))
        if h == w and 6 <= h <= 12:
            frame_candidates.append(comp)
        elif h != w and 3 <= h <= 10 and 3 <= w <= 10:
            sprite_candidates.append(comp)

    assert frame_candidates
    assert sprite_candidates

    frame_size = max(int(bbox_h(comp["bbox"])) for comp in frame_candidates)
    frames = [comp for comp in frame_candidates if int(bbox_h(comp["bbox"])) == int(frame_size)]
    assert len(frames) >= 2

    sprite_dim_counts = Counter()
    for comp in sprite_candidates:
        h = int(bbox_h(comp["bbox"]))
        w = int(bbox_w(comp["bbox"]))
        if h < frame_size and w < frame_size:
            sprite_dim_counts[(h, w)] += 1
    assert sprite_dim_counts
    (sprite_h, sprite_w), _ = max(sprite_dim_counts.items(), key=lambda kv: (kv[1], kv[0][0] * kv[0][1]))
    sprites = [comp for comp in sprite_candidates if (int(bbox_h(comp["bbox"])), int(bbox_w(comp["bbox"]))) == (int(sprite_h), int(sprite_w))]

    tb_delta = int(frame_size - sprite_h)
    lr_delta = int(frame_size - sprite_w)
    assert tb_delta % 2 == 0 and lr_delta % 2 == 0
    tb = tb_delta // 2
    lr = lr_delta // 2
    assert tb >= 1 and lr >= 1 and tb != lr
    assert len(sprites) == len(frames)

    return {
        "frame_size": int(frame_size),
        "sprite_h": int(sprite_h),
        "sprite_w": int(sprite_w),
        "tb": int(tb),
        "lr": int(lr),
        "frames": frames,
        "sprites_raw": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

    Variant D18:
    - Frames keep a square outer footprint but use asymmetric border thickness.
    - Sprites are rectangles whose size matches the frame interior.
    - The solver infers the interior rectangle from repeated sprite dimensions instead of hardcoding a 1-pixel inset.
    """
    g = np.array(grid, dtype=int)
    geom = _infer_geometry(g)
    frame_size = int(geom["frame_size"])
    sprite_h = int(geom["sprite_h"])
    sprite_w = int(geom["sprite_w"])
    tb = int(geom["tb"])
    lr = int(geom["lr"])
    frames = list(geom["frames"])
    sprites_raw = list(geom["sprites_raw"])

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
            assert region.shape == (frame_size, frame_size)
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[tb : tb + sprite_h, lr : lr + sprite_w]
            hole_mask = inner != int(frame_color)
            _lab, num_holes = label_mask(hole_mask, connectivity=4)
            frame_grid.append(
                {
                    "color": int(frame_color),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": int(num_holes),
                }
            )

    sprites = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": int(sprite_color),
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if int(i) not in used_sprites and int(f["holes"]) == int(s["holes"]):
                matches[(int(f["out_r"]), int(f["out_c"]))] = s
                used_sprites.add(int(i))
                break

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for f in frame_grid:
        fr = int(f["out_r"])
        fc = int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        sprite = matches[(fr, fc)]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)
        out[fr + tb : fr + tb + sprite_h, fc + lr : fc + lr + sprite_w] = content

    return out


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    tb: int,
    lr: int,
    sprite_h: int,
    sprite_w: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {int(k): spatch for k, spatch in sprites}
    for idx, (k, _patch) in enumerate(frames):
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = ri * frame_size
        cc = ci * frame_size
        fc = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(fc)
        content = sprite_by_k[int(k)].copy()
        content[content == BG] = int(fc)
        expected[rr + tb : rr + tb + sprite_h, cc + lr : cc + lr + sprite_w] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    tb, lr = random.choice(THICKNESS_OPTIONS)
    sprite_h = int(frame_size - 2 * tb)
    sprite_w = int(frame_size - 2 * lr)
    if sprite_h <= 0 or sprite_w <= 0 or sprite_h == sprite_w:
        return None

    frame_cap = _max_isolated_cells(sprite_h, sprite_w)
    sprite_cap = _max_isolated_cells(sprite_h - 2, sprite_w - 2)
    max_hole = min(frame_cap, sprite_cap)
    if max_hole <= 0:
        return None

    layout_choices: list[tuple[int, int, int, int, int]] = []
    for rows_grid, cols_grid in _LAYOUTS:
        n_frames = rows_grid * cols_grid
        if max_hole < n_frames - 1:
            continue
        grid_pixel_h = rows_grid * frame_size
        grid_pixel_w = cols_grid * frame_size
        min_h = max(CANVAS_MIN, grid_pixel_h + sprite_h + SPRITE_PADDING)
        min_w = max(CANVAS_MIN, grid_pixel_w + sprite_w + SPRITE_PADDING)
        if min_h <= CANVAS_MAX and min_w <= CANVAS_MAX:
            layout_choices.append((rows_grid, cols_grid, n_frames, min_h, min_w))
    if not layout_choices:
        return None

    rows_grid, cols_grid, n_frames, min_h, min_w = random.choice(layout_choices)
    H = random.randint(min_h, CANVAS_MAX)
    W = random.randint(min_w, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, n_frames)

    max_start = min(HOLE_START_MAX, max_hole - (n_frames - 1))
    if max_start < 0:
        return None
    start = random.randint(0, max_start)
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(sprite_hole_perm):
        patch = _make_sprite(sprite_h, sprite_w, int(k), int(sprite_colors[i]))
        if patch is None:
            return None
        sprites.append((int(k), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(hole_counts):
        patch = _make_frame(frame_size, tb, lr, int(k), int(frame_colors[i]))
        if patch is None:
            return None
        frames.append((int(k), patch))

    canvas = np.full((H, W), BG, dtype=int)
    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_k, fpatch) in enumerate(frames):
        ri = idx // cols_grid
        ci = idx % cols_grid
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = fpatch

    for _k, spatch in sprites:
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
        frames,
        sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
        tb,
        lr,
        sprite_h,
        sprite_w,
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
