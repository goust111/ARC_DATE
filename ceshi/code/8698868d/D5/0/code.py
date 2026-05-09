# Variant of 8698868d
# Dimension: D5 — 输出网格的构成 (grid_shape)
# Change: 输出为紧密的框架网格（rows_grid * frame_size x cols_grid * frame_size），... → 保留完整输入画布

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the D5 variant
# a) Concrete behavior:
#    The input is unchanged: a tight grid of larger square frames plus the same number of smaller square sprites
#    scattered elsewhere on an azure background. The original solver cropped the answer down to only the frame grid.
#    This variant keeps the full input canvas. The transformed frames stay at their original positions, every sprite is
#    inserted into its matched frame, and the original sprite positions disappear back into background. Example:
#    if a 2x2 frame grid starts at (3,5) on a 20x20 canvas, the old output was an 12x12 crop; the new output is still
#    20x20, with the nested 2x2 frame grid still rooted at (3,5) and blank background everywhere else.
# b) Complexity calibration:
#    The matching feature remains the connected-hole count. Keeping unique counts 0..N-1 preserves a bijection and
#    avoids ambiguous matching. The variant is visible because the output now has background margins and blanked sprite
#    source locations rather than a tight crop.
# c) Implicit assumptions review:
#    - Square-region detection still assumes frames are the largest square components and sprites are the next size down.
#      This remains valid because only the output canvas shape changes, not the input object construction.
#    - The original code stored frame output positions as compact grid offsets (ri*frame_size, ci*frame_size). That
#      assumption is exactly what breaks under D5; the variant must instead use the frames' original canvas coordinates.
#    - Starting from a cropped output canvas used to erase sprites "for free". In this variant that would wrongly delete
#      surrounding background margins, so the solver now builds a full-size canvas and paints only the transformed frames.
# d) Placement legality:
#    The generator keeps frames in a tight grid, guarantees they fit with at least a one-cell margin to the outer canvas,
#    and scatters sprites into remaining background without overlap. This ensures the solver can still infer the frame
#    lattice from equal frame-size steps and identify sprites as the smaller square components.
# e) Visual observability:
#    The generator forces extra canvas around the frame grid, so the new full-canvas output is visibly larger than the
#    old crop. The removed sprite locations are also visible when comparing input/output pairs.
# f) Visual cleanliness:
#    Only one transformed object family remains in the output: the frame grid in place. The background margins are empty,
#    so there is no confusing overlap between source sprites and destination frames.
# g) Color inference:
#    Color behavior is unchanged and remains local/structural: sprite holes are recolored to the enclosing frame color.
#    No hidden ordering by numeric color ids is introduced.


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
FRAME_MARGIN = 1
MIN_FILL_RATIO = 0.5


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _find_square_regions(grid: np.ndarray) -> dict[int, list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((h, comp))
    if not square_regions:
        raise AssertionError("no square regions found")

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)
    return sizes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, nest them into the on-canvas frame grid,
    and preserve the full input canvas in the output.
    """
    g = np.array(grid, dtype=int)
    sizes = _find_square_regions(g)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    if not frames:
        raise AssertionError("no frames found")

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
            _lab, num_holes = label_mask(inner != frame_color, connectivity=4)
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
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        _lab, num_holes = label_mask(sprite_region != sprite_color, connectivity=4)
        sprites.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
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
                matches[(int(frame["inp_r"]), int(frame["inp_c"]))] = sprite
                used_sprites.add(idx)
                break

    out = np.full_like(g, BG)
    for frame in frame_grid:
        fr = int(frame["inp_r"])
        fc = int(frame["inp_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = matches.get((fr, fc))
        if sprite is None:
            continue
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color
        sh = int(sprite["h"])
        sw = int(sprite["w"])
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
    for comp_idx, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_hole:
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
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    _lab, num = label_mask(sprite != sprite_color, connectivity=4)
    if num != hole_count:
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
    if num != hole_count:
        return None
    return frame


def _build_expected_output(
    canvas_shape: tuple[int, int],
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
    r0: int,
    c0: int,
) -> np.ndarray:
    expected = np.full(canvas_shape, BG, dtype=int)
    sprite_by_holes = {holes: spatch for holes, spatch in sprites}
    for idx, (holes, _patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_holes[holes].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING + 2 * FRAME_MARGIN)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING + 2 * FRAME_MARGIN)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, n_frames)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n_frames)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for idx, holes in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, holes, sprite_colors[idx])
        if patch is None:
            return None
        sprites.append((holes, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for idx, holes in enumerate(hole_counts):
        patch = _make_frame(frame_size, holes, frame_colors[idx])
        if patch is None:
            return None
        frames.append((holes, patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(FRAME_MARGIN, canvas_h - grid_pixel_h - FRAME_MARGIN)
    c0 = random.randint(FRAME_MARGIN, canvas_w - grid_pixel_w - FRAME_MARGIN)
    for idx, (_holes, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _holes, sprite_patch in sprites:
        try:
            r, c = random_free_location(canvas, sprite_patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, sprite_patch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        canvas.shape,
        frames,
        sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
        r0,
        c0,
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
