# Combo variant of 8698868d (index 19)
# combo_spec: D8/0+D18/0
# Name: 厚边框多像素洞框架

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask


# Design analysis for the fused variant:
# - Concrete example:
#   Original: with frame_size=6, a frame had a 1-pixel border and a 4x4 interior; the
#   solver counted isolated single-pixel frame holes inside that 4x4 area and pasted a
#   4x4 sprite at offset +1 in the output.
#   Fused variant: with frame_size=9, the border is 2 pixels thick, so the usable frame
#   cavity is 5x5 and the sprite size is exactly 5. A frame hole can now be a connected
#   domino/L-shape rather than only a singleton pixel. The solver still matches only by
#   the number of 4-connected hole components, then pastes the matched 5x5 sprite at
#   offset +2. The visible differences are therefore both structural (thick rim) and
#   internal (frame holes can span multiple cells).
# - Complexity calibration:
#   Keeping frame_size fixed at 9 makes the 2-pixel border obvious while leaving a 5x5
#   cavity that can still display several distinct hole counts. Counts remain consecutive
#   0..N-1, unique across frames/sprites, so matching is unambiguous. The generator also
#   forces at least one frame with a multi-cell hole component so the D8/0 change is not
#   optional noise.
# - Implicit assumption audit:
#   The original solver assumed:
#   1. sprite_size = frame_size - 2
#   2. a 1-pixel ring is enough to read frame color
#   3. frame holes are read from region[1:-1,1:-1]
#   4. output pastes sprites at +1 offset
#   These assumptions are invalid here. The new solver instead requires:
#   1. frames are the larger square components and sprites are the smaller square
#      components of size frame_size - 4
#   2. frame color is read from the 2-pixel outer ring
#   3. frame holes are counted only inside region[2:-2,2:-2]
#   4. output pastes sprites at +2 offset
#   5. hole counts are unique, so count-based matching stays bijective
#   The generator explicitly guarantees each of those conditions.
# - Placement legality and visual cleanliness:
#   Frames stay in a tight rectangular grid. Sprites are placed only in a reserved strip
#   outside the grid, on non-overlapping 5x5 slots, so nothing overlaps or goes out of
#   bounds. This keeps the input readable: large thick-rimmed frames occupy one block and
#   the loose sprites occupy a separate strip.
# - Human observability:
#   Across a few samples, a solver can see that the large squares have thicker rims, that
#   the interior gaps inside those large squares may be bars or blobs instead of isolated
#   dots, and that the output keeps the thick frame while inserting the loose 5x5 sprite
#   whose own holes are recolored to the frame color.


BG = 8
FRAME_BORDER = 2
SPRITE_BORDER = 1
FRAME_SIZE = 9
SPRITE_SIZE = FRAME_SIZE - 2 * FRAME_BORDER
CANVAS_SIZE = 30
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
GEN_RETRIES = 400
SEED_RETRIES = 80
MIN_FILL_RATIO = 0.6
GRID_STRIP_GAP = 1


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _ring_mask(size: int, thickness: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[:thickness, :] = True
    mask[-thickness:, :] = True
    mask[:, :thickness] = True
    mask[:, -thickness:] = True
    return mask


def _square_regions(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and h >= SPRITE_SIZE and area >= int(np.ceil(h * h * MIN_FILL_RATIO)):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    frame_size = max(sizes)
    sprite_size = frame_size - 2 * FRAME_BORDER
    frames = sizes.get(frame_size, [])
    sprites = sizes.get(sprite_size, [])
    assert frames and sprites
    return frame_size, sprite_size, frames, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match loose sprites to thick-rimmed frames by connected-hole count, then pack the
    matched sprites into a tight frame grid.

    Input structure:
    - bg is azure (8). Large same-color squares are frames with a 2-pixel-thick border;
      smaller same-color squares elsewhere are sprites whose size equals the frame cavity.
    - Frame interiors may contain several bg-colored hole components, and those components
      may themselves span multiple cells. Sprites also contain bg-colored hole components.

    Rule:
    - Detect the larger square objects as frames and the smaller square objects as sprites.
    - For each frame, count 4-connected hole components only inside the area after stripping
      the 2-pixel border. For each sprite, count 4-connected hole components inside the
      sprite shape.
    - Pair each frame with the sprite that has the same number of hole components.
    - Output only the tight frame grid; fill each frame square with its frame color and paste
      the matched sprite into the inner 5x5 cavity, recoloring the sprite's holes to the
      frame color so no background remains.

    Key insight:
    The relevant invariant is still the number of 4-connected hole components. The thick
    frame border and the fact that frame holes may be multi-cell change the appearance, but
    not the pairing rule.
    """
    g = np.array(grid, dtype=int)
    frame_size, sprite_size, frames_raw, sprites_raw = _square_regions(g)
    assert sprite_size == frame_size - 2 * FRAME_BORDER

    frames_by_pos: dict[tuple[int, int], dict] = {}
    for comp in frames_raw:
        r0, c0 = int(comp["bbox"][0]), int(comp["bbox"][1])
        frames_by_pos[(r0, c0)] = comp

    frame_rows = sorted({pos[0] for pos in frames_by_pos})
    frame_cols = sorted({pos[1] for pos in frames_by_pos})
    assert len(frame_rows) * len(frame_cols) == len(frames_raw)

    frame_grid: list[dict] = []
    border_mask = _ring_mask(frame_size, FRAME_BORDER)
    for ri, r0 in enumerate(frame_rows):
        for ci, c0 in enumerate(frame_cols):
            comp = frames_by_pos[(int(r0), int(c0))]
            region = crop_bbox(g, comp["bbox"])
            frame_color = int(Counter(map(int, region[border_mask].flatten())).most_common(1)[0][0])
            inner = region[FRAME_BORDER:-FRAME_BORDER, FRAME_BORDER:-FRAME_BORDER]
            _lab, num_holes = label_mask(inner != frame_color, connectivity=4)
            frame_grid.append(
                {
                    "color": frame_color,
                    "out_r": ri * frame_size,
                    "out_c": ci * frame_size,
                    "holes": int(num_holes),
                }
            )

    sprites_by_holes: dict[int, dict] = {}
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        _lab, num_holes = label_mask(sprite_region != sprite_color, connectivity=4)
        assert int(num_holes) not in sprites_by_holes
        sprites_by_holes[int(num_holes)] = {
            "color": sprite_color,
            "holes": int(num_holes),
            "content": sprite_region.copy(),
            "h": int(sprite_region.shape[0]),
            "w": int(sprite_region.shape[1]),
        }
    assert len(sprites_by_holes) == len(sprites_raw)
    assert len(frame_grid) == len(sprites_raw)

    out = np.full((len(frame_rows) * frame_size, len(frame_cols) * frame_size), BG, dtype=int)
    for frame in frame_grid:
        fr, fc = int(frame["out_r"]), int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr:fr + frame_size, fc:fc + frame_size] = frame_color

        assert int(frame["holes"]) in sprites_by_holes
        sprite = sprites_by_holes[int(frame["holes"])]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color
        sh, sw = int(sprite["h"]), int(sprite["w"])
        out[
            fr + FRAME_BORDER:fr + FRAME_BORDER + sh,
            fc + FRAME_BORDER:fc + FRAME_BORDER + sw,
        ] = content

    return out


def _place_isolated_seeds(side: int, count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    max_independent = (side * side + 1) // 2
    if count > max_independent:
        return None

    positions = [(r, c) for r in range(side) for c in range(side)]
    for _ in range(SEED_RETRIES):
        random.shuffle(positions)
        seeds: list[tuple[int, int]] = []
        for r, c in positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _frontier_for_component(
    comps: list[set[tuple[int, int]]],
    occupied: set[tuple[int, int]],
    idx: int,
    side: int,
) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in comps[idx]:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < side and 0 <= nc < side):
                continue
            if (nr, nc) in occupied:
                continue
            bad = False
            for j, other in enumerate(comps):
                if j == idx:
                    continue
                if (
                    (nr - 1, nc) in other
                    or (nr + 1, nc) in other
                    or (nr, nc - 1) in other
                    or (nr, nc + 1) in other
                ):
                    bad = True
                    break
            if not bad:
                out.add((nr, nc))
    return list(out)


def _make_hole_mask(side: int, hole_count: int, *, force_multicell: bool) -> np.ndarray | None:
    mask = np.zeros((side, side), dtype=bool)
    if hole_count == 0:
        return mask

    seeds = _place_isolated_seeds(side, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    occupied = set(seeds)
    growth_budget = random.randint(0, max(1, side * side // 2))

    for _ in range(growth_budget):
        choices = [idx for idx in range(len(comps)) if _frontier_for_component(comps, occupied, idx, side)]
        if not choices:
            break
        idx = random.choice(choices)
        frontier = _frontier_for_component(comps, occupied, idx, side)
        cell = random.choice(frontier)
        comps[idx].add(cell)
        occupied.add(cell)

    if force_multicell and all(len(comp) == 1 for comp in comps):
        expandable = [idx for idx in range(len(comps)) if _frontier_for_component(comps, occupied, idx, side)]
        if not expandable:
            return None
        idx = random.choice(expandable)
        frontier = _frontier_for_component(comps, occupied, idx, side)
        cell = random.choice(frontier)
        comps[idx].add(cell)
        occupied.add(cell)

    for r, c in occupied:
        mask[r, c] = True

    _lab, num_holes = label_mask(mask, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    if force_multicell and int(mask.sum()) <= int(hole_count):
        return None
    return mask


def _make_sprite(hole_count: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), sprite_color, dtype=int)
    inner_side = SPRITE_SIZE - 2 * SPRITE_BORDER
    mask = _make_hole_mask(
        inner_side,
        hole_count,
        force_multicell=hole_count > 0 and random.random() < 0.5,
    )
    if mask is None:
        return None
    for r, c in np.argwhere(mask):
        sprite[SPRITE_BORDER + r, SPRITE_BORDER + c] = BG
    return sprite


def _make_frame(hole_count: int, frame_color: int, *, force_multicell: bool) -> np.ndarray | None:
    frame = np.full((FRAME_SIZE, FRAME_SIZE), frame_color, dtype=int)
    inner_side = FRAME_SIZE - 2 * FRAME_BORDER
    mask = _make_hole_mask(inner_side, hole_count, force_multicell=force_multicell)
    if mask is None:
        return None
    for r, c in np.argwhere(mask):
        frame[FRAME_BORDER + r, FRAME_BORDER + c] = BG
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
) -> np.ndarray:
    expected = np.full((rows_grid * FRAME_SIZE, cols_grid * FRAME_SIZE), BG, dtype=int)
    sprite_by_k = {k: patch for k, patch in sprites}
    for idx, (hole_count, _frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = ri * FRAME_SIZE, ci * FRAME_SIZE
        frame_color = frame_colors[idx]
        expected[rr:rr + FRAME_SIZE, cc:cc + FRAME_SIZE] = frame_color
        content = sprite_by_k[hole_count].copy()
        content[content == BG] = frame_color
        expected[
            rr + FRAME_BORDER:rr + FRAME_BORDER + SPRITE_SIZE,
            cc + FRAME_BORDER:cc + FRAME_BORDER + SPRITE_SIZE,
        ] = content
    return expected


def _reserve_strip(rows_grid: int, cols_grid: int) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    grid_h = rows_grid * FRAME_SIZE
    grid_w = cols_grid * FRAME_SIZE

    options: list[tuple[str, str]] = []
    if CANVAS_SIZE - grid_h >= SPRITE_SIZE + GRID_STRIP_GAP:
        options.extend([("horizontal", "top"), ("horizontal", "bottom")])
    if CANVAS_SIZE - grid_w >= SPRITE_SIZE + GRID_STRIP_GAP:
        options.extend([("vertical", "left"), ("vertical", "right")])
    assert options

    orientation, side = random.choice(options)
    if orientation == "horizontal":
        strip_h = CANVAS_SIZE - grid_h - GRID_STRIP_GAP
        c0 = random.randint(0, CANVAS_SIZE - grid_w)
        if side == "top":
            return (strip_h + GRID_STRIP_GAP, c0), (0, 0, strip_h, CANVAS_SIZE)
        return (0, c0), (grid_h + GRID_STRIP_GAP, 0, strip_h, CANVAS_SIZE)

    strip_w = CANVAS_SIZE - grid_w - GRID_STRIP_GAP
    r0 = random.randint(0, CANVAS_SIZE - grid_h)
    if side == "left":
        return (r0, strip_w + GRID_STRIP_GAP), (0, 0, CANVAS_SIZE, strip_w)
    return (r0, 0), (0, grid_w + GRID_STRIP_GAP, CANVAS_SIZE, strip_w)


def _sprite_slots(strip: tuple[int, int, int, int], count: int) -> list[tuple[int, int]] | None:
    sr, sc, sh, sw = strip
    rows = sh // SPRITE_SIZE
    cols = sw // SPRITE_SIZE
    if rows * cols < count:
        return None

    extra_r = sh - rows * SPRITE_SIZE
    extra_c = sw - cols * SPRITE_SIZE
    base_r = sr + (random.randint(0, extra_r) if extra_r > 0 else 0)
    base_c = sc + (random.randint(0, extra_c) if extra_c > 0 else 0)

    slots = [
        (base_r + rr * SPRITE_SIZE, base_c + cc * SPRITE_SIZE)
        for rr in range(rows)
        for cc in range(cols)
    ]
    random.shuffle(slots)
    return slots[:count]


def _gen_attempt() -> np.ndarray | None:
    n_items = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_items])
    hole_counts = list(range(n_items))

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n_items)
    sprite_colors = random.sample(available, n_items)

    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    forced_multicell_count = 1 if 1 in hole_counts else min(hole_counts)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(sprite_hole_perm):
        patch = _make_sprite(hole_count, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((hole_count, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(hole_counts):
        patch = _make_frame(
            hole_count,
            frame_colors[i],
            force_multicell=(hole_count == forced_multicell_count),
        )
        if patch is None:
            return None
        frames.append((hole_count, patch))

    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    (r0, c0), strip = _reserve_strip(rows_grid, cols_grid)

    for idx, (_hole_count, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = r0 + ri * FRAME_SIZE, c0 + ci * FRAME_SIZE
        canvas[rr:rr + FRAME_SIZE, cc:cc + FRAME_SIZE] = frame_patch

    slots = _sprite_slots(strip, len(sprites))
    if slots is None:
        return None
    for (hole_count, sprite_patch), (sr, sc) in zip(sprites, slots):
        del hole_count
        canvas[sr:sr + SPRITE_SIZE, sc:sc + SPRITE_SIZE] = sprite_patch

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, frame_colors, rows_grid, cols_grid)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    # Fusion guard: at least one frame must contain a genuine multi-cell hole component.
    multi_cell_frames = 0
    for _, frame_patch in frames:
        inner = frame_patch[FRAME_BORDER:-FRAME_BORDER, FRAME_BORDER:-FRAME_BORDER]
        hole_area = int((inner == BG).sum())
        _lab, hole_count = label_mask(inner == BG, connectivity=4)
        if hole_area > hole_count:
            multi_cell_frames += 1
    if multi_cell_frames == 0:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
