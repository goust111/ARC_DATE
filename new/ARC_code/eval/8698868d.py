from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import bbox_coords, bbox_h, bbox_w, crop_bbox, find_components, label_mask
import random
from arc_utils import label_mask, random_free_location, paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

    Input structure:
    - bg = azure (8). Large hollow squares (frames) arranged in a grid; smaller filled squares (sprites) placed elsewhere.
    - Each frame and sprite has a distinct color. Sprites may contain holes (non-sprite-color pixels).

    Rule:
    - Frames form a regular grid; some may be bg-colored (invisible).
    - Count connected hole regions inside each frame's interior and each sprite.
    - Pair each frame with the sprite sharing the same hole count.
    - Place the matched sprite inside the frame border; recolor sprite holes to the frame's color.

    Key insight: The matching key is the number of connected hole components, not hole area or color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    BG = 8  # Azure background for this puzzle family

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # Find square regions (lower fill threshold to catch sprites with holes).
    MIN_FILL_RATIO = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((int(h), comp))

    assert not (not square_regions)

    # Group by size - frames are larger, sprites are smaller.
    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(int(frame_size), [])
    sprites_raw = sizes.get(int(sprite_size), [])
    assert not (not frames)

    # Determine frame grid structure from frame positions.
    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, c in frame_positions)
    min_c = min(c for r, c in frame_positions)
    max_r = max(r for r, c in frame_positions)
    max_c = max(c for r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    # Build frame grid (including potential background-colored frames).
    frame_grid = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != int(frame_color)
            _lab, num_holes = label_mask(hole_mask, connectivity=4)

            frame_grid.append({"color": int(frame_color), "out_r": int(ri * frame_size), "out_c": int(ci * frame_size), "holes": int(num_holes)})

    # Process sprites - count holes in each.
    sprites = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append({"color": int(sprite_color), "holes": int(num_holes), "content": sprite_region.copy(), "h": int(sprite_region.shape[0]), "w": int(sprite_region.shape[1])})

    # Match frames to sprites by hole count.
    matches: dict[tuple[int, int], tuple[dict, dict]] = {}
    used_sprites: set[int] = set()
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if int(i) not in used_sprites and int(f["holes"]) == int(s["holes"]):
                matches[(int(f["out_r"]), int(f["out_c"]))] = (f, s)
                used_sprites.add(int(i))
                break

    # Build output grid.
    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for f in frame_grid:
        fr, fc = int(f["out_r"]), int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        if (fr, fc) in matches:
            _f, sprite = matches[(fr, fc)]
            content = np.array(sprite["content"], dtype=int).copy()
            sprite_color = int(sprite["color"])
            content[content != int(sprite_color)] = int(frame_color)

            sh, sw = int(sprite["h"]), int(sprite["w"])
            out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out

# --------------- generator (attempt_1) ---------------

# ── Configuration ──────────────────────────────────────────────────────
BG = 8                      # background color (solver hardcodes azure=8)
FRAME_SIZE_RANGE = [6, 7, 8]  # square frame side length
GRID_DIM_RANGE = (1, 4)     # rows/cols of the frame grid (exclusive upper)
N_RANGE = (2, 6)             # min/max frame count (inclusive)
CANVAS_MIN = 15              # minimum input grid side
CANVAS_MAX = 30              # maximum input grid side
SPRITE_PADDING = 2           # extra canvas room for sprite placement
SPRITE_PLACEMENT_PAD = 1     # padding around sprites when placing
HOLE_START_MAX = 1           # max starting hole count (0 or 1)
SEED_RETRIES = 50            # retries for placing isolated seed cells
GEN_RETRIES = 300            # outer generate_input retry budget

# Precompute valid grid layouts: {N: [(rows, cols), ...]}
_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` cells inside a (size x size) border, each mutually 4-isolated.
    Returns seed positions or None on failure."""
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
    """Build a size x size sprite with solid border and exactly `hole_count`
    4-connected bg-hole components in the interior (grown from isolated seeds)."""
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite
    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    # Initialize each component as its seed
    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    # Grow each component randomly; keep components 4-separated
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
                    # Must not be 4-adjacent to any OTHER component
                    if any((nr + dr2, nc + dc2) in other
                           for j, other in enumerate(comps) if j != ci
                           for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]):
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
    """Build an input-style frame: solid frame_color square with `hole_count`
    single-pixel bg cells scattered in interior (each 4-isolated)."""
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


def _build_expected_output(frames, sprites, frame_colors, rows_grid, cols_grid,
                           frame_size, sprite_size) -> np.ndarray:
    """Build the expected output grid: frame borders with matched sprites nested inside."""
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {k: spatch for k, spatch in sprites}
    for idx, (k, _) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * frame_size, ci * frame_size
        fc = frame_colors[idx]
        expected[rr:rr + frame_size, cc:cc + frame_size] = fc
        content = sprite_by_k[k].copy()
        content[content == BG] = fc
        expected[rr + 1:rr + 1 + sprite_size, cc + 1:cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    # 1. Pick N and grid layout
    N = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[N])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    # 2. Canvas size (must fit frame grid + room for sprites)
    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None
    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    # 3. Colors: N distinct non-bg per group; groups may overlap
    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, N)
    sprite_colors = random.sample(available, N)

    # 4. Hole counts: consecutive range start..start+N-1
    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - N)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + N))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    # 5. Build sprites and frames
    sprites: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, k, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((k, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(hole_counts):
        patch = _make_frame(frame_size, k, frame_colors[i])
        if patch is None:
            return None
        frames.append((k, patch))

    # 6. Place frame grid on canvas
    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_, fpatch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr:rr + frame_size, cc:cc + frame_size] = fpatch

    # 7. Place sprites in remaining bg area (non-overlapping)
    for _, spatch in sprites:
        try:
            r, c = random_free_location(canvas, spatch, transparent=BG,
                                        padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, spatch, r, c, transparent=BG)

    # 8. Oracle check
    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, frame_colors,
                                      rows_grid, cols_grid, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError('generate_input: failed after %d attempts' % GEN_RETRIES)
