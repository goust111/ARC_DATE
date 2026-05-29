# Combo variant of 8698868d (index 12)
# combo_spec: D3/3+D4/0
# Name: 渐变框架反转着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for combo D3/3 + D4/0
=====================================

a) Concrete behavior and execution trace
----------------------------------------
- Original family:
  - A tight grid of large square frames.
  - Same number of smaller square sprites elsewhere.
  - Match by the number of 4-connected bg-hole components.
  - Output nests each sprite into its matched frame.
- Fusion change:
  - Each frame now has four border colors: top, bottom, left, right.
  - The frame interior is also partitioned by the same local geometry rule:
    every non-hole cell takes the color of its nearest edge, with a fixed tie
    order. So the frame already previews the output palette logic.
  - After nesting, recoloring is inverted relative to the original task:
    sprite holes stay bg; sprite solid cells fuse into the frame by taking the
    nearest-edge color at their destination cells.
- Small trace for a 6x6 frame with edge colors
  - top=2, bottom=1, left=3, right=4
  - and a 4x4 sprite mask
      7 7 7 7
      7 8 8 7
      7 7 8 7
      7 7 7 7
  - Original task would collapse everything inside to one frame color.
  - This combo instead produces a four-zone gradient base and then keeps only
    the sprite holes as 8:
      border keeps the four side colors
      interior solid cells use nearest-edge color
      the three hole cells remain 8
  - The difference is therefore twofold at once:
    1. frame color is no longer a single color,
    2. the preserved bg hole mask comes from the sprite, not the frame holes.

b) Complexity calibration
-------------------------
- A naive multi-color frame could be too unconstrained, so the design uses only
  four edge colors per sample and keeps their orientation consistent across the
  whole grid. That gives repeated evidence for the same rule.
- The interior coloring rule is geometric rather than value-based: nearest edge
  color. This is visible directly from the frame itself and does not require
  hidden color ordering.
- The sprite-to-frame matching key remains unique hole count, so the task keeps
  the original combinatorial core instead of degenerating into pure recoloring.

c) Implicit assumption review for main()
----------------------------------------
- The original solver assumed large frames appear as monochromatic components.
  That fails immediately here, so the new solver detects sprites first and then
  scans for square frame tiles by local border structure.
- The original line `hole_mask = inner != frame_color` is invalid because a
  frame interior now legitimately contains several colors. The correct hole mask
  is `inner == BG`.
- The original output step used a single `frame_color`. That also fails. The
  new output must color each nested solid pixel by destination position.
- New assumption introduced by the fused solver:
  - top row is uniform,
  - bottom row is uniform,
  - left and right columns excluding corners are uniform,
  - interior non-bg cells follow the nearest-edge rule.
  The generator enforces exactly these properties.

d) Placement legality
---------------------
- Frames are still placed as one tight grid of size 1-3 by 1-3 with 2-6 total
  tiles, so they do not overlap and stay in-bounds.
- Sprites are still placed only on remaining bg cells with padding, so no two
  sprites touch and no sprite touches the frame grid.
- Output placement remains exact because sprite size is always frame_size - 2.

e) Visual observability
-----------------------
- A solver can see the gradient rule directly in the input frames: interior
  cells near a given edge already share that edge color.
- In the output, sprite solids blend into that same gradient while sprite holes
  stay bg. This makes the inversion highly visible because every nonzero-hole
  sprite leaves cutouts.
- Across 2-4 examples, a human can infer:
  1. match by hole count,
  2. move sprite into the frame,
  3. keep its holes as bg,
  4. color its solid cells using the local frame side-color geometry.

f) Visual cleanliness
---------------------
- There are only three visual layers per frame tile:
  - four side colors arranged structurally,
  - bg frame holes in the input,
  - bg sprite holes in the output.
- No extra markers are added, and the compact grid remains easy to parse.

g) Change inferability and anti-degeneracy checks
-------------------------------------------------
- The key color rule is structural and repeated: nearest edge color, not an
  arbitrary palette map.
- The generator is designed so output interior solid cells span multiple frame
  colors. This prevents collapse to a single-color recolor variant.
- A useful quantitative check after implementation is:
  - count, per output frame, how many nested solid pixels disagree with a
    single-color baseline such as "all solids take the top-edge color";
  - this must be positive in generated samples, otherwise the gradient rule is
    not actually affecting the result.
"""


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
EDGE_NAMES = ("top", "bottom", "left", "right")
SIDE_PRIORITY = ("top", "bottom", "left", "right")
FRAME_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _rows in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _cols in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        count = _rows * _cols
        if N_RANGE[0] <= count <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(count, []).append((_rows, _cols))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _nearest_edge_name(size: int, r: int, c: int) -> str:
    distances = {
        "top": int(r),
        "bottom": int(size - 1 - r),
        "left": int(c),
        "right": int(size - 1 - c),
    }
    return min(SIDE_PRIORITY, key=lambda name: (distances[name], SIDE_PRIORITY.index(name)))


def _frame_color_at(edge_colors: dict[str, int], size: int, r: int, c: int) -> int:
    r = int(r)
    c = int(c)
    if r == 0:
        return int(edge_colors["top"])
    if r == size - 1:
        return int(edge_colors["bottom"])
    if c == 0:
        return int(edge_colors["left"])
    if c == size - 1:
        return int(edge_colors["right"])
    return int(edge_colors[_nearest_edge_name(size, r, c)])


def _render_frame_base(frame_size: int, edge_colors: dict[str, int]) -> np.ndarray:
    tile = np.full((frame_size, frame_size), BG, dtype=int)
    for r in range(frame_size):
        for c in range(frame_size):
            tile[r, c] = int(_frame_color_at(edge_colors, frame_size, r, c))
    return tile


def _validate_frame_patch(region: np.ndarray) -> dict | None:
    frame_size = int(region.shape[0])
    if region.shape[0] != region.shape[1]:
        return None
    if np.any(region[0, :] == BG) or np.any(region[-1, :] == BG):
        return None
    if np.any(region[1:-1, 0] == BG) or np.any(region[1:-1, -1] == BG):
        return None

    top = int(region[0, 0])
    bottom = int(region[-1, 0])
    left = int(region[1, 0])
    right = int(region[1, -1])
    edge_colors = {"top": top, "bottom": bottom, "left": left, "right": right}
    if len(set(edge_colors.values())) != 4:
        return None
    if not np.all(region[0, :] == top):
        return None
    if not np.all(region[-1, :] == bottom):
        return None
    if not np.all(region[1:-1, 0] == left):
        return None
    if not np.all(region[1:-1, -1] == right):
        return None

    inner = region[1:-1, 1:-1]
    for r in range(1, frame_size - 1):
        for c in range(1, frame_size - 1):
            val = int(region[r, c])
            if val == BG:
                continue
            want = int(_frame_color_at(edge_colors, frame_size, r, c))
            if val != want:
                return None

    hole_mask = inner == BG
    _lab, num_holes = label_mask(hole_mask, connectivity=4)
    return {"edge_colors": edge_colors, "holes": int(num_holes)}


def _find_sprite_candidates(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    candidates: list[dict] = []
    min_fill_ratio = 0.5
    for comp in comps:
        bbox = comp["bbox"]
        h = int(bbox_h(bbox))
        w = int(bbox_w(bbox))
        area = int(comp["area"])
        if h != w or not (4 <= h <= 6):
            continue
        if area < int(h * h * min_fill_ratio):
            continue
        patch = crop_bbox(grid, bbox)
        sprite_color = int(comp["color"])
        if not np.all(patch[0, :] == sprite_color):
            continue
        if not np.all(patch[-1, :] == sprite_color):
            continue
        if not np.all(patch[:, 0] == sprite_color):
            continue
        if not np.all(patch[:, -1] == sprite_color):
            continue
        hole_mask = patch == BG
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        candidates.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
                "content": patch.copy(),
                "h": h,
                "w": w,
                "bbox": bbox,
            }
        )
    return candidates


def _select_sprites(sprite_candidates: list[dict]) -> list[dict]:
    assert sprite_candidates
    size_counter = Counter(int(sprite["h"]) for sprite in sprite_candidates)
    sprite_size = int(size_counter.most_common(1)[0][0])
    return [sprite for sprite in sprite_candidates if int(sprite["h"]) == sprite_size]


def _find_frame_grid(grid: np.ndarray, frame_size: int) -> tuple[int, int, list[dict]]:
    H, W = grid.shape
    candidates: list[dict] = []
    pos_map: dict[tuple[int, int], dict] = {}
    for r0 in range(H - frame_size + 1):
        for c0 in range(W - frame_size + 1):
            region = grid[r0 : r0 + frame_size, c0 : c0 + frame_size]
            parsed = _validate_frame_patch(region)
            if parsed is None:
                continue
            item = {
                "inp_r": int(r0),
                "inp_c": int(c0),
                "edge_colors": parsed["edge_colors"],
                "holes": int(parsed["holes"]),
            }
            candidates.append(item)
            pos_map[(int(r0), int(c0))] = item

    assert candidates

    best: tuple[int, int, int, int] | None = None
    best_frames: list[dict] | None = None
    valid_shapes = sorted(
        [(rows, cols) for rows in range(1, 4) for cols in range(1, 4) if 2 <= rows * cols <= 6],
        key=lambda rc: rc[0] * rc[1],
        reverse=True,
    )
    for cand in candidates:
        r0 = int(cand["inp_r"])
        c0 = int(cand["inp_c"])
        for rows, cols in valid_shapes:
            frames: list[dict] = []
            ok = True
            for ri in range(rows):
                for ci in range(cols):
                    pos = (int(r0 + ri * frame_size), int(c0 + ci * frame_size))
                    if pos not in pos_map:
                        ok = False
                        break
                    tile = dict(pos_map[pos])
                    tile["out_r"] = int(ri * frame_size)
                    tile["out_c"] = int(ci * frame_size)
                    frames.append(tile)
                if not ok:
                    break
            if not ok:
                continue
            score = (rows * cols, -r0, -c0, -rows)
            if best is None or score > best:
                best = score
                best_frames = frames

    assert best_frames is not None
    n_frames = len(best_frames)
    rows = len({int(frame["out_r"]) for frame in best_frames})
    cols = n_frames // rows
    return int(rows), int(cols), sorted(best_frames, key=lambda frame: (frame["out_r"], frame["out_c"]))


def _build_nested_content(frame_size: int, edge_colors: dict[str, int], sprite_patch: np.ndarray) -> np.ndarray:
    content = _render_frame_base(frame_size, edge_colors)[1:-1, 1:-1]
    content = np.array(content, dtype=int).copy()
    content[np.array(sprite_patch, dtype=int) == BG] = BG
    return content


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to gradient frames by connected-hole count, then compact the frame grid.

    Input structure:
    - bg is fixed to azure (8).
    - Large square frames are packed in a tight grid. Each frame uses four edge colors:
      top, bottom, left, and right are all different, and interior non-hole cells already
      follow the nearest-edge color pattern.
    - Smaller monochrome square sprites are scattered elsewhere. Their bg cells form the
      hole components used for matching.

    Rule:
    - Count bg-hole components inside each frame interior and inside each sprite.
    - Pair every frame with the sprite having the same hole count.
    - Output only the compact frame grid. Keep each frame's four-color geometry.
    - Inside the matched frame, preserve sprite holes as bg, but recolor sprite solid cells
      by the frame color at that exact destination position, namely the nearest border side.

    Key insight:
    - The pairing key is still the number of connected holes, but "frame color" is now
      position-dependent rather than a single flat color: the local edge zone determines it.
    """
    g = np.array(grid, dtype=int)

    sprite_candidates = _find_sprite_candidates(g)
    sprites = _select_sprites(sprite_candidates)
    sprite_size = int(sprites[0]["h"])
    frame_size = int(sprite_size + 2)

    n_rows, n_cols, frames = _find_frame_grid(g, frame_size)

    sprite_by_holes = {int(sprite["holes"]): sprite for sprite in sprites}
    assert len(sprite_by_holes) == len(sprites)

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for frame in frames:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        edge_colors = frame["edge_colors"]
        out[fr : fr + frame_size, fc : fc + frame_size] = _render_frame_base(frame_size, edge_colors)

        sprite = sprite_by_holes[int(frame["holes"])]
        content = _build_nested_content(frame_size, edge_colors, np.array(sprite["content"], dtype=int))
        out[fr + 1 : fr + 1 + sprite_size, fc + 1 : fc + 1 + sprite_size] = content

    return out


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` interior cells of a bordered square so they stay 4-isolated."""
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
    """Build a sprite with exactly `hole_count` 4-connected bg hole components."""
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite

    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    components: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for i, component in enumerate(components):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_holes)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
            for r, c in component:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_holes:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(components)
                        if j != i
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            component.add(cell)
            all_holes.add(cell)
            sprite[cell[0], cell[1]] = BG

    _lab, num_holes = label_mask(sprite == BG, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return sprite


def _make_gradient_frame(frame_size: int, hole_count: int, edge_colors: dict[str, int]) -> np.ndarray | None:
    """Build an input frame with a four-side gradient and isolated single-pixel bg holes."""
    frame = _render_frame_base(frame_size, edge_colors)
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

    _lab, num_holes = label_mask(frame[1:-1, 1:-1] == BG, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return frame


def _sample_sprite_colors(count: int, frame_palette: list[int]) -> list[int]:
    remaining = [color for color in FRAME_COLOR_POOL if color not in set(frame_palette)]
    if not remaining:
        remaining = list(FRAME_COLOR_POOL)
    return [random.choice(remaining) for _ in range(count)]


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    edge_colors: dict[str, int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_holes = {int(hole_count): np.array(patch, dtype=int) for hole_count, patch in sprites}

    for idx, (hole_count, _frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(ri * frame_size)
        cc = int(ci * frame_size)
        expected[rr : rr + frame_size, cc : cc + frame_size] = _render_frame_base(frame_size, edge_colors)
        content = _build_nested_content(frame_size, edge_colors, sprite_by_holes[int(hole_count)])
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    return expected


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = int(frame_size - 2)

    grid_pixel_h = int(rows_grid * frame_size)
    grid_pixel_w = int(cols_grid * frame_size)
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None

    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    frame_palette = random.sample(FRAME_COLOR_POOL, 4)
    edge_colors = {
        "top": int(frame_palette[0]),
        "bottom": int(frame_palette[1]),
        "left": int(frame_palette[2]),
        "right": int(frame_palette[3]),
    }
    sprite_colors = _sample_sprite_colors(n_frames, frame_palette)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n_frames)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, hole_count, int(sprite_colors[i]))
        if patch is None:
            return None
        sprites.append((int(hole_count), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for hole_count in hole_counts:
        patch = _make_gradient_frame(frame_size, int(hole_count), edge_colors)
        if patch is None:
            return None
        frames.append((int(hole_count), patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_hole_count, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _hole_count, sprite_patch in sprites:
        try:
            sr, sc = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, int(sr), int(sc), transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
        edge_colors,
        rows_grid,
        cols_grid,
        frame_size,
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
