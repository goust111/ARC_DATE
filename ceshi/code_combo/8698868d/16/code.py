# Combo variant of 8698868d (index 16)
# combo_spec: D5/0+D6/0
# Name: 全画布输出保留精灵

from __future__ import annotations

# Design analysis for combo D5/0 + D6/0
# a) Concrete behavior and trace:
#    The input still contains one tight grid of larger square frames and the same number of smaller square sprites
#    scattered elsewhere on an azure background. The original task cropped the answer down to only the frame grid,
#    so every source sprite vanished automatically. This combo changes two linked stages together:
#    - D5 keeps the whole input canvas instead of cropping to the frame lattice.
#    - D6 keeps each scattered source sprite in place instead of erasing it.
#    Example: if a 2x2 frame grid of size 6 starts at (3,5) on a 20x20 canvas and one matching sprite sits at
#    (13,1), the old output was a 12x12 frame-only crop. The combo output stays 20x20, redraws the 2x2 grid at
#    (3,5), nests each matched sprite into its frame with holes recolored to frame color, and still leaves the sprite
#    visible at (13,1). Visually this is "copy into frame", not "move into frame".
# b) Complexity calibration:
#    The matching signal remains exactly one structural feature: the number of 4-connected hole components. Keeping
#    unique counts across 2-6 pairs preserves a bijection and prevents ambiguous matching. The canvas is forced to be
#    larger than the frame grid and the grid is kept away from the outer border, so the full-canvas change is plainly
#    visible; retaining the source sprites makes the trajectory change equally visible.
# c) Implicit assumptions review:
#    - Square-region detection still relies on frames being the largest square components and sprites the next size
#      down. The generator preserves that by keeping sprite size = frame_size - 2 and by separating objects.
#    - Inferring the frame lattice from equal frame_size steps still works because the generator places frames in a
#      gapless rows x cols rectangle.
#    - The original solver stored output positions as compact offsets (ri*frame_size, ci*frame_size). That assumption
#      fails under D5, because output coordinates must stay on the original canvas. The combo solver stores and uses
#      each frame's input top-left coordinate instead.
#    - The original cropped canvas implicitly erased source sprites. Under D6 that assumption fails, so the combo
#      solver must start from g.copy() and only overwrite frame squares.
#    - Repainting each full frame square before inserting the matched sprite remains necessary; otherwise the input
#      frame's hole pixels would survive under the nested sprite.
# d) Placement legality:
#    The generator reserves canvas room for both the frame grid and scattered sprites, keeps at least one-cell outer
#    margin around the grid, and uses random_free_location(..., padding=1) so sprites never overlap frames or each
#    other. This prevents monochromatic components of equal color from merging.
# e) Visual observability:
#    Across 2-4 examples, a human can compare input and output and see two stable facts: the output size matches the
#    input size, and every source sprite still appears outside the grid while another copy appears inside a matched
#    frame. The hole-count pairing is still readable because each frame and sprite family offers multiple distinct
#    counts in one example.
# f) Visual cleanliness:
#    Only the frame regions are rewritten; outside them the output is exactly the original scattered sprites on clean
#    background. There is no extra overlay marker, no multi-object pile-up, and no need to read hidden numeric color
#    order.
# g) Color and rule inferability:
#    Colors are not remapped globally. A sprite keeps its own body color in both places, and only its hole pixels are
#    filled with the enclosing frame color. The surrounding frame supplies the structural cue for that recoloring, so
#    a solver can infer the rule from local context rather than from arbitrary color ids.
# h) New assumptions introduced by the combo solver and how the generator guarantees them:
#    - `out = g.copy()` assumes frame-grid exterior pixels should remain unchanged; generator restricts all intended
#      transformation to the frame squares.
#    - Matching by hole count assumes a one-to-one mapping; generator uses one copy of each count in both families.
#    - Inserting the sprite at `frame top-left + (1,1)` assumes sprite_size = frame_size - 2; generator constructs
#      sprites with that exact size.

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


def _find_solvers_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "solvers" / "arc_utils.py"
        if candidate.exists():
            return candidate.parent
    raise RuntimeError("could not locate solvers/arc_utils.py")


_SOLVERS_DIR = _find_solvers_dir()
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


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
for _rows in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _cols in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        n_items = _rows * _cols
        if N_RANGE[0] <= n_items <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(n_items, []).append((_rows, _cols))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _find_square_regions(grid: np.ndarray) -> dict[int, list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        box = comp["bbox"]
        height = int(bbox_h(box))
        width = int(bbox_w(box))
        area = int(comp["area"])
        if height == width and 4 <= height <= 10 and area >= int(height * height * MIN_FILL_RATIO):
            square_regions.append((height, comp))
    if not square_regions:
        raise AssertionError("no square regions found")

    by_size: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        by_size.setdefault(int(size), []).append(comp)
    return by_size


def _infer_object_sizes(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    sizes = _find_square_regions(grid)
    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)
    frames = sizes.get(frame_size, [])
    sprites = sizes.get(sprite_size, [])
    if not frames:
        raise AssertionError("no frames found")
    return frame_size, sprite_size, frames, sprites


def _build_frame_grid(grid: np.ndarray, frame_size: int, frames: list[dict]) -> list[dict]:
    frame_positions = [(int(frame["bbox"][0]), int(frame["bbox"][1])) for frame in frames]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict] = []
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            inp_r = int(min_r + row_idx * frame_size)
            inp_c = int(min_c + col_idx * frame_size)
            region = grid[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            if region.shape != (frame_size, frame_size):
                raise AssertionError("incomplete frame region")
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            inner = region[1:-1, 1:-1]
            _labels, num_holes = label_mask(inner != frame_color, connectivity=4)
            frame_grid.append(
                {
                    "color": int(frame_color),
                    "inp_r": int(inp_r),
                    "inp_c": int(inp_c),
                    "holes": int(num_holes),
                }
            )
    return frame_grid


def _build_sprites(grid: np.ndarray, sprites_raw: list[dict]) -> list[dict]:
    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(grid, comp["bbox"])
        sprite_color = int(comp["color"])
        _labels, num_holes = label_mask(sprite_region != sprite_color, connectivity=4)
        sprites.append(
            {
                "color": int(sprite_color),
                "holes": int(num_holes),
                "content": np.array(sprite_region, dtype=int).copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )
    return sprites


def _match_frames_to_sprites(frame_grid: list[dict], sprites: list[dict]) -> dict[tuple[int, int], dict]:
    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for idx, sprite in enumerate(sprites):
            if idx in used_sprites:
                continue
            if int(frame["holes"]) != int(sprite["holes"]):
                continue
            matches[(int(frame["inp_r"]), int(frame["inp_c"]))] = sprite
            used_sprites.add(int(idx))
            break
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered sprites to frame-grid cells by connected-hole count, then rewrite the
    frame grid in place while keeping the full original canvas and the source sprites.

    Input structure:
    - The background is azure (8). Larger square frames form a tight rectangular grid somewhere on the canvas.
    - Smaller square sprites of various colors are scattered elsewhere on the same canvas, separated from the grid.
    - A frame's interior contains isolated azure holes; a sprite may contain one or more connected azure hole regions.

    Rule:
    - Count 4-connected hole components for each frame interior and for each sprite.
    - Pair every frame with the sprite that has the same hole-component count.
    - Rebuild each frame square at its original canvas position, nest the matched sprite inside the border, and recolor
      the sprite's hole pixels to the frame color.
    - Keep the output the same size as the input and leave the original scattered sprites visible at their source
      positions.

    Key insight:
    The important invariant is the number of connected hole regions, and the trajectory is copy-not-move: the source
    sprite stays where it started while a recolored copy appears inside the matched frame on the full canvas.
    """
    g = np.array(grid, dtype=int)
    frame_size, _sprite_size, frames, sprites_raw = _infer_object_sizes(g)
    frame_grid = _build_frame_grid(g, frame_size, frames)
    sprites = _build_sprites(g, sprites_raw)
    matches = _match_frames_to_sprites(frame_grid, sprites)

    out = g.copy()
    for frame in frame_grid:
        fr = int(frame["inp_r"])
        fc = int(frame["inp_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        sprite = matches.get((fr, fc))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = int(frame_color)
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

    components: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_idx, comp in enumerate(components):
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
                        for other_idx, other in enumerate(components)
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

    _labels, num = label_mask(sprite != sprite_color, connectivity=4)
    if int(num) != hole_count:
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

    _labels, num = label_mask(frame != frame_color, connectivity=4)
    if int(num) != hole_count:
        return None
    return frame


def _build_expected_output(
    canvas: np.ndarray,
    frame_records: list[dict],
    sprites: list[tuple[int, np.ndarray]],
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.array(canvas, dtype=int).copy()
    sprite_by_holes = {int(hole_count): patch for hole_count, patch in sprites}
    for frame in frame_records:
        rr = int(frame["r"])
        cc = int(frame["c"])
        holes = int(frame["holes"])
        frame_color = int(frame["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        content = np.array(sprite_by_holes[holes], dtype=int).copy()
        content[content == BG] = int(frame_color)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _assign_sprite_colors_by_hole(hole_counts: list[int], frame_colors: list[int]) -> dict[int, int] | None:
    available = [color for color in range(10) if color != BG]
    frame_color_by_hole = {int(hole): int(frame_colors[idx]) for idx, hole in enumerate(hole_counts)}
    ordered_holes = list(map(int, hole_counts))
    random.shuffle(ordered_holes)
    assignment: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(ordered_holes):
            return True
        hole = int(ordered_holes[pos])
        candidates = [color for color in available if color not in used and color != int(frame_color_by_hole[hole])]
        random.shuffle(candidates)
        for color in candidates:
            assignment[hole] = int(color)
            used.add(int(color))
            if backtrack(pos + 1):
                return True
            used.remove(int(color))
            assignment.pop(hole, None)
        return False

    if not backtrack(0):
        return None
    return assignment


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

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n_frames)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)
    sprite_color_by_hole = _assign_sprite_colors_by_hole(hole_counts, frame_colors)
    if sprite_color_by_hole is None:
        return None

    sprites: list[tuple[int, np.ndarray]] = []
    for hole_count in sprite_hole_perm:
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(sprite_color_by_hole[int(hole_count)]))
        if patch is None:
            return None
        sprites.append((int(hole_count), patch))

    frame_patches: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_size, int(hole_count), int(frame_colors[idx]))
        if patch is None:
            return None
        frame_patches.append((int(hole_count), patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(FRAME_MARGIN, canvas_h - grid_pixel_h - FRAME_MARGIN)
    c0 = random.randint(FRAME_MARGIN, canvas_w - grid_pixel_w - FRAME_MARGIN)
    frame_records: list[dict] = []
    for idx, (hole_count, frame_patch) in enumerate(frame_patches):
        row_idx, col_idx = divmod(idx, cols_grid)
        rr = int(r0 + row_idx * frame_size)
        cc = int(c0 + col_idx * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch
        frame_records.append(
            {
                "holes": int(hole_count),
                "color": int(frame_colors[idx]),
                "r": int(rr),
                "c": int(cc),
            }
        )

    for _hole_count, sprite_patch in sprites:
        try:
            rr, cc = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, rr, cc, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(canvas, frame_records, sprites, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
