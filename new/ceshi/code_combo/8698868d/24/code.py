# Combo variant of 8698868d (index 24)
# combo_spec: D1/2+D8/0+D12/0
# Name: 面积匹配+多像素洞+八连通

from __future__ import annotations

# Design analysis for combo D1/2 + D8/0 + D12/0
# a) Concrete behavior and trace:
#    The original task used the same statistic on both sides:
#      - frame_key = number of 4-connected hole components inside the frame.
#      - sprite_key = number of 4-connected hole components inside the sprite.
#    The fused variant breaks that symmetry in two coupled ways:
#      - frame_key becomes the number of 8-connected hole components.
#      - sprite_key becomes total hole area in pixels.
#    A concrete small trace:
#      - Suppose a frame interior contains one diagonal bg pair plus two isolated bg singletons.
#        Under 4-connectivity that diagonal pair splits into 2 components, so the old frame key
#        would be 4. Under 8-connectivity the same pair merges, so the new frame key is 3.
#      - Suppose a sprite has one single 4-connected hole blob of area 3. The old sprite key
#        would be 1, but the new sprite key is 3.
#      - Original logic would compare 4 (frame) with 1 (sprite) and refuse the match.
#        The fused logic compares 3 (frame 8-hole count) with 3 (sprite hole area) and matches.
#    This example shows why the three changes are not independent: multi-pixel diagonal frame
#    holes are what make the 8-connectivity change visible, and the sprite must use area rather
#    than component count to meet the new frame statistic.
# b) Complexity calibration:
#    The generator does not allow arbitrary frame-hole geometry. Instead each frame contains
#    exactly one diagonal 2-pixel hole component plus several 8-isolated singleton holes.
#    This keeps the reasoning chain visible and stable:
#      - every frame has a genuinely multi-pixel hole component (D8/0 is always active);
#      - every frame has frame_4_count = frame_8_count + 1 (D12/0 is always active);
#      - every sprite is built with one connected hole blob whose area is in 2..N+1, so
#        sprite_area and sprite_component_count visibly differ (D1/2 is always active).
#    Using keys 2..N+1 rather than 0..N-1 avoids degenerate cases where area and component
#    count coincide at 0 or 1 and makes the new rule observable in 2-4 examples.
# c) Implicit assumptions review against the original solver:
#    - `find_components(... monochromatic=True, connectivity=4)` still assumes each frame and
#      sprite body is a single monochrome object. The generator preserves solid borders and uses
#      only bg-colored holes, so object extraction remains valid even though hole connectivity
#      changes to 8 on the frame side.
#    - Square filtering `area >= 0.5 * size^2` assumes holes do not hollow the objects too much.
#      The generator caps sprite hole area at 7 and frame hole area at key+1, so fill ratios stay
#      comfortably above the threshold for all frame sizes 6..8.
#    - The original line `label_mask(inner != frame_color, connectivity=4)` encoded the hidden
#      assumption "frame holes are judged by 4-connectivity". The combo explicitly breaks this,
#      so the solver must switch the frame-side call to `connectivity=8`.
#    - The original sprite loop assumed "matching statistic = sprite hole component count". The
#      combo breaks that too, so the solver must replace that count by `np.sum(hole_mask)`.
#    - The greedy first-unused matching loop assumes keys are unique. The generator guarantees a
#      bijection by using consecutive keys 2..N+1 on both sides.
# d) Placement legality:
#    Frames are still placed as one tight rows x cols lattice with no gaps; sprites are placed on
#    the remaining bg canvas via `random_free_location(..., padding=1)`, so no two same-colored
#    objects can merge and no sprite touches the frame grid.
# e) Visual observability:
#    A human looking at examples can see three repeated cues:
#      - frame holes are not all single pixels anymore;
#      - diagonal touching inside frames must count as one group, otherwise the numbers do not line up;
#      - sprites often show one larger cavity whose area, not its component count, determines which
#        frame receives it.
#    Because every sample contains several keys in the same grid, the pairing signal has repeated
#    support instead of relying on a single exceptional object.
# f) Visual cleanliness:
#    The scene still has only two object families: one compact frame grid and scattered sprites.
#    Holes appear only inside these squares; there are no extra markers, legends, or overlapping
#    annotations. Restricting frame holes to "one diagonal pair + isolated singles" keeps the
#    interiors readable rather than noisy.
# g) Change inferability:
#    No color-id ordering is used. Colors only identify object bodies. The recoloring rule remains
#    local and structural: sprite hole pixels become the surrounding frame color after insertion.
#    The match itself is driven entirely by visible geometry.
# h) New assumptions introduced by this combo solver and how the generator enforces them:
#    - Frame matching now assumes frame 8-hole counts are distinct. The generator uses one copy of
#      each key in 2..N+1.
#    - Sprite matching now assumes sprite hole area is distinct and exactly equals one frame key.
#      The generator builds one connected hole blob of each area in 2..N+1.
#    - The output insertion still assumes `sprite_size = frame_size - 2`; the generator keeps that
#      original geometric relation unchanged.
#    - The solver treats every non-sprite-color pixel inside a sprite bbox as a hole pixel. The
#      generator never introduces third colors inside sprites, so that assumption is safe.

from collections import Counter
from functools import lru_cache
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
GEN_RETRIES = 300
MIN_FILL_RATIO = 0.5
KEY_START = 2


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _rows in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _cols in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        n_items = _rows * _cols
        if N_RANGE[0] <= n_items <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(n_items, []).append((_rows, _cols))


def _cell_bit(inner_side: int, r: int, c: int) -> int:
    return 1 << (int(r) * int(inner_side) + int(c))


def _forbidden_mask(inner_side: int, cells: tuple[tuple[int, int], ...]) -> int:
    blocked = 0
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < inner_side and 0 <= cc < inner_side:
                    blocked |= _cell_bit(inner_side, rr, cc)
    return blocked


@lru_cache(maxsize=None)
def _frame_component_template(inner_side: int, hole_count: int) -> tuple[tuple[tuple[int, int], ...], ...] | None:
    """
    Build one deterministic template with exactly `hole_count` 8-connected hole components:
    one diagonal 2-pixel component plus `hole_count-1` isolated singleton components.
    """
    all_cells = tuple((r, c) for r in range(inner_side) for c in range(inner_side))
    diag_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(inner_side - 1):
        for c in range(inner_side - 1):
            diag_pairs.append(((r, c), (r + 1, c + 1)))
            diag_pairs.append(((r, c + 1), (r + 1, c)))

    @lru_cache(maxsize=None)
    def choose_singletons(start_idx: int, need: int, blocked: int) -> tuple[tuple[int, int], ...] | None:
        if need == 0:
            return ()
        for idx in range(start_idx, len(all_cells)):
            cell = all_cells[idx]
            if blocked & _cell_bit(inner_side, cell[0], cell[1]):
                continue
            rest = choose_singletons(
                idx + 1,
                need - 1,
                blocked | _forbidden_mask(inner_side, (cell,)),
            )
            if rest is not None:
                return (cell,) + rest
        return None

    for pair in diag_pairs:
        singles = choose_singletons(0, hole_count - 1, _forbidden_mask(inner_side, pair))
        if singles is None:
            continue
        components = [pair]
        components.extend((cell,) for cell in singles)
        return tuple(components)
    return None


def _max_frame_key(frame_size: int) -> int:
    inner_side = frame_size - 2
    max_key = 0
    for hole_count in range(KEY_START, 10):
        if _frame_component_template(inner_side, hole_count) is None:
            break
        max_key = hole_count
    return max_key


def _transform_inner_cell(r: int, c: int, inner_side: int, variant: int) -> tuple[int, int]:
    last = inner_side - 1
    if variant == 0:
        return r, c
    if variant == 1:
        return c, last - r
    if variant == 2:
        return last - r, last - c
    if variant == 3:
        return last - c, r
    if variant == 4:
        return r, last - c
    if variant == 5:
        return last - r, c
    if variant == 6:
        return c, r
    return last - c, last - r


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


def _build_frame_grid(grid: np.ndarray, frame_size: int, frames: list[dict]) -> tuple[list[dict], int, int]:
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
            hole_mask = inner != frame_color
            _labels, hole_count = label_mask(hole_mask, connectivity=8)
            frame_grid.append(
                {
                    "color": int(frame_color),
                    "out_r": int(row_idx * frame_size),
                    "out_c": int(col_idx * frame_size),
                    "holes": int(hole_count),
                }
            )
    return frame_grid, n_rows, n_cols


def _build_sprites(grid: np.ndarray, sprites_raw: list[dict]) -> list[dict]:
    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(grid, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        sprites.append(
            {
                "color": int(sprite_color),
                "hole_area": int(np.sum(hole_mask)),
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
            if int(frame["holes"]) != int(sprite["hole_area"]):
                continue
            matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprite
            used_sprites.add(int(idx))
            break
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by sprite hole area versus frame 8-connected hole count,
    then place each sprite inside its matched frame and recolor the sprite holes.

    Input structure:
    - The background is azure (8). Larger square frames form one tight rectangular grid.
    - Smaller square sprites of various colors are scattered elsewhere on the background.
    - Frame interiors can contain multi-pixel hole regions; sprite interiors can contain one
      or more bg-colored cavities.

    Rule:
    - For each frame, count the number of hole components inside the frame interior using
      8-connectivity, so diagonal touching hole pixels belong to the same component.
    - For each sprite, measure the total number of hole pixels rather than the number of
      hole components.
    - Pair every frame with the sprite whose hole area equals the frame's 8-connected hole
      count, then insert that sprite inside the frame border.
    - In the output, keep only the compact frame grid and recolor every sprite hole pixel to
      the enclosing frame color.

    Key insight:
    The solver must combine two different statistics: frames are keyed by 8-connected hole
    component count, while sprites are keyed by total hole area. Multi-pixel diagonal frame
    holes are the cue that 8-connectivity matters.
    """
    g = np.array(grid, dtype=int)
    frame_size, _sprite_size, frames, sprites_raw = _infer_object_sizes(g)
    frame_grid, n_rows, n_cols = _build_frame_grid(g, frame_size, frames)
    sprites = _build_sprites(g, sprites_raw)
    matches = _match_frames_to_sprites(frame_grid, sprites)

    out = np.full((n_rows * frame_size, n_cols * frame_size), BG, dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
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


def _grow_connected_hole(size: int, hole_area: int) -> set[tuple[int, int]] | None:
    inner_side = size - 2
    if hole_area < 0 or hole_area > inner_side * inner_side:
        return None
    if hole_area == 0:
        return set()

    seed = (random.randrange(inner_side), random.randrange(inner_side))
    shape = {seed}
    while len(shape) < hole_area:
        frontier: list[tuple[int, int]] = []
        for r, c in shape:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < inner_side and 0 <= nc < inner_side and (nr, nc) not in shape:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        shape.add(random.choice(frontier))

    return {(r + 1, c + 1) for r, c in shape}


def _make_sprite(size: int, hole_area: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), sprite_color, dtype=int)
    hole_cells = _grow_connected_hole(size, hole_area)
    if hole_cells is None:
        return None
    for r, c in hole_cells:
        sprite[r, c] = BG

    hole_mask = sprite != sprite_color
    actual_area = int(np.sum(hole_mask))
    _labels, hole_components = label_mask(hole_mask, connectivity=4)
    if actual_area != hole_area:
        return None
    if hole_area > 0 and hole_components != 1:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    inner_side = frame_size - 2
    components = _frame_component_template(inner_side, hole_count)
    if components is None:
        return None

    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    variant = random.randrange(8)
    multi_pixel_seen = False
    for comp in components:
        if len(comp) > 1:
            multi_pixel_seen = True
        for r, c in comp:
            tr, tc = _transform_inner_cell(r, c, inner_side, variant)
            frame[1 + tr, 1 + tc] = BG

    hole_mask = frame[1:-1, 1:-1] != frame_color
    _lab8, num8 = label_mask(hole_mask, connectivity=8)
    _lab4, num4 = label_mask(hole_mask, connectivity=4)
    if num8 != hole_count:
        return None
    if num4 <= num8:
        return None
    if not multi_pixel_seen:
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
    sprite_by_key = {key: patch for key, patch in sprites}
    for idx, (key, _patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = ri * frame_size, ci * frame_size
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        content = np.array(sprite_by_key[key], dtype=int).copy()
        content[content == BG] = int(frame_color)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _feasible_ns(frame_size: int) -> list[int]:
    sprite_size = frame_size - 2
    max_sprite_area = (sprite_size - 2) ** 2
    max_key = min(max_sprite_area, _max_frame_key(frame_size))
    max_n = max_key - KEY_START + 1
    return [n for n in sorted(_LAYOUTS_BY_N) if n <= max_n]


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    feasible_ns = _feasible_ns(frame_size)
    if not feasible_ns:
        return None

    n_items = random.choice(feasible_ns)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_items])
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_items)
    sprite_colors = random.sample(available, n_items)

    hole_keys = list(range(KEY_START, KEY_START + n_items))
    frame_key_perm = hole_keys[:]
    sprite_key_perm = hole_keys[:]
    random.shuffle(frame_key_perm)
    random.shuffle(sprite_key_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for idx, key in enumerate(sprite_key_perm):
        patch = _make_sprite(sprite_size, key, sprite_colors[idx])
        if patch is None:
            return None
        sprites.append((key, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for idx, key in enumerate(frame_key_perm):
        patch = _make_frame(frame_size, key, frame_colors[idx])
        if patch is None:
            return None
        frames.append((key, patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_key, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _key, sprite_patch in sprites:
        try:
            rr, cc = random_free_location(canvas, sprite_patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, sprite_patch, rr, cc, transparent=BG)

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
