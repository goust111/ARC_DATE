# Combo variant of 8698868d (index 14)
# combo_spec: D4/0+D6/0
# Name: 反转着色且保留原精灵

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_ROOT = next((parent for parent in _THIS_FILE.parents if (parent / "solvers" / "arc_utils.py").exists()), None)
if _ROOT is None:
    raise RuntimeError("Could not locate project root containing solvers/arc_utils.py")
_SOLVERS_DIR = _ROOT / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for combo D4/0 + D6/0
=====================================

a) Concrete behavior and execution trace
----------------------------------------
- The input family stays the same:
  - Azure background (8).
  - Larger square frames form a tight rectangular grid.
  - Same-count smaller square sprites are scattered elsewhere on background.
  - Frames and sprites are matched only by the number of 4-connected hole components.
- Original task:
  - Build a cropped frame-grid-only output.
  - For each matched sprite, place it into the frame interior.
  - Recolor sprite holes to the frame color, so the output contains no bg.
- Fused variant:
  - Keep the full input canvas as the output carrier.
  - Rebuild each frame square in place.
  - Copy the matched sprite into the frame interior with inverted recoloring:
    sprite solid cells become frame color, sprite hole cells stay bg.
  - Leave the original scattered source sprites untouched.
- Small trace:
  - Suppose a sprite is
      2 2 2 2
      2 8 8 2
      2 2 8 2
      2 2 2 2
    and its matched frame color is 4.
  - Original nesting produced
      2 2 2 2
      2 4 4 2
      2 2 4 2
      2 2 2 2
    inside a cropped frame grid, and the scattered source sprite disappeared.
  - New nesting produces
      4 4 4 4
      4 8 8 4
      4 4 8 4
      4 4 4 4
    inside the frame, while the original scattered sprite remains visible elsewhere.
  - So the solver must infer both "copy instead of move" and "invert the recoloring target".

b) Complexity calibration
-------------------------
- The matching key is still a single structural feature: connected-hole count.
- The two added effects are visually strong but not combinatorially explosive:
  - keep_source makes the output full-canvas and preserves the source exemplars,
  - inversion makes frame interiors retain visible azure cutouts.
- Generator constraints still use 2-6 objects and consecutive hole counts, so every sample has at least one nonzero-hole sprite.
  This prevents the inversion from collapsing into an all-solid frame output.

c) Implicit assumption review for main()
----------------------------------------
- `find_components(..., bg=8, monochromatic=True)` still assumes the family uses fixed bg 8 and isolated monochrome square objects.
  Generator keeps both guarantees.
- Square classification by size still assumes frames are the largest square components and sprites are the next largest size.
  The input layout is unchanged, so this remains valid.
- Frame hole counting via `inner != frame_color` still works because input frames are not altered by the combo.
- Sprite hole counting via `sprite_region != sprite_color` still works because source sprites remain plain sprite-color plus bg.
- The original cropped output allocation is no longer valid under keep_source.
  We must start from `out = g.copy()` instead.
- The original recoloring line targeted non-sprite-color cells.
  That would destroy the holes we now need to preserve.
  The combo therefore must recolor sprite-colored cells and leave bg cells unchanged.
- The combo introduces a new assumption: everything outside frame squares should remain unchanged.
  Generator guarantees that only frame squares are transformed zones.

d) Placement legality
---------------------
- The frame grid is still placed fully inside the canvas with no gaps.
- Sprites are still placed in remaining background using `random_free_location(..., padding=1)`,
  which prevents overlap with frames or each other and avoids accidental component merging.
- Sprite size remains exactly `frame_size - 2`, so copying into the frame interior never clips.
- Because output rewrites only the frame squares, preserved source sprites outside the grid stay intact by construction.

e) Visual observability
-----------------------
- A human can compare input and output and see the same sprite shapes twice:
  once at the scattered source location and once embedded in the matching frame.
- The inversion is also directly visible because embedded sprites with holes now show azure cutouts,
  unlike the original family where frame outputs had no bg at all.
- Across a few examples this is enough to infer:
  1. pairing by hole-count,
  2. copy rather than move,
  3. invert the nested recoloring.

f) Visual cleanliness
---------------------
- The combo does not stack multiple annotations on one region.
- Frame areas show one clean composite: a frame-colored square with bg hole silhouettes.
- Outer canvas areas show only the untouched source sprites.
- The one-cell placement padding preserves whitespace so the duplicated sprites do not visually fuse with the frame grid.

g) Change inferability
----------------------
- The color rule remains local and structural rather than palette-dependent:
  - inside frames: sprite solids -> frame color, sprite holes -> bg,
  - outside frames: leave original sprite colors unchanged.
- This avoids hidden numeric-color logic and gives the solver repeated evidence for the same transformation pattern.

Generator-side non-degeneracy checks
------------------------------------
- Require at least one non-bg source pixel outside frame squares, so keep_source is visible.
- Require at least one bg pixel inside the transformed frame squares, so inversion is visible.
- Because the oracle checks `main(canvas) == expected`, these visibility checks measure real fused behavior, not just solver non-crashing.
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


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], int, int]:
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        bbox = comp["bbox"]
        height = int(bbox_h(bbox))
        width = int(bbox_w(bbox))
        area = int(comp["area"])
        if height == width and 4 <= height <= 10 and area >= int(height * height * min_fill_ratio):
            square_regions.append((height, comp))

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

    frame_positions = [(int(frame["bbox"][0]), int(frame["bbox"][1])) for frame in frames]
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
            _labels, num_holes = label_mask(hole_mask, connectivity=4)
            frame_grid.append(
                {
                    "color": frame_color,
                    "inp_r": int(inp_r),
                    "inp_c": int(inp_c),
                    "holes": int(num_holes),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _labels, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    return frame_grid, sprites, frame_size, sprite_size


def _match_by_holes(frames: list[dict], sprites: list[dict]) -> dict[tuple[int, int], dict]:
    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frames:
        for idx, sprite in enumerate(sprites):
            if idx not in used_sprites and int(frame["holes"]) == int(sprite["holes"]):
                matches[(int(frame["inp_r"]), int(frame["inp_c"]))] = sprite
                used_sprites.add(int(idx))
                break
    return matches


def _frame_mask(shape: tuple[int, int], frames: list[dict], frame_size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        rr = int(frame["inp_r"])
        cc = int(frame["inp_c"])
        mask[rr : rr + frame_size, cc : cc + frame_size] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then copy each sprite into its frame with inverted recoloring.

    Input structure:
    - bg is fixed azure (8). Larger square frames form a tight grid somewhere on the canvas, while smaller square sprites are scattered outside that grid.
    - Every frame has a solid 1-pixel border in its own color and may contain bg holes; every sprite is mostly one color and may also contain bg holes.

    Rule:
    - Count 4-connected hole components inside each frame interior and inside each sprite.
    - Pair each frame with the unique sprite having the same hole count.
    - Keep the full canvas in the output. At every frame location, rebuild the whole square in frame color, then place the matched sprite inside with inverted recoloring: sprite-colored cells become frame color and sprite holes stay bg.
    - Leave the original scattered source sprites unchanged in their original positions.

    Key insight:
    The structural match is still determined only by connected-hole count, but the result is now a full-canvas copy effect plus inverted nesting, so the source sprite remains visible and the embedded version preserves its hole silhouette.
    """
    g = np.array(grid, dtype=int)
    frames, sprites, frame_size, _sprite_size = _extract_scene(g)
    matches = _match_by_holes(frames, sprites)

    out = g.copy()
    for frame in frames:
        fr = int(frame["inp_r"])
        fc = int(frame["inp_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = matches.get((fr, fc))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content == sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` interior cells that are pairwise 4-separated."""
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
    """Build a sprite with solid border and exactly `hole_count` 4-connected bg-hole components."""
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
                    nr = r + dr
                    nc = c + dc
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

    _labels, num_holes = label_mask(sprite != sprite_color, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build an input-style frame with exactly `hole_count` isolated bg holes."""
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

    _labels, num_holes = label_mask(frame != frame_color, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return frame


def _build_expected_output(
    canvas: np.ndarray,
    frames: list[dict],
    sprites: list[tuple[int, np.ndarray]],
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    """Build the full-canvas expected output for the fused combo."""
    expected = np.array(canvas, dtype=int).copy()
    sprite_by_holes = {int(hole_count): patch for hole_count, patch in sprites}
    for frame in frames:
        rr = int(frame["r"])
        cc = int(frame["c"])
        holes = int(frame["holes"])
        frame_color = int(frame["color"])

        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = np.array(sprite_by_holes[holes], dtype=int).copy()
        content[content != BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _behavior_metrics(inp: np.ndarray, out: np.ndarray, frames: list[dict], frame_size: int) -> dict[str, int]:
    """
    Quantify the two fused behaviors:
    - preserved source content outside frame squares,
    - visible bg cutouts inside transformed frame squares.
    """
    frame_records = [{"inp_r": int(frame["r"]), "inp_c": int(frame["c"])} for frame in frames]
    frame_mask = _frame_mask(inp.shape, frame_records, frame_size)
    outside_mask = ~frame_mask
    source_pixels_outside_frames = int(np.count_nonzero(inp[outside_mask] != BG))
    outside_mismatch = int(np.count_nonzero(inp[outside_mask] != out[outside_mask]))
    bg_pixels_inside_frames = int(np.count_nonzero((out == BG) & frame_mask))
    changed_pixels_inside_frames = int(np.count_nonzero((inp != out) & frame_mask))
    return {
        "source_pixels_outside_frames": source_pixels_outside_frames,
        "outside_mismatch": outside_mismatch,
        "bg_pixels_inside_frames": bg_pixels_inside_frames,
        "changed_pixels_inside_frames": changed_pixels_inside_frames,
    }


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
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
    for idx, hole_count in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, hole_count, sprite_colors[idx])
        if patch is None:
            return None
        sprites.append((int(hole_count), patch))

    frame_patches: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_size, hole_count, frame_colors[idx])
        if patch is None:
            return None
        frame_patches.append((int(hole_count), patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    frame_records: list[dict] = []
    for idx, (hole_count, frame_patch) in enumerate(frame_patches):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
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

    metrics = _behavior_metrics(canvas, out, frame_records, frame_size)
    if metrics["source_pixels_outside_frames"] <= 0:
        return None
    if metrics["outside_mismatch"] != 0:
        return None
    if metrics["bg_pixels_inside_frames"] <= 0:
        return None
    if metrics["changed_pixels_inside_frames"] <= 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
