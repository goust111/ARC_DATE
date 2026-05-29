# Combo variant of 8698868d (index 2)
# combo_spec: D1/4+D3/3
# Name: 边缘颜色匹配渐变框架

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for this combo variant
# ---------------------------------------------------------------------------
#
# concrete_form:
#   - The frame grid remains a tight rows x cols block of larger squares, and sprites
#     remain detached smaller squares elsewhere on the canvas.
#   - The original single-color frame border is replaced with a corner-gradient border:
#       * each object has one dominant base color
#       * each side is split into two contiguous color segments
#       * these segments encode four ordered corner colors:
#           top    = (top_left,  top_right)
#           right  = (top_right, bottom_right)
#           bottom = (bottom_left, bottom_right)
#           left   = (top_left,  bottom_left)
#   - A frame and its matching sprite share the same ordered corner signature
#     (top_left, top_right, bottom_right, bottom_left), while their base fills may differ.
#   - Sprite holes are still bg-colored in the input. In the output, each hole pixel is
#     recolored by its quadrant inside the sprite:
#       top-left hole -> frame top-left corner color
#       top-right hole -> frame top-right corner color
#       bottom-right hole -> frame bottom-right corner color
#       bottom-left hole -> frame bottom-left corner color
#
# behavior_trace:
#   - Original family:
#       1. find large monochrome frames and smaller monochrome sprites
#       2. count connected bg-hole components in each
#       3. pair equal counts
#       4. recolor all sprite holes to the frame color
#   - Combo variant:
#       1. find one multicolor frame block and detached multicolor sprites
#       2. read each object's four corner colors from the segmented border
#       3. pair equal ordered corner signatures
#       4. keep the gradient frame border, nest the sprite, and recolor each hole to the
#          matched frame's corner color for that quadrant
#   - Small example:
#       frame A corners = (2,4,1,3), sprite X corners = (2,4,1,3), sprite Y corners = (3,2,4,1)
#       holes in X at top-left and bottom-right become colors 2 and 1 after nesting.
#       The original hole-count rule would miss both the pairing cue and the multi-color
#       recolor pattern.
#
# complexity_calibration:
#   - The signature uses four ordered corner colors, which is visually strong and gives
#     enough combinations for 2-6 objects without ambiguity.
#   - Each side is split into exactly two uniform segments. That is richer than a single
#     edge marker, but still simple enough for people to compare across examples.
#   - Frame interiors are kept clean so hole-count matching disappears completely instead
#     of competing with the new cue.
#
# implicit_assumption_review_for_original_main:
#   - `find_components(..., monochromatic=True)` no longer works because borders are
#     intentionally multicolored. The solver must use `monochromatic=False`.
#   - The original "largest square components are frames" assumption also breaks because
#     adjacent frames merge into one large connected foreground block. The solver now
#     identifies sprites as detached small squares and recovers frame tiles by slicing the
#     largest component bbox into equal `frame_size` tiles.
#   - Inferring frame color from border majority is invalid because the border is not
#     uniform anymore. The solver instead infers a dominant base color from all non-bg
#     pixels in the square, which works because corners stay base-colored and the interior
#     is mostly base-filled.
#   - Recoloring with `content != sprite_color` would destroy the gradient border. The new
#     solver recolors only `content == BG`.
#   - The original matching loop assumed hole counts form a bijection. Here the generator
#     makes every frame hole count equal to 0, so matching must rely on border signatures.
#
# placement_legality:
#   - Frames are pasted as a tight grid with no gaps, so the frame block is one connected
#     foreground component.
#   - Sprites are placed with `random_free_location(..., padding=1)` on remaining bg cells,
#     preventing overlap with each other and with the frame block.
#   - Hole cells are sampled only from sprite interiors, so they never break the outer
#     square bbox or touch the border segments.
#
# visual_observability:
#   - Across a few examples, each frame and exactly one detached sprite repeat the same
#     clockwise corner-color order. That is the visible pairing cue.
#   - The recolor rule is also visible because holes in different quadrants of the same
#     sprite turn into different border colors in the output.
#   - The mapping uses spatial structure only: corner order and pixel quadrant. It does not
#     depend on hidden properties such as numeric color ids or generation order.
#
# neatness_and_color_inference:
#   - Only one base color plus four corner colors are used per object, so the border stays
#     structured instead of noisy.
#   - The same four corner colors serve two purposes: object matching and hole recoloring.
#     That makes the combined rule coherent rather than two unrelated modifications.
#
# new_main_assumptions_and_generator_guarantees:
#   - Every sprite is a detached multicolor square of size `frame_size - 2`.
#   - The frame block bbox dimensions are exact multiples of `frame_size`.
#   - Every side consists of exactly two solid segments whose consistency determines one
#     ordered corner signature.
#   - Corner signatures are unique among objects, and every sprite has at least one hole.
#   - Generated examples use at least two recolor quadrants overall, so the gradient-based
#     recoloring is actually observable and not a degenerate single-color case.


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
GEN_RETRIES = 300
MAX_SPRITE_HOLES = 3

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _side_cut(size: int) -> int:
    side_len = int(size - 2)
    cut = int(side_len // 2)
    assert cut > 0 and side_len - cut > 0
    return cut


def _non_bg_counter(region: np.ndarray) -> Counter:
    return Counter(int(v) for v in region.flatten() if int(v) != BG)


def _draw_corner_gradient(
    patch: np.ndarray,
    base_color: int,
    corners: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.array(patch, dtype=int, copy=True)
    out[:, :] = np.where(out == BG, BG, int(base_color))

    top_left, top_right, bottom_right, bottom_left = map(int, corners)
    cut = _side_cut(int(out.shape[0]))

    top = out[0, 1:-1]
    top[:cut] = int(top_left)
    top[cut:] = int(top_right)

    right = out[1:-1, -1]
    right[:cut] = int(top_right)
    right[cut:] = int(bottom_right)

    bottom = out[-1, 1:-1]
    bottom[:cut] = int(bottom_left)
    bottom[cut:] = int(bottom_right)

    left = out[1:-1, 0]
    left[:cut] = int(top_left)
    left[cut:] = int(bottom_left)
    return out


def _make_frame_patch(size: int, base_color: int, corners: tuple[int, int, int, int]) -> np.ndarray:
    patch = np.full((size, size), int(base_color), dtype=int)
    return _draw_corner_gradient(patch, int(base_color), corners)


def _quadrant_index(r: int, c: int, h: int, w: int) -> int:
    top = float(r) < float(h) / 2.0
    left = float(c) < float(w) / 2.0
    if top and left:
        return 0
    if top and not left:
        return 1
    if (not top) and (not left):
        return 2
    return 3


def _recolor_holes_by_quadrant(content: np.ndarray, corners: tuple[int, int, int, int]) -> np.ndarray:
    out = np.array(content, dtype=int, copy=True)
    h, w = map(int, out.shape)
    for r, c in np.argwhere(out == BG):
        out[int(r), int(c)] = int(corners[_quadrant_index(int(r), int(c), h, w)])
    return out


def _segment_mode(values: np.ndarray, base_color: int) -> int:
    filtered = [int(v) for v in values.tolist() if int(v) not in (BG, int(base_color))]
    assert filtered
    return int(Counter(filtered).most_common(1)[0][0])


def _extract_base_and_corners(region: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    counter = _non_bg_counter(region)
    assert counter
    base_color = int(counter.most_common(1)[0][0])
    cut = _side_cut(int(region.shape[0]))

    top = region[0, 1:-1]
    right = region[1:-1, -1]
    bottom = region[-1, 1:-1]
    left = region[1:-1, 0]

    top_left = _segment_mode(top[:cut], base_color)
    top_right = _segment_mode(top[cut:], base_color)
    right_top = _segment_mode(right[:cut], base_color)
    bottom_right = _segment_mode(right[cut:], base_color)
    bottom_left = _segment_mode(bottom[:cut], base_color)
    right_bottom = _segment_mode(bottom[cut:], base_color)
    left_top = _segment_mode(left[:cut], base_color)
    left_bottom = _segment_mode(left[cut:], base_color)

    assert int(top_left) == int(left_top)
    assert int(top_right) == int(right_top)
    assert int(bottom_right) == int(right_bottom)
    assert int(bottom_left) == int(left_bottom)
    return int(base_color), (int(top_left), int(top_right), int(bottom_right), int(bottom_left))


def _locate_frame_block_and_sprites(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int, int, int], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    sprite_candidates: list[tuple[int, dict]] = []
    min_fill_ratio = 0.5
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 6 and area >= int(h * h * min_fill_ratio):
            sprite_candidates.append((int(h), comp))

    assert sprite_candidates

    sizes: dict[int, list[dict]] = {}
    for size, comp in sprite_candidates:
        sizes.setdefault(int(size), []).append(comp)

    sprite_size = int(max(sizes.keys()))
    sprites = list(sizes[int(sprite_size)])
    frame_size = int(sprite_size + 2)

    frame_block = max(comps, key=lambda comp: int(comp["area"]))
    frame_bbox = tuple(map(int, frame_block["bbox"]))
    block_h = int(bbox_h(frame_bbox))
    block_w = int(bbox_w(frame_bbox))
    assert block_h >= frame_size and block_w >= frame_size
    assert block_h % frame_size == 0 and block_w % frame_size == 0
    return int(frame_size), int(sprite_size), frame_bbox, sprites


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, int, int, int, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    frame_size, sprite_size, frame_bbox, sprites_raw = _locate_frame_block_and_sprites(g)
    min_r, min_c, max_r, max_c = map(int, frame_bbox)
    n_rows = int((max_r - min_r + 1) // frame_size)
    n_cols = int((max_c - min_c + 1) // frame_size)

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            base_color, corners = _extract_base_and_corners(region)
            frame_grid.append(
                {
                    "base_color": int(base_color),
                    "corners": tuple(map(int, corners)),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "hole_count": int(np.sum(region[1:-1, 1:-1] == BG)),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        base_color, corners = _extract_base_and_corners(sprite_region)
        sprites.append(
            {
                "base_color": int(base_color),
                "corners": tuple(map(int, corners)),
                "content": np.array(sprite_region, dtype=int, copy=True),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
                "hole_count": int(np.sum(sprite_region == BG)),
            }
        )

    return int(frame_size), int(sprite_size), int(n_rows), int(n_cols), frame_grid, sprites


def _match_frames_to_sprites(frame_grid: list[dict], sprites: list[dict]) -> dict[tuple[int, int], dict]:
    matches: dict[tuple[int, int], dict] = {}
    used: set[int] = set()
    for frame in frame_grid:
        for i, sprite in enumerate(sprites):
            if int(i) in used:
                continue
            if tuple(frame["corners"]) == tuple(sprite["corners"]):
                matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprite
                used.add(int(i))
                break
    assert len(matches) == len(frame_grid)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match detached sprites to a tight frame grid by segmented edge-gradient signatures,
    then nest each sprite and recolor its holes by the matched frame's corner colors.

    Input structure:
    - bg = azure (8). Larger square frames form one tight grid; smaller square sprites are
      scattered elsewhere.
    - Every frame and sprite has one dominant base color plus a multicolor border whose
      four sides are split into two segments, encoding four ordered corner colors.
    - Sprites may contain bg holes in their interiors; frames are kept internally solid.

    Rule:
    - Recover each object's ordered corner signature from the border segments.
    - Pair each frame with the sprite sharing the same signature.
    - Output only the compact frame grid. Keep each frame's gradient border, place the
      matched sprite inside it, and recolor every sprite hole by its quadrant:
      top-left/top-right/bottom-right/bottom-left use the corresponding frame corner color.

    Key insight:
    The important cue is the border's ordered color segmentation, not interior hole count.
    The same corner signature controls both object matching and the multi-color hole fill.
    """
    frame_size, _sprite_size, n_rows, n_cols, frame_grid, sprites = _parse_scene(grid)
    matches = _match_frames_to_sprites(frame_grid, sprites)

    out = np.full((n_rows * frame_size, n_cols * frame_size), int(BG), dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        corners = tuple(map(int, frame["corners"]))
        frame_patch = _make_frame_patch(int(frame_size), int(frame["base_color"]), corners)
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_patch

        sprite = matches[(fr, fc)]
        content = _recolor_holes_by_quadrant(np.array(sprite["content"], dtype=int, copy=True), corners)
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content
    return out


def _sample_unique_corner_signatures(n: int) -> list[tuple[int, int, int, int]] | None:
    palette = [c for c in range(10) if c != BG]
    signatures: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for _ in range(600):
        if len(signatures) == n:
            return signatures
        sig = tuple(map(int, random.sample(palette, 4)))
        if sig in seen:
            continue
        seen.add(sig)
        signatures.append(sig)
    return None


def _pick_base_color(corners: tuple[int, int, int, int]) -> int:
    choices = [c for c in range(10) if c != BG and c not in set(map(int, corners))]
    assert choices
    return int(random.choice(choices))


def _sample_hole_positions(size: int, hole_count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    chosen = random.sample(cells, hole_count)
    return [(int(r), int(c)) for r, c in chosen]


def _make_sprite_patch(
    size: int,
    base_color: int,
    corners: tuple[int, int, int, int],
    hole_positions: list[tuple[int, int]],
) -> np.ndarray:
    patch = np.full((size, size), int(base_color), dtype=int)
    for r, c in hole_positions:
        patch[int(r), int(c)] = int(BG)
    return _draw_corner_gradient(patch, int(base_color), corners)


def _build_expected_output(
    frames: list[dict],
    sprite_by_signature: dict[tuple[int, int, int, int], np.ndarray],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), int(BG), dtype=int)
    for idx, frame in enumerate(frames):
        ri = int(idx // cols_grid)
        ci = int(idx % cols_grid)
        rr = int(ri * frame_size)
        cc = int(ci * frame_size)
        corners = tuple(map(int, frame["corners"]))
        frame_patch = _make_frame_patch(int(frame_size), int(frame["base_color"]), corners)
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

        sprite = np.array(sprite_by_signature[corners], dtype=int, copy=True)
        sprite = _recolor_holes_by_quadrant(sprite, corners)
        sh, sw = sprite.shape
        expected[rr + 1 : rr + 1 + sh, cc + 1 : cc + 1 + sw] = sprite
    return expected


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    frame_size, _sprite_size, n_rows, n_cols, frame_grid, sprites = _parse_scene(grid)
    matches = _match_frames_to_sprites(frame_grid, sprites)

    signature_count = len({tuple(frame["corners"]) for frame in frame_grid})
    hole_recolor_palette: set[int] = set()
    quadrants_used: set[int] = set()
    sprite_holes = [int(sprite["hole_count"]) for sprite in sprites]
    frame_holes = [int(frame["hole_count"]) for frame in frame_grid]

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        sprite = matches[(fr, fc)]
        content = np.array(sprite["content"], dtype=int, copy=True)
        h = int(sprite["h"])
        w = int(sprite["w"])
        for r, c in np.argwhere(content == BG):
            q = _quadrant_index(int(r), int(c), h, w)
            quadrants_used.add(int(q))
            hole_recolor_palette.add(int(frame["corners"][q]))

    return {
        "frame_size": int(frame_size),
        "grid_slots": int(n_rows * n_cols),
        "unique_signatures": int(signature_count),
        "all_frame_holes_zero": int(all(v == 0 for v in frame_holes)),
        "sprite_hole_total": int(sum(sprite_holes)),
        "recolor_palette_size": int(len(hole_recolor_palette)),
        "quadrants_used": int(len(quadrants_used)),
        "hole_count_bijection_possible": int(sorted(frame_holes) == sorted(sprite_holes)),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = int(frame_size - 2)

    grid_h = int(rows_grid * frame_size)
    grid_w = int(cols_grid * frame_size)
    min_h = max(CANVAS_MIN, grid_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    h = random.randint(min_h, CANVAS_MAX)
    w = random.randint(min_w, CANVAS_MAX)

    signatures = _sample_unique_corner_signatures(n)
    if signatures is None:
        return None
    random.shuffle(signatures)

    frames: list[dict] = []
    sprites: list[dict] = []
    total_quadrants: set[int] = set()
    for idx in range(n):
        corners = tuple(map(int, signatures[idx]))
        frame_base = _pick_base_color(corners)
        sprite_base = _pick_base_color(corners)

        max_holes = min(MAX_SPRITE_HOLES, (sprite_size - 2) ** 2)
        if max_holes <= 0:
            return None
        hole_count = random.randint(1, max_holes)
        hole_positions = _sample_hole_positions(sprite_size, hole_count)
        for r, c in hole_positions:
            total_quadrants.add(_quadrant_index(int(r), int(c), sprite_size, sprite_size))

        frame_patch = _make_frame_patch(frame_size, frame_base, corners)
        sprite_patch = _make_sprite_patch(sprite_size, sprite_base, corners, hole_positions)

        frames.append(
            {
                "corners": corners,
                "base_color": int(frame_base),
                "patch": frame_patch,
            }
        )
        sprites.append(
            {
                "corners": corners,
                "base_color": int(sprite_base),
                "patch": sprite_patch,
            }
        )

    if len(total_quadrants) < 2:
        return None

    canvas = np.full((h, w), int(BG), dtype=int)
    r0 = random.randint(0, h - grid_h)
    c0 = random.randint(0, w - grid_w)
    for idx, frame in enumerate(frames):
        ri = int(idx // cols_grid)
        ci = int(idx % cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

    random.shuffle(sprites)
    for sprite in sprites:
        patch = np.array(sprite["patch"], dtype=int, copy=True)
        try:
            rr, cc = random_free_location(canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, rr, cc, transparent=BG)

    expected = _build_expected_output(
        frames,
        {tuple(sprite["corners"]): np.array(sprite["patch"], dtype=int, copy=True) for sprite in sprites},
        rows_grid,
        cols_grid,
        frame_size,
    )

    try:
        out = main(canvas)
        metrics = _fusion_metrics(canvas)
    except Exception:
        return None

    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    if metrics["unique_signatures"] != n:
        return None
    if metrics["all_frame_holes_zero"] != 1:
        return None
    if metrics["recolor_palette_size"] < 2:
        return None
    if metrics["quadrants_used"] < 2:
        return None
    if metrics["hole_count_bijection_possible"] != 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
