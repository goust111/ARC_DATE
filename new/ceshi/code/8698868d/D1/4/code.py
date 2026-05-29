# Variant of 8698868d
# Dimension: D1 — 匹配条件：连通洞计数 (match_cond)
# Change: hole_counting — 精灵与框架各自内部的 4-连通背景色洞分量数目相等即配对... → 边缘颜色匹配

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for this D1 variant
# ---------------------------------------------------------------------------
#
# concrete_form:
#   - Frames stay as larger squares arranged in a tight grid; sprites stay as smaller
#     detached squares elsewhere on the canvas.
#   - The key visible change is on the border. Each object has:
#       * one dominant base color filling most pixels
#       * four short colored marker segments, one centered on each side
#       * the four side-marker colors are all distinct on that object
#   - A frame and its matching sprite share the same ordered side signature:
#       (top_marker, right_marker, bottom_marker, left_marker)
#     while their base colors may differ.
#   - Sprites may still contain bg holes inside; output nests the sprite inside the
#     matched frame and recolors only the sprite's bg holes to the frame base color.
#
# behavior_trace:
#   - Original logic at the matching step:
#       frame_holes == sprite_holes  -> pair
#   - Variant logic at the matching step:
#       frame_signature == sprite_signature  -> pair
#   - Small example:
#       frame A edges = (2,3,4,5), no need to inspect its interior
#       sprite X edges = (2,3,4,5), sprite Y edges = (5,4,3,2)
#       original solver would look at hole counts and could pair A with Y if counts fit;
#       variant must pair A with X because only X matches the ordered border colors.
#
# complexity_calibration:
#   - Using an ordered 4-tuple keeps the rule visible and strong enough to support up
#     to 6 objects without ambiguity.
#   - Marker segments are centered and short, so humans can isolate them as "edge tags"
#     rather than confusing them with arbitrary noise.
#   - Hole counts are intentionally not made unique anymore, preventing silent fallback
#     to the original D1 behavior.
#
# implicit_assumption_review_for_original_main:
#   - The original `find_components(..., monochromatic=True)` assumed every frame was
#     a single-color component. That breaks twice here: borders are multi-colored, and
#     adjacent frames in the tight grid merge into one large foreground block.
#     Fix: detect multicolor foreground components, recover sprites as detached square
#     components, and recover frames by slicing the large frame block into equal tiles.
#   - The original frame color was inferred from border majority. That is unsafe now
#     because border cells deliberately mix colors. Fix: infer the object base color as
#     the most common non-bg color over the whole square.
#   - The original sprite recolor step used `content != sprite_color`, which was harmless
#     when sprites were monochrome except for bg holes. In this variant that would erase
#     the edge markers. Fix: recolor only `content == BG`.
#   - The original frame-grid loop tolerated possible bg-colored slots. That was tied to
#     the old family wording; this variant keeps every frame explicit and non-bg, so the
#     generator guarantees a complete visible grid.
#
# placement_legality:
#   - Frames are pasted as a tight `rows x cols` square grid with no gaps.
#   - Sprites are placed with `random_free_location(..., padding=1)` on remaining bg
#     area, so they do not overlap one another or the frame block.
#   - Marker segments never touch corners; corners remain base-colored, so the outer box
#     stays visually square and the cropped bbox is stable.
#
# visual_observability:
#   - Across 2-4 ARC examples, the repeated cue is that each frame and one detached
#     sprite share the same edge-color order. This is directly visible without using
#     hidden numeric properties.
#   - Frames are kept clean internally to avoid users inferring hole-count matching.
#   - Sprites keep some holes so the output recolor effect remains visible, but those
#     holes vary independently of pairing.
#
# neatness_and_color_inference:
#   - Only four edge markers plus one base color are used per object. The base color
#     dominates area, so markers read as tags rather than clutter.
#   - Matching uses exact edge position (top/right/bottom/left), not numeric color ids.
#     Color values themselves are never ordered or ranked.
#
# new_main_assumptions_and_generator_guarantees:
#   - Every sprite is a detached square multicolor foreground component of size 4..6.
#   - The full frame grid is one larger foreground component whose bbox dimensions are
#     multiples of `frame_size`; sprites are exactly `frame_size-2`.
#   - Each side of every object has exactly one non-base marker color segment.
#   - Side-signatures are unique among frames/sprites, so matching is bijective.
#   - The generator constructs inputs to satisfy all of the above.


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
MAX_SPRITE_HOLES = 2

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _marker_span(size: int) -> tuple[int, int]:
    side_len = int(size - 2)
    seg_len = max(1, side_len // 2)
    start = 1 + (side_len - seg_len) // 2
    end = start + seg_len
    return int(start), int(end)


def _draw_edge_markers(patch: np.ndarray, base_color: int, signature: tuple[int, int, int, int]) -> np.ndarray:
    out = np.array(patch, dtype=int, copy=True)
    out[:, :] = np.where(out == BG, BG, int(base_color))
    top_col, right_col, bottom_col, left_col = map(int, signature)
    start, end = _marker_span(int(out.shape[0]))

    out[0, start:end] = top_col
    out[-1, start:end] = bottom_col
    out[start:end, 0] = left_col
    out[start:end, -1] = right_col
    return out


def _make_clean_frame(size: int, base_color: int, signature: tuple[int, int, int, int]) -> np.ndarray:
    patch = np.full((size, size), int(base_color), dtype=int)
    return _draw_edge_markers(patch, int(base_color), signature)


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


def _make_sprite_patch(
    size: int,
    hole_count: int,
    base_color: int,
    signature: tuple[int, int, int, int],
) -> np.ndarray | None:
    sprite = np.full((size, size), int(base_color), dtype=int)
    if hole_count > 0:
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
        max_grow = max(0, inner_cells // max(hole_count, 1) - 2)
        for ci, comp in enumerate(comps):
            grow = random.randint(0, min(MAX_SPRITE_HOLES, max_grow))
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

        _, num = label_mask(sprite == BG, connectivity=4)
        if int(num) != int(hole_count):
            return None

    return _draw_edge_markers(sprite, int(base_color), signature)


def _non_bg_counter(region: np.ndarray) -> Counter:
    vals = [int(v) for v in region.flatten() if int(v) != BG]
    return Counter(vals)


def _extract_base_and_signature(region: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    counter = _non_bg_counter(region)
    assert counter
    base_color = int(counter.most_common(1)[0][0])

    side_arrays = [
        region[0, 1:-1],
        region[1:-1, -1],
        region[-1, 1:-1],
        region[1:-1, 0],
    ]
    markers: list[int] = []
    for side in side_arrays:
        side_colors = [int(v) for v in side.tolist() if int(v) not in (BG, base_color)]
        assert side_colors
        markers.append(int(Counter(side_colors).most_common(1)[0][0]))

    return int(base_color), tuple(markers)


def _locate_frame_block_and_sprites(grid: np.ndarray) -> tuple[int, int, tuple[int, int, int, int], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")

    square_sprites: list[tuple[int, dict]] = []
    min_fill_ratio = 0.5
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 6 and area >= int(h * h * min_fill_ratio):
            square_sprites.append((int(h), comp))

    assert square_sprites

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_sprites:
        sizes.setdefault(int(size), []).append(comp)

    sprite_size = int(max(sizes.keys()))
    sprites = sizes[int(sprite_size)]
    frame_size = int(sprite_size + 2)

    frame_block = max(comps, key=lambda comp: int(comp["area"]))
    fb = tuple(map(int, frame_block["bbox"]))
    fb_h = int(bbox_h(fb))
    fb_w = int(bbox_w(fb))
    assert fb_h >= frame_size and fb_w >= frame_size
    assert fb_h % frame_size == 0 and fb_w % frame_size == 0
    return int(frame_size), int(sprite_size), fb, sprites


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)

    frame_size, _sprite_size, frame_block_bbox, sprites_raw = _locate_frame_block_and_sprites(g)
    min_r, min_c, max_r, max_c = map(int, frame_block_bbox)
    n_rows = int((max_r - min_r + 1) // frame_size)
    n_cols = int((max_c - min_c + 1) // frame_size)

    frame_grid = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            base_color, signature = _extract_base_and_signature(region)
            frame_grid.append(
                {
                    "base_color": int(base_color),
                    "signature": tuple(map(int, signature)),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        base_color, signature = _extract_base_and_signature(sprite_region)
        sprites.append(
            {
                "base_color": int(base_color),
                "signature": tuple(map(int, signature)),
                "content": np.array(sprite_region, dtype=int, copy=True),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if int(i) in used_sprites:
                continue
            if tuple(f["signature"]) == tuple(s["signature"]):
                matches[(int(f["out_r"]), int(f["out_c"]))] = s
                used_sprites.add(int(i))
                break

    assert len(matches) == len(frame_grid)

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for f in frame_grid:
        fr, fc = int(f["out_r"]), int(f["out_c"])
        frame_patch = _make_clean_frame(int(frame_size), int(f["base_color"]), tuple(f["signature"]))
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_patch

        sprite = matches[(fr, fc)]
        content = np.array(sprite["content"], dtype=int, copy=True)
        content[content == int(BG)] = int(f["base_color"])
        sh, sw = int(sprite["h"]), int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _build_expected_output(
    frames: list[dict],
    sprite_by_signature: dict[tuple[int, int, int, int], np.ndarray],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    for idx, frame in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * frame_size, ci * frame_size
        fpatch = _make_clean_frame(frame_size, int(frame["base_color"]), tuple(frame["signature"]))
        expected[rr : rr + frame_size, cc : cc + frame_size] = fpatch

        sprite = np.array(sprite_by_signature[tuple(frame["signature"])], dtype=int, copy=True)
        sprite[sprite == BG] = int(frame["base_color"])
        sh, sw = sprite.shape
        expected[rr + 1 : rr + 1 + sh, cc + 1 : cc + 1 + sw] = sprite
    return expected


def _sample_unique_signatures(n: int) -> list[tuple[int, int, int, int]] | None:
    available = [c for c in range(10) if c != BG]
    seen: set[tuple[int, int, int, int]] = set()
    signatures: list[tuple[int, int, int, int]] = []
    for _ in range(500):
        if len(signatures) == n:
            return signatures
        sig = tuple(map(int, random.sample(available, 4)))
        if sig in seen:
            continue
        seen.add(sig)
        signatures.append(sig)
    return None


def _pick_base_color(signature: tuple[int, int, int, int]) -> int:
    choices = [c for c in range(10) if c != BG and c not in set(map(int, signature))]
    return int(random.choice(choices))


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    h = random.randint(min_h, CANVAS_MAX)
    w = random.randint(min_w, CANVAS_MAX)

    signatures = _sample_unique_signatures(n)
    if signatures is None:
        return None
    random.shuffle(signatures)

    sprite_holes = [random.randint(0, MAX_SPRITE_HOLES) for _ in range(n)]
    if all(hc == 0 for hc in sprite_holes):
        sprite_holes[random.randrange(n)] = 1

    frames: list[dict] = []
    sprites: list[dict] = []
    for idx in range(n):
        sig = tuple(signatures[idx])
        frame_base = _pick_base_color(sig)
        sprite_base = _pick_base_color(sig)

        frame_patch = _make_clean_frame(frame_size, frame_base, sig)
        sprite_patch = _make_sprite_patch(sprite_size, sprite_holes[idx], sprite_base, sig)
        if sprite_patch is None:
            return None

        frames.append(
            {
                "signature": sig,
                "base_color": int(frame_base),
                "patch": frame_patch,
            }
        )
        sprites.append(
            {
                "signature": sig,
                "base_color": int(sprite_base),
                "patch": sprite_patch,
            }
        )

    canvas = np.full((h, w), BG, dtype=int)
    r0 = random.randint(0, h - grid_pixel_h)
    c0 = random.randint(0, w - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame["patch"]

    random.shuffle(sprites)
    for sprite in sprites:
        patch = sprite["patch"]
        try:
            rr, cc = random_free_location(canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, rr, cc, transparent=BG)

    expected = _build_expected_output(
        frames,
        {tuple(sp["signature"]): np.array(sp["patch"], dtype=int, copy=True) for sp in sprites},
        rows_grid,
        cols_grid,
        frame_size,
    )

    try:
        out = main(canvas)
    except Exception:
        return None

    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
