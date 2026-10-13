from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_union, colors_present, crop_bbox, find_components, paste, random_sprite


_GEN_RETRY_LIMIT = 400
_MAP_SIZE_RANGE = (2, 6)
_BRUSH_SIZE_RANGE = (2, 5)
_LAYOUT_GAP_RANGE = (2, 6)


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_sprite_patches(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    comps = find_components(g, bg=Color.BLACK, connectivity=4, monochromatic=False)
    assert len(comps) >= 2, "expected at least two foreground groups"

    if len(comps) == 2:
        return [crop_bbox(g, comp["bbox"]) for comp in comps]

    best_score = None
    best_pair = None
    n = len(comps)
    for mask_bits in range(1, 1 << (n - 1)):
        left = [0]
        right = []
        for idx in range(1, n):
            if mask_bits & (1 << (idx - 1)):
                left.append(idx)
            else:
                right.append(idx)
        if not right:
            continue

        bbox_a = bbox_union(comps[idx]["bbox"] for idx in left)
        bbox_b = bbox_union(comps[idx]["bbox"] for idx in right)
        assert bbox_a is not None and bbox_b is not None
        if _rectangles_overlap(bbox_a, bbox_b):
            continue

        patch_a = crop_bbox(g, bbox_a)
        patch_b = crop_bbox(g, bbox_b)
        color_counts = sorted(
            [len(colors_present(patch_a, exclude={Color.BLACK})), len(colors_present(patch_b, exclude={Color.BLACK}))]
        )
        if color_counts != [1, 2]:
            continue

        score = (
            patch_a.size + patch_b.size,
            int(np.count_nonzero(patch_a == Color.BLACK) + np.count_nonzero(patch_b == Color.BLACK)),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (patch_a, patch_b)

    assert best_pair is not None, "failed to recover the two sprites"
    return [best_pair[0], best_pair[1]]


def _extract_roles(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    map_patch = None
    brush_patch = None
    for patch in _find_sprite_patches(grid):
        fg_colors = colors_present(patch, exclude={Color.BLACK})
        if len(fg_colors) == 2:
            assert set(fg_colors) == {Color.RED, Color.BLUE}, "map must use red and blue"
            assert map_patch is None, "expected a single two-color map"
            map_patch = patch
        elif len(fg_colors) == 1:
            assert brush_patch is None, "expected a single one-color brush"
            brush_patch = patch
        else:
            raise AssertionError("sprite roles must be determined by 2 colors vs 1 color")

    assert map_patch is not None and brush_patch is not None
    brush_color = colors_present(brush_patch, exclude={Color.BLACK})[0]
    brush_mask = brush_patch == brush_color
    assert np.any(brush_mask), "brush cannot be empty"
    assert not np.array_equal(brush_mask, np.rot90(brush_mask, 2)), "brush must change under 180-degree rotation"
    return map_patch, brush_mask


def _render_output(map_patch: np.ndarray, brush_mask: np.ndarray, *, rotate_blue: bool = True) -> np.ndarray:
    map_h, map_w = map_patch.shape
    brush_h, brush_w = brush_mask.shape
    out = np.full((map_h * brush_h, map_w * brush_w), Color.BLACK, dtype=int)
    flipped_mask = np.rot90(brush_mask, 2)

    for r in range(map_h):
        for c in range(map_w):
            cell = int(map_patch[r, c])
            if cell == Color.BLACK:
                continue
            if cell == Color.RED:
                mask = brush_mask
            else:
                assert cell == Color.BLUE, "map cells must be black, red, or blue"
                mask = flipped_mask if rotate_blue else brush_mask
            stamp = np.where(mask, cell, Color.BLACK).astype(int)
            paste(out, stamp, r * brush_h, c * brush_w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a red/blue stamp map into a large painting by pasting one brush mask in two orientations.

    Input structure:
    - The black background contains two separated sprites.
    - One sprite is a rectangular map whose visible cells are red and blue, with optional black holes inside the map bbox.
    - The other sprite is a single-color brush mask whose shape is not invariant under a 180-degree rotation.

    Rule:
    - Distinguish roles by foreground color count: the two-color sprite is the map and the one-color sprite is the brush.
    - Crop both sprites to their bounding boxes.
    - Produce an output whose size is `(map height * brush height, map width * brush width)`.
    - For each map cell, fill the corresponding brush-sized block: black cells stay black, red cells paste the brush as-is in red, and blue cells paste the 180-degree-rotated brush in blue.

    Key insight:
    - The coarse map color carries two meanings at once: it selects both the paint color and whether the fine brush mask must be flipped by 180 degrees.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    map_patch, brush_mask = _extract_roles(g)
    return _render_output(map_patch, brush_mask)


def _sample_map_dims(brush_h: int, brush_w: int) -> tuple[int, int]:
    max_map_h = min(_MAP_SIZE_RANGE[1], 30 // brush_h)
    max_map_w = min(_MAP_SIZE_RANGE[1], 30 // brush_w)
    assert max_map_h >= _MAP_SIZE_RANGE[0] and max_map_w >= _MAP_SIZE_RANGE[0]

    low_h = 3 if max_map_h >= 3 and random.random() < 0.8 else _MAP_SIZE_RANGE[0]
    low_w = 3 if max_map_w >= 3 and random.random() < 0.8 else _MAP_SIZE_RANGE[0]
    return random.randint(low_h, max_map_h), random.randint(low_w, max_map_w)


def _perimeter_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                coords.append((r, c))
    return coords


def _gen_map_patch(h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), Color.BLACK, dtype=int)
    border = _perimeter_coords(h, w)
    for r, c in border:
        patch[r, c] = random.choice([Color.RED, Color.BLUE])

    red_seed, blue_seed = random.sample(border, 2)
    patch[red_seed] = Color.RED
    patch[blue_seed] = Color.BLUE

    if h > 2 and w > 2:
        want_black = random.random() < 0.75
        black_weight = 4 if want_black else 1
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                patch[r, c] = random.choices(
                    [Color.BLACK, Color.RED, Color.BLUE],
                    weights=[black_weight, 3, 3],
                    k=1,
                )[0]
        if want_black and not np.any(patch[1:-1, 1:-1] == Color.BLACK):
            rr = random.randint(1, h - 2)
            cc = random.randint(1, w - 2)
            patch[rr, cc] = Color.BLACK

    assert np.any(patch == Color.RED) and np.any(patch == Color.BLUE)
    return patch


def _pick_brush_color() -> int:
    palette = [
        Color.RED,
        Color.BLUE,
        Color.GREEN,
        Color.YELLOW,
        Color.GREY,
        Color.PINK,
        Color.ORANGE,
        Color.TEAL,
        Color.MAROON,
    ]
    weights = [3, 3, 1, 1, 1, 1, 1, 1, 1]
    return int(random.choices(palette, weights=weights, k=1)[0])


def _gen_brush_patch(h: int, w: int) -> np.ndarray | None:
    brush_color = _pick_brush_color()
    for _ in range(120):
        sprite = random_sprite(
            h,
            w,
            density=random.uniform(0.45, 0.8),
            symmetry="not_symmetric",
            color_palette=[brush_color],
            connectivity=4,
            background=Color.BLACK,
        )
        mask = sprite == brush_color
        flipped = np.rot90(mask, 2)
        if np.array_equal(mask, flipped):
            continue
        if np.count_nonzero(mask != flipped) < 2:
            continue
        return sprite.astype(int)
    return None


def _split_padding(used: int) -> tuple[int, int]:
    free = min(6, 30 - used)
    total = random.randint(0, free)
    left = random.randint(0, total)
    return left, total - left


def _assemble_input(map_patch: np.ndarray, brush_patch: np.ndarray) -> np.ndarray | None:
    mh, mw = map_patch.shape
    bh, bw = brush_patch.shape
    gap = random.randint(*_LAYOUT_GAP_RANGE)

    layout = random.choice(["horizontal", "vertical"])
    if layout == "horizontal":
        used_h = max(mh, bh)
        used_w = mw + gap + bw
        if used_h > 30 or used_w > 30:
            return None
        top, bottom = _split_padding(used_h)
        left, right = _split_padding(used_w)
        grid = np.full((used_h + top + bottom, used_w + left + right), Color.BLACK, dtype=int)
        left_patch, right_patch = (map_patch, brush_patch) if random.random() < 0.5 else (brush_patch, map_patch)
        lp_h, lp_w = left_patch.shape
        rp_h, rp_w = right_patch.shape
        lp_r = top + random.randint(0, used_h - lp_h)
        rp_r = top + random.randint(0, used_h - rp_h)
        paste(grid, left_patch, lp_r, left)
        paste(grid, right_patch, rp_r, left + lp_w + gap)
        return grid

    used_h = mh + gap + bh
    used_w = max(mw, bw)
    if used_h > 30 or used_w > 30:
        return None
    top, bottom = _split_padding(used_h)
    left, right = _split_padding(used_w)
    grid = np.full((used_h + top + bottom, used_w + left + right), Color.BLACK, dtype=int)
    top_patch, bottom_patch = (map_patch, brush_patch) if random.random() < 0.5 else (brush_patch, map_patch)
    tp_h, tp_w = top_patch.shape
    bp_h, bp_w = bottom_patch.shape
    tp_c = left + random.randint(0, used_w - tp_w)
    bp_c = left + random.randint(0, used_w - bp_w)
    paste(grid, top_patch, top, tp_c)
    paste(grid, bottom_patch, top + tp_h + gap, bp_c)
    return grid


def _semantic_oracle(inp: np.ndarray, map_patch: np.ndarray, brush_patch: np.ndarray, expected: np.ndarray) -> bool:
    try:
        decoded_map, decoded_brush_mask = _extract_roles(inp)
        if not np.array_equal(decoded_map, map_patch):
            return False
        if not np.array_equal(decoded_brush_mask, brush_patch != Color.BLACK):
            return False

        out = main(inp)
        if not np.array_equal(out, expected):
            return False

        if np.array_equal(expected, _render_output(map_patch, brush_patch != Color.BLACK, rotate_blue=False)):
            return False

        if max(inp.shape) > 30 or max(expected.shape) > 30:
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    brush_h = random.randint(*_BRUSH_SIZE_RANGE)
    brush_w = random.randint(*_BRUSH_SIZE_RANGE)
    map_h, map_w = _sample_map_dims(brush_h, brush_w)

    map_patch = _gen_map_patch(map_h, map_w)
    brush_patch = _gen_brush_patch(brush_h, brush_w)
    if brush_patch is None:
        return None

    expected = _render_output(map_patch, brush_patch != Color.BLACK)
    inp = _assemble_input(map_patch, brush_patch)
    if inp is None:
        return None

    if _semantic_oracle(inp, map_patch, brush_patch, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 3")
