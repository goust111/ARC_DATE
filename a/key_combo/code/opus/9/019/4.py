from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
scale_sprite = _ARC_UTILS.scale_sprite


BG = int(Color.BLACK)
DIVIDER = int(Color.GREY)
SPRITE_COLOR = int(Color.MAROON)
UPSCALE = 3
PALETTE = [int(c) for c in (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL)]

_RNG = np.random.default_rng()
_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _all_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], bytes]] = set()
    for rot in range(4):
        rotated = np.rot90(mask, rot)
        for candidate in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, candidate.shape)), candidate.astype(np.uint8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(candidate.copy())
    return variants


_BASE_SHAPES = {
    3: [
        np.array([[1, 0], [1, 1]], dtype=bool),
    ],
    4: [
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    ],
}


def _choose_sprite_mask(count: int) -> np.ndarray:
    base = _BASE_SHAPES[count][int(_RNG.integers(len(_BASE_SHAPES[count])))]
    variants = [
        variant
        for variant in _all_variants(base)
        if variant.shape[0] > 1
        and variant.shape[1] > 1
        and max(int(np.sum(variant[r])) for r in range(variant.shape[0])) >= 2
        and max(int(np.sum(variant[:, c])) for c in range(variant.shape[1])) >= 2
    ]
    return variants[int(_RNG.integers(len(variants)))].copy()


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(map(int, grid.shape))).encode()).hexdigest()


def _separator_col(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(candidates) == 1
    return int(candidates[0])


def _row_majority_color(row: np.ndarray) -> int:
    colors, counts = np.unique(row[row != DIVIDER], return_counts=True)
    assert len(colors) > 0
    order = np.argsort(counts)[::-1]
    assert len(order) == 1 or int(counts[order[0]]) > int(counts[order[1]])
    return int(colors[order[0]])


def _compressed_sequence(left: np.ndarray) -> list[int]:
    majorities = [_row_majority_color(left[r]) for r in range(left.shape[0])]
    sequence: list[int] = []
    for color in majorities:
        if not sequence or sequence[-1] != color:
            sequence.append(int(color))
    assert 3 <= len(sequence) <= 5
    assert len(set(sequence)) == len(sequence)
    return sequence


def _frame_bbox(right: np.ndarray) -> tuple[int, int, int, int]:
    frames = []
    for comp in find_components(right, bg=BG, colors=[DIVIDER], connectivity=4, monochromatic=True):
        patch = crop_bbox(right, comp["bbox"])
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        if h < 4 or w < 4:
            continue
        if int(comp["area"]) != 2 * (h + w) - 4:
            continue
        if not (
            np.all(patch[0, :] == DIVIDER)
            and np.all(patch[-1, :] == DIVIDER)
            and np.all(patch[:, 0] == DIVIDER)
            and np.all(patch[:, -1] == DIVIDER)
        ):
            continue
        if np.any(patch[1:-1, 1:-1] == DIVIDER):
            continue
        frames.append(tuple(map(int, comp["bbox"])))
    assert len(frames) == 1
    return frames[0]


def _sprite_info(right: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    frame = _frame_bbox(right)
    fr0, fc0, fr1, fc1 = frame
    inner = right[fr0 + 1 : fr1, fc0 + 1 : fc1]
    comps = find_components(inner, bg=BG, colors=[SPRITE_COLOR], connectivity=4, monochromatic=True)
    assert len(comps) == 1
    sprite_bbox = tuple(map(int, comps[0]["bbox"]))
    sprite_patch = crop_bbox(inner, sprite_bbox)
    coords = sorted((int(r), int(c)) for r, c in np.argwhere(sprite_patch == SPRITE_COLOR))
    assert len(coords) == int(comps[0]["area"])
    return sprite_patch, coords


def _render_output(sequence: list[int], sprite_patch: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    assert len(sequence) == len(coords)
    colored = np.zeros_like(sprite_patch, dtype=int)
    for color, (r, c) in zip(sequence, coords):
        colored[r, c] = int(color)
    return scale_sprite(colored, UPSCALE)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the left color bands into an ordered palette and use it to color a scaled-up control sprite.

    Input structure:
    - A black-background grid is split by a single full-height grey column.
    - The left side is fully covered by 3 to 5 horizontal non-grey color bands, with row-majority colors remaining unambiguous even near ragged boundaries.
    - The right side contains one grey rectangular frame whose interior holds one 4-connected maroon control sprite on black.

    Rule:
    - Find the grey divider and read the left side row by row, taking each row's majority non-grey color.
    - Compress consecutive equal row-majority colors into an ordered sequence.
    - In the right-side control sprite's bounding box, enumerate the maroon pixels in reading order and replace pixel i with sequence color i.
    - Scale that recolored bounding box by 3 in both axes, expanding every cell into a solid 3x3 block, and return only the scaled bbox.

    Key insight:
    - The left bands do not directly appear in the output; they are first compressed into an ordered palette that parameterizes the per-pixel coloring of the control sprite before upscaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    sep = _separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0

    sequence = _compressed_sequence(left)
    sprite_patch, coords = _sprite_info(right)
    return _render_output(sequence, sprite_patch, coords)


def _random_heights(total: int, count: int, minimum: int = 2) -> list[int]:
    assert total >= count * minimum
    extras = _RNG.multinomial(total - count * minimum, np.ones(count) / count)
    return [minimum + int(extra) for extra in extras]


def _make_left_bands(height: int, width: int, sequence: list[int]) -> np.ndarray:
    left = np.zeros((height, width), dtype=int)
    heights = _random_heights(height, len(sequence), minimum=2)
    boundaries = []
    cursor = 0
    for color, band_h in zip(sequence, heights):
        left[cursor : cursor + band_h, :] = int(color)
        boundaries.append(cursor + band_h - 1)
        cursor += band_h

    mixed_rows = 0
    for i in range(len(sequence) - 1):
        top_color = int(sequence[i])
        bottom_color = int(sequence[i + 1])
        top_row = boundaries[i]
        bottom_row = top_row + 1
        for row, base_color, other_color in ((top_row, top_color, bottom_color), (bottom_row, bottom_color, top_color)):
            max_intrusion = max(1, min(width // 3, width - 1))
            intrusion = int(_RNG.integers(1, max_intrusion + 1))
            if bool(_RNG.integers(0, 2)):
                left[row, :intrusion] = other_color
                left[row, intrusion:] = base_color
            else:
                left[row, -intrusion:] = other_color
                left[row, :-intrusion] = base_color
            mixed_rows += 1
    assert mixed_rows >= 2
    return left


def _paint_frame(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top, left : left + width] = DIVIDER
    grid[top + height - 1, left : left + width] = DIVIDER
    grid[top : top + height, left] = DIVIDER
    grid[top : top + height, left + width - 1] = DIVIDER


def _place_sprite_in_frame(
    grid: np.ndarray,
    frame_top: int,
    frame_left: int,
    frame_h: int,
    frame_w: int,
    sprite_mask: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    sh, sw = sprite_mask.shape
    assert inner_h >= sh + 2 and inner_w >= sw + 2
    offset_r = int(_RNG.integers(1, inner_h - sh))
    offset_c = int(_RNG.integers(1, inner_w - sw))
    patch = np.zeros((inner_h, inner_w), dtype=int)
    patch[offset_r : offset_r + sh, offset_c : offset_c + sw][sprite_mask] = SPRITE_COLOR
    grid[frame_top + 1 : frame_top + frame_h - 1, frame_left + 1 : frame_left + frame_w - 1] = patch
    sprite_patch = crop_bbox(patch, tuple(map(int, _ARC_UTILS.bbox(patch == SPRITE_COLOR))))
    coords = sorted((int(r), int(c)) for r, c in np.argwhere(sprite_patch == SPRITE_COLOR))
    return sprite_patch, coords


def _generate_case() -> tuple[np.ndarray, dict]:
    count = int(_RNG.integers(3, 6))
    sequence = [int(c) for c in _RNG.choice(PALETTE, size=count, replace=False)]
    sprite_mask = _choose_sprite_mask(count)
    sh, sw = sprite_mask.shape

    frame_h = sh + int(_RNG.integers(4, 6))
    frame_w = sw + int(_RNG.integers(4, 6))
    min_h = max(2 * count, frame_h + 2)
    height = int(_RNG.integers(min_h, min(19, min_h + 5) + 1))

    left_w = int(_RNG.integers(6, 11))
    right_w = frame_w + int(_RNG.integers(4, 9))
    width = left_w + 1 + right_w
    assert width <= 30

    grid = np.zeros((height, width), dtype=int)
    sep = left_w
    grid[:, sep] = DIVIDER
    grid[:, :left_w] = _make_left_bands(height, left_w, sequence)

    frame_top = int(_RNG.integers(1, height - frame_h))
    frame_left = sep + 1 + int(_RNG.integers(1, right_w - frame_w))
    _paint_frame(grid, frame_top, frame_left, frame_h, frame_w)
    sprite_patch, coords = _place_sprite_in_frame(grid, frame_top, frame_left, frame_h, frame_w, sprite_mask)

    return grid, {"sequence": sequence, "sprite_patch": sprite_patch, "coords": coords}


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    try:
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 2 and max(grid.shape) <= 30
        sep = _separator_col(grid)
        left = grid[:, :sep]
        right = grid[:, sep + 1 :]
        sequence = _compressed_sequence(left)
        assert sequence == list(meta["sequence"])

        mixed_rows = sum(len(set(map(int, row))) > 1 for row in left)
        assert mixed_rows >= 2

        sprite_patch, coords = _sprite_info(right)
        assert np.array_equal(sprite_patch, meta["sprite_patch"])
        assert coords == list(meta["coords"])
        assert len(sequence) == len(coords)

        expected = _render_output(sequence, sprite_patch, coords)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        assert actual.shape == tuple(int(x) * UPSCALE for x in sprite_patch.shape)
        assert len(set(sequence)) == len(sequence)
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid, meta = _generate_case()
        if not _validate_case(grid, meta):
            continue
        input_sig = _signature(grid)
        output_sig = _signature(main(grid))
        if input_sig in _SEEN_INPUTS or output_sig in _SEEN_OUTPUTS:
            continue
        _SEEN_INPUTS.add(input_sig)
        _SEEN_OUTPUTS.add(output_sig)
        return grid
    raise RuntimeError("failed to generate a valid unique input")
