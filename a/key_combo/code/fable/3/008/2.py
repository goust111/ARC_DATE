from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, bbox_h, bbox_w, crop_bbox, scale_sprite


BACKGROUND = 0
GEN_RETRY_LIMIT = 400


def _downscale_main_candidate(g: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int], int] | None:
    b = bbox(g == int(color))
    if b is None:
        return None

    h = bbox_h(b)
    w = bbox_w(b)
    if h != w or h % 3 != 0:
        return None

    scale = h // 3
    if scale < 2:
        return None

    patch = crop_bbox(g, b)
    if np.any((patch != BACKGROUND) & (patch != int(color))):
        return None

    small = np.zeros((3, 3), dtype=bool)
    for r in range(3):
        for c in range(3):
            block = patch[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            if np.all(block == int(color)):
                small[r, c] = True
            elif np.all(block == BACKGROUND):
                small[r, c] = False
            else:
                return None

    if not small.any() or small.all():
        return None

    if not np.array_equal(scale_sprite(small.astype(int), scale).astype(bool), patch == int(color)):
        return None

    return small, b, scale


def _find_main_mask(g: np.ndarray) -> tuple[int, np.ndarray, tuple[int, int, int, int], int]:
    candidates = []
    for color in sorted(set(map(int, g.flat)) - {BACKGROUND}):
        decoded = _downscale_main_candidate(g, color)
        if decoded is not None:
            pattern, b, scale = decoded
            candidates.append((bbox_h(b) * bbox_w(b), int(color), pattern, b, scale))

    assert candidates, "expected one large 3x3 scaled main mask"
    max_area = max(item[0] for item in candidates)
    largest = [item for item in candidates if item[0] == max_area]
    assert len(largest) == 1, "expected a unique largest scaled main mask"
    _area, color, pattern, b, scale = largest[0]
    return color, pattern, b, scale


def _parse_legend_frame(
    g: np.ndarray,
    r0: int,
    c0: int,
    dot_color: int,
    main_color: int,
) -> np.ndarray | None:
    height, width = g.shape
    if r0 < 0 or c0 < 0 or r0 + 3 > height or c0 + 3 > width:
        return None

    frame = g[r0 : r0 + 3, c0 : c0 + 3]
    if np.any(frame == int(main_color)):
        return None

    mask = frame != BACKGROUND
    if not mask.any() or mask.all():
        return None

    colors = set(map(int, frame[mask]))
    if len(colors) != 1:
        return None

    shape_color = colors.pop()
    if int(shape_color) == int(dot_color):
        return None

    # The local frame is part of the visible clue: the shape must span all three
    # rows and columns so that the dot's adjacent 3x3 reference frame is unique.
    if not (mask.any(axis=0).all() and mask.any(axis=1).all()):
        return None

    return mask.copy()


def _legend_entries(g: np.ndarray, main_color: int) -> list[tuple[np.ndarray, int]]:
    entries: dict[tuple[int, int, int, int], tuple[np.ndarray, int]] = {}
    dot_candidates = np.argwhere((g != BACKGROUND) & (g != int(main_color)))

    for dot_r, dot_c in dot_candidates:
        dot_r = int(dot_r)
        dot_c = int(dot_c)
        dot_color = int(g[dot_r, dot_c])

        for offset in range(3):
            placements = (
                (dot_r - offset, dot_c + 1),  # dot on the left of the frame
                (dot_r - offset, dot_c - 3),  # dot on the right of the frame
                (dot_r + 1, dot_c - offset),  # dot above the frame
                (dot_r - 3, dot_c - offset),  # dot below the frame
            )
            for r0, c0 in placements:
                mask = _parse_legend_frame(g, int(r0), int(c0), dot_color, main_color)
                if mask is None:
                    continue
                key = (int(r0), int(c0), dot_r, dot_c)
                entries[key] = (mask, dot_color)

    return list(entries.values())


def main(grid: np.ndarray) -> np.ndarray:
    """Use the legend entry showing the main mask's negative shape to color the complement inside the main bbox.

    Input structure:
    - The background is black and contains one large monochrome main mask, exactly an integer upscale of a 3x3 binary pattern.
    - Elsewhere are several legend entries; each entry is a small monochrome shape occupying a 3x3 local frame plus one tightly adjacent single-pixel color dot.
    - Exactly one legend shape equals the foreground/background complement of the main mask's 3x3 base pattern, and one distractor legend shape equals the base pattern itself.

    Rule:
    - Locate the unique largest 3k by 3k main mask and downscale its k by k blocks to the underlying 3x3 pattern.
    - Invert that 3x3 pattern inside its frame, find the legend entry with the same inverted shape, and read that entry's adjacent dot color.
    - Return the main mask bbox only: cells occupied by the original main mask become black, and all remaining bbox cells become the matched dot color.

    Key insight:
    - The visible legend key is the negative of the large mask, not the matching positive copy; the same inversion selects the color and defines the output fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.any(g == BACKGROUND), "expected a black background"

    main_color, pattern, main_box, scale = _find_main_mask(g)
    target = ~pattern
    entries = _legend_entries(g, main_color)
    assert len(entries) >= 2, "expected multiple legend entries"

    matches = [(mask, dot_color) for mask, dot_color in entries if np.array_equal(mask, target)]
    assert len(matches) == 1, "expected exactly one negative-shape legend entry"
    paired_color = int(matches[0][1])

    main_patch = crop_bbox(g, main_box)
    out = np.full(main_patch.shape, paired_color, dtype=int)
    out[main_patch == int(main_color)] = BACKGROUND
    assert out.shape == (3 * scale, 3 * scale)
    return out


def _full_bbox(mask: np.ndarray) -> bool:
    return (
        mask.any(axis=0).all()
        and mask.any(axis=1).all()
        and (~mask).any(axis=0).all()
        and (~mask).any(axis=1).all()
    )


def _pattern_pool() -> list[np.ndarray]:
    patterns = []
    for bits in range(1, 2**9 - 1):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        comp = ~mask
        if not (3 <= int(mask.sum()) <= 6):
            continue
        if not (_full_bbox(mask) and _full_bbox(comp)):
            continue
        patterns.append(mask)
    assert patterns
    return patterns


PATTERNS = _pattern_pool()


def _same_area_distractor(pattern: np.ndarray, forbidden: list[np.ndarray]) -> np.ndarray | None:
    random.shuffle(PATTERNS)
    for candidate in PATTERNS:
        if int(candidate.sum()) != int(pattern.sum()):
            continue
        if any(np.array_equal(candidate, other) for other in forbidden):
            continue
        return candidate.copy()
    return None


def _random_distractor(forbidden: list[np.ndarray]) -> np.ndarray:
    choices = [
        candidate
        for candidate in PATTERNS
        if not any(np.array_equal(candidate, other) for other in forbidden)
    ]
    assert choices
    return random.choice(choices).copy()


def _dot_side(mask: np.ndarray) -> str:
    sides = []
    if mask[:, 0].any():
        sides.append("left")
    if mask[:, 2].any():
        sides.append("right")
    if mask[0, :].any():
        sides.append("top")
    if mask[2, :].any():
        sides.append("bottom")
    return random.choice(sides)


def _render_entry(mask: np.ndarray, shape_color: int, dot_color: int, side: str) -> np.ndarray:
    if side == "left":
        patch = np.zeros((3, 4), dtype=int)
        patch[:, 1:4] = np.where(mask, int(shape_color), BACKGROUND)
        rows = [r for r in range(3) if mask[r, 0]]
        patch[random.choice(rows), 0] = int(dot_color)
    elif side == "right":
        patch = np.zeros((3, 4), dtype=int)
        patch[:, 0:3] = np.where(mask, int(shape_color), BACKGROUND)
        rows = [r for r in range(3) if mask[r, 2]]
        patch[random.choice(rows), 3] = int(dot_color)
    elif side == "top":
        patch = np.zeros((4, 3), dtype=int)
        patch[1:4, :] = np.where(mask, int(shape_color), BACKGROUND)
        cols = [c for c in range(3) if mask[0, c]]
        patch[0, random.choice(cols)] = int(dot_color)
    elif side == "bottom":
        patch = np.zeros((4, 3), dtype=int)
        patch[0:3, :] = np.where(mask, int(shape_color), BACKGROUND)
        cols = [c for c in range(3) if mask[2, c]]
        patch[3, random.choice(cols)] = int(dot_color)
    else:
        raise ValueError(f"unknown side: {side}")
    return patch


def _can_place(reserved: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    height, width = reserved.shape
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > height or c0 + w > width:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(height, r0 + h + padding)
    cc1 = min(width, c0 + w + padding)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, padding: int = 1) -> bool:
    height, width = grid.shape
    h, w = patch.shape
    positions = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if not _can_place(reserved, patch, r0, c0, padding=padding):
            continue
        mask = patch != BACKGROUND
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        reserved[r0 : r0 + h, c0 : c0 + w] = True
        return True
    return False


def _expected_output(pattern: np.ndarray, scale: int, paired_color: int) -> np.ndarray:
    large_pattern = scale_sprite(pattern.astype(int), scale).astype(bool)
    out = np.full(large_pattern.shape, int(paired_color), dtype=int)
    out[large_pattern] = BACKGROUND
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)

    pattern = random.choice(PATTERNS).copy()
    complement = ~pattern
    scale = random.randint(2, min(5, height // 3, width // 3))

    nonblack = list(range(1, 10))
    main_color = random.choice(nonblack)
    shape_color = random.choice([c for c in nonblack if c != main_color])
    dot_palette = [c for c in nonblack if c not in (main_color, shape_color)]
    random.shuffle(dot_palette)

    legend_count = random.randint(4, min(7, len(dot_palette)))
    correct_color = dot_palette[0]
    decoy_color = dot_palette[1]

    legend_masks = [complement.copy(), pattern.copy()]
    forbidden = [pattern, complement]
    area_decoy = _same_area_distractor(complement, forbidden)
    if area_decoy is not None and len(legend_masks) < legend_count:
        legend_masks.append(area_decoy)
        forbidden.append(area_decoy)

    while len(legend_masks) < legend_count:
        distractor = _random_distractor(forbidden)
        legend_masks.append(distractor)
        forbidden.append(distractor)

    dot_colors = dot_palette[:legend_count]
    entries = list(zip(legend_masks, dot_colors))
    random.shuffle(entries)

    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    main_patch = np.where(scale_sprite(pattern.astype(int), scale).astype(bool), int(main_color), BACKGROUND)
    if not _place_patch(grid, reserved, main_patch, padding=2):
        return None

    for mask, dot_color in entries:
        entry_patch = _render_entry(mask, shape_color, int(dot_color), _dot_side(mask))
        if not _place_patch(grid, reserved, entry_patch, padding=1):
            return None

    expected = _expected_output(pattern, scale, correct_color)
    naive_positive = _expected_output(pattern, scale, decoy_color)
    if np.array_equal(expected, naive_positive):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    parsed_main_color, parsed_pattern, _main_box, parsed_scale = _find_main_mask(grid)
    parsed_entries = _legend_entries(grid, parsed_main_color)
    negative_matches = [dot for mask, dot in parsed_entries if np.array_equal(mask, ~parsed_pattern)]
    positive_matches = [dot for mask, dot in parsed_entries if np.array_equal(mask, parsed_pattern)]
    same_area_wrong = [
        dot
        for mask, dot in parsed_entries
        if int(mask.sum()) == int((~parsed_pattern).sum()) and not np.array_equal(mask, ~parsed_pattern)
    ]
    if parsed_scale != scale:
        return None
    if negative_matches != [correct_color]:
        return None
    if positive_matches != [decoy_color]:
        return None
    if not same_area_wrong:
        return None
    if len(set(map(int, actual.flat))) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid negative-legend input after {GEN_RETRY_LIMIT} attempts")
