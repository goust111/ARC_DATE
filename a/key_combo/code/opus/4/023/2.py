from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


_GEN_HEIGHTS = (12, 14, 16, 18, 20)
_GEN_WIDTH_RANGE = (11, 19)
_GEN_LENGTH_RANGE = (3, 5)
_GEN_RETRY_LIMIT = 400
_SEEN_INPUT_SIGS: set[bytes] = set()
_SEEN_OUTPUT_SIGS: set[bytes] = set()


def _input_signature(grid: np.ndarray) -> bytes:
    return grid.shape[0].to_bytes(1, "little") + grid.shape[1].to_bytes(1, "little") + grid.tobytes()


def _extract_segment(grid: np.ndarray) -> tuple[int, int, int, int]:
    blue_mask = grid == Color.BLUE
    comps = components_mask(blue_mask, connectivity=4)
    assert len(comps) == 1, "expected exactly one blue component"
    coords = comps[0]
    r0, c0, r1, c1 = bbox_coords(coords)
    assert c0 == c1, "the blue component must be vertical"
    expected = {(r, c0) for r in range(r0, r1 + 1)}
    actual = {(int(r), int(c)) for r, c in coords}
    assert actual == expected, "the blue component must be one solid vertical segment"
    return int(r0), int(c0), int(r1), int(coords.shape[0])


def _original_pyramid(shape: tuple[int, int], base_row: int, center_col: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, shape)
    depth = min(center_col, width - 1 - center_col)
    assert base_row + depth < height, "pyramid would exceed the bottom edge"

    mask = np.zeros((height, width), dtype=bool)
    colors = np.zeros((height, width), dtype=int)
    for v in range(depth + 1):
        row = base_row + v
        c0 = center_col - v
        c1 = center_col + v
        for col in range(c0, c1 + 1):
            mask[row, col] = True
            hdist = abs(col - center_col)
            colors[row, col] = Color.BLUE if (hdist + v) % 2 == 0 else Color.GREEN
    return mask, colors


def _solve_from_segment(shape: tuple[int, int], base_row: int, center_col: int) -> np.ndarray:
    orig_mask, orig_colors = _original_pyramid(shape, base_row, center_col)
    mirror_mask = orig_mask[::-1, :]

    out = np.full(shape, Color.BLACK, dtype=int)
    orig_only = orig_mask & ~mirror_mask
    mirror_only = mirror_mask & ~orig_mask
    overlap = orig_mask & mirror_mask

    out[orig_only] = orig_colors[orig_only]
    out[mirror_only] = Color.GREY
    out[overlap] = Color.RED
    return out


def _case_stats(shape: tuple[int, int], base_row: int, center_col: int) -> dict[str, int]:
    orig_mask, _ = _original_pyramid(shape, base_row, center_col)
    mirror_mask = orig_mask[::-1, :]
    overlap = int(np.count_nonzero(orig_mask & mirror_mask))
    orig_only = int(np.count_nonzero(orig_mask & ~mirror_mask))
    mirror_only = int(np.count_nonzero(mirror_mask & ~orig_mask))
    depth = min(center_col, shape[1] - 1 - center_col)
    return {
        "depth": int(depth),
        "overlap": overlap,
        "orig_only": orig_only,
        "mirror_only": mirror_only,
    }


def _case_name(shape: tuple[int, int], base_row: int, center_col: int) -> str:
    overlap = _case_stats(shape, base_row, center_col)["overlap"]
    if overlap == 0:
        return "none"
    if overlap <= 10:
        return "small"
    return "large"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the lower endpoint of an upper-half blue segment into an alternating pyramid, mirror it across the horizontal midline, and recolor overlap classes.

    Input structure:
    - Black cells form the background.
    - Exactly one blue vertical segment of length 3 to 5 appears wholly in the upper half of the grid.
    - The segment occupies one column of consecutive rows and does not cross the horizontal midline.

    Rule:
    - Use the segment's bottom row as the pyramid base row and its column as the center column.
    - Build the original pyramid downward: width 1 on the base row, then increase the width by 2 each row while the full row still fits inside the grid; color each pyramid cell blue when horizontal distance plus vertical depth is even, otherwise green.
    - Reflect that whole pyramid across the grid's horizontal midline to make a second copy.
    - Keep original-only cells in their alternating blue/green colors, recolor mirror-only cells grey, recolor overlap cells red, and leave all other cells black.

    Key insight:
    - The input segment only encodes the base point and symmetry axis for one pyramid; the final signal comes from comparing that pyramid with its horizontal reflection and classifying cells by set overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0, "the horizontal midline is defined between two rows"

    colors = set(map(int, np.unique(g)))
    assert colors <= {Color.BLACK, Color.BLUE}, "expected only black background and one blue segment"

    top_row, center_col, base_row, seg_len = _extract_segment(g)
    assert 3 <= seg_len <= 5, "segment length must be between 3 and 5"
    assert base_row < height // 2, "segment must stay in the upper half"
    assert top_row >= 0

    return _solve_from_segment(g.shape, base_row, center_col)


def _build_input(height: int, width: int, top_row: int, length: int, center_col: int) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[top_row : top_row + length, center_col] = Color.BLUE
    return grid


def _validate_instance(inp: np.ndarray) -> tuple[np.ndarray, str]:
    assert isinstance(inp, np.ndarray)
    assert inp.ndim == 2 and max(inp.shape) <= 30
    top_row, center_col, base_row, seg_len = _extract_segment(inp)
    assert 3 <= seg_len <= 5
    assert top_row >= 0
    assert base_row < inp.shape[0] // 2

    stats = _case_stats(inp.shape, base_row, center_col)
    assert 2 <= stats["depth"] <= 6, "pyramid depth should stay visually clear"
    assert stats["orig_only"] > 0 and stats["mirror_only"] > 0, "both non-overlap classes must exist"

    out = main(inp)
    expected = _solve_from_segment(inp.shape, base_row, center_col)
    assert np.array_equal(out, expected), "main() must match the geometric construction"

    fg = out != Color.BLACK
    if stats["overlap"] == 0:
        assert Color.RED not in set(map(int, np.unique(out))), "no-overlap cases must have no red"
    else:
        assert int(np.count_nonzero(out == Color.RED)) == stats["overlap"], "red must match the exact overlap"
    assert Color.GREY in set(map(int, np.unique(out))), "mirror-only area must be visible"
    assert Color.BLUE in set(map(int, np.unique(out))), "alternating original colors must remain visible"
    assert Color.GREEN in set(map(int, np.unique(out))), "alternating original colors must remain visible"
    assert np.count_nonzero(fg) < inp.size * 0.50, "foreground should stay visually sparse"

    return out, _case_name(inp.shape, base_row, center_col)


def generate_input() -> np.ndarray:
    case_target = random.choices(
        population=["none", "small", "large"],
        weights=[0.35, 0.35, 0.30],
        k=1,
    )[0]

    for _ in range(_GEN_RETRY_LIMIT):
        height = random.choice(_GEN_HEIGHTS)
        width = random.randint(*_GEN_WIDTH_RANGE)
        length = random.randint(*_GEN_LENGTH_RANGE)
        center_col = random.randint(2, width - 3)
        depth = min(center_col, width - 1 - center_col)
        if depth > 6:
            continue

        base_min = length - 1
        base_max = min(height // 2 - 1, height - 1 - depth)
        if base_min > base_max:
            continue

        matching_bases = [
            base_row
            for base_row in range(base_min, base_max + 1)
            if _case_name((height, width), base_row, center_col) == case_target
        ]
        if not matching_bases:
            continue

        base_row = random.choice(matching_bases)
        top_row = base_row - length + 1
        inp = _build_input(height, width, top_row, length, center_col)

        try:
            out, observed_case = _validate_instance(inp)
        except AssertionError:
            continue
        assert observed_case == case_target

        in_sig = _input_signature(inp)
        out_sig = _input_signature(out)
        if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue

        _SEEN_INPUT_SIGS.add(in_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return inp

    raise RuntimeError("failed to generate a valid pyramid-reflection-overlap input")
