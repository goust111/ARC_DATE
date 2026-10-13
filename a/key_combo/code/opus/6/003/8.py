from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, coords_where, get_bg


PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
HEIGHT_OPTIONS = (2, 3, 4, 5)
GRID_HEIGHT_RANGE = (10, 15)
GRID_WIDTH_RANGE = (11, 16)
ACTIVE_COUNT = 4
MAX_GENERATION_ATTEMPTS = 600

_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _extract_segments(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    segments: list[dict[str, int]] = []
    for col in range(g.shape[1]):
        mask = g[:, col] != bg
        if not np.any(mask):
            continue
        rows = np.flatnonzero(mask)
        colors = np.unique(g[rows, col])
        assert colors.size == 1
        r0 = int(rows[0])
        r1 = int(rows[-1])
        assert rows.size == r1 - r0 + 1

        local_coords = coords_where(mask[:, None])
        br0, bc0, br1, bc1 = bbox_coords(local_coords)
        assert (br0, bc0, br1, bc1) == (r0, 0, r1, 0)

        segments.append(
            {
                "col": int(col),
                "color": int(colors[0]),
                "start": r0,
                "end": r1,
                "length": int(rows.size),
            }
        )

    segments.sort(key=lambda seg: seg["col"])
    return bg, segments


def _sorted_segments_for_output(height: int, segments: list[dict[str, int]]) -> list[dict[str, int]]:
    return sorted(segments, key=lambda seg: (height - seg["length"], seg["col"]))


def _render_output(height: int, width: int, bg: int, segments: list[dict[str, int]]) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    for new_col, seg in enumerate(_sorted_segments_for_output(height, segments)):
        length = seg["length"]
        out[height - length : height, new_col] = seg["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each active vertical segment to the bottom, then sort columns by the induced gap width.

    Input structure:
    - A black background contains several active columns and many empty columns.
    - Each active column has exactly one contiguous monochrome vertical segment, with background above and below it.
    - Active segment colors are distinct and the segment lengths are intended to be distinct.

    Rule:
    - For each active column, slide its segment downward until its bottom pixel reaches the last row.
    - After sliding, compute the number of background cells above that segment; this gap equals grid height minus segment length.
    - Sort active columns by that gap in ascending order and repack them tightly from column 0, leaving every remaining column as background.

    Key insight:
    - The sort key is not a visible input label; it is created by the bottom-alignment slide, so the longest segment ends up leftmost because it leaves the smallest top gap.
    """
    g = np.asarray(grid, dtype=int)
    bg, segments = _extract_segments(g)
    return _render_output(g.shape[0], g.shape[1], bg, segments)


def _sample_columns(width: int) -> list[int] | None:
    choices = list(range(width))
    for _ in range(100):
        cols = sorted(random.sample(choices, ACTIVE_COUNT))
        if cols == list(range(ACTIVE_COUNT)):
            continue
        if not any(b - a > 1 for a, b in zip(cols, cols[1:])):
            continue
        return cols
    return None


def _sample_lengths() -> list[int]:
    while True:
        lengths = list(HEIGHT_OPTIONS)
        random.shuffle(lengths)
        if lengths != sorted(lengths, reverse=True):
            return lengths


def _sample_starts(height: int, lengths: list[int]) -> list[int] | None:
    for _ in range(200):
        starts: list[int] = []
        used: set[int] = set()
        for length in lengths:
            limit = height - length - 1
            choices = [row for row in range(limit + 1) if row not in used]
            if not choices:
                break
            start = random.choice(choices)
            starts.append(start)
            used.add(start)
        if len(starts) != len(lengths):
            continue
        return starts
    return None


def _orders_are_nontrivial(lengths: list[int], starts: list[int]) -> bool:
    want = tuple(sorted(range(ACTIVE_COUNT), key=lambda i: -lengths[i]))
    by_start = tuple(sorted(range(ACTIVE_COUNT), key=lambda i: starts[i]))
    by_bottom = tuple(sorted(range(ACTIVE_COUNT), key=lambda i: starts[i] + lengths[i] - 1))
    return want != tuple(range(ACTIVE_COUNT)) and by_start != want and by_bottom != want


def _render_input(
    height: int,
    width: int,
    cols: list[int],
    lengths: list[int],
    starts: list[int],
    colors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    for col, length, start, color in zip(cols, lengths, starts, colors):
        grid[start : start + length, col] = color
    return grid


def _validate_scene(inp: np.ndarray, expected: np.ndarray) -> None:
    bg, segments = _extract_segments(inp)
    height, width = inp.shape

    assert bg == Color.BLACK
    assert len(segments) == ACTIVE_COUNT
    assert width <= 30 and height <= 30
    assert len({seg["color"] for seg in segments}) == ACTIVE_COUNT
    assert {seg["color"] for seg in segments}.issubset(set(PALETTE))
    assert sorted(seg["length"] for seg in segments) == list(HEIGHT_OPTIONS)
    assert len({seg["start"] for seg in segments}) == ACTIVE_COUNT
    assert all(seg["end"] < height - 1 for seg in segments)
    assert any(segments[i + 1]["col"] - segments[i]["col"] > 1 for i in range(ACTIVE_COUNT - 1))
    assert [seg["col"] for seg in segments] != list(range(ACTIVE_COUNT))

    order_by_length = [seg["col"] for seg in _sorted_segments_for_output(height, segments)]
    order_by_start = [seg["col"] for seg in sorted(segments, key=lambda seg: seg["start"])]
    order_by_bottom = [seg["col"] for seg in sorted(segments, key=lambda seg: seg["end"])]
    assert order_by_length != [seg["col"] for seg in segments]
    assert order_by_start != order_by_length
    assert order_by_bottom != order_by_length

    out_bg, out_segments = _extract_segments(expected)
    assert out_bg == bg
    assert len(out_segments) == ACTIVE_COUNT
    assert [seg["col"] for seg in out_segments] == list(range(ACTIVE_COUNT))
    assert all(seg["end"] == height - 1 for seg in out_segments)
    assert np.all(expected[:, ACTIVE_COUNT:] == bg)
    assert [seg["length"] for seg in out_segments] == sorted(HEIGHT_OPTIONS, reverse=True)

    expected_colors = [seg["color"] for seg in _sorted_segments_for_output(height, segments)]
    assert [seg["color"] for seg in out_segments] == expected_colors


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    cols = _sample_columns(width)
    if cols is None:
        return None

    lengths = _sample_lengths()
    starts = _sample_starts(height, lengths)
    if starts is None or not _orders_are_nontrivial(lengths, starts):
        return None

    colors = random.sample(list(PALETTE), ACTIVE_COUNT)
    inp = _render_input(height, width, cols, lengths, starts, colors)

    segments = [
        {
            "col": col,
            "color": color,
            "start": start,
            "end": start + length - 1,
            "length": length,
        }
        for col, length, start, color in zip(cols, lengths, starts, colors)
    ]
    expected = _render_output(height, width, Color.BLACK, segments)

    _validate_scene(inp, expected)
    assert np.array_equal(main(inp), expected)

    input_sig = _signature(inp)
    output_sig = _signature(expected)
    if input_sig in _SEEN_INPUTS or output_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_INPUTS.add(input_sig)
    _SEEN_OUTPUTS.add(output_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate input for puzzle 8")
