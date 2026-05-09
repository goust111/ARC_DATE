from __future__ import annotations

# Variant of 269e22fb
# Dimension: D12 — 空间关系：裁剪窗口在图案中的位置约束 (meta)
# Change: 裁剪窗口可以位于20x20图案的任意有效位置，偏移(r,c)均匀随机选取... → 裁剪窗口位置由特殊标记像素指示
#
# Design analysis
# a) Concrete behavior and trace:
#    - Input is no longer just the crop itself. It is a 20-row canvas with two separated zones:
#      a left 20x20 empty reference board and, after one blank gap column, the visible crop rectangle.
#    - The crop still comes from one dihedral variant of the same 20x20 binary motif and still uses two colors.
#    - A single isolated marker pixel is placed somewhere on the 20x20 board. Its row and column jointly encode
#      the crop's top-left offset inside the hidden 20x20 motif.
#    - Small trace: in the original task a 9x9 crop might match the motif at two offsets, so solver searched all
#      offsets and required only one global match. In this variant the marker preselects the intended offset, so
#      solver decodes (r,c) from the marker first and then tests only that offset across orientations/polarities.
# b) Complexity calibration:
#    - If we allowed mostly large crops, the marker would often be redundant because the crop already identifies
#      the offset uniquely. To make D12 visible, the generator prefers sizes that admit content-level ambiguity
#      (mainly 8..11), but still stays inside the original legal size range.
#    - The marker encodes exactly two integers, not a larger symbolic system, so the added freedom is controlled.
# c) Hidden assumptions in the original solver that break here:
#    - `H, W = g.shape` assumed the whole input was the crop. Not valid now because input includes extra canvas.
#    - `colors_present(g)` assumed exactly two colors total. Not valid now because the outer canvas adds a third
#      visible color (the board background).
#    - Brute-force search over every offset made the crop position purely implicit. Here offset must come from the
#      marker, so searching all offsets would ignore the new rule and allow degeneration back to the old task.
# d) Placement legality:
#    - The crop is pasted as one solid non-background rectangle on the right, with a one-column blank gap from the
#      marker board, so the crop component and the marker component never touch.
#    - The marker is always a singleton inside the left 20x20 board, therefore connected-component parsing is stable.
# e) Visual observability:
#    - Across examples, the crop can look locally similar while the marker moves to different board positions.
#      The left board acts as a coordinate key that humans can inspect directly.
#    - The row and column are encoded independently by evenly spaced ticks over 20 cells, so "farther down/right
#      marker -> farther down/right crop origin" is observable from a few examples.
# f) Visual neatness:
#    - There are only three colors in the input: board background, crop foreground, crop background.
#      The marker reuses the crop foreground color, staying visually salient without adding a fourth color.
#    - Elements are spatially separated: reference board on the left, one blank column, crop on the right.
# g) Color inferability:
#    - No color id carries semantic meaning. The marker color is just one of the crop colors; the solver uses
#      isolation and location, not a special palette value.
#    - Output recoloring is unchanged: repaint the full motif with the crop's two colors.
# h) New main() assumptions guaranteed by generate_input():
#    - Non-background cells form exactly two 4-connected components: one large crop rectangle and one singleton.
#    - The singleton lies on the 20x20 reference board and lands exactly on one of the valid row/column tick marks
#      for the crop size, so offset decoding is exact.
#    - Once the offset is fixed by the marker, exactly one orientation/polarity match remains.

import random
from collections import Counter

import numpy as np

from arc_utils import colors_present, get_bg


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)

_BOARD_SIZE = 20
_CROP_ROW_RANGE = (8, 16)
_CROP_COL_RANGE = (8, 12)
_AMBIGUOUS_ROW_MAX = 11
_AMBIGUOUS_COL_MAX = 11
_RIGHT_GAP = 1
_RIGHT_START = _BOARD_SIZE + _RIGHT_GAP
_PALETTE = tuple(range(1, 10))


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _offset_marks(max_offset: int) -> tuple[int, ...]:
    if max_offset == 0:
        return (0,)
    marks = tuple(int((idx * 19 + max_offset // 2) // max_offset) for idx in range(max_offset + 1))
    assert len(set(marks)) == max_offset + 1
    return marks


def _decode_offset(marker_coord: int, crop_span: int) -> int:
    marks = _offset_marks(_BOARD_SIZE - int(crop_span))
    assert int(marker_coord) in marks
    return int(marks.index(int(marker_coord)))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    H, W = map(int, mask.shape)
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_crop_and_marker(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    bg = int(get_bg(grid))
    comps = _components(grid != bg)
    assert len(comps) == 2

    comps.sort(key=lambda comp: int(comp.shape[0]))
    marker_coords = comps[0]
    crop_coords = comps[1]
    assert int(marker_coords.shape[0]) == 1

    marker = tuple(map(int, marker_coords[0]))
    mr, mc = marker
    assert 0 <= mr < _BOARD_SIZE and 0 <= mc < _BOARD_SIZE

    r0, c0, r1, c1 = _bbox(crop_coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert c0 >= _RIGHT_START and r1 < _BOARD_SIZE
    assert crop.shape[0] in range(_CROP_ROW_RANGE[0], _CROP_ROW_RANGE[1] + 1)
    assert crop.shape[1] in range(_CROP_COL_RANGE[0], _CROP_COL_RANGE[1] + 1)
    assert len(colors_present(crop)) == 2
    return crop, marker


def _canonical_window_key(window: np.ndarray) -> bytes:
    raw = window.astype(np.uint8, copy=False)
    normal = raw.tobytes()
    flipped = (1 - raw).tobytes()
    return normal if normal <= flipped else flipped


def _build_ambiguous_cases() -> dict[tuple[int, int], list[tuple[int, int, int, int]]]:
    variants = [pattern.astype(np.uint8) for pattern in _dihedral_variants(_pattern_mask())]
    cases: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}

    for H in range(_CROP_ROW_RANGE[0], _AMBIGUOUS_ROW_MAX + 1):
        for W in range(_CROP_COL_RANGE[0], _AMBIGUOUS_COL_MAX + 1):
            groups: dict[bytes, list[tuple[int, int, int]]] = {}
            for variant_idx, pattern in enumerate(variants):
                for r in range(_BOARD_SIZE - H + 1):
                    for c in range(_BOARD_SIZE - W + 1):
                        key = _canonical_window_key(pattern[r : r + H, c : c + W])
                        groups.setdefault(key, []).append((int(variant_idx), int(r), int(c)))

            valid: list[tuple[int, int, int, int]] = []
            for matches in groups.values():
                if len(matches) <= 1:
                    continue
                counts_by_offset = Counter((int(r), int(c)) for _variant_idx, r, c in matches)
                for variant_idx, r, c in matches:
                    if counts_by_offset[(int(r), int(c))] == 1:
                        valid.append((int(variant_idx), int(r), int(c), int(len(matches))))

            if valid:
                cases[(int(H), int(W))] = valid

    assert cases
    return cases


_VARIANTS = _dihedral_variants(_pattern_mask())
_AMBIGUOUS_CASES = _build_ambiguous_cases()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the full 20x20 motif by reading a crop plus a separate position marker.

    Input structure:
    - The grid contains a two-color rectangular crop of one dihedral variant of the shared 20x20 motif.
    - Separately, there is one isolated marker pixel on a blank 20x20 reference board; its row/column encode the crop offset.

    Rule:
    - Detect the crop rectangle and the singleton marker.
    - Decode the crop's top-left offset from the marker position, then test only that offset across motif orientations and both foreground/background polarities.
    - Repaint the uniquely matched full 20x20 motif with the crop's two colors.

    Key insight:
    - The crop no longer has to reveal its own location. The singleton marker supplies the missing spatial metadata, so the crop only has to identify orientation and polarity.
    """
    g = np.asarray(grid, dtype=int)
    crop, (marker_r, marker_c) = _extract_crop_and_marker(g)
    H, W = map(int, crop.shape)

    offset_r = _decode_offset(int(marker_r), H)
    offset_c = _decode_offset(int(marker_c), W)

    crop_colors = colors_present(crop)
    assert len(crop_colors) == 2

    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in crop_colors:
        bg = int(crop_colors[0] if int(crop_colors[1]) == int(fg) else crop_colors[1])
        visible = crop == int(fg)
        for pattern in _VARIANTS:
            if np.array_equal(pattern[offset_r : offset_r + H, offset_c : offset_c + W], visible):
                matches.append((int(fg), int(bg), pattern))

    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    size = random.choice(list(_AMBIGUOUS_CASES))
    H, W = map(int, size)
    variant_idx, offset_r, offset_c, _ambiguity = random.choice(_AMBIGUOUS_CASES[size])
    pattern = _VARIANTS[int(variant_idx)]

    fg, bg = random.sample(_PALETTE, 2)
    output = np.where(pattern, int(fg), int(bg)).astype(int)
    crop = output[offset_r : offset_r + H, offset_c : offset_c + W]

    marker_r = _offset_marks(_BOARD_SIZE - H)[int(offset_r)]
    marker_c = _offset_marks(_BOARD_SIZE - W)[int(offset_c)]

    grid = np.zeros((_BOARD_SIZE, _RIGHT_START + W), dtype=int)
    grid[0:H, _RIGHT_START : _RIGHT_START + W] = crop
    grid[int(marker_r), int(marker_c)] = int(fg)

    assert np.array_equal(main(grid), output)
    return grid
