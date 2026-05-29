# Combo variant of 269e22fb (index 27)
# combo_spec: D9/3+D12/2
# Name: 双重位置编码验证

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import colors_present, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# a) Concrete behavior and trace:
#    - Input is no longer the bare crop. It is a 20-row canvas with a blank
#      20x20 reference board on the left, one empty gap column, and the visible
#      crop rectangle on the right.
#    - A singleton marker pixel sits somewhere on the left board. Its row and
#      column jointly encode the crop's top-left offset inside the hidden
#      20x20 motif using evenly spaced tick positions.
#    - Output is no longer 20x20. It is a 22x22 canvas whose center is the full
#      recovered motif and whose border repeats the same crop location:
#      top/bottom mark the crop's column span, left/right mark the row span.
#    - Example trace: if the crop is 9x10 from offset (4, 6), the input marker
#      lands on the tick for row offset 4 and column offset 6; the solver decodes
#      those coordinates, matches only windows starting at (4, 6), reconstructs
#      the full motif, then paints border marks on rows 5..13 at the left/right
#      edges and on cols 7..16 at the top/bottom edges.
#    - Relative to the original task, offset inference is no longer implicit:
#      the input specifies one position code directly, and the output exposes the
#      same position again as a span.
# b) Complexity calibration:
#    - To keep D12 meaningful, the generator samples only crop sizes whose raw
#      bitmap content admits multiple global matches when offset is ignored.
#      This prevents the marker from becoming decorative.
#    - To keep D9 meaningful, the output border always varies with both the crop
#      size and its decoded offset, so the 22x22 frame is not a fixed garnish.
#    - The construction stays visually small: three visible colors in input
#      (board background plus crop colors) and three in output (motif fg/bg plus
#      one border marker color).
# c) Hidden-assumption review against the original solver:
#    - `H, W = g.shape` no longer means "crop size"; the whole input includes a
#      board, a gap, and a crop region.
#    - `len(colors_present(g)) == 2` no longer holds on the full input because
#      the board background adds a third color.
#    - Brute-forcing every offset would ignore the new rule. The marker now
#      determines `(crop_r, crop_c)`, so the solver should search orientations
#      and polarity only at that decoded offset.
#    - Returning `np.where(pattern, fg, bg)` is no longer sufficient because the
#      output must also expose the crop span on a 22x22 border.
#    - New combo-specific assumption: the crop alone may be ambiguous, but once
#      the decoded offset is fixed there must be exactly one orientation/polarity
#      match. The generator enforces this explicitly.
# d) Placement legality:
#    - The marker is always inside the left 20x20 board and the crop always
#      starts after a one-column gap, so the singleton and the crop cannot touch.
#    - Crop colors are sampled from 1..9 while board/border marker color is 0,
#      making the crop a single solid non-background rectangle for parsing.
#    - Border spans are written to indices `1 + offset ... 1 + offset + size - 1`
#      inside a 22x22 frame, which is always in-bounds because
#      `0 <= offset <= 20 - size`.
# e) Visual observability:
#    - Across examples, moving the input marker changes both where the crop is
#      interpreted to come from and where the border spans appear in the output.
#    - Because the crop content is intentionally sampled from ambiguous cases,
#      examples can show that crop appearance alone is insufficient; the marker
#      is visibly doing real work.
# f) Visual cleanliness:
#    - Input semantics are separated into regions: left board for the code, gap,
#      right rectangle for the crop. There is no overlap.
#    - Output semantics are also separated: motif in the interior, position
#      annotation on the frame only, with corners left as plain background.
# g) Color inferability:
#    - The input marker reuses the crop foreground color, so its meaning comes
#      from isolation and position, not from a reserved color id.
#    - The output uses a fixed border marker color that never appears inside the
#      crop, making the frame annotation easy to spot without requiring any
#      hidden numeric interpretation of colors.
# h) New main() assumptions guaranteed by generate_input():
#    - Non-background cells split into exactly two 4-connected components:
#      one singleton marker and one filled crop rectangle.
#    - The marker coordinate is always one of the valid ticks for the sampled
#      crop height/width, so offset decoding is exact.
#    - The crop has more than one global `(polarity, orientation, offset)` match
#      when offset is ignored, but exactly one match at the marker-decoded
#      offset, ensuring both position encodings are genuinely active.


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
_INPUT_GAP = 1
_RIGHT_START = _BOARD_SIZE + _INPUT_GAP
_OUTPUT_SIZE = 22
_INNER_SLICE = slice(1, 21)
_BORDER_MARKER_COLOR = 0
_PALETTE = tuple(range(1, 10))
_CROP_ROW_RANGE = (8, 16)
_CROP_COL_RANGE = (8, 12)
_AMBIGUOUS_ROW_MAX = 11
_AMBIGUOUS_COL_MAX = 11


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
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = _components(g != bg)
    assert len(comps) == 2

    comps.sort(key=lambda comp: int(comp.shape[0]))
    marker_coords = comps[0]
    crop_coords = comps[1]
    assert int(marker_coords.shape[0]) == 1

    marker_r, marker_c = map(int, marker_coords[0])
    assert 0 <= marker_r < _BOARD_SIZE and 0 <= marker_c < _BOARD_SIZE

    r0, c0, r1, c1 = _bbox(crop_coords)
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    H, W = map(int, crop.shape)
    assert c0 >= _RIGHT_START and r1 < _BOARD_SIZE
    assert _CROP_ROW_RANGE[0] <= H <= _CROP_ROW_RANGE[1]
    assert _CROP_COL_RANGE[0] <= W <= _CROP_COL_RANGE[1]
    assert len(colors_present(crop)) == 2
    return crop, (marker_r, marker_c)


def _find_matches_at_offset(crop: np.ndarray, crop_r: int, crop_c: int) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(crop, dtype=int)
    H, W = map(int, g.shape)
    crop_colors = colors_present(g)
    assert len(crop_colors) == 2

    matches: list[tuple[int, int, np.ndarray]] = []
    for fg in crop_colors:
        bg = int(crop_colors[0] if int(crop_colors[1]) == int(fg) else crop_colors[1])
        visible = g == int(fg)
        for pattern in _VARIANTS:
            if np.array_equal(pattern[crop_r : crop_r + H, crop_c : crop_c + W], visible):
                matches.append((int(fg), int(bg), pattern))
    return matches


def _find_global_matches(crop: np.ndarray) -> list[tuple[int, int, np.ndarray, int, int]]:
    g = np.asarray(crop, dtype=int)
    H, W = map(int, g.shape)
    crop_colors = colors_present(g)
    assert len(crop_colors) == 2

    matches: list[tuple[int, int, np.ndarray, int, int]] = []
    for fg in crop_colors:
        bg = int(crop_colors[0] if int(crop_colors[1]) == int(fg) else crop_colors[1])
        visible = g == int(fg)
        for pattern in _VARIANTS:
            for crop_r in range(_BOARD_SIZE - H + 1):
                for crop_c in range(_BOARD_SIZE - W + 1):
                    if np.array_equal(pattern[crop_r : crop_r + H, crop_c : crop_c + W], visible):
                        matches.append((int(fg), int(bg), pattern, int(crop_r), int(crop_c)))
    return matches


def _render_output(
    pattern: np.ndarray,
    fg: int,
    bg: int,
    crop_r: int,
    crop_c: int,
    crop_h: int,
    crop_w: int,
) -> np.ndarray:
    out = np.full((_OUTPUT_SIZE, _OUTPUT_SIZE), int(bg), dtype=int)
    out[_INNER_SLICE, _INNER_SLICE] = np.where(pattern, int(fg), int(bg))
    out[0, 1 + crop_c : 1 + crop_c + crop_w] = int(_BORDER_MARKER_COLOR)
    out[21, 1 + crop_c : 1 + crop_c + crop_w] = int(_BORDER_MARKER_COLOR)
    out[1 + crop_r : 1 + crop_r + crop_h, 0] = int(_BORDER_MARKER_COLOR)
    out[1 + crop_r : 1 + crop_r + crop_h, 21] = int(_BORDER_MARKER_COLOR)
    return out


def _canonical_window_key(window: np.ndarray) -> bytes:
    raw = window.astype(np.uint8, copy=False)
    normal = raw.tobytes()
    flipped = (1 - raw).tobytes()
    return normal if normal <= flipped else flipped


def _build_ambiguous_cases() -> dict[tuple[int, int], list[tuple[int, int, int, int]]]:
    variants = [pattern.astype(np.uint8) for pattern in _VARIANTS]
    cases: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}

    for H in range(_CROP_ROW_RANGE[0], _AMBIGUOUS_ROW_MAX + 1):
        for W in range(_CROP_COL_RANGE[0], _AMBIGUOUS_COL_MAX + 1):
            groups: dict[bytes, list[tuple[int, int, int]]] = {}
            for variant_idx, pattern in enumerate(variants):
                for crop_r in range(_BOARD_SIZE - H + 1):
                    for crop_c in range(_BOARD_SIZE - W + 1):
                        key = _canonical_window_key(pattern[crop_r : crop_r + H, crop_c : crop_c + W])
                        groups.setdefault(key, []).append((int(variant_idx), int(crop_r), int(crop_c)))

            valid: list[tuple[int, int, int, int]] = []
            for matches in groups.values():
                if len(matches) <= 1:
                    continue
                counts_by_offset = Counter((int(crop_r), int(crop_c)) for _variant_idx, crop_r, crop_c in matches)
                for variant_idx, crop_r, crop_c in matches:
                    if counts_by_offset[(int(crop_r), int(crop_c))] == 1:
                        valid.append((int(variant_idx), int(crop_r), int(crop_c), int(len(matches))))

            if valid:
                cases[(int(H), int(W))] = valid

    assert cases
    return cases


_VARIANTS = _dihedral_variants(_pattern_mask())
_AMBIGUOUS_CASES = _build_ambiguous_cases()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the full motif from a marked crop and echo the decoded crop span on a border.

    Input structure:
    - The grid contains a two-color crop of one dihedral orientation of the shared 20x20 motif, pasted to the right of a blank 20x20 reference board.
    - One isolated marker pixel sits on that left board; its row and column encode the crop's top-left offset inside the hidden motif.

    Rule:
    - Separate the singleton marker from the crop rectangle, decode the crop offset from the marker coordinates, and then test only that offset across motif orientations and both foreground/background polarities.
    - Use the unique match to repaint the full 20x20 motif in the center of a 22x22 canvas.
    - Mark the same recovered crop location on the output border: top/bottom for the column span, left/right for the row span.

    Key insight:
    - Position is now stated twice in two different visual languages: a point code in the input and a span code in the output. The crop content identifies orientation and polarity, while the marker disambiguates where that crop came from.
    """
    g = np.asarray(grid, dtype=int)
    crop, (marker_r, marker_c) = _extract_crop_and_marker(g)
    H, W = map(int, crop.shape)

    crop_r = _decode_offset(int(marker_r), H)
    crop_c = _decode_offset(int(marker_c), W)

    matches = _find_matches_at_offset(crop, crop_r, crop_c)
    assert len(matches) == 1
    fg, bg, pattern = matches[0]
    return _render_output(pattern, fg, bg, crop_r, crop_c, H, W)


def generate_input() -> np.ndarray:
    size = random.choice(list(_AMBIGUOUS_CASES))
    H, W = map(int, size)
    variant_idx, crop_r, crop_c, _ambiguity = random.choice(_AMBIGUOUS_CASES[size])
    pattern = _VARIANTS[int(variant_idx)]

    fg, bg = random.sample(_PALETTE, 2)
    full = np.where(pattern, int(fg), int(bg)).astype(int)
    crop = full[crop_r : crop_r + H, crop_c : crop_c + W]

    marker_r = _offset_marks(_BOARD_SIZE - H)[int(crop_r)]
    marker_c = _offset_marks(_BOARD_SIZE - W)[int(crop_c)]

    grid = np.zeros((_BOARD_SIZE, _RIGHT_START + W), dtype=int)
    grid[0:H, _RIGHT_START : _RIGHT_START + W] = crop
    grid[int(marker_r), int(marker_c)] = int(fg)

    crop_only_matches = _find_global_matches(crop)
    assert len(crop_only_matches) > 1
    assert len(_find_matches_at_offset(crop, crop_r, crop_c)) == 1

    expected = _render_output(pattern, fg, bg, crop_r, crop_c, H, W)
    assert np.array_equal(main(grid), expected)
    return grid
