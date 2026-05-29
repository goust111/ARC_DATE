# Combo variant of 269e22fb (index 10)
# combo_spec: D12/2+D14/1
# Name: 标记像素指示位置加色块编码变体

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import colors_present, get_bg


# Design analysis
# ---------------
# a) Concrete layout and behavior trace:
#    - The original task exposed only a two-color crop, so the solver had to search all 8 dihedral variants,
#      every legal offset, and both color polarities.
#    - The fused task uses a 20-row canvas with two separated zones. On the left is a blank 20x20 reference
#      board. Its top-left 3x3 corner contains N metadata pixels in row-major order; N-1 is the dihedral
#      variant index. Elsewhere on the same board sits one isolated metadata pixel whose row/column each land
#      on a discrete tick and encode the crop's top-left offset. After one blank gap column, the right side
#      contains the original two-color crop rectangle.
#    - Trace: instead of "crop -> search variant -> search offset", the fused solver does
#      "count corner block -> decode variant, read singleton -> decode offset, then validate that one window
#      under the two possible crop polarities". The difference is that orientation and position are now carried
#      by two visibly distinct side-channel objects.
# b) Complexity calibration:
#    - The corner code uses only 8 counts, exactly matching the 8 dihedral states, so the symbolic layer stays
#      small and repeatedly observable across examples.
#    - The singleton still encodes only two integers, one for row offset and one for column offset, via evenly
#      spaced board ticks; this is enough spatial metadata without introducing a larger alphabet.
#    - Exhaustive enumeration shows a stronger motif-specific fact: every non-unique legal crop is a pairwise
#      collision between exactly one variant and one offset on each side, so there are no samples where either
#      metadata clue remains ambiguous after the crop is fixed. The generator therefore enforces global crop
#      ambiguity and balances all 8 variant codes, which keeps both auxiliary channels visible without inventing
#      off-spec coarse encodings.
# c) Hidden assumptions from the original solver that break here:
#    - `H, W = g.shape` no longer means crop size; the full input now includes a left metadata board, a blank
#      gap, and blank padding below the crop.
#    - `colors_present(g)` is no longer 2; the fused input uses four colors total: board background, crop fg,
#      crop bg, and one shared metadata color.
#    - Brute-force search over all variants and offsets is no longer the intended reasoning path; doing so would
#      ignore both the corner count and the singleton marker.
#    - The original solver never had to parse connected components. Here it must separate the crop rectangle
#      from the two metadata objects before any motif reasoning.
# d) Placement legality:
#    - The crop stays within the original 8..16 by 8..12 range and is pasted on the right after a one-column
#      blank gap, so it never touches the left-board metadata.
#    - The corner code is confined to the left board's 3x3 corner. The marker is filtered so it never lands in
#      the reserved 4x4 corner buffer, which prevents it from merging with the corner component.
#    - The left board is 20x20, matching the hidden motif size, so the marker's row and column each have enough
#      room to host all legal offset ticks exactly.
# e) Visual observability:
#    - Humans can see three clearly separated clues: a counted corner blob, a singleton coordinate marker, and
#      the crop itself. The corner count varies with the completed orientation; the singleton moves with the
#      crop location; the crop colors determine output recoloring.
#    - No invisible ordering is used. Both metadata cues are encoded by count or position, not by raw color id.
# f) Visual neatness:
#    - The input remains sparse and partitioned: metadata on the left board, crop on the right.
#    - Both metadata objects reuse one dedicated metadata color, making them feel like one instruction layer
#      without contaminating the crop's two-color appearance.
# g) Color inferability:
#    - The crop still uses exactly two colors and output ignores the metadata color entirely.
#    - The metadata color has a structural role only: counted pixels in the corner and one isolated board pixel.
# h) New main() assumptions guaranteed by generate_input():
#    - Non-background cells form exactly three 4-connected components: the crop rectangle, the corner code
#      component, and the singleton marker.
#    - The corner component lies fully inside the top-left 3x3 box and has size 1..8.
#    - The singleton lies on valid offset ticks and outside the reserved 4x4 corner buffer.
#    - The crop is globally ambiguous under the original search, but once the encoded variant and offset are
#      decoded, exactly one polarity match remains for that single window.


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
_CODE_BOX = 3
_CORNER_BUFFER = 4
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


def _other_color(colors: list[int], chosen: int) -> int:
    return int(colors[0] if int(colors[1]) == int(chosen) else colors[1])


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


def _canonical_window_key(window: np.ndarray) -> bytes:
    raw = window.astype(np.uint8, copy=False)
    normal = raw.tobytes()
    flipped = (1 - raw).tobytes()
    return normal if normal <= flipped else flipped


def _find_matches(crop: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    crop = np.asarray(crop, dtype=int)
    H, W = map(int, crop.shape)
    crop_colors = colors_present(crop)
    assert len(crop_colors) == 2

    matches: list[tuple[int, int, int, int, int]] = []
    for variant_idx, pattern in enumerate(_VARIANTS):
        for fg in crop_colors:
            bg = _other_color(crop_colors, int(fg))
            visible = crop == int(fg)
            for r in range(_BOARD_SIZE - H + 1):
                for c in range(_BOARD_SIZE - W + 1):
                    if np.array_equal(pattern[r : r + H, c : c + W], visible):
                        matches.append((int(variant_idx), int(fg), int(bg), int(r), int(c)))
    return matches


def _encode_corner_block(grid: np.ndarray, meta_color: int, variant_idx: int) -> None:
    count = int(variant_idx) + 1
    coords = [(r, c) for r in range(_CODE_BOX) for c in range(_CODE_BOX)]
    for r, c in coords[:count]:
        grid[int(r), int(c)] = int(meta_color)


def _extract_crop_marker_and_variant(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    bg = int(get_bg(grid))
    comps = _components(grid != bg)
    assert len(comps) == 3

    corner_count = int(np.sum(grid[:_CODE_BOX, :_CODE_BOX] != bg))
    assert 1 <= corner_count <= 8
    variant_idx = corner_count - 1

    comps.sort(key=lambda comp: int(comp.shape[0]), reverse=True)
    crop_coords = comps[0]
    r0, c0, r1, c1 = _bbox(crop_coords)
    assert r0 == 0
    assert c0 >= _RIGHT_START
    assert r1 < _BOARD_SIZE

    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    H, W = map(int, crop.shape)
    assert _CROP_ROW_RANGE[0] <= H <= _CROP_ROW_RANGE[1]
    assert _CROP_COL_RANGE[0] <= W <= _CROP_COL_RANGE[1]
    assert len(colors_present(crop)) == 2

    corner_comps = []
    marker_comps = []
    for comp in comps[1:]:
        rr0, cc0, rr1, cc1 = _bbox(comp)
        if rr1 < _CODE_BOX and cc1 < _CODE_BOX:
            corner_comps.append(comp)
        else:
            marker_comps.append(comp)

    assert len(corner_comps) == 1
    assert int(corner_comps[0].shape[0]) == corner_count
    assert len(marker_comps) == 1
    assert int(marker_comps[0].shape[0]) == 1

    marker_r, marker_c = map(int, marker_comps[0][0])
    assert 0 <= marker_r < _BOARD_SIZE and 0 <= marker_c < _BOARD_SIZE
    assert not (marker_r < _CORNER_BUFFER and marker_c < _CORNER_BUFFER)
    return crop, (marker_r, marker_c), int(variant_idx)


def _build_fusion_cases() -> dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]]:
    variants = [pattern.astype(np.uint8) for pattern in _VARIANTS]
    cases: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]] = {}
    collision_signatures: set[tuple[int, int]] = set()

    for H in range(_CROP_ROW_RANGE[0], _AMBIGUOUS_ROW_MAX + 1):
        for W in range(_CROP_COL_RANGE[0], _AMBIGUOUS_COL_MAX + 1):
            groups: dict[bytes, list[tuple[int, int, int]]] = {}
            for variant_idx, pattern in enumerate(variants):
                for r in range(_BOARD_SIZE - H + 1):
                    for c in range(_BOARD_SIZE - W + 1):
                        window = pattern[r : r + H, c : c + W]
                        if window.all() or (not window.any()):
                            continue
                        groups.setdefault(_canonical_window_key(window), []).append((int(variant_idx), int(r), int(c)))

            valid: list[tuple[int, int, int, int, int, int]] = []
            for matches in groups.values():
                if len(matches) <= 1:
                    continue
                counts_by_offset = Counter((int(r), int(c)) for _variant_idx, r, c in matches)
                counts_by_variant = Counter(int(variant_idx) for variant_idx, _r, _c in matches)
                for variant_idx, offset_r, offset_c in matches:
                    marker_r = _offset_marks(_BOARD_SIZE - H)[int(offset_r)]
                    marker_c = _offset_marks(_BOARD_SIZE - W)[int(offset_c)]
                    if marker_r < _CORNER_BUFFER and marker_c < _CORNER_BUFFER:
                        continue
                    collision_signatures.add(
                        (
                            int(counts_by_offset[(int(offset_r), int(offset_c))]),
                            int(counts_by_variant[int(variant_idx)]),
                        )
                    )
                    if counts_by_offset[(int(offset_r), int(offset_c))] != 1:
                        continue
                    if counts_by_variant[int(variant_idx)] != 1:
                        continue
                    valid.append(
                        (
                            int(variant_idx),
                            int(offset_r),
                            int(offset_c),
                            int(len(matches)),
                            int(marker_r),
                            int(marker_c),
                        )
                    )

            if valid:
                cases[(int(H), int(W))] = valid

    assert cases
    # Exhaustive scan of the legal crop regime shows that every ambiguous crop for this motif
    # is a two-way collision where the chosen offset and chosen variant are each unique inside
    # the collision set. That means the fusion can visibly carry both clues, but the generator
    # cannot force "both clues still ambiguous alone" samples without leaving the stated regime.
    assert collision_signatures == {(1, 1)}
    return cases


def _group_cases_by_variant(
    cases: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]]
) -> dict[int, list[tuple[int, int, int, int, int, int, int]]]:
    grouped: dict[int, list[tuple[int, int, int, int, int, int, int]]] = {variant_idx: [] for variant_idx in range(8)}
    for (H, W), case_list in cases.items():
        for variant_idx, offset_r, offset_c, ambiguity, marker_r, marker_c in case_list:
            grouped[int(variant_idx)].append(
                (
                    int(H),
                    int(W),
                    int(offset_r),
                    int(offset_c),
                    int(ambiguity),
                    int(marker_r),
                    int(marker_c),
                )
            )
    assert all(grouped[int(variant_idx)] for variant_idx in range(8))
    return grouped


_VARIANTS = _dihedral_variants(_pattern_mask())
_FUSION_CASES = _build_fusion_cases()
_FUSION_CASES_BY_VARIANT = _group_cases_by_variant(_FUSION_CASES)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the full 20x20 motif from a crop plus two decoded metadata cues.

    Input structure:
    - The right side is a two-color rectangular crop taken from one dihedral variant of the shared 20x20 motif.
    - On a separate 20x20 board at the left, a counted corner block encodes the variant index and one isolated pixel encodes the crop offset.

    Rule:
    - Parse the crop rectangle, count the top-left corner metadata cells to pick the single allowed dihedral variant, and decode the singleton's row/column into the crop offset.
    - With variant and offset fixed, test only the two possible foreground/background polarities for that one window.
    - Repaint the full 20x20 motif using the crop's two colors.

    Key insight:
    - The crop no longer carries all hidden state by itself. Orientation comes from the counted corner block and position comes from the singleton marker, so solving is mostly metadata decoding followed by a local consistency check.
    """
    g = np.asarray(grid, dtype=int)
    crop, (marker_r, marker_c), variant_idx = _extract_crop_marker_and_variant(g)
    H, W = map(int, crop.shape)

    offset_r = _decode_offset(int(marker_r), H)
    offset_c = _decode_offset(int(marker_c), W)
    pattern = _VARIANTS[int(variant_idx)]

    crop_colors = colors_present(crop)
    assert len(crop_colors) == 2

    matches: list[tuple[int, int]] = []
    for fg in crop_colors:
        bg = _other_color(crop_colors, int(fg))
        visible = crop == int(fg)
        if np.array_equal(pattern[offset_r : offset_r + H, offset_c : offset_c + W], visible):
            matches.append((int(fg), int(bg)))

    assert len(matches) == 1
    fg, bg = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def generate_input() -> np.ndarray:
    variant_idx = random.randrange(len(_VARIANTS))
    H, W, offset_r, offset_c, _ambiguity, marker_r, marker_c = random.choice(_FUSION_CASES_BY_VARIANT[int(variant_idx)])
    pattern = _VARIANTS[int(variant_idx)]

    fg, bg, meta_color = random.sample(_PALETTE, 3)
    output = np.where(pattern, int(fg), int(bg)).astype(int)
    crop = output[offset_r : offset_r + H, offset_c : offset_c + W]

    grid = np.zeros((_BOARD_SIZE, _RIGHT_START + W), dtype=int)
    _encode_corner_block(grid, int(meta_color), int(variant_idx))
    grid[0:H, _RIGHT_START : _RIGHT_START + W] = crop
    grid[int(marker_r), int(marker_c)] = int(meta_color)

    matches = _find_matches(crop)
    assert len(matches) > 1
    assert np.array_equal(main(grid), output)
    return grid
