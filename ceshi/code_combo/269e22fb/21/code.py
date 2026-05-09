# Combo variant of 269e22fb (index 21)
# combo_spec: D10/1+D16/0
# Name: 多裁剪块迭代拼图补全

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
from arc_utils import colors_present


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

_CANVAS = 20
_SEP_WIDTH = 1
_GEN_RETRIES = 300


# Design analysis for combo D10/1 + D16/0
# ----------------------------------------
# a) Concrete embodiment and behavior trace
#    - Original task:
#      one two-color crop -> identify one (polarity, dihedral variant, offset) -> repaint the
#      complete 20x20 motif.
#    - Fused task:
#      the input is a horizontal strip of 2-4 shuffled carrier panels separated by solid
#      divider columns in a third color. Each panel reveals one rectangular crop from the same
#      hidden motif instance. The source windows jointly cover the whole 20x20 board and at
#      least one pair overlaps, so the solver must reconcile several local matches into one
#      global reconstruction before repainting the single 20x20 output.
#    - Difference:
#      D16 contributes the explicit many-input-one-output panelization, while D10 contributes
#      the joint placement search across multiple overlapping windows.
#
# b) Complexity calibration
#    - The generator uses all three piece-count branches (2, 3, 4) with equal probability.
#    - Two- and three-piece cases are overlapping full-width bands; four-piece cases are an
#      overlapped 2x2 partition. These families are visually tidy and make the "same picture,
#      several windows" rule obvious to humans.
#    - Panel order is shuffled and the vertical padding inside panels is randomized so the
#      presentation does not leak the hidden source offsets.
#
# c) Hidden assumptions reviewed against the original main()
#    - The whole input may be much wider than 20, so the solver must detect separator columns
#      and recover panels before any motif matching.
#    - The whole input uses three colors, not two, so the solver must infer which color plays
#      the divider role from the frame structure rather than from color ids.
#    - Each crop has its own unknown offset. The single-window matching loop from the original
#      task therefore becomes a coordinated multi-window coverage search.
#
# d) Placement legality guaranteed by the generator
#    - Every crop is an exact rectangular window of one dihedral motif variant.
#    - The union of the chosen windows always covers the complete 20x20 board.
#    - Every sample contains genuine overlap between windows, which keeps the D10-style
#      "check consistency across several placements" behavior visible instead of reducing the
#      input to a clean non-overlapping partition.
#
# e) Visual observability and cleanliness
#    - The separator color appears only as panel padding and divider columns.
#    - Each panel contains exactly one intact rectangle, with no internal masking noise.
#    - The output remains a single recolored 20x20 motif, so the many-to-one fusion rule is
#      legible after only a few examples.


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


_VARIANTS = _dihedral_variants(_pattern_mask())


def _frame_has_single_crop(frame: np.ndarray, sep: int) -> bool:
    rows = np.flatnonzero(np.any(frame != int(sep), axis=1))
    if len(rows) == 0:
        return False
    r0 = int(rows[0])
    r1 = int(rows[-1]) + 1
    if np.any(frame[:r0, :] != int(sep)) or np.any(frame[r1:, :] != int(sep)):
        return False
    return bool(np.all(frame[r0:r1, :] != int(sep)))


def _segment_frames(grid: np.ndarray, sep: int) -> list[np.ndarray]:
    h, w = map(int, grid.shape)
    assert h == _CANVAS

    is_divider = [bool(np.all(grid[:, c] == int(sep))) for c in range(w)]
    if not any(is_divider):
        return []

    frames: list[np.ndarray] = []
    c = 0
    while c < w:
        while c < w and is_divider[c]:
            c += 1
        start = c
        while c < w and not is_divider[c]:
            c += 1
        if start < c:
            frames.append(grid[:, start:c].copy())
    return frames


def _split_frames(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    h, w = map(int, grid.shape)
    assert h == _CANVAS

    solid_colors = sorted({int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])})
    candidates: list[tuple[int, list[np.ndarray]]] = []
    for sep in solid_colors:
        frames = _segment_frames(grid, sep)
        if not (2 <= len(frames) <= 4):
            continue
        if all(_frame_has_single_crop(frame, sep) for frame in frames):
            candidates.append((int(sep), frames))

    assert len(candidates) == 1, "separator color should be structurally identifiable"
    sep, frames = candidates[0]
    return frames, int(sep)


def _extract_visible_crop(frame: np.ndarray, sep: int) -> np.ndarray:
    rows = np.flatnonzero(np.any(frame != int(sep), axis=1))
    assert len(rows) > 0
    r0 = int(rows[0])
    r1 = int(rows[-1]) + 1
    assert np.all(frame[:r0, :] == int(sep))
    assert np.all(frame[r1:, :] == int(sep))
    crop = frame[r0:r1, :]
    assert np.all(crop != int(sep))
    return crop.copy()


def _candidate_positions(pattern: np.ndarray, visible: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, visible.shape)
    out: list[tuple[int, int]] = []
    for r in range(_CANVAS - h + 1):
        for c in range(_CANVAS - w + 1):
            if np.array_equal(pattern[r : r + h, c : c + w], visible):
                out.append((int(r), int(c)))
    return out


def _find_one_joint_placement(
    visible_masks: list[np.ndarray],
    sizes: list[tuple[int, int]],
    pattern: np.ndarray,
) -> tuple[tuple[int, int], ...] | None:
    candidate_lists = [_candidate_positions(pattern, visible) for visible in visible_masks]
    if any(len(cands) == 0 for cands in candidate_lists):
        return None

    possible_cover_masks: list[np.ndarray] = []
    for placements, (h, w) in zip(candidate_lists, sizes):
        cover = np.zeros((_CANVAS, _CANVAS), dtype=bool)
        for r, c in placements:
            cover[r : r + h, c : c + w] = True
        possible_cover_masks.append(cover)

    order = sorted(range(len(sizes)), key=lambda i: (len(candidate_lists[i]), -(sizes[i][0] * sizes[i][1])))
    placements: list[tuple[int, int]] = [(-1, -1)] * len(sizes)
    found: tuple[tuple[int, int], ...] | None = None

    def dfs(idx: int, covered: np.ndarray) -> bool:
        nonlocal found
        if idx == len(order):
            if covered.all():
                found = tuple((int(r), int(c)) for r, c in placements)
                return True
            return False

        piece_idx = order[idx]
        h, w = sizes[piece_idx]
        for r, c in candidate_lists[piece_idx]:
            placements[piece_idx] = (int(r), int(c))
            new_covered = covered.copy()
            new_covered[r : r + h, c : c + w] = True

            optimistic = new_covered.copy()
            for t in range(idx + 1, len(order)):
                optimistic |= possible_cover_masks[order[t]]
            if not optimistic.all():
                continue

            if dfs(idx + 1, new_covered):
                return True

        placements[piece_idx] = (-1, -1)
        return False

    dfs(0, np.zeros((_CANVAS, _CANVAS), dtype=bool))
    return found


def _find_joint_solutions(
    grid: np.ndarray,
    limit: int | None = None,
) -> list[tuple[int, int, int, tuple[tuple[int, int], ...]]]:
    g = np.asarray(grid, dtype=int)
    frames, sep = _split_frames(g)
    crops = [_extract_visible_crop(frame, sep) for frame in frames]
    colors = colors_present(g, exclude={sep})
    assert len(colors) == 2

    solutions: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
    seen_outputs: set[bytes] = set()
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible_masks = [crop == int(fg) for crop in crops]
        sizes = [tuple(map(int, crop.shape)) for crop in crops]

        for variant_idx, pattern in enumerate(_VARIANTS):
            placements = _find_one_joint_placement(visible_masks, sizes, pattern)
            if placements is None:
                continue
            out = np.where(pattern, int(fg), int(bg)).astype(int)
            key = out.tobytes()
            if key in seen_outputs:
                continue
            seen_outputs.add(key)
            solutions.append((int(fg), int(bg), int(variant_idx), placements))
            if limit is not None and len(solutions) >= limit:
                return solutions

    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 motif from several separated crop panels that overlap on the hidden source.

    Input structure:
    - The grid is a 20-row horizontal strip split by solid full-height divider columns in a third color.
    - Each non-divider panel contains exactly one visible rectangular crop from the same dihedral variant of the fixed 20x20 motif.
    - There are 2-4 such panels; they are shuffled, vertically padded for presentation, and their source windows jointly cover the hidden 20x20 canvas.

    Rule:
    - Ignore the divider color, split the strip into panels, and extract the visible crop from each panel.
    - Try each remaining color as motif foreground and test every crop against every placement of every dihedral motif variant.
    - Keep the unique full-pattern solution for which all crops can be placed on one shared 20x20 motif and their placed windows cover all 400 cells.
    - Repaint that entire 20x20 motif with the matched foreground/background colors.

    Key insight:
    - The panels are not independent clues; they are several windows into one hidden motif instance, so the correct answer is the only global assembly where all local matches agree under a common orientation and jointly cover the whole board.
    """
    g = np.asarray(grid, dtype=int)
    solutions = _find_joint_solutions(g, limit=2)
    assert len(solutions) == 1
    fg, bg, variant_idx, _placements = solutions[0]
    pattern = _VARIANTS[variant_idx]
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _make_panel(crop: np.ndarray, sep: int, display_top: int) -> np.ndarray:
    h, w = map(int, crop.shape)
    panel = np.full((_CANVAS, w), int(sep), dtype=int)
    panel[display_top : display_top + h, :] = crop
    return panel


def _compose_input(panels: list[np.ndarray], sep: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    divider = np.full((_CANVAS, _SEP_WIDTH), int(sep), dtype=int)
    for idx, panel in enumerate(panels):
        if idx:
            pieces.append(divider)
        pieces.append(panel)
    return np.hstack(pieces)


def _transform_windows(
    windows: list[tuple[int, int, int, int]],
    k: int,
    do_flip: bool,
) -> list[tuple[int, int, int, int]]:
    transformed: list[tuple[int, int, int, int]] = []
    for r, c, h, w in windows:
        mask = np.zeros((_CANVAS, _CANVAS), dtype=bool)
        mask[r : r + h, c : c + w] = True
        moved = np.rot90(mask, k)
        if do_flip:
            moved = np.fliplr(moved)
        coords = np.argwhere(moved)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rect = moved[r0 : r1 + 1, c0 : c1 + 1]
        assert rect.all()
        transformed.append((int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1)))
    return transformed


def _band_windows(piece_count: int, rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    if piece_count == 2:
        split_r = int(rng.integers(8, 13))
        overlap = int(rng.integers(1, 4))
        windows = [
            (0, 0, split_r + overlap, _CANVAS),
            (split_r - overlap, 0, _CANVAS - (split_r - overlap), _CANVAS),
        ]
    else:
        split_r1 = int(rng.integers(5, 8))
        split_r2 = int(rng.integers(12, 16))
        overlap1 = int(rng.integers(1, 3))
        overlap2 = int(rng.integers(1, 3))
        windows = [
            (0, 0, split_r1 + overlap1, _CANVAS),
            (split_r1 - overlap1, 0, split_r2 + overlap2 - (split_r1 - overlap1), _CANVAS),
            (split_r2 - overlap2, 0, _CANVAS - (split_r2 - overlap2), _CANVAS),
        ]

    k = int(rng.choice([0, 2]))
    return _transform_windows(windows, k, bool(rng.integers(0, 2)))


def _quadrant_windows(rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    overlap_r = int(rng.integers(1, 3))
    overlap_c = int(rng.integers(1, 3))
    split_r = int(rng.integers(8, 13))
    split_c = int(rng.integers(8, 13))
    top = split_r + overlap_r
    left = split_c + overlap_c
    r0 = split_r - overlap_r
    c0 = split_c - overlap_c
    windows = [
        (0, 0, top, left),
        (0, c0, top, _CANVAS - c0),
        (r0, 0, _CANVAS - r0, left),
        (r0, c0, _CANVAS - r0, _CANVAS - c0),
    ]
    return _transform_windows(windows, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))


def _window_template(piece_count: int, rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    if piece_count in (2, 3):
        return _band_windows(piece_count, rng)
    if piece_count == 4:
        return _quadrant_windows(rng)
    raise AssertionError("piece_count must be 2, 3, or 4")


def _windows_overlap_area(windows: list[tuple[int, int, int, int]]) -> int:
    cover = np.zeros((_CANVAS, _CANVAS), dtype=bool)
    area_sum = 0
    for r, c, h, w in windows:
        cover[r : r + h, c : c + w] = True
        area_sum += int(h * w)
    return area_sum - int(cover.sum())


def _covers_full_canvas(windows: list[tuple[int, int, int, int]]) -> bool:
    cover = np.zeros((_CANVAS, _CANVAS), dtype=bool)
    for r, c, h, w in windows:
        cover[r : r + h, c : c + w] = True
    return bool(cover.all())


def _generate_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    fg, bg, sep = map(int, rng.choice(10, size=3, replace=False))
    pattern = _VARIANTS[int(rng.integers(len(_VARIANTS)))]
    output = np.where(pattern, int(fg), int(bg)).astype(int)

    piece_count = int(rng.integers(2, 5))
    windows = _window_template(piece_count, rng)
    if not _covers_full_canvas(windows):
        raise AssertionError("windows must cover the full canvas")
    if _windows_overlap_area(windows) <= 0:
        raise AssertionError("windows must overlap")
    crops = [output[r : r + h, c : c + w].copy() for r, c, h, w in windows]
    if any(len(colors_present(crop)) != 2 for crop in crops):
        raise AssertionError("monochrome crop")

    panels = [
        _make_panel(crop, sep, int(rng.integers(0, _CANVAS - crop.shape[0] + 1)))
        for crop in crops
    ]
    rng.shuffle(panels)
    grid = _compose_input(panels, sep)

    meta = {
        "piece_count": int(piece_count),
        "overlap_area": int(_windows_overlap_area(windows)),
    }
    return grid, output, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(_GEN_RETRIES):
        try:
            grid, expected, _meta = _generate_attempt(rng)
        except Exception:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid combo sample")
