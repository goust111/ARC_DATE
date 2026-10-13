from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import hashlib
import random

import numpy as np


BACKGROUND = 0
MASK_COLOR = 5
ARC_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive (r0, c0, r1, c1) bbox of a nonempty mask."""
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _row_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """List True runs as (row, inclusive_start_col, inclusive_end_col)."""
    segments: list[tuple[int, int, int]] = []
    height, width = mask.shape
    for r in range(height):
        c = 0
        while c < width:
            if not bool(mask[r, c]):
                c += 1
                continue
            c0 = c
            while c + 1 < width and bool(mask[r, c + 1]):
                c += 1
            segments.append((int(r), int(c0), int(c)))
            c += 1
    return segments


def _component_count(mask: np.ndarray) -> int:
    """Count 4-connected True components without requiring scipy."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        count += 1
        seen[r, c] = True
        queue = deque([(r, c)])
        while queue:
            rr, cc = queue.popleft()
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
    return int(count)


def _bottom_tape(g: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    """Find the unique contiguous non-background run in the bottom row."""
    occupied = np.flatnonzero(g[-1] != int(bg))
    assert occupied.size, "expected a palette tape on the bottom row"
    c0, c1 = int(occupied[0]), int(occupied[-1])
    assert occupied.size == c1 - c0 + 1, "palette tape must be contiguous"
    tape = g[-1, c0 : c1 + 1].astype(int, copy=True)
    assert np.all(tape != int(bg)), "palette tape cannot contain background gaps"
    assert np.all(tape != MASK_COLOR), "grey is reserved for the mask"
    return c0, tape


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor cropped grey mask row-segments using consecutive quotas from the bottom palette tape.

    Input structure:
    - The most common color is the background; one large 4-connected grey mask lies above the bottom row.
    - Background holes split every mask row into multiple horizontal grey segments.
    - A horizontally offset, contiguous one-row palette tape occupies part of the bottom row; its color runs have unequal lengths.

    Rule:
    - Crop to the inclusive bbox of the grey mask and list its horizontal segments in row-major order: top to bottom, then left to right.
    - Read every palette-tape pixel from left to right.  Its color run of length n therefore supplies that color to the next n mask segments.
    - Recolor every pixel of segment i with tape pixel i, while preserving all background holes inside the crop.

    Key insight:
    - Tape length is conserved as the number of mask row-segments: color-run lengths are sequential quotas, not mask lengths or aligned columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids cannot exceed 30 in either dimension"

    counts = Counter(map(int, g.ravel()))
    bg, bg_count = counts.most_common(1)[0]
    assert sum(1 for count in counts.values() if count == bg_count) == 1, "background must be uniquely most common"

    mask_full = g == MASK_COLOR
    assert mask_full.any(), "expected a grey mask"
    assert not mask_full[-1].any(), "mask and bottom palette tape must be separated"
    assert _component_count(mask_full) == 1, "grey mask must be one 4-connected object"

    r0, c0, r1, c1 = _bbox(mask_full)
    mask_crop = mask_full[r0 : r1 + 1, c0 : c1 + 1]
    segments = _row_segments(mask_crop)
    assert all(sum(1 for rr, _a, _b in segments if rr == r) >= 2 for r in range(mask_crop.shape[0])), (
        "background holes must split every mask row"
    )

    tape_c0, tape = _bottom_tape(g, bg)
    assert tape_c0 != c0, "mask and tape must be horizontally offset"
    assert len(segments) == int(tape.size), "mask segment count must equal palette tape length"

    allowed = mask_full.copy()
    allowed[-1, tape_c0 : tape_c0 + tape.size] = True
    assert np.all((g == int(bg)) | allowed), "only the mask and palette tape may be foreground"

    output = np.full(mask_crop.shape, int(bg), dtype=int)
    for color, (r, start, end) in zip(map(int, tape), segments):
        output[r, start : end + 1] = int(color)
    return output


def _random_holes(width: int, hole_count: int, forbidden: set[int]) -> set[int] | None:
    """Pick separated one-cell holes, avoiding holes in the preceding row."""
    candidates = [c for c in range(1, width - 1) if c not in forbidden]
    random.shuffle(candidates)
    chosen: list[int] = []
    for c in candidates:
        if all(abs(c - other) > 1 for other in chosen):
            chosen.append(int(c))
            if len(chosen) == hole_count:
                return set(chosen)
    return None


def _make_mask(height: int, width: int, complex_mode: bool) -> np.ndarray | None:
    """Construct a connected perforated mask with 2-4 runs in every row."""
    if complex_mode:
        segment_counts = [random.choices((2, 3, 4), weights=(2, 4, 3))[0] for _ in range(height)]
        if max(segment_counts) < 3:
            segment_counts[random.randrange(height)] = 3
        if height >= 6 and min(segment_counts) == max(segment_counts):
            segment_counts[random.randrange(height)] = 2
    else:
        segment_counts = [2] * height

    mask = np.ones((height, width), dtype=bool)
    previous_holes: set[int] = set()
    for r, segment_count in enumerate(segment_counts):
        holes = _random_holes(width, segment_count - 1, previous_holes)
        if holes is None:
            return None
        for c in holes:
            mask[r, c] = False
        previous_holes = holes

    if _component_count(mask) != 1:
        return None
    if [sum(1 for rr, _a, _b in _row_segments(mask) if rr == r) for r in range(height)] != segment_counts:
        return None
    return mask


def _quota_options(total: int) -> list[tuple[int, ...]]:
    """Unequal positive quotas; each color controls at least two segments."""
    options: list[tuple[int, ...]] = []
    for run_count in range(3, 6):
        for values in combinations(range(2, 9), run_count):
            if sum(values) == total:
                options.append(tuple(map(int, values)))
    return options


def _assigned_colors(quotas: tuple[int, ...], colors: list[int]) -> np.ndarray:
    return np.concatenate([
        np.full(quota, color, dtype=int) for quota, color in zip(quotas, colors)
    ])


def _independent_expected(mask: np.ndarray, tape: np.ndarray, bg: int) -> np.ndarray:
    """Render the stated row-major rule directly for generator oracle checks."""
    out = np.full(mask.shape, int(bg), dtype=int)
    cursor = 0
    for r in range(mask.shape[0]):
        c = 0
        while c < mask.shape[1]:
            if not mask[r, c]:
                c += 1
                continue
            start = c
            while c + 1 < mask.shape[1] and mask[r, c + 1]:
                c += 1
            if cursor >= tape.size:
                raise ValueError("more mask segments than tape pixels")
            out[r, start : c + 1] = int(tape[cursor])
            cursor += 1
            c += 1
    if cursor != tape.size:
        raise ValueError("more tape pixels than mask segments")
    return out


def _render_with_sequence(mask: np.ndarray, sequence: np.ndarray, bg: int) -> np.ndarray:
    """Small alternate oracle used to reject reverse-order degeneracy."""
    out = np.full(mask.shape, int(bg), dtype=int)
    for color, (r, c0, c1) in zip(map(int, sequence), _row_segments(mask)):
        out[r, c0 : c1 + 1] = int(color)
    return out


def _passes_anti_shortcut_checks(mask: np.ndarray, tape: np.ndarray, quotas: tuple[int, ...], bg: int) -> bool:
    """Reject examples solvable by segment length, start column, reversal, or aligned columns."""
    segments = _row_segments(mask)
    lengths = [c1 - c0 + 1 for _r, c0, c1 in segments]
    colors = list(map(int, tape))

    # Deliberate bait: at least one geometric segment length equals a quota.
    if not (set(lengths) & set(quotas)):
        return False

    # A length->color dictionary must contradict itself on the same visible length.
    length_colors: dict[int, set[int]] = {}
    for length, color in zip(lengths, colors):
        length_colors.setdefault(length, set()).add(color)
    if not any(len(values) >= 2 for values in length_colors.values()):
        return False

    # Likewise, a segment-start-column lookup must contradict itself.
    start_colors: dict[int, set[int]] = {}
    for (_r, c0, _c1), color in zip(segments, colors):
        start_colors.setdefault(c0, set()).add(color)
    if not any(len(values) >= 2 for values in start_colors.values()):
        return False

    expected = _render_with_sequence(mask, tape, bg)
    if np.array_equal(expected, _render_with_sequence(mask, tape[::-1], bg)):
        return False

    # Every quota group contains geometrically unequal segments, so quota expansion
    # cannot be mistaken for copying one representative segment shape.
    cursor = 0
    varied_groups = 0
    for quota in quotas:
        group_lengths = lengths[cursor : cursor + quota]
        varied_groups += int(len(set(group_lengths)) >= 2)
        cursor += quota
    if varied_groups < max(2, len(quotas) - 1):
        return False
    return True


def _semantic_validation(
    inp: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    tape: np.ndarray,
    quotas: tuple[int, ...],
    mask_pos: tuple[int, int],
    tape_col: int,
) -> bool:
    """Validate the puzzle semantics independently of mere generator/solver closure."""
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if expected.shape != mask.shape or int(tape.size) != len(_row_segments(mask)):
        return False
    if sum(quotas) != int(tape.size) or len(set(quotas)) != len(quotas):
        return False
    if mask_pos[1] == tape_col or _component_count(mask) != 1:
        return False
    if not all(sum(1 for rr, _a, _b in _row_segments(mask) if rr == r) >= 2 for r in range(mask.shape[0])):
        return False
    if not _passes_anti_shortcut_checks(mask, tape, quotas, BACKGROUND):
        return False
    if np.array_equal(inp, expected) or inp.shape == expected.shape:
        return False

    # Independently reconstructed truth must agree, holes must remain background,
    # and every tape color must visibly occur in the answer.
    truth = _independent_expected(mask, tape, BACKGROUND)
    if not np.array_equal(truth, expected) or not np.all(expected[~mask] == BACKGROUND):
        return False
    if set(map(int, np.unique(expected))) - {BACKGROUND} != set(map(int, np.unique(tape))):
        return False
    try:
        solved = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    return np.array_equal(solved, expected)


_SEEN_OUTPUTS: set[str] = set()


def _generation_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    complex_mode = random.random() < 0.5
    mask_h = random.randint(5, 8)
    mask_w = random.randint(9, 15)
    mask = _make_mask(mask_h, mask_w, complex_mode)
    if mask is None:
        return None

    segment_count = len(_row_segments(mask))
    if segment_count > 27:
        return None
    options = _quota_options(segment_count)
    if not options:
        return None
    quotas = list(random.choice(options))
    random.shuffle(quotas)
    quotas_tuple = tuple(quotas)

    palette_pool = [color for color in ARC_COLORS if color not in (BACKGROUND, MASK_COLOR)]
    colors = random.sample(palette_pool, len(quotas_tuple))
    tape = _assigned_colors(quotas_tuple, colors)
    if not _passes_anti_shortcut_checks(mask, tape, quotas_tuple, BACKGROUND):
        return None

    width_floor = max(mask_w, int(tape.size))
    grid_w = random.randint(width_floor + 2, min(30, width_floor + 5))
    top_margin = random.randint(1, 3)
    gap = random.randint(2, 4)
    grid_h = top_margin + mask_h + gap + 1
    if grid_h > 30:
        return None

    mask_columns = list(range(grid_w - mask_w + 1))
    tape_columns = list(range(grid_w - int(tape.size) + 1))
    random.shuffle(mask_columns)
    random.shuffle(tape_columns)
    positions = [(mc, tc) for mc in mask_columns for tc in tape_columns if mc != tc]
    if not positions:
        return None
    mask_c, tape_c = random.choice(positions)
    mask_r = top_margin

    inp = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    inp[mask_r : mask_r + mask_h, mask_c : mask_c + mask_w][mask] = MASK_COLOR
    inp[-1, tape_c : tape_c + tape.size] = tape
    expected = _independent_expected(mask, tape, BACKGROUND)

    if not _semantic_validation(
        inp, expected, mask, tape, quotas_tuple, (mask_r, mask_c), tape_c
    ):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked quota-tape puzzle input."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _generation_attempt()
        if result is None:
            continue
        inp, expected = result
        signature = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return inp
    raise RuntimeError("failed to generate a non-degenerate quota-tape input")
