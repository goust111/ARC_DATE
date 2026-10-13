from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7)
PAIR_COUNT_CHOICES = (2, 3)
HEIGHT_RANGE = (13, 18)
WIDTH_RANGE = (16, 24)
INPUT_GAPS = (4, 5, 6, 7, 8)
OBJECT_SIZE = 3
MINORITY_COUNT = 4
MAJORITY_COUNT = 5
MAX_GEN_ATTEMPTS = 500

ALL_COLOR_PAIRS = tuple(itertools.combinations(PALETTE, 2))
ALL_MINORITY_MASKS = tuple(itertools.combinations(range(OBJECT_SIZE * OBJECT_SIZE), MINORITY_COUNT))

_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _find_objects(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            pixels: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                pixels.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) != bg:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(pixels, dtype=int)
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            values = [int(v) for v in grid[coords[:, 0], coords[:, 1]]]
            counts = Counter(values)
            objects.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "patch": patch,
                    "counts": counts,
                    "colors": tuple(sorted(counts)),
                    "area": int(len(coords)),
                }
            )

    return objects


def _minority_and_majority(counts: Counter[int]) -> tuple[int, int]:
    assert len(counts) == 2, "each object must use exactly two colors"
    minority = [color for color, count in counts.items() if count == MINORITY_COUNT]
    majority = [color for color, count in counts.items() if count == MAJORITY_COUNT]
    assert len(minority) == 1 and len(majority) == 1, "expected a 4/5 color split"
    return int(minority[0]), int(majority[0])


def _swap_patch(patch: np.ndarray) -> tuple[np.ndarray, int, int]:
    values = [int(v) for v in patch.reshape(-1)]
    counts = Counter(values)
    minority, majority = _minority_and_majority(counts)
    swapped = patch.copy()
    swapped[patch == majority] = -1
    swapped[patch == minority] = majority
    swapped[swapped == -1] = minority
    return swapped.astype(int), minority, majority


def _pair_objects(objects: list[dict]) -> list[tuple[dict, dict]]:
    assert objects, "expected foreground objects"
    groups: dict[tuple[int, int, tuple[int, int]], list[dict]] = {}
    seen_color_pairs: set[tuple[int, int]] = set()

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (OBJECT_SIZE, OBJECT_SIZE), "expected 3x3 objects"
        assert obj["area"] == OBJECT_SIZE * OBJECT_SIZE, "expected each 3x3 box to be solid"
        assert len(obj["colors"]) == 2, "expected each object to be bicolored"
        _minority_and_majority(obj["counts"])
        key = (r0, r1, obj["colors"])
        groups.setdefault(key, []).append(obj)

    assert 2 <= len(groups) <= 3, "expected 2-3 pairs"
    pairs: list[tuple[dict, dict]] = []

    for (_, _, colors), members in sorted(groups.items(), key=lambda item: (item[0][0], item[0][2])):
        assert len(members) == 2, "each color-pair row band must contain exactly two objects"
        assert colors not in seen_color_pairs, "different pairs must use different color combinations"
        seen_color_pairs.add(colors)

        left, right = sorted(members, key=lambda obj: obj["bbox"][1])
        left_gap = int(right["bbox"][1] - left["bbox"][3] - 1)
        assert left_gap >= 1, "paired objects must be separated horizontally"
        pairs.append((left, right))

    return pairs


def _settled_columns(left_bbox: tuple[int, int, int, int], right_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    gap = int(right_bbox[1] - left_bbox[3] - 1)
    assert gap >= 1
    inward = gap - 1
    # Discrete tie-break for even gaps: the left object takes the extra inward step.
    left_shift = (inward + 1) // 2
    right_shift = inward // 2
    left_c0 = int(left_bbox[1] + left_shift)
    right_c0 = int(right_bbox[1] - right_shift)
    assert right_c0 - (left_c0 + OBJECT_SIZE - 1) - 1 == 1
    return left_c0, right_c0


def _place_patch(canvas: np.ndarray, r0: int, c0: int, patch: np.ndarray, bg: int = BG) -> None:
    h, w = patch.shape
    region = canvas[r0 : r0 + h, c0 : c0 + w]
    assert region.shape == patch.shape
    assert np.all(region == bg), "objects must not overlap in the output"
    region[:, :] = patch


def _explicit_expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pairs = _pair_objects(_find_objects(g, bg=BG))
    out = np.full_like(g, BG)

    parities = set()
    relation_modes = set()

    for left, right in pairs:
        r0 = int(left["bbox"][0])
        assert left["bbox"][0] == right["bbox"][0] and left["bbox"][2] == right["bbox"][2]

        left_swapped, left_minority, _ = _swap_patch(left["patch"])
        right_swapped, right_minority, _ = _swap_patch(right["patch"])
        relation_modes.add(int(left_minority != right_minority))

        gap = int(right["bbox"][1] - left["bbox"][3] - 1)
        assert 4 <= gap <= 8, "generator must use the requested input gap range"
        parities.add(gap % 2)

        left_c0, right_c0 = _settled_columns(left["bbox"], right["bbox"])
        _place_patch(out, r0, left_c0, left_swapped)
        _place_patch(out, r0, right_c0, right_swapped)
        out[r0 : r0 + OBJECT_SIZE, left_c0 + OBJECT_SIZE] = left_minority

    if len(pairs) >= 2:
        assert parities == {0, 1}, "generated grids should expose both odd and even gap behavior"
        assert relation_modes == {0, 1}, "generated grids should expose both majority-agreement modes"

    assert int(np.count_nonzero(out)) == len(pairs) * OBJECT_SIZE * (OBJECT_SIZE * 2 + 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each paired bicolored 3x3 square inward, swap each square's two colors, and fill the final gap with the left square's pre-swap minority color.

    Input structure:
    - The grid has black background 0 and contains 2-3 pairs of separated solid 3x3 objects.
    - Each object is fully filled, 4-connected, and uses exactly two non-background colors split 5-to-4.
    - The two objects in a pair share the same unordered color pair and exactly the same row span, while different pairs use different color pairs.

    Rule:
    - Identify each pair by matching same-row 3x3 objects that use the same two colors.
    - Slide the left object rightward and the right object leftward until their bounding boxes are separated by exactly one empty column.
    - After sliding, swap the two colors inside each 3x3 object without changing any pixel positions.
    - Fill the 1x3 gap column between the two settled objects with the left object's minority color from before the swap; everything else stays background.

    Key insight:
    - The final gap is created by motion, but its color is determined by the left object's original 4-cell minority color, so the solver must pair objects, simulate the collision stop, and track the pre-swap color role.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(np.min(g)) >= 0
    assert BG in set(map(int, np.unique(g))), "expected background color 0"

    out = np.full_like(g, BG)
    for left, right in _pair_objects(_find_objects(g, bg=BG)):
        r0 = int(left["bbox"][0])
        left_swapped, left_minority, _ = _swap_patch(left["patch"])
        right_swapped, _, _ = _swap_patch(right["patch"])
        left_c0, right_c0 = _settled_columns(left["bbox"], right["bbox"])

        _place_patch(out, r0, left_c0, left_swapped)
        _place_patch(out, r0, right_c0, right_swapped)
        gap_col = left_c0 + OBJECT_SIZE
        assert gap_col == right_c0 - 1
        out[r0 : r0 + OBJECT_SIZE, gap_col] = left_minority

    return out


def _sample_row_starts(height: int, pair_count: int) -> list[int] | None:
    choices = list(range(height - OBJECT_SIZE + 1))
    for _ in range(80):
        rows = sorted(random.sample(choices, pair_count))
        if all(rows[i + 1] - rows[i] >= OBJECT_SIZE + 1 for i in range(pair_count - 1)):
            return rows
    return None


def _gap_plan(pair_count: int) -> list[int]:
    gaps = [random.choice((4, 6, 8)), random.choice((5, 7))]
    while len(gaps) < pair_count:
        gaps.append(random.choice(INPUT_GAPS))
    random.shuffle(gaps)
    return gaps


def _relation_plan(pair_count: int) -> list[bool]:
    modes = [False, True]
    while len(modes) < pair_count:
        modes.append(bool(random.getrandbits(1)))
    random.shuffle(modes)
    return modes


def _build_patch(colors: tuple[int, int], minority_color: int, minority_mask: tuple[int, ...]) -> np.ndarray:
    major_color = colors[0] if colors[1] == minority_color else colors[1]
    patch = np.full((OBJECT_SIZE, OBJECT_SIZE), major_color, dtype=int)
    for idx in minority_mask:
        r, c = divmod(idx, OBJECT_SIZE)
        patch[r, c] = minority_color
    return patch


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return tuple(arr.shape), arr.tobytes()


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    row_starts = _sample_row_starts(height, pair_count)
    if row_starts is None:
        return None

    gaps = _gap_plan(pair_count)
    width = max(width, max(gaps) + 8)
    if width > WIDTH_RANGE[1]:
        return None

    color_pairs = random.sample(list(ALL_COLOR_PAIRS), pair_count)
    relation_modes = _relation_plan(pair_count)

    grid = np.zeros((height, width), dtype=int)

    for row, colors, gap, opposite_majority in zip(row_starts, color_pairs, gaps, relation_modes):
        left_shift = gap // 2
        right_shift = (gap - 1) // 2
        final_left_min = 1 + left_shift
        final_left_max = width - right_shift - 8
        if final_left_min > final_left_max:
            return None

        final_left = random.randint(final_left_min, final_left_max)
        final_right = final_left + OBJECT_SIZE + 1
        input_left = final_left - left_shift
        input_right = final_right + right_shift
        assert input_right - (input_left + OBJECT_SIZE - 1) - 1 == gap

        left_minority_color = random.choice(colors)
        if opposite_majority:
            right_minority_color = colors[0] if left_minority_color == colors[1] else colors[1]
        else:
            right_minority_color = left_minority_color

        left_mask = random.choice(ALL_MINORITY_MASKS)
        right_mask = random.choice(ALL_MINORITY_MASKS)
        if left_mask == right_mask:
            alternatives = [mask for mask in ALL_MINORITY_MASKS if mask != left_mask]
            right_mask = random.choice(alternatives)

        left_patch = _build_patch(colors, left_minority_color, left_mask)
        right_patch = _build_patch(colors, right_minority_color, right_mask)

        grid[row : row + OBJECT_SIZE, input_left : input_left + OBJECT_SIZE] = left_patch
        grid[row : row + OBJECT_SIZE, input_right : input_right + OBJECT_SIZE] = right_patch

    objects = _find_objects(grid, bg=BG)
    if len(objects) != pair_count * 2:
        return None

    expected = _explicit_expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None

    inp_sig = _signature(grid)
    out_sig = _signature(expected)
    if inp_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None

    _SEEN_INPUTS.add(inp_sig)
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
