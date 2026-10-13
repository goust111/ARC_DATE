from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


# The puzzle names these two visible ARC colors explicitly.
MAROON = 9
TEAL = 8
INPUT_COLORS = tuple(range(8))
MAX_GRID_SIZE = 30
_generation_count = 0


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color, following arc_utils.get_bg."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _foreground_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    """4-connected multicolor components as (row, column) coordinate arrays."""
    foreground = grid != int(bg)
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = grid.shape

    for start_r, start_c in np.argwhere(foreground):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        coords: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and foreground[nr, nc]
                    and not seen[nr, nc]
                ):
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive (r0, c0, r1, c1) bbox, matching arc_utils conventions."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangular_regions(grid: np.ndarray, bg: int) -> list[dict]:
    """Extract full rectangular foreground regions and their opaque crops."""
    regions: list[dict] = []
    for coords in _foreground_components(grid, bg):
        r0, c0, r1, c1 = _bbox(coords)
        height, width = r1 - r0 + 1, c1 - c0 + 1
        assert len(coords) == height * width, "each pattern region must fill its bbox"
        crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.all(crop != bg), "pattern regions may not contain canvas cells"
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (height, width),
                "crop": crop,
            }
        )
    return regions


def _transposed_pair(regions: list[dict]) -> tuple[dict, dict]:
    """Return the unique two regions whose height and width are exchanged."""
    candidates: list[tuple[dict, dict]] = []
    for i, first in enumerate(regions):
        h, w = first["shape"]
        for second in regions[i + 1 :]:
            if second["shape"] == (w, h):
                candidates.append((first, second))
    assert len(candidates) == 1, "expected exactly one transposed-size region pair"
    first, second = candidates[0]
    if first["bbox"][:2] > second["bbox"][:2]:
        first, second = second, first
    return first, second


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    """Solve while exposing the two rotation scores for generator validation."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a nonempty 2D grid"
    bg = _background(g)
    assert bg == MAROON, "expected a maroon canvas"

    regions = _rectangular_regions(g, bg)
    assert len(regions) >= 3, "expected a pair plus at least one distractor region"
    earlier, later = _transposed_pair(regions)
    reference = earlier["crop"]
    moving = later["crop"]

    ccw = np.rot90(moving, k=1)
    cw = np.rot90(moving, k=3)
    assert ccw.shape == reference.shape == cw.shape
    scores = (
        int(np.count_nonzero(reference != ccw)),
        int(np.count_nonzero(reference != cw)),
    )
    assert scores[0] != scores[1], "one quarter-turn must align uniquely better"
    chosen_k = 1 if scores[0] < scores[1] else 3
    aligned = ccw if chosen_k == 1 else cw

    out = reference.copy()
    out[reference != aligned] = TEAL
    return out, scores, chosen_k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely transposed pair, quarter-turn the later region into alignment, and mark its residual differences teal.

    Input structure:
    - A maroon canvas contains at least three mutually separated, fully filled rectangular multicolor pattern regions.
    - Exactly two regions have exchanged height and width; every other region has a different, non-transposed size and is a distractor.
    - The paired regions are near-copies seen a quarter-turn apart, with a few changed cells.

    Rule:
    - Select the unique pair with shapes h x w and w x h, then order that pair by the row-major position of each bbox top-left corner.
    - Rotate the later region 90 degrees counterclockwise and clockwise, choosing the unique direction with fewer cellwise color disagreements against the earlier region.
    - Return a grid in the earlier region's coordinate frame: keep their common color at agreeing cells and paint every disagreement teal.

    Key insight:
    - Transposed bbox dimensions identify the relevant pair, but only the uniquely better quarter-turn reveals the sparse residual; distractors and the opposite rotation rule out direct or orientation-free comparison.
    """
    out, _scores, _chosen_k = _solve_details(grid)
    return out


# ----------------------------- generator -----------------------------


def _shape_conflicts(shape: tuple[int, int], shapes: list[tuple[int, int]]) -> bool:
    """Reject duplicate and accidental transposed-size relations."""
    return any(shape == other or shape == other[::-1] for other in shapes)


def _sample_shapes(rng: random.Random) -> list[tuple[int, int]] | None:
    """Make one intended transpose pair plus both smaller and larger distractors."""
    h, w = rng.sample(range(4, 8), 2)
    pair_area = h * w
    shapes = [(h, w), (w, h)]

    small_options = [
        (a, b)
        for a in range(2, 5)
        for b in range(2, 6)
        if a * b < pair_area and not _shape_conflicts((a, b), shapes)
    ]
    rng.shuffle(small_options)
    if not small_options:
        return None
    shapes.append(small_options[0])

    large_options = [
        (a, b)
        for a in range(6, 10)
        for b in range(6, 10)
        if a * b > pair_area and not _shape_conflicts((a, b), shapes)
    ]
    rng.shuffle(large_options)
    if not large_options:
        return None
    shapes.append(large_options[0])

    for _ in range(rng.randint(0, 2)):
        options = [
            (a, b)
            for a in range(2, 9)
            for b in range(2, 9)
            if not _shape_conflicts((a, b), shapes)
        ]
        if not options:
            break
        shapes.append(rng.choice(options))
    return shapes


def _separated(
    r: int,
    c: int,
    h: int,
    w: int,
    placed: list[tuple[int, int, int, int]],
) -> bool:
    """Require at least one complete maroon cell between rectangular bboxes."""
    for other_r, other_c, other_h, other_w in placed:
        if not (
            r + h < other_r
            or other_r + other_h < r
            or c + w < other_c
            or other_c + other_w < c
        ):
            return False
    return True


def _place_shapes(
    shapes: list[tuple[int, int]],
    height: int,
    width: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    """Randomly pack all rectangles, placing large ones first for reliability."""
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)
    for index in order:
        h, w = shapes[index]
        options = [
            (r, c)
            for r in range(1, height - h)
            for c in range(1, width - w)
            if _separated(r, c, h, w, placed)
        ]
        if not options:
            return None
        r, c = rng.choice(options)
        positions[index] = (r, c)
        placed.append((r, c, h, w))
    assert all(position is not None for position in positions)
    return [position for position in positions if position is not None]


def _pattern(h: int, w: int, colors: list[int], rng: random.Random) -> np.ndarray:
    """Create a readable panel-like pattern from a base and several color blocks."""
    for _ in range(80):
        base = rng.choice(colors)
        patch = np.full((h, w), base, dtype=int)
        for _block in range(rng.randint(4, 8)):
            color = rng.choice([value for value in colors if value != base])
            block_h = rng.randint(1, max(1, h // 2))
            block_w = rng.randint(1, max(1, w // 2))
            r0 = rng.randint(0, h - block_h)
            c0 = rng.randint(0, w - block_w)
            patch[r0 : r0 + block_h, c0 : c0 + block_w] = color
        if len(np.unique(patch)) >= min(3, len(colors)):
            return patch
    raise RuntimeError("failed to make a multicolor rectangular pattern")


def _make_pair(
    reference_shape: tuple[int, int],
    colors: list[int],
    rng: random.Random,
    required_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]] | None:
    """Build a near-copy pair with sparse damage and a decisively unique rotation."""
    h, w = reference_shape
    area = h * w
    for _ in range(120):
        reference = _pattern(h, w, colors, rng)
        chosen_k = required_k
        later = np.rot90(reference, k=(-chosen_k) % 4).copy()

        damage_count = rng.randint(1, min(4, max(1, area // 8)))
        flat_indices = rng.sample(range(area), damage_count)
        for flat_index in flat_indices:
            r, c = divmod(flat_index, later.shape[1])
            old = int(later[r, c])
            later[r, c] = rng.choice([value for value in colors if value != old])

        aligned = np.rot90(later, k=chosen_k)
        opposite = np.rot90(later, k=3 if chosen_k == 1 else 1)
        right_score = int(np.count_nonzero(reference != aligned))
        wrong_score = int(np.count_nonzero(reference != opposite))
        if right_score != damage_count:
            continue
        if wrong_score < max(right_score + 4, int(np.ceil(0.35 * area))):
            continue

        expected = reference.copy()
        expected[reference != aligned] = TEAL
        if len(set(map(int, expected.ravel())) - {TEAL}) < 2:
            continue
        return reference, later, expected, chosen_k, (right_score, wrong_score)
    return None


def _unique_shape_pair(shapes: list[tuple[int, int]]) -> bool:
    pairs = [
        (i, j)
        for i, shape in enumerate(shapes)
        for j in range(i + 1, len(shapes))
        if shapes[j] == shape[::-1]
    ]
    return pairs == [(0, 1)]


def _gen_attempt(rng: random.Random, required_k: int) -> np.ndarray | None:
    shapes = _sample_shapes(rng)
    if shapes is None or not _unique_shape_pair(shapes):
        return None

    height = rng.randint(24, MAX_GRID_SIZE)
    width = rng.randint(24, MAX_GRID_SIZE)
    positions = _place_shapes(shapes, height, width, rng)
    if positions is None:
        return None

    # Construct the shared pattern in the earlier region's exact bbox frame.
    early_index = 0 if positions[0] < positions[1] else 1
    late_index = 1 - early_index
    color_count = rng.randint(3, 5)
    colors = rng.sample(INPUT_COLORS, color_count)
    pair = _make_pair(shapes[early_index], colors, rng, required_k)
    if pair is None:
        return None
    reference, later, expected, chosen_k, pair_scores = pair

    patches: list[np.ndarray | None] = [None] * len(shapes)
    patches[early_index] = reference
    patches[late_index] = later
    for index in range(2, len(shapes)):
        h, w = shapes[index]
        patches[index] = _pattern(h, w, colors, rng)

    canvas = np.full((height, width), MAROON, dtype=int)
    for patch, (r, c) in zip(patches, positions):
        assert patch is not None
        h, w = patch.shape
        canvas[r : r + h, c : c + w] = patch

    # Independent semantic checks: these are properties of the constructed scene,
    # not merely an end-to-end call back into main().
    if int(np.count_nonzero(canvas == MAROON)) <= canvas.size // 2:
        return None
    if any(np.any(patch == MAROON) or np.any(patch == TEAL) for patch in patches if patch is not None):
        return None
    if len(_foreground_components(canvas, MAROON)) != len(shapes):
        return None
    areas = [h * w for h, w in shapes]
    pair_area = areas[0]
    if not (min(areas[2:]) < pair_area < max(areas[2:])):
        return None
    if pair_scores[0] != int(np.count_nonzero(expected == TEAL)):
        return None
    if pair_scores[0] > max(4, int(np.floor(0.20 * expected.size))):
        return None

    try:
        actual, scores, actual_k = _solve_details(canvas)
    except (AssertionError, ValueError):
        return None
    expected_scores = pair_scores if chosen_k == 1 else pair_scores[::-1]
    if scores != expected_scores or actual_k != chosen_k:
        return None
    if actual.shape != shapes[early_index] or not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked instance of the transposed-pair task."""
    global _generation_count
    rng = random.Random()
    # Alternate the two genuine rule branches so any three sequential examples
    # necessarily demonstrate both clockwise and counterclockwise alignment.
    required_k = 1 if _generation_count % 2 == 0 else 3
    for _ in range(400):
        grid = _gen_attempt(rng, required_k)
        if grid is not None:
            _generation_count += 1
            return grid
    raise RuntimeError("failed to generate a valid transposed-pair input")
