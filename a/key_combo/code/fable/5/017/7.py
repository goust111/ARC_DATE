from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEUTRAL = 5
PALETTE = tuple(range(10))


def _neighbors4(r: int, c: int):
    """Yield neighbors in the canonical ARC (row, column) coordinate system."""
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _most_common(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neutral_color(grid: np.ndarray, background: int) -> int:
    """The neutral mask is the unique most frequent foreground color."""
    counts = Counter(map(int, grid.ravel()))
    counts.pop(int(background), None)
    assert counts, "expected neutral mask pixels"
    largest = max(counts.values())
    candidates = [color for color, count in counts.items() if count == largest]
    assert len(candidates) == 1, "neutral color must be the unique most frequent foreground color"
    return int(candidates[0])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    """Return 4-connected components of a boolean mask."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[list[Coord]] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        queue = deque([(r0, c0)])
        component: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(component)
    return result


def _distance_field(component: set[Coord], sources: set[Coord]) -> dict[Coord, int]:
    """Multi-source shortest-path distances restricted to one neutral component."""
    assert sources and sources <= component
    distance = {cell: 0 for cell in sources}
    queue = deque(sources)
    while queue:
        r, c = queue.popleft()
        next_distance = distance[(r, c)] + 1
        for cell in _neighbors4(r, c):
            if cell in component and cell not in distance:
                distance[cell] = next_distance
                queue.append(cell)
    assert len(distance) == len(component), "neutral component must be 4-connected"
    return distance


def _touching_sources(
    grid: np.ndarray,
    component: set[Coord],
    marker_mask: np.ndarray,
) -> dict[int, set[Coord]]:
    height, width = grid.shape
    sources: dict[int, set[Coord]] = {}
    for r, c in component:
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < height and 0 <= cc < width and marker_mask[rr, cc]:
                sources.setdefault(int(grid[rr, cc]), set()).add((r, c))
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split every touched neutral block among its marker colors by within-block geodesic distance.

    Input structure:
    - The background is the most common color, and one uniquely frequent foreground color forms neutral 4-connected blocks.
    - Other foreground cells are colored marker pixels that 4-touch the outside edge of some neutral blocks.
    - A neutral block may touch no marker color, one marker color, or several marker colors.

    Rule:
    - Erase every marker pixel to the background.
    - Leave an untouched neutral block unchanged, and recolor a block touched by one marker color completely with that color.
    - For a block touched by several colors, measure shortest 4-connected paths only through that block from every cell to each color's complete set of contact cells; use the uniquely nearest color.

    Key insight:
    - Distance is a geodesic inside each bent neutral mask, so straight-line proximity across a gap cannot shortcut around the block's turns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    background = _most_common(g)
    neutral = _neutral_color(g, background)
    marker_mask = (g != background) & (g != neutral)

    neutral_components = _components(g == neutral)
    assert neutral_components, "expected at least one neutral component"

    out = g.copy()
    out[marker_mask] = background

    for cells in neutral_components:
        component = set(cells)
        sources = _touching_sources(g, component, marker_mask)
        if not sources:
            continue
        if len(sources) == 1:
            color = next(iter(sources))
            for r, c in component:
                out[r, c] = color
            continue

        fields = {color: _distance_field(component, contact) for color, contact in sources.items()}
        for r, c in component:
            distances = {color: field[(r, c)] for color, field in fields.items()}
            nearest = min(distances.values())
            winners = [color for color, distance in distances.items() if distance == nearest]
            assert len(winners) == 1, "multi-marker components must contain no equidistant cells"
            out[r, c] = winners[0]

    return out


# ------------------------------ generator ------------------------------

GRID_SIDE_RANGE = (20, 31)  # random.randrange upper bound; ARC maximum remains 30
OBJECT_COUNT_RANGE = (3, 6)
PLACEMENT_GAP = 2
GENERATION_ATTEMPTS = 300


def _transform_pair(inp: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    turns = random.randrange(4)
    inp = np.rot90(inp, turns)
    expected = np.rot90(expected, turns)
    if random.random() < 0.5:
        inp = np.fliplr(inp)
        expected = np.fliplr(expected)
    return inp.copy(), expected.copy()


def _make_geodesic_patch(
    background: int,
    neutral: int,
    color_a: int,
    color_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a thin U path with a certified geodesic-versus-straight-line disagreement."""
    choices: list[tuple[int, int, int, int, int, int]] = []
    for height in range(8, 12):
        for width in range(4, 7):
            for a in range(0, 3):
                for b in range(a + 1, height - 1):
                    index_a = a
                    index_b = 2 * height + width - 3 - b
                    gap = index_b - index_a
                    direct_a = b - a
                    direct_b = width - 1
                    geodesic_b = index_b - b
                    if gap % 2 == 1 and direct_b < direct_a < geodesic_b:
                        choices.append((height, width, a, b, index_a, index_b))
    assert choices
    height, width, a, b, index_a, index_b = random.choice(choices)

    inp = np.full((height + 2, width + 2), background, dtype=int)
    expected = inp.copy()
    path: list[Coord] = []
    path.extend((r, 1) for r in range(1, height + 1))
    path.extend((height, c) for c in range(2, width + 1))
    path.extend((r, width) for r in range(height - 1, 0, -1))
    assert path[index_a] == (a + 1, 1)
    assert path[index_b] == (b + 1, width)

    for index, (r, c) in enumerate(path):
        inp[r, c] = neutral
        da, db = abs(index - index_a), abs(index - index_b)
        assert da != db
        expected[r, c] = color_a if da < db else color_b

    marker_a = (a + 1, 0)
    marker_b = (b + 1, width + 1)
    inp[marker_a] = color_a
    inp[marker_b] = color_b
    # Expected marker locations remain background, encoding the mandatory erasure step.
    assert expected[b + 1, 1] == color_a
    assert (width - 1) < (b - a), "right marker must look closer across the U-shaped gap"
    return _transform_pair(inp, expected)


def _random_polyomino() -> set[Coord]:
    """Grow a small connected shape with randomized extent and boundary."""
    box_h = random.randint(2, 4)
    box_w = random.randint(2, 4)
    target = random.randint(4, min(10, box_h * box_w))
    cells = {(random.randrange(box_h), random.randrange(box_w))}
    while len(cells) < target:
        frontier = {
            (rr, cc)
            for r, c in cells
            for rr, cc in _neighbors4(r, c)
            if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells
        }
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _make_simple_patch(
    background: int,
    neutral: int,
    kind: str,
    marker_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    cells = _random_polyomino()
    shape_h = max(r for r, _ in cells) + 1
    shape_w = max(c for _, c in cells) + 1
    shifted = {(r + 1, c + 1) for r, c in cells}
    inp = np.full((shape_h + 2, shape_w + 2), background, dtype=int)
    expected = inp.copy()
    for r, c in shifted:
        inp[r, c] = neutral
        expected[r, c] = neutral if kind == "untouched" else marker_color

    if kind == "single":
        candidates = []
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                if (r, c) in shifted:
                    continue
                contacts = sum((rr, cc) in shifted for rr, cc in _neighbors4(r, c))
                if contacts == 1:
                    candidates.append((r, c))
        assert candidates
        first = random.choice(candidates)
        markers = [first]
        if random.random() < 0.5:
            second_choices = [
                cell for cell in candidates
                if abs(cell[0] - first[0]) + abs(cell[1] - first[1]) > 1
            ]
            if second_choices:
                markers.append(random.choice(second_choices))
        for marker in markers:
            inp[marker] = marker_color
        # The expected patch deliberately keeps these marker locations at background.
    else:
        assert kind == "untouched"
    return _transform_pair(inp, expected)


def _placement_candidates(
    canvas_shape: tuple[int, int],
    patch: np.ndarray,
    background: int,
    forbidden: np.ndarray,
) -> list[Coord]:
    height, width = canvas_shape
    patch_h, patch_w = patch.shape
    occupied = np.argwhere(patch != background)
    candidates: list[Coord] = []
    for r0 in range(1, height - patch_h):
        for c0 in range(1, width - patch_w):
            if all(not forbidden[r0 + int(r), c0 + int(c)] for r, c in occupied):
                candidates.append((r0, c0))
    return candidates


def _place_patch(
    inp: np.ndarray,
    expected: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    background: int,
    forbidden: np.ndarray,
) -> bool:
    candidates = _placement_candidates(inp.shape, patch_in, background, forbidden)
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    patch_h, patch_w = patch_in.shape
    input_mask = patch_in != background
    output_mask = patch_out != background
    inp[r0 : r0 + patch_h, c0 : c0 + patch_w][input_mask] = patch_in[input_mask]
    expected[r0 : r0 + patch_h, c0 : c0 + patch_w][output_mask] = patch_out[output_mask]

    for pr, pc in np.argwhere(input_mask):
        r, c = r0 + int(pr), c0 + int(pc)
        rr0, rr1 = max(0, r - PLACEMENT_GAP), min(inp.shape[0], r + PLACEMENT_GAP + 1)
        cc0, cc1 = max(0, c - PLACEMENT_GAP), min(inp.shape[1], c + PLACEMENT_GAP + 1)
        forbidden[rr0:rr1, cc0:cc1] = True
    return True


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    """Check the stated semantics, branch coverage, uniqueness, and anti-shortcut witness."""
    try:
        background = _most_common(inp)
        neutral = _neutral_color(inp, background)
        if neutral != NEUTRAL:
            return False
        marker_mask = (inp != background) & (inp != neutral)
        components = _components(inp == neutral)
        if len(components) < 3 or not np.any(marker_mask):
            return False

        labels = np.zeros(inp.shape, dtype=int)
        for label, cells in enumerate(components, start=1):
            for r, c in cells:
                labels[r, c] = label

        # Every visible marker has one clear role: it touches one component at one cell.
        for r0, c0 in np.argwhere(marker_mask):
            r0, c0 = int(r0), int(c0)
            touched = {
                int(labels[r, c])
                for r, c in _neighbors4(r0, c0)
                if 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] and labels[r, c] > 0
            }
            contact_count = sum(
                0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] and labels[r, c] > 0
                for r, c in _neighbors4(r0, c0)
            )
            if len(touched) != 1 or contact_count != 1:
                return False

        branches: set[int] = set()
        straight_line_disagrees = False
        for cells in components:
            component = set(cells)
            sources = _touching_sources(inp, component, marker_mask)
            branch = min(len(sources), 2)
            branches.add(branch)
            if not sources:
                if any(expected[r, c] != neutral for r, c in component):
                    return False
                continue
            if len(sources) == 1:
                color = next(iter(sources))
                if any(expected[r, c] != color for r, c in component):
                    return False
                continue

            fields = {color: _distance_field(component, contact) for color, contact in sources.items()}
            used_colors: set[int] = set()
            for cell in component:
                geodesic = {color: field[cell] for color, field in fields.items()}
                best = min(geodesic.values())
                winners = [color for color, distance in geodesic.items() if distance == best]
                if len(winners) != 1 or int(expected[cell]) != winners[0]:
                    return False
                used_colors.add(winners[0])

                direct = {
                    color: min((cell[0] - r) ** 2 + (cell[1] - c) ** 2 for r, c in contacts)
                    for color, contacts in sources.items()
                }
                direct_best = min(direct.values())
                direct_winners = [color for color, distance in direct.items() if distance == direct_best]
                if len(direct_winners) == 1 and direct_winners[0] != winners[0]:
                    straight_line_disagrees = True
            # Every marker color owns cells, so no marker or distance field is decorative.
            if used_colors != set(sources):
                return False

        if branches != {0, 1, 2} or not straight_line_disagrees:
            return False
        actual = main(inp)
        if not np.array_equal(actual, expected) or np.array_equal(actual, inp):
            return False
        if not np.all(actual[marker_mask] == background):
            return False
        return True
    except (AssertionError, ValueError, IndexError):
        return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randrange(*GRID_SIDE_RANGE)
    width = random.randrange(*GRID_SIDE_RANGE)
    background = random.choice([color for color in PALETTE if color != NEUTRAL])
    marker_palette = [color for color in PALETTE if color not in (background, NEUTRAL)]
    random.shuffle(marker_palette)

    inp = np.full((height, width), background, dtype=int)
    expected = inp.copy()
    forbidden = np.zeros(inp.shape, dtype=bool)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    patches.append(_make_geodesic_patch(background, NEUTRAL, marker_palette[0], marker_palette[1]))
    patches.append(_make_simple_patch(background, NEUTRAL, "single", random.choice(marker_palette)))
    patches.append(_make_simple_patch(background, NEUTRAL, "untouched", marker_palette[0]))
    for _ in range(random.randrange(*OBJECT_COUNT_RANGE) - 3):
        kind = random.choice(("single", "untouched"))
        patches.append(_make_simple_patch(background, NEUTRAL, kind, random.choice(marker_palette)))

    # The large bent component goes first; the smaller components then fill random free regions.
    for patch_in, patch_out in patches:
        if not _place_patch(inp, expected, patch_in, patch_out, background, forbidden):
            return None

    color_counts = Counter(map(int, inp.ravel()))
    if color_counts[background] <= max(count for color, count in color_counts.items() if color != background):
        return None
    foreground_counts = {color: count for color, count in color_counts.items() if color != background}
    if foreground_counts.get(NEUTRAL, 0) <= max(
        (count for color, count in foreground_counts.items() if color != NEUTRAL), default=0
    ):
        return None
    if not _semantic_check(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance containing all three rule branches."""
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid geodesic-partition puzzle")
