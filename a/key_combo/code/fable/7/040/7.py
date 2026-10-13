from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SEEN_GENERATED_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()
_MAX_GENERATION_ATTEMPTS = 500


def _background(grid: np.ndarray) -> int:
    """Return the uniquely most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "expected a unique most-common background color"
    return int(winners[0])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    """Find 4-connected monochrome foreground components in (row, col) coordinates."""
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for start_r in range(height):
        for start_c in range(width):
            color = int(grid[start_r, start_c])
            if color == bg or seen[start_r, start_c]:
                continue

            queue = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the purified largest component in blocks selected by that same component.

    Input structure:
    - The background is the unique most common color in a small 2D grid.
    - Foreground pixels form several differently colored 4-connected monochrome
      components, with exactly one component having the greatest pixel area.

    Rule:
    - Replace every foreground pixel outside the unique largest component with
      background to form a full-input-size purified pattern.
    - Regard every input cell as one output block of the input grid's size. Put
      the purified pattern in exactly the blocks corresponding to pixels of the
      largest component; fill every other block entirely with background.

    Key insight:
    - The unique largest component is self-referential: its exact pixel mask
      selects the stamp locations, and the same component alone supplies the
      visible content of every stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    height, width = map(int, g.shape)
    assert height * height <= 30 and width * width <= 30, "expanded output exceeds ARC bounds"

    bg = _background(g)
    components = _monochrome_components(g, bg)
    assert len(components) >= 2, "expected one target and at least one distractor"

    # The described objects have mutually distinct colors, so no foreground
    # color may be split into multiple monochrome components.
    foreground_colors = set(map(int, np.unique(g))) - {bg}
    component_colors = [int(component["color"]) for component in components]
    assert len(component_colors) == len(foreground_colors), "each foreground color must form one component"
    assert len(set(component_colors)) == len(component_colors), "component colors must be distinct"

    largest_area = max(int(component["area"]) for component in components)
    largest = [component for component in components if int(component["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest component"
    target = largest[0]

    purified = np.full_like(g, bg)
    target_coords = target["coords"]
    purified[target_coords[:, 0], target_coords[:, 1]] = int(target["color"])

    out = np.full((height * height, width * width), bg, dtype=int)
    for r, c in target_coords:
        rr, cc = int(r), int(c)
        out[rr * height : (rr + 1) * height, cc * width : (cc + 1) * width] = purified
    return out


def _cardinally_separated(occupied: set[tuple[int, int]], cell: tuple[int, int]) -> bool:
    r, c = cell
    return cell not in occupied and all((r + dr, c + dc) not in occupied for dr, dc in _NEIGHBORS4)


def _grow_component(
    shape: tuple[int, int],
    area: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    """Grow a random component without touching earlier components cardinally."""
    height, width = shape
    allowed = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if _cardinally_separated(occupied, (r, c))
    }
    if len(allowed) < area:
        return None

    for _ in range(40):
        component = {random.choice(tuple(allowed))}
        while len(component) < area:
            frontier = {
                (r + dr, c + dc)
                for r, c in component
                for dr, dc in _NEIGHBORS4
                if (r + dr, c + dc) in allowed and (r + dr, c + dc) not in component
            }
            if not frontier:
                break
            component.add(random.choice(tuple(frontier)))
        if len(component) == area:
            return component
    return None


def _render_oracle(grid: np.ndarray, bg: int, target: set[tuple[int, int]]) -> np.ndarray:
    """Independent literal rendering of the two-level stamp rule."""
    height, width = grid.shape
    purified = np.full_like(grid, bg)
    for r, c in target:
        purified[r, c] = grid[r, c]

    expected = np.full((height * height, width * width), bg, dtype=int)
    for block_r in range(height):
        for block_c in range(width):
            if (block_r, block_c) in target:
                expected[
                    block_r * height : (block_r + 1) * height,
                    block_c * width : (block_c + 1) * width,
                ] = purified
    return expected


def _validate_generated(
    grid: np.ndarray,
    bg: int,
    intended_components: list[set[tuple[int, int]]],
) -> bool:
    """Check puzzle semantics and explicit counterexamples, not only closure."""
    try:
        if _background(grid) != bg:
            return False
        found = _monochrome_components(grid, bg)
        if len(found) != len(intended_components) or len(found) < 3:
            return False
        if len({int(component["color"]) for component in found}) != len(found):
            return False

        sizes = sorted((int(component["area"]) for component in found), reverse=True)
        if sizes[0] == sizes[1] or sum(sizes[1:]) < 2:
            return False
        target_component = max(found, key=lambda component: int(component["area"]))
        target = {tuple(map(int, rc)) for rc in target_component["coords"]}
        if target != intended_components[0]:
            return False

        # A nonrectangular target refutes replacing the pixel mask by its bbox.
        r0, c0, r1, c1 = target_component["bbox"]
        if (r1 - r0 + 1) * (c1 - c0 + 1) == len(target):
            return False
        if r0 == 0 and c0 == 0:
            return False

        expected = _render_oracle(grid, bg, target)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        height, width = grid.shape
        purified = np.full_like(grid, bg)
        for r, c in target:
            purified[r, c] = grid[r, c]

        # Wrong shortcut 1: stamp the unpurified input at the right blocks.
        unpurified = np.full_like(expected, bg)
        for r, c in target:
            unpurified[r * height : (r + 1) * height, c * width : (c + 1) * width] = grid
        if np.array_equal(unpurified, expected):
            return False

        # Wrong shortcut 2: use every foreground pixel as a block selector.
        all_foreground = np.full_like(expected, bg)
        for r, c in np.argwhere(grid != bg):
            all_foreground[r * height : (r + 1) * height, c * width : (c + 1) * width] = purified
        if np.array_equal(all_foreground, expected):
            return False

        # Every selected block and every rejected block is checked explicitly.
        for r in range(height):
            for c in range(width):
                block = actual[r * height : (r + 1) * height, c * width : (c + 1) * width]
                wanted = purified if (r, c) in target else np.full_like(grid, bg)
                if not np.array_equal(block, wanted):
                    return False
    except (AssertionError, ValueError):
        return False
    return True


def _generation_attempt() -> np.ndarray | None:
    shape = random.choice(((4, 4), (4, 5), (5, 4), (5, 5)))
    cell_count = shape[0] * shape[1]
    object_count = random.choice((3, 4)) if cell_count == 25 else 3

    min_distractor_total = object_count - 1
    max_largest = min(6, (cell_count - min_distractor_total - 1) // 2)
    largest_area = random.randint(4, max_largest)
    distractor_budget = cell_count - 2 * largest_area - 1
    if distractor_budget < min_distractor_total:
        return None

    distractor_areas = [1] * (object_count - 1)
    spare = distractor_budget - sum(distractor_areas)
    order = list(range(len(distractor_areas)))
    random.shuffle(order)
    for index in order:
        extra = random.randint(0, min(2, spare, largest_area - 2))
        distractor_areas[index] += extra
        spare -= extra

    occupied: set[tuple[int, int]] = set()
    largest = _grow_component(shape, largest_area, occupied)
    if largest is None:
        return None
    rows = [r for r, _ in largest]
    cols = [c for _, c in largest]
    bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    if bbox_area == largest_area or (min(rows) == 0 and min(cols) == 0):
        return None
    occupied.update(largest)

    components = [largest]
    for area in distractor_areas:
        component = _grow_component(shape, area, occupied)
        if component is None:
            return None
        components.append(component)
        occupied.update(component)

    bg = random.randrange(10)
    colors = random.sample([color for color in range(10) if color != bg], object_count)
    grid = np.full(shape, bg, dtype=int)
    for color, component in zip(colors, components):
        for r, c in component:
            grid[r, c] = color

    if not _validate_generated(grid, bg, components):
        return None

    out = main(grid)
    signature = (tuple(map(int, out.shape)), out.tobytes())
    if signature in _SEEN_GENERATED_OUTPUTS:
        return None
    _SEEN_GENERATED_OUTPUTS.add(signature)
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, semantically validated self-referential stamp input."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential stamp input")
