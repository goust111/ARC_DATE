import random
from collections import Counter

import numpy as np


BLACK = 0
GREEN = 3
MAROON = 9
INPUT_COLORS = (1, 2, 4, 5, 6, 7, 8)
PERIOD_CHOICES = ((3, 4), (4, 3))
GENERATION_RETRIES = 500
TILE_RETRIES = 600
CORRUPTION_RETRIES = 600


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _spanning_components(
    grid: np.ndarray, ignored_colors: tuple[int, ...] = (BLACK, MAROON)
) -> list[dict]:
    height, width = map(int, grid.shape)
    ignored = set(map(int, ignored_colors))
    out = []
    for color in sorted(set(map(int, np.unique(grid))) - ignored):
        for comp in _components_of_color(grid, color):
            rows = {r for r, _ in comp}
            cols = {c for _, c in comp}
            spans_tb = 0 in rows and (height - 1) in rows
            spans_lr = 0 in cols and (width - 1) in cols
            if spans_tb or spans_lr:
                out.append(
                    {
                        "color": int(color),
                        "coords": comp,
                        "spans_tb": spans_tb,
                        "spans_lr": spans_lr,
                    }
                )
    return out


def _period_consistent(grid: np.ndarray, period_r: int, period_c: int) -> bool:
    seen: dict[tuple[int, int], int] = {}
    height, width = map(int, grid.shape)
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color in (BLACK, MAROON):
                continue
            key = (r % period_r, c % period_c)
            prev = seen.get(key)
            if prev is not None and prev != color:
                return False
            seen[key] = color
    return True


def _infer_periods(grid: np.ndarray) -> tuple[int, int]:
    height, width = map(int, grid.shape)
    for period_r in range(1, height + 1):
        for period_c in range(1, width + 1):
            if _period_consistent(grid, period_r, period_c):
                return period_r, period_c
    raise AssertionError("failed to infer a valid period")


def _reconstruct_tile(grid: np.ndarray, period_r: int, period_c: int) -> np.ndarray:
    tile = np.full((period_r, period_c), -1, dtype=int)
    height, width = map(int, grid.shape)
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color in (BLACK, MAROON):
                continue
            rr, cc = r % period_r, c % period_c
            prev = int(tile[rr, cc])
            if prev != -1:
                assert prev == color
            tile[rr, cc] = color
    assert np.all(tile >= 0), "every tile phase must remain visible somewhere"
    return tile


def _repair_with_tile(grid: np.ndarray, tile: np.ndarray) -> np.ndarray:
    period_r, period_c = map(int, tile.shape)
    height, width = map(int, grid.shape)
    repaired = grid.copy()
    for r in range(height):
        for c in range(width):
            if int(repaired[r, c]) in (BLACK, MAROON):
                repaired[r, c] = int(tile[r % period_r, c % period_c])
    return repaired


def _repair_grid(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    periods = _infer_periods(grid)
    tile = _reconstruct_tile(grid, *periods)
    repaired = _repair_with_tile(grid, tile)
    return repaired, periods, tile


def _expected_output_from_repaired(repaired: np.ndarray) -> tuple[np.ndarray, dict]:
    spans = _spanning_components(repaired, ignored_colors=())
    assert len(spans) == 1, "expected exactly one spanning monochrome component after repair"
    corridor = spans[0]
    out = repaired.copy()
    for r, c in corridor["coords"]:
        out[r, c] = GREEN
    return out, corridor


def main(input_grid: np.ndarray) -> np.ndarray:
    """Repair the periodic wallpaper, then repaint the unique spanning corridor green.

    Input structure:
    - The grid is a repetition of a small colored tile with period roughly 3x4 or 4x3.
    - Some wallpaper cells are replaced by black holes, and the bottom-right cell is a single maroon marker.
    - Every true tile phase still appears at least once in a non-black, non-maroon cell.

    Rule:
    - Ignore black holes and the maroon marker while inferring the smallest row and column periods whose visible phase colors are consistent.
    - Reconstruct the full tile, fill every black or maroon cell with its periodic color, and recover the complete wallpaper.
    - In that repaired wallpaper, find the unique 4-connected monochrome component that spans top-to-bottom or left-to-right, recolor exactly that component green, and leave all other repaired pixels unchanged.

    Key insight:
    - The corruption hides both the repeating template and the corridor, so the corridor only becomes uniquely detectable after periodic completion.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    height, width = map(int, grid.shape)
    marker_coords = np.argwhere(grid == MAROON)
    assert marker_coords.shape[0] == 1, "expected exactly one maroon marker"
    assert tuple(map(int, marker_coords[0])) == (height - 1, width - 1), "marker must be bottom-right"

    repaired, _periods, _tile = _repair_grid(grid)
    out, _corridor = _expected_output_from_repaired(repaired)
    return out


def _random_connected_mask(height: int, width: int, size: int, rng: random.Random) -> np.ndarray:
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    while len(cells) < size:
        r, c = rng.choice(tuple(cells))
        nbrs = list(_neighbors4(r, c, height, width))
        rng.shuffle(nbrs)
        for rr, cc in nbrs:
            cells.add((rr, cc))
            break
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_tile(period_r: int, period_c: int, rng: random.Random) -> tuple[np.ndarray, int] | None:
    palette = rng.sample(list(INPUT_COLORS), rng.choice((3, 4)))
    corridor_color = rng.choice(palette)
    other_colors = [color for color in palette if color != corridor_color]
    min_size = max(4, min(period_r, period_c) + 1)
    max_size = max(min_size, (period_r * period_c) // 2)

    for _ in range(TILE_RETRIES):
        size = rng.randint(min_size, max_size)
        mask = _random_connected_mask(period_r, period_c, size, rng)
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        tile = np.empty((period_r, period_c), dtype=int)
        tile[mask] = corridor_color
        for r in range(period_r):
            for c in range(period_c):
                if not mask[r, c]:
                    tile[r, c] = rng.choice(other_colors)

        for r in range(period_r):
            if int(tile[r, 0]) == int(tile[r, -1]) != corridor_color:
                choices = [color for color in other_colors if color != int(tile[r, 0])]
                tile[r, -1] = rng.choice(choices)
        for c in range(period_c):
            if int(tile[0, c]) == int(tile[-1, c]) != corridor_color:
                choices = [color for color in other_colors if color != int(tile[0, c])]
                tile[-1, c] = rng.choice(choices)

        counts = Counter(map(int, tile.ravel()))
        if len(counts) < 3:
            continue
        if min(counts.values()) < 2:
            continue
        return tile, int(corridor_color)
    return None


def _phase_occurrences(height: int, width: int, period_r: int, period_c: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    phases: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(height):
        for c in range(width):
            phases.setdefault((r % period_r, c % period_c), []).append((r, c))
    return phases


def _hole_bounds(cell_count: int) -> tuple[int, int]:
    hole_min = (15 * cell_count + 99) // 100
    hole_max = (25 * cell_count) // 100
    return hole_min, hole_max


def _all_phases_visible(grid: np.ndarray, period_r: int, period_c: int) -> bool:
    seen = set()
    height, width = map(int, grid.shape)
    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) not in (BLACK, MAROON):
                seen.add((r % period_r, c % period_c))
    return len(seen) == period_r * period_c


def _semantic_instance_ok(
    inp: np.ndarray,
    repaired_full: np.ndarray,
    corridor_color: int,
    corridor_coords: set[tuple[int, int]],
    period_r: int,
    period_c: int,
) -> bool:
    height, width = map(int, inp.shape)
    if int(inp[height - 1, width - 1]) != MAROON:
        return False

    hole_min, hole_max = _hole_bounds(height * width - 1)
    hole_count = int(np.count_nonzero(inp == BLACK))
    if not (hole_min <= hole_count <= hole_max):
        return False
    if not _all_phases_visible(inp, period_r, period_c):
        return False
    if _infer_periods(inp) != (period_r, period_c):
        return False
    if _spanning_components(inp):
        return False

    repaired, inferred_periods, tile = _repair_grid(inp)
    if inferred_periods != (period_r, period_c):
        return False
    if not np.array_equal(repaired, repaired_full):
        return False
    if int(tile[(height - 1) % period_r, (width - 1) % period_c]) == MAROON:
        return False

    spans = _spanning_components(repaired, ignored_colors=())
    if len(spans) != 1:
        return False
    span = spans[0]
    if int(span["color"]) != int(corridor_color):
        return False
    if set(span["coords"]) != corridor_coords:
        return False

    expected = repaired_full.copy()
    for r, c in corridor_coords:
        expected[r, c] = GREEN
    return np.array_equal(main(inp), expected)


def _make_corrupted_input(
    repaired_full: np.ndarray,
    corridor_coords: set[tuple[int, int]],
    period_r: int,
    period_c: int,
    rng: random.Random,
) -> np.ndarray | None:
    height, width = map(int, repaired_full.shape)
    marker = (height - 1, width - 1)
    phases = _phase_occurrences(height, width, period_r, period_c)
    hole_min, hole_max = _hole_bounds(height * width - 1)

    for _ in range(CORRUPTION_RETRIES):
        protected = {marker}
        for cells in phases.values():
            choices = [cell for cell in cells if cell != marker]
            if not choices:
                return None
            protected.add(rng.choice(choices))

        hole_count = rng.randint(hole_min, hole_max)
        corridor_cells = [cell for cell in corridor_coords if cell not in protected and cell != marker]
        non_corridor_cells = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in corridor_coords and (r, c) not in protected and (r, c) != marker
        ]
        if not corridor_cells or not non_corridor_cells:
            continue

        min_corridor_holes = max(1, hole_count // 3)
        max_corridor_holes = min(len(corridor_cells), max(1, (7 * hole_count) // 10))
        if max_corridor_holes < min_corridor_holes:
            continue
        corridor_holes = rng.randint(min_corridor_holes, max_corridor_holes)
        other_holes = hole_count - corridor_holes
        if other_holes <= 0 or len(non_corridor_cells) < other_holes:
            continue

        chosen_holes = set(rng.sample(corridor_cells, corridor_holes))
        chosen_holes.update(rng.sample(non_corridor_cells, other_holes))

        inp = repaired_full.copy()
        for r, c in chosen_holes:
            inp[r, c] = BLACK
        inp[marker] = MAROON

        if _spanning_components(inp):
            continue
        if not _all_phases_visible(inp, period_r, period_c):
            continue
        return inp
    return None


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    period_r, period_c = rng.choice(PERIOD_CHOICES)
    reps_r = rng.randint(4, 7)
    reps_c = rng.randint(4, 7)
    height, width = period_r * reps_r, period_c * reps_c

    for _ in range(TILE_RETRIES):
        tile_and_color = _sample_tile(period_r, period_c, rng)
        if tile_and_color is None:
            continue
        tile, corridor_color = tile_and_color
        repaired_full = np.tile(tile, (reps_r, reps_c))

        if _infer_periods(repaired_full) != (period_r, period_c):
            continue

        spans = _spanning_components(repaired_full, ignored_colors=())
        if len(spans) != 1 or int(spans[0]["color"]) != int(corridor_color):
            continue

        corridor_coords = set(spans[0]["coords"])
        corridor_ratio = len(corridor_coords) / float(height * width)
        if not (0.18 <= corridor_ratio <= 0.55):
            continue

        inp = _make_corrupted_input(repaired_full, corridor_coords, period_r, period_c, rng)
        if inp is None:
            continue
        if _semantic_instance_ok(inp, repaired_full, corridor_color, corridor_coords, period_r, period_c):
            return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
