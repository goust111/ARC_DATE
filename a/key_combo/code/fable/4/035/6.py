from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
_GENERATION_RETRIES = 400
_generation_index = 0
_generated_output_signatures: set[tuple[tuple[int, int], bytes]] = set()


@dataclass(frozen=True)
class Basin:
    """A detected upright U-basin, using inclusive ARC (row, column) bounds."""

    top: int
    bottom: int
    left: int
    right: int
    palette: tuple[int, ...]
    marker_side: int

    @property
    def width(self) -> int:
        return self.right - self.left - 1

    @property
    def capacity_rows(self) -> int:
        return self.bottom - self.top


def _terrain_mask(grid: np.ndarray) -> np.ndarray:
    """Terrain includes gray walls and colored marker cells, but not blue water."""
    return (grid != BLACK) & (grid != BLUE)


def _gray_runs(row: np.ndarray, minimum: int = 5) -> list[tuple[int, int]]:
    """Return inclusive bounds of long gray runs in one row."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for c in range(len(row) + 1):
        is_gray = c < len(row) and int(row[c]) == GREY
        if is_gray and start is None:
            start = c
        elif not is_gray and start is not None:
            if c - start >= minimum:
                runs.append((start, c - 1))
            start = None
    return runs


def _detect_basins(grid: np.ndarray) -> list[Basin]:
    """Find separated U shapes from their gray floors, then read wall markers."""
    h, w = grid.shape
    terrain = _terrain_mask(grid)
    basins: list[Basin] = []

    for bottom in range(h):
        for left, right in _gray_runs(grid[bottom]):
            # A floor has open background below it (or lies at the grid edge).
            if bottom + 1 < h and np.any(grid[bottom + 1, left : right + 1] != BLACK):
                continue

            top = bottom - 1
            while top >= 0 and terrain[top, left] and terrain[top, right]:
                top -= 1
            top += 1
            if bottom - top < 4:
                continue

            # The basin's interior is unobstructed; its only solid cells are the
            # two walls and floor.  This separates floors from incidental bars.
            interior = grid[top:bottom, left + 1 : right]
            if interior.size == 0 or np.any((interior != BLACK) & (interior != BLUE)):
                continue

            left_markers = [
                (r, int(grid[r, left]))
                for r in range(top, bottom)
                if int(grid[r, left]) not in (GREY, BLACK, BLUE)
            ]
            right_markers = [
                (r, int(grid[r, right]))
                for r in range(top, bottom)
                if int(grid[r, right]) not in (GREY, BLACK, BLUE)
            ]
            if bool(left_markers) == bool(right_markers):
                continue

            marker_side = left if left_markers else right
            markers = left_markers if left_markers else right_markers
            markers.sort(key=lambda item: item[0], reverse=True)  # bottom upward
            palette = tuple(color for _, color in markers)
            if len(palette) < 2 or len(set(palette)) != len(palette):
                continue

            basins.append(Basin(top, bottom, left, right, palette, marker_side))

    basins.sort(key=lambda basin: (basin.left, basin.bottom))
    assert basins, "expected at least one marked U-basin"
    for a, b in zip(basins, basins[1:]):
        assert a.right < b.left, "basins must be horizontally separated"
    return basins


def _trace_drop(
    start: tuple[int, int], terrain: np.ndarray
) -> tuple[tuple[int, int], int]:
    """Drop through static terrain, sliding only when exactly one diagonal is free."""
    h, w = terrain.shape
    r, c = start
    diagonal_steps = 0
    for _ in range(4 * h * w):
        if r + 1 >= h:
            raise AssertionError(f"water at {start} escapes below the grid")
        if not terrain[r + 1, c]:
            r += 1
            continue

        candidates = [
            (r + 1, nc)
            for nc in (c - 1, c + 1)
            if 0 <= nc < w and not terrain[r + 1, nc]
        ]
        if len(candidates) == 1:
            r, c = candidates[0]
            diagonal_steps += 1
            continue
        if len(candidates) == 0:
            return (r, c), diagonal_steps
        raise AssertionError(f"water at {start} has two possible diagonal slides")
    raise AssertionError("water trajectory did not terminate")


def _containing_basin(stop: tuple[int, int], basins: list[Basin]) -> int:
    r, c = stop
    matches = [
        i
        for i, basin in enumerate(basins)
        if basin.top <= r < basin.bottom and basin.left < c < basin.right
    ]
    assert len(matches) == 1, f"settled water at {stop} must belong to one basin"
    return matches[0]


def _water_assignments(
    grid: np.ndarray, basins: list[Basin]
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    terrain = _terrain_mask(grid)
    assignments: list[list[tuple[int, int]]] = [[] for _ in basins]
    slide_counts: dict[tuple[int, int], int] = {}
    for rr, cc in np.argwhere(grid == BLUE):
        source = (int(rr), int(cc))
        stop, diagonal_steps = _trace_drop(source, terrain)
        basin_i = _containing_basin(stop, basins)
        assignments[basin_i].append(source)
        slide_counts[source] = diagonal_steps
    assert all(assignments), "every basin must receive water"
    return assignments, slide_counts


def main(grid: np.ndarray) -> np.ndarray:
    """Drop blue water into marked basins and recolor its depth bands with each wall palette.

    Input structure:
    - Black is the background; gray terrain forms one or more separated, upward-open U-shaped basins.
    - Blue pixels are suspended water drops above the basins, and non-gray colored pixels embedded in exactly
      one side wall of each basin are distinct palette markers.
    - A drop meeting terrain falls through the only free lower diagonal when one exists; generated inputs never
      offer two diagonal choices, so every drop reaches exactly one basin.

    Rule:
    - Trace every blue pixel vertically through the original static terrain, taking a diagonal step precisely
      when the cell below is blocked and exactly one lower diagonal is free, to determine its destination basin.
    - Pool the water assigned to each basin into complete rows from the floor upward.  Read that basin's wall
      markers by row from bottom to top, and color depth row d with palette[d modulo palette length].
    - Erase the suspended blue pixels while leaving all gray terrain and colored wall markers unchanged.

    Key insight:
    - Falling and forced diagonal slides decide which basin receives each temporary blue drop; only afterward do
      the fixed wall markers become a cyclic bottom-up palette for the pooled water's depth bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"
    assert np.any(g == BLUE), "expected suspended blue water"

    basins = _detect_basins(g)
    assignments, _ = _water_assignments(g, basins)

    out = g.copy()
    out[out == BLUE] = BLACK
    for basin, drops in zip(basins, assignments):
        count = len(drops)
        assert count % basin.width == 0, "water must fill complete depth rows"
        depth_rows = count // basin.width
        assert 0 < depth_rows <= basin.capacity_rows, "water exceeds basin capacity"
        for depth in range(depth_rows):
            r = basin.bottom - 1 - depth
            out[r, basin.left + 1 : basin.right] = basin.palette[depth % len(basin.palette)]
    return out


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    """Independent construction oracle using the generator's latent basin specs."""
    expected = grid.copy()
    expected[expected == BLUE] = BLACK
    for spec in specs:
        for depth in range(spec["layers"]):
            r = spec["bottom"] - 1 - depth
            expected[r, spec["left"] + 1 : spec["right"]] = spec["palette"][
                depth % len(spec["palette"])
            ]
    return expected


def _semantic_check(grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    """Check rule semantics, anti-shortcut witnesses, and marker dependence."""
    try:
        actual = main(grid)
        if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
            return False

        basins = _detect_basins(grid)
        assignments, slide_counts = _water_assignments(grid, basins)
        if len(basins) != len(specs):
            return False

        # Every basin has a forced-slide witness.  Therefore merely counting
        # blue pixels vertically above the open interior gives the wrong amount.
        for basin, drops, spec in zip(basins, assignments, specs):
            if len(drops) != basin.width * spec["layers"]:
                return False
            if not any(slide_counts[p] > 0 for p in drops):
                return False
            vertical_only = sum(basin.left < c < basin.right for _, c in drops)
            if vertical_only == len(drops):
                return False
            if spec["layers"] <= len(basin.palette):
                return False  # cyclic repetition must be visible

        # Terrain and marker pixels are invariant, while all output water colors
        # are taken from the correct local palette.
        terrain = _terrain_mask(grid)
        if not np.array_equal(actual[terrain], grid[terrain]):
            return False
        if np.any(actual == BLUE):
            return False

        # Swap the first two markers of one basin.  Falling destinations must be
        # unchanged, but its depth colors must change: the palette step is real.
        witness = basins[0]
        marker_rows = [
            r
            for r in range(witness.bottom - 1, witness.top - 1, -1)
            if int(grid[r, witness.marker_side]) not in (GREY, BLACK, BLUE)
        ]
        if len(marker_rows) < 2:
            return False
        altered = grid.copy()
        r0, r1 = marker_rows[:2]
        altered[r0, witness.marker_side], altered[r1, witness.marker_side] = (
            altered[r1, witness.marker_side],
            altered[r0, witness.marker_side],
        )
        changed = main(altered)
        water_region = np.zeros(grid.shape, dtype=bool)
        water_region[witness.top : witness.bottom, witness.left + 1 : witness.right] = True
        if np.array_equal(changed[water_region], actual[water_region]):
            return False
        altered_terrain = _terrain_mask(altered)
        if not np.array_equal(changed[altered_terrain], altered[altered_terrain]):
            return False
        stable_nonterrain = ~water_region & ~altered_terrain
        if not np.array_equal(changed[stable_nonterrain], actual[stable_nonterrain]):
            return False
        return True
    except (AssertionError, IndexError, ValueError):
        return False


def _sample_unique_palettes(
    rng: random.Random, basin_count: int, lengths: list[int]
) -> list[tuple[int, ...]]:
    palettes: list[tuple[int, ...]] = []
    for length in lengths:
        choices = [
            tuple(rng.sample(MARKER_COLORS, length))
            for _ in range(40)
        ]
        palette = next((p for p in choices if p not in palettes), None)
        if palette is None:
            raise RuntimeError("could not choose distinct palettes")
        palettes.append(palette)
    assert len(palettes) == basin_count
    return palettes


def _generate_attempt(basin_count: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    h = rng.randint(23, 30)
    widths = [rng.randint(3, 5 if basin_count == 2 else 4) for _ in range(basin_count)]
    gaps = [rng.randint(3, 5 if basin_count == 2 else 4) for _ in range(basin_count - 1)]
    left_margin = rng.randint(2, 3)
    right_margin = rng.randint(2, 3)
    w = left_margin + right_margin + sum(x + 2 for x in widths) + sum(gaps)
    if w > 30:
        return None

    # Distinct quantities rule out a fixed-depth shortcut.  Both palette-length
    # branches occur in every generated grid with at least two basins.
    layers = rng.sample(range(4, 8), basin_count)
    palette_lengths = [2, 3] + [rng.choice((2, 3)) for _ in range(basin_count - 2)]
    rng.shuffle(palette_lengths)
    if any(layer <= plen for layer, plen in zip(layers, palette_lengths)):
        return None
    palettes = _sample_unique_palettes(rng, basin_count, palette_lengths)

    grid = np.full((h, w), BLACK, dtype=int)
    specs: list[dict] = []
    cursor = left_margin
    marker_phase = rng.randrange(2)
    slide_phase = rng.randrange(2)
    for i, (inner_w, depth, palette) in enumerate(zip(widths, layers, palettes)):
        left = cursor
        right = left + inner_w + 1
        bottom = h - rng.randint(2, 4)
        wall_height = max(depth + 2, len(palette) + 3) + rng.randint(0, 2)
        top = bottom - wall_height + 1
        if top < 9:
            return None

        grid[top : bottom + 1, left] = GREY
        grid[top : bottom + 1, right] = GREY
        grid[bottom, left : right + 1] = GREY

        # Marker side and slide side vary independently.  Alternating defaults
        # ensure left/right branches are visible even in a three-example set.
        marker_side = left if (i + marker_phase) % 2 == 0 else right
        available_marker_rows = list(range(top + 1, bottom))
        marker_rows = sorted(rng.sample(available_marker_rows, len(palette)), reverse=True)
        for row, color in zip(marker_rows, palette):
            grid[row, marker_side] = color

        slide_from_left = (i + slide_phase) % 2 == 0
        slide_col = left if slide_from_left else right
        blocker_col = left - 1 if slide_from_left else right + 1
        grid[top, blocker_col] = GREY

        # The water amount is an exact number of full rows, but its suspended
        # positions are scattered.  At least one drop starts above a side wall
        # and can enter only by the forced diagonal slide.
        total = depth * inner_w
        source_rows = list(range(1, top))
        if len(source_rows) < 2:
            return None
        slide_positions = [(r, slide_col) for r in source_rows]
        direct_positions = [
            (r, c)
            for r in source_rows
            for c in range(left + 1, right)
        ]
        forced = rng.choice(slide_positions)
        candidates = [p for p in slide_positions + direct_positions if p != forced]
        if len(candidates) < total - 1:
            return None
        drops = [forced] + rng.sample(candidates, total - 1)
        for r, c in drops:
            if grid[r, c] != BLACK:
                return None
            grid[r, c] = BLUE

        specs.append(
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "palette": palette,
                "layers": depth,
                "slide_col": slide_col,
            }
        )
        cursor = right + 1 + (gaps[i] if i < len(gaps) else 0)

    expected = _render_expected(grid, specs)
    return grid, expected, specs


def generate_input() -> np.ndarray:
    """Generate diverse, semantically checked two- or three-basin ARC inputs."""
    global _generation_index
    basin_count = 2 if _generation_index % 3 != 1 else 3
    _generation_index += 1
    rng = random.Random()

    for _ in range(_GENERATION_RETRIES):
        candidate = _generate_attempt(basin_count, rng)
        if candidate is None:
            continue
        grid, expected, specs = candidate
        signature = (expected.shape, expected.tobytes())
        if signature in _generated_output_signatures:
            continue
        if not _semantic_check(grid, expected, specs):
            continue
        _generated_output_signatures.add(signature)
        return grid
    raise RuntimeError("failed to generate a non-degenerate basin-and-palette input")
