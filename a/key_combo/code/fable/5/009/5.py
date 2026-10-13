"""ARC task 5: Four-sided terrain ridges define and color a hidden frame."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(10))


def _background(grid: np.ndarray) -> int:
    """Return the visibly dominant (most frequent) color."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _edge_colors(grid: np.ndarray, bg: int) -> dict[str, int]:
    edge_values = {
        "top": set(map(int, grid[0])) - {bg},
        "bottom": set(map(int, grid[-1])) - {bg},
        "left": set(map(int, grid[:, 0])) - {bg},
        "right": set(map(int, grid[:, -1])) - {bg},
    }
    assert all(len(values) == 1 for values in edge_values.values()), (
        "each edge must carry exactly one terrain color"
    )
    result = {side: int(next(iter(values))) for side, values in edge_values.items()}
    assert len(set(result.values())) == 4, "the four edge terrains must have distinct colors"
    return result


def _scan_terrain(grid: np.ndarray, color: int, side: str) -> dict[str, object]:
    """Read one edge-anchored terrain in the grid's global (row, column) frame."""
    height, width = grid.shape
    expected = np.zeros_like(grid, dtype=bool)

    if side == "top":
        axes = np.flatnonzero(grid[0] == color)
        depths = []
        for col in axes:
            depth = 0
            while depth < height and int(grid[depth, col]) == color:
                depth += 1
            depths.append(depth)
            expected[:depth, col] = True
    elif side == "bottom":
        axes = np.flatnonzero(grid[-1] == color)
        depths = []
        for col in axes:
            depth = 0
            while depth < height and int(grid[height - 1 - depth, col]) == color:
                depth += 1
            depths.append(depth)
            expected[height - depth :, col] = True
    elif side == "left":
        axes = np.flatnonzero(grid[:, 0] == color)
        depths = []
        for row in axes:
            depth = 0
            while depth < width and int(grid[row, depth]) == color:
                depth += 1
            depths.append(depth)
            expected[row, :depth] = True
    elif side == "right":
        axes = np.flatnonzero(grid[:, -1] == color)
        depths = []
        for row in axes:
            depth = 0
            while depth < width and int(grid[row, width - 1 - depth]) == color:
                depth += 1
            depths.append(depth)
            expected[row, width - depth :] = True
    else:  # pragma: no cover - internal contract
        raise ValueError(f"unknown side: {side}")

    axes = np.asarray(axes, dtype=int)
    depths = np.asarray(depths, dtype=int)
    assert axes.size >= 3, "each terrain must span at least three edge positions"
    assert np.all(np.diff(axes) == 1), "each terrain must occupy one contiguous edge span"
    assert np.array_equal(expected, grid == color), (
        "terrain cells must be uninterrupted segments grown inward from their own edge"
    )

    max_depth = int(depths.max())
    winners = np.flatnonzero(depths == max_depth)
    assert winners.size == 1, "each terrain must have exactly one deepest peak"
    peak_axis = int(axes[int(winners[0])])

    if side == "top":
        tip = (max_depth - 1, peak_axis)
    elif side == "bottom":
        tip = (height - max_depth, peak_axis)
    elif side == "left":
        tip = (peak_axis, max_depth - 1)
    else:
        tip = (peak_axis, width - max_depth)

    return {
        "side": side,
        "color": int(color),
        "axes": axes,
        "depths": depths,
        "max_depth": max_depth,
        "peak_axis": peak_axis,
        "tip": tip,
        "area": int(depths.sum()),
    }


def _ray_cells(side: str, tip: tuple[int, int], rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Cells strictly between a peak tip and the first hidden-frame side it meets."""
    r0, c0, r1, c1 = rect
    row, col = tip
    if side == "top":
        return [(r, col) for r in range(row + 1, r0)]
    if side == "bottom":
        return [(r, col) for r in range(r1 + 1, row)]
    if side == "left":
        return [(row, c) for c in range(col + 1, c0)]
    if side == "right":
        return [(row, c) for c in range(c1 + 1, col)]
    raise ValueError(f"unknown side: {side}")


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 cells on either axis"

    counts = Counter(map(int, g.ravel()))
    assert len(counts) == 5, "expected one background and four terrain colors"
    bg = _background(g)
    assert counts[bg] > max(count for color, count in counts.items() if color != bg), (
        "background must be uniquely most frequent"
    )

    side_colors = _edge_colors(g, bg)
    assert set(side_colors.values()) == set(counts) - {bg}, (
        "every non-background color must be one of the four edge terrains"
    )
    terrains = {
        side: _scan_terrain(g, side_colors[side], side)
        for side in SIDES
    }

    top_col = int(terrains["top"]["peak_axis"])
    bottom_col = int(terrains["bottom"]["peak_axis"])
    left_row = int(terrains["left"]["peak_axis"])
    right_row = int(terrains["right"]["peak_axis"])
    assert top_col != bottom_col, "top and bottom peaks must define distinct vertical sides"
    assert left_row != right_row, "left and right peaks must define distinct horizontal sides"

    rect = (
        min(left_row, right_row),
        min(top_col, bottom_col),
        max(left_row, right_row),
        max(top_col, bottom_col),
    )
    r0, c0, r1, c1 = rect
    assert r1 - r0 >= 3 and c1 - c0 >= 3, "the implied rectangle must have a visible interior"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == bg), (
        "the full implied rectangle must initially be empty background"
    )

    top_tip = terrains["top"]["tip"]
    bottom_tip = terrains["bottom"]["tip"]
    left_tip = terrains["left"]["tip"]
    right_tip = terrains["right"]["tip"]
    assert top_tip[0] <= r0 - 2 and bottom_tip[0] >= r1 + 2
    assert left_tip[1] <= c0 - 2 and right_tip[1] >= c1 + 2

    rays: dict[str, list[tuple[int, int]]] = {}
    for side in SIDES:
        cells = _ray_cells(side, terrains[side]["tip"], rect)
        assert cells, "every peak must contribute a non-empty ray"
        assert all(int(g[r, c]) == bg for r, c in cells), "ray corridors must be empty"
        rays[side] = cells

    max_depth = max(int(terrain["max_depth"]) for terrain in terrains.values())
    winners = [
        side for side, terrain in terrains.items()
        if int(terrain["max_depth"]) == max_depth
    ]
    assert len(winners) == 1, "one terrain must have the unique greatest invasion depth"

    return {
        "bg": bg,
        "terrains": terrains,
        "rect": rect,
        "rays": rays,
        "winner": winners[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project four edge-terrain peaks into a hidden rectangle, color its frame by the deepest terrain, and extend four rays.

    Input structure:
    - A uniquely dominant background surrounds four differently colored solid terrains, one attached to each grid edge.
    - Top/bottom terrains are contiguous edge-rooted column profiles and left/right terrains are contiguous edge-rooted row profiles.
    - Every terrain has one unique deepest profile segment; opposite terrains have different peak coordinates and leave the central implied rectangle empty.

    Rule:
    - Use the top and bottom peak columns as the two vertical rectangle sides, and the left and right peak rows as its two horizontal sides.
    - Draw the one-cell frame in the color of the terrain whose edge-relative maximum depth is uniquely greatest, leaving the interior as background.
    - From each peak tip, extend its own color inward to but not onto the first rectangle-border cell; preserve every original terrain cell.

    Key insight:
    - Peak coordinates determine four global side lines, while the separately compared edge-relative peak depths determine the frame color.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    terrains = info["terrains"]
    r0, c0, r1, c1 = info["rect"]
    winner_color = int(terrains[info["winner"]]["color"])

    out = g.copy()
    out[r0, c0 : c1 + 1] = winner_color
    out[r1, c0 : c1 + 1] = winner_color
    out[r0 : r1 + 1, c0] = winner_color
    out[r0 : r1 + 1, c1] = winner_color

    for side in SIDES:
        color = int(terrains[side]["color"])
        for row, col in info["rays"][side]:
            out[row, col] = color
    return out


def _choose_span(axis_length: int, peak: int, width: int) -> np.ndarray | None:
    """Choose a contiguous span containing peak while keeping two corner cells clear."""
    first_allowed = 2
    last_allowed = axis_length - 3
    lo = max(first_allowed, peak - width + 1)
    hi = min(peak, last_allowed - width + 1)
    if lo > hi:
        return None
    start = random.randint(lo, hi)
    return np.arange(start, start + width, dtype=int)


def _make_profile(axes: np.ndarray, peak: int, max_depth: int) -> np.ndarray:
    """Create a connected, irregular ridge with one visibly unique deepest peak."""
    depths = []
    for axis in axes:
        distance = abs(int(axis) - peak)
        if distance == 0:
            depth = max_depth
        else:
            shoulder = max(1, max_depth - 1 - max(0, distance - 1) // 2)
            depth = random.randint(max(1, shoulder - 2), shoulder)
        depths.append(depth)
    return np.asarray(depths, dtype=int)


def _paint_terrain(
    grid: np.ndarray,
    side: str,
    color: int,
    axes: np.ndarray,
    depths: np.ndarray,
) -> bool:
    """Paint one generated terrain, rejecting accidental overlap."""
    height, width = grid.shape
    cells: list[tuple[int, int]] = []
    for axis, depth in zip(axes, depths):
        axis, depth = int(axis), int(depth)
        if side == "top":
            cells.extend((r, axis) for r in range(depth))
        elif side == "bottom":
            cells.extend((r, axis) for r in range(height - depth, height))
        elif side == "left":
            cells.extend((axis, c) for c in range(depth))
        elif side == "right":
            cells.extend((axis, c) for c in range(width - depth, width))
    if any(int(grid[r, c]) != int(grid[0, 0]) for r, c in cells):
        return False
    for row, col in cells:
        grid[row, col] = color
    return True


def _different_terrains_touch(grid: np.ndarray, bg: int) -> bool:
    """Reject even diagonal contact so the four visible objects remain unambiguous."""
    height, width = grid.shape
    for row, col in np.argwhere(grid != bg):
        color = int(grid[row, col])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(row + dr), int(col + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    other = int(grid[rr, cc])
                    if other != bg and other != color:
                        return True
    return False


def _known_ray_cells(side: str, peak: int, depth: int, shape: tuple[int, int], rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    height, width = shape
    if side == "top":
        tip = (depth - 1, peak)
    elif side == "bottom":
        tip = (height - depth, peak)
    elif side == "left":
        tip = (peak, depth - 1)
    else:
        tip = (peak, width - depth)
    return _ray_cells(side, tip, rect)


def _render_expected(grid: np.ndarray, meta: dict[str, object]) -> np.ndarray:
    """Independent construction oracle using generator metadata, not terrain detection."""
    out = grid.copy()
    r0, c0, r1, c1 = meta["rect"]
    winner_color = int(meta["terrains"][meta["winner"]]["color"])

    for col in range(c0, c1 + 1):
        out[r0, col] = winner_color
        out[r1, col] = winner_color
    for row in range(r0, r1 + 1):
        out[row, c0] = winner_color
        out[row, c1] = winner_color

    for side in SIDES:
        terrain = meta["terrains"][side]
        ray = _known_ray_cells(
            side,
            int(terrain["peak"]),
            int(terrain["max_depth"]),
            grid.shape,
            meta["rect"],
        )
        for row, col in ray:
            out[row, col] = int(terrain["color"])
    return out


def _semantic_validate(grid: np.ndarray, expected: np.ndarray, meta: dict[str, object]) -> bool:
    """Validate the stated semantics and anti-shortcut properties independently of main()."""
    height, width = grid.shape
    if max(grid.shape) > 30 or expected.shape != grid.shape:
        return False
    bg = int(meta["bg"])
    counts = Counter(map(int, grid.ravel()))
    if len(counts) != 5 or counts[bg] <= max(v for k, v in counts.items() if k != bg):
        return False

    terrain_cells: set[tuple[int, int]] = set()
    areas: dict[str, int] = {}
    for side in SIDES:
        terrain = meta["terrains"][side]
        color = int(terrain["color"])
        axes = np.asarray(terrain["axes"], dtype=int)
        depths = np.asarray(terrain["depths"], dtype=int)
        peak = int(terrain["peak"])
        max_depth = int(terrain["max_depth"])
        if axes.size < 3 or not np.all(np.diff(axes) == 1):
            return False
        if np.count_nonzero(depths == max_depth) != 1:
            return False
        if int(axes[int(np.argmax(depths))]) != peak:
            return False

        known = np.zeros_like(grid, dtype=bool)
        for axis, depth in zip(axes, depths):
            axis, depth = int(axis), int(depth)
            if side == "top":
                known[:depth, axis] = True
            elif side == "bottom":
                known[height - depth :, axis] = True
            elif side == "left":
                known[axis, :depth] = True
            else:
                known[axis, width - depth :] = True
        if not np.array_equal(known, grid == color):
            return False
        terrain_cells.update(map(tuple, np.argwhere(known)))
        areas[side] = int(known.sum())

    depths_by_side = {
        side: int(meta["terrains"][side]["max_depth"])
        for side in SIDES
    }
    deepest = max(depths_by_side.values())
    if [s for s, d in depths_by_side.items() if d == deepest] != [meta["winner"]]:
        return False
    winner_area = areas[meta["winner"]]
    if not (min(areas.values()) < winner_area < max(areas.values())):
        return False  # depth, not total area, must be needed to choose the frame color

    top_col = int(meta["terrains"]["top"]["peak"])
    bottom_col = int(meta["terrains"]["bottom"]["peak"])
    left_row = int(meta["terrains"]["left"]["peak"])
    right_row = int(meta["terrains"]["right"]["peak"])
    derived_rect = (
        min(left_row, right_row),
        min(top_col, bottom_col),
        max(left_row, right_row),
        max(top_col, bottom_col),
    )
    if derived_rect != meta["rect"] or top_col == bottom_col or left_row == right_row:
        return False
    r0, c0, r1, c1 = derived_rect
    if r1 - r0 < 3 or c1 - c0 < 3 or not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == bg):
        return False
    if _different_terrains_touch(grid, bg):
        return False

    frame = (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )
    additions = set(frame)
    ray_sets = []
    for side in SIDES:
        terrain = meta["terrains"][side]
        ray = set(_known_ray_cells(
            side,
            int(terrain["peak"]),
            int(terrain["max_depth"]),
            grid.shape,
            derived_rect,
        ))
        if not ray or ray & frame or any(ray & previous for previous in ray_sets):
            return False
        if any(int(grid[r, c]) != bg for r, c in ray):
            return False
        if any(int(expected[r, c]) != int(terrain["color"]) for r, c in ray):
            return False
        ray_sets.append(ray)
        additions |= ray

    winner_color = int(meta["terrains"][meta["winner"]]["color"])
    if any(int(grid[r, c]) != bg or int(expected[r, c]) != winner_color for r, c in frame):
        return False
    changed = set(map(tuple, np.argwhere(grid != expected)))
    terrain_rows, terrain_cols = zip(*terrain_cells)
    if changed != additions or not np.array_equal(
        grid[terrain_rows, terrain_cols], expected[terrain_rows, terrain_cols]
    ):
        return False
    return True


def _generation_attempt(winner: str) -> np.ndarray | None:
    height = random.randint(22, 30)
    width = random.randint(22, 30)

    r0 = random.randint(8, height - 14)
    r1 = random.randint(r0 + 5, height - 9)
    c0 = random.randint(8, width - 14)
    c1 = random.randint(c0 + 5, width - 9)
    rect = (r0, c0, r1, c1)

    top_peak, bottom_peak = ((c0, c1) if random.random() < 0.5 else (c1, c0))
    left_peak, right_peak = ((r0, r1) if random.random() < 0.5 else (r1, r0))
    peaks = {
        "top": top_peak,
        "bottom": bottom_peak,
        "left": left_peak,
        "right": right_peak,
    }

    depth_cap = min(r0 - 1, height - r1 - 2, c0 - 1, width - c1 - 2)
    if depth_cap < 5:
        return None
    winner_depth = random.randint(5, min(7, depth_cap))
    max_depths = {side: random.randint(2, winner_depth - 1) for side in SIDES}
    max_depths[winner] = winner_depth
    # Always include a close losing contender so the cross-side comparison is visible.
    contender = random.choice([side for side in SIDES if side != winner])
    max_depths[contender] = winner_depth - 1

    colors = random.sample(PALETTE, 5)
    bg = int(colors[0])
    side_colors = {side: int(colors[index + 1]) for index, side in enumerate(SIDES)}

    for _ in range(80):
        terrain_meta: dict[str, dict[str, object]] = {}
        for side in SIDES:
            axis_length = width if side in ("top", "bottom") else height
            span_width = random.randint(5, min(10, axis_length - 4))
            axes = _choose_span(axis_length, peaks[side], span_width)
            if axes is None:
                break
            depths = _make_profile(axes, peaks[side], max_depths[side])
            terrain_meta[side] = {
                "color": side_colors[side],
                "axes": axes,
                "depths": depths,
                "peak": peaks[side],
                "max_depth": max_depths[side],
            }
        if len(terrain_meta) != 4:
            continue
        areas = {side: int(np.sum(terrain_meta[side]["depths"])) for side in SIDES}
        if not (min(areas.values()) < areas[winner] < max(areas.values())):
            continue

        grid = np.full((height, width), bg, dtype=int)
        if not all(
            _paint_terrain(
                grid,
                side,
                side_colors[side],
                terrain_meta[side]["axes"],
                terrain_meta[side]["depths"],
            )
            for side in SIDES
        ):
            continue
        if _different_terrains_touch(grid, bg):
            continue

        meta = {
            "bg": bg,
            "terrains": terrain_meta,
            "rect": rect,
            "winner": winner,
        }
        expected = _render_expected(grid, meta)
        if not _semantic_validate(grid, expected, meta):
            continue
        try:
            solved = main(grid)
        except (AssertionError, ValueError, IndexError):
            continue
        if np.array_equal(solved, expected) and not np.array_equal(solved, grid):
            return grid
    return None


_winner_counter = random.randrange(len(SIDES))


def generate_input() -> np.ndarray:
    """Generate a varied valid instance; every four consecutive calls cover all winner directions."""
    global _winner_counter
    winner = SIDES[_winner_counter % len(SIDES)]
    for _ in range(400):
        grid = _generation_attempt(winner)
        if grid is not None:
            _winner_counter += 1
            return grid
    raise RuntimeError(f"failed to generate a valid four-ridge instance for winner {winner}")
