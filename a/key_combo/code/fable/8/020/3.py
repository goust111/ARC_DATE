import random

import numpy as np


BLACK = 0
GREEN = 3
PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _orbit_coords(height: int, width: int, row: int, col: int) -> list[tuple[int, int]]:
    """Return the unique two-axis reflection orbit of one cell."""
    orbit = {
        (int(row), int(col)),
        (int(row), int(width - 1 - col)),
        (int(height - 1 - row), int(col)),
        (int(height - 1 - row), int(width - 1 - col)),
    }
    return sorted(orbit)


def _rep_coord(height: int, width: int, row: int, col: int) -> tuple[int, int]:
    """Fold one cell into its upper-left reflection representative."""
    return min(int(row), int(height - 1 - row)), min(int(col), int(width - 1 - col))


def main(grid: np.ndarray) -> np.ndarray:
    """Fold each damaged symmetry orbit into the upper-left representative and recover its color.

    Input structure:
    - The grid is a multicolor pattern on a black background.
    - Ignoring green damaged cells, the remaining visible pattern is symmetric under
      both left-right and top-bottom reflection.
    - Every reflection orbit that contains green still shows exactly one non-green
      color on its undamaged cells.

    Rule:
    - For each orbit containing at least one green cell, inspect its reflected mates
      and read the unique visible non-green color shared by the undamaged cells.
    - Fold that damaged orbit into its upper-left representative coordinate.
    - Return the tight cropped bounding box of all folded representatives, painting
      them with their recovered colors and leaving every other output cell black.

    Key insight:
    - The green cells themselves are the hidden object to extract; full-image symmetry
      is only the mechanism for recovering each damaged cell's color and canonical
      folded location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    height, width = g.shape
    damaged = np.argwhere(g == GREEN)
    assert damaged.size > 0, "expected at least one damaged cell"

    folded: dict[tuple[int, int], int] = {}
    for row, col in damaged:
        row = int(row)
        col = int(col)
        rep = _rep_coord(height, width, row, col)
        if rep in folded:
            continue

        orbit = _orbit_coords(height, width, row, col)
        visible_colors = {int(g[r, c]) for r, c in orbit if int(g[r, c]) != GREEN}
        assert len(visible_colors) == 1, "each damaged orbit must reveal one true color"
        folded[rep] = visible_colors.pop()

    reps = np.array(list(folded.keys()), dtype=int)
    r0, c0 = reps.min(axis=0)
    r1, c1 = reps.max(axis=0)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (row, col), color in folded.items():
        out[row - r0, col - c0] = int(color)
    return out


def _sample_dimensions() -> tuple[int, int]:
    fold_h = random.randint(3, 9)
    fold_w = random.randint(3, 9)
    odd_h = random.choice([False, True])
    odd_w = random.choice([False, True])
    height = 2 * fold_h - int(odd_h)
    width = 2 * fold_w - int(odd_w)
    return height, width


def _folded_domain(height: int, width: int) -> tuple[int, int]:
    return (height + 1) // 2, (width + 1) // 2


def _neighbors4(row: int, col: int, height: int, width: int) -> list[tuple[int, int]]:
    neighbors = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width:
            neighbors.append((nr, nc))
    return neighbors


def _sample_connected_subset(
    allowed: set[tuple[int, int]],
    height: int,
    width: int,
    count: int,
    forced: tuple[int, int] | None = None,
) -> set[tuple[int, int]]:
    start = forced if forced is not None else random.choice(list(allowed))
    chosen = {start}
    frontier = {p for p in _neighbors4(start[0], start[1], height, width) if p in allowed}

    while len(chosen) < count and frontier:
        nxt = random.choice(list(frontier))
        frontier.remove(nxt)
        chosen.add(nxt)
        for neighbor in _neighbors4(nxt[0], nxt[1], height, width):
            if neighbor in allowed and neighbor not in chosen:
                frontier.add(neighbor)

    if len(chosen) < count:
        remaining = list(allowed - chosen)
        remaining.sort(key=lambda p: min(abs(p[0] - r) + abs(p[1] - c) for r, c in chosen))
        for pos in remaining:
            chosen.add(pos)
            if len(chosen) == count:
                break

    return chosen


def _assign_colors(count: int) -> list[int]:
    palette_size = random.randint(1, min(4, len(PALETTE)))
    palette = random.sample(PALETTE, palette_size)
    colors = [random.choice(palette) for _ in range(count)]
    if count >= 2 and palette_size >= 2:
        colors[0] = palette[0]
        colors[1] = palette[1]
        random.shuffle(colors)
    return colors


def _build_damage_reps(height: int, width: int) -> dict[tuple[int, int], int] | None:
    fold_h, fold_w = _folded_domain(height, width)
    all_reps = [(r, c) for r in range(fold_h) for c in range(fold_w)]
    allowed = {rep for rep in all_reps if len(_orbit_coords(height, width, rep[0], rep[1])) > 1}
    if len(allowed) < 3:
        return None

    axis_reps = [rep for rep in allowed if len(_orbit_coords(height, width, rep[0], rep[1])) == 2]
    forced = random.choice(axis_reps) if axis_reps and random.random() < 0.7 else None

    max_count = min(9, len(allowed))
    count = random.randint(3, max_count)
    reps = _sample_connected_subset(allowed, fold_h, fold_w, count, forced=forced)

    size4_reps = [rep for rep in allowed if len(_orbit_coords(height, width, rep[0], rep[1])) == 4]
    if size4_reps and not any(rep in reps for rep in size4_reps):
        reps.pop()
        reps.add(random.choice(size4_reps))
    if forced is not None and forced not in reps:
        reps.pop()
        reps.add(forced)

    colors = _assign_colors(len(reps))
    return {rep: color for rep, color in zip(sorted(reps), colors)}


def _add_filler_orbits(grid: np.ndarray, occupied_reps: set[tuple[int, int]]) -> None:
    height, width = grid.shape
    fold_h, fold_w = _folded_domain(height, width)
    remaining = [(r, c) for r in range(fold_h) for c in range(fold_w) if (r, c) not in occupied_reps]
    if not remaining:
        return

    filler_count = random.randint(1, min(6, len(remaining)))
    for rep in random.sample(remaining, filler_count):
        color = random.choice(PALETTE)
        for row, col in _orbit_coords(height, width, rep[0], rep[1]):
            grid[row, col] = color


def _render_input(height: int, width: int, damage_reps: dict[tuple[int, int], int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    _add_filler_orbits(grid, set(damage_reps))

    for rep, color in damage_reps.items():
        orbit = _orbit_coords(height, width, rep[0], rep[1])
        damage_count = random.randint(1, len(orbit) - 1)
        hidden = set(random.sample(orbit, damage_count))
        for row, col in orbit:
            grid[row, col] = GREEN if (row, col) in hidden else color
    return grid


def _gen_attempt() -> np.ndarray | None:
    height, width = _sample_dimensions()
    damage_reps = _build_damage_reps(height, width)
    if damage_reps is None:
        return None

    grid = _render_input(height, width, damage_reps)
    expected = np.array(main(grid), copy=True)
    if expected.ndim != 2 or expected.size == 0:
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-folding input")
