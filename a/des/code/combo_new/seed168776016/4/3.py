import numpy as np


GREY = 5
ALL_COLORS = tuple(range(10))
DIRECTIONS = ("up", "right", "down", "left")
MAX_GENERATE_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _line_indices(grid: np.ndarray, bg: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r] != bg)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] != bg)]


def _dominant_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).reshape(-1)
    colors, counts = np.unique(flat, return_counts=True)
    return int(colors[counts.argmax()])


def _classify_direction(local_r: int, local_c: int, height: int, width: int) -> str:
    center_r = (height - 1) / 2.0
    center_c = (width - 1) / 2.0
    dr = local_r - center_r
    dc = local_c - center_c
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    return "up" if dr < 0 else "down"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular cell with the color of the boundary side indicated by its off-center grey marker.

    Input structure:
    - The grid has a dominant background and several full-width / full-height colored divider lines.
    - These divider lines form a rectangular window-grid of cells, with colored outer borders included.
    - Each cell interior is otherwise background except for exactly one grey marker pixel placed away from the center.

    Rule:
    - Detect the divider rows and columns, then examine each cell interior separately.
    - Compare the marker position against the cell center: if its horizontal offset is larger, choose left or right; otherwise choose up or down.
    - Recolor the whole cell interior with the color of that chosen boundary side, covering the marker, while keeping all divider lines unchanged.

    Key insight:
    - The grey pixel does not point to the nearest edge by distance; it encodes a side through which directional sector around the cell center it occupies.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    row_lines = _line_indices(g, bg, axis=0)
    col_lines = _line_indices(g, bg, axis=1)

    out = g.copy()
    for r0, r1 in zip(row_lines, row_lines[1:], strict=False):
        if r1 - r0 < 2:
            continue
        for c0, c1 in zip(col_lines, col_lines[1:], strict=False):
            if c1 - c0 < 2:
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            marker_coords = np.argwhere(interior != bg)
            assert marker_coords.shape == (1, 2), "each cell must contain exactly one non-background marker"
            local_r, local_c = map(int, marker_coords[0])
            direction = _classify_direction(local_r, local_c, interior.shape[0], interior.shape[1])

            if direction == "up":
                fill = _dominant_color(g[r0, c0 + 1 : c1])
            elif direction == "down":
                fill = _dominant_color(g[r1, c0 + 1 : c1])
            elif direction == "left":
                fill = _dominant_color(g[r0 + 1 : r1, c0])
            else:
                fill = _dominant_color(g[r0 + 1 : r1, c1])

            out[r0 + 1 : r1, c0 + 1 : c1] = fill

    return out


def _boundary_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += size + 1
        positions.append(cursor)
    return positions


def _nearest_sides(local_r: int, local_c: int, height: int, width: int) -> set[str]:
    distances = {
        "up": local_r,
        "down": height - 1 - local_r,
        "left": local_c,
        "right": width - 1 - local_c,
    }
    best = min(distances.values())
    return {side for side, dist in distances.items() if dist == best}


def _marker_candidates(height: int, width: int, direction: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    center_r = (height - 1) / 2.0
    center_c = (width - 1) / 2.0
    all_candidates: list[tuple[int, int]] = []
    tricky_candidates: list[tuple[int, int]] = []

    for local_r in range(height):
        for local_c in range(width):
            dr = local_r - center_r
            dc = local_c - center_c

            if direction in {"left", "right"}:
                if direction == "left" and dc >= 0:
                    continue
                if direction == "right" and dc <= 0:
                    continue
                if abs(dc) - abs(dr) < 0.25:
                    continue
            else:
                if direction == "up" and dr >= 0:
                    continue
                if direction == "down" and dr <= 0:
                    continue
                if abs(dr) - abs(dc) < 0.25:
                    continue

            if _classify_direction(local_r, local_c, height, width) != direction:
                continue

            cell = (local_r, local_c)
            all_candidates.append(cell)
            if direction not in _nearest_sides(local_r, local_c, height, width):
                tricky_candidates.append(cell)

    assert all_candidates, f"no valid marker placement for {direction} in {height}x{width}"
    return tricky_candidates, all_candidates


def _choose_sizes(rng: np.random.Generator, count: int) -> list[int]:
    return [int(rng.integers(4, 9)) for _ in range(count)]


def _build_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    row_count = int(rng.integers(2, 4))
    col_count = int(rng.integers(2, 4))
    row_sizes = _choose_sizes(rng, row_count)
    col_sizes = _choose_sizes(rng, col_count)

    row_lines = _boundary_positions(row_sizes)
    col_lines = _boundary_positions(col_sizes)
    height = row_lines[-1] + 1
    width = col_lines[-1] + 1
    assert height <= 30 and width <= 30

    bg_choices = [color for color in ALL_COLORS if color != GREY]
    bg = int(rng.choice(bg_choices))
    line_palette = [color for color in ALL_COLORS if color not in {bg, GREY}]
    rng.shuffle(line_palette)

    needed_lines = len(row_lines) + len(col_lines)
    assert needed_lines <= len(line_palette)
    row_colors = line_palette[: len(row_lines)]
    col_colors = line_palette[len(row_lines) : needed_lines]

    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()

    for c, color in zip(col_lines, col_colors, strict=False):
        grid[:, c] = color
        expected[:, c] = color
    for r, color in zip(row_lines, row_colors, strict=False):
        grid[r, :] = color
        expected[r, :] = color

    cell_count = row_count * col_count
    directions = [DIRECTIONS[i % 4] for i in range(cell_count)]
    rng.shuffle(directions)

    used_tricky = False
    had_tricky_option = False
    idx = 0
    for i, (r0, r1) in enumerate(zip(row_lines, row_lines[1:], strict=False)):
        for j, (c0, c1) in enumerate(zip(col_lines, col_lines[1:], strict=False)):
            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            direction = directions[idx]
            idx += 1

            tricky, candidates = _marker_candidates(inner_h, inner_w, direction)
            had_tricky_option = had_tricky_option or bool(tricky)
            choices = tricky if tricky and rng.random() < 0.7 else candidates
            local_r, local_c = choices[int(rng.integers(len(choices)))]
            if (local_r, local_c) in tricky:
                used_tricky = True

            marker_r = r0 + 1 + local_r
            marker_c = c0 + 1 + local_c
            grid[marker_r, marker_c] = GREY

            if direction == "up":
                fill = int(row_colors[i])
            elif direction == "down":
                fill = int(row_colors[i + 1])
            elif direction == "left":
                fill = int(col_colors[j])
            else:
                fill = int(col_colors[j + 1])
            expected[r0 + 1 : r1, c0 + 1 : c1] = fill

    return grid, expected, had_tricky_option, used_tricky


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid, expected, had_tricky_option, used_tricky = _build_instance(rng)
        if had_tricky_option and not used_tricky:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid directional-window input")
