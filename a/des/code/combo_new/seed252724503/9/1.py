from __future__ import annotations

import random

import numpy as np


DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _marker_direction(pt: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = pt
    h, w = shape
    matches = []
    if r == 0:
        matches.append("up")
    if r == h - 1:
        matches.append("down")
    if c == 0:
        matches.append("left")
    if c == w - 1:
        matches.append("right")
    if not matches:
        distances = [
            (r, "up"),
            (h - 1 - r, "down"),
            (c, "left"),
            (w - 1 - c, "right"),
        ]
        best = min(d for d, _ in distances)
        matches = [name for d, name in distances if d == best]
    assert len(matches) == 1, f"marker {pt} must indicate one unique edge"
    return matches[0]


def _is_inside_frame(r: int, c: int, frame: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = frame
    return r0 < r < r1 and c0 < c < c1


def _find_regions(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            comp_set = set(comp)
            border = _border_cells(r0, c0, r1, c1)
            if comp_set != border:
                continue
            interior_pixels = [
                (r, c)
                for r in range(r0 + 1, r1)
                for c in range(c0 + 1, c1)
                if int(grid[r, c]) != bg
            ]
            assert len(interior_pixels) == 1, "each frame must contain exactly one seed pixel"
            seed = interior_pixels[0]
            regions.append(
                {
                    "wall_color": color,
                    "frame": (r0, c0, r1, c1),
                    "seed": seed,
                    "seed_color": int(grid[seed]),
                }
            )
    assert regions, "expected at least one closed rectangular frame"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open the marker-selected side of each seeded rectangle and extend the seed color straight to the grid edge.

    Input structure:
    - The dominant color is background.
    - One or more one-cell-thick rectangular wall frames enclose background interiors.
    - Each frame interior contains exactly one non-background seed pixel.
    - Outside the frames, singleton marker pixels sit on the grid edges; a marker's color matches one or more seeds.

    Rule:
    - Detect every rectangular frame and read the color C of its unique interior seed.
    - Build a mapping from marker color to direction by checking whether that marker lies on the top, bottom, left, or right edge.
    - For each frame, use the matching direction D to choose the aligned wall cell on side D, using the seed's column for up/down and the seed's row for left/right.
    - Paint color C from the seed through that exit cell and continue in direction D until the grid boundary; all non-exit wall cells stay unchanged.

    Key insight:
    - The seed color does not encode the action directly; it is only a lookup key that links the seed to an external edge marker, and that marker's side determines the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    regions = _find_regions(g, bg)
    frame_boxes = [region["frame"] for region in regions]
    frame_cells = {
        pt
        for region in regions
        for pt in _border_cells(*region["frame"])  # type: ignore[arg-type]
    }

    marker_dirs: dict[int, str] = {}
    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value == bg or (r, c) in frame_cells:
                continue
            if any(_is_inside_frame(r, c, frame) for frame in frame_boxes):
                continue
            assert value not in marker_dirs, f"duplicate marker color {value}"
            marker_dirs[value] = _marker_direction((r, c), g.shape)

    assert marker_dirs, "expected at least one edge marker"

    out = g.copy()
    for region in regions:
        seed_r, seed_c = region["seed"]  # type: ignore[misc]
        seed_color = int(region["seed_color"])
        r0, c0, r1, c1 = region["frame"]  # type: ignore[misc]
        assert seed_color in marker_dirs, f"missing marker for seed color {seed_color}"
        direction = marker_dirs[seed_color]
        dr, dc = DIR_STEPS[direction]

        if direction == "up":
            exit_r, exit_c = r0, seed_c
        elif direction == "down":
            exit_r, exit_c = r1, seed_c
        elif direction == "left":
            exit_r, exit_c = seed_r, c0
        else:
            exit_r, exit_c = seed_r, c1

        out[exit_r, exit_c] = bg
        r, c = seed_r, seed_c
        while 0 <= r < h and 0 <= c < w:
            out[r, c] = seed_color
            r += dr
            c += dc

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _distribute_extra(extra: int, slots: int) -> list[int]:
    adds = [0] * slots
    for _ in range(extra):
        adds[random.randrange(slots)] += 1
    return adds


def _choose_mode_and_directions() -> tuple[str, list[str]]:
    global _GEN_CALL_COUNT
    sample_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    n_regions = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 2], k=1)[0]
    if sample_index % 2 == 0:
        mode = "horizontal"
        base = ["left", "right"]
    else:
        mode = "vertical"
        base = ["up", "down"]

    start = (sample_index // 2) % 2
    directions = [base[(start + i) % 2] for i in range(n_regions)]
    random.shuffle(directions)
    return mode, directions


def _gen_horizontal(directions: list[str]) -> np.ndarray | None:
    n = len(directions)
    heights = [random.randint(4, 6) for _ in range(n)]
    widths = [random.randint(4, 7) for _ in range(n)]
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    gaps = [random.randint(1, 3) for _ in range(n - 1)]

    h_min = top_margin + bottom_margin + sum(heights) + sum(gaps)
    if h_min > 30:
        return None
    extra_h = random.randint(0, min(30 - h_min, 6))
    adds = _distribute_extra(extra_h, n + 1)
    top_margin += adds[0]
    for i in range(n - 1):
        gaps[i] += adds[i + 1]
    bottom_margin += adds[-1]
    h = top_margin + bottom_margin + sum(heights) + sum(gaps)

    min_w = max(14, max(widths) + 6)
    if min_w > 30:
        return None
    w = random.randint(min_w, 30)

    palette = list(range(10))
    bg = random.choice(palette)
    wall = random.choice([color for color in palette if color != bg])
    seed_colors = random.sample([color for color in palette if color not in {bg, wall}], n)

    grid = np.full((h, w), bg, dtype=int)
    row = top_margin
    seed_rows: list[int] = []
    for i, direction in enumerate(directions):
        height = heights[i]
        width = widths[i]
        r0 = row
        max_c0 = w - width - 2
        if max_c0 < 2:
            return None
        c0 = random.randint(2, max_c0)
        _draw_frame(grid, r0, c0, height, width, wall)

        seed_r = random.randint(r0 + 1, r0 + height - 2)
        seed_c = random.randint(c0 + 1, c0 + width - 2)
        grid[seed_r, seed_c] = seed_colors[i]
        seed_rows.append(seed_r)

        row += height
        if i < n - 1:
            row += gaps[i]

    available_rows = [r for r in range(1, h - 1) if r not in seed_rows]
    if len(available_rows) < n:
        return None
    marker_rows = random.sample(available_rows, n)
    for color, direction, marker_r in zip(seed_colors, directions, marker_rows):
        marker_c = 0 if direction == "left" else w - 1
        if grid[marker_r, marker_c] != bg:
            return None
        grid[marker_r, marker_c] = color

    return grid


def _gen_vertical(directions: list[str]) -> np.ndarray | None:
    n = len(directions)
    widths = [random.randint(4, 6) for _ in range(n)]
    heights = [random.randint(4, 7) for _ in range(n)]
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gaps = [random.randint(1, 3) for _ in range(n - 1)]

    w_min = left_margin + right_margin + sum(widths) + sum(gaps)
    if w_min > 30:
        return None
    extra_w = random.randint(0, min(30 - w_min, 6))
    adds = _distribute_extra(extra_w, n + 1)
    left_margin += adds[0]
    for i in range(n - 1):
        gaps[i] += adds[i + 1]
    right_margin += adds[-1]
    w = left_margin + right_margin + sum(widths) + sum(gaps)

    min_h = max(14, max(heights) + 6)
    if min_h > 30:
        return None
    h = random.randint(min_h, 30)

    palette = list(range(10))
    bg = random.choice(palette)
    wall = random.choice([color for color in palette if color != bg])
    seed_colors = random.sample([color for color in palette if color not in {bg, wall}], n)

    grid = np.full((h, w), bg, dtype=int)
    col = left_margin
    seed_cols: list[int] = []
    for i, direction in enumerate(directions):
        height = heights[i]
        width = widths[i]
        c0 = col
        max_r0 = h - height - 2
        if max_r0 < 2:
            return None
        r0 = random.randint(2, max_r0)
        _draw_frame(grid, r0, c0, height, width, wall)

        seed_r = random.randint(r0 + 1, r0 + height - 2)
        seed_c = random.randint(c0 + 1, c0 + width - 2)
        grid[seed_r, seed_c] = seed_colors[i]
        seed_cols.append(seed_c)

        col += width
        if i < n - 1:
            col += gaps[i]

    available_cols = [c for c in range(1, w - 1) if c not in seed_cols]
    if len(available_cols) < n:
        return None
    marker_cols = random.sample(available_cols, n)
    for color, direction, marker_c in zip(seed_colors, directions, marker_cols):
        marker_r = 0 if direction == "up" else h - 1
        if grid[marker_r, marker_c] != bg:
            return None
        grid[marker_r, marker_c] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    mode, directions = _choose_mode_and_directions()
    if mode == "horizontal":
        grid = _gen_horizontal(directions)
    else:
        grid = _gen_vertical(directions)
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if grid.shape != out.shape or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
