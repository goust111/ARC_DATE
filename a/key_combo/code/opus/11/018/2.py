from __future__ import annotations

import numpy as np


BG = 0
TOP_COLOR = 1
RIGHT_COLOR = 2
BOTTOM_COLOR = 3
LEFT_COLOR = 4
CONTROL_COLOR = 5

QUADRANTS = ("ul", "ur", "lr", "ll")
DIRECTIONS = ("top", "right", "bottom", "left")
BASE_MISMATCH_SPECS = (
    ("ul", "left"),
    ("ur", "top"),
    ("lr", "right"),
    ("ll", "bottom"),
)
MATCH_SPECS = (
    ("ul", "top"),
    ("ur", "right"),
    ("lr", "bottom"),
    ("ll", "left"),
)
GRID_SIZE_CHOICES = (13, 15, 17, 19)
CONTROL_COUNT_RANGE = (5, 8)
MAX_GENERATION_ATTEMPTS = 400


def _majority_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _edge_colors(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    top_slice = grid[0, 1:-1] if w > 2 else grid[0, :]
    bottom_slice = grid[h - 1, 1:-1] if w > 2 else grid[h - 1, :]
    left_slice = grid[1:-1, 0] if h > 2 else grid[:, 0]
    right_slice = grid[1:-1, w - 1] if h > 2 else grid[:, w - 1]
    return {
        "top": _majority_color(top_slice),
        "right": _majority_color(right_slice),
        "bottom": _majority_color(bottom_slice),
        "left": _majority_color(left_slice),
    }


def _quadrant_of(r: int, c: int, h: int, w: int) -> str:
    mid_r = (h - 1) / 2.0
    mid_c = (w - 1) / 2.0
    if r == mid_r or c == mid_c:
        raise AssertionError("control pixels must not lie on the center axes")
    if r < mid_r and c < mid_c:
        return "ul"
    if r < mid_r and c > mid_c:
        return "ur"
    if r > mid_r and c > mid_c:
        return "lr"
    return "ll"


def _quadrant_color(quadrant: str, edge_colors: dict[str, int]) -> int:
    mapping = {
        "ul": edge_colors["top"],
        "ur": edge_colors["right"],
        "lr": edge_colors["bottom"],
        "ll": edge_colors["left"],
    }
    return int(mapping[quadrant])


def _nearest_direction(r: int, c: int, h: int, w: int) -> str:
    distances = {
        "top": int(r),
        "right": int(w - 1 - c),
        "bottom": int(h - 1 - r),
        "left": int(c),
    }
    best = min(distances.values())
    winners = [direction for direction, dist in distances.items() if dist == best]
    assert len(winners) == 1, "nearest edge must be unique"
    return winners[0]


def _step_from(r: int, c: int, direction: str) -> tuple[int, int]:
    if direction == "top":
        return r - 1, c
    if direction == "right":
        return r, c + 1
    if direction == "bottom":
        return r + 1, c
    return r, c - 1


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    edge_colors = _edge_colors(g)
    out = g.copy()

    for r0, c0 in np.argwhere(g == CONTROL_COLOR):
        r = int(r0)
        c = int(c0)
        direction = _nearest_direction(r, c, h, w)
        nr, nc = _step_from(r, c, direction)
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if g[nr, nc] != BG:
            continue
        quadrant = _quadrant_of(r, c, h, w)
        out[nr, nc] = _quadrant_color(quadrant, edge_colors)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each gray control pixel one step toward its nearest edge while coloring the extension by quadrant membership.

    Input structure
    - The grid is a black background with four colored edge bars on the outer boundary and several gray control pixels in the interior.
    - The top, right, bottom, and left edge bars are visually distinct, and each control pixel lies strictly inside one of the four quadrants around the grid center.

    Rule
    - For each gray control pixel, compare its orthogonal distances to the top, right, bottom, and left edges and find the unique nearest edge direction.
    - Paint exactly the adjacent cell one step in that nearest direction, unless that destination is already occupied.
    - The new cell's color does not come from the nearest edge; it comes from the control pixel's quadrant: upper-left uses the top edge color, upper-right uses the right edge color, lower-right uses the bottom edge color, and lower-left uses the left edge color.
    - Keep the original control pixels and the edge bars unchanged.

    Key insight
    - Direction and color are chosen by two independent spatial cues: nearest-edge distance decides where to extend, while quadrant membership decides which edge color to use.
    """
    return _apply_rule(grid)


def _make_canvas(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[:, 0] = LEFT_COLOR
    grid[:, w - 1] = RIGHT_COLOR
    grid[0, :] = TOP_COLOR
    grid[h - 1, :] = BOTTOM_COLOR
    return grid


def _quadrant_ranges(quadrant: str, h: int, w: int) -> tuple[range, range]:
    cr = h // 2
    cc = w // 2
    if quadrant == "ul":
        return range(2, cr), range(2, cc)
    if quadrant == "ur":
        return range(2, cr), range(cc + 1, w - 2)
    if quadrant == "lr":
        return range(cr + 1, h - 2), range(cc + 1, w - 2)
    return range(cr + 1, h - 2), range(2, cc)


def _candidate_cells(quadrant: str, direction: str, h: int, w: int) -> list[tuple[int, int]]:
    rows, cols = _quadrant_ranges(quadrant, h, w)
    candidates = []
    for r in rows:
        for c in cols:
            try:
                nearest = _nearest_direction(r, c, h, w)
            except AssertionError:
                continue
            if nearest == direction:
                candidates.append((int(r), int(c)))
    return candidates


def _neighbors8(cell: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = cell
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _pick_cell(
    rng: np.random.Generator,
    spec: tuple[str, str],
    h: int,
    w: int,
    controls: set[tuple[int, int]],
    targets: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    quadrant, direction = spec
    options = []
    for r, c in _candidate_cells(quadrant, direction, h, w):
        target = _step_from(r, c, direction)
        if (r, c) in forbidden or target in forbidden:
            continue
        if (r, c) in controls or target in controls:
            continue
        if target in targets:
            continue
        options.append((r, c))
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def _build_specs(rng: np.random.Generator, total_controls: int) -> list[tuple[str, str]]:
    specs = list(BASE_MISMATCH_SPECS)
    extra_count = total_controls - len(specs)
    extra_pool = list(MATCH_SPECS) + list(BASE_MISMATCH_SPECS)
    for _ in range(extra_count):
        specs.append(extra_pool[int(rng.integers(len(extra_pool)))])
    rng.shuffle(specs)
    return specs


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.choice(GRID_SIZE_CHOICES))
    w = int(rng.choice(GRID_SIZE_CHOICES))
    total_controls = int(rng.integers(*CONTROL_COUNT_RANGE))
    grid = _make_canvas(h, w)
    controls: set[tuple[int, int]] = set()
    targets: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    for spec in _build_specs(rng, total_controls):
        cell = _pick_cell(rng, spec, h, w, controls, targets, forbidden)
        if cell is None:
            return None
        r, c = cell
        direction = _nearest_direction(r, c, h, w)
        target = _step_from(r, c, direction)
        controls.add(cell)
        targets.add(target)
        forbidden.update(_neighbors8(cell))
        forbidden.add(target)
        grid[r, c] = CONTROL_COLOR

    solved = _apply_rule(grid)

    control_quadrants = {_quadrant_of(r, c, h, w) for r, c in controls}
    if control_quadrants != set(QUADRANTS):
        return None

    control_directions = {_nearest_direction(r, c, h, w) for r, c in controls}
    if control_directions != set(DIRECTIONS):
        return None

    mismatches = sum(
        _nearest_direction(r, c, h, w)
        != {"ul": "top", "ur": "right", "lr": "bottom", "ll": "left"}[_quadrant_of(r, c, h, w)]
        for r, c in controls
    )
    if mismatches < 1:
        return None

    if np.count_nonzero(solved != grid) != len(controls):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
