import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_RELS = {
    "up": ((0, 0), (-1, 0), (0, -1), (0, 1)),
    "down": ((0, 0), (1, 0), (0, -1), (0, 1)),
    "left": ((0, 0), (0, -1), (-1, 0), (1, 0)),
    "right": ((0, 0), (0, 1), (-1, 0), (1, 0)),
}
LEAF_TO_DIR = {
    frozenset({(-1, 0), (0, -1), (0, 1)}): "up",
    frozenset({(1, 0), (0, -1), (0, 1)}): "down",
    frozenset({(0, -1), (-1, 0), (1, 0)}): "left",
    frozenset({(0, 1), (-1, 0), (1, 0)}): "right",
}
SIDES = ("up", "down", "left", "right")
SIDE_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
INWARD_DIR = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
OUTWARD_BY_QUADRANT = {
    "tl": ("up", "left"),
    "tr": ("up", "right"),
    "bl": ("down", "left"),
    "br": ("down", "right"),
}

GEN_RETRIES = 300
GRID_MIN = 25
GRID_MAX = 30
RAY_SPACING = 4
EXTRA_ARROW_MAX = 2
PLACEMENT_TRIES = 120


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each arrow by how many other arrows' rays terminate on it.

    Input structure:
    - The background is the dominant color.
    - Above the bottom row, the foreground objects are small monochrome arrow sprites.
    - Each arrow is a 4-cell 'T' shape inside a 3x3 neighborhood: one center cell,
      one tip cell, and two wing cells flanking the tip.
    - The bottom row contains a 4-color legend of singleton pixels ordered from left
      to right for hit counts 0, 1, 2, and 3+.

    Rule:
    - Parse every foreground component above the legend row as an arrow and infer its
      pointing direction from the sprite geometry.
    - From each arrow tip, cast a ray one cell beyond the tip and keep stepping in the
      pointing direction until either the grid boundary is crossed or another arrow
      pixel is reached.
    - If a ray first reaches another arrow, count that as one hit on the struck arrow.
    - Recolor every arrow with the legend color at index min(hit_count, 3); keep the
      legend row unchanged and do not draw the rays themselves.

    Key insight:
    - The 4-cell arrow shape simultaneously encodes the object identity and the ray
      direction, so solving the puzzle requires decoding sprite orientation before the
      hit-count-to-color lookup can be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a legend row"

    bg = _get_bg(g)
    legend = _extract_legend(g, bg)
    arrows = _extract_arrows(g, bg)
    hits = _compute_hits(g.shape, arrows)

    out = g.copy()
    for arrow, hit_count in zip(arrows, hits):
        new_color = legend[min(hit_count, 3)]
        for r, c in arrow["cells"]:
            out[r, c] = new_color
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _extract_legend(grid: np.ndarray, bg: int) -> list[int]:
    last_row = grid[-1]
    cols = [int(c) for c in np.flatnonzero(last_row != bg)]
    assert len(cols) == 4, "legend row must contain exactly four colored cells"
    colors = [int(last_row[c]) for c in cols]
    assert len(set(colors)) == 4, "legend colors must be distinct"
    return colors


def _find_components(body: np.ndarray, bg: int) -> list[dict]:
    h, w = body.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(body[r, c]) == bg:
                continue
            color = int(body[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS.values():
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(body[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _extract_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    body = grid[:-1]
    arrows: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    for comp in _find_components(body, bg):
        parsed = _parse_arrow(comp["coords"])
        assert parsed is not None, "every foreground component above the legend must be an arrow"
        center, direction = parsed
        cells = frozenset((int(r), int(c)) for r, c in comp["coords"])
        assert occupied.isdisjoint(cells), "arrow components must not overlap"
        occupied.update(cells)
        arrows.append(
            {
                "color": int(comp["color"]),
                "center": center,
                "direction": direction,
                "cells": cells,
                "tip": _tip(center, direction),
            }
        )

    assert arrows, "expected at least one arrow"
    return arrows


def _parse_arrow(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], str] | None:
    if len(coords) != 4:
        return None

    coord_set = {(int(r), int(c)) for r, c in coords}
    degrees = {}
    for r, c in coord_set:
        degree = 0
        for dr, dc in DIRS.values():
            if (r + dr, c + dc) in coord_set:
                degree += 1
        degrees[(r, c)] = degree

    centers = [cell for cell, degree in degrees.items() if degree == 3]
    leaves = [cell for cell, degree in degrees.items() if degree == 1]
    if len(centers) != 1 or len(leaves) != 3:
        return None

    center = centers[0]
    leaf_offsets = frozenset((r - center[0], c - center[1]) for r, c in leaves)
    direction = LEAF_TO_DIR.get(leaf_offsets)
    if direction is None:
        return None
    return center, direction


def _tip(center: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return center[0] + dr, center[1] + dc


def _compute_hits(shape: tuple[int, int], arrows: list[dict]) -> list[int]:
    height, width = map(int, shape)
    occupancy: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["cells"]:
            occupancy[cell] = idx

    hits = [0] * len(arrows)
    for idx, arrow in enumerate(arrows):
        dr, dc = DIRS[arrow["direction"]]
        r, c = arrow["tip"]
        r += dr
        c += dc
        while 0 <= r < height and 0 <= c < width:
            target = occupancy.get((r, c))
            if target is not None and target != idx:
                hits[target] += 1
                break
            r += dr
            c += dc
    return hits


def _arrow_cells(center: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in ARROW_RELS[direction]}


def _paint_arrow(grid: np.ndarray, center: tuple[int, int], direction: str, color: int) -> None:
    for r, c in _arrow_cells(center, direction):
        grid[r, c] = color


def _blocked_zone(cells: set[tuple[int, int]], body_height: int, width: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < body_height and 0 <= cc < width:
                    blocked.add((rr, cc))
    return blocked


def _can_place_arrow(
    center: tuple[int, int],
    direction: str,
    body_height: int,
    width: int,
    blocked: set[tuple[int, int]],
) -> bool:
    cells = _arrow_cells(center, direction)
    for r, c in cells:
        if not (0 <= r < body_height and 0 <= c < width):
            return False
        if (r, c) in blocked:
            return False
    return True


def _try_place_arrow(
    specs: list[dict],
    center: tuple[int, int],
    direction: str,
    body_height: int,
    width: int,
    blocked: set[tuple[int, int]],
) -> bool:
    if not _can_place_arrow(center, direction, body_height, width, blocked):
        return False
    cells = _arrow_cells(center, direction)
    specs.append({"center": center, "direction": direction, "cells": frozenset(cells), "tip": _tip(center, direction)})
    blocked.update(_blocked_zone(cells, body_height, width))
    return True


def _legend_positions(width: int, rng: np.random.Generator) -> list[int]:
    gaps = [int(rng.integers(1, 3)) for _ in range(3)]
    span = 4 + sum(gaps)
    start = int(rng.integers(1, width - span))
    cols = [start]
    for gap in gaps:
        cols.append(cols[-1] + gap + 1)
    return cols


def _cluster_arrows(
    center: tuple[int, int],
    quadrant: str,
    target_hits: int,
    rng: np.random.Generator,
) -> list[tuple[tuple[int, int], str]]:
    arrows: list[tuple[tuple[int, int], str]] = []
    if target_hits == 0:
        target_dir = str(rng.choice(OUTWARD_BY_QUADRANT[quadrant]))
        arrows.append((center, target_dir))
        return arrows

    sides = list(rng.choice(SIDES, size=target_hits, replace=False))
    target_dir = str(rng.choice(sides))
    arrows.append((center, target_dir))
    for side in sides:
        dr, dc = SIDE_STEPS[side]
        shooter_center = (center[0] + RAY_SPACING * dr, center[1] + RAY_SPACING * dc)
        arrows.append((shooter_center, INWARD_DIR[side]))
    return arrows


def _extra_direction(center: tuple[int, int], body_height: int, width: int, rng: np.random.Generator) -> str:
    r, c = center
    distances = {
        "up": r,
        "down": (body_height - 1) - r,
        "left": c,
        "right": (width - 1) - c,
    }
    best = min(distances.values())
    choices = [side for side, dist in distances.items() if dist == best]
    return str(rng.choice(choices))


def _build_grid(
    height: int,
    width: int,
    bg: int,
    arrow_color: int,
    legend_colors: list[int],
    legend_cols: list[int],
    specs: list[dict],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for col, color in zip(legend_cols, legend_colors):
        grid[height - 1, col] = color
    for spec in specs:
        _paint_arrow(grid, spec["center"], spec["direction"], arrow_color)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    body_height = height - 1

    palette = list(map(int, rng.permutation(10)[:6]))
    bg, arrow_color = palette[:2]
    legend_colors = palette[2:]
    legend_cols = _legend_positions(width, rng)

    top_rows = [r for r in range(5, 8) if r < body_height - 6]
    bottom_rows = [r for r in range(max(12, body_height - 8), body_height - 4) if r <= body_height - 5]
    left_cols = [c for c in range(5, 8) if c < width - 6]
    right_cols = [c for c in range(max(12, width - 8), width - 4) if c <= width - 5]
    if not top_rows or not bottom_rows or not left_cols or not right_cols:
        return None

    centers = {
        "tl": (int(rng.choice(top_rows)), int(rng.choice(left_cols))),
        "tr": (int(rng.choice(top_rows)), int(rng.choice(right_cols))),
        "bl": (int(rng.choice(bottom_rows)), int(rng.choice(left_cols))),
        "br": (int(rng.choice(bottom_rows)), int(rng.choice(right_cols))),
    }

    quadrant_order = list(centers)
    target_hits = [0, 1, 2, int(rng.choice([3, 4]))]
    rng.shuffle(target_hits)

    specs: list[dict] = []
    blocked: set[tuple[int, int]] = set()
    for quadrant, desired_hits in zip(quadrant_order, target_hits):
        for center, direction in _cluster_arrows(centers[quadrant], quadrant, desired_hits, rng):
            if not _try_place_arrow(specs, center, direction, body_height, width, blocked):
                return None

    extra_target = int(rng.integers(0, EXTRA_ARROW_MAX + 1))
    for _ in range(extra_target):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            center = (int(rng.integers(1, body_height - 1)), int(rng.integers(1, width - 1)))
            direction = _extra_direction(center, body_height, width, rng)
            if _try_place_arrow(specs, center, direction, body_height, width, blocked):
                placed = True
                break
        if not placed:
            break

    grid = _build_grid(height, width, bg, arrow_color, legend_colors, legend_cols, specs)
    try:
        arrows = _extract_arrows(grid, bg)
        hits = _compute_hits(grid.shape, arrows)
        if {min(hit, 3) for hit in hits} != {0, 1, 2, 3}:
            return None
        _ = main(grid)
    except AssertionError:
        return None
    return grid
