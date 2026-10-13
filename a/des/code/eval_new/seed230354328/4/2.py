import random
from collections import Counter, deque

import numpy as np


OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
SIDES = ("top", "bottom", "left", "right")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append(pts)

    return components


def _edge_color_map(grid: np.ndarray, bg: int) -> dict[str, int]:
    h, w = grid.shape
    edge_arrays = {
        "top": grid[0, 1 : w - 1],
        "bottom": grid[h - 1, 1 : w - 1],
        "left": grid[1 : h - 1, 0],
        "right": grid[1 : h - 1, w - 1],
    }

    mapping: dict[str, int] = {}
    for side, arr in edge_arrays.items():
        counts = Counter(int(v) for v in arr.ravel() if int(v) != bg)
        assert counts, f"edge {side} must contain a non-background stripe"
        mapping[side] = counts.most_common(1)[0][0]

    assert len(set(mapping.values())) == 4, "outer edge stripes must use four distinct colors"
    return mapping


def _wall_color(grid: np.ndarray, bg: int) -> int:
    inner = grid[1:-1, 1:-1].ravel()
    counts = Counter(int(v) for v in inner if int(v) != bg)
    assert counts, "expected a non-background room-frame color inside the grid"
    return counts.most_common(1)[0][0]


def _find_rooms(grid: np.ndarray, bg: int, wall: int) -> list[tuple[int, int, int, int, str]]:
    rooms: list[tuple[int, int, int, int, str]] = []

    for pts in _components(grid == wall):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        comp = set(pts)
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0 + 1, r1):
            border.add((r, c0))
            border.add((r, c1))

        if not comp <= border:
            continue

        missing = border - comp
        extra = comp - border
        if extra or len(missing) != 1:
            continue

        gr, gc = next(iter(missing))
        if gr == r0 and c0 < gc < c1:
            side = "top"
        elif gr == r1 and c0 < gc < c1:
            side = "bottom"
        elif gc == c0 and r0 < gr < r1:
            side = "left"
        elif gc == c1 and r0 < gr < r1:
            side = "right"
        else:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior == wall):
            continue

        rooms.append((r0, c0, r1, c1, side))

    rooms.sort()
    assert rooms, "expected at least one valid room"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room with the color from the opposite border stripe based on the room's doorway side.

    Input structure:
    - bg = most common color.
    - Grey line rectangles form several rooms; each room has a bg interior and exactly one single-cell gap on one side of its border.
    - The top, bottom, left, and right outer edges each carry a distinct stripe color.

    Rule:
    - Read the four outer edge stripes to build a side->color mapping.
    - Find every rectangular grey room and locate the side containing its unique missing border cell.
    - Fill the room's interior bg cells with the color from the opposite outer edge: top gap uses bottom stripe, left gap uses right stripe, and vice versa.
    - Keep the grey borders and edge stripes unchanged.

    Key insight: The doorway side does not match its own edge color; each room takes the stripe color from the opposite side of the grid.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    edge_colors = _edge_color_map(g, bg)
    wall = _wall_color(g, bg)
    rooms = _find_rooms(g, bg, wall)

    out = g.copy()
    for r0, c0, r1, c1, side in rooms:
        fill_color = edge_colors[OPPOSITE_SIDE[side]]
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[patch == bg] = fill_color

    return out


# --------------- generator ---------------

LAYOUT_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
INTERIOR_SIZE_RANGE = (2, 5)
MARGIN_RANGE = (1, 2)
GAP_RANGE = (1, 2)
MAX_GENERATION_RETRIES = 400


def _sample_layout(rng: random.Random) -> dict[str, object]:
    for _ in range(200):
        room_rows, room_cols = rng.choice(LAYOUT_CHOICES)
        interior_heights = [
            rng.randint(INTERIOR_SIZE_RANGE[0], INTERIOR_SIZE_RANGE[1]) for _ in range(room_rows)
        ]
        interior_widths = [
            rng.randint(INTERIOR_SIZE_RANGE[0], INTERIOR_SIZE_RANGE[1]) for _ in range(room_cols)
        ]

        top_margin = rng.randint(MARGIN_RANGE[0], MARGIN_RANGE[1])
        bottom_margin = rng.randint(MARGIN_RANGE[0], MARGIN_RANGE[1])
        left_margin = rng.randint(MARGIN_RANGE[0], MARGIN_RANGE[1])
        right_margin = rng.randint(MARGIN_RANGE[0], MARGIN_RANGE[1])
        row_gaps = [rng.randint(GAP_RANGE[0], GAP_RANGE[1]) for _ in range(room_rows - 1)]
        col_gaps = [rng.randint(GAP_RANGE[0], GAP_RANGE[1]) for _ in range(room_cols - 1)]

        height = (
            2
            + top_margin
            + bottom_margin
            + sum(v + 2 for v in interior_heights)
            + sum(row_gaps)
        )
        width = (
            2
            + left_margin
            + right_margin
            + sum(v + 2 for v in interior_widths)
            + sum(col_gaps)
        )

        if height <= 30 and width <= 30:
            return {
                "room_rows": room_rows,
                "room_cols": room_cols,
                "interior_heights": interior_heights,
                "interior_widths": interior_widths,
                "top_margin": top_margin,
                "bottom_margin": bottom_margin,
                "left_margin": left_margin,
                "right_margin": right_margin,
                "row_gaps": row_gaps,
                "col_gaps": col_gaps,
                "height": height,
                "width": width,
            }

    raise RuntimeError("failed to sample a valid layout")


def _axis_starts(lengths: list[int], margin: int, gaps: list[int]) -> list[int]:
    starts: list[int] = []
    pos = 1 + margin
    for i, inner in enumerate(lengths):
        starts.append(pos)
        pos += inner + 2
        if i < len(gaps):
            pos += gaps[i]
    return starts


def _direction_pool(rng: random.Random, count: int) -> list[str]:
    pool: list[str] = []
    while len(pool) < count:
        chunk = list(SIDES)
        rng.shuffle(chunk)
        pool.extend(chunk)
    return pool[:count]


def _expected_output(
    grid: np.ndarray,
    bg: int,
    room_specs: list[tuple[int, int, int, int, str]],
    edge_colors: dict[str, int],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1, side in room_specs:
        fill_color = edge_colors[OPPOSITE_SIDE[side]]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _sample_layout(rng)
    bg, wall, top_color, bottom_color, left_color, right_color = rng.sample(list(range(10)), 6)

    grid = np.full((layout["height"], layout["width"]), bg, dtype=int)
    grid[0, 1:-1] = top_color
    grid[-1, 1:-1] = bottom_color
    grid[1:-1, 0] = left_color
    grid[1:-1, -1] = right_color

    row_starts = _axis_starts(
        layout["interior_heights"], layout["top_margin"], layout["row_gaps"]
    )
    col_starts = _axis_starts(
        layout["interior_widths"], layout["left_margin"], layout["col_gaps"]
    )

    positions = [(ri, ci) for ri in range(layout["room_rows"]) for ci in range(layout["room_cols"])]
    rng.shuffle(positions)
    direction_map = dict(zip(positions, _direction_pool(rng, len(positions))))

    room_specs: list[tuple[int, int, int, int, str]] = []
    for ri, r0 in enumerate(row_starts):
        for ci, c0 in enumerate(col_starts):
            interior_h = layout["interior_heights"][ri]
            interior_w = layout["interior_widths"][ci]
            r1 = r0 + interior_h + 1
            c1 = c0 + interior_w + 1

            grid[r0, c0 : c1 + 1] = wall
            grid[r1, c0 : c1 + 1] = wall
            grid[r0 : r1 + 1, c0] = wall
            grid[r0 : r1 + 1, c1] = wall

            side = direction_map[(ri, ci)]
            if side == "top":
                gap = (r0, rng.randint(c0 + 1, c1 - 1))
            elif side == "bottom":
                gap = (r1, rng.randint(c0 + 1, c1 - 1))
            elif side == "left":
                gap = (rng.randint(r0 + 1, r1 - 1), c0)
            else:
                gap = (rng.randint(r0 + 1, r1 - 1), c1)

            grid[gap] = bg
            room_specs.append((r0, c0, r1, c1, side))

    if _background_color(grid) != bg:
        return None
    if _wall_color(grid, bg) != wall:
        return None

    found_rooms = _find_rooms(grid, bg, wall)
    if sorted(found_rooms) != sorted(room_specs):
        return None

    edge_colors = {
        "top": top_color,
        "bottom": bottom_color,
        "left": left_color,
        "right": right_color,
    }
    expected = _expected_output(grid, bg, room_specs, edge_colors)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
