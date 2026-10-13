import random
from collections import Counter

import numpy as np


DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE_DIR = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            out.append(
                {
                    "color": int(color),
                    "coords": np.asarray(coords, dtype=int),
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return out


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    candidates = []
    for color in _colors_present(grid, exclude={bg}):
        for comp in _components_of_color(grid, color):
            r0, c0, r1, c1 = comp["bbox"]
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in comp["coords"]:
                mask[int(rr) - r0, int(cc) - c0] = True
            border = np.zeros_like(mask)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.array_equal(mask, border):
                candidates.append(comp)
    return candidates


def _parse_side_distance(marker: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> tuple[str, int]:
    r, c = marker
    r0, c0, r1, c1 = frame_bbox
    if r0 <= r <= r1 and c < c0:
        return "left", c0 - c
    if r0 <= r <= r1 and c > c1:
        return "right", c - c1
    if c0 <= c <= c1 and r < r0:
        return "up", r0 - r
    if c0 <= c <= c1 and r > r1:
        return "down", r - r1
    raise AssertionError("marker must lie unambiguously outside one frame side")


def _extract_state(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frames = _frame_candidates(g, bg)
    assert len(frames) == 1, "expected exactly one hollow rectangular container"
    frame = frames[0]
    frame_color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]
    assert r1 - r0 >= 2 and c1 - c0 >= 2

    all_colors = set(_colors_present(g, exclude={bg, frame_color}))
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    interior_colors = set(map(int, np.unique(interior))) - {bg, frame_color}
    assert interior_colors, "expected interior colored pixels"
    assert all_colors == interior_colors, "every non-frame color must appear inside and on exactly one outside marker"

    instructions = []
    distances = []
    for color in sorted(interior_colors):
        coords = [tuple(map(int, p)) for p in np.argwhere(g == int(color))]
        inside = [(r, c) for r, c in coords if r0 < r < r1 and c0 < c < c1]
        outside = [(r, c) for r, c in coords if r < r0 or r > r1 or c < c0 or c > c1]
        assert inside, "each marker color must have interior pixels"
        assert len(outside) == 1, "each marker color must have exactly one outside marker"
        direction, distance = _parse_side_distance(outside[0], frame["bbox"])
        distances.append(distance)
        instructions.append(
            {
                "color": int(color),
                "direction": direction,
                "distance": int(distance),
                "marker": outside[0],
            }
        )

    assert len(set(distances)) == len(distances), "marker distances must define a unique order"
    instructions.sort(key=lambda item: int(item["distance"]))
    return bg, frame, instructions


def _slide_color(out: np.ndarray, color: int, frame_bbox: tuple[int, int, int, int], direction: str, bg: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    coords = []
    for rr, cc in np.argwhere(out == int(color)):
        r = int(rr)
        c = int(cc)
        if ir0 <= r <= ir1 and ic0 <= c <= ic1:
            coords.append((r, c))
    if not coords:
        return

    for r, c in coords:
        out[r, c] = bg

    if direction == "left":
        coords.sort(key=lambda p: (p[1], p[0]))
    elif direction == "right":
        coords.sort(key=lambda p: (-p[1], p[0]))
    elif direction == "up":
        coords.sort(key=lambda p: (p[0], p[1]))
    elif direction == "down":
        coords.sort(key=lambda p: (-p[0], p[1]))
    else:
        raise AssertionError("unknown direction")

    dr, dc = DIR_VECS[direction]
    for r, c in coords:
        cr, cc = r, c
        while True:
            nr = cr + dr
            nc = cc + dc
            if not (ir0 <= nr <= ir1 and ic0 <= nc <= ic1):
                break
            if int(out[nr, nc]) != int(bg):
                break
            cr, cc = nr, nc
        out[cr, cc] = int(color)


def _apply_transform(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], instructions: list[dict], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for item in instructions:
        mr, mc = item["marker"]
        out[mr, mc] = bg
    for item in sorted(instructions, key=lambda x: int(x["distance"])):
        _slide_color(out, int(item["color"]), frame_bbox, str(item["direction"]), bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each interior color toward the side named by its outside marker, ordered by marker distance.

    Input structure:
    - A dominant background surrounds one unique 1-cell-thick hollow rectangular container.
    - The container interior contains scattered colored pixels from several colors.
    - For each interior color, exactly one same-color singleton marker lies outside the container on one side.

    Rule:
    - A marker's side determines that color's slide direction: left, right, up, or down from the container toward that side.
    - Process colors by increasing straight distance from the marker to the container's outer wall.
    - When one color is processed, all of its interior pixels slide inside the container until blocked by the inner wall or already-settled pixels; pixels nearer the destination wall settle first.
    - Erase all outside markers and keep the container border unchanged.

    Key insight:
    - The outside markers are not moved into the box; they jointly encode both direction and priority, so earlier colors become the blockers that later colors stack against.
    """
    bg, frame, instructions = _extract_state(np.asarray(grid, dtype=int))
    return _apply_transform(np.asarray(grid, dtype=int), frame["bbox"], instructions, bg)


def _sorted_instructions(instructions: list[dict]) -> list[dict]:
    return sorted(
        [{k: v for k, v in item.items()} for item in instructions],
        key=lambda item: int(item["distance"]),
    )


def _build_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _pick_side_assignments(n_colors: int) -> list[str]:
    sides = ["up", "down", "left", "right"]
    priority_side = random.choice(sides)
    others = [side for side in sides if side != priority_side]
    picked = [priority_side, priority_side, random.choice(others)]
    if n_colors == 4:
        picked.append(random.choice(others + [priority_side]))
    return picked


def _pick_marker_coords(frame_bbox: tuple[int, int, int, int], directions: list[str], distances: list[int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    side_slots = {
        "left": list(range(r0 + 1, r1)),
        "right": list(range(r0 + 1, r1)),
        "up": list(range(c0 + 1, c1)),
        "down": list(range(c0 + 1, c1)),
    }
    for slots in side_slots.values():
        random.shuffle(slots)

    coords = []
    used = {side: set() for side in side_slots}
    for side, dist in zip(directions, distances):
        slots = side_slots[side]
        anchor = None
        for candidate in slots:
            if candidate not in used[side]:
                anchor = candidate
                break
        if anchor is None:
            raise AssertionError("not enough side slots for markers")
        used[side].add(anchor)
        if side == "left":
            coords.append((anchor, c0 - dist))
        elif side == "right":
            coords.append((anchor, c1 + dist))
        elif side == "up":
            coords.append((r0 - dist, anchor))
        else:
            coords.append((r1 + dist, anchor))
    return coords


def _free_cells(frame_bbox: tuple[int, int, int, int], reserved_row: int | None, reserved_col: int | None) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if reserved_row is not None and r == reserved_row:
                continue
            if reserved_col is not None and c == reserved_col:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _place_priority_gadget(frame_bbox: tuple[int, int, int, int], direction: str) -> tuple[tuple[int, int], tuple[int, int], int | None, int | None]:
    r0, c0, r1, c1 = frame_bbox
    inner_rows = list(range(r0 + 1, r1))
    inner_cols = list(range(c0 + 1, c1))
    if direction in {"left", "right"}:
        row = random.choice(inner_rows)
        left_choices = inner_cols[1 : max(2, len(inner_cols) // 2 + 1)]
        right_choices = inner_cols[max(1, len(inner_cols) // 2 - 1) : -1]
        if not left_choices:
            left_choices = inner_cols[:-1]
        if not right_choices:
            right_choices = inner_cols[1:]
        if direction == "left":
            b_col = random.choice(left_choices)
            a_col = random.choice([c for c in right_choices if c != b_col] or [c for c in inner_cols if c != b_col])
        else:
            a_col = random.choice(left_choices)
            b_col = random.choice([c for c in right_choices if c != a_col] or [c for c in inner_cols if c != a_col])
        return (row, a_col), (row, b_col), row, None

    col = random.choice(inner_cols)
    top_choices = inner_rows[1 : max(2, len(inner_rows) // 2 + 1)]
    bottom_choices = inner_rows[max(1, len(inner_rows) // 2 - 1) : -1]
    if not top_choices:
        top_choices = inner_rows[:-1]
    if not bottom_choices:
        bottom_choices = inner_rows[1:]
    if direction == "up":
        b_row = random.choice(top_choices)
        a_row = random.choice([r for r in bottom_choices if r != b_row] or [r for r in inner_rows if r != b_row])
    else:
        a_row = random.choice(top_choices)
        b_row = random.choice([r for r in bottom_choices if r != a_row] or [r for r in inner_rows if r != a_row])
    return (a_row, col), (b_row, col), None, col


def _choose_extra_cell(
    direction: str,
    occupied: set[tuple[int, int]],
    pool: list[tuple[int, int]],
    seed: tuple[int, int],
) -> tuple[int, int] | None:
    preferred = []
    sr, sc = seed
    for cell in pool:
        if cell in occupied:
            continue
        r, c = cell
        if direction in {"left", "right"} and r == sr:
            preferred.append(cell)
        if direction in {"up", "down"} and c == sc:
            preferred.append(cell)
    if preferred and random.random() < 0.6:
        return random.choice(preferred)
    for cell in pool:
        if cell not in occupied:
            return cell
    return None


def _gadget_front_neighbor(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIR_VECS[direction]
    return cell[0] + dr, cell[1] + dc


def _gen_attempt() -> np.ndarray | None:
    outer_h = random.randint(8, 12)
    outer_w = random.randint(8, 12)
    grid_h = random.randint(outer_h + 8, min(26, outer_h + 12))
    grid_w = random.randint(outer_w + 8, min(26, outer_w + 12))
    r0 = random.randint(4, grid_h - outer_h - 4)
    c0 = random.randint(4, grid_w - outer_w - 4)
    frame_bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)

    bg = random.randint(0, 2)
    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    n_colors = random.choice([3, 4])
    frame_color = palette[0]
    colors = palette[1 : 1 + n_colors]

    directions = _pick_side_assignments(n_colors)
    priority_side = directions[0]
    if directions[1] != priority_side or len(set(directions)) < 2:
        return None

    distances = list(range(1, n_colors + 1))
    markers = _pick_marker_coords(frame_bbox, directions, distances)

    a_seed, b_seed, reserved_row, reserved_col = _place_priority_gadget(frame_bbox, priority_side)
    pool = _free_cells(frame_bbox, reserved_row, reserved_col)

    counts = {color: random.randint(2, 4) for color in colors}
    placements = {colors[0]: [a_seed], colors[1]: [b_seed]}
    occupied = {a_seed, b_seed}

    for idx, color in enumerate(colors[2:], start=2):
        if not pool:
            return None
        cell = pool.pop()
        placements[color] = [cell]
        occupied.add(cell)

    for idx, color in enumerate(colors):
        direction = directions[idx]
        while len(placements[color]) < counts[color]:
            cell = _choose_extra_cell(direction, occupied, pool, placements[color][0])
            if cell is None:
                return None
            placements[color].append(cell)
            occupied.add(cell)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    _build_frame(grid, frame_bbox, frame_color)

    instructions = []
    for idx, color in enumerate(colors):
        for r, c in placements[color]:
            grid[r, c] = color
        mr, mc = markers[idx]
        if int(grid[mr, mc]) != int(bg):
            return None
        grid[mr, mc] = color
        instructions.append(
            {
                "color": int(color),
                "direction": directions[idx],
                "distance": distances[idx],
                "marker": (mr, mc),
            }
        )

    try:
        parsed_bg, parsed_frame, _parsed_instructions = _extract_state(grid)
    except AssertionError:
        return None
    if parsed_bg != bg or parsed_frame["bbox"] != frame_bbox:
        return None

    oracle = _apply_transform(grid, frame_bbox, instructions, bg)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, oracle):
        return None

    swapped = _sorted_instructions(instructions)
    swapped[0]["distance"], swapped[1]["distance"] = swapped[1]["distance"], swapped[0]["distance"]
    swapped_out = _apply_transform(grid, frame_bbox, swapped, bg)
    if np.array_equal(swapped_out, oracle):
        return None

    dir_test = _sorted_instructions(instructions)
    dir_test[2]["direction"] = OPPOSITE_DIR[dir_test[2]["direction"]]
    if np.array_equal(_apply_transform(grid, frame_bbox, dir_test, bg), oracle):
        return None

    gadget_color_a = colors[0]
    gadget_color_b = colors[1]
    gadget_b_cells = [tuple(map(int, p)) for p in np.argwhere(oracle == int(gadget_color_b))]
    reserved_hits = [
        cell
        for cell in gadget_b_cells
        if (reserved_row is not None and cell[0] == reserved_row) or (reserved_col is not None and cell[1] == reserved_col)
    ]
    if len(reserved_hits) != 1:
        return None
    fr, fc = _gadget_front_neighbor(reserved_hits[0], priority_side)
    if int(oracle[fr, fc]) != int(gadget_color_a):
        return None

    if max(grid.shape) > 30 or max(oracle.shape) > 30:
        return None
    if np.array_equal(grid, oracle):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
