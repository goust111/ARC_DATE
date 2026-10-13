from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
MARKER = 9
BAR = 5
OBJECT_COLORS = (1, 2, 3, 4)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "down", "left", "right")


MARKER_TEMPLATES = (
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
)


UP_PROTOTYPES = (
    ((0, 0), (-1, 0)),
    ((0, 0), (-1, 0), (-2, 0)),
    ((0, 0), (-1, 0), (-1, 1)),
    ((0, 0), (-1, 0), (-1, -1)),
    ((0, 0), (-1, 0), (-2, 0), (-1, 1)),
    ((0, 0), (-1, 0), (-2, 0), (-1, -1)),
    ((0, 0), (-1, 0), (-1, -1), (-1, 1)),
    ((0, 0), (-1, 0), (-2, 0), (-1, -1), (-1, 1)),
    ((0, 0), (-1, 0), (-2, 0), (-2, 1), (-1, 1)),
    ((0, 0), (-1, 0), (-2, 0), (-2, -1), (-1, -1)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(grid: np.ndarray, *, colors: tuple[int, ...] | list[int] | None = None) -> list[dict]:
    h, w = grid.shape
    if colors is None:
        colors = sorted(set(map(int, np.unique(grid))) - {BG})
    else:
        colors = [int(color) for color in colors if int(color) != BG]

    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for color in colors:
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(grid[r, c]) != color:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                cells: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for nr, nc in _neighbors4(rr, cc):
                        if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                coords = np.array(cells, dtype=int)
                out.append({"color": color, "coords": coords, "area": len(cells)})
    return out


def _round_half_up(coords: np.ndarray) -> tuple[int, int]:
    mean_r = float(coords[:, 0].mean())
    mean_c = float(coords[:, 1].mean())
    return int(np.floor(mean_r + 0.5)), int(np.floor(mean_c + 0.5))


def _direction_name(vec: tuple[int, int]) -> str:
    for name, candidate in DIRS.items():
        if candidate == vec:
            return name
    raise ValueError(vec)


def _direction_from_marker_set(
    marker_set: set[tuple[int, int]],
    shape: tuple[int, int],
    coords: np.ndarray,
) -> str | None:
    touched: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS.values():
            nr = rr + dr
            nc = cc + dc
            if _in_bounds(shape, nr, nc) and (nr, nc) in marker_set:
                touched.add((-dr, -dc))
    if len(touched) != 1:
        return None
    return _direction_name(next(iter(touched)))


def _object_direction(grid: np.ndarray, coords: np.ndarray) -> str:
    marker_set = {tuple(cell) for cell in np.argwhere(grid == MARKER).tolist()}
    direction = _direction_from_marker_set(marker_set, grid.shape, coords)
    assert direction is not None, "each object must touch the marker from exactly one cardinal side"
    return direction


def _steps_to_edge(shape: tuple[int, int], coords: np.ndarray, direction: str) -> int:
    dr, dc = DIRS[direction]
    h, w = shape
    if dr < 0:
        return int(coords[:, 0].min())
    if dr > 0:
        return int(h - 1 - coords[:, 0].max())
    if dc < 0:
        return int(coords[:, 1].min())
    return int(w - 1 - coords[:, 1].max())


def _translate(coords: np.ndarray, direction: str, steps: int) -> np.ndarray:
    dr, dc = DIRS[direction]
    return coords + np.array([dr, dc], dtype=int) * int(steps)


def _swept_cells(coords: np.ndarray, direction: str, steps: int) -> set[tuple[int, int]]:
    own = {tuple(map(int, cell)) for cell in coords.tolist()}
    swept: set[tuple[int, int]] = set()
    for step in range(1, steps + 1):
        shifted = _translate(coords, direction, step)
        for r, c in shifted:
            pos = (int(r), int(c))
            if pos not in own:
                swept.add(pos)
    return swept


def _visible_histogram_cells(
    shape: tuple[int, int],
    center: tuple[int, int],
    counts: dict[str, int],
) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for name in DIR_ORDER:
        dr, dc = DIRS[name]
        cells: list[tuple[int, int]] = []
        for step in range(1, counts[name] + 1):
            rr = center[0] + dr * step
            cc = center[1] + dc * step
            assert _in_bounds(shape, rr, cc)
            cells.append((rr, cc))
        out[name] = cells
    return out


def _analyze_input(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict[str, int]]:
    comps = _components(grid)
    marker_comps = [comp for comp in comps if int(comp["color"]) == MARKER]
    assert len(marker_comps) == 1, "expected one maroon marker component"
    marker = marker_comps[0]["coords"]
    assert 5 <= int(marker.shape[0]) <= 9, "marker size must stay within the described range"

    objects: list[dict] = []
    counts = {name: 0 for name in DIR_ORDER}
    for comp in comps:
        color = int(comp["color"])
        if color == MARKER:
            continue
        assert color in OBJECT_COLORS
        coords = comp["coords"]
        assert 2 <= int(coords.shape[0]) <= 5
        direction = _object_direction(grid, coords)
        counts[direction] += 1
        steps = _steps_to_edge(grid.shape, coords, direction)
        assert steps > 0
        objects.append(
            {
                "color": color,
                "coords": coords,
                "direction": direction,
                "steps": steps,
                "shifted": _translate(coords, direction, steps),
                "swept": _swept_cells(coords, direction, steps),
            }
        )

    assert 4 <= len(objects) <= 8
    return marker, objects, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push marker-touching objects outward and draw a central histogram of the push directions.

    Input structure:
    - The background is black, and a single maroon 4-connected cross or T-like marker sits near the center.
    - Several small monochromatic 4-connected objects in colors 1-4 surround that marker.
    - Each moving object touches the maroon marker on exactly one cardinal side, which determines whether it lies above, below, left, or right of the marker.

    Rule:
    - For each non-maroon object, infer the outward push direction as the direction opposite its maroon contact side.
    - Erase the object from its starting cells, translate it straight outward until it first touches the matching grid edge, and keep the maroon marker fixed.
    - Count how many objects move up, down, left, and right; then, from the rounded centroid of the maroon marker, draw a gray bar of that length in each of those four directions.
    - If a histogram pixel would overlap the maroon marker, preserve the maroon pixel.

    Key insight:
    - The same local contact-side relation both launches every object away from the marker and contributes one directional vote to the central compass histogram.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert _get_bg(g) == BG

    marker, objects, counts = _analyze_input(g)
    out = g.copy()
    out[g != MARKER] = BG

    occupied_after: set[tuple[int, int]] = set()
    for obj in objects:
        shifted = obj["shifted"]
        for r, c in shifted:
            pos = (int(r), int(c))
            assert pos not in occupied_after
            assert int(out[pos]) == BG
            occupied_after.add(pos)
        out[shifted[:, 0], shifted[:, 1]] = int(obj["color"])

    center = _round_half_up(marker)
    marker_set = {tuple(cell) for cell in marker.tolist()}
    for cells in _visible_histogram_cells(g.shape, center, counts).values():
        for r, c in cells:
            pos = (int(r), int(c))
            assert pos in marker_set or int(out[pos]) == BG
            if pos not in marker_set:
                out[pos] = BAR

    return out


def _rotate_up(proto: tuple[tuple[int, int], ...], direction: str) -> tuple[tuple[int, int], ...]:
    if direction == "up":
        return proto
    if direction == "down":
        return tuple((-r, -c) for r, c in proto)
    if direction == "left":
        return tuple((c, r) for r, c in proto)
    return tuple((-c, -r) for r, c in proto)


def _marker_grid(size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((size, size), dtype=int)
    template = MARKER_TEMPLATES[rng.randrange(len(MARKER_TEMPLATES))]
    h, w = template.shape
    base_r = size // 2 - h // 2 + rng.randint(-1, 1)
    base_c = size // 2 - w // 2 + rng.randint(-1, 1)
    base_r = max(2, min(size - h - 3, base_r))
    base_c = max(2, min(size - w - 3, base_c))
    cells = np.argwhere(template == 1)
    cells = cells + np.array([base_r, base_c], dtype=int)
    grid[cells[:, 0], cells[:, 1]] = MARKER
    return grid, cells


def _side_anchors(marker_cells: np.ndarray, shape: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    marker_set = {tuple(cell) for cell in marker_cells.tolist()}
    out = {name: [] for name in DIR_ORDER}
    for mr, mc in marker_set:
        for name, (dr, dc) in DIRS.items():
            nr = mr + dr
            nc = mc + dc
            if _in_bounds(shape, nr, nc) and (nr, nc) not in marker_set:
                out[name].append((nr, nc))
    for name in DIR_ORDER:
        out[name] = sorted(set(out[name]))
    return out


def _candidate_from_anchor(
    anchor: tuple[int, int],
    direction: str,
    proto: tuple[tuple[int, int], ...],
    marker_set: set[tuple[int, int]],
    shape: tuple[int, int],
) -> dict | None:
    coords = np.array([(anchor[0] + dr, anchor[1] + dc) for dr, dc in proto], dtype=int)
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        return None
    if not np.all((0 <= coords[:, 0]) & (coords[:, 0] < shape[0]) & (0 <= coords[:, 1]) & (coords[:, 1] < shape[1])):
        return None
    if any((int(r), int(c)) in marker_set for r, c in coords):
        return None
    if _direction_from_marker_set(marker_set, shape, coords) != direction:
        return None

    steps = _steps_to_edge(shape, coords, direction)
    if steps <= 0:
        return None
    shifted = _translate(coords, direction, steps)
    if any((int(r), int(c)) in marker_set for r, c in shifted):
        return None

    return {
        "coords": coords,
        "direction": direction,
        "steps": steps,
        "shifted": shifted,
        "swept": _swept_cells(coords, direction, steps),
    }


def _candidates_by_direction(marker_cells: np.ndarray, shape: tuple[int, int]) -> dict[str, list[dict]]:
    marker_set = {tuple(cell) for cell in marker_cells.tolist()}
    anchors = _side_anchors(marker_cells, shape)
    out: dict[str, list[dict]] = {name: [] for name in DIR_ORDER}
    for direction in DIR_ORDER:
        seen: set[tuple[tuple[int, int], ...]] = set()
        for anchor in anchors[direction]:
            for base_proto in UP_PROTOTYPES:
                cand = _candidate_from_anchor(anchor, direction, _rotate_up(base_proto, direction), marker_set, shape)
                if cand is None:
                    continue
                key = tuple(sorted((int(r), int(c)) for r, c in cand["coords"].tolist()))
                if key in seen:
                    continue
                seen.add(key)
                out[direction].append(cand)
    return out


def _halo4(cells: np.ndarray) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        out.add((rr, cc))
        for nr, nc in _neighbors4(rr, cc):
            out.add((nr, nc))
    return out


def _choose_counts(
    anchors: dict[str, list[tuple[int, int]]],
    center: tuple[int, int],
    size: int,
    rng: random.Random,
) -> dict[str, int] | None:
    caps = {}
    for name in DIR_ORDER:
        if name == "up":
            bar_cap = center[0]
        elif name == "down":
            bar_cap = size - 1 - center[0]
        elif name == "left":
            bar_cap = center[1]
        else:
            bar_cap = size - 1 - center[1]
        caps[name] = min(3, len(anchors[name]), bar_cap)
        if caps[name] < 1:
            return None

    counts = {name: 1 for name in DIR_ORDER}
    extras_cap = sum(caps[name] - 1 for name in DIR_ORDER)
    extra_choices = [0, 1, 1, 2, 2, 3, 4]
    extra = min(rng.choice(extra_choices), extras_cap)
    pool = [name for name in DIR_ORDER for _ in range(caps[name] - 1)]
    rng.shuffle(pool)
    for name in pool[:extra]:
        counts[name] += 1
    return counts


def _choose_placements(
    candidates: dict[str, list[dict]],
    counts: dict[str, int],
    marker_set: set[tuple[int, int]],
    rng: random.Random,
) -> list[dict] | None:
    slots = [name for name in DIR_ORDER for _ in range(counts[name])]
    slots.sort(key=lambda name: len(candidates[name]))
    shuffled = {name: rng.sample(candidates[name], len(candidates[name])) for name in DIR_ORDER}

    def search(
        index: int,
        input_halo: set[tuple[int, int]],
        output_halo: set[tuple[int, int]],
        occupied_input: set[tuple[int, int]],
        occupied_output: set[tuple[int, int]],
        reserved_swept: set[tuple[int, int]],
        chosen: list[dict],
    ) -> list[dict] | None:
        if index == len(slots):
            return chosen.copy()

        direction = slots[index]
        for cand in shuffled[direction]:
            coords = {(int(r), int(c)) for r, c in cand["coords"].tolist()}
            shifted = {(int(r), int(c)) for r, c in cand["shifted"].tolist()}
            if not coords.isdisjoint(input_halo):
                continue
            if not shifted.isdisjoint(output_halo):
                continue
            if not cand["swept"].isdisjoint(occupied_input):
                continue
            if not cand["swept"].isdisjoint(reserved_swept):
                continue
            if not shifted.isdisjoint(marker_set):
                continue

            found = search(
                index + 1,
                input_halo | _halo4(cand["coords"]),
                output_halo | _halo4(cand["shifted"]),
                occupied_input | coords,
                occupied_output | shifted,
                reserved_swept | cand["swept"],
                chosen + [cand],
            )
            if found is not None:
                return found
        return None

    return search(0, set(), set(), marker_set.copy(), marker_set.copy(), set(), [])


def _assign_colors(count: int, rng: random.Random) -> list[int]:
    color_count = rng.randint(2, min(4, count))
    palette = rng.sample(list(OBJECT_COLORS), color_count)
    colors = palette.copy()
    while len(colors) < count:
        colors.append(rng.choice(palette))
    rng.shuffle(colors)
    return colors


def _render_input(
    size: int,
    marker_cells: np.ndarray,
    placements: list[dict],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    grid[marker_cells[:, 0], marker_cells[:, 1]] = MARKER
    for placement, color in zip(placements, colors):
        coords = placement["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(color)
    return grid


def _expected_output(
    size: int,
    marker_cells: np.ndarray,
    placements: list[dict],
    colors: list[int],
) -> np.ndarray:
    out = np.zeros((size, size), dtype=int)
    out[marker_cells[:, 0], marker_cells[:, 1]] = MARKER
    counts = {name: 0 for name in DIR_ORDER}
    for placement, color in zip(placements, colors):
        counts[placement["direction"]] += 1
        shifted = placement["shifted"]
        out[shifted[:, 0], shifted[:, 1]] = int(color)

    center = _round_half_up(marker_cells)
    marker_set = {tuple(cell) for cell in marker_cells.tolist()}
    for cells in _visible_histogram_cells((size, size), center, counts).values():
        for r, c in cells:
            if (r, c) not in marker_set:
                out[r, c] = BAR
    return out


def _validate_scene(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        marker, objects, counts = _analyze_input(inp)
    except AssertionError:
        return False

    if any(counts[name] < 1 for name in DIR_ORDER):
        return False

    distinct_colors = {int(obj["color"]) for obj in objects}
    if not (2 <= len(distinct_colors) <= 4):
        return False

    marker_set = {tuple(cell) for cell in marker.tolist()}
    shifted_occupied: set[tuple[int, int]] = set()
    for obj in objects:
        shifted = {tuple(map(int, cell)) for cell in obj["shifted"].tolist()}
        if shifted & shifted_occupied:
            return False
        shifted_occupied |= shifted

    center = _round_half_up(marker)
    visible = 0
    for cells in _visible_histogram_cells(inp.shape, center, counts).values():
        for cell in cells:
            if cell in shifted_occupied:
                return False
            if cell not in marker_set:
                visible += 1
    if visible < 2:
        return False

    try:
        pred = main(inp)
    except AssertionError:
        return False
    return np.array_equal(pred, out)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.randint(15, 19)
    marker_grid, marker_cells = _marker_grid(size, rng)
    center = _round_half_up(marker_cells)
    anchors = _side_anchors(marker_cells, marker_grid.shape)
    counts = _choose_counts(anchors, center, size, rng)
    if counts is None:
        return None

    candidates = _candidates_by_direction(marker_cells, marker_grid.shape)
    if any(len(candidates[name]) < counts[name] for name in DIR_ORDER):
        return None

    marker_set = {tuple(cell) for cell in marker_cells.tolist()}
    placements = _choose_placements(candidates, counts, marker_set, rng)
    if placements is None:
        return None

    rng.shuffle(placements)
    colors = _assign_colors(len(placements), rng)
    inp = _render_input(size, marker_cells, placements, colors)
    out = _expected_output(size, marker_cells, placements, colors)
    if not _validate_scene(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
