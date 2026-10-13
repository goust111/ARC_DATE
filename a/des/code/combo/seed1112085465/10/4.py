from __future__ import annotations

import random

import numpy as np


BG = 8
SEP = 5
MARKER = 2
PATTERN_COLORS = (0, 1, 3, 4, 6, 7, 9)
DIRECTIONS = ("up", "down", "left", "right")
STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

PANEL_SIZE_RANGE = (4, 7)
PATTERN_AREA_RANGE = (2, 8)
GEN_ATTEMPTS = 300
COLLISION_PROB = 0.5
LOSS_PROB = 0.5
SHAPE_GROW_ATTEMPTS = 80


def _panel_origin(panel_r: int, panel_c: int, panel_h: int, panel_w: int) -> tuple[int, int]:
    return panel_r * (panel_h + 1), panel_c * (panel_w + 1)


def _find_layout(g: np.ndarray) -> tuple[int, int, int]:
    h, w = g.shape
    candidates: list[tuple[int, int, int]] = []

    for color in map(int, np.unique(g)):
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if len(rows) != 2 or len(cols) != 2:
            continue
        panel_h = rows[0]
        panel_w = cols[0]
        if panel_h < 2 or panel_w < 2:
            continue
        if rows[1] != 2 * panel_h + 1 or cols[1] != 2 * panel_w + 1:
            continue
        if h != 3 * panel_h + 2 or w != 3 * panel_w + 2:
            continue
        candidates.append((color, panel_h, panel_w))

    assert len(candidates) == 1, "expected one 3x3 separator layout"
    return candidates[0]


def _marker_direction(marker_r: int, marker_c: int, panel_h: int, panel_w: int) -> str:
    distances = {
        "up": marker_r,
        "down": panel_h - 1 - marker_r,
        "left": marker_c,
        "right": panel_w - 1 - marker_c,
    }
    best = min(distances.values())
    winners = [direction for direction, dist in distances.items() if dist == best]
    assert len(winners) == 1, "marker must be closest to exactly one edge"
    return winners[0]


def _parse_panels(g: np.ndarray) -> tuple[int, int, list[dict]]:
    sep_color, panel_h, panel_w = _find_layout(g)
    panels: list[dict] = []
    for panel_r in range(3):
        for panel_c in range(3):
            r0, c0 = _panel_origin(panel_r, panel_c, panel_h, panel_w)
            panel = g[r0 : r0 + panel_h, c0 : c0 + panel_w]
            markers = np.argwhere(panel == MARKER)
            assert markers.shape[0] == 1, "each panel must contain one red marker"
            marker_r, marker_c = map(int, markers[0])
            direction = _marker_direction(marker_r, marker_c, panel_h, panel_w)

            pattern_mask = (panel != BG) & (panel != MARKER)
            coords = np.argwhere(pattern_mask)
            assert coords.shape[0] > 0, "each panel must contain a non-empty pattern"
            colors = panel[pattern_mask].astype(int, copy=True)
            panels.append(
                {
                    "panel_r": panel_r,
                    "panel_c": panel_c,
                    "direction": direction,
                    "coords": coords.astype(int, copy=True),
                    "colors": colors,
                }
            )

    assert np.all(g[_separator_rows(panel_h), :] == sep_color)
    assert np.all(g[:, _separator_cols(panel_w)] == sep_color)
    return panel_h, panel_w, panels


def _separator_rows(panel_h: int) -> list[int]:
    return [panel_h, 2 * panel_h + 1]


def _separator_cols(panel_w: int) -> list[int]:
    return [panel_w, 2 * panel_w + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each panel pattern into the adjacent panel indicated by its red edge marker.

    The grid is a 3x3 array of equal panels on an azure background, separated by full
    grey rows and columns. Each panel contains one non-red pattern and one red marker
    closest to a single edge. The marker gives the destination neighbor panel. Source
    patterns are erased, markers and separators stay, and incoming patterns are pasted
    in row-major source order so later ones overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panel_h, panel_w, panels = _parse_panels(g)
    out = g.copy()

    for panel in panels:
        panel_r = int(panel["panel_r"])
        panel_c = int(panel["panel_c"])
        r0, c0 = _panel_origin(panel_r, panel_c, panel_h, panel_w)
        coords = np.asarray(panel["coords"], dtype=int)
        out[r0 + coords[:, 0], c0 + coords[:, 1]] = BG

    for panel in panels:
        panel_r = int(panel["panel_r"])
        panel_c = int(panel["panel_c"])
        dr, dc = STEP[str(panel["direction"])]
        target_r = panel_r + dr
        target_c = panel_c + dc
        if not (0 <= target_r < 3 and 0 <= target_c < 3):
            continue

        r0, c0 = _panel_origin(target_r, target_c, panel_h, panel_w)
        coords = np.asarray(panel["coords"], dtype=int)
        colors = np.asarray(panel["colors"], dtype=int)
        for (local_r, local_c), color in zip(coords, colors):
            out[r0 + int(local_r), c0 + int(local_c)] = int(color)

    return out


def _neighbors_pushing_into(target_r: int, target_c: int) -> list[tuple[int, int, str]]:
    options: list[tuple[int, int, str]] = []
    if target_r > 0:
        options.append((target_r - 1, target_c, "down"))
    if target_r < 2:
        options.append((target_r + 1, target_c, "up"))
    if target_c > 0:
        options.append((target_r, target_c - 1, "right"))
    if target_c < 2:
        options.append((target_r, target_c + 1, "left"))
    return options


def _outward_directions(panel_r: int, panel_c: int) -> list[str]:
    options: list[str] = []
    if panel_r == 0:
        options.append("up")
    if panel_r == 2:
        options.append("down")
    if panel_c == 0:
        options.append("left")
    if panel_c == 2:
        options.append("right")
    return options


def _sample_marker(direction: str, panel_h: int, panel_w: int) -> tuple[int, int]:
    if direction == "up":
        return 0, random.randint(1, panel_w - 2)
    if direction == "down":
        return panel_h - 1, random.randint(1, panel_w - 2)
    if direction == "left":
        return random.randint(1, panel_h - 2), 0
    return random.randint(1, panel_h - 2), panel_w - 1


def _grow_shape(
    panel_h: int,
    panel_w: int,
    area: int,
    forbidden: set[tuple[int, int]],
    required: tuple[int, int] | None,
) -> set[tuple[int, int]] | None:
    free = {(r, c) for r in range(panel_h) for c in range(panel_w)} - forbidden
    if required is not None and required not in free:
        return None
    if len(free) < area:
        return None

    start = required if required is not None else random.choice(list(free))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in STEP.values():
            rr, cc = r + dr, c + dc
            if (rr, cc) in free and (rr, cc) not in cells:
                frontier.add((rr, cc))

    add_frontier(*start)
    while len(cells) < area:
        if not frontier:
            return None
        nxt = random.choice(list(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        add_frontier(*nxt)

    return cells


def _sample_pattern_cells(
    panel_h: int,
    panel_w: int,
    forbidden: set[tuple[int, int]],
    required: tuple[int, int] | None,
) -> set[tuple[int, int]] | None:
    max_area = min(PATTERN_AREA_RANGE[1], panel_h * panel_w - len(forbidden))
    min_area = min(PATTERN_AREA_RANGE[0], max_area)
    if max_area < 1 or min_area < 1:
        return None
    for _ in range(SHAPE_GROW_ATTEMPTS):
        area = random.randint(min_area, max_area)
        cells = _grow_shape(panel_h, panel_w, area, forbidden, required)
        if cells is not None:
            return cells
    return None


def _assign_colors(
    cells: set[tuple[int, int]],
    forced: dict[tuple[int, int], int] | None = None,
) -> dict[tuple[int, int], int]:
    ordered = list(cells)
    random.shuffle(ordered)
    palette_size = random.randint(1, min(3, len(ordered)))
    palette = random.sample(PATTERN_COLORS, palette_size)

    colors: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(ordered):
        if idx < palette_size:
            colors[cell] = palette[idx]
        else:
            colors[cell] = random.choice(palette)

    if forced is not None:
        for cell, color in forced.items():
            assert cell in colors
            colors[cell] = int(color)
    return colors


def _render_input(
    panel_h: int,
    panel_w: int,
    markers: dict[tuple[int, int], tuple[int, int]],
    patterns: dict[tuple[int, int], dict[tuple[int, int], int]],
) -> np.ndarray:
    h = 3 * panel_h + 2
    w = 3 * panel_w + 2
    grid = np.full((h, w), BG, dtype=int)
    grid[_separator_rows(panel_h), :] = SEP
    grid[:, _separator_cols(panel_w)] = SEP

    for panel_r in range(3):
        for panel_c in range(3):
            r0, c0 = _panel_origin(panel_r, panel_c, panel_h, panel_w)
            marker_r, marker_c = markers[(panel_r, panel_c)]
            grid[r0 + marker_r, c0 + marker_c] = MARKER
            for (local_r, local_c), color in patterns[(panel_r, panel_c)].items():
                grid[r0 + local_r, c0 + local_c] = int(color)

    return grid


def _render_expected_output(
    panel_h: int,
    panel_w: int,
    directions: dict[tuple[int, int], str],
    markers: dict[tuple[int, int], tuple[int, int]],
    patterns: dict[tuple[int, int], dict[tuple[int, int], int]],
) -> np.ndarray:
    out = _render_input(panel_h, panel_w, markers, patterns)

    for panel_r in range(3):
        for panel_c in range(3):
            r0, c0 = _panel_origin(panel_r, panel_c, panel_h, panel_w)
            for local_r, local_c in patterns[(panel_r, panel_c)]:
                out[r0 + local_r, c0 + local_c] = BG

    for panel_r in range(3):
        for panel_c in range(3):
            dr, dc = STEP[directions[(panel_r, panel_c)]]
            target_r = panel_r + dr
            target_c = panel_c + dc
            if not (0 <= target_r < 3 and 0 <= target_c < 3):
                continue
            r0, c0 = _panel_origin(target_r, target_c, panel_h, panel_w)
            for (local_r, local_c), color in patterns[(panel_r, panel_c)].items():
                out[r0 + local_r, c0 + local_c] = int(color)

    return out


def _sample_directions() -> tuple[dict[tuple[int, int], str], dict | None, dict | None]:
    directions = {
        (panel_r, panel_c): random.choice(DIRECTIONS)
        for panel_r in range(3)
        for panel_c in range(3)
    }

    collision_plan = None
    if random.random() < COLLISION_PROB:
        target = (random.randint(0, 2), random.randint(0, 2))
        choices = _neighbors_pushing_into(*target)
        chosen = random.sample(choices, 2)
        sources = []
        for panel_r, panel_c, direction in chosen:
            directions[(panel_r, panel_c)] = direction
            sources.append((panel_r, panel_c))
        collision_plan = {"target": target, "sources": tuple(sources)}

    loss_plan = None
    if random.random() < LOSS_PROB:
        blocked = set() if collision_plan is None else set(collision_plan["sources"])
        candidates = [
            (panel_r, panel_c, direction)
            for panel_r in range(3)
            for panel_c in range(3)
            if (panel_r, panel_c) not in blocked
            for direction in _outward_directions(panel_r, panel_c)
        ]
        if candidates:
            panel_r, panel_c, direction = random.choice(candidates)
            directions[(panel_r, panel_c)] = direction
            loss_plan = {"source": (panel_r, panel_c)}

    return directions, collision_plan, loss_plan


def _choose_overlap_cell(
    panel_h: int,
    panel_w: int,
    markers: dict[tuple[int, int], tuple[int, int]],
    collision_plan: dict,
) -> tuple[int, int] | None:
    target = tuple(collision_plan["target"])
    sources = [tuple(source) for source in collision_plan["sources"]]
    preferred = [
        (r, c)
        for r in range(1, panel_h - 1)
        for c in range(1, panel_w - 1)
        if (r, c) != markers[target] and all((r, c) != markers[source] for source in sources)
    ]
    fallback = [
        (r, c)
        for r in range(panel_h)
        for c in range(panel_w)
        if (r, c) != markers[target] and all((r, c) != markers[source] for source in sources)
    ]
    choices = preferred if preferred else fallback
    if not choices:
        return None
    return random.choice(choices)


def _build_sample() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    directions, collision_plan, _loss_plan = _sample_directions()

    markers = {
        (panel_r, panel_c): _sample_marker(directions[(panel_r, panel_c)], panel_h, panel_w)
        for panel_r in range(3)
        for panel_c in range(3)
    }

    overlap_cell = None
    overlap_colors: dict[tuple[int, int], int] = {}
    if collision_plan is not None:
        overlap_cell = _choose_overlap_cell(panel_h, panel_w, markers, collision_plan)
        if overlap_cell is None:
            return None
        forced_colors = random.sample(PATTERN_COLORS, len(collision_plan["sources"]))
        overlap_colors = {
            tuple(source): int(color)
            for source, color in zip(collision_plan["sources"], forced_colors)
        }

    patterns: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    for panel_r in range(3):
        for panel_c in range(3):
            source = (panel_r, panel_c)
            forbidden = {markers[source]}
            dr, dc = STEP[directions[source]]
            target = (panel_r + dr, panel_c + dc)
            if 0 <= target[0] < 3 and 0 <= target[1] < 3:
                forbidden.add(markers[target])

            required = None
            forced = None
            if collision_plan is not None and source in collision_plan["sources"]:
                required = overlap_cell
                forced = {overlap_cell: overlap_colors[source]}

            cells = _sample_pattern_cells(panel_h, panel_w, forbidden, required)
            if cells is None:
                return None
            patterns[source] = _assign_colors(cells, forced)

    inp = _render_input(panel_h, panel_w, markers, patterns)
    expected = _render_expected_output(panel_h, panel_w, directions, markers, patterns)

    if collision_plan is not None:
        target = tuple(collision_plan["target"])
        later_source = max(collision_plan["sources"])
        target_r0, target_c0 = _panel_origin(target[0], target[1], panel_h, panel_w)
        rr = target_r0 + overlap_cell[0]
        cc = target_c0 + overlap_cell[1]
        if expected[rr, cc] != overlap_colors[later_source]:
            return None

    if int(np.count_nonzero(inp == MARKER)) != 9 or int(np.count_nonzero(expected == MARKER)) != 9:
        return None

    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _build_sample()
        if sample is not None:
            inp, _out = sample
            return inp
    raise RuntimeError("failed to generate a valid panel-push puzzle input")
