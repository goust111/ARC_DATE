from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]

UP: Direction = (-1, 0)
RIGHT: Direction = (0, 1)
DOWN: Direction = (1, 0)
LEFT: Direction = (0, -1)
DIRECTIONS = (UP, RIGHT, DOWN, LEFT)
SIDE_DIRECTION = {"top": UP, "right": RIGHT, "bottom": DOWN, "left": LEFT}
INWARD = {"top": DOWN, "right": LEFT, "bottom": UP, "left": RIGHT}


@dataclass(frozen=True)
class Parsed:
    background: int
    traveler_color: int
    traveler: Coord
    initial_direction: Direction
    color_direction: dict[int, Direction]


def _parse(grid: np.ndarray) -> Parsed:
    """Read the four structural border colors and the unique traveler."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    assert len(set(corners)) == 1, "the four corners must show the background"
    bg = corners[0]

    side_values = {
        "top": g[0, 1:-1],
        "right": g[1:-1, -1],
        "bottom": g[-1, 1:-1],
        "left": g[1:-1, 0],
    }
    side_colors: dict[str, int] = {}
    for side, values in side_values.items():
        assert values.size and np.all(values == values.flat[0]), f"{side} border is not pure"
        side_colors[side] = int(values.flat[0])
    assert len(set(side_colors.values())) == 4
    assert bg not in side_colors.values()

    frame_colors = set(side_colors.values())
    interior = g[1:-1, 1:-1]
    unusual = ~np.isin(interior, [bg, *frame_colors])
    coords = np.argwhere(unusual)
    assert len(coords) == 1, "expected exactly one uniquely colored traveler"
    tr = (int(coords[0, 0]) + 1, int(coords[0, 1]) + 1)
    traveler_color = int(g[tr])

    touching: list[str] = []
    r, c = tr
    if r == 1:
        touching.append("top")
    if r == h - 2:
        touching.append("bottom")
    if c == 1:
        touching.append("left")
    if c == w - 2:
        touching.append("right")
    assert len(touching) == 1, "traveler must be just inside exactly one frame side"

    color_direction = {
        side_colors[side]: SIDE_DIRECTION[side]
        for side in ("top", "right", "bottom", "left")
    }
    return Parsed(bg, traveler_color, tr, INWARD[touching[0]], color_direction)


def _trace(grid: np.ndarray, parsed: Parsed) -> tuple[list[Coord], list[Coord], list[Direction]]:
    """Trace the one-shot instructions without modifying the input."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r, c = parsed.traveler
    dr, dc = parsed.initial_direction
    path: list[Coord] = [(r, c)]
    hits: list[Coord] = []
    outgoing: list[Direction] = []
    used: set[Coord] = set()

    # With k instructions there are at most k+1 straight finite segments.
    sign_count = int(np.isin(g[1:-1, 1:-1], list(parsed.color_direction)).sum())
    limit = (sign_count + 1) * (h + w) + 1
    for _ in range(limit):
        nr, nc = r + dr, c + dc
        if nr in (0, h - 1) or nc in (0, w - 1):
            return path, hits, outgoing
        r, c = nr, nc
        path.append((r, c))
        color = int(g[r, c])
        if color in parsed.color_direction and (r, c) not in used:
            used.add((r, c))
            hits.append((r, c))
            dr, dc = parsed.color_direction[color]
            outgoing.append((dr, dc))
    raise AssertionError("path did not reach the frame")


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the traveler through one-shot colored signposts and paint its complete route.

    Input structure:
    - A one-cell frame has four distinct uniform side colors and background-colored corners.
    - Interior signpost pixels use frame-side colors; one differently colored traveler sits just inside exactly one side.

    Rule:
    - The traveler initially moves perpendicularly inward from its adjacent frame side.
    - On first reaching a signpost, it turns toward the frame side of the same color; later visits to that signpost do not turn it.
    - Stop immediately before entering the frame and recolor every visited interior cell with the traveler color, preserving missed signs and the frame.

    Key insight:
    - Frame colors form a visible direction legend, while hit interior pixels are consumable routing commands and all missed signposts are distractors.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse(g)
    path, _hits, _outgoing = _trace(g, parsed)
    out = g.copy()
    for r, c in path:
        out[r, c] = parsed.traveler_color
    return out


def _inside(shape: tuple[int, int], p: Coord) -> bool:
    h, w = shape
    return 1 <= p[0] < h - 1 and 1 <= p[1] < w - 1


def _ray_cells(shape: tuple[int, int], start: Coord, direction: Direction) -> list[Coord]:
    cells: list[Coord] = []
    r, c = start
    dr, dc = direction
    while True:
        r, c = r + dr, c + dc
        if not _inside(shape, (r, c)):
            return cells
        cells.append((r, c))


def _plan_route(
    shape: tuple[int, int],
    start: Coord,
    initial: Direction,
    n_signs: int,
    require_same: bool,
    require_reverse: bool,
) -> tuple[list[Coord], list[Direction], list[Coord]] | None:
    """Construct commands in temporal order; no future command lies on an earlier segment."""
    current = start
    direction = initial
    visited: list[Coord] = [start]
    visited_set = {start}
    signs: list[Coord] = []
    outgoing: list[Direction] = []
    same_left = require_same
    reverse_left = require_reverse

    for i in range(n_signs):
        ray = _ray_cells(shape, current, direction)
        candidates = []
        for j, p in enumerate(ray):
            # New commands must not have been crossed earlier and should leave a
            # visibly nonzero incoming segment.
            if p not in visited_set and j >= 1:
                candidates.append((j, p))
        if not candidates:
            return None

        # Avoid always selecting the nearest possible command.
        j, sign = random.choice(candidates[: min(len(candidates), 7)])
        segment = ray[: j + 1]
        visited.extend(segment)
        visited_set.update(segment)
        signs.append(sign)

        remaining = n_signs - i - 1
        choices = list(DIRECTIONS)
        random.shuffle(choices)
        if same_left and remaining <= 1:
            choices = [direction] + [d for d in choices if d != direction]
        elif reverse_left and remaining <= 2:
            rev = (-direction[0], -direction[1])
            choices = [rev] + [d for d in choices if d != rev]

        picked: Direction | None = None
        for new_direction in choices:
            if i < n_signs - 1:
                future = _ray_cells(shape, sign, new_direction)
                if not any(p not in visited_set for p in future[1:]):
                    continue
            picked = new_direction
            break
        if picked is None:
            return None
        if picked == direction:
            same_left = False
        if picked == (-direction[0], -direction[1]):
            reverse_left = False
        outgoing.append(picked)
        current, direction = sign, picked

    if same_left or reverse_left:
        return None
    visited.extend(_ray_cells(shape, current, direction))
    return signs, outgoing, visited


def _render_candidate() -> tuple[np.ndarray, list[Coord], list[Direction], list[Coord]] | None:
    h = random.randint(13, 22)
    w = random.randint(13, 22)
    shape = (h, w)
    colors = random.sample(range(10), 6)
    bg, traveler_color = colors[:2]
    frame_colors = colors[2:]
    sides = ("top", "right", "bottom", "left")
    side_color = dict(zip(sides, random.sample(frame_colors, 4)))

    side = random.choice(sides)
    if side in ("top", "bottom"):
        c = random.randint(3, w - 4)
        start = (1 if side == "top" else h - 2, c)
    else:
        r = random.randint(3, h - 4)
        start = (r, 1 if side == "left" else w - 2)
    initial = INWARD[side]

    n_signs = random.randint(4, 6)
    # Same-direction and reverse commands are deliberately oversampled so a
    # small ARC training set is likely to expose both stated edge cases.
    require_same = random.random() < 0.45
    require_reverse = random.random() < 0.55
    plan = _plan_route(shape, start, initial, n_signs, require_same, require_reverse)
    if plan is None:
        return None
    signs, outgoing, planned_path = plan

    # At least two commands must make genuine perpendicular turns. This keeps
    # border decoding indispensable rather than reducing the task to one ray.
    incoming = [initial, *outgoing[:-1]]
    perpendicular = sum(a[0] * b[0] + a[1] * b[1] == 0 for a, b in zip(incoming, outgoing))
    # Every generated example exercises all four visible color-to-side mappings;
    # thus even a very small training set cannot leave one legend entry untested.
    if perpendicular < 2 or len(set(outgoing)) < 4:
        return None
    if len(set(planned_path)) < 9 or len(set(planned_path)) > int(0.48 * (h - 2) * (w - 2)):
        return None

    g = np.full(shape, bg, dtype=int)
    g[0, 1:-1] = side_color["top"]
    g[1:-1, -1] = side_color["right"]
    g[-1, 1:-1] = side_color["bottom"]
    g[1:-1, 0] = side_color["left"]
    direction_color = {SIDE_DIRECTION[s]: side_color[s] for s in sides}
    for p, direction in zip(signs, outgoing):
        g[p] = direction_color[direction]
    g[start] = traveler_color

    path_set = set(planned_path)
    forbidden = set(path_set)
    for r, c in path_set:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((r + dr, c + dc))
    options = [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if (r, c) not in forbidden and g[r, c] == bg
    ]
    random.shuffle(options)
    distractors: list[Coord] = []
    for p in options:
        if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) < 2 for q in distractors):
            continue
        g[p] = random.choice(frame_colors)
        distractors.append(p)
        if len(distractors) == random.randint(4, 7):
            break
    if len(distractors) < 4:
        return None
    return g, signs, outgoing, distractors


def _semantic_check(
    grid: np.ndarray,
    expected_signs: list[Coord],
    expected_outgoing: list[Direction],
    distractors: list[Coord],
) -> bool:
    """Check the described semantics, including non-degeneracy, not just closure."""
    try:
        parsed = _parse(grid)
        path, hits, outgoing = _trace(grid, parsed)
        out = main(grid)
    except (AssertionError, ValueError, IndexError):
        return False
    if hits != expected_signs or outgoing != expected_outgoing:
        return False
    if len(hits) < 4 or len(set(outgoing)) != 4 or len(distractors) < 4 or set(path) & set(distractors):
        return False
    if any(int(out[p]) != int(grid[p]) for p in distractors):
        return False
    changed = set(map(tuple, np.argwhere(out != grid)))
    expected_changed = {p for p in path if int(grid[p]) != parsed.traveler_color}
    if changed != expected_changed:
        return False
    # The border legend and all corners are immutable.
    if not (
        np.array_equal(out[0, :], grid[0, :])
        and np.array_equal(out[-1, :], grid[-1, :])
        and np.array_equal(out[:, 0], grid[:, 0])
        and np.array_equal(out[:, -1], grid[:, -1])
    ):
        return False
    incoming = [parsed.initial_direction, *outgoing[:-1]]
    return sum(a[0] * b[0] + a[1] * b[1] == 0 for a, b in zip(incoming, outgoing)) >= 2


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked signpost-routing instance."""
    for _ in range(1500):
        made = _render_candidate()
        if made is None:
            continue
        grid, signs, outgoing, distractors = made
        if _semantic_check(grid, signs, outgoing, distractors):
            return grid
    raise RuntimeError("failed to generate a valid signpost-routing instance")
