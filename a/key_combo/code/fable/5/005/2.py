from __future__ import annotations

from collections import Counter, deque
import hashlib
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_GRAY = 5
_DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PALETTE = tuple(range(10))
_MAX_GENERATION_ATTEMPTS = 500

# Body coordinates are relative to the differently colored nose at (0, 0).
# All templates face up, are four-connected, and include asymmetric variants so
# that the legend's exact shape cannot be replaced by a generic arrow.
_BODY_TEMPLATES: tuple[tuple[Coord, ...], ...] = (
    ((1, 0), (2, -1), (2, 0), (2, 1), (3, -1), (3, 1)),
    ((1, -1), (1, 0), (1, 1), (2, -1), (2, 0), (2, 1), (3, 0)),
    ((1, 0), (2, -1), (2, 0), (2, 1), (3, -1), (3, 0), (4, 0)),
    (
        (1, 0), (2, -1), (2, 0), (2, 1),
        (3, -2), (3, -1), (3, 0), (3, 1), (3, 2),
        (4, -1), (4, 1),
    ),
    ((1, 0), (1, 1), (2, -1), (2, 0), (2, 1), (3, -1), (3, 0), (4, -1)),
    ((1, 0), (2, 0), (2, 1), (3, -1), (3, 0), (3, 1), (4, -1), (4, 0)),
)

# Prevent independently generated examples in one run from collapsing to the
# same answer even if random choices happen to collide.
_SEEN_OUTPUTS: set[str] = set()
_DIRECTION_BAG: list[Coord] = []
_TEMPLATE_BAG: list[int] = []


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert ordered and (len(ordered) == 1 or ordered[0][1] > ordered[1][1]), (
        "background must be uniquely most common"
    )
    return int(ordered[0][0])


def _neighbors(shape: tuple[int, int], cell: Coord):
    h, w = shape
    r, c = cell
    for dr, dc in _DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[set[Coord]]:
    """Return four-connected components of a boolean mask in (row, col) coordinates."""
    unseen = set(map(tuple, np.argwhere(mask)))
    result: list[set[Coord]] = []
    while unseen:
        seed = unseen.pop()
        comp = {seed}
        queue: deque[Coord] = deque([seed])
        while queue:
            cell = queue.popleft()
            for nxt in _neighbors(mask.shape, cell):
                if nxt in unseen:
                    unseen.remove(nxt)
                    comp.add(nxt)
                    queue.append(nxt)
        result.append(comp)
    return result


def _bbox(cells: set[Coord]) -> BBox:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _gray_path(grid: np.ndarray) -> tuple[set[Coord], tuple[Coord, Coord]]:
    gray_components = _components(grid == _GRAY)
    assert len(gray_components) == 1, "expected exactly one four-connected gray path"
    path = gray_components[0]
    assert len(path) >= 4, "gray trace must contain several cells"
    degrees = {cell: sum(nxt in path for nxt in _neighbors(grid.shape, cell)) for cell in path}
    assert all(degree in (1, 2) for degree in degrees.values()), "gray trace must be a simple path"
    endpoints = tuple(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "gray trace must have exactly two endpoints"
    return path, (endpoints[0], endpoints[1])


def _find_legend(
    grid: np.ndarray, bg: int, path: set[Coord], endpoints: tuple[Coord, Coord]
) -> tuple[set[Coord], BBox, list[tuple[Coord, int]], Coord]:
    """Find the unique closed rectangular frame and read its two-color sprite."""
    candidates: list[tuple[set[Coord], BBox, list[tuple[Coord, int]], Coord]] = []
    colors = set(map(int, np.unique(grid))) - {bg, _GRAY}

    for color in colors:
        for comp in _components(grid == color):
            box = _bbox(comp)
            r0, c0, r1, c1 = box
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            border = _border_cells(box)
            if comp != border:
                continue

            touching = [
                endpoint
                for endpoint in endpoints
                if any(nxt in border for nxt in _neighbors(grid.shape, endpoint))
            ]
            if len(touching) != 1:
                continue
            # The start endpoint is the only trace cell allowed next to the frame.
            adjacent_trace = {
                cell for cell in path
                if any(nxt in border for nxt in _neighbors(grid.shape, cell))
            }
            if adjacent_trace != {touching[0]}:
                continue

            sprite: list[tuple[Coord, int]] = []
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    value = int(grid[r, c])
                    if value != bg:
                        sprite.append(((r, c), value))
            sprite_colors = Counter(color for _, color in sprite)
            if len(sprite_colors) != 2:
                continue
            singleton_colors = [color for color, count in sprite_colors.items() if count == 1]
            if len(singleton_colors) != 1 or len(sprite) < 5:
                continue
            nose_color = singleton_colors[0]
            nose = next(cell for cell, color in sprite if color == nose_color)
            body = {cell for cell, color in sprite if color != nose_color}
            # This visible local relation defines the legend's canonical "up".
            if nose[0] != min(r for (r, _), _color in sprite):
                continue
            if (nose[0] + 1, nose[1]) not in body:
                continue
            candidates.append((border, box, sprite, touching[0]))

    assert len(candidates) == 1, "expected one unambiguous framed sprite legend"
    return candidates[0]


def _rotate_offset(offset: Coord, direction: Coord) -> Coord:
    """Rotate an up-facing relative coordinate so up points in direction."""
    dr, dc = offset
    if direction == (-1, 0):       # up: identity
        return dr, dc
    if direction == (0, 1):        # clockwise
        return dc, -dr
    if direction == (1, 0):        # half-turn
        return -dr, -dc
    if direction == (0, -1):       # counter-clockwise
        return -dc, dr
    raise AssertionError("last path step must be cardinal")


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild the vanished legend sprite at the gray trace endpoint, oriented by the final step.

    Input structure:
    - A uniquely dominant background contains one closed monochromatic rectangular frame.
    - Inside the frame is a two-color up-facing sprite whose differently colored nose is a singleton.
    - One four-connected, non-branching gray trace has exactly one endpoint adjacent to the frame;
      its other endpoint hangs free in the scene.

    Rule:
    - Follow the gray trace from the frame-adjacent endpoint to its free endpoint.
    - Rotate every sprite pixel about its nose so the up-facing nose points in the direction of the
      trace's final step, then place the nose exactly on the free endpoint.
    - Fill every other cell with the background, thereby erasing the frame, legend, and full trace.

    Key insight:
    - Shape and colors come from the framed legend, position comes from the far trace endpoint, and
      orientation comes only from the last step; all three independent clues are required.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    bg = _background_color(g)
    assert bg != _GRAY, "gray is reserved for the visible trace"

    path, endpoints = _gray_path(g)
    _frame, _box, sprite, start = _find_legend(g, bg, path, endpoints)
    end = endpoints[1] if endpoints[0] == start else endpoints[0]
    previous = next(cell for cell in _neighbors(g.shape, end) if cell in path)
    direction = (end[0] - previous[0], end[1] - previous[1])

    nose_colors = Counter(color for _, color in sprite)
    nose_color = next(color for color, count in nose_colors.items() if count == 1)
    nose = next(cell for cell, color in sprite if color == nose_color)

    out = np.full_like(g, bg)
    for (r, c), color in sprite:
        rr_off, cc_off = _rotate_offset((r - nose[0], c - nose[1]), direction)
        rr, cc = end[0] + rr_off, end[1] + cc_off
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], (
            "rotated sprite must fit completely inside the canvas"
        )
        out[rr, cc] = color
    return out


def _step(a: Coord, b: Coord) -> Coord:
    return b[0] - a[0], b[1] - a[1]


def _turn_count(path: list[Coord]) -> int:
    directions = [_step(a, b) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _dominant_direction(start: Coord, end: Coord) -> Coord | None:
    dr, dc = end[0] - start[0], end[1] - start[1]
    if abs(dr) > abs(dc):
        return (1 if dr > 0 else -1, 0)
    if abs(dc) > abs(dr):
        return (0, 1 if dc > 0 else -1)
    return None


def _can_extend(
    shape: tuple[int, int], path: list[Coord], cell: Coord,
    frame_box: BBox, frame: set[Coord]
) -> bool:
    h, w = shape
    r, c = cell
    if not (0 <= r < h and 0 <= c < w) or cell in path:
        return False
    r0, c0, r1, c1 = frame_box
    if r0 <= r <= r1 and c0 <= c <= c1:
        return False
    neighbors_on_path = [nxt for nxt in _neighbors(shape, cell) if nxt in path]
    if neighbors_on_path != [path[-1]]:
        return False
    if any(nxt in frame for nxt in _neighbors(shape, cell)):
        return False
    return True


def _rotated_cells(offsets: tuple[Coord, ...], anchor: Coord, direction: Coord) -> set[Coord]:
    result = set()
    ar, ac = anchor
    for offset in offsets:
        dr, dc = _rotate_offset(offset, direction)
        result.add((ar + dr, ac + dc))
    return result


def _find_path(
    shape: tuple[int, int], start: Coord, initial: Coord, desired: Coord,
    frame_box: BBox, frame: set[Coord], sprite_offsets: tuple[Coord, ...]
) -> list[Coord] | None:
    """Grow an induced winding path and reserve a clear 2--4 cell final segment."""
    h, w = shape
    for _ in range(220):
        path = [start]
        prefix_length = random.randint(8, 18)
        while len(path) < prefix_length:
            candidates = []
            for dr, dc in _DIR4:
                nxt = (path[-1][0] + dr, path[-1][1] + dc)
                if _can_extend(shape, path, nxt, frame_box, frame):
                    candidates.append(nxt)
            if not candidates:
                break
            # The first move away from the frame is structurally forced; later
            # choices remain genuinely random and create varied detours.
            random.shuffle(candidates)
            path.append(candidates[0])
        if len(path) != prefix_length or _step(path[-2], path[-1]) == desired:
            continue

        final_run = random.randint(2, 4)
        candidate_path = list(path)
        for _ in range(final_run):
            nxt = (candidate_path[-1][0] + desired[0], candidate_path[-1][1] + desired[1])
            if not _can_extend(shape, candidate_path, nxt, frame_box, frame):
                break
            candidate_path.append(nxt)
        else:
            end = candidate_path[-1]
            cells = _rotated_cells(sprite_offsets, end, desired)
            if not all(0 <= r < h and 0 <= c < w for r, c in cells):
                continue
            r0, c0, r1, c1 = frame_box
            if any(r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1 for r, c in cells):
                continue
            if _turn_count(candidate_path) < 3:
                continue
            if _step(candidate_path[0], candidate_path[1]) != initial:
                continue
            if _dominant_direction(start, end) in (None, desired):
                continue
            return candidate_path
    return None


def _independent_expected(
    shape: tuple[int, int], bg: int, endpoint: Coord, direction: Coord,
    body_color: int, nose_color: int, body: tuple[Coord, ...]
) -> np.ndarray:
    """Render the intended answer with matrix rotations independent of main()."""
    matrices = {
        (-1, 0): np.array(((1, 0), (0, 1)), dtype=int),
        (0, 1): np.array(((0, 1), (-1, 0)), dtype=int),
        (1, 0): np.array(((-1, 0), (0, -1)), dtype=int),
        (0, -1): np.array(((0, -1), (1, 0)), dtype=int),
    }
    out = np.full(shape, bg, dtype=int)
    out[endpoint] = nose_color
    matrix = matrices[direction]
    for offset in body:
        transformed = matrix @ np.asarray(offset, dtype=int)
        out[endpoint[0] + int(transformed[0]), endpoint[1] + int(transformed[1])] = body_color
    return out


def _frame_start(box: BBox, side: str) -> tuple[Coord, Coord]:
    r0, c0, r1, c1 = box
    if side == "top":
        c = random.randint(c0 + 1, c1 - 1)
        return (r0 - 1, c), (-1, 0)
    if side == "bottom":
        c = random.randint(c0 + 1, c1 - 1)
        return (r1 + 1, c), (1, 0)
    if side == "left":
        r = random.randint(r0 + 1, r1 - 1)
        return (r, c0 - 1), (0, -1)
    r = random.randint(r0 + 1, r1 - 1)
    return (r, c1 + 1), (0, 1)


def _semantic_checks(
    inp: np.ndarray, expected: np.ndarray, path: list[Coord], start: Coord,
    desired: Coord, initial: Coord, body: tuple[Coord, ...], body_color: int,
    nose_color: int, nose_cell: Coord, bg: int
) -> bool:
    """Check clue uniqueness, non-degeneracy, and independent semantic dependencies."""
    try:
        actual = main(inp)
    except (AssertionError, StopIteration):
        return False
    if not np.array_equal(actual, expected):
        return False
    endpoint = path[-1]
    if int(expected[endpoint]) != nose_color or int(np.sum(expected == nose_color)) != 1:
        return False
    if int(np.sum(expected == body_color)) != len(body):
        return False
    if any(int(np.sum(expected == color)) for color in ({_GRAY} - {body_color, nose_color})):
        return False
    if _turn_count(path) < 3 or desired == initial or _step(path[-2], path[-1]) != desired:
        return False

    # A shortcut using the direction at the legend end must produce a different answer.
    wrong_cells = _rotated_cells(((0, 0),) + body, endpoint, initial)
    if all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] for r, c in wrong_cells):
        wrong_orientation = _independent_expected(
            inp.shape, bg, endpoint, initial, body_color, nose_color, body
        )
        if np.array_equal(wrong_orientation, expected):
            return False

    # Remove a non-anchor boundary pixel while keeping the remaining sprite
    # connected. The result must lose that exact output pixel, proving that the
    # shape is read from the legend rather than hard-coded.
    body_set = set(body)
    removable = []
    for cell in body:
        if cell == (1, 0):
            continue
        remaining = (body_set - {cell}) | {(0, 0)}
        reached = {(0, 0)}
        queue = deque([(0, 0)])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DIR4:
                nxt = (r + dr, c + dc)
                if nxt in remaining and nxt not in reached:
                    reached.add(nxt)
                    queue.append(nxt)
        if reached == remaining:
            removable.append(cell)
    if not removable:
        return False
    altered = inp.copy()
    removed = removable[0]
    altered[nose_cell[0] + removed[0], nose_cell[1] + removed[1]] = bg
    try:
        altered_out = main(altered)
    except (AssertionError, StopIteration):
        return False
    if np.array_equal(altered_out, expected):
        return False
    return True


def _gen_attempt(
    required_direction: Coord | None = None, required_template: int | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = random.randint(18, 30), random.randint(18, 30)
    bg = random.choice([color for color in _PALETTE if color != _GRAY])
    frame_color, body_color, nose_color = random.sample(
        [color for color in _PALETTE if color not in (bg, _GRAY)], 3
    )
    body = (
        _BODY_TEMPLATES[required_template]
        if required_template is not None
        else random.choice(_BODY_TEMPLATES)
    )
    sprite_offsets = ((0, 0),) + body

    rows = [r for r, _ in sprite_offsets]
    cols = [c for _, c in sprite_offsets]
    min_r, max_r, min_c, max_c = min(rows), max(rows), min(cols), max(cols)
    pad_top, pad_bottom, pad_left, pad_right = (random.randint(0, 1) for _ in range(4))
    inner_h = max_r - min_r + 1 + pad_top + pad_bottom
    inner_w = max_c - min_c + 1 + pad_left + pad_right
    frame_h, frame_w = inner_h + 2, inner_w + 2
    if frame_h + 8 > h or frame_w + 8 > w:
        return None

    r0 = random.randint(3, h - frame_h - 4)
    c0 = random.randint(3, w - frame_w - 4)
    box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
    frame = _border_cells(box)

    side_initial = {
        "top": (-1, 0), "bottom": (1, 0),
        "left": (0, -1), "right": (0, 1),
    }
    sides = list(side_initial)
    if required_direction is not None:
        sides = [side for side in sides if side_initial[side] != required_direction]
    side = random.choice(sides)
    start, initial = _frame_start(box, side)
    desired = required_direction or random.choice(
        [direction for direction in _DIR4 if direction != initial]
    )
    path = _find_path(inp_shape := (h, w), start, initial, desired, box, frame, sprite_offsets)
    if path is None:
        return None

    inp = np.full(inp_shape, bg, dtype=int)
    for cell in frame:
        inp[cell] = frame_color

    nose_cell = (r0 + 1 + pad_top - min_r, c0 + 1 + pad_left - min_c)
    inp[nose_cell] = nose_color
    for dr, dc in body:
        inp[nose_cell[0] + dr, nose_cell[1] + dc] = body_color
    for cell in path:
        if int(inp[cell]) != bg:
            return None
        inp[cell] = _GRAY

    counts = Counter(map(int, inp.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    expected = _independent_expected(
        inp_shape, bg, path[-1], desired, body_color, nose_color, body
    )
    if not _semantic_checks(
        inp, expected, path, start, desired, initial, body, body_color,
        nose_color, nose_cell, bg
    ):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a diverse, semantically validated input with a unique output signature."""
    if not _DIRECTION_BAG:
        _DIRECTION_BAG.extend(random.sample(list(_DIR4), len(_DIR4)))
    if not _TEMPLATE_BAG:
        _TEMPLATE_BAG.extend(random.sample(range(len(_BODY_TEMPLATES)), len(_BODY_TEMPLATES)))
    required_direction = _DIRECTION_BAG[0]
    required_template = _TEMPLATE_BAG[0]
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(required_direction, required_template)
        if result is None:
            continue
        inp, expected = result
        signature = hashlib.md5(expected.tobytes() + str(expected.shape).encode()).hexdigest()
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        _DIRECTION_BAG.pop(0)
        _TEMPLATE_BAG.pop(0)
        return inp
    raise RuntimeError("failed to generate a valid trace-end sprite input")
