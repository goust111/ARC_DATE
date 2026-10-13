"""ARC task: cast a diagonal billiard ray from a two-pixel pointer."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


# Grey is a visible, fixed tail marker.  The head and target colors vary.
TAIL_COLOR = 5
ARC_COLORS = tuple(range(10))
RECT_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
MAX_GENERATION_ATTEMPTS = 500

Coord = tuple[int, int]
Rect = tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1)
Step = tuple[Coord, bool, bool]  # next position, row bounce, column bounce


def _background(grid: np.ndarray) -> int:
    """Return the dominant color, following the shared ARC toolbox convention."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _locate_pointer(grid: np.ndarray, bg: int) -> tuple[Coord, Coord, Coord, int]:
    """Find the unique grey tail and its isolated diagonal head."""
    tails = np.argwhere(grid == TAIL_COLOR)
    assert len(tails) == 1, "expected one fixed grey tail marker"
    tail = tuple(map(int, tails[0]))

    candidates: list[Coord] = []
    for dr in (-1, 1):
        for dc in (-1, 1):
            r, c = tail[0] + dr, tail[1] + dc
            if _in_bounds(grid.shape, r, c) and int(grid[r, c]) not in (bg, TAIL_COLOR):
                candidates.append((r, c))
    assert len(candidates) == 1, "the tail must have one diagonal foreground head"
    head = candidates[0]
    direction = (head[0] - tail[0], head[1] - tail[1])
    ray_color = int(grid[head])

    pointer = {tail, head}
    for pr, pc in pointer:
        adjacent_foreground = {
            (pr + dr, pc + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc)
            and _in_bounds(grid.shape, pr + dr, pc + dc)
            and int(grid[pr + dr, pc + dc]) != bg
        }
        assert adjacent_foreground == pointer - {(pr, pc)}, "pointer must be isolated"
    return tail, head, direction, ray_color


def _component_at(grid: np.ndarray, start: Coord, forbidden: set[Coord]) -> set[Coord]:
    """4-connected monochromatic component containing start."""
    color = int(grid[start])
    stack = [start]
    seen: set[Coord] = set()
    h, w = grid.shape
    while stack:
        r, c = stack.pop()
        if (r, c) in seen or (r, c) in forbidden:
            continue
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != color:
            continue
        seen.add((r, c))
        stack.extend(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
    return seen


def _target_components(grid: np.ndarray, bg: int, pointer: set[Coord]) -> list[set[Coord]]:
    """Collect and validate all solid monochrome rectangular target blocks."""
    unseen = {
        (int(r), int(c))
        for r, c in np.argwhere(grid != bg)
        if (int(r), int(c)) not in pointer
    }
    components: list[set[Coord]] = []
    while unseen:
        start = next(iter(unseen))
        comp = _component_at(grid, start, pointer)
        unseen -= comp
        assert len(comp) >= 4, "each target must be a nontrivial solid block"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        box = {
            (r, c)
            for r in range(min(rows), max(rows) + 1)
            for c in range(min(cols), max(cols) + 1)
        }
        assert comp == box, "each target must fill its bounding rectangle"
        components.append(comp)
    assert len(components) >= 2, "expected several target blocks"
    return components


def _advance(
    shape: tuple[int, int], position: Coord, direction: Coord
) -> tuple[Coord, Coord, bool, bool]:
    """Advance one cell, reflecting exactly the coordinate that would leave the grid."""
    h, w = shape
    r, c = position
    dr, dc = direction
    bounce_r = not (0 <= r + dr < h)
    bounce_c = not (0 <= c + dc < w)
    if bounce_r:
        dr = -dr
    if bounce_c:
        dc = -dc
    return (r + dr, c + dc), (dr, dc), bounce_r, bounce_c


def main(grid: np.ndarray) -> np.ndarray:
    """Cast the pointer's diagonal ray with boundary bounces, then recolor the first target it hits.

    Input structure:
    - A dominant uniform background contains one isolated diagonal two-pixel pointer.
    - Its fixed grey tail touches one differently colored head; the head color is the ray color.
    - Several separated solid monochrome rectangular target blocks use other colors.

    Rule:
    - Start at the head and move one diagonal cell at a time in the tail-to-head direction.
    - Before an out-of-bounds step, reverse each offending row or column direction component.
    - Paint traversed background cells with the head color.  On first entering a target,
      stop and recolor that target's complete 4-connected block with the head color.

    Key insight:
    - The grey-to-colored pointer jointly determines start, direction, and paint color;
      the first target is found on the reflected polyline, not on an unbent straight ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2
    bg = _background(g)
    assert bg != TAIL_COLOR, "the fixed grey tail must differ from the background"
    tail, head, direction, ray_color = _locate_pointer(g, bg)
    pointer = {tail, head}
    targets = _target_components(g, bg, pointer)
    target_of = {cell: comp for comp in targets for cell in comp}

    out = g.copy()
    position = head
    seen_states: set[tuple[Coord, Coord]] = set()
    while (position, direction) not in seen_states:
        seen_states.add((position, direction))
        position, direction, _bounce_r, _bounce_c = _advance(g.shape, position, direction)

        if position in target_of:
            for r, c in target_of[position]:
                out[r, c] = ray_color
            return out
        if position not in pointer:
            assert int(g[position]) == bg, "ray encountered an invalid non-target object"
            out[position] = ray_color

    raise AssertionError("the reflected ray must hit a target before repeating")


def _billiard_cycle(shape: tuple[int, int], head: Coord, direction: Coord) -> list[Step]:
    """List one full deterministic billiard state cycle for generator reasoning."""
    position = head
    state = (position, direction)
    seen: set[tuple[Coord, Coord]] = set()
    steps: list[Step] = []
    while state not in seen:
        seen.add(state)
        position, direction, br, bc = _advance(shape, position, direction)
        steps.append((position, br, bc))
        state = (position, direction)
    return steps


def _rect_cells(rect: Rect) -> set[Coord]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _rectangles_containing(shape: tuple[int, int], cell: Coord) -> list[Rect]:
    """Candidate target rectangles containing cell, with a one-cell outer margin."""
    h, w = shape
    r, c = cell
    result: list[Rect] = []
    for rh, rw in RECT_SHAPES:
        for r0 in range(r - rh + 1, r + 1):
            for c0 in range(c - rw + 1, c + 1):
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if 1 <= rect[0] and rect[2] < h - 1 and 1 <= rect[1] and rect[3] < w - 1:
                    result.append(rect)
    random.shuffle(result)
    return result


def _well_separated(cells: set[Coord], occupied: set[Coord], gap: int = 1) -> bool:
    return all(
        not any(abs(r - rr) <= gap and abs(c - cc) <= gap for rr, cc in occupied)
        for r, c in cells
    )


def _choose_hit(
    shape: tuple[int, int], steps: list[Step], pointer: set[Coord]
) -> tuple[int, Rect] | None:
    """Choose a collision that visibly depends on both row and column reflections."""
    prefix: set[Coord] = set()
    row_bounces: set[int] = set()
    col_bounces: set[int] = set()
    candidates: list[tuple[int, Rect]] = []
    full_path = {position for position, _br, _bc in steps}

    for index, (position, br, bc) in enumerate(steps):
        if br:
            row_bounces.add(index)
        if bc:
            col_bounces.add(index)
        prefix.add(position)
        distinct_bounce_steps = row_bounces | col_bounces
        has_visible_kink = any(i in row_bounces ^ col_bounces for i in distinct_bounce_steps)
        if (
            index < 7
            or not row_bounces
            or not col_bounces
            or len(distinct_bounce_steps) < 2
            or not has_visible_kink
            or position in pointer
        ):
            continue

        earlier = {p for p, _x, _y in steps[:index]}
        for rect in _rectangles_containing(shape, position):
            cells = _rect_cells(rect)
            if cells & earlier or not _well_separated(cells, pointer, gap=1):
                continue
            # Whole-block recoloring must add pixels beyond the thin trajectory.
            if len(cells - full_path) < 2:
                continue
            candidates.append((index, rect))
        if len(candidates) >= 50 or index >= 3 * sum(shape):
            break
    return random.choice(candidates) if candidates else None


def _choose_later_path_target(
    shape: tuple[int, int],
    steps: list[Step],
    hit_index: int,
    forbidden: set[Coord],
) -> Rect | None:
    """Place a target later on the same route so collision-stop is indispensable."""
    prefix = {p for p, _br, _bc in steps[: hit_index + 1]}
    candidates: list[Rect] = []
    for index in range(hit_index + 5, len(steps)):
        position = steps[index][0]
        for rect in _rectangles_containing(shape, position):
            cells = _rect_cells(rect)
            if cells & prefix or not _well_separated(cells, forbidden, gap=1):
                continue
            candidates.append(rect)
        if len(candidates) >= 40:
            break
    return random.choice(candidates) if candidates else None


def _choose_scattered_target(
    shape: tuple[int, int], forbidden: set[Coord], protected_prefix: set[Coord]
) -> Rect | None:
    """Place a distractor target away from the actual pre-collision trajectory."""
    h, w = shape
    candidates: list[Rect] = []
    for rh, rw in RECT_SHAPES:
        for r0 in range(1, h - rh):
            for c0 in range(1, w - rw):
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                cells = _rect_cells(rect)
                if cells & protected_prefix or not _well_separated(cells, forbidden, gap=1):
                    continue
                candidates.append(rect)
    return random.choice(candidates) if candidates else None


def _straight_ray_misses_targets(
    shape: tuple[int, int], head: Coord, direction: Coord, targets: set[Coord]
) -> bool:
    r, c = head
    dr, dc = direction
    while _in_bounds(shape, r + dr, c + dc):
        r, c = r + dr, c + dc
        if (r, c) in targets:
            return False
    return True


def _expected_output(
    grid: np.ndarray,
    bg: int,
    ray_color: int,
    steps: list[Step],
    hit_index: int,
    hit_cells: set[Coord],
) -> np.ndarray:
    """Independent construction oracle; it does not call main()."""
    expected = grid.copy()
    for position, _br, _bc in steps[:hit_index]:
        if int(grid[position]) == bg:
            expected[position] = ray_color
    for r, c in hit_cells:
        expected[r, c] = ray_color
    return expected


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    target_color: int,
    tail: Coord,
    head: Coord,
    direction: Coord,
    steps: list[Step],
    hit_index: int,
    rectangles: list[Rect],
) -> bool:
    """Check puzzle semantics, uniqueness, necessary dependencies, and non-degeneracy."""
    if grid.ndim != 2 or max(grid.shape) > 30 or _background(grid) != bg:
        return False
    if int(np.count_nonzero(grid == TAIL_COLOR)) != 1 or tail == head:
        return False

    pointer = {tail, head}
    blocks = [_rect_cells(rect) for rect in rectangles]
    hit_block = blocks[0]
    all_targets = set().union(*blocks)
    prefix = [p for p, _br, _bc in steps[:hit_index]]
    collision = steps[hit_index][0]

    # The deterministic first collision is unique and occurs only after a real fold.
    if collision not in hit_block or any(p in all_targets for p in prefix):
        return False
    bounce_steps = [(i, br, bc) for i, (_p, br, bc) in enumerate(steps[:hit_index]) if br or bc]
    if len(bounce_steps) < 2:
        return False
    if not any(br for _i, br, _bc in bounce_steps) or not any(bc for _i, _br, bc in bounce_steps):
        return False
    if not any(br != bc for _i, br, bc in bounce_steps):
        return False
    if not _straight_ray_misses_targets(grid.shape, head, direction, all_targets):
        return False

    # Whole-block recoloring and collision-stop each visibly matter.
    path_cells = {p for p, _br, _bc in steps[: hit_index + 1]}
    if len(hit_block - path_cells) < 2:
        return False
    future_contacts = [
        i
        for i, (p, _br, _bc) in enumerate(steps[hit_index + 1 :], start=hit_index + 1)
        if p in blocks[1]
    ]
    if not future_contacts:
        return False
    if any(int(expected[r, c]) != target_color for r, c in blocks[1]):
        return False
    if int(np.count_nonzero(expected != grid)) < 8:
        return False

    try:
        parsed_tail, parsed_head, parsed_direction, parsed_color = _locate_pointer(grid, bg)
        if (parsed_tail, parsed_head, parsed_direction) != (tail, head, direction):
            return False
        if parsed_color != int(grid[head]):
            return False
        parsed_blocks = _target_components(grid, bg, pointer)
        if len(parsed_blocks) != len(rectangles):
            return False
        actual = main(grid)
    except (AssertionError, IndexError, ValueError):
        return False
    return bool(np.array_equal(actual, expected))


def _generation_attempt() -> np.ndarray | None:
    h = random.randint(15, 24)
    w = random.randint(15, 24)
    bg, ray_color, target_color = random.sample([c for c in ARC_COLORS if c != TAIL_COLOR], 3)

    direction = random.choice(((-1, -1), (-1, 1), (1, -1), (1, 1)))
    possible_heads = [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if 1 <= r - direction[0] < h - 1 and 1 <= c - direction[1] < w - 1
    ]
    head = random.choice(possible_heads)
    tail = (head[0] - direction[0], head[1] - direction[1])
    pointer = {tail, head}
    steps = _billiard_cycle((h, w), head, direction)

    hit_choice = _choose_hit((h, w), steps, pointer)
    if hit_choice is None:
        return None
    hit_index, hit_rect = hit_choice
    hit_cells = _rect_cells(hit_rect)

    later_rect = _choose_later_path_target((h, w), steps, hit_index, pointer | hit_cells)
    if later_rect is None:
        return None
    rectangles = [hit_rect, later_rect]
    occupied = pointer | hit_cells | _rect_cells(later_rect)
    protected_prefix = {p for p, _br, _bc in steps[: hit_index + 1]}

    target_count = random.randint(3, 5)
    while len(rectangles) < target_count:
        rect = _choose_scattered_target((h, w), occupied, protected_prefix)
        if rect is None:
            return None
        rectangles.append(rect)
        occupied |= _rect_cells(rect)

    grid = np.full((h, w), bg, dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = target_color
    grid[tail] = TAIL_COLOR
    grid[head] = ray_color

    expected = _expected_output(grid, bg, ray_color, steps, hit_index, hit_cells)
    if not _validate_generated(
        grid,
        expected,
        bg,
        target_color,
        tail,
        head,
        direction,
        steps,
        hit_index,
        rectangles,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked billiard-ray puzzle input."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _generation_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a non-degenerate billiard-ray input")
