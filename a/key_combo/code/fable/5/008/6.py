from __future__ import annotations

import random
from collections import deque

import numpy as np


# Directions are ordered by the rotation encoded in the puzzle:
# up=0, right=1, down=2, left=3 (clockwise quarter-turns).
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_PALETTE = tuple(range(10))
_GEN_RETRIES = 400


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    top = int(counts.max())
    assert int(np.count_nonzero(counts == top)) == 1, "background must be uniquely most common"
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    """Return 4-connected components using the common ARC (row, col) convention."""
    h, w = mask.shape
    unseen = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    result: list[set[tuple[int, int]]] = []
    while unseen:
        seed = unseen.pop()
        comp = {seed}
        queue = [seed]
        while queue:
            r, c = queue.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if 0 <= p[0] < h and 0 <= p[1] < w and p in unseen:
                    unseen.remove(p)
                    comp.add(p)
                    queue.append(p)
        result.append(comp)
    return result


def _bbox(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _long_cross_info(
    points: set[tuple[int, int]],
) -> tuple[tuple[int, int], int] | None:
    """Recognize a whole component that is exactly a six-cell long-arm cross."""
    if len(points) != 6:
        return None
    matches: list[tuple[tuple[int, int], int]] = []
    for center in points:
        core = {center}
        core.update((center[0] + dr, center[1] + dc) for dr, dc in _DIRS)
        if not core <= points:
            continue
        extra = points - core
        if len(extra) != 1:
            continue
        er, ec = next(iter(extra))
        for k, (dr, dc) in enumerate(_DIRS):
            if (er, ec) == (center[0] + 2 * dr, center[1] + 2 * dc):
                matches.append((center, k))
    return matches[0] if len(matches) == 1 else None


def _rotate_offset(point: tuple[int, int], quarter_turns: int) -> tuple[int, int]:
    """Rotate a row/column offset clockwise about (0, 0)."""
    r, c = point
    for _ in range(int(quarter_turns) % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotated(
    points: set[tuple[int, int]], quarter_turns: int
) -> set[tuple[int, int]]:
    return {_rotate_offset(p, quarter_turns) for p in points}


def _analyze(grid: np.ndarray) -> dict:
    """Infer every role only from visible component structure."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    bg = _background(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert len(colors) >= 2, "field and template colors must both be present"

    by_color: dict[int, list[set[tuple[int, int]]]] = {
        color: _components(g == color) for color in colors
    }
    instructions_by_color: dict[int, list[tuple[set[tuple[int, int]], tuple[int, int], int]]] = {}
    for color, comps in by_color.items():
        found = []
        for comp in comps:
            info = _long_cross_info(comp)
            if info is not None:
                center, direction = info
                found.append((comp, center, direction))
        if found:
            instructions_by_color[color] = found

    assert len(instructions_by_color) == 1, "the instruction field color must be structurally unique"
    field_color = next(iter(instructions_by_color))
    instructions = instructions_by_color[field_color]

    template_candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
    for color, comps in by_color.items():
        if color == field_color:
            continue
        for comp in comps:
            template_candidates.append((len(comp), color, comp))
    assert template_candidates, "a differently colored template component is required"
    largest_area = max(area for area, _, _ in template_candidates)
    largest = [item for item in template_candidates if item[0] == largest_area]
    assert len(largest) == 1, "the largest non-field component must uniquely identify the template"
    _, template_color, template_abs = largest[0]

    r0, c0, r1, c1 = _bbox(template_abs)
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, \
        "template bbox must have an integral geometric center"
    anchor = ((r0 + r1) // 2, (c0 + c1) // 2)
    template_offsets = {(r - anchor[0], c - anchor[1]) for r, c in template_abs}

    return {
        "background": bg,
        "field_color": field_color,
        "template_color": template_color,
        "template_abs": template_abs,
        "template_offsets": template_offsets,
        "instructions": instructions,
        "by_color": by_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase isolated long-arm crosses and stamp centered, direction-rotated copies of the template at their centers.

    Input structure:
    - A uniquely most-common background surrounds field-color components and components of at least one other color.
    - The field color is structurally identified by one or more isolated 4-connected components that are exactly six cells: a five-cell cross plus one cell extending one arm.
    - The template is the unique largest 4-connected component among all non-background, non-field colors; its odd-sized bbox gives an integral center.

    Rule:
    - For every whole field-color component that is exactly a long-arm cross, read its extended arm as up, right, down, or left.
    - Erase all six instruction cells, rotate the template clockwise by 0, 90, 180, or 270 degrees respectively about its bbox center, and paint that copy with the template color at the cross center.
    - Preserve the source template, smaller same-color components, and every field component that is not exactly a six-cell long-arm cross.

    Key insight:
    - Each valid cross encodes both the stamp center and its rotation, while whole-component shape checking rejects locally cross-like patterns embedded in larger field blobs.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze(g)
    bg = int(analysis["background"])
    template_color = int(analysis["template_color"])
    offsets = analysis["template_offsets"]
    instructions = analysis["instructions"]

    out = g.copy()
    for comp, _, _ in instructions:
        for r, c in comp:
            out[r, c] = bg

    h, w = out.shape
    for _, (cr, cc), direction in instructions:
        for dr, dc in _rotated(offsets, direction):
            r, c = cr + dr, cc + dc
            assert 0 <= r < h and 0 <= c < w, "rotated stamp must fit inside the grid"
            out[r, c] = template_color
    return out


def _connected_relative(points: set[tuple[int, int]]) -> bool:
    if not points:
        return False
    seen = {next(iter(points))}
    queue = deque(seen)
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            q = (r + dr, c + dc)
            if q in points and q not in seen:
                seen.add(q)
                queue.append(q)
    return len(seen) == len(points)


def _random_template() -> set[tuple[int, int]] | None:
    """Sample a centered, connected, rotation-sensitive template."""
    for _ in range(500):
        h, w = random.choice(((3, 3), (3, 5), (5, 3), (5, 5)))
        rh, rw = h // 2, w // 2
        cells = [(r, c) for r in range(-rh, rh + 1) for c in range(-rw, rw + 1)]
        area_lo = 5 if (h, w) == (3, 3) else max(7, (h * w + 2) // 3)
        area_hi = min(h * w - 1, area_lo + 4)
        area = random.randint(area_lo, area_hi)

        required = {
            (0, 0),
            (-rh, random.randint(-rw, rw)),
            (rh, random.randint(-rw, rw)),
            (random.randint(-rh, rh), -rw),
            (random.randint(-rh, rh), rw),
        }
        if len(required) > area:
            continue
        remaining = [p for p in cells if p not in required]
        shape = required | set(random.sample(remaining, area - len(required)))
        if not _connected_relative(shape):
            continue
        if _long_cross_info(shape) is not None:
            continue
        variants = {_shape_signature(_rotated(shape, k)) for k in range(4)}
        if len(variants) != 4:
            continue
        return shape
    return None


def _shape_signature(points: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(points))


def _long_cross_offsets(direction: int) -> set[tuple[int, int]]:
    result = {(0, 0)} | set(_DIRS)
    dr, dc = _DIRS[direction]
    result.add((2 * dr, 2 * dc))
    return result


def _pseudo_cross_offsets() -> tuple[set[tuple[int, int]], int]:
    """Build a larger component containing a complete local long-cross subset."""
    direction = random.randrange(4)
    shape = _long_cross_offsets(direction)
    dr, dc = _DIRS[direction]
    pr, pc = -dc, dr
    shape.add((dr + pr, dc + pc))
    target = random.randint(8, 12)
    while len(shape) < target:
        frontier = []
        for r, c in shape:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) not in shape and -3 <= nr <= 3 and -3 <= nc <= 3:
                    frontier.append((nr, nc))
        if not frontier:
            break
        shape.add(random.choice(frontier))
    return shape, direction


def _small_blob(area: int) -> set[tuple[int, int]]:
    shape = {(0, 0)}
    while len(shape) < area:
        r, c = random.choice(tuple(shape))
        dr, dc = random.choice(_DIRS)
        shape.add((r + dr, c + dc))
    r0, c0, _, _ = _bbox(shape)
    return {(r - r0, c - c0) for r, c in shape}


def _too_close(
    points: set[tuple[int, int]], occupied: set[tuple[int, int]], gap: int = 1
) -> bool:
    for r, c in points:
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _positions_for_relative(
    relative: set[tuple[int, int]], h: int, w: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(relative)
    centers = [
        (r, c)
        for r in range(-r0, h - r1)
        for c in range(-c0, w - c1)
    ]
    random.shuffle(centers)
    return centers


def _place_preserved(
    grid: np.ndarray,
    relative: set[tuple[int, int]],
    color: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    for cr, cc in _positions_for_relative(relative, h, w):
        points = {(cr + dr, cc + dc) for dr, dc in relative}
        if _too_close(points, occupied, gap=1):
            continue
        for r, c in points:
            grid[r, c] = color
        occupied.update(points)
        return points
    return None


def _expected_output(
    grid: np.ndarray,
    bg: int,
    template_color: int,
    template: set[tuple[int, int]],
    instruction_data: list[tuple[set[tuple[int, int]], tuple[int, int], int]],
) -> np.ndarray:
    out = grid.copy()
    for cross, _, _ in instruction_data:
        for p in cross:
            out[p] = bg
    for _, (cr, cc), direction in instruction_data:
        for dr, dc in _rotated(template, direction):
            out[cr + dr, cc + dc] = template_color
    return out


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    meta: dict,
) -> bool:
    """Check puzzle semantics, not merely generator/solver round-tripping."""
    try:
        analysis = _analyze(grid)
        solved = main(grid)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(solved, expected) or np.array_equal(solved, grid):
        return False
    if analysis["field_color"] != meta["field_color"]:
        return False
    if analysis["template_color"] != meta["template_color"]:
        return False
    if analysis["template_abs"] != meta["template_abs"]:
        return False

    instructions = analysis["instructions"]
    if len(instructions) != len(meta["instructions"]):
        return False
    if {direction for _, _, direction in instructions} != {0, 1, 2, 3}:
        return False
    template = analysis["template_offsets"]
    if len({_shape_signature(_rotated(template, k)) for k in range(4)}) != 4:
        return False

    # Erasing must matter: at least one original instruction cell per cross becomes bg.
    for cross, _, _ in instructions:
        if not any(int(solved[p]) == analysis["background"] for p in cross):
            return False

    # Whole-component checking must matter: the preserved pseudo blob contains a
    # complete local six-cell instruction subset but is larger than six cells.
    pseudo = meta["pseudo_abs"]
    pseudo_center = meta["pseudo_center"]
    pseudo_direction = meta["pseudo_direction"]
    local = {
        (pseudo_center[0] + dr, pseudo_center[1] + dc)
        for dr, dc in _long_cross_offsets(pseudo_direction)
    }
    if not (len(pseudo) > 6 and local <= pseudo):
        return False
    if any(int(solved[p]) != meta["field_color"] for p in pseudo):
        return False

    # A no-rotation shortcut and a no-erasure shortcut must each disagree.
    no_rotation = grid.copy()
    for cross, _, _ in instructions:
        for p in cross:
            no_rotation[p] = analysis["background"]
    for _, (cr, cc), _ in instructions:
        for dr, dc in template:
            no_rotation[cr + dr, cc + dc] = analysis["template_color"]
    if np.array_equal(no_rotation, solved):
        return False

    no_erase = grid.copy()
    for _, (cr, cc), direction in instructions:
        for dr, dc in _rotated(template, direction):
            no_erase[cr + dr, cc + dc] = analysis["template_color"]
    if np.array_equal(no_erase, solved):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(25, 30)
    w = random.randint(25, 30)
    bg, field_color, template_color = random.sample(_PALETTE, 3)
    grid = np.full((h, w), bg, dtype=int)

    template = _random_template()
    if template is None:
        return None
    occupied: set[tuple[int, int]] = set()
    template_abs = _place_preserved(grid, template, template_color, occupied)
    if template_abs is None:
        return None

    # Smaller same-color components make "largest component" observable.
    for _ in range(random.randint(1, 2)):
        decoy = _small_blob(random.randint(1, min(4, len(template) - 1)))
        if _place_preserved(grid, decoy, template_color, occupied) is None:
            return None

    pseudo_rel, pseudo_direction = _pseudo_cross_offsets()
    pseudo_abs = _place_preserved(grid, pseudo_rel, field_color, occupied)
    if pseudo_abs is None:
        return None
    pr0, pc0, _, _ = _bbox(pseudo_abs)
    rr0, rc0, _, _ = _bbox(pseudo_rel)
    pseudo_center = (pr0 - rr0, pc0 - rc0)

    # Optional extra field blobs vary the number and outline of preserved objects.
    for _ in range(random.randint(0, 2)):
        junk = _small_blob(random.randint(2, 5))
        if _place_preserved(grid, junk, field_color, occupied) is None:
            return None

    preserved = set(occupied)
    reserved_targets: set[tuple[int, int]] = set()
    directions = [0, 1, 2, 3]
    if random.random() < 0.5:
        directions.append(random.randrange(4))
    random.shuffle(directions)

    instruction_data: list[tuple[set[tuple[int, int]], tuple[int, int], int]] = []
    all_stamp_orientations: set[tuple[int, int]] = set()
    for k in range(4):
        all_stamp_orientations.update(_rotated(template, k))
    for direction in directions:
        cross_rel = _long_cross_offsets(direction)
        stamp_rel = _rotated(template, direction)
        # Reserve enough room for every orientation.  Besides keeping the examples
        # visually separated, this makes the "ignore the direction" counterexample
        # a valid in-grid alternative that can be explicitly rejected.
        union_rel = cross_rel | all_stamp_orientations
        placed = False
        for cr, cc in _positions_for_relative(union_rel, h, w):
            cross_abs = {(cr + dr, cc + dc) for dr, dc in cross_rel}
            stamp_abs = {(cr + dr, cc + dc) for dr, dc in stamp_rel}
            union_abs = cross_abs | stamp_abs
            if _too_close(cross_abs, occupied, gap=1):
                continue
            if _too_close(stamp_abs, preserved, gap=1):
                continue
            if _too_close(union_abs, reserved_targets, gap=1):
                continue
            for r, c in cross_abs:
                grid[r, c] = field_color
            occupied.update(cross_abs)
            reserved_targets.update(union_abs)
            instruction_data.append((cross_abs, (cr, cc), direction))
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(
        grid, bg, template_color, template, instruction_data
    )
    meta = {
        "field_color": field_color,
        "template_color": template_color,
        "template_abs": template_abs,
        "instructions": instruction_data,
        "pseudo_abs": pseudo_abs,
        "pseudo_center": pseudo_center,
        "pseudo_direction": pseudo_direction,
    }
    if not _validate_generated(grid, expected, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance with all four direction branches."""
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid long-arm cross instance")
