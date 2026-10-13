"""ARC task: stamp an unrotated sprite that turns clockwise on collision."""

from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GRAY = 5
PINK = 6

_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
_SOURCE_COLORS = (1, 3, 7, 8, 9)
_MAX_GENERATION_ATTEMPTS = 500
_generation_counter = 0
_seen_outputs: set[tuple[tuple[int, int], bytes]] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive (r0, c0, r1, c1) bbox used by arc_utils."""
    coords = np.argwhere(mask)
    assert len(coords), "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_8_connected(mask: np.ndarray) -> bool:
    coords = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not coords:
        return False
    seen = {next(iter(coords))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r + dr, c + dc) in coords and (r + dr, c + dc) not in seen:
                    seen.add((r + dr, c + dc))
                    stack.append((r + dr, c + dc))
    return seen == coords


def _extract_sprite(grid: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    """Identify the sole non-wall foreground sprite and return its bbox anchor/mask."""
    source_cells = (grid != BLACK) & (grid != GRAY)
    source_colors = np.unique(grid[source_cells])
    assert len(source_colors) == 1, "expected exactly one non-gray foreground color"
    source_color = int(source_colors[0])
    r0, c0, r1, c1 = _bbox(source_cells)
    mask = source_cells[r0 : r1 + 1, c0 : c1 + 1]
    assert _is_8_connected(mask), "source sprite must be 8-connected"
    assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()

    # The source and every input wall must be separated in the 8-neighborhood.
    walls = set(map(tuple, np.argwhere(grid == GRAY)))
    assert walls, "expected at least one gray wall pixel"
    for raw_r, raw_c in np.argwhere(source_cells):
        r, c = int(raw_r), int(raw_c)
        assert not any((r + dr, c + dc) in walls for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    return r0, c0, source_color, mask


def _run(grid: np.ndarray, *, keep_trace: bool = False):
    """Execute the collision walk; optionally return semantic trace records."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30x30"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette values 0..9"

    r, c, source_color, mask = _extract_sprite(g)
    sprite_h, sprite_w = map(int, mask.shape)
    steps = (sprite_w + 1, sprite_h + 1, sprite_w + 1, sprite_h + 1)
    segment_colors = (source_color, PINK, RED, YELLOW)

    out = g.copy()
    trace: list[dict] = []
    direction = 0
    turns = 0
    blocked_streak = 0
    successes = 0

    while blocked_streak < 4:
        dr, dc = _DIRECTIONS[direction]
        nr = r + dr * steps[direction]
        nc = c + dc * steps[direction]

        if nr < 0 or nc < 0 or nr + sprite_h > g.shape[0] or nc + sprite_w > g.shape[1]:
            reasons = ("edge",)
        else:
            view = out[nr : nr + sprite_h, nc : nc + sprite_w]
            wall_hit = bool(np.any(mask & (view == GRAY)))
            ink_hit = bool(np.any(mask & (view != BLACK) & (view != GRAY)))
            reasons = tuple(name for name, hit in (("wall", wall_hit), ("self", ink_hit)) if hit)

        if reasons:
            if keep_trace:
                trace.append(
                    {"kind": "turn", "anchor": (nr, nc), "turns": turns,
                     "direction": direction, "reasons": reasons}
                )
            direction = (direction + 1) % 4
            turns += 1
            blocked_streak += 1
            continue

        color = int(segment_colors[turns % 4])
        view = out[nr : nr + sprite_h, nc : nc + sprite_w]
        view[mask] = color
        r, c = nr, nc
        blocked_streak = 0
        successes += 1
        assert successes <= g.size, "collision walk did not make finite progress"
        if keep_trace:
            trace.append(
                {"kind": "stamp", "anchor": (r, c), "turns": turns,
                 "direction": direction, "color": color}
            )

    return (out, trace) if keep_trace else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp an unrotated sprite until clockwise collision turns leave no available direction.

    Input structure:
    - A black grid contains one tightly bounded, monochromatic, 8-connected non-gray
      sprite and several gray wall pixels or wall blocks; sprite and walls do not touch.

    Rule:
    - Starting to the right, test bbox-aligned copies one blank column or row beyond the
      previous successful copy. If the full bbox is out of bounds, or the sprite mask
      overlaps a gray wall or earlier foreground ink, do not stamp and turn clockwise.
      Stamp colors cycle by accumulated turns as source, pink, red, yellow; preserve the
      original sprite and walls. Stop after four consecutive blocked directions.

    Key insight:
    - Walls and old stamps are turn switches for one unrotated mask, while color records
      the number of turns rather than the copy's position along the resulting spiral.
    """
    return _run(grid)


def _random_mask() -> np.ndarray:
    """Generate a tight irregular 8-connected mask with at least one transparent cell."""
    for _ in range(200):
        h = random.randint(2, 3)
        w = random.randint(2, 3)
        mask = np.random.random((h, w)) < random.uniform(0.52, 0.78)
        if int(mask.sum()) < 3 or mask.all():
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if _is_8_connected(mask):
            return mask
    raise RuntimeError("failed to generate an irregular sprite mask")


def _mask_cells(anchor: tuple[int, int], mask: np.ndarray) -> set[tuple[int, int]]:
    r0, c0 = anchor
    return {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}


def _add_wall_group(
    grid: np.ndarray,
    anchor: tuple[int, int],
    mask: np.ndarray,
    occupied_ink: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    """Put a one/two-pixel wall on the candidate mask, guaranteeing a real collision."""
    true_local = [tuple(map(int, p)) for p in np.argwhere(mask)]
    random.shuffle(true_local)
    for lr, lc in true_local:
        cell = (anchor[0] + lr, anchor[1] + lc)
        if cell in occupied_ink or int(grid[cell]) != BLACK:
            continue
        group = {cell}
        if random.random() < 0.55:
            neighbors = [
                (anchor[0] + rr, anchor[1] + cc)
                for rr, cc in true_local
                if max(abs(rr - lr), abs(cc - lc)) == 1
                and (anchor[0] + rr, anchor[1] + cc) not in occupied_ink
                and int(grid[anchor[0] + rr, anchor[1] + cc]) == BLACK
            ]
            if neighbors:
                group.add(random.choice(neighbors))
        for p in group:
            grid[p] = GRAY
        return group
    return None


def _make_candidate(mode: str):
    mask = _random_mask()
    sprite_h, sprite_w = map(int, mask.shape)
    row_step, col_step = sprite_h + 1, sprite_w + 1

    height = random.randint(max(20, sprite_h + 5 * row_step), 30)
    width = random.randint(max(20, sprite_w + 6 * col_step), 30)
    length_right = random.randint(3, min(5, (width - sprite_w - 2) // col_step - 1))
    length_down = random.randint(2, min(4, (height - sprite_h - 2) // row_step - 1))
    length_left = random.randint(1, length_right - 1)

    max_r0 = height - sprite_h - (length_down + 1) * row_step
    if max_r0 < 1:
        return None
    r0 = random.randint(1, max_r0)

    if mode == "edge":
        # The last successful rightward bbox is flush with the right edge.
        c0 = width - sprite_w - length_right * col_step
        if c0 < 1:
            return None
    else:
        max_c0 = width - sprite_w - (length_right + 1) * col_step
        if max_c0 < 1:
            return None
        c0 = random.randint(1, max_c0)

    source_color = random.choice(_SOURCE_COLORS)
    grid = np.zeros((height, width), dtype=int)
    source_anchor = (r0, c0)
    source_cells = _mask_cells(source_anchor, mask)
    for p in source_cells:
        grid[p] = source_color

    occupied = set(source_cells)
    planned_by_turn: dict[int, list[tuple[int, int]]] = {}
    wall_groups: list[set[tuple[int, int]]] = []
    current = source_anchor
    lengths = (length_right, length_down, length_left)

    # Plan the first three straight segments explicitly. Their failures are the
    # counterexample-bearing wall/edge turns; later motion is left to the solver.
    for direction, segment_length in enumerate(lengths):
        dr, dc = _DIRECTIONS[direction]
        step = col_step if direction in (0, 2) else row_step
        anchors = []
        for _ in range(segment_length):
            candidate = (current[0] + dr * step, current[1] + dc * step)
            cells = _mask_cells(candidate, mask)
            if cells & occupied or any(not (0 <= r < height and 0 <= c < width) for r, c in cells):
                return None
            if any(int(grid[p]) != BLACK for p in cells):
                return None
            occupied.update(cells)
            current = candidate
            anchors.append(candidate)
        planned_by_turn[direction] = anchors

        blocked_anchor = (current[0] + dr * step, current[1] + dc * step)
        if direction == 0 and mode == "edge":
            if blocked_anchor[1] + sprite_w <= width:
                return None
        else:
            if not (0 <= blocked_anchor[0] <= height - sprite_h and 0 <= blocked_anchor[1] <= width - sprite_w):
                return None
            group = _add_wall_group(grid, blocked_anchor, mask, occupied)
            if group is None:
                return None
            wall_groups.append(group)

    # A wall in a transparent hole of a successful bbox must *not* block stamping.
    hole_local = [tuple(map(int, p)) for p in np.argwhere(~mask)]
    hole_anchor = planned_by_turn[0][0]
    lr, lc = random.choice(hole_local)
    hole_wall = (hole_anchor[0] + lr, hole_anchor[1] + lc)
    if int(grid[hole_wall]) != BLACK or hole_wall in occupied:
        return None
    grid[hole_wall] = GRAY

    plan = {
        "mode": mode,
        "mask": mask,
        "source_anchor": source_anchor,
        "source_color": source_color,
        "planned": planned_by_turn,
        "lengths": lengths,
        "wall_groups": wall_groups,
        "hole_wall": hole_wall,
        "hole_anchor": hole_anchor,
    }
    return grid, plan


def _valid_semantics(grid: np.ndarray, plan: dict) -> bool:
    """Check the stated rule's dependencies, not merely main(generate_input()) closure."""
    try:
        out, trace = _run(grid, keep_trace=True)
    except (AssertionError, ValueError, IndexError):
        return False

    mask = plan["mask"]
    source_color = int(plan["source_color"])
    if grid.shape != out.shape or max(grid.shape) > 30 or np.array_equal(grid, out):
        return False
    if not np.array_equal(out[grid != BLACK], grid[grid != BLACK]):
        return False

    # The four deliberately visible segments must have their exact planned masks,
    # followed respectively by wall/edge, wall, wall, and old-ink collisions.
    expected_counts = (*plan["lengths"], plan["lengths"][1] - 1)
    expected_reasons = ((plan["mode"],), ("wall",), ("wall",), ("self",))
    expected_colors = (source_color, PINK, RED, YELLOW)
    for turns in range(4):
        stamps = [e for e in trace if e["kind"] == "stamp" and e["turns"] == turns]
        failures = [e for e in trace if e["kind"] == "turn" and e["turns"] == turns]
        if len(stamps) != expected_counts[turns] or not failures:
            return False
        if tuple(e["anchor"] for e in stamps) != tuple(plan["planned"].get(turns, [])) and turns < 3:
            return False
        if stamps and any(e["color"] != expected_colors[turns] for e in stamps):
            return False
        if failures[0]["reasons"] != expected_reasons[turns]:
            return False

    # Reconstruct the expected upward anchors independently from the planned bottom.
    bottom = plan["planned"][2][-1]
    row_step = mask.shape[0] + 1
    expected_up = tuple((bottom[0] - k * row_step, bottom[1]) for k in range(1, plan["lengths"][1]))
    actual_up = tuple(e["anchor"] for e in trace if e["kind"] == "stamp" and e["turns"] == 3)
    if actual_up != expected_up:
        return False

    # The transparent-hole wall is a direct negative test for bbox collision.
    if out[plan["hole_wall"]] != GRAY:
        return False
    hole_stamp = [e for e in trace if e["kind"] == "stamp" and e["anchor"] == plan["hole_anchor"]]
    if not hole_stamp:
        return False

    # Every planned wall is causally necessary: removing that complete wall group
    # must permit a different walk and therefore a different final output.
    for group in plan["wall_groups"]:
        modified = grid.copy()
        for p in group:
            modified[p] = BLACK
        try:
            if np.array_equal(_run(modified), out):
                return False
        except AssertionError:
            return False

    # An immediate re-turn must affect a later visible stamp, rather than occur
    # only among the four terminal failures. This rules out a one-turn-per-segment
    # shortcut and makes the accumulated turn counter causally observable.
    consecutive_turns = 0
    visible_multi_turn = False
    for event in trace:
        if event["kind"] == "turn":
            consecutive_turns += 1
        else:
            if consecutive_turns >= 2:
                visible_multi_turn = True
            consecutive_turns = 0
    if not visible_multi_turn:
        return False

    # Termination must be precisely four consecutive blocked turns.
    if len(trace) < 4 or any(e["kind"] != "turn" for e in trace[-4:]):
        return False
    if not {source_color, PINK, RED, YELLOW}.issubset(set(map(int, np.unique(out)))):
        return False
    return True


def generate_input() -> np.ndarray:
    """Return a varied, semantically validated input; edge/wall branches alternate."""
    global _generation_counter
    mode = "wall" if _generation_counter % 2 == 0 else "edge"
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        candidate = _make_candidate(mode)
        if candidate is None:
            continue
        grid, plan = candidate
        if not _valid_semantics(grid, plan):
            continue
        output = main(grid)
        signature = (tuple(map(int, output.shape)), output.tobytes())
        if signature in _seen_outputs:
            continue
        _seen_outputs.add(signature)
        _generation_counter += 1
        return grid
    raise RuntimeError(f"failed to generate a valid {mode}-collision stamping input")
