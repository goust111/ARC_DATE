from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


# Cardinal directions use the shared ARC convention: coordinates are (row, col).
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

MIN_SIDE = 20
MAX_SIDE = 30
MIN_CHAIN = 3
MAX_CHAIN = 4
MIN_DECOYS = 1
MAX_DECOYS = 2
MAX_GENERATION_ATTEMPTS = 500


def _arrow_offsets(direction: tuple[int, int]) -> frozenset[tuple[int, int]]:
    """Six-cell offset T whose nearer stem endpoint is the tip at (0, 0)."""
    if direction == UP:
        return frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (1, -1), (1, 1)})
    if direction == DOWN:
        return frozenset({(0, 0), (-1, 0), (-2, 0), (-3, 0), (-1, -1), (-1, 1)})
    if direction == LEFT:
        return frozenset({(0, 0), (0, 1), (0, 2), (0, 3), (-1, 1), (1, 1)})
    if direction == RIGHT:
        return frozenset({(0, 0), (0, -1), (0, -2), (0, -3), (-1, -1), (1, -1)})
    raise ValueError(f"not a cardinal direction: {direction}")


ARROW_TEMPLATES = {direction: _arrow_offsets(direction) for direction in DIRECTIONS}


def _component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    """Return one 4-connected component of a boolean mask."""
    height, width = mask.shape
    todo = deque([start])
    seen = {start}
    while todo:
        r, c = todo.popleft()
        for dr, dc in DIRECTIONS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                todo.append((rr, cc))
    return seen


def _parse_arrow(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Recognize an exact six-pixel offset-crossbar T and return (tip, direction)."""
    if len(cells) != 6:
        return None
    matches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for tip in cells:
        tr, tc = tip
        for direction, offsets in ARROW_TEMPLATES.items():
            shifted = {(tr + dr, tc + dc) for dr, dc in offsets}
            if shifted == cells:
                matches.append((tip, direction))
    assert len(matches) <= 1, "an arrow sprite must have a unique tip and orientation"
    return matches[0] if matches else None


def _find_arrows(g: np.ndarray, bg: int) -> list[dict]:
    """Find all exact monochrome arrow components, regardless of their color ids."""
    height, width = g.shape
    arrows: list[dict] = []
    for color in sorted(set(map(int, g.flat)) - {int(bg)}):
        mask = g == color
        unseen = {(int(r), int(c)) for r, c in np.argwhere(mask)}
        while unseen:
            start = next(iter(unseen))
            cells = _component(mask, start)
            unseen.difference_update(cells)
            parsed = _parse_arrow(cells)
            if parsed is None:
                continue
            tip, direction = parsed
            assert all(0 < r < height - 1 and 0 < c < width - 1 for r, c in cells), (
                "arrow sprites must lie strictly inside the frame"
            )
            arrows.append(
                {
                    "cells": frozenset(cells),
                    "tip": tip,
                    "direction": direction,
                    "color": int(color),
                }
            )
    return arrows


def _frame_start(g: np.ndarray) -> tuple[int, tuple[int, int], tuple[int, int], int]:
    """Infer the perimeter frame, its unique odd pixel, and the inward direction."""
    height, width = g.shape
    perimeter = (
        [int(g[0, c]) for c in range(width)]
        + [int(g[height - 1, c]) for c in range(width)]
        + [int(g[r, 0]) for r in range(1, height - 1)]
        + [int(g[r, width - 1]) for r in range(1, height - 1)]
    )
    counts = Counter(perimeter)
    frame_color, frame_count = counts.most_common(1)[0]
    assert frame_count == len(perimeter) - 1, "the frame must contain exactly one differently colored start"

    starts: list[tuple[int, int]] = []
    for c in range(width):
        if int(g[0, c]) != frame_color:
            starts.append((0, c))
        if int(g[height - 1, c]) != frame_color:
            starts.append((height - 1, c))
    for r in range(1, height - 1):
        if int(g[r, 0]) != frame_color:
            starts.append((r, 0))
        if int(g[r, width - 1]) != frame_color:
            starts.append((r, width - 1))
    assert len(starts) == 1, "expected a unique frame start"
    start = starts[0]
    r, c = start
    assert r not in (0, height - 1) or c not in (0, width - 1), "the start may not be a corner"
    if r == 0:
        inward = DOWN
    elif r == height - 1:
        inward = UP
    elif c == 0:
        inward = RIGHT
    else:
        assert c == width - 1
        inward = LEFT
    return int(frame_color), start, inward, int(g[start])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a color-changing relay line from the unique frame start through offset-crossbar T arrows.

    Input structure:
    - A monochrome one-cell perimeter frame has exactly one differently colored, non-corner start pixel.
    - The interior background contains separated monochrome six-pixel T arrows: a four-pixel stem
      crossed by a three-pixel bar one step from one endpoint, plus off-path arrows as distractors.

    Rule:
    - Enter perpendicular to the side containing the start and paint the first ray in the start color.
    - On hitting any pixel of an arrow, paint that collision pixel with the incoming color, jump to
      the nearer stem endpoint, then follow the arrow's direction using that arrow's color.
    - Repeat until the current ray reaches the frame, painting that frame collision pixel too;
      arrows never hit by the relay remain unchanged.

    Key insight:
    - The crossbar offset uniquely identifies each arrow's tip and direction, while segment colors
      record the most recently hit arrow, so the complete result requires causal step-by-step tracing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 7 and width >= 7, "grid is too small for framed arrow sprites"

    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    _frame_color, position, direction, segment_color = _frame_start(g)
    arrows = _find_arrows(g, bg)
    assert arrows, "expected at least one arrow sprite"

    at_cell: dict[tuple[int, int], int] = {}
    for index, arrow in enumerate(arrows):
        for cell in arrow["cells"]:
            assert cell not in at_cell, "arrow sprites may not overlap"
            at_cell[cell] = index

    out = g.copy()
    used: set[int] = set()
    # A valid relay is acyclic; this guard also turns malformed loops into a clear failure.
    for _ in range(height * width + len(arrows) + 1):
        r = position[0] + direction[0]
        c = position[1] + direction[1]
        assert 0 <= r < height and 0 <= c < width, "a relay ray left the grid without meeting its frame"
        position = (r, c)

        if r in (0, height - 1) or c in (0, width - 1):
            out[r, c] = segment_color
            return out

        hit = at_cell.get(position)
        out[r, c] = segment_color
        if hit is None:
            continue

        assert hit not in used, "relay chain must not revisit an arrow"
        used.add(hit)
        arrow = arrows[hit]
        position = arrow["tip"]
        direction = arrow["direction"]
        segment_color = arrow["color"]
        out[position] = segment_color

    raise AssertionError("relay did not terminate at the frame")


def _cells_for(tip: tuple[int, int], direction: tuple[int, int]) -> frozenset[tuple[int, int]]:
    tr, tc = tip
    return frozenset((tr + dr, tc + dc) for dr, dc in ARROW_TEMPLATES[direction])


def _on_border(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    height, width = shape
    return r in (0, height - 1) or c in (0, width - 1)


def _ray_event(
    position: tuple[int, int],
    direction: tuple[int, int],
    arrows: list[dict],
    shape: tuple[int, int],
) -> tuple[str, int | None, tuple[int, int], list[tuple[int, int]]]:
    """Trace to the first original arrow pixel or frame cell."""
    occupied = {cell: i for i, arrow in enumerate(arrows) for cell in arrow["cells"]}
    cells: list[tuple[int, int]] = []
    r, c = position
    for _ in range(max(shape) + 1):
        r += direction[0]
        c += direction[1]
        cell = (r, c)
        cells.append(cell)
        if _on_border(cell, shape):
            return "border", None, cell, cells
        if cell in occupied:
            return "arrow", occupied[cell], cell, cells
    raise AssertionError("ray failed to meet an arrow or border")


def _far_enough(cells: frozenset[tuple[int, int]], arrows: list[dict], gap: int = 1) -> bool:
    """Require at least `gap` empty Chebyshev rings between arrow sprites."""
    for r, c in cells:
        for arrow in arrows:
            if any(max(abs(r - rr), abs(c - cc)) <= gap for rr, cc in arrow["cells"]):
                return False
    return True


def _prospective_arrows(
    position: tuple[int, int],
    incoming: tuple[int, int],
    arrows: list[dict],
    trace: set[tuple[int, int]],
    shape: tuple[int, int],
    preferred: list[tuple[int, int]],
) -> list[tuple[dict, tuple[int, int], list[tuple[int, int]]]]:
    """Enumerate unambiguous next arrows hit away from their tips."""
    height, width = shape
    result: list[tuple[dict, tuple[int, int], list[tuple[int, int]]]] = []
    for outgoing in preferred:
        if outgoing == OPPOSITE[incoming]:
            continue
        for tr in range(2, height - 2):
            for tc in range(2, width - 2):
                tip = (tr, tc)
                cells = _cells_for(tip, outgoing)
                if not all(0 < r < height - 1 and 0 < c < width - 1 for r, c in cells):
                    continue
                if not _far_enough(cells, arrows, gap=1):
                    continue
                candidate = {"cells": cells, "tip": tip, "direction": outgoing}
                kind, index, collision, line = _ray_event(position, incoming, arrows + [candidate], shape)
                if kind != "arrow" or index != len(arrows):
                    continue
                # Every relay visibly needs the jump: the first collision is not the tip,
                # and the tip lies on a parallel/perpendicular coordinate line.
                if collision == tip:
                    continue
                if incoming[0] == 0 and collision[0] == tip[0]:
                    continue
                if incoming[1] == 0 and collision[1] == tip[1]:
                    continue
                if not 4 <= len(line) <= 11:
                    continue
                if any(cell in trace for cell in line[:-1]):
                    continue
                result.append((candidate, collision, line))
    random.shuffle(result)
    return result


def _known_simulation(
    grid: np.ndarray,
    start: tuple[int, int],
    initial_direction: tuple[int, int],
    start_color: int,
    arrows: list[dict],
) -> tuple[np.ndarray, list[int], list[dict]]:
    """Independent generator oracle using the already known arrow identities."""
    out = grid.copy()
    position = start
    direction = initial_direction
    color = int(start_color)
    visited: list[int] = []
    events: list[dict] = []
    for _ in range(len(arrows) + 1):
        kind, index, collision, line = _ray_event(position, direction, arrows, grid.shape)
        for cell in line:
            out[cell] = color
        if kind == "border":
            events.append({"kind": kind, "line": line, "color": color, "collision": collision})
            return out, visited, events
        assert index is not None and index not in visited
        arrow = arrows[index]
        events.append(
            {
                "kind": kind,
                "line": line,
                "color": color,
                "collision": collision,
                "tip": arrow["tip"],
                "incoming": direction,
                "outgoing": arrow["direction"],
            }
        )
        visited.append(index)
        position = arrow["tip"]
        direction = arrow["direction"]
        color = int(arrow["color"])
        out[position] = color
    raise AssertionError("known relay did not terminate")


def _choose_start(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    height, width = shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return (0, random.randint(3, width - 4)), DOWN
    if side == "bottom":
        return (height - 1, random.randint(3, width - 4)), UP
    if side == "left":
        return (random.randint(3, height - 4), 0), RIGHT
    return (random.randint(3, height - 4), width - 1), LEFT


def _place_decoys(
    arrows: list[dict],
    trace: set[tuple[int, int]],
    shape: tuple[int, int],
    count: int,
) -> list[dict] | None:
    """Place off-path arrows; prefer a close decoy that punishes path-free selection."""
    height, width = shape
    decoys: list[dict] = []
    for decoy_index in range(count):
        candidates: list[dict] = []
        for direction in DIRECTIONS:
            for tr in range(2, height - 2):
                for tc in range(2, width - 2):
                    cells = _cells_for((tr, tc), direction)
                    if not all(0 < r < height - 1 and 0 < c < width - 1 for r, c in cells):
                        continue
                    if not _far_enough(cells, arrows + decoys, gap=1):
                        continue
                    if any(cell in trace for cell in cells):
                        continue
                    distance = min(abs(r - rr) + abs(c - cc) for r, c in cells for rr, cc in trace)
                    if distance < 2:
                        continue
                    candidates.append(
                        {"cells": cells, "tip": (tr, tc), "direction": direction, "distance": distance}
                    )
        if not candidates:
            return None
        if decoy_index == 0:
            near = [candidate for candidate in candidates if candidate["distance"] <= 5]
            if near:
                candidates = near
        chosen = random.choice(candidates)
        chosen.pop("distance")
        decoys.append(chosen)
    return decoys


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIDE, MAX_SIDE)
    width = random.randint(MIN_SIDE, MAX_SIDE)
    shape = (height, width)
    chain_count = random.randint(MIN_CHAIN, MAX_CHAIN)
    decoy_count = random.randint(MIN_DECOYS, MAX_DECOYS)
    total_arrow_count = chain_count + decoy_count

    colors = random.sample(range(10), 3 + total_arrow_count)
    bg, frame_color, start_color = colors[:3]
    arrow_colors = colors[3:]
    start, incoming = _choose_start(shape)

    arrows: list[dict] = []
    trace: set[tuple[int, int]] = {start}
    position = start
    directions_used: list[tuple[int, int]] = []
    incoming_directions: list[tuple[int, int]] = []
    straight_count = 0

    for chain_index in range(chain_count):
        # Same-direction relays are a real branch, but turns are deliberately common.
        preferred = [d for d in DIRECTIONS if d != OPPOSITE[incoming]]
        if chain_index < 2:
            turns = [d for d in preferred if d != incoming]
            straights = [d for d in preferred if d == incoming]
            preferred = turns + straights
        candidates = _prospective_arrows(position, incoming, arrows, trace, shape, preferred)
        if not candidates:
            return None

        # Weight the first half toward turns without excluding the straight/shift branch.
        turn_candidates = [item for item in candidates if item[0]["direction"] != incoming]
        straight_candidates = [item for item in candidates if item[0]["direction"] == incoming]
        must_cover_straight = chain_index == chain_count - 1 and straight_count == 0
        if must_cover_straight:
            if not straight_candidates:
                return None
            candidate, _collision, line = random.choice(straight_candidates)
        elif turn_candidates and (chain_index < 2 or random.random() < 0.55):
            candidate, _collision, line = random.choice(turn_candidates)
        else:
            candidate, _collision, line = random.choice(candidates)
        candidate["color"] = int(arrow_colors[chain_index])
        arrows.append(candidate)
        trace.update(line)
        trace.add(candidate["tip"])
        incoming_directions.append(incoming)
        directions_used.append(candidate["direction"])
        straight_count += int(candidate["direction"] == incoming)
        position = candidate["tip"]
        incoming = candidate["direction"]

    # Ensure orientation decoding, not mere straight continuation, controls the chain.
    turn_count = sum(a != b for a, b in zip(incoming_directions, directions_used))
    if turn_count < 2 or straight_count < 1:
        return None

    kind, index, _collision, final_line = _ray_event(position, incoming, arrows, shape)
    if kind != "border" or index is not None or len(final_line) < 4:
        return None
    if any(cell in trace for cell in final_line[:-1]):
        return None
    trace.update(final_line)

    decoys = _place_decoys(arrows, trace, shape, decoy_count)
    if decoys is None:
        return None
    for i, decoy in enumerate(decoys):
        decoy["color"] = int(arrow_colors[chain_count + i])
    all_arrows = arrows + decoys

    grid = np.full(shape, int(bg), dtype=int)
    grid[0, :] = int(frame_color)
    grid[-1, :] = int(frame_color)
    grid[:, 0] = int(frame_color)
    grid[:, -1] = int(frame_color)
    grid[start] = int(start_color)
    for arrow in all_arrows:
        for cell in arrow["cells"]:
            assert int(grid[cell]) == int(bg)
            grid[cell] = int(arrow["color"])

    # Semantic oracle: validate the intended causal order, jump dependence, decoys,
    # visible color changes, and exact agreement with the independently parsed solver.
    try:
        expected, visited, events = _known_simulation(grid, start, _choose_inward(start, shape), start_color, all_arrows)
        solved = main(grid)
    except (AssertionError, ValueError):
        return None
    if visited != list(range(chain_count)):
        return None
    if not np.array_equal(solved, expected) or np.array_equal(solved, grid):
        return None
    if len(events) != chain_count + 1 or any(len(event["line"]) < 4 for event in events):
        return None
    for event in events[:-1]:
        collision = event["collision"]
        tip = event["tip"]
        if collision == tip:
            return None
        incoming_dir = event["incoming"]
        if incoming_dir[0] == 0 and collision[0] == tip[0]:
            return None
        if incoming_dir[1] == 0 and collision[1] == tip[1]:
            return None
    for decoy in decoys:
        if any(int(expected[cell]) != int(grid[cell]) for cell in decoy["cells"]):
            return None
    return grid


def _choose_inward(start: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    height, width = shape
    if r == 0:
        return DOWN
    if r == height - 1:
        return UP
    if c == 0:
        return RIGHT
    assert c == width - 1
    return LEFT


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked framed relay with both chain and decoy arrows."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid arrow relay after {MAX_GENERATION_ATTEMPTS} attempts")
