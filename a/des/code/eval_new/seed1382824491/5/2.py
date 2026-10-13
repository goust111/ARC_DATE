from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _distance_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return abs(a[0] - b[0]), abs(a[1] - b[1])


def _non_bg_components(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            comp_idx = len(components)
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                cell_to_comp[(rr, cc)] = comp_idx
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components, cell_to_comp


def _extract_structure(
    grid: np.ndarray,
) -> tuple[
    set[tuple[int, int]],
    list[tuple[int, int, int]],
    int,
    tuple[int, int],
    dict[int, list[tuple[int, int]]],
    dict[tuple[int, int], int],
]:
    components, cell_to_comp = _non_bg_components(grid)
    assert components, "expected at least one non-background component"

    color_cells: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(grid != BG):
        color = int(grid[int(r), int(c)])
        color_cells.setdefault(color, []).append((int(r), int(c)))

    source_candidates: list[int] = []
    for comp_idx, cells in enumerate(components):
        counts: dict[int, int] = {}
        for r, c in cells:
            color = int(grid[r, c])
            counts[color] = counts.get(color, 0) + 1
        if len(cells) > 1 and len(counts) >= 2 and any(count > 1 for count in counts.values()):
            source_candidates.append(comp_idx)

    assert len(source_candidates) == 1, "expected a unique multicolor source component"
    source_idx = source_candidates[0]

    source_cells = set(components[source_idx])
    source_colors = {int(grid[r, c]) for r, c in source_cells}
    assert 2 <= len(source_colors) <= 3

    marker_candidates = []
    for color, coords in color_cells.items():
        inside = [coord for coord in coords if coord in source_cells]
        outside = [coord for coord in coords if coord not in source_cells]
        if len(inside) == 1 and len(outside) == 2:
            marker_candidates.append((color, inside[0]))
    assert len(marker_candidates) == 1, "expected a unique start-marker color"
    start_color, start_pos = marker_candidates[0]

    signposts_by_color: dict[int, list[tuple[int, int]]] = {}
    signpost_color_at: dict[tuple[int, int], int] = {}
    outside_cells = set()
    for color, coords in color_cells.items():
        outside = [coord for coord in coords if coord not in source_cells]
        if not outside:
            continue
        assert len(outside) == 2, "every signpost color must appear exactly twice outside the source"
        signposts_by_color[color] = outside
        for coord in outside:
            signpost_color_at[coord] = color
            outside_cells.add(coord)

    assert start_color in signposts_by_color
    assert outside_cells == {
        (int(r), int(c))
        for r, c in np.argwhere(grid != BG)
        if (int(r), int(c)) not in source_cells
    }

    source_offsets = [
        (r - start_pos[0], c - start_pos[1], int(grid[r, c]))
        for r, c in source_cells
    ]
    return source_cells, source_offsets, start_color, start_pos, signposts_by_color, signpost_color_at


def _build_chain(
    start_pos: tuple[int, int],
    start_color: int,
    signposts_by_color: dict[int, list[tuple[int, int]]],
    signpost_color_at: dict[tuple[int, int], int],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    chain: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    visited_colors: set[int] = set()
    current_anchor = start_pos
    current_color = start_color

    while True:
        assert current_color in signposts_by_color
        pair = signposts_by_color[current_color]
        ordered = sorted(pair, key=lambda pos: _distance_key(pos, current_anchor))
        assert _distance_key(ordered[0], current_anchor) != _distance_key(ordered[1], current_anchor)
        entrance, exit_pos = ordered
        chain.append((current_color, entrance, exit_pos))
        visited_colors.add(current_color)

        adjacent_next = []
        for nbr in _neighbors4(*exit_pos):
            if nbr in signpost_color_at and signpost_color_at[nbr] != current_color:
                adjacent_next.append(nbr)
        assert len(adjacent_next) <= 1, "each exit may have at most one next-link clue"
        if not adjacent_next:
            break

        next_color = signpost_color_at[adjacent_next[0]]
        assert next_color not in visited_colors, "teleport chain must not cycle"
        current_anchor = exit_pos
        current_color = next_color

    assert visited_colors == set(signposts_by_color), "all signpost colors must belong to the same chain"
    return chain


def _paint_copies(
    base: np.ndarray,
    source_offsets: list[tuple[int, int, int]],
    start_color: int,
    chain: list[tuple[int, tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    out = base.copy()
    h, w = out.shape
    for station_color, entrance, _exit_pos in chain:
        for dr, dc, color in source_offsets:
            rr = entrance[0] + dr
            cc = entrance[1] + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = station_color if color == start_color else color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the source pattern through a hidden signpost chain and recolor the anchor pixel at each stop.

    Input structure:
    - Black is the background.
    - One compact multicolor source pattern contains a unique single-pixel start marker whose color also appears on one signpost pair.
    - Outside the source, every signpost color appears on exactly two single cells; some differently colored signposts may touch to encode chain links.

    Rule:
    - Find the start-marker color, then locate its two outside signposts.
    - Relative to the current anchor point, the nearer same-color signpost under row-priority comparison (first absolute row difference, then absolute column difference) is the station entrance and the other is the exit.
    - If the exit has one 4-neighbor signpost pixel of a different color, that color gives the next pair and the same entrance/exit rule repeats from the current exit; otherwise the chain ends.
    - At every station entrance, copy the whole source pattern with the start-marker cell recolored to that station color while all other source colors stay unchanged.

    Key insight:
    - The puzzle is solved by separating the source component from the scattered signposts, then reading the adjacent exit-to-next-entrance clues as a directed teleport chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _source_cells, source_offsets, start_color, start_pos, signposts_by_color, signpost_color_at = _extract_structure(g)
    chain = _build_chain(start_pos, start_color, signposts_by_color, signpost_color_at)
    return _paint_copies(g, source_offsets, start_color, chain)


def _translated_cells(offsets: list[tuple[int, int]], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    return {(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets}


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _is_edge(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _pick_cell(candidates: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int]:
    edge = [cell for cell in candidates if _is_edge(cell, shape)]
    interior = [cell for cell in candidates if not _is_edge(cell, shape)]
    if edge and interior:
        return random.choice(edge if random.random() < 0.5 else interior)
    return random.choice(candidates)


def _random_pattern(
    station_colors: list[int], body_color_count: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    ph = random.randint(3, 5)
    pw = random.randint(3, 5)
    anchor = (random.randrange(ph), random.randrange(pw))
    target_area = random.randint(5, min(ph * pw, 9))

    cells = {anchor}
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors4(*anchor)
        if 0 <= nr < ph and 0 <= nc < pw
    }
    while len(cells) < target_area:
        if not frontier:
            return _random_pattern(station_colors, body_color_count)
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in cells:
                frontier.add((nr, nc))

    other_cells = [cell for cell in cells if cell != anchor]
    random.shuffle(other_cells)
    body_colors = random.sample([c for c in range(1, 10) if c not in station_colors], body_color_count)

    color_at = {anchor: station_colors[0]}
    if body_color_count == 1:
        for cell in other_cells:
            color_at[cell] = body_colors[0]
    else:
        split = random.randint(2, len(other_cells) - 2)
        first_group = set(other_cells[:split])
        for cell in other_cells:
            color_at[cell] = body_colors[0] if cell in first_group else body_colors[1]

    offsets_with_color = [
        (r - anchor[0], c - anchor[1], color_at[(r, c)])
        for r, c in cells
    ]
    shape_offsets = [(dr, dc) for dr, dc, _color in offsets_with_color]
    return offsets_with_color, shape_offsets


def _source_anchor_candidates(
    shape: tuple[int, int], offsets: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    h, w = shape
    min_dr = min(dr for dr, _dc in offsets)
    max_dr = max(dr for dr, _dc in offsets)
    min_dc = min(dc for _dr, dc in offsets)
    max_dc = max(dc for _dr, dc in offsets)

    for margin in (2, 1):
        rows = range(margin - min_dr, h - margin - max_dr)
        cols = range(margin - min_dc, w - margin - max_dc)
        candidates = [(r, c) for r in rows for c in cols]
        if candidates:
            return candidates
    return []


def _entrance_candidates(
    shape: tuple[int, int],
    offsets: list[tuple[int, int]],
    reserved_patterns: set[tuple[int, int]],
    signposts: set[tuple[int, int]],
    forbidden_signposts: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            anchor = (r, c)
            if anchor in forbidden_signposts or anchor in signposts:
                continue
            if any(nbr in signposts for nbr in _neighbors4(r, c)):
                continue
            cells = _translated_cells(offsets, anchor)
            if any(not _in_bounds(shape, rr, cc) for rr, cc in cells):
                continue
            if cells & reserved_patterns:
                continue
            if any(cell in signposts for cell in cells if cell != anchor):
                continue
            candidates.append(anchor)
    return candidates


def _link_exit_candidates(
    shape: tuple[int, int],
    next_entrance: tuple[int, int],
    previous_anchor: tuple[int, int],
    current_entrance: tuple[int, int],
    reserved_patterns: set[tuple[int, int]],
    signposts: set[tuple[int, int]],
    forbidden_signposts: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    current_key = _distance_key(current_entrance, previous_anchor)
    for rr, cc in _neighbors4(*next_entrance):
        pos = (rr, cc)
        if not _in_bounds(shape, rr, cc):
            continue
        if pos in forbidden_signposts or pos in signposts or pos in reserved_patterns:
            continue
        if _distance_key(pos, previous_anchor) <= current_key:
            continue
        adjacent = {nbr for nbr in _neighbors4(rr, cc) if nbr in signposts}
        if adjacent != {next_entrance}:
            continue
        candidates.append(pos)
    return candidates


def _terminal_exit_candidates(
    shape: tuple[int, int],
    previous_anchor: tuple[int, int],
    current_entrance: tuple[int, int],
    reserved_patterns: set[tuple[int, int]],
    signposts: set[tuple[int, int]],
    forbidden_signposts: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    current_key = _distance_key(current_entrance, previous_anchor)
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            if pos in forbidden_signposts or pos in signposts or pos in reserved_patterns:
                continue
            if _distance_key(pos, previous_anchor) <= current_key:
                continue
            if any(nbr in signposts for nbr in _neighbors4(r, c)):
                continue
            candidates.append(pos)
    return candidates


def _render_input(
    shape: tuple[int, int],
    source_anchor: tuple[int, int],
    pattern: list[tuple[int, int, int]],
    station_colors: list[int],
    entrances: list[tuple[int, int]],
    exits: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for dr, dc, color in pattern:
        grid[source_anchor[0] + dr, source_anchor[1] + dc] = color
    for color, entrance, exit_pos in zip(station_colors, entrances, exits):
        grid[entrance] = color
        grid[exit_pos] = color
    return grid


def _expected_output(
    grid: np.ndarray,
    pattern: list[tuple[int, int, int]],
    start_color: int,
    station_colors: list[int],
    entrances: list[tuple[int, int]],
) -> np.ndarray:
    chain = [(color, entrance, entrance) for color, entrance in zip(station_colors, entrances)]
    return _paint_copies(grid, pattern, start_color, chain)


def _attempt_generate() -> np.ndarray | None:
    station_count = random.choice([1, 2, 3, 4])
    body_color_count = random.choice([1, 2])
    station_colors = random.sample(range(1, 10), station_count)
    pattern, shape_offsets = _random_pattern(station_colors, body_color_count)

    low_side = max(14, 10 + 2 * station_count)
    height = random.randint(low_side, 24)
    width = random.randint(low_side, 24)
    shape = (height, width)

    src_candidates = _source_anchor_candidates(shape, shape_offsets)
    if not src_candidates:
        return None
    source_anchor = random.choice(src_candidates)
    source_cells = _translated_cells(shape_offsets, source_anchor)
    forbidden_signposts = _halo(source_cells, shape)

    reserved_patterns = set(source_cells)
    signposts: set[tuple[int, int]] = set()
    entrances: list[tuple[int, int]] = []
    exits: list[tuple[int, int]] = []

    first_candidates = _entrance_candidates(shape, shape_offsets, reserved_patterns, signposts, forbidden_signposts)
    if not first_candidates:
        return None
    first_entrance = _pick_cell(first_candidates, shape)
    entrances.append(first_entrance)
    signposts.add(first_entrance)
    reserved_patterns |= _translated_cells(shape_offsets, first_entrance)

    for idx in range(station_count - 1):
        prev_anchor = source_anchor if idx == 0 else exits[-1]
        current_entrance = entrances[idx]
        next_candidates = _entrance_candidates(shape, shape_offsets, reserved_patterns, signposts, forbidden_signposts)
        random.shuffle(next_candidates)

        chosen_entrance = None
        chosen_exit = None
        for candidate in next_candidates:
            new_copy = _translated_cells(shape_offsets, candidate)
            trial_signposts = set(signposts)
            trial_signposts.add(candidate)
            exit_candidates = _link_exit_candidates(
                shape,
                candidate,
                prev_anchor,
                current_entrance,
                reserved_patterns | new_copy,
                trial_signposts,
                forbidden_signposts,
            )
            if exit_candidates:
                chosen_entrance = candidate
                chosen_exit = _pick_cell(exit_candidates, shape)
                break
        if chosen_entrance is None or chosen_exit is None:
            return None

        entrances.append(chosen_entrance)
        signposts.add(chosen_entrance)
        reserved_patterns |= _translated_cells(shape_offsets, chosen_entrance)
        exits.append(chosen_exit)
        signposts.add(chosen_exit)

    prev_anchor = source_anchor if station_count == 1 else exits[-1]
    last_entrance = entrances[-1]
    terminal_choices = _terminal_exit_candidates(
        shape,
        prev_anchor,
        last_entrance,
        reserved_patterns,
        signposts,
        forbidden_signposts,
    )
    if not terminal_choices:
        return None
    last_exit = _pick_cell(terminal_choices, shape)
    exits.append(last_exit)
    signposts.add(last_exit)

    for idx, entrance in enumerate(entrances):
        copy_cells = _translated_cells(shape_offsets, entrance)
        forbidden_covered = signposts - {entrance}
        if copy_cells & forbidden_covered:
            return None
        if idx > 0:
            prev_exit = exits[idx - 1]
            assert prev_exit in _neighbors4(*entrance)

    signpost_neighbors = {
        cell: {nbr for nbr in _neighbors4(*cell) if nbr in signposts}
        for cell in signposts
    }
    for idx, entrance in enumerate(entrances):
        expected = {exits[idx - 1]} if idx > 0 else set()
        if signpost_neighbors[entrance] != expected:
            return None
    for idx, exit_pos in enumerate(exits):
        expected = {entrances[idx + 1]} if idx + 1 < len(entrances) else set()
        other_color_neighbors = {
            nbr
            for nbr in signpost_neighbors[exit_pos]
            if nbr not in {entrances[idx], exits[idx]}
        }
        if other_color_neighbors != expected:
            return None
        prev_anchor = source_anchor if idx == 0 else exits[idx - 1]
        if not (_distance_key(entrances[idx], prev_anchor) < _distance_key(exit_pos, prev_anchor)):
            return None

    grid = _render_input(shape, source_anchor, pattern, station_colors, entrances, exits)
    expected = _expected_output(grid, pattern, station_colors[0], station_colors, entrances)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-chain puzzle")
