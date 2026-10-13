from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
MIRROR_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
HORIZONTAL = 1
VERTICAL = 2

GRID_MIN = 12
GRID_MAX = 24
RETRY_LIMIT = 500
DISTRACTOR_LIMIT = 3

SCENARIO_PATTERNS = (
    (),
    ("H",),
    ("V",),
    ("H", "V"),
    ("V", "H"),
    ("H", "H", "V"),
    ("V", "V", "H"),
    ("H", "V", "V"),
    ("V", "H", "H"),
)


def _in_bounds(height: int, width: int, r: int, c: int) -> bool:
    return 0 <= r < height and 0 <= c < width


def _is_edge(height: int, width: int, r: int, c: int) -> bool:
    return r == 0 or r == height - 1 or c == 0 or c == width - 1


def _find_source_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    yellow_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)]
    assert len(yellow_cells) == 2, "expected exactly two yellow marker pixels"

    height, width = grid.shape
    edge_cells = [rc for rc in yellow_cells if _is_edge(height, width, rc[0], rc[1])]
    assert len(edge_cells) == 1, "expected exactly one edge yellow source"

    source = edge_cells[0]
    guide = yellow_cells[0] if yellow_cells[1] == source else yellow_cells[1]
    dr = int(guide[0] - source[0])
    dc = int(guide[1] - source[1])
    assert abs(dr) == 1 and abs(dc) == 1, "yellow markers must be diagonal neighbors"
    return source, guide, (dr, dc)


def _mirror_kind_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    kinds = np.zeros((height, width), dtype=np.int8)
    seen = np.zeros((height, width), dtype=bool)

    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color in (BLACK, YELLOW) or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not _in_bounds(height, width, nr, nc):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = sorted({rr for rr, _ in coords})
            cols = sorted({cc for _, cc in coords})
            if len(rows) == 1:
                expected = list(range(cols[0], cols[-1] + 1))
                assert cols == expected, "horizontal mirror must be contiguous"
                kind = HORIZONTAL
            else:
                assert len(cols) == 1, "mirror must be horizontal or vertical"
                expected = list(range(rows[0], rows[-1] + 1))
                assert rows == expected, "vertical mirror must be contiguous"
                kind = VERTICAL

            for rr, cc in coords:
                kinds[rr, cc] = kind

    return kinds


def _trace_beam(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, str]], set[tuple[int, int]], bool]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    _source, guide, (dr, dc) = _find_source_and_direction(g)
    mirror_kinds = _mirror_kind_grid(g)

    out = g.copy()
    visited_cells = set([_source, guide])
    events: list[tuple[int, int, str]] = []
    seen_states: set[tuple[int, int, int, int]] = set()

    r, c = guide
    while True:
        state = (r, c, dr, dc)
        if state in seen_states:
            return out, events, visited_cells, True
        seen_states.add(state)

        nr = r + dr
        nc = c + dc
        if not _in_bounds(height, width, nr, nc):
            break

        visited_cells.add((nr, nc))
        kind = int(mirror_kinds[nr, nc])
        if kind == HORIZONTAL:
            events.append((nr, nc, "H"))
            dr = -dr
        elif kind == VERTICAL:
            events.append((nr, nc, "V"))
            dc = -dc
        elif int(g[nr, nc]) == BLACK:
            out[nr, nc] = YELLOW

        r, c = nr, nc

    return out, events, visited_cells, False


def main(grid: np.ndarray) -> np.ndarray:
    """Trace a diagonal yellow beam that reflects from horizontal and vertical colored mirrors."""
    out, _events, _visited_cells, cycled = _trace_beam(np.asarray(grid, dtype=int))
    assert not cycled, "beam path should stop at the boundary"
    return out


def _reflect(direction: tuple[int, int], kind: str) -> tuple[int, int]:
    dr, dc = direction
    if kind == "H":
        return (-dr, dc)
    assert kind == "V"
    return (dr, -dc)


def _sample_source(height: int, width: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    choices = []
    if width >= 2:
        choices.append(("top", (1, 1)))
        choices.append(("top", (1, -1)))
        choices.append(("bottom", (-1, 1)))
        choices.append(("bottom", (-1, -1)))
    if height >= 2:
        choices.append(("left", (-1, 1)))
        choices.append(("left", (1, 1)))
        choices.append(("right", (-1, -1)))
        choices.append(("right", (1, -1)))

    for _ in range(40):
        side, direction = random.choice(choices)
        dr, dc = direction
        if side == "top":
            if dc == 1:
                c = random.randint(0, width - 3)
            else:
                c = random.randint(2, width - 1)
            source = (0, c)
        elif side == "bottom":
            if dc == 1:
                c = random.randint(0, width - 3)
            else:
                c = random.randint(2, width - 1)
            source = (height - 1, c)
        elif side == "left":
            if dr == 1:
                r = random.randint(0, height - 3)
            else:
                r = random.randint(2, height - 1)
            source = (r, 0)
        else:
            if dr == 1:
                r = random.randint(0, height - 3)
            else:
                r = random.randint(2, height - 1)
            source = (r, width - 1)

        guide = (source[0] + dr, source[1] + dc)
        if _in_bounds(height, width, guide[0], guide[1]):
            return source, guide, direction

    raise RuntimeError("failed to sample a valid source")


def _forward_free_cells(
    height: int,
    width: int,
    occupied: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    r = start[0] + dr
    c = start[1] + dc
    while _in_bounds(height, width, r, c) and not bool(occupied[r, c]):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _ray_length_to_boundary(height: int, width: int, start: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    length = 0
    r = start[0] + dr
    c = start[1] + dc
    while _in_bounds(height, width, r, c):
        length += 1
        r += dr
        c += dc
    return length


def _spec_cells(spec: dict[str, int | str]) -> list[tuple[int, int]]:
    if spec["kind"] == "H":
        row = int(spec["row"])
        return [(row, c) for c in range(int(spec["start"]), int(spec["end"]) + 1)]
    col = int(spec["col"])
    return [(r, col) for r in range(int(spec["start"]), int(spec["end"]) + 1)]


def _mark_spec(occupied: np.ndarray, spec: dict[str, int | str]) -> None:
    for r, c in _spec_cells(spec):
        occupied[r, c] = True


def _segment_candidates(
    height: int,
    width: int,
    hit: tuple[int, int],
    kind: str,
    occupied: np.ndarray,
) -> list[dict[str, int | str]]:
    r, c = hit
    if kind == "H":
        if r in (0, height - 1):
            return []
    else:
        if c in (0, width - 1):
            return []

    candidates: list[dict[str, int | str]] = []
    lengths = list(range(3, 7))
    random.shuffle(lengths)
    for length in lengths:
        if kind == "H":
            starts = list(range(c - length + 1, c + 1))
            random.shuffle(starts)
            for start in starts:
                end = start + length - 1
                if start < 0 or end >= width:
                    continue
                rr0 = max(0, r - 1)
                rr1 = min(height - 1, r + 1)
                cc0 = max(0, start - 1)
                cc1 = min(width - 1, end + 1)
                if bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                    continue
                candidates.append({"kind": "H", "row": r, "start": start, "end": end})
        else:
            starts = list(range(r - length + 1, r + 1))
            random.shuffle(starts)
            for start in starts:
                end = start + length - 1
                if start < 0 or end >= height:
                    continue
                rr0 = max(0, start - 1)
                rr1 = min(height - 1, end + 1)
                cc0 = max(0, c - 1)
                cc1 = min(width - 1, c + 1)
                if bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                    continue
                candidates.append({"kind": "V", "col": c, "start": start, "end": end})
    return candidates


def _render_grid(
    height: int,
    width: int,
    source: tuple[int, int],
    guide: tuple[int, int],
    specs: list[dict[str, int | str]],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[source] = YELLOW
    grid[guide] = YELLOW

    colors = random.sample(MIRROR_COLORS, len(specs))
    for spec, color in zip(specs, colors):
        for r, c in _spec_cells(spec):
            grid[r, c] = color
    return grid


def _add_distractors(
    height: int,
    width: int,
    occupied: np.ndarray,
    beam_cells: set[tuple[int, int]],
    min_count: int,
) -> list[dict[str, int | str]] | None:
    target = random.randint(min_count, DISTRACTOR_LIMIT)
    added: list[dict[str, int | str]] = []
    attempts = 0
    while len(added) < target and attempts < 120:
        attempts += 1
        kind = random.choice(("H", "V"))
        if kind == "H":
            hit = (random.randint(1, height - 2), random.randint(0, width - 1))
        else:
            hit = (random.randint(0, height - 1), random.randint(1, width - 2))

        candidates = _segment_candidates(height, width, hit, kind, occupied)
        random.shuffle(candidates)
        for spec in candidates:
            cells = _spec_cells(spec)
            if any(cell in beam_cells for cell in cells):
                continue
            _mark_spec(occupied, spec)
            added.append(spec)
            break

    if len(added) < min_count:
        return None
    return added


def _generate_attempt() -> np.ndarray | None:
    pattern = random.choice(SCENARIO_PATTERNS)
    min_size = max(GRID_MIN, 10 + 2 * len(pattern))
    height = random.randint(min_size, GRID_MAX)
    width = random.randint(min_size, GRID_MAX)

    source, guide, direction = _sample_source(height, width)
    occupied = np.zeros((height, width), dtype=bool)
    occupied[source] = True
    occupied[guide] = True

    specs: list[dict[str, int | str]] = []
    current = guide
    current_direction = direction

    for idx, kind in enumerate(pattern):
        free_cells = _forward_free_cells(height, width, occupied, current, current_direction)
        if len(free_cells) < 2:
            return None

        remaining = len(pattern) - idx - 1
        positions = free_cells[1:] if len(free_cells) > 2 else free_cells
        random.shuffle(positions)

        placed = False
        for hit in positions:
            new_direction = _reflect(current_direction, kind)
            if not _in_bounds(height, width, hit[0] + new_direction[0], hit[1] + new_direction[1]):
                continue

            candidates = _segment_candidates(height, width, hit, kind, occupied)
            random.shuffle(candidates)
            for spec in candidates:
                temp_occupied = occupied.copy()
                _mark_spec(temp_occupied, spec)
                next_free = _forward_free_cells(height, width, temp_occupied, hit, new_direction)
                if len(next_free) < remaining:
                    continue
                _mark_spec(occupied, spec)
                specs.append(spec)
                current = hit
                current_direction = new_direction
                placed = True
                break
            if placed:
                break

        if not placed:
            return None

    final_free = _forward_free_cells(height, width, occupied, current, current_direction)
    if len(final_free) != _ray_length_to_boundary(height, width, current, current_direction):
        return None

    primary_grid = _render_grid(height, width, source, guide, specs)
    primary_out, events, beam_cells, cycled = _trace_beam(primary_grid)
    if cycled or tuple(kind for _, _, kind in events) != pattern:
        return None

    distractors = _add_distractors(
        height,
        width,
        occupied.copy(),
        beam_cells,
        1 if not specs else 0,
    )
    if distractors is None:
        return None

    grid = _render_grid(height, width, source, guide, specs + distractors)
    out, events, _beam_cells, cycled = _trace_beam(grid)
    if cycled or tuple(kind for _, _, kind in events) != pattern:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-reflection puzzle")
