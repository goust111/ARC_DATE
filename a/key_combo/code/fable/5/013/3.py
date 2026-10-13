from __future__ import annotations

import random
from collections import Counter

import numpy as np


# The two mirror colors are visible ARC conventions from the puzzle statement.
SLASH = 8      # "/" mirror
BACKSLASH = 7  # "\\" mirror
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size, "expected a nonempty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells += [(r1, c) for c in range(c0, c1 + 1)]
    cells += [(r, c0) for r in range(r0 + 1, r1)]
    cells += [(r, c1) for r in range(r0 + 1, r1)]
    return cells


def _find_frame(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], set[tuple[int, int]]]:
    """Find the unique nearly complete monochrome rectangular perimeter."""
    candidates: list[tuple[int, int, tuple[int, int, int, int], set[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        if color in (bg, SLASH, BACKSLASH):
            continue
        mask = grid == color
        if int(mask.sum()) < 12:
            continue
        b = _bbox_of(mask)
        r0, c0, r1, c1 = b
        if r1 - r0 + 1 < 7 or c1 - c0 + 1 < 7:
            continue
        if not all(int(grid[r, c]) == color for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
            continue
        border = _perimeter(b)
        wall_count = sum(int(grid[r, c]) == color for r, c in border)
        # Several one-cell holes are allowed, but the visible rectangle must dominate.
        if wall_count * 4 < len(border) * 3:
            continue
        gaps = {(r, c) for r, c in border if int(grid[r, c]) == bg}
        if not gaps or any(int(grid[r, c]) not in (bg, color) for r, c in border):
            continue
        candidates.append((wall_count, color, b, gaps))

    assert candidates, "expected a rectangular frame with pinhole gaps"
    candidates.sort(reverse=True, key=lambda x: x[0])
    if len(candidates) > 1:
        assert candidates[0][0] > candidates[1][0], "frame must be uniquely identifiable"
    _, frame_color, bbox, gaps = candidates[0]
    return int(frame_color), bbox, gaps


def _gap_inward_direction(
    gap: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r, c = gap
    r0, c0, r1, c1 = bbox
    assert r not in (r0, r1) or c not in (c0, c1), "frame corners cannot be gaps"
    if r == r0:
        return 1, 0
    if r == r1:
        return -1, 0
    if c == c0:
        return 0, 1
    assert c == c1
    return 0, -1


def _outside_source(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], int]:
    r0, c0, r1, c1 = bbox
    possible: list[tuple[int, int]] = []
    for r, c in np.argwhere(
        (grid != bg) & (grid != frame_color) & (grid != SLASH) & (grid != BACKSLASH)
    ):
        rr, cc = int(r), int(c)
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            possible.append((rr, cc))
    assert len(possible) == 1, "expected exactly one colored source pixel outside the frame"
    source = possible[0]
    return source, int(grid[source])


def _entry_candidates(
    grid: np.ndarray,
    bg: int,
    source: tuple[int, int],
    bbox: tuple[int, int, int, int],
    gaps: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    sr, sc = source
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for gap in gaps:
        inward = _gap_inward_direction(gap, bbox)
        gr, gc = gap
        # The source must lie on the outward normal, not merely on the wall's axis.
        if inward[0] != 0:
            aligned = sc == gc and (sr - gr) * inward[0] < 0
        else:
            aligned = sr == gr and (sc - gc) * inward[1] < 0
        if not aligned:
            continue
        r, c = sr + inward[0], sc + inward[1]
        clear = True
        while (r, c) != gap:
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or int(grid[r, c]) != bg:
                clear = False
                break
            r, c = r + inward[0], c + inward[1]
        if clear and int(grid[gap]) == bg:
            out.append((gap, inward))
    return out


def _reflected(direction: tuple[int, int], mirror: int) -> tuple[int, int]:
    dr, dc = direction
    if mirror == SLASH:
        return -dc, -dr
    assert mirror == BACKSLASH
    return dc, dr


def _trace_ray(
    grid: np.ndarray,
    source: tuple[int, int],
    direction: tuple[int, int],
    source_color: int,
    bg: int,
    frame_color: int,
    bbox: tuple[int, int, int, int],
    gaps: set[tuple[int, int]],
) -> tuple[np.ndarray, dict]:
    """Simulate in canvas coordinates; mirrors turn before the following step."""
    out = grid.copy()
    h, w = out.shape
    r, c = source
    dr, dc = direction
    seen: set[tuple[int, int, int, int]] = set()
    painted: list[tuple[int, int]] = []
    mirror_hits: list[tuple[int, int]] = []
    gap_hits: list[tuple[int, int]] = []
    reason = "edge"

    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            reason = "edge"
            break
        state = (nr, nc, dr, dc)
        if state in seen:
            reason = "loop"
            break
        seen.add(state)
        value = int(grid[nr, nc])
        if value == frame_color:
            r0, c0, r1, c1 = bbox
            reason = "wall" if nr in (r0, r1) or nc in (c0, c1) else "obstacle"
            break
        if value in (SLASH, BACKSLASH):
            mirror_hits.append((nr, nc))
            dr, dc = _reflected((dr, dc), value)
            r, c = nr, nc
            continue
        assert value == bg or (nr, nc) == source, "ray encountered an unsupported colored object"
        if value == bg:
            out[nr, nc] = source_color
            painted.append((nr, nc))
        if (nr, nc) in gaps:
            gap_hits.append((nr, nc))
        r, c = nr, nc

    info = {
        "painted": painted,
        "mirror_hits": mirror_hits,
        "gap_hits": gap_hits,
        "reason": reason,
    }
    return out, info


def _parse_and_trace(grid: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids cannot exceed 30x30"
    bg = _background(g)
    assert bg not in (SLASH, BACKSLASH), "mirror colors must remain visually distinct from the background"
    frame_color, bbox, gaps = _find_frame(g, bg)
    source, source_color = _outside_source(g, bg, frame_color, bbox)
    entries = _entry_candidates(g, bg, source, bbox, gaps)
    assert len(entries) == 1, "source alignment and clear approach must select exactly one gap"
    entry, direction = entries[0]
    out, trace = _trace_ray(g, source, direction, source_color, bg, frame_color, bbox, gaps)
    parsed = {
        "bg": bg,
        "frame_color": frame_color,
        "bbox": bbox,
        "gaps": gaps,
        "source": source,
        "source_color": source_color,
        "entry": entry,
        "direction": direction,
    }
    return out, trace, parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast the uniquely admitted source ray through a pinholed mirror hall, reflecting it until absorption or escape.

    Input structure:
    - The most common color is the background; a different color makes one nearly complete rectangular frame with several isolated one-pixel gaps.
    - Exactly one colored source pixel is outside the frame. Teal (8) and orange (7) interior pixels are respectively "/" and "\\" mirrors.
    - Sparse pixels of the frame color inside the rectangle are absorbing obstacles; the other frame-color pixels are its walls.

    Rule:
    - Select the sole gap lying on the source's row or column along that side's outward normal, requiring every cell between source and gap to be background.
    - Paint the ray in the source color one cell at a time in the original canvas coordinates. At a teal "/" or orange "\\" mirror, preserve the mirror and apply the corresponding diagonal reflection.
    - Stop before an absorbing frame-color pixel or solid wall. If the ray reaches any gap, pass through it; after an exit gap, continue painting to the grid edge.

    Key insight:
    - Exterior alignment uniquely determines only the entrance; the exit or absorber is determined by following the complete ordered reflection cascade, with every gap acting as a bidirectional portal.
    """
    out, _trace, _parsed = _parse_and_trace(grid)
    return out


# ------------------------------ generator ------------------------------

PALETTE = tuple(range(10))
NON_MIRROR_COLORS = tuple(c for c in PALETTE if c not in (SLASH, BACKSLASH))
BRANCHES = ("exit", "obstacle", "wall")
_branch_index = random.randrange(len(BRANCHES))


def _side_geometry(
    side: str, bbox: tuple[int, int, int, int], rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return (r0, rng.randint(c0 + 2, c1 - 2)), (1, 0)
    if side == "bottom":
        return (r1, rng.randint(c0 + 2, c1 - 2)), (-1, 0)
    if side == "left":
        return (rng.randint(r0 + 2, r1 - 2), c0), (0, 1)
    return (rng.randint(r0 + 2, r1 - 2), c1), (0, -1)


def _inside(p: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = p
    r0, c0, r1, c1 = bbox
    return r0 < r < r1 and c0 < c < c1


def _segment(
    start: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    return [(start[0] + direction[0] * k, start[1] + direction[1] * k) for k in range(1, length + 1)]


def _distance_to_border(
    start: tuple[int, int], direction: tuple[int, int], bbox: tuple[int, int, int, int]
) -> int:
    r, c = start
    dr, dc = direction
    r0, c0, r1, c1 = bbox
    if dr < 0:
        return r - r0
    if dr > 0:
        return r1 - r
    if dc < 0:
        return c - c0
    return c1 - c


def _mirror_for_turn(incoming: tuple[int, int], outgoing: tuple[int, int]) -> int:
    if _reflected(incoming, SLASH) == outgoing:
        return SLASH
    assert _reflected(incoming, BACKSLASH) == outgoing
    return BACKSLASH


def _build_active_chain(
    entry: tuple[int, int],
    inward: tuple[int, int],
    bbox: tuple[int, int, int, int],
    branch: str,
    rng: random.Random,
) -> dict | None:
    mirror_target = rng.randint(2, 4)
    mirrors: dict[tuple[int, int], int] = {}
    trace_cells: set[tuple[int, int]] = {entry}
    current = entry
    direction = inward

    for _ in range(mirror_target):
        options: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
        for length in range(2, 6):
            cells = _segment(current, direction, length)
            if not all(_inside(p, bbox) for p in cells):
                continue
            if any(p in mirrors or p in trace_cells for p in cells):
                continue
            for outgoing in ((direction[1], -direction[0]), (-direction[1], direction[0])):
                # Leave room for at least one painted cell after this mirror.
                if _distance_to_border(cells[-1], outgoing, bbox) >= 2:
                    options.append((cells, outgoing))
        if not options:
            return None
        cells, outgoing = rng.choice(options)
        trace_cells.update(cells[:-1])
        mirror = cells[-1]
        mirrors[mirror] = _mirror_for_turn(direction, outgoing)
        current, direction = mirror, outgoing

    distance = _distance_to_border(current, direction, bbox)
    if distance < 2:
        return None
    terminal_line = _segment(current, direction, distance)
    if any(p in mirrors or p in trace_cells for p in terminal_line):
        return None
    trace_cells.update(terminal_line[:-1])
    terminal = terminal_line[-1]

    result = {
        "mirrors": mirrors,
        "trace_cells": trace_cells,
        "terminal": terminal,
        "direction": direction,
        "exit": None,
        "obstacle": None,
    }
    if branch == "exit":
        result["exit"] = terminal
        trace_cells.add(terminal)
    elif branch == "obstacle":
        # Move the absorber inward, leaving the frame impact point solid and unused.
        max_len = min(distance - 1, 4)
        if max_len < 2:
            return None
        obstacle_len = rng.randint(2, max_len)
        obstacle = _segment(current, direction, obstacle_len)[-1]
        trace_cells.difference_update(terminal_line[obstacle_len - 1 :])
        trace_cells.update(_segment(current, direction, obstacle_len - 1))
        result["obstacle"] = obstacle
        result["terminal"] = obstacle
    return result


def _gap_pool(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return (
        [(r0, c) for c in range(c0 + 1, c1)]
        + [(r1, c) for c in range(c0 + 1, c1)]
        + [(r, c0) for r in range(r0 + 1, r1)]
        + [(r, c1) for r in range(r0 + 1, r1)]
    )


def _not_adjacent_to_any(p: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    return all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1 for q in others)


def _pick_distractor_cell(
    bbox: tuple[int, int, int, int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    pool = [
        (r, c)
        for r in range(r0 + 2, r1 - 1)
        for c in range(c0 + 2, c1 - 1)
        if (r, c) not in forbidden
        and all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in forbidden)
    ]
    return rng.choice(pool) if pool else None


def _render_candidate(branch: str, rng: random.Random) -> tuple[np.ndarray, dict] | None:
    frame_h = rng.randint(10, 15)
    frame_w = rng.randint(10, 15)
    margins = [rng.randint(3, 5) for _ in range(4)]  # top, bottom, left, right
    h = frame_h + margins[0] + margins[1]
    w = frame_w + margins[2] + margins[3]
    if h > 30 or w > 30:
        return None
    bbox = (margins[0], margins[2], margins[0] + frame_h - 1, margins[2] + frame_w - 1)

    entry, inward = _side_geometry(rng.choice(("top", "bottom", "left", "right")), bbox, rng)
    chain = None
    for _ in range(120):
        chain = _build_active_chain(entry, inward, bbox, branch, rng)
        if chain is not None and chain["exit"] != entry:
            break
    if chain is None:
        return None

    bg, frame_color, source_color = rng.sample(NON_MIRROR_COLORS, 3)
    grid = np.full((h, w), bg, dtype=int)
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    gaps = {entry}
    if chain["exit"] is not None:
        gaps.add(chain["exit"])
    impact = chain["terminal"] if branch == "wall" else None

    # Add dead gaps that cannot compete with the source-aligned entry.
    source_distance = rng.randint(2, 3)
    source = (entry[0] - inward[0] * source_distance, entry[1] - inward[1] * source_distance)
    gap_candidates = [
        p
        for p in _gap_pool(bbox)
        if p not in gaps
        and p != impact
        and p[0] != source[0]
        and p[1] != source[1]
        and _not_adjacent_to_any(p, gaps)
    ]
    rng.shuffle(gap_candidates)
    dead_target = rng.randint(1, 3)
    for p in gap_candidates:
        if len(gaps) >= dead_target + (2 if branch == "exit" else 1):
            break
        if _not_adjacent_to_any(p, gaps):
            gaps.add(p)
    if len(gaps) < (2 if branch == "exit" else 1) + 1:
        return None
    for p in gaps:
        grid[p] = bg

    mirrors: dict[tuple[int, int], int] = dict(chain["mirrors"])
    for p, color in mirrors.items():
        grid[p] = color

    obstacles: set[tuple[int, int]] = set()
    if chain["obstacle"] is not None:
        obstacles.add(chain["obstacle"])

    forbidden = set(chain["trace_cells"]) | set(mirrors) | obstacles
    for _ in range(rng.randint(1, 2)):
        p = _pick_distractor_cell(bbox, forbidden, rng)
        if p is not None:
            obstacles.add(p)
            forbidden.add(p)
    for p in obstacles:
        grid[p] = frame_color

    # Both mirror glyphs are always visible; add off-path mirrors when necessary.
    needed = [color for color in (SLASH, BACKSLASH) if color not in mirrors.values()]
    while len(mirrors) + len(needed) < 3:
        needed.append(rng.choice((SLASH, BACKSLASH)))
    for color in needed:
        p = _pick_distractor_cell(bbox, forbidden, rng)
        if p is None:
            return None
        mirrors[p] = color
        forbidden.add(p)
        grid[p] = color

    grid[source] = source_color
    metadata = {
        "branch": branch,
        "entry": entry,
        "active_mirrors": set(chain["mirrors"]),
        "exit": chain["exit"],
        "terminal": chain["terminal"],
    }
    return grid, metadata


def _semantic_validation(grid: np.ndarray, metadata: dict) -> bool:
    """Reject shortcuts and verify intended semantics, not only oracle closure."""
    try:
        out, trace, parsed = _parse_and_trace(grid)
    except (AssertionError, ValueError):
        return False
    active_hits = trace["mirror_hits"]
    if len(active_hits) < 2 or set(active_hits) != metadata["active_mirrors"]:
        return False
    if len(set(map(int, grid.ravel())) & {SLASH, BACKSLASH}) != 2:
        return False
    if parsed["entry"] != metadata["entry"] or len(trace["painted"]) < 7:
        return False
    if len(parsed["gaps"] - set(trace["gap_hits"])) < 1:
        return False

    branch = metadata["branch"]
    if branch == "exit":
        if trace["reason"] != "edge" or trace["gap_hits"] != [metadata["entry"], metadata["exit"]]:
            return False
        er, ec = metadata["exit"]
        r0, c0, r1, c1 = parsed["bbox"]
        after_exit = [p for p in trace["painted"] if p != parsed["source"] and not (r0 <= p[0] <= r1 and c0 <= p[1] <= c1)]
        # This includes the entry approach, so require exterior paint on both sides.
        if not any((p[0] - er) * (er - active_hits[-1][0]) > 0 or (p[1] - ec) * (ec - active_hits[-1][1]) > 0 for p in after_exit):
            return False
    else:
        if trace["gap_hits"] != [metadata["entry"]] or trace["reason"] != branch:
            return False

    if np.array_equal(out, grid):
        return False
    # Removing an active mirror must change the result: the reflection cascade is necessary.
    altered = grid.copy()
    altered[active_hits[0]] = parsed["bg"]
    try:
        shortcut_out = main(altered)
    except AssertionError:
        shortcut_out = altered
    if np.array_equal(shortcut_out, out):
        return False
    return True


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked halls while cycling all terminal branches."""
    global _branch_index
    branch = BRANCHES[_branch_index % len(BRANCHES)]
    rng = random.Random()
    for _ in range(500):
        candidate = _render_candidate(branch, rng)
        if candidate is None:
            continue
        grid, metadata = candidate
        if _semantic_validation(grid, metadata):
            _branch_index += 1
            return grid
    raise RuntimeError(f"failed to generate a valid mirror-hall input for branch {branch}")
