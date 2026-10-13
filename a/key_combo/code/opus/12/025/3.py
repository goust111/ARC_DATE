from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

QUADRANT_SIZE = 9
FULL_SIZE = 2 * QUADRANT_SIZE + 1
CENTER = QUADRANT_SIZE

MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
PATTERN_COLOR_POOL = (BLUE, RED, GREEN, YELLOW, PINK, TEAL)
QUADRANT_ORDER = ("TL", "TR", "BL", "BR")

VERTEX_AXIS = (0, 3, 6, 8)
LINE_ROWS = (3, 6)
LINE_COLS = (3, 6)
LINE_HOLES = (1, 4, 7)


def _line_mask() -> np.ndarray:
    mask = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=bool)
    for r in LINE_ROWS:
        mask[r, :] = True
        mask[r, list(LINE_HOLES)] = False
    for c in LINE_COLS:
        mask[:, c] = True
        mask[list(LINE_HOLES), c] = False
    return mask


def _vertex_mask() -> np.ndarray:
    mask = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=bool)
    for r in VERTEX_AXIS:
        for c in VERTEX_AXIS:
            mask[r, c] = True
    return mask


LINE_MASK = _line_mask()
VERTEX_MASK = _vertex_mask()
ALLOWED_PATTERN_MASK = ~(LINE_MASK | VERTEX_MASK)
ALLOWED_CELLS = [(int(r), int(c)) for r, c in np.argwhere(ALLOWED_PATTERN_MASK)]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < QUADRANT_SIZE and 0 <= cc < QUADRANT_SIZE:
            out.append((rr, cc))
    return out


def _extract_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (FULL_SIZE, FULL_SIZE), f"expected {FULL_SIZE}x{FULL_SIZE}, got {g.shape}"
    assert np.all(g[CENTER, :] == ORANGE)
    assert np.all(g[:, CENTER] == ORANGE)
    return [
        g[:CENTER, :CENTER],
        g[:CENTER, CENTER + 1 :],
        g[CENTER + 1 :, :CENTER],
        g[CENTER + 1 :, CENTER + 1 :],
    ]


def _pattern_mask(quadrant: np.ndarray) -> np.ndarray:
    return (quadrant != BACKGROUND) & ALLOWED_PATTERN_MASK


def _pattern_color(quadrant: np.ndarray) -> int:
    colors = {int(quadrant[r, c]) for r, c in ALLOWED_CELLS if int(quadrant[r, c]) != BACKGROUND}
    assert len(colors) == 1, f"expected one pattern color, got {colors}"
    return colors.pop()


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    q = deque([coords[0]])
    while q:
        cell = q.popleft()
        for nxt in _neighbors4(cell):
            if mask[nxt] and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(coords)


def _consistent_cell_count(quadrant: np.ndarray) -> int:
    count = 0
    for r in range(3):
        for c in range(3):
            corners = [
                (VERTEX_AXIS[r], VERTEX_AXIS[c]),
                (VERTEX_AXIS[r], VERTEX_AXIS[c + 1]),
                (VERTEX_AXIS[r + 1], VERTEX_AXIS[c]),
                (VERTEX_AXIS[r + 1], VERTEX_AXIS[c + 1]),
            ]
            vals = [int(quadrant[rr, cc]) for rr, cc in corners]
            if vals[0] != BACKGROUND and len(set(vals)) == 1:
                count += 1
    return count


def _overlay_patterns(quadrants: list[np.ndarray]) -> np.ndarray:
    counts = [_consistent_cell_count(quadrant) for quadrant in quadrants]
    valid = [idx for idx, count in enumerate(counts) if count > 0]
    out = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=int)
    # Paint lower priority first: fewer consistent cells, then later quadrants.
    for idx in sorted(valid, key=lambda i: (counts[i], -i)):
        quadrant = quadrants[idx]
        mask = _pattern_mask(quadrant)
        out[mask] = quadrant[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay only the structurally valid quadrant patterns, with priority set by vertex-consistency counts.

    Input structure:
    - A 19x19 grid is split by an orange separator cross into four aligned 9x9 quadrants.
    - Each quadrant contains a grey internal lattice plus colored single-pixel vertex markers on the lattice intersections.
    - Each quadrant also contains one connected foreground pattern in its own color, drawn away from the lattice.

    Rule:
    - For each quadrant, inspect the 3x3 logical cell layout induced by the vertex positions.
    - A cell is consistent when its four corner markers all exist and share one color; count how many consistent cells each quadrant has.
    - Quadrants with zero consistent cells are ignored; the remaining quadrant patterns are overlaid onto a 9x9 output, with larger counts on top and row-major order breaking ties.

    Key insight:
    - The marker colors never paint the answer directly; they only decide which quadrant layers survive and which surviving layer covers the others at overlaps.
    """
    quadrants = _extract_quadrants(np.asarray(grid, dtype=int))
    return _overlay_patterns(quadrants)


def _symmetry_transform(cell: tuple[int, int], rot: int, flip: bool) -> tuple[int, int]:
    r, c = cell
    for _ in range(rot):
        r, c = c, 2 - r
    if flip:
        c = 2 - c
    return r, c


def _cell_template(count: int, rng: random.Random) -> list[tuple[int, int]]:
    if count == 0:
        return []
    base = [(0, 0)]
    if count >= 2:
        base.append((1, 1))
    if count >= 3:
        base.append((2, 2))
    rot = rng.randrange(4)
    flip = bool(rng.randrange(2))
    return [_symmetry_transform(cell, rot, flip) for cell in base]


def _count_from_lattice(lattice: np.ndarray) -> int:
    count = 0
    for r in range(3):
        for c in range(3):
            vals = [
                int(lattice[r, c]),
                int(lattice[r + 1, c]),
                int(lattice[r, c + 1]),
                int(lattice[r + 1, c + 1]),
            ]
            if vals[0] != BACKGROUND and len(set(vals)) == 1:
                count += 1
    return count


def _render_lattice(target_count: int, rng: random.Random) -> np.ndarray:
    base_colors = list(MARKER_COLORS)
    rng.shuffle(base_colors)
    a, b, active, spare = base_colors
    lattice = np.array([[a if (r + c) % 2 == 0 else b for c in range(4)] for r in range(4)], dtype=int)
    for r, c in _cell_template(target_count, rng):
        lattice[r : r + 2, c : c + 2] = active

    # Add a little marker variation without changing the consistency count.
    candidates = [(r, c) for r in range(4) for c in range(4)]
    rng.shuffle(candidates)
    for r, c in candidates[: rng.randint(0, 3)]:
        old = int(lattice[r, c])
        proposal = rng.choice((BACKGROUND, spare))
        lattice[r, c] = proposal
        if _count_from_lattice(lattice) != target_count:
            lattice[r, c] = old

    assert _count_from_lattice(lattice) == target_count
    assert np.count_nonzero(lattice) >= 8
    return lattice


def _shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    allowed: np.ndarray,
) -> list[tuple[int, int]] | None:
    q = deque([start])
    parent = {start: None}
    while q:
        cell = q.popleft()
        if cell == goal:
            path = []
            cur = cell
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path
        for nxt in _neighbors4(cell):
            if allowed[nxt] and nxt not in parent:
                parent[nxt] = cell
                q.append(nxt)
    return None


def _grow_connected_shape(
    required: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    rng: random.Random,
    min_size: int = 5,
    max_size: int = 12,
) -> np.ndarray | None:
    allowed = ALLOWED_PATTERN_MASK.copy()
    for cell in blocked:
        allowed[cell] = False
    for cell in required:
        if not allowed[cell]:
            return None

    shape = set(required)
    req_list = list(required)
    if req_list:
        anchor = req_list[0]
        for other in req_list[1:]:
            path = _shortest_path(anchor, other, allowed)
            if path is None:
                return None
            shape.update(path)

    if len(shape) > max_size:
        return None
    target = rng.randint(max(min_size, len(shape)), max_size)
    while len(shape) < target:
        frontier = []
        for cell in shape:
            for nxt in _neighbors4(cell):
                if allowed[nxt] and nxt not in shape:
                    frontier.append(nxt)
        if not frontier:
            return None
        frontier = sorted(set(frontier))
        shape.add(rng.choice(frontier))

    mask = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=bool)
    for cell in shape:
        mask[cell] = True
    if not _is_connected(mask):
        return None
    return mask


def _bfs_distances(start: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        cell = q.popleft()
        for nxt in _neighbors4(cell):
            if ALLOWED_PATTERN_MASK[nxt] and nxt not in dist:
                dist[nxt] = dist[cell] + 1
                q.append(nxt)
    return dist


def _sample_counts(valid_indices: list[int], rng: random.Random) -> list[int]:
    counts = [0, 0, 0, 0]
    if rng.random() < 0.45:
        top_value = rng.choice((2, 3))
        if len(valid_indices) == 2:
            tied = valid_indices
        else:
            tied = rng.sample(valid_indices, 2)
        for idx in tied:
            counts[idx] = top_value
        for idx in valid_indices:
            if idx not in tied:
                counts[idx] = 1 if top_value > 1 else 2
    else:
        values = [3, 1] if len(valid_indices) == 2 else [3, 2, 1]
        rng.shuffle(values)
        for idx, value in zip(valid_indices, values):
            counts[idx] = value
    return counts


def _paint_quadrant(
    lattice: np.ndarray,
    pattern_mask: np.ndarray,
    pattern_color: int,
) -> np.ndarray:
    quadrant = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=int)
    quadrant[LINE_MASK] = GREY
    for vr in range(4):
        for vc in range(4):
            color = int(lattice[vr, vc])
            if color == BACKGROUND:
                continue
            quadrant[VERTEX_AXIS[vr], VERTEX_AXIS[vc]] = color
    quadrant[pattern_mask] = pattern_color
    return quadrant


def _assemble_input(quadrants: list[np.ndarray]) -> np.ndarray:
    grid = np.zeros((FULL_SIZE, FULL_SIZE), dtype=int)
    grid[CENTER, :] = ORANGE
    grid[:, CENTER] = ORANGE
    grid[:CENTER, :CENTER] = quadrants[0]
    grid[:CENTER, CENTER + 1 :] = quadrants[1]
    grid[CENTER + 1 :, :CENTER] = quadrants[2]
    grid[CENTER + 1 :, CENTER + 1 :] = quadrants[3]
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    quadrants = _extract_quadrants(grid)
    counts = [_consistent_cell_count(quadrant) for quadrant in quadrants]
    valid = [idx for idx, count in enumerate(counts) if count > 0]
    invalid = [idx for idx, count in enumerate(counts) if count == 0]
    assert len(valid) >= 2
    assert len(invalid) >= 1

    pattern_masks = [_pattern_mask(quadrant) for quadrant in quadrants]
    pattern_colors = [_pattern_color(quadrant) for quadrant in quadrants]
    assert len(set(pattern_colors)) == 4

    for mask in pattern_masks:
        size = int(np.count_nonzero(mask))
        assert 5 <= size <= 12
        assert _is_connected(mask)
        assert not np.any(mask & LINE_MASK)
        assert not np.any(mask & VERTEX_MASK)

    valid_union = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=bool)
    for idx in valid:
        valid_union |= pattern_masks[idx]

    overlap = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=int)
    for idx in valid:
        overlap += pattern_masks[idx].astype(int)
    assert np.any(overlap >= 2)

    for idx in valid:
        others = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=bool)
        for jdx in valid:
            if jdx != idx:
                others |= pattern_masks[jdx]
        assert np.any(pattern_masks[idx] & ~others)

    assert any(np.any(pattern_masks[idx] & ~valid_union) for idx in invalid)

    top_count = max(counts[idx] for idx in valid)
    top_valid = [idx for idx in valid if counts[idx] == top_count]
    if len(top_valid) >= 2:
        top_overlap = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=int)
        for idx in top_valid:
            top_overlap += pattern_masks[idx].astype(int)
        assert np.any(top_overlap >= 2)

    out = main(grid)
    assert out.shape == (QUADRANT_SIZE, QUADRANT_SIZE)
    assert np.array_equal(out, _overlay_patterns(quadrants))


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    colors = list(PATTERN_COLOR_POOL)
    rng.shuffle(colors)
    quadrant_colors = colors[:4]

    num_valid = rng.choice((2, 3))
    valid_indices = sorted(rng.sample(range(4), num_valid))
    counts = _sample_counts(valid_indices, rng)

    shared_seed = rng.choice(ALLOWED_CELLS)
    dist = _bfs_distances(shared_seed)
    near_cells = [
        cell
        for cell, d in dist.items()
        if 2 <= d <= 4 and cell != shared_seed
    ]
    if len(near_cells) < 4:
        return None
    rng.shuffle(near_cells)

    valid_private: dict[int, tuple[int, int]] = {}
    invalid_private: dict[int, tuple[int, int]] = {}
    used = {shared_seed}

    for idx in valid_indices:
        choices = [cell for cell in near_cells if cell not in used]
        if not choices:
            return None
        cell = choices[0]
        valid_private[idx] = cell
        used.add(cell)

    invalid_indices = [idx for idx in range(4) if idx not in valid_indices]
    far_cells = [cell for cell in ALLOWED_CELLS if cell not in used]
    rng.shuffle(far_cells)
    for idx in invalid_indices:
        if not far_cells:
            return None
        invalid_private[idx] = far_cells.pop()
        used.add(invalid_private[idx])

    reserved_private = set(valid_private.values()) | set(invalid_private.values())
    pattern_masks: list[np.ndarray] = []
    for idx in range(4):
        if idx in valid_private:
            required = {shared_seed, valid_private[idx]}
            blocked = (reserved_private - {valid_private[idx]})
            mask = _grow_connected_shape(required, blocked, rng)
        else:
            required = {invalid_private[idx]}
            blocked = (reserved_private - {invalid_private[idx]}) | {shared_seed}
            mask = _grow_connected_shape(required, blocked, rng)
        if mask is None:
            return None
        pattern_masks.append(mask)

    quadrants = []
    for idx in range(4):
        lattice = _render_lattice(counts[idx], rng)
        quadrants.append(_paint_quadrant(lattice, pattern_masks[idx], quadrant_colors[idx]))

    grid = _assemble_input(quadrants)
    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
