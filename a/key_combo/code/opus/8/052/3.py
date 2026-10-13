from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9

MOVE_BY_COLOR = {
    ORANGE: (-2, 0),
    RED: (0, -2),
    MAROON: (2, 0),
    BLUE: (0, 1),
}
MOTION_COLORS = tuple(MOVE_BY_COLOR)
RING_STEPS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

GRID_SIZE_RANGE = (11, 16)
FOREGROUND_COUNT_RANGE = (6, 10)
MAX_GENERATION_ATTEMPTS = 400

_GENERATION_PHASE = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _landing_map(grid: np.ndarray) -> dict[tuple[int, int], list[int]]:
    h, w = grid.shape
    landings: dict[tuple[int, int], list[int]] = {}
    for r, c in np.argwhere(grid != BLACK):
        r = int(r)
        c = int(c)
        color = int(grid[r, c])
        if color not in MOVE_BY_COLOR:
            raise ValueError(f"unsupported foreground color {color}")
        dr, dc = MOVE_BY_COLOR[color]
        nr = r + dr
        nc = c + dc
        if _in_bounds((h, w), nr, nc):
            landings.setdefault((nr, nc), []).append(color)
    return landings


def main(grid: np.ndarray) -> np.ndarray:
    """Translate motion-colored pixels, mark landing collisions green, and paint yellow rings around those collision points.

    Input structure:
    - The grid is a black background with sparse isolated foreground pixels.
    - Every foreground pixel is one of four motion colors: orange, red, maroon, or blue.
    - No input cell contains more than one pixel, and any collision only emerges after movement.

    Rule:
    - Clear the canvas to black.
    - Move every orange pixel up 2, every red pixel left 2, every maroon pixel down 2, and every blue pixel right 1.
    - Drop moved pixels that land out of bounds.
    - If exactly one moved pixel lands on a cell, paint that landing cell with its original color.
    - If two or more moved pixels land on the same cell, paint that cell green, then paint its in-bounds 8-neighbors yellow.
    - Keep priorities as green collision center over original singleton landings over yellow ring over black background.

    Key insight:
    - The green centers and yellow rings are determined only after simulating all color-coded translations, because collisions are emergent properties of the moved endpoints rather than visible input objects.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    collisions: list[tuple[int, int]] = []

    for (r, c), colors in _landing_map(g).items():
        if len(colors) == 1:
            out[r, c] = colors[0]
        else:
            collisions.append((r, c))

    h, w = g.shape
    for r, c in collisions:
        for dr, dc in RING_STEPS:
            nr = r + dr
            nc = c + dc
            if _in_bounds((h, w), nr, nc) and out[nr, nc] == BLACK:
                out[nr, nc] = YELLOW

    for r, c in collisions:
        out[r, c] = GREEN

    return out


def _source_from_dest(dest: tuple[int, int], color: int) -> tuple[int, int]:
    dr, dc = MOVE_BY_COLOR[color]
    return dest[0] - dr, dest[1] - dc


def _landing_sources_if_valid(
    dest: tuple[int, int], colors: list[int] | tuple[int, ...], shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    srcs = [_source_from_dest(dest, color) for color in colors]
    if len(set(srcs)) != len(srcs):
        return None
    if not all(_in_bounds(shape, r, c) for r, c in srcs):
        return None
    return srcs


def _empty_state() -> dict[str, object]:
    return {"sources": set(), "destinations": set(), "placements": []}


def _can_add_landing(
    state: dict[str, object],
    shape: tuple[int, int],
    dest: tuple[int, int],
    colors: list[int] | tuple[int, ...],
) -> bool:
    srcs = _landing_sources_if_valid(dest, colors, shape)
    if srcs is None:
        return False
    sources: set[tuple[int, int]] = state["sources"]  # type: ignore[assignment]
    destinations: set[tuple[int, int]] = state["destinations"]  # type: ignore[assignment]
    if dest in destinations or dest in sources:
        return False
    for src in srcs:
        if src in sources or src in destinations:
            return False
    return True


def _add_landing(
    state: dict[str, object],
    shape: tuple[int, int],
    dest: tuple[int, int],
    colors: list[int] | tuple[int, ...],
) -> bool:
    if not _can_add_landing(state, shape, dest, colors):
        return False
    srcs = _landing_sources_if_valid(dest, colors, shape)
    assert srcs is not None
    sources: set[tuple[int, int]] = state["sources"]  # type: ignore[assignment]
    destinations: set[tuple[int, int]] = state["destinations"]  # type: ignore[assignment]
    placements: list[tuple[int, int, int]] = state["placements"]  # type: ignore[assignment]

    destinations.add(dest)
    for src, color in zip(srcs, colors):
        sources.add(src)
        placements.append((src[0], src[1], int(color)))
    return True


def _pick_and_add_landing(
    state: dict[str, object],
    shape: tuple[int, int],
    colors: list[int] | tuple[int, ...],
    predicate,
) -> tuple[int, int] | None:
    h, w = shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if predicate(r, c) and _can_add_landing(state, shape, (r, c), colors)
    ]
    if not candidates:
        return None
    dest = random.choice(candidates)
    ok = _add_landing(state, shape, dest, colors)
    assert ok
    return dest


def _pick_and_add_specific_dest(
    state: dict[str, object],
    shape: tuple[int, int],
    colors: list[int] | tuple[int, ...],
    candidates: list[tuple[int, int]],
) -> tuple[int, int] | None:
    usable = [dest for dest in candidates if _can_add_landing(state, shape, dest, colors)]
    if not usable:
        return None
    dest = random.choice(usable)
    ok = _add_landing(state, shape, dest, colors)
    assert ok
    return dest


def _drop_candidates(
    state: dict[str, object], shape: tuple[int, int], color: int
) -> list[tuple[int, int]]:
    h, w = shape
    sources: set[tuple[int, int]] = state["sources"]  # type: ignore[assignment]
    destinations: set[tuple[int, int]] = state["destinations"]  # type: ignore[assignment]
    candidates = []
    dr, dc = MOVE_BY_COLOR[color]
    for r in range(h):
        for c in range(w):
            if (r, c) in sources or (r, c) in destinations:
                continue
            nr = r + dr
            nc = c + dc
            if not _in_bounds(shape, nr, nc):
                candidates.append((r, c))
    return candidates


def _pick_and_add_drop(
    state: dict[str, object], shape: tuple[int, int], color: int
) -> bool:
    candidates = _drop_candidates(state, shape, color)
    if not candidates:
        return False
    r, c = random.choice(candidates)
    sources: set[tuple[int, int]] = state["sources"]  # type: ignore[assignment]
    placements: list[tuple[int, int, int]] = state["placements"]  # type: ignore[assignment]
    sources.add((r, c))
    placements.append((r, c, color))
    return True


def _materialize_grid(state: dict[str, object], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    placements: list[tuple[int, int, int]] = state["placements"]  # type: ignore[assignment]
    for r, c, color in placements:
        grid[r, c] = color
    return grid


def _state_copy(state: dict[str, object]) -> dict[str, object]:
    return {
        "sources": set(state["sources"]),  # type: ignore[arg-type]
        "destinations": set(state["destinations"]),  # type: ignore[arg-type]
        "placements": list(state["placements"]),  # type: ignore[arg-type]
    }


def _add_fillers(
    state: dict[str, object],
    shape: tuple[int, int],
    target_count: int,
    anchor_points: list[tuple[int, int]],
) -> None:
    attempts = 0
    while len(state["placements"]) < target_count and attempts < 60:  # type: ignore[arg-type]
        attempts += 1
        if random.random() < 0.25:
            if _pick_and_add_drop(state, shape, random.choice(MOTION_COLORS)):
                continue
        color = random.choice(MOTION_COLORS)
        min_gap = random.choice((2, 3))
        added = _pick_and_add_landing(
            state,
            shape,
            [color],
            lambda r, c: all(_chebyshev((r, c), point) >= min_gap for point in anchor_points),
        )
        if added is not None:
            anchor_points.append(added)


def _build_phase0(shape: tuple[int, int]) -> np.ndarray | None:
    state = _empty_state()
    h, w = shape
    center = _pick_and_add_landing(
        state,
        shape,
        [ORANGE, RED],
        lambda r, c: 2 <= r <= h - 4 and 2 <= c <= w - 4,
    )
    if center is None:
        return None

    ring_positions = [
        (center[0] + dr, center[1] + dc)
        for dr, dc in RING_STEPS
        if _in_bounds(shape, center[0] + dr, center[1] + dc)
    ]
    protected = _pick_and_add_specific_dest(state, shape, [BLUE], ring_positions)
    if protected is None:
        return None

    anchors = [center, protected]
    if _pick_and_add_landing(
        state,
        shape,
        [MAROON],
        lambda r, c: all(_chebyshev((r, c), point) >= 3 for point in anchors),
    ) is None:
        return None
    if _pick_and_add_landing(
        state,
        shape,
        [random.choice([ORANGE, RED])],
        lambda r, c: all(_chebyshev((r, c), point) >= 4 for point in anchors),
    ) is None:
        return None
    if not _pick_and_add_drop(state, shape, random.choice([BLUE, RED, MAROON])):
        return None

    target_count = random.randint(6, 8)
    _add_fillers(state, shape, target_count, anchors)
    return _materialize_grid(state, shape)


def _build_phase1(shape: tuple[int, int]) -> np.ndarray | None:
    state = _empty_state()
    h, w = shape
    candidates = []
    for r in range(2, h - 3):
        for c in range(2, w - 3):
            a = (r, c)
            b = (r + 1, c + 1)
            if not _can_add_landing(state, shape, a, [ORANGE, RED]):
                continue
            temp = _state_copy(state)
            assert _add_landing(temp, shape, a, [ORANGE, RED])
            if _can_add_landing(temp, shape, b, [MAROON, BLUE]):
                candidates.append(a)
    if not candidates:
        return None

    a = random.choice(candidates)
    b = (a[0] + 1, a[1] + 1)
    assert _add_landing(state, shape, a, [ORANGE, RED])
    assert _add_landing(state, shape, b, [MAROON, BLUE])

    anchors = [a, b]
    if _pick_and_add_landing(
        state,
        shape,
        [random.choice([ORANGE, MAROON])],
        lambda r, c: all(_chebyshev((r, c), point) >= 3 for point in anchors),
    ) is None:
        return None
    if _pick_and_add_landing(
        state,
        shape,
        [random.choice([RED, BLUE])],
        lambda r, c: all(_chebyshev((r, c), point) >= 4 for point in anchors),
    ) is None:
        return None

    target_count = random.randint(6, 8)
    _add_fillers(state, shape, target_count, anchors)
    return _materialize_grid(state, shape)


def _build_phase2(shape: tuple[int, int]) -> np.ndarray | None:
    state = _empty_state()
    h, w = shape
    center = _pick_and_add_landing(
        state,
        shape,
        [ORANGE, RED],
        lambda r, c: r == 0 and 2 <= c <= w - 4,
    )
    if center is None:
        return None

    anchors = [center]
    if _pick_and_add_landing(
        state,
        shape,
        [MAROON],
        lambda r, c: r >= h // 2 and all(_chebyshev((r, c), point) >= 4 for point in anchors),
    ) is None:
        return None
    if _pick_and_add_landing(
        state,
        shape,
        [BLUE],
        lambda r, c: r >= h // 2 and c <= w // 2 and all(_chebyshev((r, c), point) >= 3 for point in anchors),
    ) is None:
        return None
    if _pick_and_add_landing(
        state,
        shape,
        [random.choice([ORANGE, RED])],
        lambda r, c: r >= 1 and all(_chebyshev((r, c), point) >= 4 for point in anchors),
    ) is None:
        return None
    if not _pick_and_add_drop(state, shape, random.choice([ORANGE, BLUE, MAROON])):
        return None

    target_count = random.randint(6, 8)
    _add_fillers(state, shape, target_count, anchors)
    return _materialize_grid(state, shape)


def _build_phase3(shape: tuple[int, int]) -> np.ndarray | None:
    state = _empty_state()
    h, w = shape
    center = _pick_and_add_landing(
        state,
        shape,
        [ORANGE, RED, MAROON],
        lambda r, c: 2 <= r <= h - 4 and 2 <= c <= w - 4,
    )
    if center is None:
        return None

    ring_positions = [
        (center[0] + dr, center[1] + dc)
        for dr, dc in RING_STEPS
        if _in_bounds(shape, center[0] + dr, center[1] + dc)
    ]
    protected = _pick_and_add_specific_dest(state, shape, [BLUE], ring_positions)
    if protected is None:
        return None

    anchors = [center, protected]
    if _pick_and_add_landing(
        state,
        shape,
        [ORANGE],
        lambda r, c: all(_chebyshev((r, c), point) >= 3 for point in anchors),
    ) is None:
        return None
    if _pick_and_add_landing(
        state,
        shape,
        [random.choice([RED, MAROON])],
        lambda r, c: all(_chebyshev((r, c), point) >= 4 for point in anchors),
    ) is None:
        return None
    if not _pick_and_add_drop(state, shape, random.choice([BLUE, RED, MAROON])):
        return None

    target_count = random.randint(7, 9)
    _add_fillers(state, shape, target_count, anchors)
    return _materialize_grid(state, shape)


def _validate_generated(grid: np.ndarray, phase: int) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED, GREEN, YELLOW, ORANGE, MAROON}):
        return False

    foreground = int(np.count_nonzero(grid))
    if not (FOREGROUND_COUNT_RANGE[0] <= foreground <= FOREGROUND_COUNT_RANGE[1]):
        return False

    landings = _landing_map(grid)
    if not landings:
        return False

    collisions = {pos: colors for pos, colors in landings.items() if len(colors) >= 2}
    singles = {pos: colors[0] for pos, colors in landings.items() if len(colors) == 1}
    if not collisions or not singles:
        return False
    if max(len(colors) for colors in collisions.values()) < (3 if phase == 3 else 2):
        return False

    out = main(grid)
    if out.shape != grid.shape or not np.any(out == GREEN) or not np.any(out == YELLOW):
        return False
    if not all(out[r, c] == color for (r, c), color in singles.items()):
        return False
    if not all(out[r, c] == GREEN for r, c in collisions):
        return False

    moved_foreground = sum(len(colors) for colors in landings.values())
    if phase == 2 and moved_foreground >= foreground:
        return False
    if phase in (0, 3):
        if not any(
            _chebyshev(single_pos, collision_pos) == 1
            for single_pos in singles
            for collision_pos in collisions
        ):
            return False
    if phase == 1:
        collision_positions = list(collisions)
        if not any(
            _chebyshev(collision_positions[i], collision_positions[j]) == 1
            for i in range(len(collision_positions))
            for j in range(i + 1, len(collision_positions))
        ):
            return False
    return True


def generate_input() -> np.ndarray:
    global _GENERATION_PHASE

    for _ in range(MAX_GENERATION_ATTEMPTS):
        phase = _GENERATION_PHASE % 4
        shape = (
            random.randint(*GRID_SIZE_RANGE),
            random.randint(*GRID_SIZE_RANGE),
        )
        builders = (_build_phase0, _build_phase1, _build_phase2, _build_phase3)
        grid = builders[phase](shape)
        if grid is None:
            continue
        if _validate_generated(grid, phase):
            _GENERATION_PHASE += 1
            return grid

    raise RuntimeError("failed to generate a valid collision-ring ARC input")
