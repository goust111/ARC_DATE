from __future__ import annotations

import numpy as np


BG = 0
MARKER = 2
SPRITE = 3
FRAME = 5

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Offsets are relative to the head pixel at (0, 0).
TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    "hollow3": {
        0: [(0, 0), (1, -1), (1, 1)],
        1: [(0, 0), (-1, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 1)],
        3: [(0, 0), (-1, 1), (1, 1)],
    },
    "filled4": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1)],
    },
    "stem4": {
        0: [(0, 0), (1, 0), (2, -1), (2, 1)],
        1: [(0, 0), (0, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, 0), (-2, -1), (-2, 1)],
        3: [(0, 0), (0, 1), (-1, 2), (1, 2)],
    },
}


GRID_MIN = 17
GRID_MAX = 30
STEP_COUNT_RANGE = (4, 6)
STEP_DISTANCES = (2, 3, 4)
MAX_ATTEMPTS = 500


def _opposite(direction: int) -> int:
    return (int(direction) + 2) % 4


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    ar, ac = a
    br, bc = b
    if ac == bc and br < ar:
        return 0
    if ar == br and bc > ac:
        return 1
    if ac == bc and br > ar:
        return 2
    if ar == br and bc < ac:
        return 3
    return None


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _template_bounds(template: str, direction: int) -> tuple[int, int, int, int]:
    offsets = TEMPLATES[template][direction]
    rows = [dr for dr, _ in offsets]
    cols = [dc for _, dc in offsets]
    return min(rows), max(rows), min(cols), max(cols)


def _sprite_cells(head: tuple[int, int], template: str, direction: int) -> set[tuple[int, int]]:
    hr, hc = map(int, head)
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[template][direction]}


def _fits_sprite(
    shape: tuple[int, int],
    head: tuple[int, int],
    template: str,
    direction: int,
) -> bool:
    h, w = map(int, shape)
    for r, c in _sprite_cells(head, template, direction):
        if not (0 <= r < h and 0 <= c < w):
            return False
    return True


def _fits_frame(shape: tuple[int, int], center: tuple[int, int], radius: int) -> bool:
    h, w = map(int, shape)
    r, c = map(int, center)
    return radius <= r < h - radius and radius <= c < w - radius


def _draw_hollow_square(grid: np.ndarray, center: tuple[int, int], radius: int, color: int) -> None:
    r, c = map(int, center)
    r0, r1 = r - radius, r + radius
    c0, c1 = c - radius, c + radius
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _find_sprite(grid: np.ndarray) -> tuple[str, int, tuple[int, int]]:
    pts = {tuple(map(int, rc)) for rc in np.argwhere(grid == SPRITE)}
    assert 3 <= len(pts) <= 4, f"expected a 3-4 pixel sprite, got {len(pts)}"

    matches: list[tuple[str, int, tuple[int, int]]] = []
    for head in pts:
        hr, hc = head
        for template, per_dir in TEMPLATES.items():
            for direction, offsets in per_dir.items():
                placed = {(hr + dr, hc + dc) for dr, dc in offsets}
                if placed == pts:
                    matches.append((template, int(direction), (int(hr), int(hc))))
    assert len(matches) == 1, f"expected exactly one sprite interpretation, got {len(matches)}"
    return matches[0]


def _reachable_markers(
    head: tuple[int, int],
    facing: int,
    markers: set[tuple[int, int]],
    shape: tuple[int, int],
    template: str,
) -> dict[int, list[tuple[int, tuple[int, int]]]]:
    hr, hc = map(int, head)
    reachable: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for direction in range(4):
        if direction == _opposite(facing):
            continue
        candidates: list[tuple[int, tuple[int, int]]] = []
        for mr, mc in markers:
            mr = int(mr)
            mc = int(mc)
            if direction == 0 and mc == hc and mr < hr:
                target = (mr, mc)
            elif direction == 1 and mr == hr and mc > hc:
                target = (mr, mc)
            elif direction == 2 and mc == hc and mr > hr:
                target = (mr, mc)
            elif direction == 3 and mr == hr and mc < hc:
                target = (mr, mc)
            else:
                continue
            if _fits_sprite(shape, target, template, direction):
                candidates.append((_distance(head, target), target))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            reachable[int(direction)] = candidates
    return reachable


def _simulate_path(
    shape: tuple[int, int],
    template: str,
    initial_head: tuple[int, int],
    initial_facing: int,
    markers: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[int]]:
    heads = [tuple(map(int, initial_head))]
    facings = [int(initial_facing)]
    remaining = {tuple(map(int, p)) for p in markers}

    while remaining:
        reachable = _reachable_markers(heads[-1], facings[-1], remaining, shape, template)
        assert len(reachable) == 1, f"expected exactly one reachable direction, got {sorted(reachable)}"
        direction = next(iter(reachable))
        _dist, target = reachable[direction][0]
        heads.append(target)
        facings.append(int(direction))
        remaining.remove(target)

    return heads, facings


def _render_output(
    shape: tuple[int, int],
    template: str,
    heads: list[tuple[int, int]],
    facings: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for step_index, center in enumerate(heads[1:], start=1):
        _draw_hollow_square(out, center, step_index, FRAME)
    for r, c in _sprite_cells(heads[-1], template, facings[-1]):
        out[r, c] = SPRITE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the triangular sprite through the uniquely reachable markers and draw larger hollow frames at each landing.

    Input structure:
    - The background is black.
    - Exactly one green triangular sprite of 3-4 pixels appears; its tip pixel is the head and its geometry encodes one of the four cardinal facings.
    - Four to six red single-pixel markers are scattered on the board.

    Rule:
    - Read the sprite head and facing from the green template.
    - Repeatedly look from the head along cardinal half-lines, but never in the exact reverse of the current facing; valid inputs are constructed so that exactly one allowed direction contains reachable markers, and the nearest marker on that half-line is the next stop.
    - Move the whole sprite so its head lands on that marker, consume the marker, rotate the sprite to the movement direction, and after the k-th consumption draw a grey hollow square of side 2k+1 centered on that landing cell.
    - Keep only the final sprite position and all grey frames; the travel path leaves no trail.

    Key insight:
    - The frame size depends on visit order rather than on static geometry, so the solver must simulate the full consume path before it can know which border belongs at each marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template, facing, head = _find_sprite(g)
    markers = {tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)}
    assert 4 <= len(markers) <= 6, f"expected 4-6 markers, got {len(markers)}"

    heads, facings = _simulate_path(g.shape, template, head, facing, markers)
    return _render_output(g.shape, template, heads, facings)


def _sample_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    n_segments = int(rng.integers(3, min(4, n_steps) + 1))
    lengths = np.ones(n_segments, dtype=int)
    for _ in range(n_steps - n_segments):
        lengths[int(rng.integers(0, n_segments))] += 1
    if lengths.max() < 2:
        idx = int(rng.integers(0, n_segments))
        donor = int((idx + 1) % n_segments)
        lengths[idx] += 1
        lengths[donor] -= 1

    segments = [int(rng.integers(0, 4))]
    for _ in range(1, n_segments):
        turn = int(rng.choice([-1, 1]))
        segments.append((segments[-1] + turn) % 4)

    step_dirs: list[int] = []
    for direction, length in zip(segments, lengths):
        step_dirs.extend([int(direction)] * int(length))
    return step_dirs


def _sequence_is_interesting(step_dirs: list[int]) -> bool:
    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, len(step_dirs)))
    if turns < 2:
        return False
    if len(set(step_dirs)) < 3:
        return False
    if all(step_dirs[i] != step_dirs[i - 1] for i in range(1, len(step_dirs))):
        return False
    if not ((0 in step_dirs and 2 in step_dirs) or (1 in step_dirs and 3 in step_dirs)):
        return False
    return True


def _candidate_clear_of_past_sprites(
    candidate: tuple[int, int],
    heads: list[tuple[int, int]],
    facings: list[int],
    template: str,
) -> bool:
    return all(candidate not in _sprite_cells(head, template, facing) for head, facing in zip(heads, facings))


def _future_marker_is_compatible(
    candidate: tuple[int, int],
    heads: list[tuple[int, int]],
    facings: list[int],
    step_dirs: list[int],
    template: str,
) -> bool:
    if not _candidate_clear_of_past_sprites(candidate, heads, facings, template):
        return False

    for idx, head in enumerate(heads[:-1]):
        direction = _direction(head, candidate)
        if direction is None or direction == _opposite(facings[idx]):
            continue
        if direction != step_dirs[idx]:
            return False
        if _distance(head, candidate) < _distance(head, heads[idx + 1]):
            return False
    return True


def _rollout_relative_path(
    rng: np.random.Generator,
    template: str,
    step_dirs: list[int],
) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
    heads = [(0, 0)]
    facings = [int(step_dirs[0])]
    step_distances: list[int] = []

    for step_index, move_dir in enumerate(step_dirs):
        current_head = heads[-1]
        current_facing = facings[-1]
        if move_dir == _opposite(current_facing):
            return None

        dr, dc = DIRS4[move_dir]
        distances = list(STEP_DISTANCES)
        rng.shuffle(distances)
        chosen = None
        for distance in distances:
            candidate = (current_head[0] + dr * int(distance), current_head[1] + dc * int(distance))
            if candidate in heads:
                continue
            if not _future_marker_is_compatible(candidate, heads, facings, step_dirs, template):
                continue
            chosen = candidate
            step_distances.append(int(distance))
            break
        if chosen is None:
            return None

        heads.append(chosen)
        facings.append(int(move_dir))

        # The new state must not already expose two different legal directions.
        partial_remaining = set(heads[step_index + 2 :])
        if partial_remaining:
            reachable = _reachable_markers(chosen, move_dir, partial_remaining, (999, 999), template)
            if len(reachable) > 1:
                return None

    if len(set(step_distances)) < 2:
        return None
    return heads, facings, step_distances


def _translation_interval(
    size: int,
    coords: list[int],
    lo_offsets: list[int],
    hi_offsets: list[int],
    frame_radii: list[int],
) -> tuple[int, int] | None:
    lo = -10**9
    hi = 10**9
    for coord, off_lo, off_hi in zip(coords, lo_offsets, hi_offsets):
        lo = max(lo, -int(coord) - int(off_lo))
        hi = min(hi, int(size) - 1 - int(coord) - int(off_hi))
    for coord, radius in zip(coords[1:], frame_radii):
        lo = max(lo, int(radius) - int(coord))
        hi = min(hi, int(size) - 1 - int(radius) - int(coord))
    if lo > hi:
        return None
    return int(lo), int(hi)


def _pick_canvas_and_translation(
    rng: np.random.Generator,
    template: str,
    heads: list[tuple[int, int]],
    facings: list[int],
) -> tuple[int, int, int, int] | None:
    row_coords = [int(r) for r, _ in heads]
    col_coords = [int(c) for _, c in heads]
    row_lo_offsets = []
    row_hi_offsets = []
    col_lo_offsets = []
    col_hi_offsets = []
    for facing in facings:
        r0, r1, c0, c1 = _template_bounds(template, facing)
        row_lo_offsets.append(int(r0))
        row_hi_offsets.append(int(r1))
        col_lo_offsets.append(int(c0))
        col_hi_offsets.append(int(c1))

    frame_radii = list(range(1, len(heads)))
    row_choices: list[tuple[int, int, int]] = []
    col_choices: list[tuple[int, int, int]] = []
    for h in range(GRID_MIN, GRID_MAX + 1):
        interval = _translation_interval(h, row_coords, row_lo_offsets, row_hi_offsets, frame_radii)
        if interval is not None:
            row_choices.append((h, interval[0], interval[1]))
    for w in range(GRID_MIN, GRID_MAX + 1):
        interval = _translation_interval(w, col_coords, col_lo_offsets, col_hi_offsets, frame_radii)
        if interval is not None:
            col_choices.append((w, interval[0], interval[1]))
    if not row_choices or not col_choices:
        return None

    h, tr_lo, tr_hi = row_choices[int(rng.integers(0, len(row_choices)))]
    w, tc_lo, tc_hi = col_choices[int(rng.integers(0, len(col_choices)))]
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    return int(h), int(w), tr, tc


def _translate_points(points: list[tuple[int, int]], tr: int, tc: int) -> list[tuple[int, int]]:
    return [(int(r + tr), int(c + tc)) for r, c in points]


def _validate_semantics(
    shape: tuple[int, int],
    template: str,
    heads: list[tuple[int, int]],
    facings: list[int],
    step_dirs: list[int],
) -> bool:
    h, w = map(int, shape)
    if len(heads) != len(facings) or len(heads) != len(step_dirs) + 1:
        return False

    for idx, (head, facing) in enumerate(zip(heads, facings)):
        if not _fits_sprite(shape, head, template, facing):
            return False
        if idx >= 1 and not _fits_frame(shape, head, idx):
            return False

    for idx in range(len(step_dirs)):
        remaining = set(heads[idx + 1 :])
        reachable = _reachable_markers(heads[idx], facings[idx], remaining, shape, template)
        if len(reachable) != 1:
            return False
        direction = next(iter(reachable))
        if direction != step_dirs[idx]:
            return False
        nearest = reachable[direction][0][1]
        if nearest != heads[idx + 1]:
            return False

    for idx in range(1, len(heads)):
        if _direction(heads[idx - 1], heads[idx]) != step_dirs[idx - 1]:
            return False
        if idx >= 2 and step_dirs[idx - 1] == _opposite(step_dirs[idx - 2]):
            return False
    return True


def _render_input(
    shape: tuple[int, int],
    template: str,
    initial_head: tuple[int, int],
    initial_facing: int,
    markers: list[tuple[int, int]],
) -> np.ndarray:
    inp = np.full(shape, BG, dtype=int)
    for r, c in _sprite_cells(initial_head, template, initial_facing):
        inp[r, c] = SPRITE
    for r, c in markers:
        inp[r, c] = MARKER
    return inp


def _build_case(rng: np.random.Generator) -> np.ndarray | None:
    n_steps = int(rng.integers(STEP_COUNT_RANGE[0], STEP_COUNT_RANGE[1] + 1))
    template = str(rng.choice(list(TEMPLATES)))
    step_dirs = _sample_step_dirs(rng, n_steps)
    if not _sequence_is_interesting(step_dirs):
        return None

    rollout = _rollout_relative_path(rng, template, step_dirs)
    if rollout is None:
        return None
    rel_heads, rel_facings, _step_distances = rollout

    canvas = _pick_canvas_and_translation(rng, template, rel_heads, rel_facings)
    if canvas is None:
        return None
    h, w, tr, tc = canvas

    heads = _translate_points(rel_heads, tr, tc)
    facings = list(map(int, rel_facings))
    shape = (int(h), int(w))
    if not _validate_semantics(shape, template, heads, facings, step_dirs):
        return None

    inp = _render_input(shape, template, heads[0], facings[0], heads[1:])
    expected = _render_output(shape, template, heads, facings)

    # Independent checks beyond solver self-consistency.
    if np.count_nonzero(inp == MARKER) != n_steps:
        return None
    if np.count_nonzero(inp == SPRITE) not in (3, 4):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    # Prevent degenerate examples where removing visit order would still work too easily.
    frame_sizes = {2 * i + 1 for i in range(1, n_steps + 1)}
    if len(frame_sizes) != n_steps:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
