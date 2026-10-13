from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Coord = Tuple[int, int]
Segment = Tuple[Coord, Coord, int]

DIRS4: dict[int, Coord] = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}

TEMPLATES: dict[str, dict[int, list[Coord]]] = {
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
    "midfilled6": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1), (2, -1), (2, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1), (-2, -1), (-2, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2)],
    },
}

# --------------- generator configuration ---------------

STEP_COUNT_RANGE = (5, 8)
STEP_DISTANCE_RANGE = (2, 5)
PADDING_RANGE = (1, 3)
MAX_CANVAS = 30
MAX_GENERATION_ATTEMPTS = 800


def _add(a: Coord, b: Coord) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def _scale(v: Coord, k: int) -> Coord:
    return v[0] * k, v[1] * k


def _line_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segment must be orthogonal")


def _template_cells(template_name: str, direction: int, head: Coord) -> set[Coord]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[template_name][direction]}


def _match_sprite(cells: set[Coord]) -> tuple[str, int, Coord]:
    matches: list[tuple[str, int, Coord]] = []
    for head in cells:
        for template_name, variants in TEMPLATES.items():
            for direction, offsets in variants.items():
                placed = {(head[0] + dr, head[1] + dc) for dr, dc in offsets}
                if placed == cells:
                    matches.append((template_name, direction, head))
    assert len(matches) == 1, "expected exactly one arrow sprite match"
    return matches[0]


def _parse_grid(grid: np.ndarray) -> tuple[str, int, Coord, int, dict[Coord, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(c) for c in np.unique(g) if int(c) != 0]
    actor_matches: list[tuple[str, int, Coord, int]] = []
    for color in colors:
        pts = set(map(tuple, np.argwhere(g == color)))
        if not (3 <= len(pts) <= 6):
            continue
        try:
            template_name, direction, head = _match_sprite(pts)
        except AssertionError:
            continue
        actor_matches.append((template_name, direction, head, color))

    assert len(actor_matches) == 1, "expected exactly one arrow-colored sprite"
    template_name, direction, head, actor_color = actor_matches[0]

    markers: dict[Coord, int] = {}
    for color in colors:
        if color == actor_color:
            continue
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2), "each marker color must occur exactly once"
        markers[(int(coords[0, 0]), int(coords[0, 1]))] = color

    return template_name, direction, head, actor_color, markers


def _visible_markers(head: Coord, markers: dict[Coord, int]) -> dict[int, list[tuple[int, Coord, int]]]:
    hr, hc = head
    visible: dict[int, list[tuple[int, Coord, int]]] = {}
    for (r, c), color in markers.items():
        if r == hr and c > hc:
            visible.setdefault(1, []).append((c - hc, (r, c), color))
        elif r == hr and c < hc:
            visible.setdefault(3, []).append((hc - c, (r, c), color))
        elif c == hc and r > hr:
            visible.setdefault(2, []).append((r - hr, (r, c), color))
        elif c == hc and r < hr:
            visible.setdefault(0, []).append((hr - r, (r, c), color))
    for items in visible.values():
        items.sort(key=lambda item: item[0])
    return visible


def _simulate(head: Coord, markers: dict[Coord, int]) -> tuple[list[Coord], list[int], list[Segment]]:
    remaining = dict(markers)
    heads = [head]
    step_dirs: list[int] = []
    segments: list[Segment] = []
    current = head

    while remaining:
        visible = _visible_markers(current, remaining)
        if not visible:
            break
        assert len(visible) == 1, "exactly one search direction must be available"
        step_dir, items = next(iter(visible.items()))
        _, target, color = items[0]
        step_dirs.append(step_dir)
        segments.append((current, target, color))
        del remaining[target]
        current = target
        heads.append(current)

    assert not remaining, "all markers must be consumed"
    return heads, step_dirs, segments


def _render_trajectory(shape: Sequence[int], segments: Sequence[Segment]) -> np.ndarray:
    out = np.zeros(tuple(shape), dtype=int)
    for idx, (start, end, color) in enumerate(segments):
        cells = _line_cells(start, end)
        if idx > 0:
            cells = cells[1:]
        for r, c in cells:
            out[r, c] = int(color)
    return out


def _analyze(grid: np.ndarray) -> tuple[str, int, Coord, int, dict[Coord, int], list[Coord], list[int], list[Segment]]:
    template_name, init_dir, head, actor_color, markers = _parse_grid(grid)
    heads, step_dirs, segments = _simulate(head, markers)
    return template_name, init_dir, head, actor_color, markers, heads, step_dirs, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the arrow's whole eating route and recolor each walked segment by the consumed marker at its end.

    Input structure:
    - A black background contains one arrow-shaped sprite made of one repeated color.
    - The remaining non-black cells are isolated single-pixel markers, each with its own color.
    - The arrow tip identifies the head cell and the current facing of the sprite.

    Rule:
    - Repeatedly inspect the four cardinal half-lines starting at the head.
    - At every step, markers lie on exactly one such half-line; move to the nearest marker on that unique direction, consume it, and update the facing to that direction.
    - Output a black canvas of the same size containing only the walked polyline: step k is colored entirely with the color of the marker eaten at the end of that step; include the first start cell in step 1, include every step endpoint, and exclude later step start cells.

    Key insight:
    - The answer is not the sprite's final pose but its movement history, and each segment can only be colored after identifying which endpoint marker that segment eventually reaches.
    """
    g = np.asarray(grid, dtype=int)
    _, _, _, _, _, _, _, segments = _analyze(g)
    return _render_trajectory(g.shape, segments)


def _relative_heads(step_dirs: Sequence[int], step_dists: Sequence[int]) -> list[Coord]:
    heads = [(0, 0)]
    for step_dir, dist in zip(step_dirs, step_dists):
        heads.append(_add(heads[-1], _scale(DIRS4[step_dir], dist)))
    return heads


def _has_axis_reversal(step_dirs: Sequence[int]) -> bool:
    return (0 in step_dirs and 2 in step_dirs) or (1 in step_dirs and 3 in step_dirs)


def _validate_head_path(heads: Sequence[Coord], step_dirs: Sequence[int]) -> bool:
    if len(set(heads)) != len(heads):
        return False
    if len(set(step_dirs)) < 3:
        return False
    if not _has_axis_reversal(step_dirs):
        return False
    if not any(step_dirs[i] == step_dirs[i + 1] for i in range(len(step_dirs) - 1)):
        return False

    visited: set[Coord] = set()
    for idx in range(len(step_dirs)):
        cells = _line_cells(heads[idx], heads[idx + 1])
        for k, cell in enumerate(cells):
            if idx > 0 and k == 0:
                continue
            if cell in visited:
                return False
            visited.add(cell)

    for idx, step_dir in enumerate(step_dirs):
        current = heads[idx]
        visible: list[tuple[int, int, int]] = []
        for later in range(idx + 1, len(heads)):
            r, c = heads[later]
            cr, cc = current
            if r == cr and c > cc:
                visible.append((c - cc, 1, later))
            elif r == cr and c < cc:
                visible.append((cc - c, 3, later))
            elif c == cc and r > cr:
                visible.append((r - cr, 2, later))
            elif c == cc and r < cr:
                visible.append((cr - r, 0, later))
        if not visible:
            return False
        dirs = {direction for _, direction, _ in visible}
        if dirs != {step_dir}:
            return False
        nearest = min(visible)
        if nearest[2] != idx + 1:
            return False
    return True


def _state_dirs(init_dir: int, step_dirs: Sequence[int]) -> list[int]:
    return [init_dir] + list(step_dirs)


def _validate_sprite_vs_markers(
    template_name: str,
    init_dir: int,
    heads: Sequence[Coord],
    step_dirs: Sequence[int],
) -> bool:
    dirs_by_state = _state_dirs(init_dir, step_dirs)
    for state_idx, (head, direction) in enumerate(zip(heads, dirs_by_state)):
        body = _template_cells(template_name, direction, head)
        for future_idx in range(state_idx + 1, len(heads)):
            if heads[future_idx] in body:
                return False
    return True


def _sample_step_dirs(n_steps: int) -> list[int]:
    for _ in range(400):
        dirs = [random.randrange(4) for _ in range(n_steps)]
        if len(set(dirs)) < 3:
            continue
        if not _has_axis_reversal(dirs):
            continue
        if not any(dirs[i] == dirs[i + 1] for i in range(n_steps - 1)):
            continue
        return dirs
    raise RuntimeError("failed to sample direction sequence")


def _sample_step_dists(n_steps: int) -> list[int]:
    for _ in range(200):
        dists = [random.randint(*STEP_DISTANCE_RANGE) for _ in range(n_steps)]
        if len(set(dists)) >= 2:
            return dists
    raise RuntimeError("failed to sample distance sequence")


def _split_extra(extra: int) -> tuple[int, int]:
    left = random.randint(1, extra - 1)
    return left, extra - left


def _place_path(
    template_name: str,
    init_dir: int,
    heads: Sequence[Coord],
    step_dirs: Sequence[int],
) -> tuple[tuple[int, int], dict[Coord, Coord]] | None:
    dirs_by_state = _state_dirs(init_dir, step_dirs)
    all_coords: set[Coord] = set(heads)
    for head, direction in zip(heads, dirs_by_state):
        all_coords |= _template_cells(template_name, direction, head)

    rows = [r for r, _ in all_coords]
    cols = [c for _, c in all_coords]
    base_h = max(rows) - min(rows) + 1
    base_w = max(cols) - min(cols) + 1
    max_extra_h = MAX_CANVAS - base_h
    max_extra_w = MAX_CANVAS - base_w
    if max_extra_h < 2 or max_extra_w < 2:
        return None

    extra_h = random.randint(2, min(max_extra_h, 2 * PADDING_RANGE[1]))
    extra_w = random.randint(2, min(max_extra_w, 2 * PADDING_RANGE[1]))
    top, bottom = _split_extra(extra_h)
    left, right = _split_extra(extra_w)
    if not (PADDING_RANGE[0] <= top <= PADDING_RANGE[1] and PADDING_RANGE[0] <= bottom <= PADDING_RANGE[1]):
        return None
    if not (PADDING_RANGE[0] <= left <= PADDING_RANGE[1] and PADDING_RANGE[0] <= right <= PADDING_RANGE[1]):
        return None

    shift_r = top - min(rows)
    shift_c = left - min(cols)
    shifted = {(r, c): (r + shift_r, c + shift_c) for r, c in all_coords}
    height = base_h + extra_h
    width = base_w + extra_w
    if height > MAX_CANVAS or width > MAX_CANVAS:
        return None
    return (height, width), shifted


def _translate_coords(coords: Iterable[Coord], mapping: dict[Coord, Coord]) -> list[Coord]:
    return [mapping[coord] for coord in coords]


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    n_steps = random.randint(*STEP_COUNT_RANGE)
    template_name = random.choice(list(TEMPLATES))
    init_dir = random.randrange(4)
    actor_color = random.randint(1, 9)

    step_dirs = _sample_step_dirs(n_steps)
    step_dists = _sample_step_dists(n_steps)
    heads = _relative_heads(step_dirs, step_dists)
    if not _validate_head_path(heads, step_dirs):
        return None
    if not _validate_sprite_vs_markers(template_name, init_dir, heads, step_dirs):
        return None

    placement = _place_path(template_name, init_dir, heads, step_dirs)
    if placement is None:
        return None
    shape, mapping = placement
    shifted_heads = _translate_coords(heads, mapping)

    marker_colors = random.sample([c for c in range(1, 10) if c != actor_color], n_steps)
    grid = np.zeros(shape, dtype=int)
    for r, c in _template_cells(template_name, init_dir, shifted_heads[0]):
        grid[r, c] = actor_color
    for pos, color in zip(shifted_heads[1:], marker_colors):
        if grid[pos] != 0:
            return None
        grid[pos] = color

    segments = [(shifted_heads[i], shifted_heads[i + 1], marker_colors[i]) for i in range(n_steps)]
    expected = _render_trajectory(shape, segments)
    if np.count_nonzero(expected) <= n_steps:
        return None

    try:
        template_check, init_dir_check, head_check, actor_check, markers_check, parsed_heads, parsed_dirs, parsed_segments = _analyze(grid)
    except AssertionError:
        return None

    if template_check != template_name or init_dir_check != init_dir or head_check != shifted_heads[0] or actor_check != actor_color:
        return None
    if parsed_heads != shifted_heads or parsed_dirs != step_dirs:
        return None
    if parsed_segments != segments:
        return None
    if len(markers_check) != n_steps or len(set(markers_check.values())) != n_steps:
        return None
    if not _validate_head_path(parsed_heads, parsed_dirs):
        return None
    if not _validate_sprite_vs_markers(template_name, init_dir, heads, step_dirs):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid trajectory-reconstruction instance")
