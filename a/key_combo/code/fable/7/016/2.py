"""ARC task: redirect a phase-continuous two-color ray with diagonal mirrors."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], p: tuple[int, int]) -> bool:
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _grey_components(grid: np.ndarray, ignored: tuple[int, int]) -> list[list[tuple[int, int]]]:
    """Return 8-connected grey components after removing the direction indicator."""
    unseen = {tuple(map(int, p)) for p in np.argwhere(grid == GREY)} - {ignored}
    result = []
    while unseen:
        seed = unseen.pop()
        comp = [seed]
        queue = deque([seed])
        while queue:
            r, c = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in unseen:
                        unseen.remove(q)
                        comp.append(q)
                        queue.append(q)
        result.append(comp)
    return result


def _decode(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    assert YELLOW not in g, "yellow is reserved for the output trail"

    marker_cells = np.argwhere((g != BLACK) & (g != GREY))
    assert len(marker_cells) == 1, "expected one colored marker pixel"
    marker = tuple(map(int, marker_cells[0]))
    marker_color = int(g[marker])

    adjacent_grey = []
    for dr, dc in CARDINAL:
        q = (marker[0] + dr, marker[1] + dc)
        if _in_bounds(g.shape, q) and int(g[q]) == GREY:
            adjacent_grey.append((q, (dr, dc)))
    assert len(adjacent_grey) == 1, "marker must have one cardinally adjacent grey indicator"
    indicator, direction = adjacent_grey[0]

    mirrors: dict[tuple[int, int], str] = {}
    components = _grey_components(g, indicator)
    assert components, "expected at least one mirror"
    for comp in components:
        assert len(comp) == 2, "each mirror must be an isolated two-pixel component"
        (r1, c1), (r2, c2) = comp
        assert abs(r1 - r2) == 1 and abs(c1 - c2) == 1, "mirror pixels must be diagonal"
        kind = "backslash" if (r1 - r2) * (c1 - c2) > 0 else "slash"
        mirrors[(r1, c1)] = kind
        mirrors[(r2, c2)] = kind
    return marker, marker_color, indicator, direction, mirrors


def _redirect(direction: tuple[int, int], kind: str) -> tuple[int, int]:
    dr, dc = direction
    return (dc, dr) if kind == "backslash" else (-dc, -dr)


def main(grid: np.ndarray) -> np.ndarray:
    """Trace a phase-continuous two-color ray, redirecting and swapping colors at diagonal mirrors.

    Input structure:
    - A black grid contains one non-grey colored marker and one cardinally adjacent
      grey direction indicator. Every other grey component is an isolated pair of
      diagonally adjacent pixels, whose slope visibly specifies a / or \\ mirror.
    - Yellow does not occur in the input and is reserved for the added trail.

    Rule:
    - Starting just beyond the direction indicator, advance one cell at a time.
      Paint traversed black cells with a repeating primary, primary, secondary
      rhythm, initially using marker color as primary and yellow as secondary.
    - A grey mirror cell stays grey and consumes no beat. Its diagonal slope turns
      the ray by 90 degrees; at that same reflection primary and secondary swap.
      The beat phase continues across every reflection and the ray ends at the edge.

    Key insight:
    - Each mirror couples a geometric redirection with a color-role swap, while the
      three-cell phase belongs to the whole journey rather than to individual segments.
    """
    g = np.asarray(grid, dtype=int)
    marker, marker_color, indicator, direction, mirrors = _decode(g)
    out = g.copy()
    primary, secondary = marker_color, YELLOW
    phase = 0
    position = indicator
    seen_states = set()

    while True:
        position = (position[0] + direction[0], position[1] + direction[1])
        if not _in_bounds(g.shape, position):
            break
        state = (position, direction)
        assert state not in seen_states, "ray must terminate rather than cycle"
        seen_states.add(state)

        if position in mirrors:
            direction = _redirect(direction, mirrors[position])
            primary, secondary = secondary, primary
            continue
        assert int(g[position]) == BLACK, "ray may cross only background or mirror cells"
        out[position] = primary if phase in (0, 1) else secondary
        phase = (phase + 1) % 3
    return out


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], distance: int):
    return [(start[0] + i * direction[0], start[1] + i * direction[1]) for i in range(1, distance + 1)]


def _build_candidate() -> tuple[np.ndarray, dict] | None:
    h = random.randint(13, 25)
    w = random.randint(13, 25)
    shape = (h, w)
    marker_color = random.choice((1, 2, 3, 6, 7, 8, 9))
    turns = random.randint(2, 4)
    direction = random.choice(CARDINAL)

    # Leave room for an indicator, a visible pre-reflection rhythm, and a mirror.
    marker = (random.randint(2, h - 3), random.randint(2, w - 3))
    indicator = (marker[0] + direction[0], marker[1] + direction[1])
    if not _in_bounds(shape, indicator):
        return None

    path = {marker, indicator}
    hit_points: list[tuple[int, int]] = []
    kinds: list[str] = []
    directions = [direction]
    position = marker
    painted_before_hits = []
    painted_count = 0

    for turn_index in range(turns):
        dr, dc = direction
        max_dist = (h - 2 - position[0] if dr > 0 else position[0] - 1) if dr else (
            w - 2 - position[1] if dc > 0 else position[1] - 1
        )
        min_dist = 5 if turn_index == 0 else 4
        choices = list(range(min_dist, max_dist + 1))
        random.shuffle(choices)
        chosen = None
        for dist in choices:
            cells = _segment_cells(position, direction, dist)
            # The first segment deliberately includes the already-reserved indicator.
            overlap = set(cells) & path
            allowed = {indicator} if turn_index == 0 else set()
            if overlap <= allowed:
                chosen = (dist, cells)
                break
        if chosen is None:
            return None
        dist, cells = chosen
        path.update(cells)
        position = cells[-1]
        hit_points.append(position)
        newly_painted = dist - (2 if turn_index == 0 else 1)
        painted_count += newly_painted
        painted_before_hits.append(painted_count)

        left = (-dc, dr)
        right = (dc, -dr)
        new_direction = random.choice((left, right))
        kind = "backslash" if new_direction == (dc, dr) else "slash"
        kinds.append(kind)
        direction = new_direction
        directions.append(direction)

    # The last segment must visibly show both roles after the final swap.
    dr, dc = direction
    exit_dist = (h - 1 - position[0] if dr > 0 else position[0]) if dr else (
        w - 1 - position[1] if dc > 0 else position[1]
    )
    if exit_dist < 4:
        return None
    final_cells = _segment_cells(position, direction, exit_dist)
    if set(final_cells) & path:
        return None
    path.update(final_cells)

    # Pick the unused endpoint of each diagonal mirror only after the complete
    # centerline is known, so no decorative mirror pixel lies on the trajectory.
    companions = []
    forbidden = set(path)
    for hit, kind in zip(hit_points, kinds):
        offsets = ((1, 1), (-1, -1)) if kind == "backslash" else ((1, -1), (-1, 1))
        options = [(hit[0] + a, hit[1] + b) for a, b in offsets]
        options = [q for q in options if _in_bounds(shape, q) and q not in forbidden]
        random.shuffle(options)
        if not options:
            return None
        companion = options[0]
        companions.append(companion)
        forbidden.add(companion)

    grey_groups = [{indicator}] + [{a, b} for a, b in zip(hit_points, companions)]
    for i, group in enumerate(grey_groups):
        for other in grey_groups[i + 1 :]:
            if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for a in group for b in other):
                return None

    # At least one bounce must expose phase continuity (a phase reset would differ).
    if all(count % 3 == 0 for count in painted_before_hits):
        return None

    grid = np.zeros(shape, dtype=int)
    grid[marker] = marker_color
    grid[indicator] = GREY
    for a, b in zip(hit_points, companions):
        grid[a] = GREY
        grid[b] = GREY
    meta = {"hits": set(hit_points), "turns": turns, "painted_before_hits": painted_before_hits}
    return grid, meta


def generate_input() -> np.ndarray:
    """Generate varied, non-self-intersecting examples with two to four necessary mirrors."""
    for _ in range(2000):
        candidate = _build_candidate()
        if candidate is None:
            continue
        grid, meta = candidate
        try:
            out = main(grid)
            marker, marker_color, indicator, direction, mirrors = _decode(grid)
        except AssertionError:
            continue
        changed = set(map(tuple, np.argwhere(out != grid)))
        # Semantic checks: every intended mirror is encountered, both output colors
        # occur, and the transform adds a substantial but non-dominating trail.
        if not meta["hits"].issubset(mirrors):
            continue
        if not ({marker_color, YELLOW} <= set(map(int, np.unique(out)))):
            continue
        if len(changed) < 10 or len(changed) > grid.size // 2:
            continue
        return grid
    raise RuntimeError("failed to generate a valid mirror-ray input")
