import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

MIRROR_COLORS = (BLUE, RED, GREEN, MAGENTA, ORANGE, AZURE, MAROON)
ORIENTATION_SCENARIOS = (
    ("horizontal",),
    ("vertical",),
    ("horizontal", "vertical"),
    ("vertical", "horizontal"),
    ("horizontal", "vertical", "horizontal"),
    ("vertical", "horizontal", "vertical"),
)
DIRECTOR_OPTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
GRID_H_RANGE = (12, 21)
GRID_W_RANGE = (12, 21)
EXTRA_MIRROR_RANGE = (0, 4)
GENERATE_RETRIES = 400

_GEN_CALL_COUNT = 0


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(nr, nc, h, w) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _toward_center_sign(index: int, size: int) -> int:
    return 1 if index * 2 <= size - 1 else -1


def _initial_direction(source: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = source
    touches_top = r == 0
    touches_bottom = r == h - 1
    touches_left = c == 0
    touches_right = c == w - 1
    assert any((touches_top, touches_bottom, touches_left, touches_right)), "source must lie on the grid edge"

    if touches_top:
        dr = 1
    elif touches_bottom:
        dr = -1
    else:
        dr = _toward_center_sign(r, h)

    if touches_left:
        dc = 1
    elif touches_right:
        dc = -1
    else:
        dc = _toward_center_sign(c, w)

    return dr, dc


def _extract_source(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == YELLOW)
    assert coords.shape == (1, 2), "expected exactly one yellow source pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _mirror_cells_from_hit(
    hit: tuple[int, int],
    orientation: str,
    offset: int,
    h: int,
    w: int,
) -> tuple[tuple[int, int], ...] | None:
    hr, hc = hit
    if orientation == "horizontal":
        start_c = hc - offset
        cells = tuple((hr, start_c + i) for i in range(3))
    else:
        start_r = hr - offset
        cells = tuple((start_r + i, hc) for i in range(3))
    if any(not _in_bounds(r, c, h, w) for r, c in cells):
        return None
    if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
        return None
    return cells


def _halo(cells: tuple[tuple[int, int], ...], h: int, w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(rr, cc, h, w):
                    blocked.add((rr, cc))
    return blocked


def _extract_mirrors(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    mask = (grid != BLACK) & (grid != YELLOW)
    mirrors: list[dict] = []
    mirror_at: dict[tuple[int, int], int] = {}
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each mirror must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        assert len(comp) == 3, "each mirror must have area 3"
        assert (hh, ww) in {(1, 3), (3, 1)}, "each mirror must be 1x3 or 3x1"
        assert r0 > 0 and c0 > 0 and r1 < grid.shape[0] - 1 and c1 < grid.shape[1] - 1, "mirrors may not touch the edge"
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                assert grid[rr, cc] == color, "mirror bbox must be solid"
        orientation = "horizontal" if hh == 1 else "vertical"
        idx = len(mirrors)
        cells = tuple(sorted(comp))
        mirrors.append({"cells": cells, "color": color, "orientation": orientation})
        for cell in cells:
            assert cell not in mirror_at, "mirrors may not overlap"
            mirror_at[cell] = idx
    return mirrors, mirror_at


def _beam_events(
    shape: tuple[int, int],
    source: tuple[int, int],
    mirrors: list[dict],
) -> dict:
    h, w = shape
    mirror_at: dict[tuple[int, int], int] = {}
    for idx, mirror in enumerate(mirrors):
        for cell in mirror["cells"]:
            mirror_at[cell] = idx

    r, c = source
    dr, dc = _initial_direction(source, h, w)
    color = GREY
    seen_states: set[tuple[int, int, int, int]] = set()
    events: list[tuple[str, tuple[int, int], int]] = []
    hit_indices: list[int] = []
    next_cells: set[tuple[int, int]] = set()
    segment_lengths: list[int] = []
    current_segment = 0

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "beam entered a cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(nr, nc, h, w):
            segment_lengths.append(current_segment)
            break

        next_cells.add((nr, nc))
        mirror_idx = mirror_at.get((nr, nc))
        if mirror_idx is not None:
            events.append(("reflect", (r, c), mirror_idx))
            hit_indices.append(mirror_idx)
            segment_lengths.append(current_segment)
            current_segment = 0
            color = mirrors[mirror_idx]["color"]
            if mirrors[mirror_idx]["orientation"] == "horizontal":
                dr = -dr
            else:
                dc = -dc
            continue

        r, c = nr, nc
        events.append(("move", (r, c), color))
        current_segment += 1

    return {
        "events": events,
        "hits": hit_indices,
        "next_cells": next_cells,
        "segment_lengths": segment_lengths,
    }


def _steps_to_exit(pos: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = pos
    dr, dc = direction
    steps = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(r, c, h, w):
            return steps
        steps += 1


def _place_hit_sequence(
    h: int,
    w: int,
    source: tuple[int, int],
    orientations: tuple[str, ...],
) -> list[dict] | None:
    start_dir = _initial_direction(source, h, w)
    max_steps = max(h, w)

    def dfs(
        pos: tuple[int, int],
        direction: tuple[int, int],
        idx: int,
        mirrors: list[dict],
        blocked: set[tuple[int, int]],
    ) -> list[dict] | None:
        if idx == len(orientations):
            if _steps_to_exit(pos, direction, h, w) >= 2:
                return mirrors
            return None

        orientation = orientations[idx]
        step_options = list(range(2, max_steps))
        random.shuffle(step_options)
        offset_options = [0, 1, 2]

        for steps in step_options:
            rp = (pos[0] + steps * direction[0], pos[1] + steps * direction[1])
            hit = (rp[0] + direction[0], rp[1] + direction[1])
            if not _in_bounds(rp[0], rp[1], h, w) or not _in_bounds(hit[0], hit[1], h, w):
                continue

            random.shuffle(offset_options)
            for offset in offset_options:
                cells = _mirror_cells_from_hit(hit, orientation, offset, h, w)
                if cells is None:
                    continue
                if any(cell in blocked for cell in cells):
                    continue

                new_direction = (-direction[0], direction[1]) if orientation == "horizontal" else (direction[0], -direction[1])
                candidate = mirrors + [{"cells": cells, "orientation": orientation, "color": -1}]
                result = dfs(rp, new_direction, idx + 1, candidate, blocked | _halo(cells, h, w))
                if result is not None:
                    return result
        return None

    return dfs(source, start_dir, 0, [], set())


def _source_for_direction(direction: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for c in range(w):
        choices.append((0, c))
        choices.append((h - 1, c))
    for r in range(1, h - 1):
        choices.append((r, 0))
        choices.append((r, w - 1))

    choices = [pos for pos in dict.fromkeys(choices) if _initial_direction(pos, h, w) == direction]
    assert choices, "expected at least one valid source location"
    return random.choice(choices)


def _all_extra_candidates(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    candidates: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for orientation in ("horizontal", "vertical"):
        if orientation == "horizontal":
            for r in range(1, h - 1):
                for c in range(1, w - 3):
                    cells = ((r, c), (r, c + 1), (r, c + 2))
                    if any(cell in blocked or cell in forbidden for cell in cells):
                        continue
                    candidates.append((orientation, cells))
        else:
            for r in range(1, h - 3):
                for c in range(1, w - 1):
                    cells = ((r, c), (r + 1, c), (r + 2, c))
                    if any(cell in blocked or cell in forbidden for cell in cells):
                        continue
                    candidates.append((orientation, cells))
    random.shuffle(candidates)
    return candidates


def _build_grid(h: int, w: int, source: tuple[int, int], mirrors: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[source] = YELLOW
    for mirror in mirrors:
        for r, c in mirror["cells"]:
            grid[r, c] = int(mirror["color"])
    return grid


def _gen_attempt(orientations: tuple[str, ...], direction: tuple[int, int]) -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    source = _source_for_direction(direction, h, w)

    mirrors = _place_hit_sequence(h, w, source, orientations)
    if mirrors is None:
        return None

    extra_count = random.randrange(*EXTRA_MIRROR_RANGE)
    total_needed = len(mirrors) + extra_count
    if total_needed > len(MIRROR_COLORS):
        return None

    sim = _beam_events((h, w), source, mirrors)
    hit_orientations = tuple(mirrors[idx]["orientation"] for idx in sim["hits"])
    if hit_orientations != orientations:
        return None
    if len(sim["hits"]) != len(orientations):
        return None
    if min(sim["segment_lengths"]) < 2:
        return None

    blocked: set[tuple[int, int]] = set()
    for mirror in mirrors:
        blocked |= _halo(mirror["cells"], h, w)
    forbidden = set(sim["next_cells"])

    extras: list[dict] = []
    extra_candidates = _all_extra_candidates(h, w, blocked, forbidden)
    for orientation_extra, cells_extra in extra_candidates:
        if len(extras) >= extra_count:
            break
        extras.append({"cells": cells_extra, "orientation": orientation_extra, "color": -1})
        blocked |= _halo(cells_extra, h, w)
    if len(extras) != extra_count:
        return None

    all_mirrors = mirrors + extras
    colors = random.sample(MIRROR_COLORS, len(all_mirrors))
    random.shuffle(colors)
    for mirror, color in zip(all_mirrors, colors):
        mirror["color"] = color

    final_sim = _beam_events((h, w), source, all_mirrors)
    final_orientations = tuple(all_mirrors[idx]["orientation"] for idx in final_sim["hits"])
    if final_orientations != orientations:
        return None
    if len(final_sim["hits"]) != len(orientations):
        return None
    if min(final_sim["segment_lengths"]) < 2:
        return None

    grid = _build_grid(h, w, source, all_mirrors)
    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    painted = int(np.count_nonzero((out != grid) & (grid == BLACK)))
    if painted < 6:
        return None
    if not np.any(out == GREY):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a diagonal light ray through colored 1x3 or 3x1 mirrors, recoloring the ray after each reflection.

    Input structure:
    - The background is black.
    - Exactly one yellow source pixel lies on the grid edge and emits a 45-degree diagonal ray inward.
    - Each other connected colored region is a solid 1x3 or 3x1 mirror bar fully inside the grid.
    - A horizontal mirror flips the ray's vertical component; a vertical mirror flips the horizontal component.

    Rule:
    - Start from the yellow source and advance the ray one diagonal cell at a time.
    - Every black cell the ray enters is painted grey until the ray encounters its first mirror.
    - If the next diagonal cell belongs to a mirror, the ray reflects from its current cell without entering the mirror, and from that reflection point onward the ray color becomes the hit mirror's color.
    - Continue with repeated reflections until the ray leaves the grid; keep all mirrors and the yellow source unchanged.

    Key insight:
    - The bar shape itself encodes the reflection axis, so solving the puzzle is a step-by-step diagonal ray simulation whose trail color updates at each mirror hit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source = _extract_source(g)
    mirrors, _mirror_at = _extract_mirrors(g)
    out = g.copy()

    for event_type, (r, c), payload in _beam_events(g.shape, source, mirrors)["events"]:
        if event_type == "move":
            if out[r, c] == BLACK:
                out[r, c] = payload
        else:
            mirror = mirrors[payload]
            if (r, c) != source:
                out[r, c] = int(mirror["color"])

    return out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    scenario = ORIENTATION_SCENARIOS[_GEN_CALL_COUNT % len(ORIENTATION_SCENARIOS)]
    direction = DIRECTOR_OPTIONS[(_GEN_CALL_COUNT // len(ORIENTATION_SCENARIOS)) % len(DIRECTOR_OPTIONS)]
    _GEN_CALL_COUNT += 1

    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(scenario, direction)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray input")
