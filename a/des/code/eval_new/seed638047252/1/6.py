from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

SCENARIOS = (
    {"mirror_mode": "/", "need_crossing": False, "need_multi": False},
    {"mirror_mode": "\\", "need_crossing": False, "need_multi": False},
    {"mirror_mode": "mixed", "need_crossing": True, "need_multi": False},
    {"mirror_mode": "mixed", "need_crossing": False, "need_multi": True},
    {"mirror_mode": "mixed", "need_crossing": True, "need_multi": True},
)

MAX_GEN_ATTEMPTS = 600
MAX_LAYOUT_ATTEMPTS = 120
MAX_SEGMENT_ATTEMPTS = 120
MAX_SUBSET_ATTEMPTS = 600


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored edge rays through diagonal grey mirrors and mark path crossings in maroon.

    Input structure:
    - Background is black.
    - Grey pixels form one or more straight diagonal mirror segments, each running only as "/" or "\" and having length at least 3.
    - Colored single-pixel light sources sit on the grid edges, never in corners; each source color is unique and emits inward perpendicular to its edge.

    Rule:
    - Each source launches a ray from its own edge cell.
    - The ray advances one cardinal step at a time; when its next step would enter a grey mirror pixel, it reflects by 90 degrees without entering that grey pixel.
    - A "/" mirror maps down->right, up->left, right->up, left->down; a "\" mirror maps down->left, up->right, right->down, left->up.
    - Rays continue until the next step would leave the grid. Grey mirrors stay grey, source pixels stay their source color, and traversed black cells are painted with that source color.
    - If two different source colors traverse the same black cell, that cell becomes maroon.

    Key insight:
    - The grey diagonal components must first be classified as "/" or "\" mirrors; only then can each ray be simulated step by step, with crossings detected by overlaying the traced paths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape

    mirrors = _mirror_map(g)
    sources = _find_sources(g)

    out = np.full_like(g, BG)
    out[g == GREY] = GREY

    seen_colors: dict[tuple[int, int], set[int]] = defaultdict(set)

    for r, c, color, direction in sources:
        out[r, c] = color
        trace = _trace_ray(g, mirrors, (r, c), direction)
        for rr, cc in trace["paint_cells"]:
            colors_here = seen_colors[(rr, cc)]
            if color in colors_here:
                continue
            colors_here.add(color)
            if len(colors_here) == 1:
                out[rr, cc] = color
            else:
                out[rr, cc] = MAROON

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        scenario = random.choice(SCENARIOS)
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray ARC input")


def _gen_attempt(scenario: dict[str, object]) -> np.ndarray | None:
    need_crossing = bool(scenario["need_crossing"])
    need_multi = bool(scenario["need_multi"])
    mirror_mode = str(scenario["mirror_mode"])

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        H, W = _choose_dims(need_crossing=need_crossing, need_multi=need_multi)
        grid = np.full((H, W), BG, dtype=int)
        placed = _place_mirrors(grid, mirror_mode=mirror_mode, need_multi=need_multi)
        if not placed:
            continue

        mirrors = _mirror_map(grid)
        candidates = _candidate_sources(grid, mirrors)
        if len(candidates) < 2:
            continue

        required_orients = {mirror_mode} if mirror_mode in ("/", "\\") else {"/", "\\"}
        subset = _select_source_subset(
            candidates,
            required_orients=required_orients,
            need_crossing=need_crossing,
            need_multi=need_multi,
            shape=(H, W),
        )
        if subset is None:
            continue

        final = grid.copy()
        colors = random.sample(SOURCE_COLORS, len(subset))
        random.shuffle(colors)
        for cand, color in zip(subset, colors):
            r, c = cand["start"]
            final[r, c] = color

        out = main(final)
        if np.array_equal(out, final):
            continue
        if need_crossing and not np.any(out == MAROON):
            continue
        if _non_bg_density(out) > 0.48:
            continue
        return final

    return None


def _choose_dims(*, need_crossing: bool, need_multi: bool) -> tuple[int, int]:
    if need_crossing and need_multi:
        return random.randint(18, 24), random.randint(18, 24)
    if need_crossing or need_multi:
        return random.randint(15, 23), random.randint(15, 23)
    return random.randint(12, 20), random.randint(12, 20)


def _place_mirrors(grid: np.ndarray, *, mirror_mode: str, need_multi: bool) -> bool:
    H, W = grid.shape
    segments: list[list[tuple[int, int]]] = []
    blocked: set[tuple[int, int]] = set()

    min_segments = 3 if mirror_mode == "mixed" else 2
    max_segments = 5 if mirror_mode == "mixed" else 4
    if need_multi:
        min_segments += 1
        max_segments += 1
    num_segments = random.randint(min_segments, max_segments)

    if mirror_mode == "mixed":
        pending_orients = ["/", "\\"]
        while len(pending_orients) < num_segments:
            pending_orients.append(random.choice(["/", "\\"]))
    else:
        pending_orients = [mirror_mode] * num_segments
    random.shuffle(pending_orients)

    for orient in pending_orients:
        placed = False
        for _ in range(MAX_SEGMENT_ATTEMPTS):
            length = random.randint(3, min(7, H - 2, W - 2))
            cells = _random_segment_cells(H, W, orient, length)
            if cells is None:
                continue
            if _segment_blocked(cells, blocked, H, W):
                continue
            for r, c in cells:
                grid[r, c] = GREY
            segments.append(cells)
            for cell in cells:
                blocked.update(_neighbors8(cell, H, W))
            placed = True
            break
        if not placed:
            for cells in segments:
                for r, c in cells:
                    grid[r, c] = BG
            return False

    return True


def _random_segment_cells(H: int, W: int, orient: str, length: int) -> list[tuple[int, int]] | None:
    if orient == "\\":
        if H - length - 1 < 1 or W - length - 1 < 1:
            return None
        r0 = random.randint(1, H - length - 1)
        c0 = random.randint(1, W - length - 1)
        return [(r0 + i, c0 + i) for i in range(length)]

    if H - length - 1 < 1 or length > W - 2:
        return None
    r0 = random.randint(1, H - length - 1)
    c0 = random.randint(length, W - 2)
    return [(r0 + i, c0 - i) for i in range(length)]


def _segment_blocked(
    cells: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    H: int,
    W: int,
) -> bool:
    return any((r, c) in blocked or r <= 0 or r >= H - 1 or c <= 0 or c >= W - 1 for r, c in cells)


def _candidate_sources(grid: np.ndarray, mirrors: dict[tuple[int, int], str]) -> list[dict[str, object]]:
    H, W = grid.shape
    candidates: list[dict[str, object]] = []
    starts = []

    for c in range(1, W - 1):
        starts.append(((0, c), DOWN))
        starts.append(((H - 1, c), UP))
    for r in range(1, H - 1):
        starts.append(((r, 0), RIGHT))
        starts.append(((r, W - 1), LEFT))

    for start, direction in starts:
        r, c = start
        if grid[r, c] != BG:
            continue
        trace = _trace_ray(grid, mirrors, start, direction)
        if trace["cycle"]:
            continue
        paint_cells = tuple(trace["paint_cells"])
        if len(paint_cells) < 3:
            continue
        candidates.append(
            {
                "start": start,
                "direction": direction,
                "paint_cells": paint_cells,
                "paint_set": set(paint_cells),
                "visited_set": set(trace["visited_cells"]),
                "reflections": trace["reflections"],
                "used_orients": set(trace["used_orients"]),
            }
        )

    return candidates


def _select_source_subset(
    candidates: list[dict[str, object]],
    *,
    required_orients: set[str],
    need_crossing: bool,
    need_multi: bool,
    shape: tuple[int, int],
) -> list[dict[str, object]] | None:
    cross_pairs = [
        (i, j)
        for i in range(len(candidates))
        for j in range(i + 1, len(candidates))
        if _interior_intersection(candidates[i]["paint_set"], candidates[j]["paint_set"], shape)
    ]
    multi_idxs = [i for i, cand in enumerate(candidates) if int(cand["reflections"]) >= 2]

    if need_crossing and not cross_pairs:
        return None
    if need_multi and not multi_idxs:
        return None

    indices = list(range(len(candidates)))
    max_sources = min(4, len(candidates), len(SOURCE_COLORS))

    for _ in range(MAX_SUBSET_ATTEMPTS):
        chosen: list[int] = []

        if need_crossing:
            a, b = random.choice(cross_pairs)
            chosen.extend([a, b])

        if need_multi and not any(idx in multi_idxs for idx in chosen):
            chosen.append(random.choice(multi_idxs))

        chosen = list(dict.fromkeys(chosen))
        min_sources = max(2, len(chosen))
        if min_sources > max_sources:
            continue

        target_count = random.randint(min_sources, max_sources)
        extras = [idx for idx in indices if idx not in chosen]
        random.shuffle(extras)
        for idx in extras:
            if len(chosen) >= target_count:
                break
            chosen.append(idx)

        subset = [candidates[idx] for idx in chosen]
        if not _subset_ok(
            subset,
            required_orients=required_orients,
            need_crossing=need_crossing,
            need_multi=need_multi,
            shape=shape,
        ):
            continue
        return subset

    return None


def _subset_ok(
    subset: list[dict[str, object]],
    *,
    required_orients: set[str],
    need_crossing: bool,
    need_multi: bool,
    shape: tuple[int, int],
) -> bool:
    H, W = shape
    starts = [cand["start"] for cand in subset]
    if len(starts) != len(set(starts)):
        return False

    for i, (r0, c0) in enumerate(starts):
        for j, (r1, c1) in enumerate(starts):
            if i >= j:
                continue
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False

    for i, cand in enumerate(subset):
        visited = cand["visited_set"]
        for j, other in enumerate(subset):
            if i == j:
                continue
            if other["start"] in visited:
                return False

    union_orients: set[str] = set()
    has_crossing = False
    for cand in subset:
        union_orients.update(cand["used_orients"])

    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            if _interior_intersection(subset[i]["paint_set"], subset[j]["paint_set"], shape):
                has_crossing = True
                break
        if has_crossing:
            break

    if not required_orients.issubset(union_orients):
        return False
    if need_crossing and not has_crossing:
        return False
    if need_multi and not any(int(cand["reflections"]) >= 2 for cand in subset):
        return False
    if sum(int(cand["reflections"]) for cand in subset) < 1:
        return False

    painted_union: set[tuple[int, int]] = set()
    for cand in subset:
        painted_union.update(cand["paint_set"])
    if len(painted_union) < 8:
        return False
    if len(painted_union) > int(0.45 * H * W):
        return False

    return True


def _find_sources(grid: np.ndarray) -> list[tuple[int, int, int, tuple[int, int]]]:
    H, W = grid.shape
    sources: list[tuple[int, int, int, tuple[int, int]]] = []
    seen_colors: set[int] = set()

    colored = np.argwhere((grid != BG) & (grid != GREY))
    for rr, cc in colored:
        r = int(rr)
        c = int(cc)
        assert _is_boundary(r, c, H, W), "all non-grey foreground pixels must be edge sources"
        assert not _is_corner(r, c, H, W), "sources must not occupy corners"
        color = int(grid[r, c])
        assert color != MAROON, "maroon is reserved for output crossings"
        assert color not in seen_colors, "source colors must be unique"
        seen_colors.add(color)
        sources.append((r, c, color, _source_dir(r, c, H, W)))

    assert sources, "at least one source is required"
    sources.sort(key=lambda item: (item[0], item[1]))
    return sources


def _mirror_map(grid: np.ndarray) -> dict[tuple[int, int], str]:
    H, W = grid.shape
    grey_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == GREY)}
    seen: set[tuple[int, int]] = set()
    mirrors: dict[tuple[int, int], str] = {}

    for cell in grey_cells:
        if cell in seen:
            continue
        stack = [cell]
        comp: list[tuple[int, int]] = []
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in _neighbors8((r, c), H, W):
                if (nr, nc) in grey_cells and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))

        assert len(comp) >= 3, "mirror segments must have length at least 3"
        orient = _component_orientation(comp)
        for rc in comp:
            mirrors[rc] = orient

    return mirrors


def _component_orientation(comp: list[tuple[int, int]]) -> str:
    diffs = {r - c for r, c in comp}
    sums = {r + c for r, c in comp}

    if len(diffs) == 1:
        ordered = sorted(comp)
        assert all(
            ordered[i + 1][0] - ordered[i][0] == 1 and ordered[i + 1][1] - ordered[i][1] == 1
            for i in range(len(ordered) - 1)
        ), "backslash mirror must be contiguous"
        return "\\"

    assert len(sums) == 1, "grey components must be straight diagonal segments"
    ordered = sorted(comp)
    assert all(
        ordered[i + 1][0] - ordered[i][0] == 1 and ordered[i + 1][1] - ordered[i][1] == -1
        for i in range(len(ordered) - 1)
    ), "slash mirror must be contiguous"
    return "/"


def _trace_ray(
    grid: np.ndarray,
    mirrors: dict[tuple[int, int], str],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict[str, object]:
    H, W = grid.shape
    current = start
    move_dir = direction
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    visited_cells: list[tuple[int, int]] = []
    paint_cells: list[tuple[int, int]] = []
    used_orients: set[str] = set()
    reflections = 0

    while True:
        state = (current, move_dir)
        if state in seen_states:
            return {
                "visited_cells": tuple(visited_cells),
                "paint_cells": tuple(paint_cells),
                "used_orients": frozenset(used_orients),
                "reflections": reflections,
                "cycle": True,
            }
        seen_states.add(state)

        nr = current[0] + move_dir[0]
        nc = current[1] + move_dir[1]
        if not (0 <= nr < H and 0 <= nc < W):
            break

        mirror = mirrors.get((nr, nc))
        if mirror is not None:
            move_dir = _reflect(move_dir, mirror)
            reflections += 1
            used_orients.add(mirror)
            continue

        current = (nr, nc)
        visited_cells.append(current)
        if int(grid[current]) == BG:
            paint_cells.append(current)

    return {
        "visited_cells": tuple(visited_cells),
        "paint_cells": tuple(paint_cells),
        "used_orients": frozenset(used_orients),
        "reflections": reflections,
        "cycle": False,
    }


def _reflect(direction: tuple[int, int], mirror: str) -> tuple[int, int]:
    if mirror == "/":
        return {
            DOWN: RIGHT,
            UP: LEFT,
            RIGHT: UP,
            LEFT: DOWN,
        }[direction]
    return {
        DOWN: LEFT,
        UP: RIGHT,
        RIGHT: DOWN,
        LEFT: UP,
    }[direction]


def _source_dir(r: int, c: int, H: int, W: int) -> tuple[int, int]:
    if r == 0:
        return DOWN
    if r == H - 1:
        return UP
    if c == 0:
        return RIGHT
    assert c == W - 1
    return LEFT


def _interior_intersection(
    a: set[tuple[int, int]],
    b: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    H, W = shape
    for r, c in a & b:
        if 0 < r < H - 1 and 0 < c < W - 1:
            return True
    return False


def _neighbors8(cell: tuple[int, int], H: int, W: int) -> set[tuple[int, int]]:
    r, c = cell
    out = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out.add((rr, cc))
    return out


def _is_boundary(r: int, c: int, H: int, W: int) -> bool:
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def _is_corner(r: int, c: int, H: int, W: int) -> bool:
    return (r, c) in {(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)}


def _non_bg_density(grid: np.ndarray) -> float:
    return float(np.count_nonzero(grid != BG) / grid.size)
