from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9
WHITE = 10

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_MAP = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
SOURCE_COLOR_POOL = (BLUE, RED, GREEN, PINK, TEAL)
RESERVED_INPUT_COLORS = {BLACK, GREY, YELLOW, ORANGE, WHITE, MAROON}
MAX_GENERATION_ATTEMPTS = 300


def _build_adjacency(mask: np.ndarray) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    adjacency: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        neighbors: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
                neighbors.append((int(nr), int(nc)))
        adjacency[(rr, cc)] = tuple(neighbors)
    return adjacency


def _classify_endpoints(
    grid: np.ndarray,
    adjacency: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> tuple[list[tuple[int, tuple[int, int]]], set[tuple[int, int]]]:
    sources: list[tuple[int, tuple[int, int]]] = []
    terminals: set[tuple[int, int]] = set()

    for cell, neighbors in adjacency.items():
        r, c = cell
        color = int(grid[r, c])
        degree = len(neighbors)

        if color == WHITE:
            assert degree == 1, "terminal must be a degree-1 endpoint"
            terminals.add(cell)
            continue

        if color in (YELLOW, ORANGE):
            assert degree in (3, 4), "junction markers must sit on T or cross intersections"
            continue

        if color == GREY:
            assert degree == 2, "plain pipe cells must be straight-through cells"
            n0, n1 = neighbors
            assert n0[0] == n1[0] or n0[1] == n1[1], "plain pipe cells must not bend"
            continue

        assert color not in RESERVED_INPUT_COLORS, "unexpected reserved color in input"
        assert degree == 1, "sources must be colored degree-1 endpoints"
        sources.append((color, cell))

    source_colors = [color for color, _cell in sources]
    assert source_colors, "expected at least one source"
    assert len(source_colors) == len(set(source_colors)), "source colors must be unique"
    return sources, terminals


def _signal_exits(
    grid: np.ndarray,
    adjacency: dict[tuple[int, int], tuple[tuple[int, int], ...]],
    prev: tuple[int, int] | None,
    curr: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    neighbors = adjacency[curr]
    if prev is None:
        candidates = list(neighbors)
    else:
        candidates = [cell for cell in neighbors if cell != prev]

    if not candidates:
        return ()

    r, c = curr
    color = int(grid[r, c])
    if color == YELLOW:
        preferred = [cell for cell in candidates if cell[0] == r]
        return tuple(preferred if preferred else candidates)
    if color == ORANGE:
        preferred = [cell for cell in candidates if cell[1] == c]
        return tuple(preferred if preferred else candidates)
    return tuple(candidates)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route simultaneous colored signals through a pipe network and recolor reached terminals.

    Input structure
    - Black background.
    - Grey one-pixel pipes form horizontal/vertical segments.
    - Degree-1 endpoints are colored sources or white terminals.
    - T/cross intersection centers are yellow (prefer horizontal exits) or orange
      (prefer vertical exits).

    Rule
    - Each source emits a signal carrying its own color.
    - Signals move synchronously one cell per step.
    - At a yellow/orange junction, a signal takes every non-backtracking exit on
      the preferred axis; if that axis has no available exit, it takes all
      non-backtracking exits on the other axis.
    - If different colors arrive at the same pixel in the same step, they all
      disappear and that pixel becomes maroon.
    - A white terminal reached by a single color is recolored to that color.
    - Pipe pixels and junction markers otherwise remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    adjacency = _build_adjacency(g != BLACK)
    sources, terminals = _classify_endpoints(g, adjacency)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()
    claimed_terminals: set[tuple[int, int]] = set()

    active: list[tuple[int, tuple[int, int] | None, tuple[int, int]]] = [
        (int(color), None, cell) for color, cell in sources
    ]
    seen_states: set[tuple[int, int, int, int, int]] = set()

    for _step in range(8 * g.size):
        proposals: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
        for color, prev, curr in active:
            if curr in blocked or curr in claimed_terminals:
                continue
            for nxt in _signal_exits(g, adjacency, prev, curr):
                proposals.setdefault(nxt, []).append((int(color), curr))

        if not proposals:
            break

        next_active: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        next_keys_this_step: set[tuple[int, int, int, int, int]] = set()

        for target, arrivals in proposals.items():
            if target in blocked:
                continue

            if target in claimed_terminals:
                continue

            incoming_colors = {int(color) for color, _curr in arrivals}
            if len(incoming_colors) >= 2:
                out[target] = MAROON
                blocked.add(target)
                continue

            color = int(next(iter(incoming_colors)))
            if target in terminals:
                out[target] = color
                claimed_terminals.add(target)
                continue

            for _color, prev_cell in arrivals:
                key = (color, int(prev_cell[0]), int(prev_cell[1]), int(target[0]), int(target[1]))
                if key in seen_states or key in next_keys_this_step:
                    continue
                seen_states.add(key)
                next_keys_this_step.add(key)
                next_active.append((color, prev_cell, target))

        active = next_active

    return out


def _crop_patch(cells: dict[tuple[int, int], int]) -> np.ndarray:
    rows = [coord[0] for coord in cells]
    cols = [coord[1] for coord in cells]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for (r, c), color in cells.items():
        patch[int(r - r0), int(c - c0)] = int(color)
    return patch


def _build_junction_patch(center_color: int, arms: dict[str, tuple[int, int]]) -> np.ndarray:
    cells: dict[tuple[int, int], int] = {(0, 0): int(center_color)}
    for direction, (length, endpoint_color) in arms.items():
        assert length >= 1
        dr, dc = DIR_MAP[direction]
        for step in range(1, length):
            cells[(int(dr * step), int(dc * step))] = GREY
        cells[(int(dr * length), int(dc * length))] = int(endpoint_color)
    return _crop_patch(cells)


def _build_line_patch(axis: str, total_len: int, start_color: int, end_color: int) -> np.ndarray:
    assert axis in ("h", "v")
    assert total_len >= 3
    cells: dict[tuple[int, int], int] = {}
    for idx in range(total_len):
        if idx == 0:
            color = int(start_color)
        elif idx == total_len - 1:
            color = int(end_color)
        else:
            color = GREY
        cells[(0, idx) if axis == "h" else (idx, 0)] = color
    return _crop_patch(cells)


def _random_mirror(patch: np.ndarray) -> np.ndarray:
    out = patch
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, copy=True)


def _rand_len(lo: int, hi: int) -> int:
    return int(random.randint(lo, hi))


def _make_preferred_t(source_color: int) -> np.ndarray:
    prefer_axis = random.choice(("h", "v"))
    if prefer_axis == "h":
        source_dir = random.choice(("L", "R"))
        branch_dir = random.choice(("U", "D"))
        arms = {
            source_dir: (_rand_len(1, 4), source_color),
            OPPOSITE[source_dir]: (_rand_len(1, 4), WHITE),
            branch_dir: (_rand_len(1, 3), WHITE),
        }
        patch = _build_junction_patch(YELLOW, arms)
    else:
        source_dir = random.choice(("U", "D"))
        branch_dir = random.choice(("L", "R"))
        arms = {
            source_dir: (_rand_len(1, 4), source_color),
            OPPOSITE[source_dir]: (_rand_len(1, 4), WHITE),
            branch_dir: (_rand_len(1, 3), WHITE),
        }
        patch = _build_junction_patch(ORANGE, arms)
    return _random_mirror(patch)


def _make_fallback_t(source_color: int) -> np.ndarray:
    prefer_axis = random.choice(("h", "v"))
    if prefer_axis == "h":
        source_dir = random.choice(("L", "R"))
        arms = {
            source_dir: (_rand_len(1, 4), source_color),
            "U": (_rand_len(1, 3), WHITE),
            "D": (_rand_len(1, 3), WHITE),
        }
        patch = _build_junction_patch(YELLOW, arms)
    else:
        source_dir = random.choice(("U", "D"))
        arms = {
            source_dir: (_rand_len(1, 4), source_color),
            "L": (_rand_len(1, 3), WHITE),
            "R": (_rand_len(1, 3), WHITE),
        }
        patch = _build_junction_patch(ORANGE, arms)
    return _random_mirror(patch)


def _make_cross_split(source_color: int) -> np.ndarray:
    prefer_axis = random.choice(("h", "v"))
    if prefer_axis == "h":
        source_dir = random.choice(("U", "D"))
        arms = {
            source_dir: (_rand_len(1, 4), source_color),
            OPPOSITE[source_dir]: (_rand_len(1, 3), WHITE),
            "L": (_rand_len(1, 3), WHITE),
            "R": (_rand_len(1, 3), WHITE),
        }
        patch = _build_junction_patch(YELLOW, arms)
    else:
        source_dir = random.choice(("L", "R"))
        arms = {
            source_dir: (_rand_len(1, 4), source_color),
            OPPOSITE[source_dir]: (_rand_len(1, 3), WHITE),
            "U": (_rand_len(1, 3), WHITE),
            "D": (_rand_len(1, 3), WHITE),
        }
        patch = _build_junction_patch(ORANGE, arms)
    return _random_mirror(patch)


def _make_straight_delivery(source_color: int) -> np.ndarray:
    axis = random.choice(("h", "v"))
    return _build_line_patch(axis, _rand_len(4, 8), source_color, WHITE)


def _make_collision_line(source_a: int, source_b: int) -> np.ndarray:
    axis = random.choice(("h", "v"))
    total_len = random.choice((5, 7, 9))
    return _build_line_patch(axis, total_len, source_a, source_b)


def _patch_nonblack_coords(patch: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(patch != BLACK)]


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, padding: int = 1) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False

    for pr, pc in _patch_nonblack_coords(patch):
        r = top + pr
        c = left + pc
        if int(grid[r, c]) != BLACK:
            return False
        for rr in range(max(0, r - padding), min(h, r + padding + 1)):
            for cc in range(max(0, c - padding), min(w, c + padding + 1)):
                if int(grid[rr, cc]) != BLACK:
                    return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for pr, pc in _patch_nonblack_coords(patch):
        grid[top + pr, left + pc] = int(patch[pr, pc])


def _place_patch_random(grid: np.ndarray, patch: np.ndarray) -> bool:
    ph, pw = patch.shape
    if ph > grid.shape[0] or pw > grid.shape[1]:
        return False

    for _ in range(200):
        top = random.randint(0, grid.shape[0] - ph)
        left = random.randint(0, grid.shape[1] - pw)
        if _can_place_patch(grid, patch, top, left):
            _paste_patch(grid, patch, top, left)
            return True
    return False


def _sample_patches() -> list[np.ndarray]:
    branch_mode = random.choice(("preferred", "fallback"))
    motif_kinds: list[str] = []

    if branch_mode == "preferred":
        motif_kinds.append(random.choice(("preferred_t", "cross_split")))
    else:
        motif_kinds.append("fallback_t")

    motif_kinds.append("straight")

    if random.random() < 0.5:
        motif_kinds.append("collision")

    if random.random() < 0.35:
        motif_kinds.append(random.choice(("preferred_t", "fallback_t", "cross_split")))

    source_count = sum(2 if kind == "collision" else 1 for kind in motif_kinds)
    assert source_count <= len(SOURCE_COLOR_POOL)
    source_colors = random.sample(list(SOURCE_COLOR_POOL), source_count)
    color_idx = 0
    patches: list[np.ndarray] = []

    for kind in motif_kinds:
        if kind == "preferred_t":
            patches.append(_make_preferred_t(source_colors[color_idx]))
            color_idx += 1
        elif kind == "fallback_t":
            patches.append(_make_fallback_t(source_colors[color_idx]))
            color_idx += 1
        elif kind == "cross_split":
            patches.append(_make_cross_split(source_colors[color_idx]))
            color_idx += 1
        elif kind == "straight":
            patches.append(_make_straight_delivery(source_colors[color_idx]))
            color_idx += 1
        else:
            assert kind == "collision"
            patches.append(_make_collision_line(source_colors[color_idx], source_colors[color_idx + 1]))
            color_idx += 2

    return patches


def _gen_attempt() -> np.ndarray | None:
    patches = _sample_patches()
    min_h = max(max(patch.shape[0] for patch in patches) + 2, 16)
    min_w = max(max(patch.shape[1] for patch in patches) + 2, 16)
    height = random.randint(min_h, max(min_h, 28))
    width = random.randint(min_w, max(min_w, 28))
    grid = np.full((height, width), BLACK, dtype=int)

    patches.sort(key=lambda patch: (-int(np.count_nonzero(patch != BLACK)), -patch.shape[0] * patch.shape[1]))
    for patch in patches:
        if not _place_patch_random(grid, patch):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-routing instance")
