from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from arc_utils import Color, get_bg, in_bounds
except Exception:
    _FILE = Path(__file__).resolve()
    for _p in _FILE.parents:
        candidate = _p / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            import sys

            sys.path.append(str(candidate.parent))
            break
    from arc_utils import Color, get_bg, in_bounds


def _reflect(dr: int, dc: int, mirror: str) -> Tuple[int, int]:
    """Reflect a directional delta by a diagonal mirror."""
    if mirror == "\\":
        return int(dc), int(dr)
    if mirror == "/":
        return int(-dc), int(-dr)
    raise ValueError(f"unsupported mirror orientation: {mirror}")


def _build_reflection_rules() -> Dict[Tuple[int, int], Dict[str, Tuple[int, int]]]:
    rules: Dict[Tuple[int, int], Dict[str, Tuple[int, int]]] = {}
    for dr, dc in [( -1, 0), (1, 0), (0, -1), (0, 1)]:
        rules[(dr, dc)] = {
            "\\": _reflect(dr, dc, "\\"),
            "/": _reflect(dr, dc, "/"),
        }
    return rules


_REFLECTION_RULES = _build_reflection_rules()


def _mirror_pairs(g: np.ndarray, bg: int) -> Dict[Tuple[int, int], Tuple[str, int]]:
    """Collect every diagonal mirror pixel as (orientation, color)."""
    h, w = g.shape
    visited: set[Tuple[int, int]] = set()
    out: Dict[Tuple[int, int], Tuple[str, int]] = {}

    for r in range(h):
        for c in range(w):
            if (r, c) in visited:
                continue
            color = int(g[r, c])
            if color == bg or color in {Color.RED, Color.PINK}:
                continue

            for mirror, (dr, dc) in {"\\": (1, 1), "/": (1, -1)}.items():
                nr, nc = r + dr, c + dc
                if not in_bounds((h, w), nr, nc):
                    continue
                if (nr, nc) in visited:
                    continue
                if int(g[nr, nc]) == color:
                    visited.add((r, c))
                    visited.add((nr, nc))
                    out[(r, c)] = (mirror, color)
                    out[(nr, nc)] = (mirror, color)
                    break

    return out


def _find_source_and_incoming(grid: np.ndarray, bg: int) -> Tuple[int, int, int, int]:
    """Return source and initial direction."""
    src_coords = np.argwhere(grid == Color.RED)
    assert src_coords.size == 2, "expected exactly one red source"
    sr, sc = map(int, src_coords[0])

    pink = np.argwhere(grid == Color.PINK)
    assert pink.size, "expected pink trajectory clues"

    pink_neighbors: List[Tuple[int, int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = sr + dr, sc + dc
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.PINK:
            pink_neighbors.append((nr, nc, dr, dc))
    assert pink_neighbors, "expected a pink cell adjacent to the source"
    assert len(pink_neighbors) == 1, "expected exactly one incoming pink direction"

    _, _, tdr, tdc = pink_neighbors[0]
    dr, dc = -tdr, -tdc
    assert (dr, dc) in _REFLECTION_RULES, "initial direction must be cardinal"
    return sr, sc, dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a color-changing beam from the red source, turning on diagonal mirrors.

    Input structure:
    - The background is the dominant color.
    - One red pixel marks the source, with an adjacent straight pink trail segment giving the incoming ray direction.
    - Mirrors are isolated two-pixel diagonal pairs of identical color, each forming either '/' or '\\'.

    Rule:
    - Set beam direction as opposite to the pink segment direction starting at the red source.
    - Move the beam cell by cell along straight lines, recoloring every visited background cell with the current beam color.
    - When the next step is a mirror cell, recolor to the mirror's color and reflect 90° according to the mirror orientation.
    - Continue until stepping outside the grid and leave the trail in the original grid untouched.

    Key insight:
    - The mirror is not a blocker: each diagonal pixel pair both defines the reflection law and carries the post-bounce beam color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    mirrors = _mirror_pairs(g, bg)

    r, c, dr, dc = _find_source_and_incoming(g, bg)
    color = Color.RED

    out = g.copy()
    seen_states = set()

    while True:
        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break

        state = (r, c, nr, nc, dr, dc, color)
        assert state not in seen_states, "beam path must be acyclic per task guarantee"
        seen_states.add(state)

        if (nr, nc) in mirrors:
            mirror, mcolor = mirrors[(nr, nc)]
            color = mcolor
            out[nr, nc] = color
            dr, dc = _REFLECTION_RULES[(dr, dc)][mirror]
        else:
            if out[nr, nc] == bg:
                out[nr, nc] = color

        r, c = nr, nc

    return out


def _trace_until_exit(
    h: int,
    w: int,
    source: Tuple[int, int],
    direction: Tuple[int, int],
    mirrors: Dict[Tuple[int, int], Tuple[str, int]],
) -> tuple[Tuple[int, int, int, int, int], ...]:
    """Trace a candidate layout and return a bounded state path."""
    sr, sc = source
    dr, dc = direction
    r, c = sr, sc
    color = Color.RED
    seen: set[Tuple[int, int, int, int]] = set()
    path: list[Tuple[int, int, int, int, int]] = []

    while True:
        nr, nc = r + dr, c + dc
        if not in_bounds((h, w), nr, nc):
            break

        state = (nr, nc, dr, dc, color)
        if state in seen:
            return tuple(path)
        seen.add(state)

        if (nr, nc) in mirrors:
            mirror, mcolor = mirrors[(nr, nc)]
            color = mcolor
            dr, dc = _REFLECTION_RULES[(dr, dc)][mirror]
            path.append((nr, nc, dr, dc, color))
        else:
            path.append((nr, nc, dr, dc, color))

        r, c = nr, nc
        if len(path) > h * w * 4:
            return tuple(path)

    return tuple(path)


def _mirror_candidates(
    h: int,
    w: int,
    occupied: set[Tuple[int, int]],
    mirror_color: int,
    orientation: str,
    pivot: Tuple[int, int],
) -> Dict[Tuple[int, int], tuple[str, int]] | None:
    pr, pc = pivot
    deltas = [(1, 1), (-1, -1)] if orientation == "\\" else [(1, -1), (-1, 1)]

    for dr, dc in deltas:
        sr, sc = pr + dr, pc + dc
        if not in_bounds((h, w), sr, sc):
            continue
        if (sr, sc) in occupied:
            continue
        return {(pr, pc): (orientation, mirror_color), (sr, sc): (orientation, mirror_color)}

    return None


def _distance_to_edge(h: int, w: int, pos: Tuple[int, int], direction: Tuple[int, int]) -> int:
    r, c = pos
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def generate_input() -> np.ndarray:
    """
    Generate random inputs matching the puzzle grammar.

    Randomly chooses grid size, source orientation, pink-segment length, mirror count,
    mirror colors, and reflection count branch (zero bounce / one bounce / repeated bounce).
    Mirror layouts are built so the beam is guaranteed to exit the grid.
    """
    bg = int(Color.BACKGROUND)
    color_pool = [c for c in Color.ALL_COLORS if c not in {bg, Color.RED, Color.PINK}]
    hit_targets = [0, 1, 1, 2, 2, 3]
    max_attempts = 400

    for _ in range(max_attempts):
        h = random.randint(14, 28)
        w = random.randint(14, 28)

        source_r = random.randint(3, h - 4)
        source_c = random.randint(3, w - 4)

        incoming = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        init_dir = (-incoming[0], -incoming[1])

        pink_len = random.randint(2, 5)
        pink: List[Tuple[int, int]] = []
        ok = True
        for k in range(1, pink_len + 1):
            nr, nc = source_r + incoming[0] * k, source_c + incoming[1] * k
            if not in_bounds((h, w), nr, nc):
                ok = False
                break
            pink.append((nr, nc))
        if not ok:
            continue

        occupied = {(source_r, source_c)} | set(pink)

        mirrors: Dict[Tuple[int, int], Tuple[str, int]] = {}
        dr, dc = init_dir
        r, c = source_r, source_c

        target_hits = random.choice(hit_targets)
        current_color = Color.RED
        success_hits = 0

        if target_hits:
            for _hit in range(target_hits):
                max_step = _distance_to_edge(h, w, (r, c), (dr, dc))
                if max_step <= 0:
                    break

                attempts_left = 40
                placed = False
                while attempts_left > 0 and not placed:
                    dist = random.randint(1, max_step)
                    pr, pc = r + dist * dr, c + dist * dc
                    if not in_bounds((h, w), pr, pc):
                        attempts_left -= 1
                        continue
                    if (pr, pc) in occupied:
                        attempts_left -= 1
                        continue

                    orientation = random.choice(("/", "\\"))
                    mirror_color = random.choice(color_pool)
                    candidate = _mirror_candidates(
                        h,
                        w,
                        occupied,
                        mirror_color,
                        orientation,
                        (pr, pc),
                    )
                    if candidate is None:
                        attempts_left -= 1
                        continue

                    mirrors.update(candidate)
                    occupied.update(candidate.keys())

                    # Advance to the mirror cell and reflect.
                    for _ in range(dist):
                        r += dr
                        c += dc
                    mirror, _mcolor = candidate[(pr, pc)]
                    current_color = mirror_color
                    dr, dc = _REFLECTION_RULES[(dr, dc)][mirror]
                    success_hits += 1
                    placed = True

                if not placed:
                    break

        if success_hits != target_hits:
            continue

        path = _trace_until_exit(h, w, (source_r, source_c), init_dir, mirrors)
        if not path:
            continue
        if len(path) > h * w * 4:
            continue

        # optional additional mirror off-path to test parser robustness
        extra_count = 0
        if random.random() < 0.25:
            extra_count = 1
        elif random.random() < 0.12:
            extra_count = 2

        extra_tries = 0
        while extra_count > 0 and extra_tries < 80:
            extra_tries += 1
            orientation = random.choice(("/", "\\"))
            mirror_color = random.choice(color_pool)
            candidate_cells = [(r0, c0) for r0 in range(h) for c0 in range(w) if (r0, c0) not in occupied]
            random.shuffle(candidate_cells)
            candidate = None
            for pivot in candidate_cells:
                mirror_cells = _mirror_candidates(h, w, occupied, mirror_color, orientation, pivot)
                if mirror_cells is None:
                    continue
                candidate = mirror_cells
                break
            if candidate is None:
                break

            mirrors.update(candidate)
            occupied.update(candidate.keys())
            extra_count -= 1

        grid = np.full((h, w), bg, dtype=int)
        grid[source_r, source_c] = Color.RED
        for rr, cc in pink:
            grid[rr, cc] = Color.PINK
        for (mr, mc), (_ori, mcol) in mirrors.items():
            grid[mr, mc] = mcol

        # ensure generated input can be transformed deterministically
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        if out.ndim != 2 or max(out.shape) > 30 or max(grid.shape) > 30:
            continue

        return grid

    raise RuntimeError("failed to generate a valid arc task input")
