from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
MIRROR = 4
CROSS = 6
TOWER_PALETTE = [1, 2, 3, 5, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _axis(dr: int, dc: int) -> int:
    return 0 if dc != 0 else 1


def _split_dirs(dr: int, dc: int) -> list[tuple[int, int]]:
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _simulate(grid: np.ndarray, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    tower_mask = (g != BG) & (g != MIRROR)
    towers = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(tower_mask)
    }
    assert towers, "expected at least one signal tower"
    assert all(color not in (BG, MIRROR, CROSS) for color in towers.values())

    out = g.copy()
    horiz = np.full((h, w), -1, dtype=int)
    vert = np.full((h, w), -1, dtype=int)
    seen_states: set[tuple[int, int, int, int, int]] = set()
    active: list[tuple[int, int, int, int, int]] = []

    for (r, c), color in sorted(towers.items()):
        for dr, dc in DIRS4:
            state = (int(r), int(c), int(dr), int(dc), int(color))
            seen_states.add(state)
            active.append(state)

    hit_mirrors: set[tuple[int, int]] = set()
    diff_crosses: set[tuple[int, int]] = set()
    same_crosses: set[tuple[int, int]] = set()

    while active:
        proposals: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = defaultdict(list)
        next_active: list[tuple[int, int, int, int, int]] = []

        for r, c, dr, dc, color in active:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in towers:
                continue
            if int(g[nr, nc]) == MIRROR:
                hit_mirrors.add((nr, nc))
                for ndr, ndc in _split_dirs(dr, dc):
                    state = (nr, nc, int(ndr), int(ndc), color)
                    if state in seen_states:
                        continue
                    seen_states.add(state)
                    next_active.append(state)
                continue
            proposals[(nr, nc)].append((nr, nc, dr, dc, color))

        for (r, c), arrivals in sorted(proposals.items()):
            by_axis = {0: [], 1: []}
            before = {
                0: int(horiz[r, c]),
                1: int(vert[r, c]),
            }
            for state in arrivals:
                by_axis[_axis(state[2], state[3])].append(state)

            admitted: list[tuple[int, int, int, int, int]] = []
            for axis in (0, 1):
                if before[axis] != -1 or not by_axis[axis]:
                    continue
                colors = {int(state[4]) for state in by_axis[axis]}
                assert len(colors) == 1, "ambiguous same-axis collision"
                color = next(iter(colors))
                if axis == 0:
                    horiz[r, c] = int(color)
                else:
                    vert[r, c] = int(color)
                admitted.extend(by_axis[axis])

            h_color = int(horiz[r, c])
            v_color = int(vert[r, c])
            if h_color != -1 and v_color != -1:
                if h_color == v_color:
                    out[r, c] = int(h_color)
                    same_crosses.add((r, c))
                else:
                    out[r, c] = CROSS
                    diff_crosses.add((r, c))
            elif h_color != -1:
                out[r, c] = int(h_color)
            elif v_color != -1:
                out[r, c] = int(v_color)

            for _nr, _nc, dr, dc, color in admitted:
                state = (r, c, int(dr), int(dc), int(color))
                if state in seen_states:
                    continue
                seen_states.add(state)
                next_active.append(state)

        active = next_active

    if return_stats:
        return out, {
            "hit_mirrors": hit_mirrors,
            "diff_crosses": diff_crosses,
            "same_crosses": same_crosses,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four-direction rays from every non-yellow tower pixel.

    Rays stop at the boundary or before another tower. When a ray reaches a yellow
    mirror, the forward beam ends and two perpendicular beams continue from the
    mirror. A blank cell occupied by both a horizontal and a vertical ray becomes
    magenta when the colors differ; same-color crossings keep that color.
    """
    return _simulate(grid)


def _normalize(points: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    min_r = min(r for r, _ in points.values())
    min_c = min(c for _, c in points.values())
    return {
        name: (int(r - min_r), int(c - min_c))
        for name, (r, c) in points.items()
    }


def _pattern_mode_a(include_c: bool) -> dict[str, tuple[int, int]]:
    d1 = random.randint(3, 5)
    du = random.randint(2, 4)
    db = random.randint(2, 4)
    points = {
        "A": (0, 0),
        "M1": (0, d1),
        "B": (-du, d1 + db),
    }
    if include_c:
        dd = random.randint(2, 4)
        dc = random.randint(2, 4)
        points["C"] = (dd, d1 + dc)
    return _normalize(points)


def _pattern_mode_b(include_c: bool) -> dict[str, tuple[int, int]]:
    d1 = random.randint(4, 6)
    du2 = random.randint(2, 4)
    x = random.randint(1, d1 - 1)
    dbu = random.randint(2, 4)
    points = {
        "A": (0, 0),
        "M1": (0, d1),
        "M2": (-du2, d1),
        "B": (-du2 - dbu, x),
    }
    if include_c:
        dd = random.randint(2, 4)
        dc = random.randint(2, 4)
        points["C"] = (dd, d1 + dc)
    return _normalize(points)


def _build_patch(mode: str, include_c: bool) -> tuple[np.ndarray, bool]:
    if mode == "a":
        points = _pattern_mode_a(include_c)
        expect_same = False
    else:
        points = _pattern_mode_b(include_c)
        expect_same = True

    tower_names = [name for name in ("A", "B", "C") if name in points]
    colors = random.sample(TOWER_PALETTE, len(tower_names))
    color_map = {name: color for name, color in zip(tower_names, colors)}

    max_r = max(r for r, _ in points.values())
    max_c = max(c for _, c in points.values())
    patch = np.full((max_r + 1, max_c + 1), BG, dtype=int)

    for name, (r, c) in points.items():
        if name.startswith("M"):
            patch[r, c] = MIRROR
        else:
            patch[r, c] = int(color_map[name])

    return patch, expect_same


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k)
    return np.asarray(out, dtype=int).copy()


def _embed_patch(patch: np.ndarray) -> np.ndarray:
    top = random.randint(2, 6)
    bottom = random.randint(2, 6)
    left = random.randint(2, 6)
    right = random.randint(2, 6)
    h, w = patch.shape
    grid = np.full((h + top + bottom, w + left + right), BG, dtype=int)
    grid[top : top + h, left : left + w] = patch
    return grid


def _valid_generated(grid: np.ndarray, expect_same: bool) -> bool:
    if np.any(grid == CROSS):
        return False
    try:
        out, stats = _simulate(grid, return_stats=True)
    except AssertionError:
        return False
    mirror_count = int(np.count_nonzero(grid == MIRROR))
    if len(stats["hit_mirrors"]) != mirror_count:
        return False
    if len(stats["diff_crosses"]) < 1:
        return False
    if expect_same and len(stats["same_crosses"]) < 1:
        return False
    if np.count_nonzero(out == CROSS) < 1:
        return False
    density = float(np.count_nonzero(out != BG)) / float(out.size)
    if density > 0.65:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["a", "b"])
    include_c = random.random() < 0.5
    patch, expect_same = _build_patch(mode, include_c)
    grid = _embed_patch(_random_symmetry(patch))
    if not _valid_generated(grid, expect_same):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Build sparse tower-and-mirror layouts with guaranteed split rays and crossings.
    """
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle")
