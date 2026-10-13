from __future__ import annotations

import random
from typing import Iterable

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

MOVABLE_COLORS = (BLUE, RED, YELLOW, PINK, ORANGE, TEAL, MAROON)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _movable_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != GREY)


def _colored_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components(_movable_mask(grid))


def _settle_one(occupancy: np.ndarray, r: int, c: int) -> tuple[int, int, list[str]]:
    """
    Simulate one colored pixel falling through black cells.

    The caller provides an occupancy grid containing only grey obstacles and
    already-settled colored pixels.
    """
    h, w = occupancy.shape
    rr = int(r)
    cc = int(c)
    branches: list[str] = []
    for _ in range(h * w):
        if rr + 1 < h and occupancy[rr + 1, cc] == BLACK:
            rr += 1
            continue

        left_open = rr + 1 < h and cc - 1 >= 0 and occupancy[rr + 1, cc - 1] == BLACK
        right_open = rr + 1 < h and cc + 1 < w and occupancy[rr + 1, cc + 1] == BLACK

        if left_open and right_open:
            branches.append("both_left")
            rr += 1
            cc -= 1
            continue
        if left_open:
            branches.append("left_only")
            rr += 1
            cc -= 1
            continue
        if right_open:
            branches.append("right_only")
            rr += 1
            cc += 1
            continue
        break
    else:
        raise AssertionError("pixel did not settle")
    return rr, cc, branches


def _simulate_gravity(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources = sorted(
        [
            (int(r), int(c), int(g[r, c]))
            for r, c in np.argwhere(_movable_mask(g))
        ]
    )
    occupancy = g.copy()
    occupancy[_movable_mask(g)] = BLACK

    traces: list[dict] = []
    for r, c, color in sources:
        rr, cc, branches = _settle_one(occupancy, r, c)
        assert occupancy[rr, cc] == BLACK, "two pixels cannot settle into one cell"
        occupancy[rr, cc] = color
        traces.append(
            {
                "start": (r, c),
                "end": (rr, cc),
                "color": color,
                "branches": tuple(branches),
                "moved": (rr, cc) != (r, c),
            }
        )

    if return_trace:
        return occupancy, traces
    return occupancy


def _paint_domino_border(out: np.ndarray, settled: np.ndarray, coords: np.ndarray) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = _bbox(coords)
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                if settled[r, c] == BLACK:
                    out[r, c] = GREEN


def _apply_size_rules(settled: np.ndarray) -> np.ndarray:
    out = np.asarray(settled, dtype=int).copy()
    comps = _colored_components(settled)

    for coords in comps:
        if len(coords) >= 3:
            out[coords[:, 0], coords[:, 1]] = BLACK

    for coords in comps:
        if len(coords) == 2:
            _paint_domino_border(out, settled, coords)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Apply gravity to colored pixels, then keep singles, frame pairs, and erase larger clusters.

    Input structure:
    - The grid uses black as background and grey as static terrain/obstacles.
    - Non-black, non-grey pixels are movable colored particles initially placed in the upper area.
    - Grey cells stay fixed and colored particles are processed in reading order.

    Rule:
    - Remove the movable pixels from the occupancy map, then settle them one by one from top to bottom and left to right.
    - A particle falls straight down through black cells; when blocked below, it may slide to left-down or right-down if that diagonal cell is black, choosing left when both diagonals are open.
    - After all particles settle, compute 4-connected components over all colored pixels only.
    - Area-1 components stay unchanged, area-2 components stay and receive a green perimeter on the bbox expanded by one cell, and area>=3 components are erased to black.

    Key insight:
    - The final component sizes are determined only after the gravity simulation, so the obstacle-guided landing pattern decides which colors survive, which get highlighted, and which disappear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _apply_size_rules(_simulate_gravity(g))


GEN_HEIGHT_RANGE = (13, 18)
GEN_WIDTH_RANGE = (14, 18)
GEN_ATTEMPTS = 400


def _distribute_extra(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _build_surface(h: int, w: int, domino_vertical: bool) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build a connected lower-half grey terrain using a skyline profile.

    Three separated catchments are reserved for the size-2, size>=3, and size-1
    outcomes after gravity.
    """
    deep = h - 1
    mid = max(h // 2, h - random.choice((4, 5)))
    high = max(h // 2, mid - 1)
    surface = np.full(w, mid, dtype=int)

    extra = w - 14
    add_gap1, add_gap2, add_tail = _distribute_extra(extra, 3)
    gap1 = 1 + add_gap1
    gap2 = 1 + add_gap2
    tail = 1 + add_tail

    domino_start = 0
    cluster_start = 4 + gap1
    singleton_start = cluster_start + 5 + gap2
    assert singleton_start + 2 + tail == w

    if domino_vertical:
        surface[domino_start : domino_start + 4] = (mid, deep, mid, mid)
    else:
        surface[domino_start : domino_start + 4] = (mid, deep, deep, mid)

    surface[cluster_start : cluster_start + 5] = (mid, deep, mid, high, deep)
    surface[singleton_start : singleton_start + 2] = (deep, mid)

    safe_fill = set(range(w))
    critical = {
        domino_start,
        domino_start + 1,
        domino_start + 2,
        domino_start + 3,
        cluster_start - 1,
        cluster_start,
        cluster_start + 1,
        cluster_start + 2,
        cluster_start + 3,
        cluster_start + 4,
        singleton_start,
        singleton_start + 1,
    }
    safe_fill -= {c for c in critical if 0 <= c < w}
    for c in random.sample(sorted(safe_fill), k=min(len(safe_fill), random.randint(0, 2))):
        surface[c] = min(deep - 1, mid + 1)

    meta = {
        "domino_start": domino_start,
        "cluster_start": cluster_start,
        "singleton_start": singleton_start,
        "mid": mid,
        "deep": deep,
        "high": high,
        "domino_vertical": int(domino_vertical),
    }
    return surface, meta


def _paint_terrain(surface: np.ndarray, h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for c, top in enumerate(surface):
        grid[int(top) :, int(c)] = GREY
    return grid


def _assign_rows(columns: list[int], top_max: int) -> dict[int, int] | None:
    for _ in range(120):
        rows: dict[int, int] = {}
        for c in sorted(columns):
            options = list(range(top_max + 1))
            random.shuffle(options)
            picked = None
            for r in options:
                if all(abs(r - rr) + abs(c - cc) != 1 for cc, rr in rows.items()):
                    picked = r
                    break
            if picked is None:
                break
            rows[c] = picked
        if len(rows) == len(columns):
            return rows
    return None


def _classify_components_by_size(grid: np.ndarray) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for comp in _colored_components(grid):
        result[len(comp)] = comp
    return result


def _input_rule_only(grid: np.ndarray) -> np.ndarray:
    return _apply_size_rules(np.asarray(grid, dtype=int))


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    source_coords = np.argwhere(_movable_mask(g))
    if len(source_coords) not in (6, 7):
        return False
    source_colors = [int(g[r, c]) for r, c in source_coords]
    if len(set(source_colors)) != len(source_colors):
        return False
    if max(int(r) for r, _ in source_coords) >= h // 2:
        return False
    if any(len(comp) != 1 for comp in _colored_components(g)):
        return False

    grey_comps = _components(g == GREY)
    if len(grey_comps) != 1:
        return False

    settled, traces = _simulate_gravity(g, return_trace=True)
    branch_types = {branch for trace in traces for branch in trace["branches"]}
    if not {"left_only", "right_only", "both_left"} <= branch_types:
        return False
    if sum(1 for trace in traces if trace["moved"]) < 4:
        return False

    comps_by_size = _classify_components_by_size(settled)
    big_size = len(source_coords) - 3
    if sorted(comps_by_size) != [1, 2, big_size]:
        return False

    out = main(g)
    if not np.array_equal(out, _apply_size_rules(settled)):
        return False
    if not np.array_equal(out[g == GREY], np.full(np.sum(g == GREY), GREY, dtype=int)):
        return False

    single = comps_by_size[1][0]
    if out[int(single[0]), int(single[1])] != settled[int(single[0]), int(single[1])]:
        return False

    domino = comps_by_size[2]
    for r, c in domino:
        if out[int(r), int(c)] != settled[int(r), int(c)]:
            return False

    big = comps_by_size[big_size]
    for r, c in big:
        if out[int(r), int(c)] != BLACK:
            return False

    r0, c0, r1, c1 = _bbox(domino)
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    green_count = 0
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                if settled[r, c] == BLACK:
                    if out[r, c] != GREEN:
                        return False
                    green_count += 1
                elif out[r, c] != settled[r, c]:
                    return False
    if green_count == 0:
        return False

    if np.array_equal(out, _input_rule_only(g)):
        return False
    if np.array_equal(settled, g):
        return False
    return True


def _build_candidate() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    domino_vertical = bool(random.getrandbits(1))
    cluster_extra = bool(random.getrandbits(1))

    surface, meta = _build_surface(h, w, domino_vertical)
    grid = _paint_terrain(surface, h, w)

    d = meta["domino_start"]
    c = meta["cluster_start"]
    s = meta["singleton_start"]

    source_columns: list[int] = []
    if domino_vertical:
        source_columns.extend([d, d + 1])
    else:
        source_columns.extend([d, d + 2])
    source_columns.extend([c + 1, c + 2, c + 3, s + 1])
    if cluster_extra:
        source_columns.append(c)

    top_max = max(2, h // 2 - 2)
    rows = _assign_rows(source_columns, top_max)
    if rows is None:
        return None

    colors = random.sample(MOVABLE_COLORS, len(source_columns))
    placements: list[tuple[int, int]] = []
    for col in source_columns:
        placements.append((rows[col], col))
    random.shuffle(colors)
    for (r, c0), color in zip(placements, colors):
        grid[int(r), int(c0)] = int(color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid gravity-area ARC input")
