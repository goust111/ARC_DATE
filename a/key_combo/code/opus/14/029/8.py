from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


BG = 0
BAR = 1
MOVING = 3

MOVE_AWAY = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

SPLASH_TOWARD = {
    "top": ((-1, -1), (-1, 1)),
    "bottom": ((1, -1), (1, 1)),
    "left": ((-1, -1), (1, -1)),
    "right": ((-1, 1), (1, 1)),
}


@dataclass(frozen=True)
class PixelPlan:
    src: tuple[int, int]
    edge: str
    moved: tuple[int, int]
    trigger: bool
    footprint: frozenset[tuple[int, int]]


def _find_bar_row(grid: np.ndarray) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r] == BAR)
    ]
    assert len(rows) == 1, "expected exactly one full-width blue bar"
    return int(rows[0])


def _nearest_edge(r: int, c: int, h: int, w: int) -> str:
    dists = {
        "top": r,
        "bottom": h - 1 - r,
        "left": c,
        "right": w - 1 - c,
    }
    min_dist = min(dists.values())
    winners = [name for name, dist in dists.items() if dist == min_dist]
    assert len(winners) == 1, "nearest edge must be unique"
    return winners[0]


def _is_adjacent_to_bar(r: int, c: int, bar_r: int) -> bool:
    return abs(r - bar_r) == 1


def _paint_ray_cells(
    h: int,
    w: int,
    bar_r: int,
    start: tuple[int, int],
    edge: str,
) -> set[tuple[int, int]]:
    cells = {start}
    for dr, dc in SPLASH_TOWARD[edge]:
        rr = start[0] + dr
        cc = start[1] + dc
        while 0 <= rr < h and 0 <= cc < w:
            if rr != bar_r:
                cells.add((rr, cc))
            rr += dr
            cc += dc
    return cells


def _plan_pixel(
    h: int,
    w: int,
    bar_r: int,
    src: tuple[int, int],
) -> PixelPlan:
    r, c = src
    edge = _nearest_edge(r, c, h, w)
    dr, dc = MOVE_AWAY[edge]
    moved = (r + dr, c + dc)
    assert 0 <= moved[0] < h and 0 <= moved[1] < w
    assert moved[0] != bar_r
    trigger = _is_adjacent_to_bar(moved[0], moved[1], bar_r)
    footprint = _paint_ray_cells(h, w, bar_r, moved, edge) if trigger else {moved}
    return PixelPlan(src=src, edge=edge, moved=moved, trigger=trigger, footprint=frozenset(footprint))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each green pixel one step away from its unique nearest grid edge, then splash diagonals back toward that edge if the move makes it touch the blue bar.

    Input structure:
    A black background contains exactly one full-width horizontal blue bar and several isolated green single pixels above and below it. Every green pixel has a unique nearest grid edge among top, bottom, left, and right, and no green pixel starts 4-neighbor-adjacent to the bar.

    Rule:
    For each green pixel, find its unique nearest grid edge by raw row/column distance. Move the pixel one step away from that edge, erasing the original cell. If the moved pixel is then 4-neighbor-adjacent to the blue bar, emit two green 45-degree rays from the moved pixel in the opposite direction, namely back toward the nearest edge: top gives up-left and up-right, bottom gives down-left and down-right, left gives up-left and down-left, and right gives up-right and down-right. Keep the blue bar unchanged and paint every non-bar ray cell until each ray leaves the grid.

    Key insight:
    The nearest edge does not directly draw anything; it first selects a per-pixel movement direction, and only the post-move adjacency to the bar decides whether a backward diagonal splash appears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert set(np.unique(g)).issubset({BG, BAR, MOVING})

    h, w = g.shape
    bar_r = _find_bar_row(g)
    sources = [tuple(map(int, rc)) for rc in np.argwhere(g == MOVING)]
    assert sources, "expected at least one moving pixel"

    out = g.copy()
    out[g == MOVING] = BG

    plans = []
    moved_seen: set[tuple[int, int]] = set()
    for src in sources:
        assert not _is_adjacent_to_bar(src[0], src[1], bar_r), "sources must not start adjacent to the bar"
        plan = _plan_pixel(h, w, bar_r, src)
        assert plan.moved not in moved_seen, "moved pixels must remain distinct"
        moved_seen.add(plan.moved)
        plans.append(plan)

    for plan in plans:
        for r, c in plan.footprint:
            if r == bar_r:
                continue
            out[r, c] = MOVING

    out[bar_r, :] = BAR
    return out


def _always_down_reference(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bar_r = _find_bar_row(g)
    out = g.copy()
    out[g == MOVING] = BG
    for r, c in np.argwhere(g == MOVING):
        nr = int(r) + 1
        if nr >= h or nr == bar_r:
            continue
        out[nr, int(c)] = MOVING
        if nr == bar_r - 1:
            for dc in (-1, 1):
                rr = nr - 1
                cc = int(c) + dc
                while 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = MOVING
                    rr -= 1
                    cc += dc
    out[bar_r, :] = BAR
    return out


def _fixed_up_splash_reference(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bar_r = _find_bar_row(g)
    out = g.copy()
    out[g == MOVING] = BG
    for r, c in np.argwhere(g == MOVING):
        rr = int(r)
        cc = int(c)
        edge = _nearest_edge(rr, cc, h, w)
        dr, dc = MOVE_AWAY[edge]
        nr, nc = rr + dr, cc + dc
        if nr == bar_r:
            continue
        out[nr, nc] = MOVING
        if _is_adjacent_to_bar(nr, nc, bar_r):
            for diag_dc in (-1, 1):
                ray_r = nr - 1
                ray_c = nc + diag_dc
                while 0 <= ray_r < h and 0 <= ray_c < w:
                    out[ray_r, ray_c] = MOVING
                    ray_r -= 1
                    ray_c += diag_dc
    out[bar_r, :] = BAR
    return out


def _all_candidates(h: int, w: int, bar_r: int) -> list[PixelPlan]:
    plans: list[PixelPlan] = []
    for r in range(1, h - 1):
        if r == bar_r or abs(r - bar_r) == 1:
            continue
        for c in range(1, w - 1):
            try:
                plan = _plan_pixel(h, w, bar_r, (r, c))
            except AssertionError:
                continue
            plans.append(plan)
    return plans


def _role_filters() -> dict[str, callable]:
    return {
        "top_trigger": lambda p: p.edge == "top" and p.trigger,
        "bottom_trigger": lambda p: p.edge == "bottom" and p.trigger,
        "top_nontrigger": lambda p: p.edge == "top" and not p.trigger,
        "bottom_nontrigger": lambda p: p.edge == "bottom" and not p.trigger,
        "left_nontrigger": lambda p: p.edge == "left" and not p.trigger,
        "right_nontrigger": lambda p: p.edge == "right" and not p.trigger,
    }


def _isolated(src: tuple[int, int], chosen: Iterable[PixelPlan]) -> bool:
    return all(
        max(abs(src[0] - other.src[0]), abs(src[1] - other.src[1])) > 1
        for other in chosen
    )


def _compatible(plan: PixelPlan, chosen: list[PixelPlan], used: set[tuple[int, int]]) -> bool:
    if plan.moved in {other.moved for other in chosen}:
        return False
    if not _isolated(plan.src, chosen):
        return False
    return plan.footprint.isdisjoint(used)


def _select_plans(
    rng: np.random.Generator,
    buckets: dict[str, list[PixelPlan]],
    roles: list[str],
) -> list[PixelPlan] | None:
    ordered_roles = sorted(roles, key=lambda role: len(buckets[role]))
    chosen: list[PixelPlan] = []
    used: set[tuple[int, int]] = set()

    def backtrack(i: int) -> bool:
        if i == len(ordered_roles):
            return True
        role = ordered_roles[i]
        candidates = buckets[role][:]
        rng.shuffle(candidates)
        for plan in candidates:
            if not _compatible(plan, chosen, used):
                continue
            chosen.append(plan)
            used.update(plan.footprint)
            if backtrack(i + 1):
                return True
            chosen.pop()
            used.difference_update(plan.footprint)
        return False

    if not backtrack(0):
        return None

    extras = [p for plans in buckets.values() for p in plans]
    rng.shuffle(extras)
    for plan in extras:
        if any(plan.src == other.src for other in chosen):
            continue
        if len(chosen) >= 7:
            break
        if _compatible(plan, chosen, used) and rng.random() < 0.35:
            chosen.append(plan)
            used.update(plan.footprint)
    return chosen


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    h, w = grid.shape
    bar_r = _find_bar_row(grid)
    plans = [_plan_pixel(h, w, bar_r, tuple(map(int, rc))) for rc in np.argwhere(grid == MOVING)]
    if len(plans) < 5:
        return False

    edges = {plan.edge for plan in plans}
    if edges != {"top", "bottom", "left", "right"}:
        return False

    if not any(plan.edge == "top" and plan.trigger for plan in plans):
        return False
    if not any(plan.edge == "bottom" and plan.trigger for plan in plans):
        return False
    if not any(plan.edge in {"top", "bottom"} and not plan.trigger for plan in plans):
        return False
    if not any(plan.edge in {"left", "right"} and not plan.trigger for plan in plans):
        return False

    used: set[tuple[int, int]] = set()
    for plan in plans:
        if _is_adjacent_to_bar(plan.src[0], plan.src[1], bar_r):
            return False
        if not plan.footprint.isdisjoint(used):
            return False
        used.update(plan.footprint)

    if np.array_equal(grid, out):
        return False
    if np.array_equal(out, _always_down_reference(grid)):
        return False
    if np.array_equal(out, _fixed_up_splash_reference(grid)):
        return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(11, 17))
    w = int(rng.integers(11, 17))
    bar_r = int(rng.integers(4, h - 4))

    all_plans = _all_candidates(h, w, bar_r)
    filters = _role_filters()
    buckets = {
        role: [plan for plan in all_plans if pred(plan)]
        for role, pred in filters.items()
    }
    if any(not plans for plans in buckets.values()):
        return None

    vertical_nontrigger = "top_nontrigger" if rng.random() < 0.5 else "bottom_nontrigger"
    roles = ["top_trigger", "bottom_trigger", vertical_nontrigger, "left_nontrigger", "right_nontrigger"]
    chosen = _select_plans(rng, buckets, roles)
    if chosen is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[bar_r, :] = BAR
    for plan in chosen:
        grid[plan.src] = MOVING

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
