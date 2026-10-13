from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
RECT_COUNT_RANGE = (2, 5)
INNER_H_RANGE = (1, 4)
INNER_W_RANGE = (1, 5)
MAX_FOREIGN_RAY_HITS = 4


@dataclass(frozen=True)
class RectInfo:
    r0: int
    c0: int
    r1: int
    c1: int
    gap_r: int
    gap_c: int
    gap_color: int
    key_r: int
    key_c: int
    key_color: int
    dr: int
    dc: int
    area: int

    @property
    def matched(self) -> bool:
        return self.key_color == self.gap_color


@dataclass(frozen=True)
class RectPlan:
    inner_h: int
    inner_w: int
    side: str
    gap_offset: int
    key_color: int
    gap_color: int
    ray_mode: str

    @property
    def outer_h(self) -> int:
        return self.inner_h + 2

    @property
    def outer_w(self) -> int:
        return self.inner_w + 2

    @property
    def area(self) -> int:
        return self.inner_h * self.inner_w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _direction_from_gap(r0: int, c0: int, r1: int, c1: int, gap_r: int, gap_c: int) -> tuple[int, int]:
    edge_hits = int(gap_r == r0) + int(gap_r == r1) + int(gap_c == c0) + int(gap_c == c1)
    assert edge_hits == 1, "gap must lie on exactly one side of the frame"
    if gap_r == r0:
        return -1, 0
    if gap_r == r1:
        return 1, 0
    if gap_c == c0:
        return 0, -1
    return 0, 1


def _extract_rectangles(grid: np.ndarray) -> list[RectInfo]:
    g = np.asarray(grid, dtype=int)
    grey_components = _components(g == GREY)
    rects: list[RectInfo] = []

    for comp in grey_components:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        outer_h = r1 - r0 + 1
        outer_w = c1 - c0 + 1
        if outer_h < 3 or outer_w < 3:
            continue

        local = np.zeros((outer_h, outer_w), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True

        perimeter = np.zeros_like(local, dtype=bool)
        perimeter[0, :] = True
        perimeter[-1, :] = True
        perimeter[:, 0] = True
        perimeter[:, -1] = True

        if np.any(local & ~perimeter):
            continue
        if np.any(local[1:-1, 1:-1]):
            continue

        missing = np.argwhere(perimeter & ~local)
        if missing.shape != (1, 2):
            continue
        gap_r = int(r0 + missing[0, 0])
        gap_c = int(c0 + missing[0, 1])
        gap_color = int(g[gap_r, gap_c])
        if gap_color in (BLACK, GREY):
            continue

        dr, dc = _direction_from_gap(r0, c0, r1, c1, gap_r, gap_c)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        if np.any(inner == GREY):
            continue
        key_cells = np.argwhere(inner != BLACK)
        if key_cells.shape != (1, 2):
            continue
        key_r = int(r0 + 1 + key_cells[0, 0])
        key_c = int(c0 + 1 + key_cells[0, 1])
        key_color = int(g[key_r, key_c])
        if key_color in (BLACK, GREY):
            continue

        rects.append(
            RectInfo(
                r0=r0,
                c0=c0,
                r1=r1,
                c1=c1,
                gap_r=gap_r,
                gap_c=gap_c,
                gap_color=gap_color,
                key_r=key_r,
                key_c=key_c,
                key_color=key_color,
                dr=dr,
                dc=dc,
                area=(outer_h - 2) * (outer_w - 2),
            )
        )

    assert rects, "expected at least one keyed frame"
    assert len(rects) == len(grey_components), "every grey component must be a valid frame"
    rects.sort(key=lambda rect: (rect.r0, rect.c0, rect.r1, rect.c1))
    return rects


def _ray_cells(rect: RectInfo, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for step in range(1, rect.area + 1):
        rr = rect.gap_r + rect.dr * step
        cc = rect.gap_c + rect.dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock each grey rectangular frame whose key color matches its border lock color,
    then cast an outward ray whose visible length equals the frame's inner area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    h, w = g.shape
    for rect in _extract_rectangles(g):
        if not rect.matched:
            continue
        for rr, cc in _ray_cells(rect, h, w):
            out[rr, cc] = rect.key_color
    return out


def _sample_match_flags(rng: random.Random, n_rects: int) -> list[bool]:
    true_count = n_rects // 2 if rng.random() < 0.5 else (n_rects + 1) // 2
    flags = [True] * true_count + [False] * (n_rects - true_count)
    rng.shuffle(flags)
    return flags


def _sample_plans(rng: random.Random) -> list[RectPlan]:
    n_rects = rng.randint(*RECT_COUNT_RANGE)
    key_colors = rng.sample(PALETTE, n_rects)
    match_flags = _sample_match_flags(rng, n_rects)
    plans: list[RectPlan] = []

    for key_color, matched in zip(key_colors, match_flags):
        inner_h = rng.randint(*INNER_H_RANGE)
        inner_w = rng.randint(*INNER_W_RANGE)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        side = rng.choice(SIDES)
        gap_offset = rng.randint(1, (outer_w - 2) if side in ("top", "bottom") else (outer_h - 2))
        if matched:
            gap_color = key_color
            ray_mode = "full" if inner_h * inner_w == 1 or rng.random() < 0.65 else "truncated"
        else:
            gap_color = rng.choice([color for color in PALETTE if color != key_color])
            ray_mode = "locked"
        plans.append(
            RectPlan(
                inner_h=inner_h,
                inner_w=inner_w,
                side=side,
                gap_offset=gap_offset,
                key_color=key_color,
                gap_color=gap_color,
                ray_mode=ray_mode,
            )
        )
    return plans


def _margin_to_edge(r0: int, c0: int, outer_h: int, outer_w: int, side: str, h: int, w: int) -> int:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    if side == "top":
        return r0
    if side == "bottom":
        return h - 1 - r1
    if side == "left":
        return c0
    return w - 1 - c1


def _difficulty(plan: RectPlan) -> tuple[int, int, int]:
    mode_rank = {"full": 2, "truncated": 1, "locked": 0}[plan.ray_mode]
    return mode_rank, plan.area, max(plan.outer_h, plan.outer_w)


def _place_plan(
    rng: random.Random,
    plan: RectPlan,
    blocked: np.ndarray,
) -> RectInfo | None:
    h, w = blocked.shape
    if h < plan.outer_h + 2 or w < plan.outer_w + 2:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(1, h - plan.outer_h - 1)
        c0 = rng.randint(1, w - plan.outer_w - 1)
        r1 = r0 + plan.outer_h - 1
        c1 = c0 + plan.outer_w - 1
        if blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue

        margin = _margin_to_edge(r0, c0, plan.outer_h, plan.outer_w, plan.side, h, w)
        if plan.ray_mode == "full" and margin < plan.area:
            continue
        if plan.ray_mode == "truncated" and not (1 <= margin < plan.area):
            continue

        if plan.side == "top":
            gap_r, gap_c = r0, c0 + plan.gap_offset
        elif plan.side == "bottom":
            gap_r, gap_c = r1, c0 + plan.gap_offset
        elif plan.side == "left":
            gap_r, gap_c = r0 + plan.gap_offset, c0
        else:
            gap_r, gap_c = r0 + plan.gap_offset, c1

        key_r = r0 + 1 + rng.randrange(plan.inner_h)
        key_c = c0 + 1 + rng.randrange(plan.inner_w)
        dr, dc = _direction_from_gap(r0, c0, r1, c1, gap_r, gap_c)

        blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
        return RectInfo(
            r0=r0,
            c0=c0,
            r1=r1,
            c1=c1,
            gap_r=gap_r,
            gap_c=gap_c,
            gap_color=plan.gap_color,
            key_r=key_r,
            key_c=key_c,
            key_color=plan.key_color,
            dr=dr,
            dc=dc,
            area=plan.area,
        )
    return None


def _render_input(h: int, w: int, rects: list[RectInfo]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for rect in rects:
        grid[rect.r0, rect.c0 : rect.c1 + 1] = GREY
        grid[rect.r1, rect.c0 : rect.c1 + 1] = GREY
        grid[rect.r0 : rect.r1 + 1, rect.c0] = GREY
        grid[rect.r0 : rect.r1 + 1, rect.c1] = GREY
        grid[rect.gap_r, rect.gap_c] = rect.gap_color
        grid[rect.key_r, rect.key_c] = rect.key_color
    return grid


def _expected_output(grid: np.ndarray, rects: list[RectInfo]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for rect in rects:
        if not rect.matched:
            continue
        for rr, cc in _ray_cells(rect, h, w):
            out[rr, cc] = rect.key_color
    return out


def _valid_ray_layout(grid: np.ndarray, rects: list[RectInfo]) -> bool:
    ray_counts: dict[tuple[int, int], int] = {}
    foreign_hits = 0
    h, w = grid.shape

    for rect in rects:
        if not rect.matched:
            continue
        for cell in _ray_cells(rect, h, w):
            ray_counts[cell] = ray_counts.get(cell, 0) + 1
            if ray_counts[cell] > 1:
                return False
            if grid[cell] != BLACK:
                foreign_hits += 1

    return foreign_hits <= MAX_FOREIGN_RAY_HITS


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    blocked = np.zeros((h, w), dtype=bool)

    plans = _sample_plans(rng)
    plans.sort(key=_difficulty, reverse=True)

    rects: list[RectInfo] = []
    for plan in plans:
        rect = _place_plan(rng, plan, blocked)
        if rect is None:
            return None
        rects.append(rect)

    grid = _render_input(h, w, rects)
    if not _valid_ray_layout(grid, rects):
        return None

    expected = _expected_output(grid, rects)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
