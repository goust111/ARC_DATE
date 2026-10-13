from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300


def _dilate_1d(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    return out


def _grey_components(grid: np.ndarray) -> list[np.ndarray]:
    mask = grid == GREY
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_bars(grid: np.ndarray, light: tuple[int, int]) -> dict[str, dict[int, list[dict]]]:
    lr, lc = light
    bars_by_side: dict[str, dict[int, list[dict]]] = {
        "up": {},
        "down": {},
        "left": {},
        "right": {},
    }

    for coords in _grey_components(grid):
        rows = coords[:, 0]
        cols = coords[:, 1]
        row_vals = sorted(set(map(int, rows)))
        col_vals = sorted(set(map(int, cols)))

        if len(row_vals) == 1:
            row = row_vals[0]
            c0 = int(cols.min())
            c1 = int(cols.max())
            if row < lr:
                side = "up"
            elif row > lr:
                side = "down"
            else:
                raise AssertionError("horizontal bar cannot share the light row")
            bars_by_side[side].setdefault(row, []).append(
                {"side": side, "row": row, "start": c0, "end": c1}
            )
        elif len(col_vals) == 1:
            col = col_vals[0]
            r0 = int(rows.min())
            r1 = int(rows.max())
            if col < lc:
                side = "left"
            elif col > lc:
                side = "right"
            else:
                raise AssertionError("vertical bar cannot share the light column")
            bars_by_side[side].setdefault(col, []).append(
                {"side": side, "col": col, "start": r0, "end": r1}
            )
        else:
            raise AssertionError("each grey baffle must be a straight 1-cell-thick bar")

    return bars_by_side


def _paint_row_side(
    grid: np.ndarray,
    out: np.ndarray,
    rows: range,
    bars_by_row: dict[int, list[dict]],
) -> list[dict]:
    width = grid.shape[1]
    shadow_prev = np.zeros(width, dtype=bool)
    seed_prev = np.zeros(width, dtype=bool)
    statuses: list[dict] = []

    for row in rows:
        cand = _dilate_1d(shadow_prev) | seed_prev
        shadow_curr = cand & (grid[row, :] == BLACK)
        seed_curr = np.zeros(width, dtype=bool)

        for bar in bars_by_row.get(row, []):
            seg = slice(bar["start"], bar["end"] + 1)
            hit = bool(np.any(cand[seg]))
            status = dict(bar)
            status["disabled"] = hit
            statuses.append(status)
            if hit:
                out[row, seg] = BLUE
                shadow_curr[seg] = True
            else:
                seed_curr[seg] = True

        out[row, shadow_curr] = BLUE
        shadow_prev = shadow_curr
        seed_prev = seed_curr

    return statuses


def _paint_col_side(
    grid: np.ndarray,
    out: np.ndarray,
    cols: range,
    bars_by_col: dict[int, list[dict]],
) -> list[dict]:
    height = grid.shape[0]
    shadow_prev = np.zeros(height, dtype=bool)
    seed_prev = np.zeros(height, dtype=bool)
    statuses: list[dict] = []

    for col in cols:
        cand = _dilate_1d(shadow_prev) | seed_prev
        shadow_curr = cand & (grid[:, col] == BLACK)
        seed_curr = np.zeros(height, dtype=bool)

        for bar in bars_by_col.get(col, []):
            seg = slice(bar["start"], bar["end"] + 1)
            hit = bool(np.any(cand[seg]))
            status = dict(bar)
            status["disabled"] = hit
            statuses.append(status)
            if hit:
                out[seg, col] = BLUE
                shadow_curr[seg] = True
            else:
                seed_curr[seg] = True

        out[shadow_curr, col] = BLUE
        shadow_prev = shadow_curr
        seed_prev = seed_curr

    return statuses


def _solve_with_status(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    lights = np.argwhere(g == YELLOW)
    if lights.shape != (1, 2):
        raise AssertionError("expected exactly one yellow light source")
    light = tuple(map(int, lights[0]))
    lr, lc = light

    out = g.copy()
    bars = _parse_bars(g, light)
    statuses: list[dict] = []
    statuses.extend(_paint_row_side(g, out, range(lr - 1, -1, -1), bars["up"]))
    statuses.extend(_paint_row_side(g, out, range(lr + 1, g.shape[0]), bars["down"]))
    statuses.extend(_paint_col_side(g, out, range(lc - 1, -1, -1), bars["left"]))
    statuses.extend(_paint_col_side(g, out, range(lc + 1, g.shape[1]), bars["right"]))
    out[light] = YELLOW
    return out, statuses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast widening blue shadow wedges away from a yellow light, while any grey baffle touched by an earlier shadow turns blue and stops acting as a blocker or source.

    Input structure:
    - A black background contains exactly one yellow light pixel and several grey straight baffles.
    - Each baffle is a solid horizontal or vertical 1-cell-thick bar.
    - Horizontal bars lie strictly above or below the light; vertical bars lie strictly left or right of it.

    Rule:
    - Process each side of the light from near to far.
    - An effective grey baffle stays grey and projects blue shadow away from the light: the first shadow row or column keeps the same span, then every further step expands by one cell on both ends.
    - If a farther baffle has any cell inside an already effective shadow wedge, recolor the whole baffle blue; it becomes part of the shadow and does not cast a new wedge.
    - Paint shadow only onto background cells; the yellow light always stays yellow.

    Key insight:
    - The shadow is not a fixed union of independent wedges: near-to-far propagation matters because surviving grey baffles block and restart the wave, while shadowed blue baffles no longer block it, creating the intended occlusion chain.
    """
    return _solve_with_status(np.asarray(grid, dtype=int))[0]


def _find_segments(mask: np.ndarray, length: int, min_margin: int = 1) -> list[int]:
    n = mask.size
    starts: list[int] = []
    lo = min_margin
    hi = n - length - min_margin
    if hi < lo:
        return starts
    for start in range(lo, hi + 1):
        starts.append(start)
    return starts


def _choose_segment(
    rng: random.Random,
    cand: np.ndarray,
    line_len: int,
    mode: str,
) -> tuple[int, int, bool]:
    max_len = min(8, line_len - 2)
    min_len = min(3, max_len)
    options: list[tuple[int, int, bool]] = []

    for length in range(min_len, max_len + 1):
        for start in _find_segments(cand, length):
            end = start + length - 1
            overlap = int(np.count_nonzero(cand[start : end + 1]))
            if mode == "clear" and overlap == 0:
                options.append((start, end, False))
            elif mode == "shadowed" and overlap > 0:
                options.append((start, end, True))
            elif mode == "edge_touch" and 1 <= overlap < length:
                options.append((start, end, True))

    if not options:
        raise ValueError(f"no placement for mode={mode}")
    return rng.choice(options)


def _advance_empty(shadow_prev: np.ndarray, seed_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cand = _dilate_1d(shadow_prev) | seed_prev
    return cand, np.zeros_like(seed_prev)


def _plan_side(
    rng: random.Random,
    side: str,
    line_len: int,
    max_distance: int,
    force_chain: bool,
) -> list[dict]:
    n_bars = 3 if force_chain else rng.randint(1, 3)
    modes = ["clear"]
    if force_chain:
        modes.extend(["edge_touch", "shadowed"])
    else:
        modes.extend(rng.choice([["clear"], ["shadowed"], ["clear", "shadowed"]]))
    modes = modes[:n_bars]
    while len(modes) < n_bars:
        modes.append(rng.choice(["clear", "shadowed"]))

    shadow_prev = np.zeros(line_len, dtype=bool)
    seed_prev = np.zeros(line_len, dtype=bool)
    dist_cursor = 0
    plan: list[dict] = []

    for idx, mode in enumerate(modes):
        remaining = n_bars - idx - 1
        min_gap = 2
        max_gap = min(4, max_distance - dist_cursor - 2 * remaining)
        if max_gap < min_gap:
            raise ValueError("not enough room for planned baffles")
        gap = rng.randint(min_gap, max_gap)

        for _ in range(gap - 1):
            shadow_prev, seed_prev = _advance_empty(shadow_prev, seed_prev)

        dist = dist_cursor + gap
        cand = _dilate_1d(shadow_prev) | seed_prev
        start, end, hit = _choose_segment(rng, cand, line_len, mode)
        segment = np.zeros(line_len, dtype=bool)
        segment[start : end + 1] = True

        if hit:
            shadow_prev = cand.copy()
            shadow_prev[start : end + 1] = True
            seed_prev = np.zeros(line_len, dtype=bool)
        else:
            shadow_prev = cand.copy()
            seed_prev = segment

        plan.append(
            {
                "side": side,
                "distance": dist,
                "start": start,
                "end": end,
                "expected_disabled": hit,
            }
        )
        dist_cursor = dist

    return plan


def _place_plan(grid: np.ndarray, light: tuple[int, int], plan: list[dict]) -> None:
    lr, lc = light
    for bar in plan:
        side = bar["side"]
        start = int(bar["start"])
        end = int(bar["end"])
        dist = int(bar["distance"])
        if side == "up":
            row = lr - dist
            grid[row, start : end + 1] = GREY
        elif side == "down":
            row = lr + dist
            grid[row, start : end + 1] = GREY
        elif side == "left":
            col = lc - dist
            grid[start : end + 1, col] = GREY
        elif side == "right":
            col = lc + dist
            grid[start : end + 1, col] = GREY
        else:
            raise AssertionError(f"bad side {side}")


def _build_attempt() -> np.ndarray | None:
    rng = random.Random()
    horizontal_scene = rng.random() < 0.5

    if horizontal_scene:
        h = rng.randint(20, 29)
        w = rng.randint(18, 28)
        light = (rng.randint(7, h - 8), rng.randint(4, w - 5))
        candidate_sides = ["up", "down"]
        line_len = w
        max_dists = {"up": light[0], "down": h - 1 - light[0]}
    else:
        h = rng.randint(18, 28)
        w = rng.randint(20, 29)
        light = (rng.randint(4, h - 5), rng.randint(7, w - 8))
        candidate_sides = ["left", "right"]
        line_len = h
        max_dists = {"left": light[1], "right": w - 1 - light[1]}

    usable_sides = [side for side in candidate_sides if max_dists[side] >= 8]
    if not usable_sides:
        return None

    n_sides = 1 if len(usable_sides) == 1 else rng.randint(1, 2)
    selected = rng.sample(usable_sides, n_sides)
    chain_side = rng.choice(selected)

    plans: list[dict] = []
    for side in selected:
        try:
            plans.extend(_plan_side(rng, side, line_len, max_dists[side], force_chain=(side == chain_side)))
        except ValueError:
            return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[light] = YELLOW
    _place_plan(grid, light, plans)

    try:
        out, statuses = _solve_with_status(grid)
    except Exception:
        return None

    disabled = sum(1 for s in statuses if s["disabled"])
    active = len(statuses) - disabled
    if len(statuses) < 3 or disabled == 0 or active == 0:
        return None
    if not np.any(out == BLUE):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-chain input")
