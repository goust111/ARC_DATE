from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("top", "right", "bottom", "left")
STEP_BY_SIDE = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(comp) != expected:
        return None
    pts = set(comp)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in pts) != on_border:
                return None
    return r0, c0, r1, c1


def _boundary_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        return [(r0, c) for c in range(c0, c1 + 1)]
    if c0 == c1:
        return [(r, c0) for r in range(r0, r1 + 1)]
    cycle = [(r0, c) for c in range(c0, c1 + 1)]
    cycle.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    cycle.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    cycle.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return cycle


def _nearest_side(seed_r: int, seed_c: int, r0: int, c0: int, r1: int, c1: int) -> tuple[str, int]:
    distances = [
        ("top", seed_r - r0),
        ("right", c1 - seed_c),
        ("bottom", r1 - seed_r),
        ("left", seed_c - c0),
    ]
    best = min(dist for _side, dist in distances)
    sides = [side for side, dist in distances if dist == best]
    assert len(sides) == 1
    return sides[0], int(best)


def _trace_vine(
    out: np.ndarray,
    seed: tuple[int, int],
    color: int,
    inner_bbox: tuple[int, int, int, int],
) -> None:
    ir0, ic0, ir1, ic1 = inner_bbox
    sr, sc = seed
    side, straight_len = _nearest_side(sr, sc, ir0, ic0, ir1, ic1)
    dr, dc = STEP_BY_SIDE[side]
    target = {
        "top": (ir0, sc),
        "right": (sr, ic1),
        "bottom": (ir1, sc),
        "left": (sr, ic0),
    }[side]

    current = seed
    for _ in range(straight_len):
        current = (current[0] + dr, current[1] + dc)
        if out[current] == BG:
            out[current] = color

    cycle = _boundary_cycle(ir0, ic0, ir1, ic1)
    cycle_index = {pos: idx for idx, pos in enumerate(cycle)}
    idx = cycle_index[target]
    budget = len(cycle)
    for _ in range(budget - straight_len):
        idx = (idx + 1) % len(cycle)
        nxt = cycle[idx]
        if nxt == seed:
            break
        if out[nxt] == BG:
            out[nxt] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed into a vine: first straight toward its nearest inner wall, then clockwise along the wall.

    The total movement length equals the perimeter-cell count of the rectangle's interior.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frame_infos: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components(g == color):
            if len(comp) <= 1:
                continue
            bbox = _frame_bbox(comp)
            if bbox is not None:
                frame_infos.append((color, bbox))

    frame_infos.sort(key=lambda item: item[1])
    for frame_color, (r0, c0, r1, c1) in frame_infos:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_pts = np.argwhere((inner != BG) & (inner != frame_color))
        assert len(seed_pts) == 1
        sr = int(seed_pts[0, 0] + r0 + 1)
        sc = int(seed_pts[0, 1] + c0 + 1)
        _trace_vine(out, (sr, sc), int(g[sr, sc]), (r0 + 1, c0 + 1, r1 - 1, c1 - 1))

    return out


def _seed_candidates(interior_h: int, interior_w: int, side: str, require_touch: bool) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(interior_h):
        for c in range(interior_w):
            distances = {
                "top": r,
                "right": interior_w - 1 - c,
                "bottom": interior_h - 1 - r,
                "left": c,
            }
            best = min(distances.values())
            if distances[side] != best:
                continue
            if sum(dist == best for dist in distances.values()) != 1:
                continue
            if require_touch and best != 0:
                continue
            if not require_touch and best == 0:
                continue
            candidates.append((r, c))
    return candidates


def _sample_seed(interior_h: int, interior_w: int, side: str, rng: np.random.Generator) -> tuple[int, int]:
    prefer_touch = bool(rng.random() < 0.3)
    preferred = _seed_candidates(interior_h, interior_w, side, prefer_touch)
    if preferred:
        choice = preferred[int(rng.integers(len(preferred)))]
        return int(choice[0]), int(choice[1])

    fallback = _seed_candidates(interior_h, interior_w, side, not prefer_touch)
    assert fallback
    choice = fallback[int(rng.integers(len(fallback)))]
    return int(choice[0]), int(choice[1])


def _try_place_rectangles(
    height: int,
    width: int,
    specs: list[dict[str, int | str | tuple[int, int]]],
    rng: np.random.Generator,
) -> list[dict[str, int | str | tuple[int, int]]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, int | str | tuple[int, int]]] = []
    for spec in specs:
        frame_h = int(spec["interior_h"]) + 2
        frame_w = int(spec["interior_w"]) + 2
        if frame_h + 2 > height or frame_w + 2 > width:
            return None

        success = False
        for _ in range(200):
            top = int(rng.integers(1, height - frame_h))
            left = int(rng.integers(1, width - frame_w))
            if blocked[top - 1 : top + frame_h + 1, left - 1 : left + frame_w + 1].any():
                continue
            blocked[top - 1 : top + frame_h + 1, left - 1 : left + frame_w + 1] = True
            item = dict(spec)
            item["top"] = top
            item["left"] = left
            placed.append(item)
            success = True
            break
        if not success:
            return None
    return placed


def _build_grid(
    height: int,
    width: int,
    placed: list[dict[str, int | str | tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for item in placed:
        top = int(item["top"])
        left = int(item["left"])
        interior_h = int(item["interior_h"])
        interior_w = int(item["interior_w"])
        color = int(item["color"])
        seed_r, seed_c = item["seed"]

        grid[top, left : left + interior_w + 2] = FRAME_COLOR
        grid[top + interior_h + 1, left : left + interior_w + 2] = FRAME_COLOR
        grid[top : top + interior_h + 2, left] = FRAME_COLOR
        grid[top : top + interior_h + 2, left + interior_w + 1] = FRAME_COLOR
        grid[top + 1 + int(seed_r), left + 1 + int(seed_c)] = color
    return grid


def _sample_specs(rng: np.random.Generator, n_frames: int) -> list[dict[str, int | str | tuple[int, int]]]:
    colors = rng.choice(np.array(SEED_COLORS, dtype=int), size=n_frames, replace=False)
    dirs = list(DIRS)
    rng.shuffle(dirs)
    specs: list[dict[str, int | str | tuple[int, int]]] = []
    for idx in range(n_frames):
        interior_h = int(rng.integers(5, 10))
        interior_w = int(rng.integers(5, 10))
        side = dirs[idx]
        seed = _sample_seed(interior_h, interior_w, side, rng)
        specs.append(
            {
                "interior_h": interior_h,
                "interior_w": interior_w,
                "side": side,
                "seed": seed,
                "color": int(colors[idx]),
            }
        )
    return specs


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        n_frames = int(rng.integers(1, 5))
        height = int(rng.integers(18 + 2 * n_frames, 31))
        width = int(rng.integers(18 + 2 * n_frames, 31))
        specs = _sample_specs(rng, n_frames)
        placed = _try_place_rectangles(height, width, specs, rng)
        if placed is None:
            continue
        grid = _build_grid(height, width, placed)
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("generate_input failed")
