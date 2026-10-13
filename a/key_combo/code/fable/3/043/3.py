from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


CoordArray = np.ndarray


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: CoordArray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(grid: np.ndarray, bg: int, stripe_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or stripe_mask[sr, sc] or int(grid[sr, sc]) == bg:
                continue
            color = int(grid[sr, sc])
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or stripe_mask[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(pts, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _overlaps_mask(coords: CoordArray, mask: np.ndarray) -> bool:
    return bool(np.any(mask[coords[:, 0], coords[:, 1]]))


def _overlaps_coords(a: CoordArray, b: CoordArray) -> bool:
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    large_set = {(int(r), int(c)) for r, c in large}
    return any((int(r), int(c)) in large_set for r, c in small)


def _in_bounds(coords: CoordArray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.all(coords[:, 0] >= 0)
        and np.all(coords[:, 0] < h)
        and np.all(coords[:, 1] >= 0)
        and np.all(coords[:, 1] < w)
    )


def _solve(grid: np.ndarray, *, trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    stripe_color = int(g[0, 0])
    assert stripe_color != bg and np.all(g[0, :] == stripe_color), "top row must be the ceiling stripe"
    stripe_mask = np.zeros((h, w), dtype=bool)
    stripe_mask[0, :] = True

    objects = _components(g, bg, stripe_mask)
    assert objects, "expected moving objects below the stripe"
    top_rows = [int(obj["bbox"][0]) for obj in objects]
    assert len(set(top_rows)) == len(top_rows), "object top rows must be distinct"
    objects = sorted(objects, key=lambda obj: int(obj["bbox"][0]))

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    fixed_mask = stripe_mask.copy()
    settled: list[dict] = []
    traces: list[dict] = []

    for order, obj in enumerate(objects):
        color = int(obj["color"])
        coords = obj["coords"].copy()
        quota = int(obj["area"])
        steps: list[str] = []
        blockers_seen: set[int] = set()
        ceiling_stop = False

        while quota > 0:
            up = coords + np.array([-1, 0], dtype=int)
            if not _in_bounds(up, g.shape) or _overlaps_mask(up, stripe_mask):
                ceiling_stop = True
                break

            if not _overlaps_mask(up, fixed_mask):
                coords = up
                quota -= 1
                steps.append("U")
                continue

            hit_indices = [
                idx for idx, settled_obj in enumerate(settled)
                if _overlaps_coords(up, settled_obj["coords"])
            ]
            assert hit_indices, "only settled objects may trigger detours"
            for idx in hit_indices:
                blockers_seen.add(int(settled[idx]["order"]))

            blocker_coords = np.concatenate([settled[idx]["coords"] for idx in hit_indices], axis=0)
            blocker_left = int(blocker_coords[:, 1].min())
            blocker_right = int(blocker_coords[:, 1].max())
            obj_left = int(coords[:, 1].min())
            obj_right = int(coords[:, 1].max())

            left_steps = obj_right - blocker_left + 1
            right_steps = blocker_right - obj_left + 1
            dc = 1 if right_steps <= left_steps else -1

            side = coords + np.array([0, dc], dtype=int)
            assert _in_bounds(side, g.shape), "detour step must stay in bounds"
            assert not _overlaps_mask(side, fixed_mask), "detour step must not collide"

            coords = side
            quota -= 1
            steps.append("R" if dc > 0 else "L")

        assert not _overlaps_mask(coords, fixed_mask), "settled objects must not overlap"
        out[coords[:, 0], coords[:, 1]] = color
        fixed_mask[coords[:, 0], coords[:, 1]] = True

        rec = {
            "order": order,
            "color": color,
            "area": int(obj["area"]),
            "initial_bbox": obj["bbox"],
            "final_bbox": _bbox(coords),
            "steps": tuple(steps),
            "ceiling_stop": ceiling_stop,
            "blockers": tuple(sorted(blockers_seen)),
            "detoured": any(step in ("L", "R") for step in steps),
            "coords": coords.copy(),
        }
        traces.append(rec)
        settled.append({"coords": coords.copy(), "order": order})

    if trace:
        return out, traces
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move ordered objects upward using area as path budget, detouring around settled blockers.

    Input structure:
    - The grid has a dominant background color and row 0 is one solid full-width
      non-background ceiling stripe.
    - Below the stripe are several 4-connected monochrome foreground objects.
    - The objects have distinct initial top bbox rows, giving a unique high-to-low
      processing order.

    Rule:
    - Clear all original object cells, keep the ceiling stripe fixed, and process
      objects by increasing initial top row.
    - Each object's movement quota is its pixel count. Each one-cell move costs
      one quota point: move up when possible; if upward motion would overlap a
      previously settled object, move horizontally toward the side that exits the
      blocker column span sooner, breaking equal distances to the right, then try
      upward again on the next step.
    - If upward motion would hit the ceiling stripe, stop the object immediately
      and discard any remaining quota. Settled objects block all later objects.

    Key insight:
    - Area is a route budget rather than a vertical displacement: horizontal
      detours spend the same quota, and earlier settled paths change later paths.
    """
    return _solve(grid)


# ---------------- generator ----------------

_RETRY_LIMIT = 500
_COLORS = tuple(range(10))


def _line_vertical(height: int) -> list[tuple[int, int]]:
    return [(r, 0) for r in range(height)]


def _line_horizontal(width: int) -> list[tuple[int, int]]:
    return [(0, c) for c in range(width)]


def _l_shape(area: int) -> list[tuple[int, int]]:
    if area == 3:
        return [(0, 0), (1, 0), (1, 1)]
    if area == 4:
        return [(0, 0), (1, 0), (2, 0), (2, 1)]
    return [(0, 0), (1, 0)]


def _paint_object(
    grid: np.ndarray,
    rel: list[tuple[int, int]],
    top: int,
    left: int,
    color: int,
    bg: int,
    used_tops: set[int],
) -> bool:
    pts = [(top + r, left + c) for r, c in rel]
    h, w = grid.shape
    if top in used_tops:
        return False
    if any(r <= 0 or r >= h or c < 0 or c >= w for r, c in pts):
        return False
    if any(int(grid[r, c]) != bg for r, c in pts):
        return False
    for r, c in pts:
        grid[r, c] = color
    used_tops.add(top)
    return True


def _choose_colors() -> tuple[int, int, list[int]]:
    bg = random.choice(_COLORS)
    stripe = random.choice([c for c in _COLORS if c != bg])
    object_colors = random.sample([c for c in _COLORS if c not in (bg, stripe)], 7)
    return bg, stripe, object_colors


def _pure_shape() -> list[tuple[int, int]]:
    area = random.choice([2, 3, 4])
    kind = random.choice(["v", "h", "l"])
    if kind == "v":
        return _line_vertical(area)
    if kind == "h":
        return _line_horizontal(area)
    return _l_shape(area)


def _shape_size(rel: list[tuple[int, int]]) -> tuple[int, int, int]:
    rows = [r for r, _ in rel]
    cols = [c for _, c in rel]
    return len(rel), max(rows) + 1, max(cols) + 1


def _add_pure_object(
    grid: np.ndarray,
    color: int,
    bg: int,
    used_tops: set[int],
    preferred_left: int,
) -> bool:
    rel = _pure_shape()
    area, height, width = _shape_size(rel)
    h, w = grid.shape
    lo = area + 2
    hi = h - height - 2
    if lo > hi:
        return False

    top_options = [r for r in range(lo, hi + 1) if r not in used_tops]
    random.shuffle(top_options)
    left_options = list(range(max(1, preferred_left - 1), min(w - width - 1, preferred_left + 2) + 1))
    random.shuffle(left_options)
    for top in top_options:
        for left in left_options:
            if _paint_object(grid, rel, top, left, color, bg, used_tops):
                return True
    return False


def _has_semantic_coverage(grid: np.ndarray) -> bool:
    try:
        out, traces = _solve(grid, trace=True)
    except Exception:
        return False
    if np.array_equal(out, grid):
        return False

    detoured_orders = {rec["order"] for rec in traces if rec["detoured"]}
    has_ceiling = any(rec["ceiling_stop"] for rec in traces)
    has_plain_budget = any(
        rec["steps"]
        and len(rec["steps"]) == rec["area"]
        and all(step == "U" for step in rec["steps"])
        for rec in traces
    )
    has_detour_resume = any(
        any(step in ("L", "R") for step in rec["steps"])
        and any(
            rec["steps"][i] in ("L", "R") and "U" in rec["steps"][i + 1 :]
            for i in range(len(rec["steps"]))
        )
        for rec in traces
    )
    has_chain = any(
        rec["detoured"] and any(blocker in detoured_orders for blocker in rec["blockers"])
        for rec in traces
    )
    has_left_or_right = any("L" in rec["steps"] for rec in traces) or any("R" in rec["steps"] for rec in traces)
    return has_ceiling and has_plain_budget and has_detour_resume and has_chain and has_left_or_right


def _gen_attempt() -> np.ndarray | None:
    side_mode = random.choice(["tie_right", "left", "right"])
    block_width = 1 if side_mode == "tie_right" else random.choice([3, 4])
    block_top = 2 + block_width
    detour_top = block_top + 2
    detour_height = block_width + 3
    detour_bottom = detour_top + detour_height - 1
    chain_top = detour_bottom + 2
    chain_height = block_width + 5
    chain_bottom = chain_top + chain_height - 1

    min_h = chain_bottom + 2
    if min_h > 30:
        return None
    height = random.randint(min_h, 30)
    width = random.randint(20, 27)

    bg, stripe, colors = _choose_colors()
    grid = np.full((height, width), bg, dtype=int)
    grid[0, :] = stripe
    used_tops: set[int] = set()

    if side_mode == "tie_right":
        block_left = random.randint(7, width - 8)
        detour_left = block_left
        expected_detour_dir = 1
    else:
        block_left = random.randint(7, width - block_width - 8)
        if side_mode == "left":
            detour_left = block_left
            expected_detour_dir = -1
        else:
            detour_left = block_left + block_width - 1
            expected_detour_dir = 1
    chain_left = detour_left + expected_detour_dir

    # Ceiling stopper: it moves once, then the stripe discards remaining quota.
    stopper_rel = _line_horizontal(3) if random.random() < 0.5 else _line_vertical(3)
    stopper_left = 1
    if not _paint_object(grid, stopper_rel, 2, stopper_left, colors[0], bg, used_tops):
        return None

    # First blocker settles above the next object. Its width controls left/right branch choice.
    blocker_rel = _line_horizontal(block_width)
    if not _paint_object(grid, blocker_rel, block_top, block_left, colors[1], bg, used_tops):
        return None

    # This object must spend quota on the blocker detour and then resume upward.
    detour_rel = _line_vertical(detour_height)
    if not _paint_object(grid, detour_rel, detour_top, detour_left, colors[2], bg, used_tops):
        return None

    # This object is blocked by the already-detoured object, forcing chain behavior.
    chain_rel = _line_vertical(chain_height)
    if not _paint_object(grid, chain_rel, chain_top, chain_left, colors[3], bg, used_tops):
        return None

    # A far object rules out the shortcut "all objects only react to blockers".
    if not _add_pure_object(grid, colors[4], bg, used_tops, width - 4):
        return None

    # Sometimes add another harmless shape to vary object count and row layout.
    if random.random() < 0.55:
        extra_left = width - 8 if side_mode == "right" else 4
        _add_pure_object(grid, colors[5], bg, used_tops, extra_left)

    if not _has_semantic_coverage(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quota-detour-chain input")
