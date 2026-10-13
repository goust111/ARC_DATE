from __future__ import annotations

import random

import numpy as np

BG = 8
BLUE = 1
RED = 2
GREEN = 3

GEN_ATTEMPTS = 3000


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    cid = 0
    for comp in _components(mask):
        cid += 1
        rows = comp[:, 0]
        cols = comp[:, 1]
        labels[rows, cols] = cid
        bboxes[cid] = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
    return labels, bboxes


def _extract_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components((g == BLUE) | (g == RED))
    bars: list[dict] = []
    for comp in comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert c0 == c1, "each obstacle bar must occupy a single column"
        top = int(rows.min())
        bottom = int(rows.max())
        expected_rows = list(range(top, bottom + 1))
        assert sorted(map(int, rows.tolist())) == expected_rows, "bar rows must be contiguous"
        colors = [int(g[r, c0]) for r in expected_rows]
        blue_rows = tuple(i for i, color in enumerate(colors) if color == BLUE)
        assert 1 <= len(blue_rows) <= 3
        bars.append(
            {
                "top": top,
                "bottom": bottom,
                "col": c0,
                "rows": tuple(expected_rows),
                "colors": tuple(colors),
                "blue_rows": blue_rows,
                "blue_count": len(blue_rows),
                "height": bottom - top + 1,
                "bbox": (top, c0, bottom, c0),
            }
        )
    assert 2 <= len(bars) <= 4
    return bars


def _bar_sort_key(bar: dict) -> tuple[int, int, int]:
    return (-int(bar["col"]), int(bar["top"]), -int(bar["height"]))


def _settle_bars(bars: list[dict], shape: tuple[int, int]) -> tuple[list[dict], list[str]]:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    settled: list[dict] = []
    reasons: list[str] = []
    for bar in sorted(bars, key=_bar_sort_key):
        rows = np.asarray(bar["rows"], dtype=int)
        col = int(bar["col"])
        shift = 0
        reason = "wall"
        while True:
            next_col = col + shift + 1
            if next_col >= w:
                reason = "wall"
                break
            if np.any(occupied[rows, next_col]):
                reason = "object"
                break
            shift += 1
        final_col = col + shift
        for row in rows:
            occupied[row, final_col] = True
        colors = tuple(int(x) for x in bar["colors"])
        blue_rows = tuple(int(x) for x in bar["blue_rows"])
        settled.append(
            {
                "top": int(bar["top"]),
                "bottom": int(bar["bottom"]),
                "col": int(final_col),
                "rows": tuple(int(x) for x in rows.tolist()),
                "colors": colors,
                "blue_rows": blue_rows,
                "height": int(bar["height"]),
                "blue_count": len(blue_rows),
                "bbox": (int(bar["top"]), int(final_col), int(bar["bottom"]), int(final_col)),
            }
        )
        reasons.append(reason)
    return settled, reasons


def _render_bars(shape: tuple[int, int], bars: list[dict], *, recolor_blue: bool = False) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for bar in bars:
        col = int(bar["col"])
        for offset, row in enumerate(bar["rows"]):
            color = int(bar["colors"][offset])
            if recolor_blue and color == BLUE:
                color = RED
            out[int(row), col] = color
    return out


def _forbidden_mask(shape: tuple[int, int], bars: list[dict], *, radius_mode: str = "actual") -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for bar in bars:
        if radius_mode == "actual":
            radius = int(bar["blue_count"])
        elif radius_mode == "zero":
            radius = 0
        elif radius_mode == "one":
            radius = 1
        else:
            raise ValueError(f"unknown radius_mode: {radius_mode}")
        top = max(0, int(bar["top"]) - radius)
        bottom = min(h - 1, int(bar["bottom"]) + radius)
        left = max(0, int(bar["col"]) - radius)
        right = min(w - 1, int(bar["col"]) + radius)
        forbidden[top : bottom + 1, left : right + 1] = True
    return forbidden


def _paint_output(shape: tuple[int, int], settled: list[dict], path: list[tuple[int, int]]) -> np.ndarray:
    out = _render_bars(shape, settled, recolor_blue=True)
    for r, c in path:
        out[r, c] = GREEN
    return out


def _trace_path(blocked: np.ndarray, start_row: int) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    if blocked[start_row, 0]:
        return None
    labels, bboxes = _label_components(blocked)
    r = int(start_row)
    c = 0
    target_row = int(start_row)
    path = [(r, c)]
    active_right_edge = -1

    while c < w - 1:
        if active_right_edge >= 0 and c > active_right_edge and r != target_row:
            step = -1 if r > target_row else 1
            if 0 <= r + step < h and not blocked[r + step, c]:
                r += step
                if path[-1] != (r, c):
                    path.append((r, c))
                continue

        if not blocked[r, c + 1]:
            c += 1
            if path[-1] != (r, c):
                path.append((r, c))
            continue

        comp_id = int(labels[r, c + 1])
        if comp_id == 0:
            return None
        top, _left, bottom, right = bboxes[comp_id]
        candidates: list[tuple[int, int, int]] = []
        up_row = top - 1
        down_row = bottom + 1
        if up_row >= 0:
            col_slice = blocked[min(r, up_row) : max(r, up_row) + 1, c]
            if not col_slice.any() and not blocked[up_row, c + 1]:
                candidates.append((abs(up_row - target_row), 0, up_row))
        if down_row < h:
            col_slice = blocked[min(r, down_row) : max(r, down_row) + 1, c]
            if not col_slice.any() and not blocked[down_row, c + 1]:
                candidates.append((abs(down_row - target_row), 1, down_row))
        if not candidates:
            return None
        _, _, bypass_row = min(candidates)
        step = -1 if bypass_row < r else 1
        while r != bypass_row:
            r += step
            if blocked[r, c]:
                return None
            if path[-1] != (r, c):
                path.append((r, c))
        active_right_edge = int(right)

    return path


def _solve_variant(
    grid: np.ndarray,
    *,
    settle: bool = True,
    radius_mode: str = "actual",
    return_meta: bool = False,
):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts = np.argwhere(g == GREEN)
    assert starts.shape[0] == 1, "expected exactly one green start pixel"
    sr, sc = map(int, starts[0])
    assert sc == 0, "green start must lie on the left border"

    bars = _extract_bars(g)
    active_bars, reasons = (_settle_bars(bars, (h, w)) if settle else (bars, ["original"] * len(bars)))
    forbidden = _forbidden_mask((h, w), active_bars, radius_mode=radius_mode)
    obstacle_mask = _render_bars((h, w), active_bars) != BG
    blocked = forbidden | obstacle_mask
    blocked[sr, sc] = False
    path = _trace_path(blocked, sr)
    if path is None:
        raise AssertionError("path must exist")
    out = _paint_output((h, w), active_bars, path)
    if return_meta:
        return out, {
            "bars": active_bars,
            "stop_reasons": reasons,
            "path": path,
            "start_row": sr,
            "forbidden": forbidden,
            "blocked": blocked,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide the mixed-color obstacle bars rightward, then draw a green route that detours around their settled keep-out zones.

    Input structure:
    - The background is azure (8).
    - A single green (3) start pixel lies on the left border.
    - There are 2 to 4 obstacle bars, each a single vertical contiguous column of red (2) and blue (1) pixels.

    Rule:
    - Treat each bar as a rigid object and settle the bars from right to left, moving each one horizontally right until the next step would leave the grid or overlap a bar that has already settled.
    - After settling, count the blue pixels on each bar; that count is the bar's keep-out radius. Every cell within Chebyshev distance at most that radius from any bar cell becomes forbidden.
    - Starting from the green border pixel, extend a one-cell-wide green path to the right border. The path prefers moving right; when the next right cell is forbidden it chooses the upper or lower detour around that blocked region by staying closer to the start row, with ties going upward, and after clearing the region it returns toward the start row as soon as possible.
    - Finally recolor every blue bar pixel to red at the settled positions.

    Key insight:
    - The path is constrained by the bars only after horizontal gravity settles them, and each settled bar's blue count controls how wide its forbidden zone becomes.
    """
    out = _solve_variant(grid, settle=True, radius_mode="actual", return_meta=False)
    return np.asarray(out, dtype=int)


def _sample_columns(rng: random.Random, low: int, high: int, count: int, gap: int) -> list[int] | None:
    pool = list(range(low, high + 1))
    rng.shuffle(pool)
    chosen: list[int] = []
    for col in pool:
        if all(abs(col - prev) >= gap for prev in chosen):
            chosen.append(col)
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _random_bar(rng: random.Random, h: int, col: int, start_row: int, kind: str) -> dict:
    height = rng.randint(3, 6)
    blue_count = rng.randint(1, min(3, height))
    if kind == "cross":
        top_lo = max(0, start_row - height + 1)
        top_hi = min(h - height, start_row)
    elif kind == "high":
        center = max(1, start_row - rng.randint(2, 4))
        top_lo = max(0, center - height + 1)
        top_hi = min(h - height, center)
    elif kind == "low":
        center = min(h - 2, start_row + rng.randint(2, 4))
        top_lo = max(0, center - height + 1)
        top_hi = min(h - height, center)
    else:
        top_lo = 0
        top_hi = h - height
    top = rng.randint(top_lo, top_hi)
    colors = [RED] * height
    for idx in rng.sample(range(height), blue_count):
        colors[idx] = BLUE
    rows = tuple(range(top, top + height))
    return {
        "top": top,
        "bottom": top + height - 1,
        "col": col,
        "rows": rows,
        "colors": tuple(colors),
        "blue_rows": tuple(i for i, color in enumerate(colors) if color == BLUE),
        "height": height,
    }


def _path_stats(path: list[tuple[int, int]], start_row: int) -> dict[str, int]:
    vertical_cells = sum(1 for r, _c in path if r != start_row)
    turns = 0
    prev_dir = None
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        cur_dir = (int(np.sign(r1 - r0)), int(np.sign(c1 - c0)))
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return {"vertical_cells": vertical_cells, "turns": turns, "length": len(path)}


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        out, meta = _solve_variant(inp, settle=True, radius_mode="actual", return_meta=True)
        no_gravity = _solve_variant(inp, settle=False, radius_mode="actual", return_meta=False)
        no_radius = _solve_variant(inp, settle=True, radius_mode="zero", return_meta=False)
    except Exception:
        return False

    bars = meta["bars"]
    reasons = meta["stop_reasons"]
    path = meta["path"]
    start_row = int(meta["start_row"])
    stats = _path_stats(path, start_row)

    if stats["vertical_cells"] == 0 or stats["turns"] < 2:
        return False
    if out.shape != inp.shape:
        return False
    if not any(reason == "object" for reason in reasons):
        return False
    final_cols = {int(bar["col"]) for bar in bars}
    if len(final_cols) < 2:
        return False
    if max(int(bar["blue_count"]) for bar in bars) <= 1:
        return False
    if np.array_equal(out, no_gravity):
        return False
    if np.array_equal(out, no_radius):
        return False
    if sum(1 for bar in bars if int(bar["top"]) - int(bar["blue_count"]) <= start_row <= int(bar["bottom"]) + int(bar["blue_count"])) == 0:
        return False
    if int((inp == GREEN).sum()) != 1:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 18)
    w = rng.randint(17, 22)
    start_row = rng.randint(3, h - 4)
    n_bars = rng.randint(2, 4)

    cols = _sample_columns(rng, 2, max(5, w // 2 + 2), n_bars, gap=2)
    if cols is None:
        return None

    kinds = ["cross"]
    while len(kinds) < n_bars:
        kinds.append(rng.choice(["cross", "high", "low"]))
    rng.shuffle(kinds)

    bars = [_random_bar(rng, h, col, start_row, kind) for col, kind in zip(cols, kinds)]

    grid = np.full((h, w), BG, dtype=int)
    grid[start_row, 0] = GREEN
    for bar in bars:
        col = int(bar["col"])
        for offset, row in enumerate(bar["rows"]):
            grid[int(row), col] = int(bar["colors"][offset])

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
