from __future__ import annotations

from collections import Counter
import random

import numpy as np


LEGEND_LEN = 5
FRAME_COLOR = 5
ALL_COLORS = tuple(range(10))
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _non_bg_counts(grid: np.ndarray, bg: int) -> Counter:
    return Counter(int(v) for v in grid.flat if int(v) != int(bg))


def _line_candidates(h: int, w: int) -> list[list[tuple[int, int]]]:
    seqs: list[list[tuple[int, int]]] = []
    if w >= LEGEND_LEN:
        seqs.extend(
            [
                [(0, i) for i in range(LEGEND_LEN)],
                [(0, w - 1 - i) for i in range(LEGEND_LEN)],
                [(h - 1, i) for i in range(LEGEND_LEN)],
                [(h - 1, w - 1 - i) for i in range(LEGEND_LEN)],
            ]
        )
    if h >= LEGEND_LEN:
        seqs.extend(
            [
                [(i, 0) for i in range(LEGEND_LEN)],
                [(i, w - 1) for i in range(LEGEND_LEN)],
                [(h - 1 - i, 0) for i in range(LEGEND_LEN)],
                [(h - 1 - i, w - 1) for i in range(LEGEND_LEN)],
            ]
        )
    return seqs


def _find_legend(grid: np.ndarray, bg: int, frame_color: int) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    counts = Counter(map(int, grid.flat))
    best: tuple[list[int], set[tuple[int, int]]] | None = None

    for coords in _line_candidates(h, w):
        vals = [int(grid[r, c]) for r, c in coords]
        if any(v in (bg, frame_color) for v in vals):
            continue
        if len(set(vals)) != LEGEND_LEN:
            continue
        if any(counts[v] != 1 for v in vals):
            continue
        coord_set = set(coords)
        if best is not None:
            raise AssertionError("legend detection should be unique")
        best = (vals, coord_set)

    assert best is not None, "failed to locate the 5-cell legend"
    return best


def _mask_prefix(mask: np.ndarray) -> np.ndarray:
    return mask.astype(int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r1, c) for c in range(c0, c1 + 1))
    coords.extend((r, c0) for r in range(r0 + 1, r1))
    coords.extend((r, c1) for r in range(r0 + 1, r1))
    return coords


def _candidate_rectangles(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    grey_coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not grey_coords:
        return []

    coord_to_id = {coord: idx for idx, coord in enumerate(grey_coords)}
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    prefix = _mask_prefix(mask)
    candidates: list[dict] = []

    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            r0 = int(r0)
            r1 = int(r1)
            if r1 - r0 < 2:
                continue
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    c0 = int(c0)
                    c1 = int(c1)
                    if c1 - c0 < 2:
                        continue

                    top = _rect_sum(prefix, r0, c0, r0, c1)
                    bottom = _rect_sum(prefix, r1, c0, r1, c1)
                    left_full = _rect_sum(prefix, r0, c0, r1, c0)
                    right_full = _rect_sum(prefix, r0, c1, r1, c1)
                    if min(top, bottom, left_full, right_full) <= 0:
                        continue

                    interior = _rect_sum(prefix, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                    if interior != 0:
                        continue

                    perimeter = 2 * ((r1 - r0) + (c1 - c0))
                    left = _rect_sum(prefix, r0 + 1, c0, r1 - 1, c0)
                    right = _rect_sum(prefix, r0 + 1, c1, r1 - 1, c1)
                    border_count = top + bottom + left + right
                    missing = perimeter - border_count
                    if not (1 <= missing <= LEGEND_LEN):
                        continue

                    obs_ids: set[int] = set()
                    missing_cells: list[tuple[int, int]] = []
                    border = _border_coords(r0, c0, r1, c1)
                    for cell in border:
                        if bool(mask[cell]):
                            obs_ids.add(coord_to_id[cell])
                        else:
                            missing_cells.append(cell)

                    if len(missing_cells) != missing:
                        continue

                    candidates.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "observed_ids": frozenset(obs_ids),
                            "border": frozenset(border),
                            "missing_count": int(missing),
                            "missing_cells": tuple(missing_cells),
                        }
                    )

    return candidates


def _solve_rectangles(mask: np.ndarray) -> list[dict]:
    grey_coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not grey_coords:
        return []

    candidates = _candidate_rectangles(mask)
    assert candidates, "expected at least one valid incomplete rectangle"

    by_pixel: dict[int, list[int]] = {idx: [] for idx in range(len(grey_coords))}
    for ci, cand in enumerate(candidates):
        for pid in cand["observed_ids"]:
            by_pixel[int(pid)].append(ci)

    def dfs(uncovered: frozenset[int], used_border: frozenset[tuple[int, int]]) -> list[dict] | None:
        if not uncovered:
            return []

        best_pixel = None
        best_options = None
        for pid in uncovered:
            opts = []
            for ci in by_pixel[int(pid)]:
                cand = candidates[ci]
                if not cand["observed_ids"].issubset(uncovered):
                    continue
                if not cand["border"].isdisjoint(used_border):
                    continue
                opts.append(ci)
            if not opts:
                return None
            if best_options is None or len(opts) < len(best_options):
                best_pixel = int(pid)
                best_options = opts

        assert best_pixel is not None and best_options is not None
        best_options.sort(
            key=lambda ci: (
                -len(candidates[ci]["observed_ids"]),
                candidates[ci]["missing_count"],
                candidates[ci]["bbox"],
            )
        )

        for ci in best_options:
            cand = candidates[ci]
            result = dfs(
                uncovered - cand["observed_ids"],
                used_border | cand["border"],
            )
            if result is not None:
                return [cand] + result
        return None

    solution = dfs(frozenset(range(len(grey_coords))), frozenset())
    assert solution is not None, "failed to partition grey pixels into rectangles"
    return solution


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    counts = _non_bg_counts(g, bg)
    assert counts, "input must contain non-background objects"
    frame_color = max(counts, key=lambda col: (counts[col], col))

    legend_colors, legend_coords = _find_legend(g, bg, int(frame_color))
    mask = g == int(frame_color)
    rects = _solve_rectangles(mask)

    out = g.copy()
    for r, c in legend_coords:
        out[r, c] = int(bg)

    for rect in rects:
        fill_color = int(legend_colors[int(rect["missing_count"]) - 1])
        for r, c in rect["missing_cells"]:
            out[r, c] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair incomplete grey rectangle frames using a corner legend keyed by
    the number of missing border pixels.
    """
    return _transform(grid)


def _legend_layout(h: int, w: int, corner: str, vertical: bool) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(i, 0) for i in range(LEGEND_LEN)] if vertical else [(0, i) for i in range(LEGEND_LEN)]
    if corner == "tr":
        return [(i, w - 1) for i in range(LEGEND_LEN)] if vertical else [(0, w - 1 - i) for i in range(LEGEND_LEN)]
    if corner == "bl":
        return [(h - 1 - i, 0) for i in range(LEGEND_LEN)] if vertical else [(h - 1, i) for i in range(LEGEND_LEN)]
    if corner == "br":
        return (
            [(h - 1 - i, w - 1) for i in range(LEGEND_LEN)]
            if vertical
            else [(h - 1, w - 1 - i) for i in range(LEGEND_LEN)]
        )
    raise ValueError(corner)


def _side_nonempty(remaining: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> bool:
    top = any((r0, c) in remaining for c in range(c0, c1 + 1))
    bottom = any((r1, c) in remaining for c in range(c0, c1 + 1))
    left = any((r, c0) in remaining for r in range(r0, r1 + 1))
    right = any((r, c1) in remaining for r in range(r0, r1 + 1))
    return top and bottom and left and right


def _sample_missing(border: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int, missing: int) -> set[tuple[int, int]] | None:
    for _ in range(120):
        removed = set(random.sample(border, missing))
        remaining = set(border) - removed
        if _side_nonempty(remaining, r0, c0, r1, c1):
            return removed
    return None


def _paint_expected(grid: np.ndarray, legend_coords: list[tuple[int, int]], frames: list[dict], legend_colors: list[int], bg: int) -> np.ndarray:
    out = grid.copy()
    for r, c in legend_coords:
        out[r, c] = int(bg)
    for frame in frames:
        fill = int(legend_colors[int(frame["missing_count"]) - 1])
        for r, c in frame["missing_cells"]:
            out[r, c] = fill
    return out


def _placeable(used: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(used.shape[0] - 1, r1 + 1)
    cc1 = min(used.shape[1] - 1, c1 + 1)
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(used: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(used.shape[0] - 1, r1 + 1)
    cc1 = min(used.shape[1] - 1, c1 + 1)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    bg = random.choice([c for c in ALL_COLORS if c != FRAME_COLOR])
    legend_colors = random.sample([c for c in ALL_COLORS if c not in (bg, FRAME_COLOR)], LEGEND_LEN)
    corner = random.choice(["tl", "tr", "bl", "br"])
    vertical = bool(random.getrandbits(1))

    grid = np.full((h, w), int(bg), dtype=int)
    legend_coords = _legend_layout(h, w, corner, vertical)
    for idx, (r, c) in enumerate(legend_coords):
        grid[r, c] = int(legend_colors[idx])

    used = np.zeros((h, w), dtype=bool)
    for r, c in legend_coords:
        used[r, c] = True

    frame_target = random.randint(1, 4)
    frames: list[dict] = []

    for _ in range(200):
        if len(frames) >= frame_target:
            break

        max_h = min(10, h - 4)
        max_w = min(10, w - 4)
        if max_h < 4 or max_w < 4:
            return None

        fh = random.randint(4, max_h)
        fw = random.randint(4, max_w)
        r0 = random.randint(2, h - fh - 2)
        c0 = random.randint(2, w - fw - 2)
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1

        if not _placeable(used, r0, c0, r1, c1):
            continue

        border = _border_coords(r0, c0, r1, c1)
        missing_count = random.randint(1, LEGEND_LEN)
        removed = _sample_missing(border, r0, c0, r1, c1, missing_count)
        if removed is None:
            continue

        for cell in border:
            if cell not in removed:
                grid[cell] = FRAME_COLOR

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "missing_count": int(missing_count),
                "missing_cells": tuple(sorted(removed)),
            }
        )
        _reserve(used, r0, c0, r1, c1)

    if not frames:
        return None

    expected = _paint_expected(grid, legend_coords, frames, legend_colors, bg)
    if not np.array_equal(_transform(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC example")
