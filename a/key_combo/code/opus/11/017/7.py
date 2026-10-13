from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

FILL_COLORS = (BLUE, YELLOW, ORANGE)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _border_cycle(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    border = [(0, c) for c in range(w)]
    border.extend((r, w - 1) for r in range(1, h))
    border.extend((h - 1, c) for c in range(w - 2, -1, -1))
    border.extend((r, 0) for r in range(h - 2, 0, -1))
    return border


def _find_single_border_segment(grid: np.ndarray) -> tuple[list[tuple[int, int]], int, int]:
    border = _border_cycle(grid.shape)
    green = np.array([grid[r, c] == GREEN for r, c in border], dtype=bool)
    assert green.any(), "expected a green border segment"
    starts = [i for i, flag in enumerate(green) if flag and not green[i - 1]]
    assert len(starts) == 1, "expected exactly one cyclic green segment on the border"
    start = starts[0]

    run_len = 0
    while green[(start + run_len) % len(border)]:
        run_len += 1
        if run_len > len(border):
            raise AssertionError("border segment cannot cover the entire border")

    assert int(green.sum()) == run_len, "green border must be a single contiguous run"
    return border, start, run_len


def _tile_border(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    border, start, run_len = _find_single_border_segment(out)
    stride = 2 * run_len
    assert len(border) % stride == 0, "border perimeter must be divisible by 2n"
    for offset in range(0, len(border), stride):
        for k in range(run_len):
            r, c = border[(start + offset + k) % len(border)]
            out[r, c] = GREEN
    return out


def _reachable_background(work: np.ndarray) -> np.ndarray:
    h, w = work.shape
    walls = (work == GREEN) | (work == RED)
    seen = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    border = _border_cycle(work.shape)
    for r, c in border:
        if not walls[r, c] and not seen[r, c]:
            seen[r, c] = True
            queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _region_infos(work: np.ndarray) -> list[dict]:
    h, w = work.shape
    reachable = _reachable_background(work)
    interior = np.zeros((h, w), dtype=bool)
    interior[1:-1, 1:-1] = True
    enclosed = (work == BLACK) & interior & ~reachable

    infos = []
    for coords in _components(enclosed):
        border_neighbors = set()
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and work[nr, nc] == GREEN:
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        border_neighbors.add((nr, nc))
        infos.append(
            {
                "coords": coords,
                "anchor": min(coords),
                "contact_count": len(border_neighbors),
            }
        )

    infos.sort(key=lambda info: (-info["contact_count"], info["anchor"]))
    return infos


def _solve(grid: np.ndarray) -> np.ndarray:
    tiled = _tile_border(grid)
    infos = _region_infos(tiled)
    assert 1 <= len(infos) <= 3, "expected one to three enclosed regions after tiling"
    out = tiled.copy()
    for color, info in zip(FILL_COLORS, infos):
        for r, c in info["coords"]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the single green border segment around the frame and fill the enclosed pockets by border-contact rank.

    Input structure:
    - A black background grid has exactly one contiguous green segment on the outer border.
    - The interior contains one-cell-thick red horizontal and vertical line segments that do not touch the border.
    - The border otherwise stays black, and blue/yellow/orange appear only in the output fills.

    Rule:
    - Measure the green border segment length n along the clockwise perimeter cycle.
    - Repeat that segment around the border with period 2n, painting n green cells then leaving n black cells.
    - Treat the tiled green border and the red interior lines as walls, flood-filling background from every non-green border gap.
    - Each remaining enclosed background region is ranked by the number of distinct green border pixels touching it by 4-neighborhood; largest gets blue, second gets yellow, third gets orange, with row-major top-left tiebreaks.

    Key insight:
    - The border repetition creates the missing wall pieces; only after that cyclic tiling do the red interior barriers turn open pockets into sealed regions whose fill colors depend on how much green border they touch.
    """
    return _solve(np.asarray(grid, dtype=int))


def _buffer(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    runs = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _add_candidate(
    candidates: list[dict],
    shape: tuple[int, int],
    green_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    region_cells: set[tuple[int, int]],
    original_segment: set[tuple[int, int]],
    wall_lengths: tuple[int, int, int],
    side: str,
) -> None:
    if green_cells.issubset(original_segment):
        return
    if any(length < 3 or length > 8 for length in wall_lengths):
        return
    all_cells = green_cells | wall_cells | region_cells
    candidates.append(
        {
            "side": side,
            "count": len(green_cells),
            "greens": green_cells,
            "walls": wall_cells,
            "region": region_cells,
            "buffer": _buffer(all_cells, shape),
        }
    )


def _enumerate_candidates(
    shape: tuple[int, int],
    tiled_green: np.ndarray,
    original_segment: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []

    max_v_depth = min(8, h // 2)
    max_h_depth = min(8, w // 2)

    top_cols = [c for c in range(1, w - 1) if tiled_green[0, c]]
    for run_start, run_end in _runs(top_cols):
        run_len = run_end - run_start + 1
        for count in range(1, run_len + 1):
            for a in range(run_start, run_end - count + 2):
                b = a + count - 1
                if a < 2 or b > w - 3:
                    continue
                for depth in range(3, max_v_depth + 1):
                    green_cells = {(0, c) for c in range(a, b + 1)}
                    wall_cells = {(r, a - 1) for r in range(1, depth + 1)}
                    wall_cells |= {(r, b + 1) for r in range(1, depth + 1)}
                    wall_cells |= {(depth, c) for c in range(a - 1, b + 2)}
                    region_cells = {(r, c) for r in range(1, depth) for c in range(a, b + 1)}
                    _add_candidate(
                        candidates,
                        shape,
                        green_cells,
                        wall_cells,
                        region_cells,
                        original_segment,
                        (depth, depth, count + 2),
                        "top",
                    )

    bottom_cols = [c for c in range(1, w - 1) if tiled_green[h - 1, c]]
    for run_start, run_end in _runs(bottom_cols):
        run_len = run_end - run_start + 1
        for count in range(1, run_len + 1):
            for a in range(run_start, run_end - count + 2):
                b = a + count - 1
                if a < 2 or b > w - 3:
                    continue
                for depth in range(3, max_v_depth + 1):
                    top_row = h - 1 - depth
                    green_cells = {(h - 1, c) for c in range(a, b + 1)}
                    wall_cells = {(r, a - 1) for r in range(top_row, h - 1)}
                    wall_cells |= {(r, b + 1) for r in range(top_row, h - 1)}
                    wall_cells |= {(top_row, c) for c in range(a - 1, b + 2)}
                    region_cells = {(r, c) for r in range(top_row + 1, h - 1) for c in range(a, b + 1)}
                    _add_candidate(
                        candidates,
                        shape,
                        green_cells,
                        wall_cells,
                        region_cells,
                        original_segment,
                        (depth, depth, count + 2),
                        "bottom",
                    )

    left_rows = [r for r in range(1, h - 1) if tiled_green[r, 0]]
    for run_start, run_end in _runs(left_rows):
        run_len = run_end - run_start + 1
        for count in range(1, run_len + 1):
            for a in range(run_start, run_end - count + 2):
                b = a + count - 1
                if a < 2 or b > h - 3:
                    continue
                for depth in range(3, max_h_depth + 1):
                    green_cells = {(r, 0) for r in range(a, b + 1)}
                    wall_cells = {(a - 1, c) for c in range(1, depth + 1)}
                    wall_cells |= {(b + 1, c) for c in range(1, depth + 1)}
                    wall_cells |= {(r, depth) for r in range(a - 1, b + 2)}
                    region_cells = {(r, c) for r in range(a, b + 1) for c in range(1, depth)}
                    _add_candidate(
                        candidates,
                        shape,
                        green_cells,
                        wall_cells,
                        region_cells,
                        original_segment,
                        (depth, depth, count + 2),
                        "left",
                    )

    right_rows = [r for r in range(1, h - 1) if tiled_green[r, w - 1]]
    for run_start, run_end in _runs(right_rows):
        run_len = run_end - run_start + 1
        for count in range(1, run_len + 1):
            for a in range(run_start, run_end - count + 2):
                b = a + count - 1
                if a < 2 or b > h - 3:
                    continue
                for depth in range(3, max_h_depth + 1):
                    left_col = w - 1 - depth
                    green_cells = {(r, w - 1) for r in range(a, b + 1)}
                    wall_cells = {(a - 1, c) for c in range(left_col, w - 1)}
                    wall_cells |= {(b + 1, c) for c in range(left_col, w - 1)}
                    wall_cells |= {(r, left_col) for r in range(a - 1, b + 2)}
                    region_cells = {(r, c) for r in range(a, b + 1) for c in range(left_col + 1, w - 1)}
                    _add_candidate(
                        candidates,
                        shape,
                        green_cells,
                        wall_cells,
                        region_cells,
                        original_segment,
                        (depth, depth, count + 2),
                        "right",
                    )

    return candidates


def _choose_candidates(candidates: list[dict], region_count: int) -> list[dict] | None:
    groups: dict[int, list[dict]] = {}
    for candidate in candidates:
        groups.setdefault(candidate["count"], []).append(candidate)

    available_counts = [count for count, group in groups.items() if group]
    if len(available_counts) < region_count:
        return None

    desired_counts = random.sample(sorted(available_counts), region_count)
    for group in groups.values():
        random.shuffle(group)

    chosen: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    def backtrack(index: int) -> bool:
        if index == len(desired_counts):
            return True
        count = desired_counts[index]
        for candidate in groups[count]:
            if candidate["buffer"] & occupied:
                continue
            chosen.append(candidate)
            new_cells = candidate["buffer"] - occupied
            occupied.update(new_cells)
            if backtrack(index + 1):
                return True
            occupied.difference_update(new_cells)
            chosen.pop()
        return False

    if backtrack(0):
        return chosen
    return None


def _build_input(shape: tuple[int, int], start: int, run_len: int, chosen: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    border = _border_cycle(shape)
    for offset in range(run_len):
        r, c = border[(start + offset) % len(border)]
        grid[r, c] = GREEN
    for candidate in chosen:
        for r, c in candidate["walls"]:
            grid[r, c] = RED
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    perimeter = 2 * h + 2 * w - 4
    valid_lengths = [n for n in (3, 4, 5) if perimeter % (2 * n) == 0]
    if not valid_lengths:
        return None

    run_len = random.choice(valid_lengths)
    start = random.randrange(perimeter)
    border = _border_cycle((h, w))
    original_segment = {border[(start + k) % perimeter] for k in range(run_len)}

    tiled = np.zeros((h, w), dtype=int)
    for offset in range(0, perimeter, 2 * run_len):
        for k in range(run_len):
            r, c = border[(start + offset + k) % perimeter]
            tiled[r, c] = GREEN

    candidates = _enumerate_candidates((h, w), tiled == GREEN, original_segment)
    if not candidates:
        return None

    region_count = random.choice((2, 3))
    chosen = _choose_candidates(candidates, region_count)
    if chosen is None:
        return None

    inp = _build_input((h, w), start, run_len, chosen)

    if _region_infos(inp):
        return None

    tiled_infos = _region_infos(_tile_border(inp))
    if len(tiled_infos) != region_count:
        return None
    contact_counts = [info["contact_count"] for info in tiled_infos]
    if len(set(contact_counts)) != region_count:
        return None
    if any(count < 1 for count in contact_counts):
        return None

    out = _solve(inp)
    if np.array_equal(out, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid border-tiling flood-fill puzzle")
