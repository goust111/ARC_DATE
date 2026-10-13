from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MAROON = 9
SEED_COLORS = (1, 3, 4, 6, 7, 8)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# All templates use orthogonal offsets so the "near-side" red marker is unambiguous.
TEMPLATE_LIBRARY = (
    ((-2, 0), (-1, 0), (1, 0), (2, 0)),
    ((0, 1), (0, 2), (-1, 0), (1, 0)),
    ((-2, 0), (-1, 0), (0, -1), (0, 1)),
    ((1, 0), (2, 0), (0, 1), (0, 2)),
    ((-1, 0), (-2, 0), (0, -1), (0, -2)),
    ((0, -2), (0, -1), (0, 1), (0, 2)),
    ((1, 0), (2, 0), (0, -1), (0, 1)),
    ((0, 1), (0, 2), (1, 0), (2, 0)),
)

GEN_RETRIES = 400
PLACEMENT_RETRIES = 500


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _expand_points(points: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in points:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _mark_reserved(reserved: np.ndarray, points: set[tuple[int, int]], pad: int = 1) -> None:
    h, w = reserved.shape
    for r, c in _expand_points(points, h, w, pad=pad):
        reserved[r, c] = True


def _parse_legend_frame(
    grid: np.ndarray, points: list[tuple[int, int]]
) -> tuple[tuple[int, int, int, int], int, tuple[tuple[int, int], ...]] | None:
    r0, c0, r1, c1 = _bbox(points)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return None
    if len(points) != 2 * height + 2 * width - 4:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == GREY):
        return None
    if np.any(patch[~border] == GREY):
        return None

    inner = patch[1:-1, 1:-1]
    seed_cells = np.argwhere((inner != BLACK) & (inner != MAROON))
    if seed_cells.shape[0] != 1:
        return None
    if not np.any(inner == MAROON):
        return None

    seed_r, seed_c = map(int, seed_cells[0])
    seed_color = int(inner[seed_r, seed_c])
    maroon_cells = np.argwhere(inner == MAROON)
    offsets = tuple(
        sorted((int(rr) - seed_r, int(cc) - seed_c) for rr, cc in maroon_cells)
    )
    return (r0, c0, r1, c1), seed_color, offsets


def _extract_templates(grid: np.ndarray) -> tuple[dict[int, tuple[tuple[int, int], ...]], np.ndarray]:
    grey_mask = grid == GREY
    candidates = []
    for comp in _components(grey_mask):
        parsed = _parse_legend_frame(grid, comp)
        if parsed is not None:
            candidates.append(parsed)
    assert candidates, "expected at least one legend frame"

    grouped: dict[int, list[tuple[tuple[int, int, int, int], int, tuple[tuple[int, int], ...]]]] = {}
    for item in candidates:
        grouped.setdefault(item[0][1], []).append(item)
    best_left = min(
        c0 for c0, items in grouped.items() if len(items) == max(len(v) for v in grouped.values())
    )
    frames = sorted(grouped[best_left], key=lambda item: item[0][0])

    legend_mask = np.zeros_like(grid, dtype=bool)
    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    for (r0, c0, r1, c1), seed_color, offsets in frames:
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        assert seed_color not in templates, "legend colors must be unique"
        templates[seed_color] = offsets
    return templates, legend_mask


def _apply_growth(
    grid: np.ndarray, templates: dict[int, tuple[tuple[int, int], ...]], legend_mask: np.ndarray
) -> np.ndarray:
    h, w = grid.shape
    growth: dict[tuple[int, int], int] = {}
    red_marks: set[tuple[int, int]] = set()

    seed_cells = [
        (r, c, int(grid[r, c]))
        for r in range(h)
        for c in range(w)
        if not legend_mask[r, c] and int(grid[r, c]) in templates
    ]

    for r, c, color in sorted(seed_cells):
        for dr, dc in templates[color]:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if legend_mask[rr, cc]:
                continue
            cell = int(grid[rr, cc])
            if cell == GREY:
                step_r = _sign(dr)
                step_c = _sign(dc)
                mark_r = rr - step_r
                mark_c = cc - step_c
                if (
                    0 <= mark_r < h
                    and 0 <= mark_c < w
                    and not legend_mask[mark_r, mark_c]
                    and int(grid[mark_r, mark_c]) == BLACK
                ):
                    red_marks.add((mark_r, mark_c))
                continue
            if cell == BLACK:
                growth.setdefault((rr, cc), color)

    out = np.array(grid, copy=True)
    for (r, c), color in growth.items():
        out[r, c] = color
    for r, c in red_marks:
        out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Grow seed-colored templates from the legend and mark obstacle truncation with red."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    templates, legend_mask = _extract_templates(g)
    return _apply_growth(g, templates, legend_mask)


# --------------- generator ---------------


def _in_main(r: int, c: int, h: int, w: int, main_left: int) -> bool:
    return 0 <= r < h and main_left <= c < w


def _rect_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _draw_legend(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    left: int,
    top: int,
    box_size: int,
    color: int,
    offsets: tuple[tuple[int, int], ...],
) -> None:
    bottom = top + box_size - 1
    right = left + box_size - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    legend_mask[top : bottom + 1, left : right + 1] = True

    center_r = top + box_size // 2
    center_c = left + box_size // 2
    grid[center_r, center_c] = color
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = MAROON


def _targets_for(seed: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    r, c = seed
    return [(r + dr, c + dc) for dr, dc in offsets]


def _overlaps_reserved(reserved: np.ndarray, points: set[tuple[int, int]]) -> bool:
    return any(reserved[r, c] for r, c in points)


def _free_seed(
    grid: np.ndarray,
    reserved: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
    main_left: int,
) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(0, h - 1)
        c = random.randint(main_left, w - 1)
        seed = (r, c)
        if grid[seed] != BLACK or reserved[seed]:
            continue

        targets = _targets_for(seed, offsets)
        cells = {seed}
        ok = True
        for rr, cc in targets:
            if not _in_main(rr, cc, h, w, main_left):
                ok = False
                break
            if grid[rr, cc] != BLACK:
                ok = False
                break
            cells.add((rr, cc))
        if not ok:
            continue
        footprint = _expand_points(cells, h, w, pad=1)
        if _overlaps_reserved(reserved, footprint):
            continue
        return seed, set(), cells
    return None


def _edge_seed(
    grid: np.ndarray,
    reserved: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
    main_left: int,
) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(0, h - 1)
        c = random.randint(main_left, w - 1)
        seed = (r, c)
        if grid[seed] != BLACK or reserved[seed]:
            continue
        if not (r <= 1 or r >= h - 2 or c >= w - 2):
            continue

        targets = _targets_for(seed, offsets)
        in_bounds = []
        cells = {seed}
        out_count = 0
        ok = True
        for rr, cc in targets:
            if not (0 <= rr < h and 0 <= cc < w):
                out_count += 1
                continue
            if cc < main_left:
                ok = False
                break
            if grid[rr, cc] != BLACK:
                ok = False
                break
            in_bounds.append((rr, cc))
            cells.add((rr, cc))
        if not ok or out_count == 0 or len(in_bounds) < 2:
            continue
        footprint = _expand_points(cells, h, w, pad=1)
        if _overlaps_reserved(reserved, footprint):
            continue
        return seed, set(), cells
    return None


def _rect_candidates(
    target: tuple[int, int], offset: tuple[int, int], h: int, w: int, main_left: int
) -> list[set[tuple[int, int]]]:
    tr, tc = target
    dr, dc = offset
    candidates: list[set[tuple[int, int]]] = []

    if dr == 0 and dc != 0:
        for height in (1, 2, 3):
            for width in (1, 2, 3):
                if height * width < 2:
                    continue
                if dc > 0:
                    c0, c1 = tc, tc + width - 1
                else:
                    c0, c1 = tc - width + 1, tc
                for top in range(tr - height + 1, tr + 1):
                    bottom = top + height - 1
                    if top < 0 or bottom >= h or c0 < main_left or c1 >= w:
                        continue
                    candidates.append(_rect_cells(top, c0, bottom, c1))
    elif dc == 0 and dr != 0:
        for height in (1, 2, 3):
            for width in (1, 2, 3):
                if height * width < 2:
                    continue
                if dr > 0:
                    r0, r1 = tr, tr + height - 1
                else:
                    r0, r1 = tr - height + 1, tr
                for left in range(tc - width + 1, tc + 1):
                    right = left + width - 1
                    if r0 < 0 or r1 >= h or left < main_left or right >= w:
                        continue
                    candidates.append(_rect_cells(r0, left, r1, right))
    return candidates


def _blocked_seed(
    grid: np.ndarray,
    reserved: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
    main_left: int,
) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    long_offsets = [offset for offset in offsets if max(abs(offset[0]), abs(offset[1])) >= 2]
    assert long_offsets

    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(0, h - 1)
        c = random.randint(main_left, w - 1)
        seed = (r, c)
        if grid[seed] != BLACK or reserved[seed]:
            continue

        targets = _targets_for(seed, offsets)
        if not all(_in_main(rr, cc, h, w, main_left) for rr, cc in targets):
            continue

        blocked_offset = random.choice(long_offsets)
        block_target = (r + blocked_offset[0], c + blocked_offset[1])
        mark = (
            block_target[0] - _sign(blocked_offset[0]),
            block_target[1] - _sign(blocked_offset[1]),
        )
        if not _in_main(mark[0], mark[1], h, w, main_left):
            continue
        if grid[mark] != BLACK or reserved[mark]:
            continue

        other_targets = {pos for pos in targets if pos != block_target}
        if any(grid[pos] != BLACK for pos in other_targets):
            continue

        rectangles = _rect_candidates(block_target, blocked_offset, h, w, main_left)
        random.shuffle(rectangles)
        for rect in rectangles:
            if _overlaps_reserved(reserved, rect):
                continue
            if any(grid[pos] != BLACK for pos in rect):
                continue
            if rect & other_targets:
                continue
            cells = {seed, mark, *targets} | rect
            footprint = _expand_points(cells, h, w, pad=1)
            if _overlaps_reserved(reserved, footprint):
                continue
            return seed, rect, cells
    return None


def _extra_obstacles(grid: np.ndarray, reserved: np.ndarray, main_left: int) -> None:
    h, w = grid.shape
    for _ in range(random.randint(0, 2)):
        for _ in range(PLACEMENT_RETRIES):
            height = random.randint(1, 3)
            width = random.randint(1, 4)
            if height * width < 2:
                continue
            r0 = random.randint(0, h - height)
            c0 = random.randint(main_left, w - width)
            cells = _rect_cells(r0, c0, r0 + height - 1, c0 + width - 1)
            footprint = _expand_points(cells, h, w, pad=1)
            if _overlaps_reserved(reserved, footprint):
                continue
            if any(grid[pos] != BLACK for pos in cells):
                continue
            for rr, cc in cells:
                grid[rr, cc] = GREY
            _mark_reserved(reserved, cells, pad=1)
            break


def _place_seed_and_obstacle(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    seed: tuple[int, int],
    obstacle: set[tuple[int, int]],
    cells: set[tuple[int, int]],
) -> None:
    sr, sc = seed
    grid[sr, sc] = color
    for r, c in obstacle:
        grid[r, c] = GREY
    _mark_reserved(reserved, cells, pad=1)


def _generate_attempt() -> np.ndarray | None:
    legend_count = random.randint(2, 4)
    box_size = random.choice((7, 9))
    left_margin = random.randint(0, 2)
    top_margin = random.randint(1, 3)
    between_legend = random.randint(1, 2)
    main_gap = random.randint(2, 5)
    bottom_margin = random.randint(1, 3)

    legend_height = top_margin + legend_count * box_size + (legend_count - 1) * between_legend + bottom_margin
    h_low = max(18, legend_height)
    h_high = max(h_low, 28)
    h = random.randint(h_low, h_high)

    legend_right = left_margin + box_size - 1
    main_left = legend_right + 1 + main_gap
    w_low = max(main_left + 14, 24)
    w_high = max(w_low, 36)
    w = random.randint(w_low, w_high)

    grid = np.zeros((h, w), dtype=int)
    legend_mask = np.zeros((h, w), dtype=bool)

    colors = random.sample(SEED_COLORS, legend_count)
    templates = random.sample(TEMPLATE_LIBRARY, legend_count)
    template_by_color = dict(zip(colors, templates, strict=True))

    for idx, color in enumerate(colors):
        top = top_margin + idx * (box_size + between_legend)
        _draw_legend(grid, legend_mask, left_margin, top, box_size, color, template_by_color[color])

    reserved = np.zeros((h, w), dtype=bool)
    _mark_reserved(reserved, set(map(tuple, np.argwhere(legend_mask))), pad=1)

    seed_count = random.randint(max(3, legend_count), min(6, legend_count + 2))
    seed_colors = list(colors)
    while len(seed_colors) < seed_count:
        seed_colors.append(random.choice(colors))
    random.shuffle(seed_colors)

    modes = ["blocked", "edge", "free"]
    while len(modes) < seed_count:
        modes.append(random.choice(("blocked", "edge", "free")))
    random.shuffle(modes)

    tasks = list(zip(seed_colors, modes, strict=True))
    tasks.sort(key=lambda item: {"blocked": 0, "edge": 1, "free": 2}[item[1]])

    for color, mode in tasks:
        offsets = template_by_color[color]
        if mode == "blocked":
            placement = _blocked_seed(grid, reserved, offsets, main_left)
        elif mode == "edge":
            placement = _edge_seed(grid, reserved, offsets, main_left)
        else:
            placement = _free_seed(grid, reserved, offsets, main_left)
        if placement is None:
            return None
        seed, obstacle, cells = placement
        _place_seed_and_obstacle(grid, reserved, color, seed, obstacle, cells)

    _extra_obstacles(grid, reserved, main_left)

    out = main(grid)
    if not np.any(out == RED):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
