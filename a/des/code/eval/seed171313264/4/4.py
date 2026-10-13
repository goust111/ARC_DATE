from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_RETRIES = 400
GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (22, 31)
FRAME_COUNT_CHOICES = (2, 3)
INNER_H_RANGE = (3, 5)
INNER_W_RANGE = (3, 5)
PLACEMENT_TRIES = 250
PALETTE = tuple(range(1, 10))
DIRECTIONS = ("top", "right", "bottom", "left")

_DIRECTION_BAG: list[str] = []
_GAP_BAG: list[int] = []


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _draw_balanced_direction() -> str:
    global _DIRECTION_BAG
    if not _DIRECTION_BAG:
        _DIRECTION_BAG = random.sample(list(DIRECTIONS), len(DIRECTIONS))
    return _DIRECTION_BAG.pop()


def _draw_balanced_gap_count() -> int:
    global _GAP_BAG
    if not _GAP_BAG:
        _GAP_BAG = random.sample([1, 2, 3], 3)
    return _GAP_BAG.pop()


def _side_gap_positions(line: np.ndarray, color: int) -> list[int] | None:
    gaps: list[int] = []
    for idx, value in enumerate(line):
        value = int(value)
        if value == BG:
            gaps.append(idx)
        elif value != int(color):
            return None
    return gaps


def _extract_frame(
    g: np.ndarray, coords: list[tuple[int, int]], color: int
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 5 or outer_w < 5:
        return None

    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    if any(int(g[r, c]) != int(color) for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
        return None

    top = _side_gap_positions(g[r0, c0 + 1 : c1], color)
    bottom = _side_gap_positions(g[r1, c0 + 1 : c1], color)
    left = _side_gap_positions(g[r0 + 1 : r1, c0], color)
    right = _side_gap_positions(g[r0 + 1 : r1, c1], color)
    if any(side is None for side in (top, bottom, left, right)):
        return None

    assert top is not None and bottom is not None and left is not None and right is not None
    side_to_gaps = {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }
    nonempty = [(side, gaps) for side, gaps in side_to_gaps.items() if gaps]
    if len(nonempty) != 1:
        return None

    side, gaps = nonempty[0]
    if not (1 <= len(gaps) <= 3):
        return None

    expected_area = 2 * (outer_h + outer_w) - 4 - len(gaps)
    if len(coords) != expected_area:
        return None

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    colors_inside = {int(v) for v in np.unique(inner) if int(v) not in (BG, int(color))}
    if not colors_inside:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "side": side,
        "gaps": [int(x) for x in gaps],
    }


def _find_frames(g: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in (int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components_mask(g == int(color)):
            frame = _extract_frame(g, comp, color)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: tuple(int(x) for x in item["bbox"]))  # type: ignore[arg-type]
    return frames


def _repeat(values: np.ndarray, length: int) -> np.ndarray:
    return np.resize(np.asarray(values, dtype=int), length)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrude interior slices outward through one-sided border gaps.

    Input structure:
    - bg = black.
    - Several hollow rectangular frames share one border color.
    - Each frame has one monochrome interior sprite and 1-3 single-cell gaps on exactly one side.

    Rule:
    - The side containing the gaps determines the extrusion direction.
    - For each gap, select the interior row/column aligned to that gap.
    - Repeat that slice outward for (interior extent along the extrusion axis) * (number of gaps).
    - Keep the original frame, pattern, and open gaps unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        side = str(frame["side"])
        gaps = [int(x) for x in frame["gaps"]]  # type: ignore[arg-type]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = map(int, inner.shape)

        if side in ("top", "bottom"):
            length = inner_h * len(gaps)
            for gap in gaps:
                inner_c = int(gap)
                values = inner[:, inner_c]
                if side == "bottom":
                    values = values[::-1]
                repeated = _repeat(values, length)
                for step, value in enumerate(repeated, start=1):
                    rr = r0 - step if side == "top" else r1 + step
                    if 0 <= rr < g.shape[0]:
                        out[rr, c0 + 1 + inner_c] = int(value)
        else:
            length = inner_w * len(gaps)
            for gap in gaps:
                inner_r = int(gap)
                values = inner[inner_r, :]
                if side == "right":
                    values = values[::-1]
                repeated = _repeat(values, length)
                for step, value in enumerate(repeated, start=1):
                    cc = c0 - step if side == "left" else c1 + step
                    if 0 <= cc < g.shape[1]:
                        out[r0 + 1 + inner_r, cc] = int(value)

    return out


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [a]
    while (r, c) != (tr, tc):
        steps: list[tuple[int, int]] = []
        if r < tr:
            steps.append((r + 1, c))
        elif r > tr:
            steps.append((r - 1, c))
        if c < tc:
            steps.append((r, c + 1))
        elif c > tc:
            steps.append((r, c - 1))
        r, c = random.choice(steps)
        path.append((r, c))
    return path


def _seed_cells(inner_h: int, inner_w: int, side: str, gaps: list[int]) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, gap) for gap in gaps]
    if side == "bottom":
        return [(inner_h - 1, gap) for gap in gaps]
    if side == "left":
        return [(gap, 0) for gap in gaps]
    return [(gap, inner_w - 1) for gap in gaps]


def _grow_pattern(inner_h: int, inner_w: int, side: str, gaps: list[int]) -> np.ndarray:
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    seeds = _seed_cells(inner_h, inner_w, side, sorted(gaps))
    for seed in seeds:
        mask[seed] = True
    for start, end in zip(seeds, seeds[1:], strict=False):
        for r, c in _manhattan_path(start, end):
            mask[r, c] = True

    area = inner_h * inner_w
    min_target = max(int(mask.sum()), (area + 2) // 3)
    max_target = max(min_target, min(area - 1, (2 * area) // 3 + 1))
    target = random.randint(min_target, max_target)

    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in zip(*np.where(mask), strict=False):
            r = int(r)
            c = int(c)
            if r > 0 and not mask[r - 1, c]:
                frontier.append((r - 1, c))
            if r + 1 < inner_h and not mask[r + 1, c]:
                frontier.append((r + 1, c))
            if c > 0 and not mask[r, c - 1]:
                frontier.append((r, c - 1))
            if c + 1 < inner_w and not mask[r, c + 1]:
                frontier.append((r, c + 1))
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
    return mask


def _spec_blank_side_bbox(
    spec: dict[str, object], r0: int, c0: int
) -> tuple[int, int, int, int]:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    length = int(spec["length"])
    side = str(spec["side"])
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    if side == "top":
        return r0 - length, c0, r0 - 1, c1
    if side == "bottom":
        return r1 + 1, c0, r1 + length, c1
    if side == "left":
        return r0, c0 - length, r1, c0 - 1
    return r0, c1 + 1, r1, c1 + length


def _guard_bbox(spec: dict[str, object], r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    frame_r1 = r0 + outer_h - 1
    frame_c1 = c0 + outer_w - 1
    br0, bc0, br1, bc1 = _spec_blank_side_bbox(spec, r0, c0)

    rr0 = max(0, min(r0, br0) - 1)
    cc0 = max(0, min(c0, bc0) - 1)
    rr1 = min(h - 1, max(frame_r1, br1) + 1)
    cc1 = min(w - 1, max(frame_c1, bc1) + 1)
    return rr0, cc0, rr1, cc1


def _placement_ranges(spec: dict[str, object], h: int, w: int) -> tuple[range, range] | None:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    length = int(spec["length"])
    side = str(spec["side"])

    if side == "top":
        r_lo, r_hi = length, h - outer_h
        c_lo, c_hi = 0, w - outer_w
    elif side == "bottom":
        r_lo, r_hi = 0, h - outer_h - length
        c_lo, c_hi = 0, w - outer_w
    elif side == "left":
        r_lo, r_hi = 0, h - outer_h
        c_lo, c_hi = length, w - outer_w
    else:
        r_lo, r_hi = 0, h - outer_h
        c_lo, c_hi = 0, w - outer_w - length

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return range(r_lo, r_hi + 1), range(c_lo, c_hi + 1)


def _candidate_positions(spec: dict[str, object], h: int, w: int) -> list[tuple[int, int]]:
    ranges = _placement_ranges(spec, h, w)
    if ranges is None:
        return []
    row_range, col_range = ranges
    positions = [(r, c) for r in row_range for c in col_range]
    random.shuffle(positions)
    return positions[:PLACEMENT_TRIES]


def _draw_frame(
    grid: np.ndarray,
    border_color: int,
    pattern_color: int,
    spec: dict[str, object],
    r0: int,
    c0: int,
) -> None:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    side = str(spec["side"])
    gaps = [int(x) for x in spec["gaps"]]  # type: ignore[arg-type]
    pattern_mask = np.asarray(spec["pattern_mask"], dtype=bool)

    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color

    if side == "top":
        for gap in gaps:
            grid[r0, c0 + 1 + gap] = BG
    elif side == "bottom":
        for gap in gaps:
            grid[r1, c0 + 1 + gap] = BG
    elif side == "left":
        for gap in gaps:
            grid[r0 + 1 + gap, c0] = BG
    else:
        for gap in gaps:
            grid[r0 + 1 + gap, c1] = BG

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner[pattern_mask] = pattern_color


def _make_frame_spec() -> dict[str, object]:
    side = _draw_balanced_direction()
    gap_count = _draw_balanced_gap_count()

    while True:
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        axis_len = inner_w if side in ("top", "bottom") else inner_h
        if axis_len >= gap_count:
            break

    gaps = sorted(random.sample(range(axis_len), gap_count))
    pattern_mask = _grow_pattern(inner_h, inner_w, side, gaps)
    length = (inner_h if side in ("top", "bottom") else inner_w) * gap_count

    return {
        "side": side,
        "gaps": gaps,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
        "length": length,
        "pattern_mask": pattern_mask,
        "weight": (inner_h + 2) * (inner_w + 2) + length * max(inner_h + 2, inner_w + 2),
    }


def _build_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_frames = random.choice(FRAME_COUNT_CHOICES)

    border_color = random.choice(PALETTE)
    pattern_colors = random.sample([c for c in PALETTE if c != border_color], n_frames)
    specs = [_make_frame_spec() for _ in range(n_frames)]
    specs.sort(key=lambda item: int(item["weight"]), reverse=True)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[dict[str, object], int, int]] = []

    for spec in specs:
        placed = False
        for r0, c0 in _candidate_positions(spec, h, w):
            rr0, cc0, rr1, cc1 = _guard_bbox(spec, r0, c0, h, w)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placements.append((spec, r0, c0))
            placed = True
            break
        if not placed:
            return None

    for (spec, r0, c0), pattern_color in zip(placements, pattern_colors, strict=False):
        _draw_frame(grid, border_color, int(pattern_color), spec, r0, c0)

    frames = _find_frames(grid)
    if len(frames) != n_frames:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
