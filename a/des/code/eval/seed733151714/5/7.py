from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
COLOR_POOL = (1, 2, 3, 5, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GAP_SIDES = ("top", "right", "bottom", "left")
CW_SIDES = ("top", "right")
CCW_SIDES = ("bottom", "left")
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_left_legend_row(row: np.ndarray) -> tuple[int, int] | None:
    nonblack = np.flatnonzero(row != BLACK)
    if len(nonblack) == 0:
        return None
    if nonblack[0] != 0:
        return None
    seg_len = len(nonblack)
    if seg_len < 2 or seg_len > 4:
        return None
    if not np.array_equal(nonblack, np.arange(seg_len)):
        return None
    if np.any(row[seg_len:] != BLACK):
        return None
    color = int(row[0])
    if color in (BLACK, YELLOW):
        return None
    if not np.all(row[1:seg_len] == YELLOW):
        return None
    return color, seg_len - 1


def _parse_right_legend_row(row: np.ndarray) -> tuple[int, int] | None:
    w = row.shape[0]
    nonblack = np.flatnonzero(row != BLACK)
    if len(nonblack) == 0:
        return None
    seg_len = len(nonblack)
    if seg_len < 2 or seg_len > 4:
        return None
    if nonblack[-1] != w - 1:
        return None
    if not np.array_equal(nonblack, np.arange(w - seg_len, w)):
        return None
    if np.any(row[: w - seg_len] != BLACK):
        return None
    color = int(row[-1])
    if color in (BLACK, YELLOW):
        return None
    if not np.all(row[w - seg_len : w - 1] == YELLOW):
        return None
    return color, seg_len - 1


def _collect_legend_lines(g: np.ndarray, from_top: bool) -> tuple[str, list[int], list[tuple[int, int]]] | None:
    h = g.shape[0]
    rows = range(h) if from_top else range(h - 1, -1, -1)
    side: str | None = None
    row_ids: list[int] = []
    lines: list[tuple[int, int]] = []
    for idx, r in enumerate(rows):
        row = g[r]
        left = _parse_left_legend_row(row)
        right = _parse_right_legend_row(row)
        if left is None and right is None:
            if idx == 0:
                return None
            if lines:
                break
            continue
        if left is not None and right is not None:
            return None
        cur_side = "left" if left is not None else "right"
        cur_line = left if left is not None else right
        assert cur_line is not None
        if side is None:
            side = cur_side
        elif side != cur_side:
            return None
        row_ids.append(int(r))
        lines.append((int(cur_line[0]), int(cur_line[1])))
    if side is None or len(lines) < 2:
        return None
    return side, row_ids, lines


def _read_legend(g: np.ndarray) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    top = _collect_legend_lines(g, from_top=True)
    bottom = _collect_legend_lines(g, from_top=False)
    candidates = [cand for cand in (top, bottom) if cand is not None]
    assert len(candidates) == 1, "expected exactly one corner legend"
    side, rows, lines = candidates[0]
    mapping: dict[int, int] = {}
    for color, count in lines:
        assert color not in mapping, "legend colors must be unique"
        mapping[int(color)] = int(count)
    max_count = max(count for _color, count in lines)
    r0 = min(rows)
    r1 = max(rows)
    if side == "left":
        c0 = 0
        c1 = max_count
    else:
        c0 = g.shape[1] - max_count - 1
        c1 = g.shape[1] - 1
    return mapping, (int(r0), int(c0), int(r1), int(c1))


def _perimeter_coords(r0: int, c0: int, size: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _gap_side(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str | None:
    r, c = gap
    sides: list[str] = []
    if r == r0 and c0 < c < c1:
        sides.append("top")
    if c == c1 and r0 < r < r1:
        sides.append("right")
    if r == r1 and c0 < c < c1:
        sides.append("bottom")
    if c == c0 and r0 < r < r1:
        sides.append("left")
    if len(sides) != 1:
        return None
    return sides[0]


def _find_turntables(g: np.ndarray) -> list[dict[str, int | bool]]:
    turntables: list[dict[str, int | bool]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, YELLOW):
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            size = r1 - r0 + 1
            if size != c1 - c0 + 1 or not (5 <= size <= 7):
                continue
            if len(comp) != 4 * size - 5:
                continue
            coord_set = set(comp)
            perimeter = _perimeter_coords(r0, c0, size)
            if not coord_set.issubset(perimeter):
                continue
            missing = perimeter - coord_set
            if len(missing) != 1:
                continue
            gap = next(iter(missing))
            side = _gap_side(gap, r0, c0, r1, c1)
            if side is None:
                continue
            turntables.append(
                {
                    "color": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "size": int(size),
                    "clockwise": side in CW_SIDES,
                }
            )
    assert turntables, "expected at least one turntable"
    return turntables


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read color-coded rotation counts from a corner legend, use each square ring's
    gap side to decide clockwise vs counterclockwise, rotate only the inner
    pattern, and erase the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    mapping, legend_bbox = _read_legend(g)
    turntables = _find_turntables(g)

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BLACK

    for turntable in turntables:
        color = int(turntable["color"])
        assert color in mapping, "every turntable color must appear in the legend"
        turns = int(mapping[color]) % 4
        r0 = int(turntable["r0"])
        c0 = int(turntable["c0"])
        size = int(turntable["size"])
        inner = g[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
        if bool(turntable["clockwise"]):
            rotated = np.rot90(inner, -turns)
        else:
            rotated = np.rot90(inner, turns)
        out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = rotated

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
OBJECT_COUNT_RANGE = (2, 4)
TURN_SIZE_RANGE = (5, 7)
PATTERN_RETRIES = 200
PLACEMENT_RETRIES = 300
GENERATION_RETRIES = 500
PAD = 1


def _sample_steps(count: int) -> list[int]:
    base = [1, 2, 3]
    random.shuffle(base)
    if count == 2:
        steps = base[:2]
    elif count == 3:
        steps = base[:]
    else:
        steps = base + [random.choice(base)]
    random.shuffle(steps)
    return steps


def _random_patch(size: int, forbidden: set[int]) -> np.ndarray:
    cells = [(r, c) for r in range(size) for c in range(size)]
    usable = [color for color in COLOR_POOL if color not in forbidden]
    for _ in range(PATTERN_RETRIES):
        patch = np.zeros((size, size), dtype=int)
        fill_count = random.randint(max(3, size), size * size - 1)
        chosen = random.sample(cells, fill_count)
        color_count = random.randint(2, min(4, fill_count, len(usable)))
        colors = random.sample(usable, color_count)
        assignments = colors[:]
        assignments.extend(random.choice(colors) for _ in range(fill_count - color_count))
        random.shuffle(assignments)
        for (r, c), color in zip(chosen, assignments):
            patch[r, c] = int(color)
        if len(set(int(v) for v in patch[patch != BLACK])) < 2:
            continue
        if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
            continue
        return patch
    raise RuntimeError("failed to generate an asymmetric inner pattern")


def _gap_cell(side: str, size: int) -> tuple[int, int]:
    idx = random.randint(1, size - 2)
    if side == "top":
        return 0, idx
    if side == "right":
        return idx, size - 1
    if side == "bottom":
        return size - 1, idx
    return idx, 0


def _place_turntable(
    g: np.ndarray,
    r0: int,
    c0: int,
    size: int,
    ring_color: int,
    side: str,
    inner: np.ndarray,
) -> None:
    g[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = inner
    for r, c in _perimeter_coords(r0, c0, size):
        g[r, c] = int(ring_color)
    gr, gc = _gap_cell(side, size)
    g[r0 + gr, c0 + gc] = BLACK


def _can_place(blocked: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = blocked.shape
    if r0 < 1 or c0 < 1 or r0 + size >= h or c0 + size >= w:
        return False
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(h, r0 + size + PAD)
    cc1 = min(w, c0 + size + PAD)
    return not bool(np.any(blocked[rr0:rr1, cc0:cc1]))


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, size: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(h, r0 + size + PAD)
    cc1 = min(w, c0 + size + PAD)
    blocked[rr0:rr1, cc0:cc1] = True


def _legend_rows(count: int, corner: str, h: int) -> list[int]:
    if corner.startswith("top"):
        return list(range(count))
    return list(range(h - count, h))


def _paint_legend(g: np.ndarray, rows: list[int], entries: list[tuple[int, int]], corner: str) -> None:
    h, w = g.shape
    assert len(rows) == len(entries)
    for r, (color, steps) in zip(rows, entries):
        if corner.endswith("left"):
            g[r, 0] = int(color)
            g[r, 1 : 1 + steps] = YELLOW
        else:
            g[r, w - 1] = int(color)
            g[r, w - 1 - steps : w - 1] = YELLOW
    if corner.startswith("top"):
        guard_row = len(rows)
        if guard_row < h:
            g[guard_row, :] = np.where(g[guard_row, :] == BLACK, BLACK, g[guard_row, :])
    else:
        guard_row = h - len(rows) - 1
        if guard_row >= 0:
            g[guard_row, :] = np.where(g[guard_row, :] == BLACK, BLACK, g[guard_row, :])


def _make_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    sizes = [random.randint(*TURN_SIZE_RANGE) for _ in range(object_count)]
    legend_corner = random.choice(LEGEND_CORNERS)

    min_h = max(GRID_H_RANGE[0], object_count + max(sizes) + 4)
    min_w = max(GRID_W_RANGE[0], max(sizes) + 4)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(min_w, GRID_W_RANGE[1])
    g = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    rows = _legend_rows(object_count, legend_corner, h)
    if legend_corner.startswith("top"):
        blocked[: object_count + 1, :] = True
    else:
        blocked[h - object_count - 1 :, :] = True

    ring_colors = random.sample(COLOR_POOL, object_count)
    steps = _sample_steps(object_count)
    gap_sides = [random.choice(CW_SIDES), random.choice(CCW_SIDES)]
    gap_sides.extend(random.choice(GAP_SIDES) for _ in range(object_count - 2))
    random.shuffle(gap_sides)

    order = list(range(object_count))
    order.sort(key=lambda idx: sizes[idx], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * object_count
    for idx in order:
        size = sizes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - size):
            for c0 in range(1, w - size):
                if _can_place(blocked, r0, c0, size):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (int(r0), int(c0))
        _mark_blocked(blocked, int(r0), int(c0), size)

    for idx in range(object_count):
        placement = placements[idx]
        assert placement is not None
        r0, c0 = placement
        ring_color = int(ring_colors[idx])
        inner_size = sizes[idx] - 2
        inner = _random_patch(inner_size, forbidden={ring_color, YELLOW})
        _place_turntable(g, r0, c0, sizes[idx], ring_color, gap_sides[idx], inner)

    legend_entries = list(zip(ring_colors, steps))
    random.shuffle(legend_entries)
    _paint_legend(g, rows, legend_entries, legend_corner)

    try:
        out = main(g)
        mapping, legend_bbox = _read_legend(g)
        turntables = _find_turntables(g)
    except AssertionError:
        return None

    if len(mapping) != object_count or len(turntables) != object_count:
        return None
    if np.array_equal(out, g):
        return None
    lr0, lc0, lr1, lc1 = legend_bbox
    if np.any(out[lr0 : lr1 + 1, lc0 : lc1 + 1] != BLACK):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
