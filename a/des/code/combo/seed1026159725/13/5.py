import random
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _count_bg_components(patch: np.ndarray, bg: int) -> int:
    return len(_components(patch == bg))


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    assert h >= 3 and w >= 2
    specs = [
        ([0, 1, 2], 0, 1),
        ([h - 3, h - 2, h - 1], 0, 1),
        ([0, 1, 2], w - 1, w - 2),
        ([h - 3, h - 2, h - 1], w - 1, w - 2),
    ]

    candidates = []
    for rows, col, inward_col in specs:
        vals = [int(grid[r, col]) for r in rows]
        if any(v == bg for v in vals) or len(set(vals)) != 3:
            continue
        if any(int(grid[r, inward_col]) != bg for r in rows):
            continue
        before = rows[0] - 1
        after = rows[-1] + 1
        if 0 <= before < h and int(grid[before, col]) != bg:
            continue
        if 0 <= after < h and int(grid[after, col]) != bg:
            continue
        candidates.append((rows, col, vals))

    assert len(candidates) == 1, "expected exactly one 3-pixel corner legend"
    rows, col, vals = candidates[0]
    mapping = {1: vals[0], 2: vals[1], 3: vals[2]}
    coords = {(r, col) for r in rows}
    return mapping, coords


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend, legend_cells = _detect_legend(g, bg)

    work = g.copy()
    out = g.copy()
    for r, c in legend_cells:
        work[r, c] = bg
        out[r, c] = bg

    for comp in _components(work != bg):
        r0, c0, r1, c1 = _bbox(comp)
        patch = work[r0 : r1 + 1, c0 : c1 + 1]
        missing = _count_bg_components(patch, bg)
        if missing == 0:
            continue
        assert missing in legend, "concave objects must have 1..3 gaps"
        new_color = legend[missing]
        for r, c in comp:
            out[r, c] = new_color

    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    transformed = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _shape_convex() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    return np.ones((h, w), dtype=bool)


def _shape_one_gap() -> np.ndarray:
    style = random.choice(["bite", "L", "U"])
    if style == "bite":
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        mask = np.ones((h, w), dtype=bool)
        remove_r = random.randrange(h)
        remove_c = random.randrange(w)
        mask[remove_r, remove_c] = False
    elif style == "L":
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[:, 0] = True
    else:
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    mask = _random_transform(mask)
    assert len(_components(mask)) == 1
    assert _count_bg_components(mask.astype(int) ^ 1, 1) == 1
    return mask


def _shape_two_gaps() -> np.ndarray:
    if random.random() < 0.5:
        h = random.randint(4, 7)
        w = random.randint(5, 8)
        mid = w // 2
        mask = np.zeros((h, w), dtype=bool)
        mask[:, mid] = True
        mask[0, : mid + 1] = True
        mask[-1, mid:] = True
    else:
        h = random.randint(5, 8)
        w = random.randint(4, 7)
        mid = h // 2
        mask = np.zeros((h, w), dtype=bool)
        mask[mid, :] = True
        mask[: mid + 1, 0] = True
        mask[mid:, -1] = True
    mask = _random_transform(mask)
    assert len(_components(mask)) == 1
    assert _count_bg_components(mask.astype(int) ^ 1, 1) == 2
    return mask


def _shape_three_gaps() -> np.ndarray:
    if random.random() < 0.5:
        h = random.randint(4, 7)
        w = random.randint(5, 8)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 1] = True
        mask[:, -2] = True
        mask[0, 1:-1] = True
        mask[-1, 0] = True
        mask[-1, -1] = True
    else:
        h = random.randint(5, 8)
        w = random.randint(4, 7)
        mask = np.zeros((h, w), dtype=bool)
        mask[1, :] = True
        mask[-2, :] = True
        mask[1:-1, 0] = True
        mask[0, -1] = True
        mask[-1, -1] = True
    mask = _random_transform(mask)
    assert len(_components(mask)) == 1
    assert _count_bg_components(mask.astype(int) ^ 1, 1) == 3
    return mask


def _make_shape(gap_count: int) -> np.ndarray:
    if gap_count == 0:
        return _shape_convex()
    if gap_count == 1:
        return _shape_one_gap()
    if gap_count == 2:
        return _shape_two_gaps()
    if gap_count == 3:
        return _shape_three_gaps()
    raise ValueError(gap_count)


def _legend_spec(h: int, w: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    corner = random.choice(["tl", "bl", "tr", "br"])
    if corner == "tl":
        coords = [(0, 0), (1, 0), (2, 0)]
        reserved = {(r, c) for r in range(0, 4) for c in range(0, 3)}
    elif corner == "bl":
        coords = [(h - 3, 0), (h - 2, 0), (h - 1, 0)]
        reserved = {(r, c) for r in range(h - 4, h) for c in range(0, 3)}
    elif corner == "tr":
        coords = [(0, w - 1), (1, w - 1), (2, w - 1)]
        reserved = {(r, c) for r in range(0, 4) for c in range(w - 3, w)}
    else:
        coords = [(h - 3, w - 1), (h - 2, w - 1), (h - 1, w - 1)]
        reserved = {(r, c) for r in range(h - 4, h) for c in range(w - 3, w)}
    return coords, reserved


def _can_place(occupied: np.ndarray, r0: int, c0: int, shape: np.ndarray) -> bool:
    h, w = occupied.shape
    sh, sw = shape.shape
    r1 = r0 + sh
    c1 = c0 + sw
    if r0 < 1 or c0 < 1 or r1 > h - 1 or c1 > w - 1:
        return False
    rr0 = r0 - 1
    cc0 = c0 - 1
    rr1 = r1 + 1
    cc1 = c1 + 1
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_shape(grid: np.ndarray, occupied: np.ndarray, shape: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    sh, sw = shape.shape
    positions = [(r, c) for r in range(1, h - sh) for c in range(1, w - sw)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if not _can_place(occupied, r0, c0, shape):
            continue
        patch = grid[r0 : r0 + sh, c0 : c0 + sw]
        patch[shape] = color
        occupied[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1] = True
        return True
    return False


def _object_gap_counts(grid: np.ndarray, bg: int) -> list[int]:
    legend, legend_cells = _detect_legend(grid, bg)
    del legend
    work = grid.copy()
    for r, c in legend_cells:
        work[r, c] = bg
    counts = []
    for comp in _components(work != bg):
        r0, c0, r1, c1 = _bbox(comp)
        counts.append(_count_bg_components(work[r0 : r1 + 1, c0 : c1 + 1], bg))
    return sorted(counts)


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        bg = random.randint(0, 9)
        grid = np.full((h, w), bg, dtype=int)

        legend_cells, reserved = _legend_spec(h, w)
        occupied = np.zeros((h, w), dtype=bool)
        for r, c in reserved:
            if 0 <= r < h and 0 <= c < w:
                occupied[r, c] = True

        usable_colors = [c for c in range(10) if c != bg]
        random.shuffle(usable_colors)
        extra_count = random.randint(0, 2)
        gap_counts = [0, 1, 2, 3] + [random.randint(0, 3) for _ in range(extra_count)]
        needed = 3 + len(gap_counts)
        if len(usable_colors) < needed:
            continue

        legend_colors = usable_colors[:3]
        object_colors = usable_colors[3:needed]
        for (r, c), color in zip(legend_cells, legend_colors):
            grid[r, c] = color

        ok = True
        for gap_count, color in zip(gap_counts, object_colors):
            shape = _make_shape(gap_count)
            if not _place_shape(grid, occupied, shape, color):
                ok = False
                break
        if not ok:
            continue

        if _most_common_color(grid) != bg:
            continue
        if _object_gap_counts(grid, bg).count(0) < 1:
            continue
        if not all(k in _object_gap_counts(grid, bg) for k in (0, 1, 2, 3)):
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
