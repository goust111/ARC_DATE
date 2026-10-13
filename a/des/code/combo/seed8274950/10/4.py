from __future__ import annotations

from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _max_run_length(values: np.ndarray) -> int:
    best = 0
    cur = 0
    prev = None
    for value in values.tolist():
        if prev is None or value != prev:
            cur = 1
            prev = value
        else:
            cur += 1
        if cur > best:
            best = cur
    return best


def _classify_stripes(patch: np.ndarray) -> str:
    h, w = patch.shape

    row_uniform = [int(np.unique(patch[r]).size) == 1 for r in range(h)]
    col_uniform = [int(np.unique(patch[:, c]).size) == 1 for c in range(w)]

    horizontal = all(row_uniform) and not all(col_uniform)
    vertical = all(col_uniform) and not all(row_uniform)
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"

    row_score = sum(_max_run_length(patch[r]) for r in range(h))
    col_score = sum(_max_run_length(patch[:, c]) for c in range(w))
    if row_score == col_score:
        raise AssertionError("ambiguous stripe orientation")
    return "horizontal" if row_score > col_score else "vertical"


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.tolist()))
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "expected a unique larger-area color"
    return int(winners[0])


def _draw_arrow(out: np.ndarray, bbox: tuple[int, int, int, int], orientation: str, color: int):
    r0, c0, r1, c1 = bbox
    if orientation == "horizontal":
        mid_r = (int(r0) + int(r1)) // 2
        out[mid_r, int(c0) : int(c1) + 1] = int(color)
        head_c = int(c1 - 1)
        if mid_r - 1 >= int(r0) and head_c >= int(c0):
            out[mid_r - 1, head_c] = int(color)
        if mid_r + 1 <= int(r1) and head_c >= int(c0):
            out[mid_r + 1, head_c] = int(color)
    else:
        mid_c = (int(c0) + int(c1)) // 2
        out[int(r0) : int(r1) + 1, mid_c] = int(color)
        head_r = int(r1 - 1)
        if mid_c - 1 >= int(c0) and head_r >= int(r0):
            out[head_r, mid_c - 1] = int(color)
        if mid_c + 1 <= int(c1) and head_r >= int(r0):
            out[head_r, mid_c + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each two-color striped object with an arrow whose direction matches its stripe orientation.

    Horizontal stripes become right-pointing arrows centered in the object's bbox.
    Vertical stripes become down-pointing arrows centered in the object's bbox.
    The arrow color is the object's larger-area color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    comps = find_components(g != bg)
    out = g.copy()

    for coords in comps:
        values = g[coords[:, 0], coords[:, 1]]
        colors = np.unique(values)
        assert len(colors) == 2, "expected each object to use exactly two colors"

        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch != bg)), "striped object must fill its bbox"

        orientation = _classify_stripes(patch)
        color = _majority_color(values)

        out[coords[:, 0], coords[:, 1]] = int(bg)
        _draw_arrow(out, (r0, c0, r1, c1), orientation, color)

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()
_PALETTE = tuple(range(10))
_GRID_H_RANGE = (16, 30)
_GRID_W_RANGE = (16, 30)
_OBJECT_COUNT_RANGE = (2, 5)
_PLACEMENT_ATTEMPTS = 200
_GEN_ATTEMPTS = 300
_MARGIN = 1

_H_STRIPE_HEIGHTS = (3, 5, 7, 9)
_H_STRIPE_WIDTH_RANGE = (5, 11)
_V_STRIPE_HEIGHT_RANGE = (5, 11)
_V_STRIPE_WIDTHS = (3, 5, 7, 9)


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _sample_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(_RNG.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    values = [cuts[0]]
    for idx in range(1, len(cuts)):
        values.append(int(cuts[idx] - cuts[idx - 1]))
    values.append(int(total - cuts[-1]))
    return [int(v) for v in values]


def _build_striped_patch(orientation: str, bg: int) -> tuple[np.ndarray, str, int]:
    available = [color for color in _PALETTE if color != bg]
    color_a, color_b = _RNG.choice(available, size=2, replace=False).tolist()

    for _ in range(200):
        if orientation == "horizontal":
            height = int(_RNG.choice(_H_STRIPE_HEIGHTS))
            width = _randint(*_H_STRIPE_WIDTH_RANGE)
            run_count = _randint(3, min(6, height))
            runs = _sample_partition(height, run_count)
            patch = np.empty((height, width), dtype=int)
            row = 0
            start = _randint(0, 1)
            for idx, run in enumerate(runs):
                color = int(color_a if (start + idx) % 2 == 0 else color_b)
                patch[row : row + run, :] = int(color)
                row += int(run)
        else:
            height = _randint(*_V_STRIPE_HEIGHT_RANGE)
            width = int(_RNG.choice(_V_STRIPE_WIDTHS))
            run_count = _randint(3, min(6, width))
            runs = _sample_partition(width, run_count)
            patch = np.empty((height, width), dtype=int)
            col = 0
            start = _randint(0, 1)
            for idx, run in enumerate(runs):
                color = int(color_a if (start + idx) % 2 == 0 else color_b)
                patch[:, col : col + run] = int(color)
                col += int(run)

        counts = Counter(map(int, patch.reshape(-1).tolist()))
        if counts[int(color_a)] == counts[int(color_b)]:
            continue
        majority = int(color_a if counts[int(color_a)] > counts[int(color_b)] else color_b)
        return patch, orientation, majority

    raise RuntimeError("failed to build a non-degenerate striped patch")


def _orientation_plan(count: int) -> list[str]:
    horizontal = count // 2
    vertical = count // 2
    if count % 2 == 1:
        if _randint(0, 1) == 0:
            horizontal += 1
        else:
            vertical += 1
    plan = ["horizontal"] * int(horizontal) + ["vertical"] * int(vertical)
    _RNG.shuffle(plan)
    return plan


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    rr0 = max(0, int(r0 - _MARGIN))
    cc0 = max(0, int(c0 - _MARGIN))
    rr1 = min(blocked.shape[0], int(r1 + _MARGIN + 1))
    cc1 = min(blocked.shape[1], int(c1 + _MARGIN + 1))
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int):
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    rr0 = max(0, int(r0 - _MARGIN))
    cc0 = max(0, int(c0 - _MARGIN))
    rr1 = min(blocked.shape[0], int(r1 + _MARGIN + 1))
    cc1 = min(blocked.shape[1], int(c1 + _MARGIN + 1))
    blocked[rr0:rr1, cc0:cc1] = True


def _render_expected(shape: tuple[int, int], bg: int, objects: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        patch = obj["patch"]
        h, w = patch.shape
        _draw_arrow(out, (r0, c0, r0 + h - 1, c0 + w - 1), str(obj["orientation"]), int(obj["majority"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    h = _randint(*_GRID_H_RANGE)
    w = _randint(*_GRID_W_RANGE)
    bg = _randint(0, 9)
    object_count = _randint(*_OBJECT_COUNT_RANGE)
    plan = _orientation_plan(object_count)

    objects: list[dict] = []
    total_area = 0
    for orientation in plan:
        patch, orient, majority = _build_striped_patch(orientation, bg)
        total_area += int(patch.size)
        objects.append({"patch": patch, "orientation": orient, "majority": majority})

    if total_area * 3 >= h * w:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = list(range(len(objects)))
    order.sort(key=lambda idx: -int(objects[idx]["patch"].size))

    for idx in order:
        patch = objects[idx]["patch"]
        ph, pw = patch.shape
        placed = False
        for _ in range(_PLACEMENT_ATTEMPTS):
            r0 = _randint(0, h - ph)
            c0 = _randint(0, w - pw)
            if not _can_place(blocked, r0, c0, ph, pw):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            _mark_blocked(blocked, r0, c0, ph, pw)
            objects[idx]["r0"] = int(r0)
            objects[idx]["c0"] = int(c0)
            placed = True
            break
        if not placed:
            return None

    counts = Counter(map(int, grid.reshape(-1).tolist()))
    bg_count = counts[int(bg)]
    if any(color != int(bg) and count >= bg_count for color, count in counts.items()):
        return None

    expected = _render_expected(grid.shape, bg, objects)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-arrow input")
