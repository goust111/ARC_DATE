from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
EDGES = ("left", "right", "top", "bottom")
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int, ignore: np.ndarray | None = None) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    if ignore is None:
        ignore = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []
    for r in range(h):
        for c in range(w):
            if bool(seen[r, c]) or bool(ignore[r, c]) or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(seen[nr, nc]) or bool(ignore[nr, nc]) or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append((color, np.array(pts, dtype=int)))
    return comps


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, _ = grid.shape
    candidates: list[tuple[int, np.ndarray]] = []
    for pts in _components(grid != bg):
        rows = pts[:, 0]
        cols = pts[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        if r1 - r0 + 1 != 2:
            continue
        if r0 not in (0, h - 2):
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg):
            continue
        if pts.shape[0] != sub.size:
            continue
        if sub.shape[1] < 2:
            continue
        candidates.append((sub.shape[1], pts))

    assert candidates, "legend must be a full 2xN border block"
    legend_pts = max(candidates, key=lambda item: item[0])[1]
    rows = legend_pts[:, 0]
    cols = legend_pts[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    mapping: dict[int, int] = {}
    for c in range(c0, c1 + 1):
        src = int(grid[r0, c])
        dst = int(grid[r0 + 1, c])
        assert src != bg and dst != bg
        assert src not in mapping, "legend source colors must be unique"
        mapping[src] = dst
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each object to its nearest edge along its longest bbox axis and paint the swept path using the legend color map.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    mapping, legend_mask = _find_legend(g, bg)
    objects = _monochrome_components(g, bg, ignore=legend_mask)
    out = np.full_like(g, bg)

    objects.sort(key=lambda item: (int(item[1][:, 0].min()), int(item[1][:, 1].min())))
    for color, coords in objects:
        trail_color = mapping[color]
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        assert box_h != box_w, "object bbox must have a unique longest axis"

        if box_w > box_h:
            left_dist = c0
            right_dist = w - 1 - c1
            if left_dist <= right_dist:
                step_dr, step_dc = 0, -1
                steps = left_dist
            else:
                step_dr, step_dc = 0, 1
                steps = right_dist
        else:
            top_dist = r0
            bottom_dist = h - 1 - r1
            if top_dist <= bottom_dist:
                step_dr, step_dc = -1, 0
                steps = top_dist
            else:
                step_dr, step_dc = 1, 0
                steps = bottom_dist

        source = {(int(r), int(c)) for r, c in coords}
        for step in range(1, steps + 1):
            for r, c in source:
                rr = r + step_dr * step
                cc = c + step_dc * step
                if (rr, cc) not in source:
                    out[rr, cc] = trail_color

        for r, c in source:
            out[r + step_dr * steps, c + step_dc * steps] = color

    return out


def _horizontal_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(1, 4)
    w_min = max(3, h + 1)
    w_max = min(7, h + 5)
    w = rng.randint(w_min, w_max)
    mask = np.zeros((h, w), dtype=bool)
    spine = rng.randrange(h)
    intervals: list[tuple[int, int]] = [(-1, -1) for _ in range(h)]
    intervals[spine] = (0, w - 1)

    def sample_overlap(prev_start: int, prev_end: int) -> tuple[int, int]:
        if rng.random() < 0.3:
            return 0, w - 1
        min_len = max(1, w // 2)
        max_len = w - 1 if w > 1 else 1
        length = rng.randint(min_len, max_len)
        low = max(0, prev_start - length + 1)
        high = min(prev_end, w - length)
        start = rng.randint(low, high)
        return start, start + length - 1

    for r in range(spine - 1, -1, -1):
        intervals[r] = sample_overlap(*intervals[r + 1])
    for r in range(spine + 1, h):
        intervals[r] = sample_overlap(*intervals[r - 1])

    for r, (start, end) in enumerate(intervals):
        mask[r, start : end + 1] = True

    return mask


def _random_shape(orientation: str, rng: random.Random) -> np.ndarray:
    base = _horizontal_shape(rng)
    return base if orientation == "horizontal" else base.T


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _rect_is_free(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _interior_choices(total: int, span: int) -> list[int]:
    choices = list(range(1, total - span))
    if choices:
        return choices
    return list(range(0, total - span + 1))


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    edge: str,
    color: int,
    mask: np.ndarray,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    pts = np.argwhere(mask)

    if edge in ("left", "right"):
        max_shift = (w - mw - 1) // 2
        if max_shift < 1:
            return False
        orth_choices = _interior_choices(h, mh)
        shifts = list(range(1, max_shift + 1))
        rng.shuffle(orth_choices)
        rng.shuffle(shifts)
        for shift in shifts:
            for r0 in orth_choices:
                if edge == "left":
                    c0 = shift
                    sweep = (r0, 0, r0 + mh - 1, shift + mw - 1)
                else:
                    c0 = w - mw - shift
                    sweep = (r0, c0, r0 + mh - 1, w - 1)
                if not _rect_is_free(reserved, *sweep, pad=1):
                    continue
                for dr, dc in pts:
                    grid[r0 + int(dr), c0 + int(dc)] = color
                _mark_reserved(reserved, *sweep, pad=1)
                return True
        return False

    max_shift = (h - mh - 1) // 2
    if max_shift < 1:
        return False
    orth_choices = _interior_choices(w, mw)
    shifts = list(range(1, max_shift + 1))
    rng.shuffle(orth_choices)
    rng.shuffle(shifts)
    for shift in shifts:
        for c0 in orth_choices:
            if edge == "top":
                r0 = shift
                sweep = (0, c0, shift + mh - 1, c0 + mw - 1)
            else:
                r0 = h - mh - shift
                sweep = (r0, c0, h - 1, c0 + mw - 1)
            if not _rect_is_free(reserved, *sweep, pad=1):
                continue
            for dr, dc in pts:
                grid[r0 + int(dr), c0 + int(dc)] = color
            _mark_reserved(reserved, *sweep, pad=1)
            return True
    return False


def _validate_generated(grid: np.ndarray, expected_objects: int) -> None:
    bg = _background_color(grid)
    mapping, legend_mask = _find_legend(grid, bg)
    assert len(mapping) == expected_objects

    border = np.zeros_like(grid, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert not bool(((grid != bg) & border & ~legend_mask).any())

    objects = _monochrome_components(grid, bg, ignore=legend_mask)
    assert len(objects) == expected_objects
    assert {color for color, _coords in objects} == set(mapping)

    h, w = grid.shape
    for color, coords in objects:
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        assert box_h != box_w
        assert r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1
        if box_w > box_h:
            assert c0 != w - 1 - c1
        else:
            assert r0 != h - 1 - r1
        assert color in mapping

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(2, 4)
    h = rng.randint(13, 19)
    w = rng.randint(13, 19)
    if n_objects == 4:
        h = max(h, 15)
        w = max(w, 15)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    object_colors = palette[:n_objects]
    trail_colors = palette[n_objects : 2 * n_objects]
    if len(trail_colors) < n_objects:
        return None

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    legend_side = rng.choice(("top", "bottom"))
    legend_row = 0 if legend_side == "top" else h - 2
    legend_start = rng.randint(0, w - n_objects)
    order = list(range(n_objects))
    rng.shuffle(order)
    for idx, col in enumerate(range(legend_start, legend_start + n_objects)):
        color_idx = order[idx]
        grid[legend_row, col] = object_colors[color_idx]
        grid[legend_row + 1, col] = trail_colors[color_idx]
    _mark_reserved(reserved, legend_row, legend_start, legend_row + 1, legend_start + n_objects - 1, pad=1)

    edges = list(rng.sample(EDGES, n_objects))
    rng.shuffle(edges)
    for color, edge in zip(object_colors, edges):
        orientation = "horizontal" if edge in ("left", "right") else "vertical"
        placed = False
        for _ in range(120):
            mask = _random_shape(orientation, rng)
            if _place_object(grid, reserved, edge, color, mask, rng):
                placed = True
                break
        if not placed:
            return None

    _validate_generated(grid, n_objects)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
