import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GRID_MIN = 20
GRID_MAX = 25
OBJ_COUNT_CHOICES = (3, 4)
TYPE_CHOICES = ("h", "v", "hv", "none")
INNER_MARGIN = 2
MIN_GAP = 1
SIZE_MIN = 3
SIZE_MAX = 5
GEN_RETRIES = 400
MASK_RETRIES = 300
OBJ_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_RNG = np.random.default_rng()


def _mode_color(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=int).ravel()
    counts = Counter(int(v) for v in vals)
    best_count = max(counts.values())
    return min(color for color, count in counts.items() if count == best_count)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _symmetry_axes(mask: np.ndarray) -> tuple[bool, bool]:
    return np.array_equal(mask, np.flipud(mask)), np.array_equal(mask, np.fliplr(mask))


def _project_top(out: np.ndarray, patch: np.ndarray, mask: np.ndarray, c0: int) -> None:
    for local_c in range(mask.shape[1]):
        colors = patch[mask[:, local_c], local_c]
        if colors.size:
            out[0, c0 + local_c] = _mode_color(colors)


def _project_left(out: np.ndarray, patch: np.ndarray, mask: np.ndarray, r0: int) -> None:
    for local_r in range(mask.shape[0]):
        colors = patch[local_r, mask[local_r]]
        if colors.size:
            out[r0 + local_r, 0] = _mode_color(colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each foreground object to the top and/or left receiver strip
    according to whether its shape is horizontally and/or vertically symmetric.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3

    bg = _mode_color(g)
    edge_values = np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))
    frame = _mode_color(edge_values)
    assert frame != bg

    obj_mask = (g != bg) & (g != frame)
    comps = _components(obj_mask)
    assert comps

    out = g.copy()
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[r - r0, c - c0] = True

        has_h, has_v = _symmetry_axes(mask)
        if has_h:
            _project_top(out, patch, mask, c0)
        if has_v:
            _project_left(out, patch, mask, r0)

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _full_span(mask: np.ndarray) -> bool:
    return bool(mask.any(axis=0).all() and mask.any(axis=1).all())


def _mirror_h(base: np.ndarray, full_h: int) -> np.ndarray:
    if full_h % 2 == 0:
        return np.vstack([base, base[::-1]])
    return np.vstack([base, base[:-1][::-1]])


def _mirror_v(base: np.ndarray, full_w: int) -> np.ndarray:
    if full_w % 2 == 0:
        return np.hstack([base, base[:, ::-1]])
    return np.hstack([base, base[:, :-1][:, ::-1]])


def _ensure_rows_cols(base: np.ndarray) -> np.ndarray:
    out = base.copy()
    for r in range(out.shape[0]):
        if not bool(out[r].any()):
            out[r, random.randrange(out.shape[1])] = True
    for c in range(out.shape[1]):
        if not bool(out[:, c].any()):
            out[random.randrange(out.shape[0]), c] = True
    return out


def _random_density() -> float:
    return random.uniform(0.45, 0.75)


def _mask_matches_kind(mask: np.ndarray, kind: str) -> bool:
    has_h, has_v = _symmetry_axes(mask)
    if kind == "h":
        return has_h and not has_v
    if kind == "v":
        return has_v and not has_h
    if kind == "hv":
        return has_h and has_v
    return (not has_h) and (not has_v)


def _generate_mask(kind: str, h: int, w: int) -> np.ndarray:
    min_cells = max(h + w - 2, (h * w + 1) // 2)
    max_cells = h * w - 1

    for _ in range(MASK_RETRIES):
        if kind == "h":
            base = _RNG.random(((h + 1) // 2, w)) < _random_density()
            mask = _mirror_h(_ensure_rows_cols(base), h)
        elif kind == "v":
            base = _RNG.random((h, (w + 1) // 2)) < _random_density()
            mask = _mirror_v(_ensure_rows_cols(base), w)
        elif kind == "hv":
            base = _RNG.random(((h + 1) // 2, (w + 1) // 2)) < _random_density()
            mask = _mirror_h(_mirror_v(_ensure_rows_cols(base), w), h)
        else:
            mask = _RNG.random((h, w)) < _random_density()
            mask = _ensure_rows_cols(mask)

        cells = int(mask.sum())
        if not (min_cells <= cells <= max_cells):
            continue
        if not _full_span(mask):
            continue
        if not _is_connected(mask):
            continue
        if not _mask_matches_kind(mask, kind):
            continue
        return mask.astype(bool)

    raise RuntimeError(f"failed to generate a {kind} mask of shape {(h, w)}")


def _bounded_partition(total: int, count: int, low: int, high: int) -> list[int]:
    parts: list[int] = []
    remaining = total
    for idx in range(count):
        left = count - idx - 1
        cur_low = max(low, remaining - left * high)
        cur_high = min(high, remaining - left * low)
        parts.append(random.randint(cur_low, cur_high))
        remaining -= parts[-1]
    random.shuffle(parts)
    return parts


def _sample_lengths(count: int, total: int) -> list[int]:
    budget = total - 2 * INNER_MARGIN - MIN_GAP * (count - 1)
    min_total = SIZE_MIN * count
    max_total = SIZE_MAX * count
    assert budget >= min_total
    target = random.randint(min_total, min(max_total, budget))
    return _bounded_partition(target, count, SIZE_MIN, SIZE_MAX)


def _random_composition(total: int, parts: int) -> list[int]:
    vals = [0] * parts
    for _ in range(total):
        vals[random.randrange(parts)] += 1
    return vals


def _assign_positions(order: list[int], lengths: list[int], total: int) -> dict[int, int]:
    ordered_lengths = [lengths[idx] for idx in order]
    need = sum(ordered_lengths) + MIN_GAP * (len(order) - 1) + 2 * INNER_MARGIN
    slack = total - need
    assert slack >= 0
    extras = _random_composition(slack, len(order) + 1)

    starts: dict[int, int] = {}
    pos = INNER_MARGIN + extras[0]
    for slot_idx, obj_idx in enumerate(order):
        starts[obj_idx] = pos
        pos += ordered_lengths[slot_idx]
        if slot_idx < len(order) - 1:
            pos += MIN_GAP + extras[slot_idx + 1]
    return starts


def _pick_types(count: int) -> list[str]:
    if count == 4:
        result = list(TYPE_CHOICES)
        random.shuffle(result)
        return result
    omitted = random.choice(TYPE_CHOICES)
    result = [kind for kind in TYPE_CHOICES if kind != omitted]
    random.shuffle(result)
    return result


def _paint_frame(grid: np.ndarray) -> None:
    grid[0, :] = FRAME
    grid[-1, :] = FRAME
    grid[:, 0] = FRAME
    grid[:, -1] = FRAME


def _place_object(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    window[mask] = int(color)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    count = random.choice(OBJ_COUNT_CHOICES)
    types = _pick_types(count)
    colors = random.sample(OBJ_COLORS, count)

    heights = _sample_lengths(count, h)
    widths = _sample_lengths(count, w)

    row_order = random.sample(range(count), count)
    col_order = random.sample(range(count), count)
    row_starts = _assign_positions(row_order, heights, h)
    col_starts = _assign_positions(col_order, widths, w)

    objects = []
    for idx in range(count):
        kind = types[idx]
        obj_h = heights[idx]
        obj_w = widths[idx]
        mask = _generate_mask(kind, obj_h, obj_w)
        objects.append(
            {
                "kind": kind,
                "color": colors[idx],
                "mask": mask,
                "r0": row_starts[idx],
                "c0": col_starts[idx],
            }
        )

    grid = np.full((h, w), BG, dtype=int)
    _paint_frame(grid)
    for obj in objects:
        _place_object(grid, obj["mask"], int(obj["color"]), int(obj["r0"]), int(obj["c0"]))

    out = main(grid)
    if not np.any(out[0] != grid[0]):
        return None
    if not np.any(out[:, 0] != grid[:, 0]):
        return None

    for obj in objects:
        mask = np.asarray(obj["mask"], dtype=bool)
        has_h, has_v = _symmetry_axes(mask)
        kind = str(obj["kind"])
        if kind == "h" and (not has_h or has_v):
            return None
        if kind == "v" and (not has_v or has_h):
            return None
        if kind == "hv" and (not has_h or not has_v):
            return None
        if kind == "none" and (has_h or has_v):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
