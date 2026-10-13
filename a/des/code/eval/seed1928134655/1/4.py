from __future__ import annotations

import random
from collections import deque

import numpy as np


AXIS_COLOR = 5
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
ORTHO4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _full_line_color(line: np.ndarray) -> int | None:
    first = int(line[0])
    if np.all(line == first):
        return first
    return None


def _detect_axes_and_stripes(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    axes: list[dict[str, object]] = []

    for c in range(w):
        if not np.all(g[:, c] == AXIS_COLOR):
            continue
        stripes: dict[int, int] = {}
        for side in (-1, 1):
            cc = c + side
            if 0 <= cc < w:
                color = _full_line_color(g[:, cc])
                if color is not None and color not in (bg, AXIS_COLOR):
                    stripes[side] = int(color)
        axes.append({"kind": "v", "pos": c, "stripes": stripes})

    for r in range(h):
        if not np.all(g[r, :] == AXIS_COLOR):
            continue
        stripes = {}
        for side in (-1, 1):
            rr = r + side
            if 0 <= rr < h:
                color = _full_line_color(g[rr, :])
                if color is not None and color not in (bg, AXIS_COLOR):
                    stripes[side] = int(color)
        axes.append({"kind": "h", "pos": r, "stripes": stripes})

    assert axes, "expected at least one grey symmetry axis"
    return axes


def _line_mask(grid: np.ndarray, axes: list[dict[str, object]]) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    for axis in axes:
        pos = int(axis["pos"])
        stripes = axis["stripes"]
        if axis["kind"] == "v":
            mask[:, pos] = True
            for side in stripes:
                mask[:, pos + int(side)] = True
        else:
            mask[pos, :] = True
            for side in stripes:
                mask[pos + int(side), :] = True
    return mask


def _component_axis_distance(
    comp: list[tuple[int, int]], axis: dict[str, object]
) -> tuple[int, int] | None:
    pos = int(axis["pos"])
    if axis["kind"] == "v":
        diffs = [c - pos for _r, c in comp]
    else:
        diffs = [r - pos for r, _c in comp]
    if any(diff == 0 for diff in diffs):
        return None
    signs = {1 if diff > 0 else -1 for diff in diffs}
    if len(signs) != 1:
        return None
    side = int(next(iter(signs)))
    dist = min(abs(diff) for diff in diffs)
    return dist, side


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each colored pattern across its nearest grey axis and recolor the copy
    with the stripe color adjacent to that axis on the pattern's side.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    axes = _detect_axes_and_stripes(g, bg)
    lines = _line_mask(g, axes)
    pattern_mask = (g != bg) & (~lines)
    comps = _components(pattern_mask)

    out = g.copy()
    for comp in comps:
        options: list[tuple[int, int, int]] = []
        for idx, axis in enumerate(axes):
            info = _component_axis_distance(comp, axis)
            if info is None:
                continue
            dist, side = info
            options.append((dist, idx, side))

        assert options, "pattern must lie on one side of at least one axis"
        options.sort()
        best_dist = options[0][0]
        tied = [item for item in options if item[0] == best_dist]
        assert len(tied) == 1, "nearest axis must be unambiguous"

        _dist, axis_idx, side = tied[0]
        axis = axes[axis_idx]
        stripe_color = axis["stripes"].get(side)
        assert stripe_color is not None, "nearest axis side must expose a stripe"
        pos = int(axis["pos"])

        for r, c in comp:
            if axis["kind"] == "v":
                rr, cc = r, 2 * pos - c
            else:
                rr, cc = 2 * pos - r, c
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(stripe_color)
    return out


def _is_connected_4(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ORTHO4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def _shape_spans_box(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    return (
        0 in rows
        and height - 1 in rows
        and 0 in cols
        and width - 1 in cols
        and len(rows) == height
        and len(cols) == width
    )


def _random_shape(rng: random.Random, max_width: int) -> list[tuple[int, int]]:
    for _ in range(200):
        height = rng.randint(3, 5)
        width = rng.randint(3, min(5, max_width))
        target = rng.randint(max(5, (height * width + 1) // 2), height * width)
        cells: set[tuple[int, int]] = {(r, c) for r in range(height) for c in range(width)}
        stagnant = 0
        while len(cells) > target and stagnant < 200:
            cell = rng.choice(tuple(cells))
            trial = set(cells)
            trial.remove(cell)
            if _is_connected_4(trial) and _shape_spans_box(trial, height, width):
                cells = trial
                stagnant = 0
            else:
                stagnant += 1
        if _shape_spans_box(cells, height, width) and _is_connected_4(cells):
            return sorted(cells)
    raise RuntimeError("failed to sample a shape")


def _nearest_axis_map(width: int, axes: list[int]) -> list[int]:
    nearest = [-1] * width
    for c in range(width):
        dists = [abs(c - axis) for axis in axes]
        best = min(dists)
        if dists.count(best) == 1:
            nearest[c] = int(dists.index(best))
    return nearest


def _column_zone_map(width: int, axes: list[int]) -> dict[tuple[int, int], list[int]]:
    nearest = _nearest_axis_map(width, axes)
    zones: dict[tuple[int, int], list[int]] = {}
    for idx, axis in enumerate(axes):
        zones[(idx, -1)] = [c for c in range(width) if nearest[c] == idx and c < axis - 1]
        zones[(idx, 1)] = [c for c in range(width) if nearest[c] == idx and c > axis + 1]
    return zones


def _choose_axes(rng: random.Random, width: int, num_axes: int) -> list[int] | None:
    if num_axes == 1:
        if width < 15:
            return None
        return [rng.randint(6, width - 7)]

    pairs: list[tuple[int, int]] = []
    for a0 in range(6, width - 12):
        for a1 in range(a0 + 11, width - 6):
            zones = _column_zone_map(width, [a0, a1])
            if min(len(cols) for cols in zones.values()) >= 4:
                pairs.append((a0, a1))
    if not pairs:
        return None
    a0, a1 = rng.choice(pairs)
    return [a0, a1]


def _populate_sides(rng: random.Random, num_axes: int) -> list[tuple[int, int]]:
    sides: list[tuple[int, int]] = []
    for idx in range(num_axes):
        if rng.random() < 0.5:
            sides.append((idx, rng.choice((-1, 1))))
        else:
            sides.extend([(idx, -1), (idx, 1)])
    return sides


def _pattern_counts(rng: random.Random, side_count: int) -> list[int]:
    if side_count <= 2:
        return [1 + int(rng.random() < 0.5) for _ in range(side_count)]
    if side_count == 3:
        counts = [1, 1, 2]
        rng.shuffle(counts)
        return counts
    return [1] * side_count


def _cells_clear(
    cells: list[tuple[int, int]], exact_blocked: np.ndarray, halo_blocked: np.ndarray
) -> bool:
    h, w = exact_blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if exact_blocked[r, c] or halo_blocked[r, c]:
            return False
    return True


def _mark_halo(halo_blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = halo_blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                halo_blocked[rr, cc] = True


def _transpose_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(c, r) for r, c in cells]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_axes = rng.choice((1, 2))
    height = rng.randint(16, 24)
    width = rng.randint(18 if num_axes == 1 else 22, 28)
    axes = _choose_axes(rng, width, num_axes)
    if axes is None:
        return None

    bg = rng.choice([c for c in range(10) if c != AXIS_COLOR])
    grid = np.full((height, width), bg, dtype=int)
    exact_blocked = np.zeros((height, width), dtype=bool)
    halo_blocked = np.zeros((height, width), dtype=bool)

    side_specs = _populate_sides(rng, num_axes)
    palette = [c for c in range(10) if c not in (bg, AXIS_COLOR)]
    if len(side_specs) > len(palette):
        return None
    stripe_colors = rng.sample(palette, len(side_specs))
    stripe_by_side = {side: color for side, color in zip(side_specs, stripe_colors)}

    for axis in axes:
        grid[:, axis] = AXIS_COLOR
        exact_blocked[:, axis] = True

    for (axis_idx, side), color in stripe_by_side.items():
        col = axes[axis_idx] + side
        if not (0 <= col < width):
            return None
        grid[:, col] = color
        exact_blocked[:, col] = True

    zones = _column_zone_map(width, axes)
    counts = _pattern_counts(rng, len(side_specs))
    placements: list[tuple[list[tuple[int, int]], list[tuple[int, int]], int]] = []

    items = list(zip(side_specs, counts))
    rng.shuffle(items)
    for (axis_idx, side), pattern_count in items:
        stripe_color = stripe_by_side[(axis_idx, side)]
        zone_cols = zones[(axis_idx, side)]
        if len(zone_cols) < 3:
            return None
        col_min, col_max = min(zone_cols), max(zone_cols)
        axis = axes[axis_idx]

        for _ in range(pattern_count):
            placed = False
            for _ in range(250):
                shape = _random_shape(rng, max_width=len(zone_cols))
                shape_h = max(r for r, _c in shape) + 1
                shape_w = max(c for _r, c in shape) + 1

                if shape_w > len(zone_cols) or height - shape_h - 1 < 1:
                    continue

                row0 = rng.randint(1, height - shape_h - 1)
                col0_low = col_min
                col0_high = col_max - shape_w + 1
                if col0_low > col0_high:
                    continue
                col0 = rng.randint(col0_low, col0_high)

                source = [(row0 + r, col0 + c) for r, c in shape]
                if not all(
                    0 <= c < width
                    and _nearest_axis_map(width, axes)[c] == axis_idx
                    and ((side == -1 and c < axis - 1) or (side == 1 and c > axis + 1))
                    for _r, c in source
                ):
                    continue

                mirrored = [(r, 2 * axis - c) for r, c in source]
                if not _cells_clear(source, exact_blocked, halo_blocked):
                    continue
                if not _cells_clear(mirrored, exact_blocked, halo_blocked):
                    continue

                color_choices = [c for c in palette if c != stripe_color]
                color = rng.choice(color_choices)
                for r, c in source:
                    grid[r, c] = color
                _mark_halo(halo_blocked, source)
                _mark_halo(halo_blocked, mirrored)
                placements.append((source, mirrored, stripe_color))
                placed = True
                break
            if not placed:
                return None

    if not placements:
        return None

    if rng.random() < 0.5:
        grid = grid.T.copy()
        placements = [
            (_transpose_cells(src), _transpose_cells(dst), color)
            for src, dst, color in placements
        ]

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    for source, mirrored, color in placements:
        if any(int(out[r, c]) != int(grid[r, c]) for r, c in source):
            return None
        if any(int(out[r, c]) != color for r, c in mirrored):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 300 attempts")
