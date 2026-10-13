from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(10))
MAX_GEN_ATTEMPTS = 400
MODE_CHOICES = ("unique", "size_tie", "same_row_tie")
CLUSTER_COUNT_RANGE = (3, 6)
CLUSTER_AREA_RANGE = (1, 8)
BODY_HEIGHT_RANGE = (8, 20)
BODY_WIDTH_MAX = 28


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _cluster_key(coords: np.ndarray) -> tuple[int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (int(len(coords)), int(rows.min()), int(cols.min()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor scattered same-color clusters by ordering them with a bottom seed row.

    Input structure:
    - A dominant background color fills most of the grid.
    - The bottom row is a seed row containing several distinct non-background
      single pixels, read from left to right.
    - Above that row, the non-background pixels form the same number of
      4-connected monochrome clusters.

    Rule:
    - Read the seed colors from the bottom row in left-to-right order.
    - In the rows above, find every 4-connected non-background cluster and
      measure its pixel count.
    - Sort clusters by increasing size; break ties by topmost row, then by
      leftmost column.
    - Repaint the sorted clusters with the seed colors in that order, and erase
      the entire seed row back to the background color.

    Key insight:
    - The seed row provides only the output color order, while the cluster areas
      determine which object receives each color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 1

    bg = _dominant_color(g)
    seed_row = g[-1]
    seed_colors = [int(seed_row[c]) for c in range(g.shape[1]) if int(seed_row[c]) != bg]
    assert seed_colors, "expected at least one seed color in the bottom row"
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must be distinct"

    body_mask = g[:-1] != bg
    comps = _components4(body_mask)
    assert len(comps) == len(seed_colors), "seed count must match cluster count"

    ordered = sorted(comps, key=_cluster_key)
    out = g.copy()
    out[-1, :] = bg
    if body_mask.any():
        out[:-1, :] = np.where(body_mask, bg, out[:-1, :])
    for color, coords in zip(seed_colors, ordered):
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    def rotate(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        return (int(c), int(-r))

    variants: set[tuple[tuple[int, int], ...]] = set()
    cur = list(shape)
    for _ in range(4):
        norm = _normalize_shape(set(cur))
        variants.add(norm)
        mirrored = _normalize_shape({(r, -c) for r, c in cur})
        variants.add(mirrored)
        cur = [rotate(cell) for cell in cur]
    return random.choice(sorted(variants))


def _random_connected_shape(area: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (int(r + dr), int(c + dc))
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    return _transform_shape(_normalize_shape(cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    return (int(max_r + 1), int(max_c + 1))


def _translate_shape(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in shape]


def _add_blocked(blocked: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((int(r + dr), int(c + dc)))


def _can_place(
    cells: list[tuple[int, int]],
    body_h: int,
    body_w: int,
    blocked: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not (0 <= r < body_h and 0 <= c < body_w):
            return False
        if (int(r), int(c)) in blocked:
            return False
    return True


def _sample_spaced_positions(width: int, count: int, gap: int) -> list[int] | None:
    if count <= 0 or width < count:
        return None
    for _ in range(200):
        cols = sorted(random.sample(range(width), count))
        if all(cols[i + 1] - cols[i] >= gap for i in range(len(cols) - 1)):
            return cols
    return None


def _make_size_profile(count: int, mode: str) -> list[int]:
    pool = list(range(CLUSTER_AREA_RANGE[0], CLUSTER_AREA_RANGE[1] + 1))
    if mode == "unique":
        sizes = random.sample(pool, count)
        random.shuffle(sizes)
        return sizes

    tie_size = random.choice(pool)
    others = random.sample([v for v in pool if v != tie_size], count - 2)
    random.shuffle(others)
    return [tie_size, tie_size, *others]


def _place_clusters(
    body_h: int,
    body_w: int,
    shapes: list[tuple[tuple[int, int], ...]],
    mode: str,
) -> list[list[tuple[int, int]]] | None:
    blocked: set[tuple[int, int]] = set()
    placements: list[list[tuple[int, int]] | None] = [None] * len(shapes)

    if mode == "same_row_tie":
        shape_a = shapes[0]
        shape_b = shapes[1]
        ha, wa = _shape_dims(shape_a)
        hb, wb = _shape_dims(shape_b)
        if body_h < max(ha, hb):
            return None
        row_choices = list(range(body_h - max(ha, hb) + 1))
        random.shuffle(row_choices)
        placed_pair = False
        for r0 in row_choices:
            col_choices_a = list(range(body_w - wa + 1))
            random.shuffle(col_choices_a)
            for c0 in col_choices_a:
                cells_a = _translate_shape(shape_a, r0, c0)
                if not _can_place(cells_a, body_h, body_w, blocked):
                    continue
                trial_blocked = set(blocked)
                _add_blocked(trial_blocked, cells_a)
                col_choices_b = list(range(body_w - wb + 1))
                random.shuffle(col_choices_b)
                for c1 in col_choices_b:
                    cells_b = _translate_shape(shape_b, r0, c1)
                    if not _can_place(cells_b, body_h, body_w, trial_blocked):
                        continue
                    placements[0] = cells_a
                    placements[1] = cells_b
                    _add_blocked(blocked, cells_a)
                    _add_blocked(blocked, cells_b)
                    placed_pair = True
                    break
                if placed_pair:
                    break
            if placed_pair:
                break
        if not placed_pair:
            return None

    remaining = [i for i in range(len(shapes)) if placements[i] is None]
    remaining.sort(key=lambda i: len(shapes[i]), reverse=True)
    for idx in remaining:
        shape = shapes[idx]
        sh, sw = _shape_dims(shape)
        if body_h < sh or body_w < sw:
            return None
        for _ in range(300):
            r0 = random.randint(0, body_h - sh)
            c0 = random.randint(0, body_w - sw)
            cells = _translate_shape(shape, r0, c0)
            if not _can_place(cells, body_h, body_w, blocked):
                continue
            placements[idx] = cells
            _add_blocked(blocked, cells)
            break
        if placements[idx] is None:
            return None

    return [cells for cells in placements if cells is not None]


def _summarize_case(grid: np.ndarray) -> list[tuple[int, int, int]]:
    bg = _dominant_color(grid)
    comps = _components4(grid[:-1] != bg)
    return sorted(_cluster_key(comp) for comp in comps)


def _mode_is_covered(summary: list[tuple[int, int, int]], mode: str) -> bool:
    sizes = [item[0] for item in summary]
    if mode == "unique":
        return len(set(sizes)) == len(sizes)

    equal_pairs = [
        (summary[i], summary[j])
        for i in range(len(summary))
        for j in range(i + 1, len(summary))
        if summary[i][0] == summary[j][0]
    ]
    if not equal_pairs:
        return False
    if mode == "size_tie":
        return any(a[1] != b[1] for a, b in equal_pairs)
    return any(a[1] == b[1] for a, b in equal_pairs)


def _validate_generated(grid: np.ndarray, mode: str, cluster_color: int) -> bool:
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2 and max(grid.shape) <= 30):
        return False

    bg = _dominant_color(grid)
    seed_colors = [int(grid[-1, c]) for c in range(grid.shape[1]) if int(grid[-1, c]) != bg]
    if len(seed_colors) < 3 or len(seed_colors) != len(set(seed_colors)):
        return False

    body = grid[:-1]
    body_vals = {int(v) for v in body[body != bg]}
    if body_vals != {int(cluster_color)}:
        return False

    summary = _summarize_case(grid)
    if len(summary) != len(seed_colors):
        return False
    if not _mode_is_covered(summary, mode):
        return False

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.all(out[-1] == bg):
        return False
    out_colors = {int(v) for v in out[:-1][out[:-1] != bg]}
    if out_colors != set(seed_colors):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODE_CHOICES)
    count = random.randint(*CLUSTER_COUNT_RANGE)
    bg = random.choice(PALETTE)
    remaining = [c for c in PALETTE if c != bg]
    cluster_color = random.choice(remaining)
    seed_palette = [c for c in remaining if c != cluster_color]
    if len(seed_palette) < count:
        return None
    seed_colors = random.sample(seed_palette, count)

    sizes = _make_size_profile(count, mode)
    shapes = [_random_connected_shape(area) for area in sizes]
    max_h = max(_shape_dims(shape)[0] for shape in shapes)
    max_w = max(_shape_dims(shape)[1] for shape in shapes)

    min_body_h = max(BODY_HEIGHT_RANGE[0], max_h + 1)
    if min_body_h > BODY_HEIGHT_RANGE[1]:
        return None
    body_h = random.randint(min_body_h, BODY_HEIGHT_RANGE[1])

    min_body_w = max(2 * count - 1, max_w + 2)
    if mode == "same_row_tie":
        _, wa = _shape_dims(shapes[0])
        _, wb = _shape_dims(shapes[1])
        min_body_w = max(min_body_w, wa + wb + 2)
    if min_body_w > BODY_WIDTH_MAX:
        return None
    body_w = random.randint(min_body_w, BODY_WIDTH_MAX)

    placements = _place_clusters(body_h, body_w, shapes, mode)
    if placements is None or len(placements) != count:
        return None

    seed_gap = 2 if body_w >= 2 * count - 1 and random.random() < 0.7 else 1
    seed_cols = _sample_spaced_positions(body_w, count, seed_gap)
    if seed_cols is None:
        return None

    grid = np.full((body_h + 1, body_w), bg, dtype=int)
    for cells in placements:
        for r, c in cells:
            grid[r, c] = cluster_color
    for c, color in zip(seed_cols, seed_colors):
        grid[body_h, c] = int(color)

    if not _validate_generated(grid, mode, cluster_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
