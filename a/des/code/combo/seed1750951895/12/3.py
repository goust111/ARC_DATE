import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
PALETTE = tuple(c for c in range(1, 10) if c != GREY)
RNG = np.random.default_rng()
MAX_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_of(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def bbox_area(coords: np.ndarray) -> int:
    r0, r1, c0, c1 = bbox_of(coords)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def dominant_color(grid: np.ndarray, coords: np.ndarray) -> int:
    counts = Counter(int(grid[r, c]) for r, c in coords)
    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    if len(winners) != 1:
        raise AssertionError("dominant color must be unique")
    return int(winners[0])


def find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], set[int]]:
    grey_comps = connected_components(grid == GREY)
    candidates = []
    for comp in grey_comps:
        r0, r1, c0, c1 = bbox_of(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if not np.all(box[border_mask] == GREY):
            continue
        if np.any(box[~border_mask] == GREY):
            continue
        inner_colors = {int(v) for v in box[1:-1, 1:-1].ravel() if int(v) not in (BG, GREY)}
        if 2 <= len(inner_colors) <= 3:
            candidates.append(((r0, r1, c0, c1), inner_colors))
    if not candidates:
        raise AssertionError("legend not found")
    candidates.sort(key=lambda item: (item[0][0] + item[0][2], item[0][0], item[0][2], bbox_area(np.array(
        [(item[0][0], item[0][2]), (item[0][1], item[0][3])]
    ))))
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete any 4-connected object containing all legend colors, then encode
    the remaining objects as a single color column sorted by bbox area.
    """
    g = np.asarray(grid, dtype=int)
    legend_bbox, target_colors = find_legend(g)
    r0, r1, c0, c1 = legend_bbox

    work = g.copy()
    work[r0 : r1 + 1, c0 : c1 + 1] = BG

    survivors: list[tuple[int, int]] = []
    for comp in connected_components(work != BG):
        colors = {int(work[r, c]) for r, c in comp}
        if target_colors.issubset(colors):
            continue
        survivors.append((bbox_area(comp), dominant_color(work, comp)))

    if not survivors:
        raise AssertionError("there must be at least one remaining object")

    survivors.sort(key=lambda item: (-item[0], item[1]))
    return np.array([[color] for _area, color in survivors], dtype=int)


def _random_connected_shape(height: int, width: int, cell_count: int) -> np.ndarray:
    cells = {(RNG.integers(height), RNG.integers(width))}
    while len(cells) < cell_count:
        r, c = random.choice(tuple(cells))
        dirs = list(neighbors4(r, c))
        RNG.shuffle(dirs)
        for nr, nc in dirs:
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                cells.add((nr, nc))
                break
        else:
            frontier = []
            for rr, cc in cells:
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if frontier:
                cells.add(frontier[int(RNG.integers(len(frontier)))])
    shape = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        shape[r, c] = True
    rows = np.where(shape.any(axis=1))[0]
    cols = np.where(shape.any(axis=0))[0]
    return shape[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _color_object(shape: np.ndarray, allowed_colors: list[int], dominant: int) -> np.ndarray | None:
    coords = np.argwhere(shape)
    n = len(coords)
    if n < len(allowed_colors) + 1:
        return None
    colored = np.full(shape.shape, BG, dtype=int)
    order = coords.copy()
    RNG.shuffle(order)

    dominant_count = int(RNG.integers(max(2, n // 3), n - len(allowed_colors) + 2))
    dominant_count = min(dominant_count, n - (len(allowed_colors) - (1 if dominant in allowed_colors else 0)))
    dominant_count = max(dominant_count, 2)

    ptr = 0
    placed = 0
    while placed < dominant_count and ptr < n:
        r, c = order[ptr]
        colored[r, c] = dominant
        ptr += 1
        placed += 1

    for color in allowed_colors:
        if color == dominant:
            continue
        if ptr >= n:
            return None
        r, c = order[ptr]
        colored[r, c] = color
        ptr += 1

    remaining_palette = [c for c in allowed_colors if c != dominant]
    while ptr < n:
        r, c = order[ptr]
        if not remaining_palette or RNG.random() < 0.6:
            colored[r, c] = dominant
        else:
            colored[r, c] = int(remaining_palette[RNG.integers(len(remaining_palette))])
        ptr += 1

    counts = Counter(int(colored[r, c]) for r, c in coords)
    if counts[dominant] <= max(v for k, v in counts.items() if k != dominant):
        return None
    return colored


def _build_object(target_colors: list[int], deleted: bool, used_survivor_areas: set[int]):
    for _ in range(120):
        if deleted:
            bbox_h = int(RNG.integers(2, 6))
            bbox_w = int(RNG.integers(3, 8))
        else:
            bbox_h = int(RNG.integers(2, 7))
            bbox_w = int(RNG.integers(2, 8))
        bbox_area_value = bbox_h * bbox_w
        if not deleted and bbox_area_value in used_survivor_areas:
            continue

        min_cells = max(len(target_colors) + 1, (bbox_h * bbox_w) // 2)
        cell_count = int(RNG.integers(min_cells, bbox_h * bbox_w + 1))
        shape = _random_connected_shape(bbox_h, bbox_w, cell_count)
        if shape.shape != (bbox_h, bbox_w):
            continue

        extra_pool = [c for c in PALETTE if c not in target_colors]
        RNG.shuffle(extra_pool)
        if deleted:
            colors = list(target_colors) + extra_pool[: int(RNG.integers(0, 3))]
        else:
            keep_count = 1 if len(target_colors) == 2 else int(RNG.integers(1, len(target_colors)))
            kept_targets = list(RNG.choice(target_colors, size=keep_count, replace=False))
            extra_needed = max(1, int(RNG.integers(1, 3)))
            colors = kept_targets + extra_pool[:extra_needed]
        colors = list(dict.fromkeys(colors))
        if len(colors) < 2:
            continue

        dominant = int(colors[RNG.integers(len(colors))])
        colored = _color_object(shape, colors, dominant)
        if colored is None:
            continue

        comp_colors = {int(v) for v in colored.ravel() if int(v) != BG}
        if deleted != set(target_colors).issubset(comp_colors):
            continue
        if not deleted and bbox_area_value in used_survivor_areas:
            continue
        return colored, bbox_area_value
    return None, None


def _can_place(occupied: np.ndarray, top: int, left: int, obj: np.ndarray) -> bool:
    h, w = obj.shape
    H, W = occupied.shape
    r0 = max(0, top - 1)
    r1 = min(H, top + h + 1)
    c0 = max(0, left - 1)
    c1 = min(W, left + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place(grid: np.ndarray, occupied: np.ndarray, obj: np.ndarray) -> bool:
    H, W = grid.shape
    h, w = obj.shape
    positions = [(r, c) for r in range(1, H - h) for c in range(1, W - w)]
    RNG.shuffle(positions)
    for top, left in positions:
        if _can_place(occupied, top, left, obj):
            mask = obj != BG
            grid[top : top + h, left : left + w][mask] = obj[mask]
            occupied[top : top + h, left : left + w][mask] = True
            return True
    return False


def _make_legend(grid: np.ndarray, occupied: np.ndarray, target_colors: list[int]) -> None:
    interior_h = int(RNG.integers(2, 4))
    interior_w = len(target_colors) if RNG.random() < 0.5 else len(target_colors) + 1
    h = interior_h + 2
    w = interior_w + 2
    top = int(RNG.integers(1, 3))
    left = int(RNG.integers(1, 3))
    grid[top : top + h, left : left + w] = BG
    grid[top, left : left + w] = GREY
    grid[top + h - 1, left : left + w] = GREY
    grid[top : top + h, left] = GREY
    grid[top : top + h, left + w - 1] = GREY

    slots = [(r, c) for r in range(top + 1, top + h - 1) for c in range(left + 1, left + w - 1)]
    RNG.shuffle(slots)
    for color in target_colors:
        r, c = slots.pop()
        grid[r, c] = color
    for r, c in slots:
        if RNG.random() < 0.4:
            grid[r, c] = int(target_colors[RNG.integers(len(target_colors))])
    occupied[top - 1 : top + h + 1, left - 1 : left + w + 1] = True


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _gen_attempt() -> np.ndarray | None:
    H = int(RNG.integers(17, 28))
    W = int(RNG.integers(17, 28))
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    target_count = 2 if RNG.random() < 0.5 else 3
    target_colors = list(RNG.choice(PALETTE, size=target_count, replace=False))
    _make_legend(grid, occupied, target_colors)

    used_survivor_areas: set[int] = set()
    num_deleted = int(RNG.integers(1, 4))
    num_survivors = int(RNG.integers(2, 5))

    objects = []
    for deleted in [True] * num_deleted + [False] * num_survivors:
        obj, area = _build_object(target_colors, deleted, used_survivor_areas)
        if obj is None:
            return None
        if not deleted:
            used_survivor_areas.add(int(area))
        objects.append(obj)
    RNG.shuffle(objects)

    for obj in objects:
        if not _place(grid, occupied, obj):
            return None

    try:
        out = _expected_output(grid)
    except AssertionError:
        return None

    if out.shape[0] != num_survivors or out.shape[1] != 1:
        return None
    if len(set(int(v) for v in out[:, 0])) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
