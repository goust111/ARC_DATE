import random
from collections import Counter

import numpy as np


BG_COLOR = 0
RED_COLOR = 2
CROSS_COLOR = 5
ARROW_COLORS = [1, 3, 4, 6, 7, 8, 9]
GRID_SIZE_CHOICES = (9, 11, 13, 15, 17, 19)
MAX_GENERATION_ATTEMPTS = 300

DIRECTIONS = ("up", "down", "left", "right")
DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_TEMPLATES = {
    "up": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    "down": ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    "left": ((0, 0), (1, 0), (1, 1), (1, 2), (2, 0)),
    "right": ((0, 2), (1, 0), (1, 1), (1, 2), (2, 2)),
}
NORMALIZED_TEMPLATES = {
    direction: frozenset(template) for direction, template in ARROW_TEMPLATES.items()
}


def _most_common_color(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(values))


def _orientation_from_coords(coords: list[tuple[int, int]]) -> str:
    assert len(coords) == 5, f"arrow must contain 5 pixels, got {len(coords)}"
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    assert r1 - r0 == 2 and c1 - c0 == 2, f"arrow bbox must be 3x3, got {(r0, c0, r1, c1)}"
    normalized = frozenset((r - r0, c - c0) for r, c in coords)
    for direction, template in NORMALIZED_TEMPLATES.items():
        if normalized == template:
            return direction
    raise AssertionError(f"unrecognized arrow shape: {sorted(normalized)}")


def _extract_arrow_counts(grid: np.ndarray) -> tuple[Counter, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1, "grid must be an odd-sized square"

    bg = _most_common_color(g)
    mid = h // 2
    cross_color = int(g[mid, mid])
    assert np.all(g[mid, :] == cross_color), "middle row must be the cross"
    assert np.all(g[:, mid] == cross_color), "middle column must be the cross"

    visited = np.zeros((h, w), dtype=bool)
    counts: Counter[str] = Counter()

    for r in range(h):
        for c in range(w):
            if visited[r, c] or r == mid or c == mid:
                continue
            color = int(g[r, c])
            if color == bg:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR_STEPS.values():
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or nr == mid or nc == mid:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            counts[_orientation_from_coords(coords)] += 1

    assert sum(counts.values()) > 0, "expected at least one arrow"
    return counts, bg, cross_color, mid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the scattered 5-pixel arrows by orientation and stack those counts on the central cross.
    """
    g = np.asarray(grid, dtype=int)
    counts, bg, cross_color, mid = _extract_arrow_counts(g)

    out = np.full_like(g, bg)
    out[mid, :] = cross_color
    out[:, mid] = cross_color

    for direction in DIRECTIONS:
        dr, dc = DIR_STEPS[direction]
        for step in range(counts[direction]):
            rr = mid + dr * step
            cc = mid + dc * step
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "arrow count exceeds arm length"
            out[rr, cc] = RED_COLOR

    return out


def _candidate_boxes(size: int) -> list[tuple[int, int]]:
    mid = size // 2
    rows = [r0 for r0 in range(size - 2) if r0 <= mid - 4 or r0 >= mid + 2]
    cols = [c0 for c0 in range(size - 2) if c0 <= mid - 4 or c0 >= mid + 2]
    return [(r0, c0) for r0 in rows for c0 in cols]


def _sample_non_touching_boxes(size: int) -> list[tuple[int, int]]:
    candidates = _candidate_boxes(size)
    random.shuffle(candidates)

    forbidden = np.zeros((size, size), dtype=bool)
    mid = size // 2
    forbidden[mid, :] = True
    forbidden[:, mid] = True

    boxes: list[tuple[int, int]] = []
    for r0, c0 in candidates:
        r_lo = max(0, r0 - 1)
        r_hi = min(size, r0 + 4)
        c_lo = max(0, c0 - 1)
        c_hi = min(size, c0 + 4)
        if np.any(forbidden[r_lo:r_hi, c_lo:c_hi]):
            continue
        boxes.append((r0, c0))
        forbidden[r_lo:r_hi, c_lo:c_hi] = True

    return boxes


def _sample_directions(count: int) -> list[str]:
    if count == 1 or random.random() < 0.35:
        direction = random.choice(DIRECTIONS)
        return [direction] * count

    directions = [random.choice(DIRECTIONS) for _ in range(count)]
    if len(set(directions)) == 1:
        idx = random.randrange(count)
        other_dirs = [d for d in DIRECTIONS if d != directions[idx]]
        directions[idx] = random.choice(other_dirs)
    random.shuffle(directions)
    return directions


def _validate_generated(grid: np.ndarray, expected: Counter) -> bool:
    try:
        observed, bg, cross_color, mid = _extract_arrow_counts(grid)
    except AssertionError:
        return False

    if observed != expected or bg != BG_COLOR or cross_color != CROSS_COLOR:
        return False

    out = main(grid)
    cross_mask = np.zeros_like(grid, dtype=bool)
    cross_mask[mid, :] = True
    cross_mask[:, mid] = True
    if not np.all(out[~cross_mask] == BG_COLOR):
        return False

    for direction in DIRECTIONS:
        dr, dc = DIR_STEPS[direction]
        for step in range(mid + 1):
            rr = mid + dr * step
            cc = mid + dc * step
            expected_color = RED_COLOR if step < observed[direction] else CROSS_COLOR
            if int(out[rr, cc]) != expected_color:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(GRID_SIZE_CHOICES)
    grid = np.full((size, size), BG_COLOR, dtype=int)
    mid = size // 2
    grid[mid, :] = CROSS_COLOR
    grid[:, mid] = CROSS_COLOR

    boxes = _sample_non_touching_boxes(size)
    max_arrows = min(mid + 1, len(boxes))
    if max_arrows == 0:
        return None

    arrow_count = random.randint(1, max_arrows)
    selected_boxes = random.sample(boxes, arrow_count)
    directions = _sample_directions(arrow_count)

    palette_size = random.randint(1, min(4, len(ARROW_COLORS)))
    palette = random.sample(ARROW_COLORS, palette_size)
    colors = [random.choice(palette) for _ in range(arrow_count)]

    expected = Counter()
    for (r0, c0), direction, color in zip(selected_boxes, directions, colors):
        expected[direction] += 1
        for dr, dc in ARROW_TEMPLATES[direction]:
            rr = r0 + dr
            cc = c0 + dc
            if int(grid[rr, cc]) != BG_COLOR:
                return None
            grid[rr, cc] = color

    if not _validate_generated(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
