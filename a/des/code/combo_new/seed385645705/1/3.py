from __future__ import annotations

import random
from collections import deque

import numpy as np


FENCE_COLOR = 2
SEED_COLORS = (1, 3)
BACKGROUND_CHOICES = (0, 4, 5, 6, 7, 8, 9)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    assert int(np.sum(counts == max_count)) == 1, "background color must be dominant"
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _line_from_component(coords: np.ndarray) -> dict[str, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        r = int(rows[0])
        c0, c1 = int(cols.min()), int(cols.max())
        assert len(coords) >= 3 and len(coords) == c1 - c0 + 1, "red component must be a solid line"
        return {"orient": 0, "r": r, "c0": c0, "c1": c1}
    if np.all(cols == cols[0]):
        c = int(cols[0])
        r0, r1 = int(rows.min()), int(rows.max())
        assert len(coords) >= 3 and len(coords) == r1 - r0 + 1, "red component must be a solid line"
        return {"orient": 1, "c": c, "r0": r0, "r1": r1}
    raise AssertionError("red fence must be horizontal or vertical")


def _side_seed(
    grid: np.ndarray,
    bg: int,
    line: dict[str, int],
    side: int,
) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]]] | None:
    h, w = grid.shape
    forbidden = {bg, FENCE_COLOR}
    if line["orient"] == 0:
        r = line["r"]
        sr = r + side
        mr = r - side
        if not (0 <= sr < h and 0 <= mr < h):
            return None
        c0, c1 = line["c0"], line["c1"]
        cols = [c for c in range(c0, c1 + 1) if int(grid[sr, c]) not in forbidden]
        if not cols:
            return None
        lo, hi = min(cols), max(cols)
        assert cols == list(range(lo, hi + 1)), "seed segment must be contiguous"
        seed = [(sr, c) for c in cols]
        mirror = [(mr, c) for c in cols]
    else:
        c = line["c"]
        sc = c + side
        mc = c - side
        if not (0 <= sc < w and 0 <= mc < w):
            return None
        r0, r1 = line["r0"], line["r1"]
        rows = [r for r in range(r0, r1 + 1) if int(grid[r, sc]) not in forbidden]
        if not rows:
            return None
        lo, hi = min(rows), max(rows)
        assert rows == list(range(lo, hi + 1)), "seed segment must be contiguous"
        seed = [(r, sc) for r in rows]
        mirror = [(r, mc) for r in rows]

    values = [int(grid[r, c]) for r, c in seed]
    assert all(v not in forbidden for v in values), "seed colors must differ from background and fence"
    return seed, values, mirror


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect every seed segment across its red fence while swapping the two seed colors.

    Input structure:
    - The background is the dominant color.
    - Red pixels form one or more horizontal or vertical one-cell-thick fence
      segments of length at least three.
    - Exactly one side of each fence has an adjacent parallel seed segment made
      from two alternating colors such as blue and green, fully within the
      fence span.

    Rule:
    - Keep the background, red fences, and original seed segments unchanged.
    - For each fence, place a same-length segment in the symmetric cells on the
      other side of the fence.
    - Each mirrored pixel receives the other one of the two seed colors.

    Key insight:
    - The red fence gives the spatial reflection axis, while the alternating
      seed colors reveal that the reflected copy must also invert the color
      pair instead of being copied directly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0

    bg = _background_color(g)
    red_components = _components(g == FENCE_COLOR)
    assert red_components, "expected at least one red fence"

    tasks = []
    all_seed_colors: set[int] = set()
    for comp in red_components:
        line = _line_from_component(comp)
        candidates = []
        for side in (-1, 1):
            seed = _side_seed(g, bg, line, side)
            if seed is not None:
                candidates.append(seed)
        assert len(candidates) == 1, "each fence must have exactly one adjacent seed side"
        seed_cells, values, mirror_cells = candidates[0]
        assert len(seed_cells) <= len(comp), "seed cannot be longer than its fence"
        all_seed_colors.update(values)
        tasks.append((values, mirror_cells))

    assert len(all_seed_colors) == 2, "expected exactly two seed colors"
    a, b = tuple(all_seed_colors)
    swap = {a: b, b: a}

    out = g.copy()
    for values, mirror_cells in tasks:
        for value, (r, c) in zip(values, mirror_cells):
            assert int(g[r, c]) == bg, "mirrored seed position must be background"
            out[r, c] = swap[int(value)]
    return out


# --------------- generator ---------------

GRID_SIZE_RANGE = (12, 24)
OBJECT_COUNT_RANGE = (1, 5)
FENCE_LENGTH_RANGE = (3, 9)
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 120
OBJECT_PADDING = 1


def _neighbors_with_padding(cells: set[tuple[int, int]], h: int, w: int, padding: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(max(0, r - padding), min(h, r + padding + 1)):
            for cc in range(max(0, c - padding), min(w, c + padding + 1)):
                out.add((rr, cc))
    return out


def _can_place(reserved: np.ndarray, cells: set[tuple[int, int]], padding: int) -> bool:
    h, w = reserved.shape
    if not cells:
        return False
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
    for r, c in _neighbors_with_padding(cells, h, w, padding):
        if reserved[r, c]:
            return False
    return True


def _attempt_place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    orientation: str,
) -> dict[str, list[tuple[int, int]] | list[int]] | None:
    h, w = grid.shape
    max_len = min(FENCE_LENGTH_RANGE[1], w - 2 if orientation == "h" else h - 2)
    if max_len < FENCE_LENGTH_RANGE[0]:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        length = random.randint(FENCE_LENGTH_RANGE[0], max_len)
        seed_len = random.randint(2, length)
        offset = random.randint(0, length - seed_len)
        side = random.choice((-1, 1))
        start_color = random.choice(SEED_COLORS)
        other_color = SEED_COLORS[1] if start_color == SEED_COLORS[0] else SEED_COLORS[0]
        seed_values = [start_color if i % 2 == 0 else other_color for i in range(seed_len)]

        if orientation == "h":
            if h < 5 or w - length - 1 < 1:
                return None
            r = random.randint(2, h - 3)
            c0 = random.randint(1, w - length - 1)
            fence = [(r, c) for c in range(c0, c0 + length)]
            seed = [(r + side, c0 + offset + i) for i in range(seed_len)]
            mirror = [(r - side, c0 + offset + i) for i in range(seed_len)]
        else:
            if w < 5 or h - length - 1 < 1:
                return None
            c = random.randint(2, w - 3)
            r0 = random.randint(1, h - length - 1)
            fence = [(r, c) for r in range(r0, r0 + length)]
            seed = [(r0 + offset + i, c + side) for i in range(seed_len)]
            mirror = [(r0 + offset + i, c - side) for i in range(seed_len)]

        footprint = set(fence) | set(seed) | set(mirror)
        if not _can_place(reserved, footprint, OBJECT_PADDING):
            continue

        for r, c in fence:
            grid[r, c] = FENCE_COLOR
        for (r, c), value in zip(seed, seed_values):
            grid[r, c] = value
        for r, c in _neighbors_with_padding(footprint, h, w, OBJECT_PADDING):
            reserved[r, c] = True
        return {"values": seed_values, "mirror": mirror}
    return None


def _build_expected(grid: np.ndarray, objects: list[dict[str, list[tuple[int, int]] | list[int]]]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        values = obj["values"]
        mirror = obj["mirror"]
        for value, (r, c) in zip(values, mirror):
            out[r, c] = SEED_COLORS[1] if value == SEED_COLORS[0] else SEED_COLORS[0]
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    min_side = max(GRID_SIZE_RANGE[0], 8 + 2 * n_objects)
    h = random.randint(min_side, GRID_SIZE_RANGE[1])
    w = random.randint(min_side, GRID_SIZE_RANGE[1])
    bg = random.choice(BACKGROUND_CHOICES)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    orientations = [random.choice(("h", "v")) for _ in range(n_objects)]
    if n_objects >= 2 and random.random() < 0.65:
        orientations[0:2] = ["h", "v"]
        random.shuffle(orientations)

    objects = []
    for orientation in orientations:
        obj = _attempt_place_object(grid, reserved, orientation)
        if obj is None:
            return None
        objects.append(obj)

    if _background_color(grid) != bg:
        return None
    expected = _build_expected(grid, objects)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed385645705/1/3 input")
