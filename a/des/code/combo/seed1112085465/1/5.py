import random
from collections import deque

import numpy as np


BG = 0
FRAME = 3
DIRS = ("up", "right", "down", "left")
DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _shape_from_pattern(pattern: tuple[str, ...]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, row in enumerate(pattern)
        for c, ch in enumerate(row)
        if ch == "#"
    }


def _rotate_cw(coords: set[tuple[int, int]], height: int, width: int) -> tuple[set[tuple[int, int]], int, int]:
    rotated = {(c, height - 1 - r) for r, c in coords}
    return rotated, width, height


def _direction_from_cells(coords: list[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    centroid_r = sum(r for r, _ in coords) / len(coords)
    centroid_c = sum(c for _, c in coords) / len(coords)
    dr = centroid_r - center_r
    dc = centroid_c - center_c
    assert abs(dr) > 1e-9 or abs(dc) > 1e-9, "sprite centroid must not match bbox center"
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _build_sprite_library() -> dict[str, list[list[tuple[int, int]]]]:
    base_up_patterns = [
        ("###", ".#."),
        ("###", ".#.", ".#."),
        ("###", "..#", "..#"),
        ("##.", ".##", ".#."),
        (".##", "##.", ".#."),
    ]
    library = {direction: [] for direction in DIRS}
    for pattern in base_up_patterns:
        coords = _shape_from_pattern(pattern)
        current = coords
        h = len(pattern)
        w = len(pattern[0])
        for _ in range(4):
            normalized = sorted(current)
            direction = _direction_from_cells(normalized)
            library[direction].append(normalized)
            current, h, w = _rotate_cw(current, h, w)
    for direction in DIRS:
        dedup = []
        seen = set()
        for sprite in library[direction]:
            key = tuple(sprite)
            if key not in seen:
                seen.add(key)
                dedup.append(sprite)
        assert dedup, f"missing templates for direction {direction}"
        library[direction] = dedup
    return library


SPRITE_LIBRARY = _build_sprite_library()


def _ray_anchor_and_cells(
    frame_bbox: tuple[int, int, int, int],
    sprite_bbox: tuple[int, int, int, int],
    direction: str,
    length: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    sr0, sc0, sr1, sc1 = sprite_bbox
    anchor_r = (sr0 + sr1) // 2
    anchor_c = (sc0 + sc1) // 2
    dr, dc = DIR_VECS[direction]
    if direction == "up":
        start_r, start_c = r0 - 1, anchor_c
    elif direction == "down":
        start_r, start_c = r1 + 1, anchor_c
    elif direction == "left":
        start_r, start_c = anchor_r, c0 - 1
    else:
        start_r, start_c = anchor_r, c1 + 1

    cells = []
    rr, cc = start_r, start_c
    for _ in range(length):
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each sprite's direction from its centroid bias inside a green frame,
    then cast an outward ray whose length equals the sprite's pixel count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    frame_components = _components(g == FRAME)
    assert frame_components, "expected at least one green frame"

    out = g.copy()
    for comp in frame_components:
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose interior space"
        expected_border = _frame_border(r0, c0, r1, c1)
        assert set(comp) == expected_border, "green component must be a 1-pixel rectangular frame"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        sprite_mask = (interior != BG) & (interior != FRAME)
        sprite_local = [tuple(map(int, rc)) for rc in np.argwhere(sprite_mask)]
        assert sprite_local, "frame interior must contain one sprite"

        sprite_colors = np.unique(interior[sprite_mask])
        assert sprite_colors.size == 1, "sprite must be monochromatic"
        sprite_color = int(sprite_colors[0])

        sprite_coords = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in sprite_local]
        assert 2 <= len(sprite_coords) <= 5, "sprite size must be between 2 and 5"
        direction = _direction_from_cells(sprite_coords)
        sprite_bbox = _bbox(sprite_coords)
        ray_cells = _ray_anchor_and_cells((r0, c0, r1, c1), sprite_bbox, direction, len(sprite_coords))

        for rr, cc in ray_cells:
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = sprite_color

    return out


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
FRAME_COUNT_CHOICES = [1, 2, 3]
FRAME_COUNT_WEIGHTS = [2, 3, 2]
INNER_EXTRA_RANGE = (2, 4)
PLACEMENT_RETRIES = 200
GENERATE_RETRIES = 400
COLOR_POOL = [1, 2, 4, 5, 6, 7, 8, 9]

_GEN_COUNTER = 0


def _halo(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    halo = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    halo.add((rr, cc))
    return halo


def _rect_cells(r0: int, c0: int, r1: int, c1: int, height: int, width: int) -> set[tuple[int, int]]:
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(height - 1, r1)
    cc1 = min(width - 1, c1)
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _direction_cycle(n_frames: int) -> list[str]:
    global _GEN_COUNTER
    offset = _GEN_COUNTER % len(DIRS)
    cycle = list(DIRS[offset:] + DIRS[:offset])
    chosen = cycle[:n_frames]
    random.shuffle(chosen)
    return chosen


def _place_object(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    direction: str,
    color: int,
) -> tuple[set[tuple[int, int]], int] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        sprite = random.choice(SPRITE_LIBRARY[direction])
        sprite_h = max(r for r, _ in sprite) + 1
        sprite_w = max(c for _, c in sprite) + 1
        inner_h = sprite_h + random.randint(*INNER_EXTRA_RANGE)
        inner_w = sprite_w + random.randint(*INNER_EXTRA_RANGE)
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        length = len(sprite)

        min_r = 0
        max_r = h - frame_h
        min_c = 0
        max_c = w - frame_w
        if direction == "up":
            min_r = length
        elif direction == "down":
            max_r = h - frame_h - length
        elif direction == "left":
            min_c = length
        else:
            max_c = w - frame_w - length
        if min_r > max_r or min_c > max_c:
            continue

        frame_r0 = random.randint(min_r, max_r)
        frame_c0 = random.randint(min_c, max_c)
        frame_r1 = frame_r0 + frame_h - 1
        frame_c1 = frame_c0 + frame_w - 1

        sprite_r0 = frame_r0 + 1 + random.randint(1, inner_h - sprite_h - 1)
        sprite_c0 = frame_c0 + 1 + random.randint(1, inner_w - sprite_w - 1)

        sprite_cells = {(sprite_r0 + r, sprite_c0 + c) for r, c in sprite}
        sprite_bbox = _bbox(list(sprite_cells))
        frame_cells = _frame_border(frame_r0, frame_c0, frame_r1, frame_c1)
        ray_cells = set(
            _ray_anchor_and_cells((frame_r0, frame_c0, frame_r1, frame_c1), sprite_bbox, direction, length)
        )

        if any(not (0 <= r < h and 0 <= c < w) for r, c in ray_cells):
            continue

        drawn = frame_cells | sprite_cells | ray_cells
        if len(drawn) != len(frame_cells) + len(sprite_cells) + len(ray_cells):
            continue

        protected = _rect_cells(frame_r0 - 1, frame_c0 - 1, frame_r1 + 1, frame_c1 + 1, h, w)
        protected |= _halo(ray_cells, h, w)
        if protected & forbidden:
            continue

        for r, c in frame_cells:
            grid[r, c] = FRAME
        for r, c in sprite_cells:
            grid[r, c] = color
        return protected, len(ray_cells)
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    n_frames = random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]

    directions = _direction_cycle(n_frames)
    colors = random.sample(COLOR_POOL, k=n_frames)
    forbidden: set[tuple[int, int]] = set()
    total_ray_pixels = 0

    for direction, color in zip(directions, colors):
        result = _place_object(grid, forbidden, direction, color)
        if result is None:
            return None
        protected, ray_len = result
        forbidden |= protected
        total_ray_pixels += ray_len

    out = main(grid)
    changed = int((out != grid).sum())
    if changed != total_ray_pixels:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate valid input")
