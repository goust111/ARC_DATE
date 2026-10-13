import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 160


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue

            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components


def _slide_component(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    moved: set[tuple[int, int]] = set()
    for r, c in coords:
        has_support = any(rr > r and cc > c for rr, cc in coords)
        moved.add((r, c + 1) if has_support else (r, c))
    return moved


def _right_contour(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(
        [(r, c) for r, c in coords if not any(rr == r and cc > c for rr, cc in coords)],
        key=lambda rc: (rc[0], rc[1]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each object rightward by southeast support, then emit rightward rays from the slid right contour.

    Input structure:
    - The grid is a black background with 2-4 small monochrome 4-connected objects in colors 1-4.
    - Objects lie in the left half of the canvas, are separated from one another, and the right half starts empty.
    - Each object has some pixels with a same-object pixel strictly down-right of them and some without.

    Rule:
    - Process each monochrome 4-connected object independently. A pixel moves one column right iff the object contains
      another pixel with both larger row and larger column; otherwise it stays put. Moved pixels merge on collision.
    - After sliding, mark the object's right-contour pixels: those with no same-object pixel further right on the same row.
    - Paint the slid objects onto the output, then fire horizontal rays to the right from each contour pixel. Rays are
      processed by ascending object color, then by ascending source row, and fill black cells up to the right border.
      A ray stops immediately before any existing non-black pixel of a different color.

    Key insight:
    - The southeast-support slide is not just a deformation step: it changes which pixels remain on the post-slide right
      contour, and only those surviving contour pixels become ray sources.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    ray_sources: list[tuple[int, list[tuple[int, int]]]] = []
    for component in sorted(_find_components(g), key=lambda comp: comp["color"]):
        color = int(component["color"])
        slid = _slide_component(component["coords"])
        for r, c in slid:
            out[r, c] = color
        ray_sources.append((color, _right_contour(slid)))

    h, w = out.shape
    for color, sources in ray_sources:
        for r, c in sources:
            for nc in range(c + 1, w):
                cell = int(out[r, nc])
                if cell not in (BLACK, color):
                    break
                out[r, nc] = color

    return out


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _shape_has_movable_and_fixed(coords: list[tuple[int, int]]) -> bool:
    flags = [any(rr > r and cc > c for rr, cc in coords) for r, c in coords]
    return any(flags) and not all(flags)


def _sprite_from_coords(coords: list[tuple[int, int]], color: int = 1) -> np.ndarray:
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    sprite = np.zeros((max_r + 1, max_c + 1), dtype=int)
    for r, c in coords:
        sprite[r, c] = color
    return sprite


def _random_shape(rng: random.Random) -> np.ndarray:
    bbox_choices = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3)]

    for _ in range(SHAPE_ATTEMPTS):
        target = rng.randint(4, 8)
        viable = [shape for shape in bbox_choices if shape[0] * shape[1] >= target + 1]
        h, w = rng.choice(viable)

        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != target:
            continue

        coords = _normalize(cells)
        rows = {r for r, _ in coords}
        cols = {c for _, c in coords}
        area = (max(rows) + 1) * (max(cols) + 1)
        if len(rows) < 2 or len(cols) < 2:
            continue
        if area == len(coords):
            continue
        if not _shape_has_movable_and_fixed(coords):
            continue

        slid = _slide_component(coords)
        if len(_right_contour(slid)) == 0:
            continue
        return _sprite_from_coords(coords)

    raise RuntimeError("failed to generate a valid object shape")


def _can_place(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, padding: int = 2) -> bool:
    h, w = grid.shape
    sh, sw = sprite.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False

    sprite_coords = np.argwhere(sprite != BLACK)
    if sprite_coords.size == 0:
        return False
    if max(int(left + c) for _, c in sprite_coords) >= w / 2:
        return False

    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + sh + padding)
    c1 = min(w, left + sw + padding)
    if np.any(grid[r0:r1, c0:c1] != BLACK):
        return False

    return True


def _paste(grid: np.ndarray, sprite: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(sprite != BLACK)
    for r, c in coords:
        grid[top + int(r), left + int(c)] = color


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 16)
    w = rng.randint(max(h + 3, 16), min(24, h + 10))
    n_objects = rng.choice((2, 2, 3, 3, 4))
    colors = rng.sample(PALETTE, n_objects)

    grid = np.zeros((h, w), dtype=int)
    align_rows = rng.random() < 0.5
    anchor_rows: list[int] = []

    for index, color in enumerate(colors):
        sprite = _random_shape(rng)
        sh, sw = sprite.shape

        candidates: list[tuple[int, int]] = []
        for top in range(1, h - sh):
            for left in range(1, w - sw):
                if _can_place(grid, sprite, top, left):
                    candidates.append((top, left))
        if not candidates:
            return None

        if align_rows and anchor_rows and rng.random() < 0.7:
            preferred = [
                pos
                for pos in candidates
                if min(abs(pos[0] - base) for base in anchor_rows) <= 2
            ]
            if preferred:
                candidates = preferred

        top, left = rng.choice(candidates[:PLACEMENT_ATTEMPTS] if len(candidates) > PLACEMENT_ATTEMPTS else candidates)
        _paste(grid, sprite, top, left, color)
        anchor_rows.append(top + sh // 2)

        if index == 0 and rng.random() < 0.5:
            anchor_rows.append(top)

    components = sorted(_find_components(grid), key=lambda comp: comp["color"])
    if len(components) != n_objects:
        return None
    if len({comp["color"] for comp in components}) != n_objects:
        return None

    for comp in components:
        coords = comp["coords"]
        if not (4 <= len(coords) <= 8):
            return None
        if not _shape_has_movable_and_fixed(coords):
            return None
        if max(c for _, c in coords) >= w / 2:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any(out[:, int(np.ceil(w / 2)) :] != BLACK):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
