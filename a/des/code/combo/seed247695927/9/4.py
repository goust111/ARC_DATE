import random
from collections import Counter, deque

import numpy as np


NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_RETRIES = 400
_NEXT_SPECIAL = False


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _component_patch(grid: np.ndarray, coords: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch.astype(int) * int(color), (r0, c0, r1, c1)


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _paste(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = out[r0:r0 + h, c0:c0 + w]
    mask = patch != bg
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored object vertically until it touches the matching top or bottom border bar.

    Input structure
    - The background is the dominant color.
    - The top row is a full-width bar of one foreground color and the bottom row is a full-width bar of another.
    - Every other foreground pixel belongs to an 8-connected monochrome object whose color matches one of the bars.

    Rule
    - Objects of the top-bar color move upward so their bounding-box top lands on row 1.
    - Objects of the bottom-bar color move downward so their bounding-box bottom lands on row H-2.
    - A single-column downward mover whose destination overlaps its source becomes one continuous vertical segment.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 4 and w >= 3

    bg = _dominant_color(g)
    top_color = int(g[0, 0])
    bottom_color = int(g[h - 1, 0])
    assert np.all(g[0] == top_color)
    assert np.all(g[h - 1] == bottom_color)
    assert top_color != bg and bottom_color != bg and top_color != bottom_color

    interior = g[1:h - 1]
    allowed = {bg, top_color, bottom_color}
    assert set(map(int, np.unique(g))).issubset(allowed)

    objects: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []
    for color in (top_color, bottom_color):
        mask = interior == color
        for local_coords in _find_components(mask):
            coords = local_coords.copy()
            coords[:, 0] += 1
            patch, bbox = _component_patch(g, coords, color)
            objects.append((int(color), patch, bbox))

    out = np.full_like(g, bg)
    out[0, :] = top_color
    out[h - 1, :] = bottom_color

    for color, patch, bbox in objects:
        r0, c0, r1, c1 = bbox
        ph, pw = patch.shape
        if color == top_color:
            dest_r = 1
            _paste(out, patch, dest_r, c0, bg)
            continue

        dest_r = h - 1 - ph
        if pw == 1:
            dest_r1 = dest_r + ph - 1
            if dest_r <= r1:
                out[r0:dest_r1 + 1, c0] = color
                continue
        _paste(out, patch, dest_r, c0, bg)

    return out


def _normalize(cells: set[tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = min(rs)
    c0 = min(cs)
    h = max(rs) - r0 + 1
    w = max(cs) - c0 + 1
    patch = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        patch[r - r0, c - c0] = True
    return patch


def _random_connected_patch(height: int, width: int, area: int) -> np.ndarray:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return _random_connected_patch(height, width, area)
        cells.add(random.choice(tuple(frontier)))
    patch = _normalize(cells)
    if patch.shape[0] > height or patch.shape[1] > width:
        return _random_connected_patch(height, width, area)
    return patch


def _sample_patch(max_h: int, max_w: int, *, allow_single_col: bool) -> np.ndarray:
    shape_mode = random.choice(["line", "compact", "irregular"])
    if allow_single_col and random.random() < 0.3:
        h = random.randint(1, max_h)
        return np.ones((h, 1), dtype=bool)

    if shape_mode == "line":
        if random.random() < 0.5:
            h = random.randint(1, max_h)
            w = random.randint(2, max_w)
            return np.ones((h, w), dtype=bool) if h == 1 else _random_connected_patch(h, w, random.randint(max(w, h), h * w))
        w = random.randint(1, max_w)
        h = random.randint(2, max_h)
        return np.ones((h, 1), dtype=bool) if w == 1 else _random_connected_patch(h, w, random.randint(max(w, h), h * w))

    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    area_min = 1 if min(h, w) == 1 else max(2, (h * w + 1) // 2)
    area = random.randint(area_min, h * w)
    return _random_connected_patch(h, w, area)


def _can_place(patch: np.ndarray, top: int, left: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for pr, pc in np.argwhere(patch):
        rr = top + int(pr)
        cc = left + int(pc)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, color: int, top: int, left: int, occupied: np.ndarray) -> None:
    for pr, pc in np.argwhere(patch):
        rr = top + int(pr)
        cc = left + int(pc)
        grid[rr, cc] = color
        occupied[rr, cc] = True


def _build_expected(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _component_count(grid: np.ndarray, color: int) -> int:
    return len(_find_components(grid[1:-1] == color))


def _gen_attempt(force_special: bool) -> np.ndarray | None:
    h = random.randint(9, 16)
    w = random.randint(8, 16)
    colors = random.sample(range(10), 3)
    bg, top_color, bottom_color = colors

    grid = np.full((h, w), bg, dtype=int)
    grid[0, :] = top_color
    grid[h - 1, :] = bottom_color
    occupied = np.zeros((h, w), dtype=bool)

    top_count = random.randint(1, 4)
    bottom_count = random.randint(1, 4)
    if force_special:
        bottom_count = max(1, bottom_count)

    placed_bottom_special = False
    bottom_specs: list[tuple[np.ndarray, int, int]] = []

    if force_special:
        max_special_h = max(2, min(5, (h - 3) // 2))
        if max_special_h < 2:
            return None
        special_h = random.randint(2, max_special_h)
        low = max(2, h - 2 * special_h)
        high = h - special_h - 2
        if low > high:
            return None
        special_top = random.randint(low, high)
        special_col = random.randint(0, w - 1)
        bottom_specs.append((np.ones((special_h, 1), dtype=bool), special_top, special_col))
        placed_bottom_special = True

    for _ in range(top_count):
        patch = _sample_patch(min(4, h - 4), min(4, w), allow_single_col=True)
        ph, pw = patch.shape
        if ph > h - 4 or pw > w:
            return None
        tries = 0
        placed = False
        while tries < 80 and not placed:
            tries += 1
            top = random.randint(2, max(2, h // 2))
            if top + ph - 1 > h - 3:
                continue
            left = random.randint(0, w - pw)
            if _can_place(patch, top, left, occupied):
                _place_patch(grid, patch, top_color, top, left, occupied)
                placed = True
        if not placed:
            return None

    for patch, top, left in bottom_specs:
        if not _can_place(patch, top, left, occupied):
            return None
        _place_patch(grid, patch, bottom_color, top, left, occupied)

    extra_bottom = bottom_count - len(bottom_specs)
    for _ in range(extra_bottom):
        patch = _sample_patch(min(5, h - 4), min(4, w), allow_single_col=True)
        ph, pw = patch.shape
        if ph > h - 4 or pw > w:
            return None
        tries = 0
        placed = False
        while tries < 80 and not placed:
            tries += 1
            top_max = h - ph - 2
            if top_max < 2:
                return None
            top = random.randint(2, top_max)
            left = random.randint(0, w - pw)
            if _can_place(patch, top, left, occupied):
                _place_patch(grid, patch, bottom_color, top, left, occupied)
                placed = True
        if not placed:
            return None

    out = _build_expected(grid)
    if np.array_equal(out, grid):
        return None

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= counts[top_color] or counts[bg] <= counts[bottom_color]:
        return None

    if _component_count(grid, top_color) != top_count:
        return None
    if _component_count(grid, bottom_color) != bottom_count:
        return None
    if _component_count(out, top_color) != top_count:
        return None
    if _component_count(out, bottom_color) != bottom_count:
        return None

    if force_special:
        has_special = False
        for c in range(w):
            runs_in = np.flatnonzero(grid[1:h - 1, c] == bottom_color)
            runs_out = np.flatnonzero(out[1:h - 1, c] == bottom_color)
            if runs_in.size >= 2 and runs_out.size > runs_in.size:
                has_special = True
                break
        if not has_special:
            return None

    # Reject accidental touching/degenerate full interior rows.
    interior = grid[1:h - 1]
    if any(np.all(interior[r] == top_color) or np.all(interior[r] == bottom_color) for r in range(interior.shape[0])):
        return None

    return grid


def generate_input() -> np.ndarray:
    global _NEXT_SPECIAL
    force_special = _NEXT_SPECIAL
    _NEXT_SPECIAL = not _NEXT_SPECIAL

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(force_special)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
