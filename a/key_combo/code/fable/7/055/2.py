import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
BORDER_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ROTATE_DIR = {0: "up", 1: "right", 2: "down", 3: "left"}
SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def label_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def random_shape(rng: random.Random) -> np.ndarray:
    mask = SHAPES[rng.randrange(len(SHAPES))].copy()
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    k = rng.randrange(4)
    if k:
        mask = np.rot90(mask, k)
    return normalize_mask(mask)


def side_colors(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 5 and w >= 5
    top_vals = set(map(int, g[0, 1:-1]))
    bottom_vals = set(map(int, g[h - 1, 1:-1]))
    left_vals = set(map(int, g[1:-1, 0]))
    right_vals = set(map(int, g[1:-1, w - 1]))
    assert len(top_vals) == 1 and len(bottom_vals) == 1
    assert len(left_vals) == 1 and len(right_vals) == 1
    top = top_vals.pop()
    bottom = bottom_vals.pop()
    left = left_vals.pop()
    right = right_vals.pop()
    assert len({top, bottom, left, right}) == 4
    return {"up": top, "down": bottom, "left": left, "right": right}


def vote_counts(grid: np.ndarray, colors: dict[str, int]) -> dict[str, int]:
    interior = np.asarray(grid, dtype=int)[1:-1, 1:-1]
    return {d: int(np.sum(interior == color)) for d, color in colors.items()}


def winner_direction(grid: np.ndarray) -> str:
    counts = vote_counts(grid, side_colors(grid))
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    assert ordered[0][1] > ordered[1][1]
    return ordered[0][0]


def component_distance(coords: np.ndarray, direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox_coords(coords)
    if direction == "up":
        return r0 - 1
    if direction == "down":
        return (h - 2) - r1
    if direction == "left":
        return c0 - 1
    return (w - 2) - c1


def sort_key(coords: np.ndarray, direction: str, shape: tuple[int, int]) -> tuple[int, int, int]:
    r0, c0, r1, c1 = bbox_coords(coords)
    dist = component_distance(coords, direction, shape)
    if direction in ("up", "down"):
        return dist, c0, r0
    return dist, r0, c0


def slide_component(
    coords: np.ndarray, occupied: np.ndarray, direction: str
) -> np.ndarray:
    dr, dc = DIRS[direction]
    cur = coords.copy()
    while True:
        nxt = cur + np.array([dr, dc], dtype=int)
        if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= occupied.shape[0]):
            break
        if np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= occupied.shape[1]):
            break
        if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
            break
        cur = nxt
    return cur


def clear_votes(grid: np.ndarray, colors: dict[str, int], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    palette = set(colors.values())
    for r in range(1, out.shape[0] - 1):
        for c in range(1, out.shape[1] - 1):
            if int(out[r, c]) in palette:
                out[r, c] = int(bg)
    return out


def simulate(grid: np.ndarray, direction: str, near_first: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    colors = side_colors(g)
    bg = get_bg(g)
    base = clear_votes(g, colors, bg)
    comps = label_components(base == GREY)

    out = base.copy()
    out[out == GREY] = bg
    occupied = out != bg

    ordered = sorted(comps, key=lambda cc: sort_key(cc, direction, g.shape), reverse=not near_first)
    for coords in ordered:
        settled = slide_component(coords, occupied, direction)
        out[settled[:, 0], settled[:, 1]] = GREY
        occupied[settled[:, 0], settled[:, 1]] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the interior color votes to choose one border wall as gravity, then slide all grey objects toward it.

    Input structure:
    - The outer border is a four-sided frame: the top row, bottom row, left edge, and right edge each use a different solid color.
    - Interior non-background pixels are either grey 8-connected objects or isolated single-pixel votes whose colors match one of the four border walls.
    - The border colors may reappear only as those interior vote pixels.

    Rule:
    - Count how many interior vote pixels use each border color and pick the unique strict majority color.
    - Erase every vote pixel.
    - Treat the matching border wall as the gravity direction and slide every grey object rigidly toward that wall.
    - Process grey objects from nearest to farthest from the chosen wall; each object stops just before the wall or a previously settled grey object.

    Key insight:
    - The colored singletons are a global vote, not local targets: geometry does not choose the direction, and the near-to-far settling order is necessary because earlier objects become obstacles for later ones.
    """
    return simulate(grid, winner_direction(grid), near_first=True)


def nearest_border_direction(coords: np.ndarray, shape: tuple[int, int]) -> str | None:
    h, w = shape
    r0, c0, r1, c1 = bbox_coords(coords)
    distances = {
        "up": r0 - 1,
        "down": (h - 2) - r1,
        "left": c0 - 1,
        "right": (w - 2) - c1,
    }
    best = min(distances.values())
    winners = [d for d, v in distances.items() if v == best]
    return winners[0] if len(winners) == 1 else None


def paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = int(color)


def can_place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if r0 < 1 or c0 < 1 or r0 + mh > h - 1 or c0 + mw > w - 1:
        return False
    coords = {(r0 + r, c0 + c) for r, c in np.argwhere(mask)}
    for rr, cc in coords:
        if grid[rr, cc] != BG:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    return False
                if (nr, nc) in coords:
                    continue
                if grid[nr, nc] != BG:
                    return False
    return True


def can_place_vote(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if grid[r, c] != BG:
        return False
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if grid[nr, nc] != BG and (nr, nc) != (r, c):
                return False
    return True


def add_frame(grid: np.ndarray, colors: tuple[int, int, int, int]) -> None:
    top, bottom, left, right = colors
    grid[0, :] = top
    grid[-1, :] = bottom
    grid[1:-1, 0] = left
    grid[1:-1, -1] = right


def vote_plan(rng: random.Random) -> dict[str, int]:
    winner = rng.randint(4, 6)
    others = [rng.randint(1, min(3, winner - 1)) for _ in range(3)]
    rng.shuffle(others)
    return {"up": winner, "down": others[0], "left": others[1], "right": others[2]}


def rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, -k).copy()


def validate_example(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    colors = side_colors(g)
    if GREY in colors.values() or BG in colors.values():
        return False

    interior = g[1:-1, 1:-1]
    allowed = {BG, GREY, *colors.values()}
    if not set(map(int, np.unique(interior))).issubset(allowed):
        return False

    counts = vote_counts(g, colors)
    if min(counts.values()) < 1:
        return False
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if ordered[0][1] <= ordered[1][1]:
        return False
    direction = ordered[0][0]

    base = clear_votes(g, colors, BG)
    comps = label_components(base == GREY)
    if len(comps) < 3:
        return False

    distances = [component_distance(comp, direction, g.shape) for comp in comps]
    if len(distances) != len(set(distances)):
        return False

    if not any(nearest_border_direction(comp, g.shape) not in (None, direction) for comp in comps):
        return False

    result = simulate(g, direction, near_first=True)
    if np.array_equal(result, base):
        return False
    if np.array_equal(result, simulate(g, direction, near_first=False)):
        return False
    for other in DIRS:
        if other != direction and np.array_equal(result, simulate(g, other, near_first=True)):
            return False

    grey_before = int(np.sum(base == GREY))
    grey_after = int(np.sum(result == GREY))
    if grey_before != grey_after:
        return False

    palette = set(colors.values())
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            val = int(g[r, c])
            if val in palette:
                if np.any(g[r - 1 : r + 2, c - 1 : c + 2] != BG) and np.count_nonzero(g[r - 1 : r + 2, c - 1 : c + 2]) != 1:
                    return False
    return True


def generate_canonical_up(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 18)
    w = rng.randint(12, 18)
    grid = np.full((h, w), BG, dtype=int)
    frame = tuple(rng.sample(BORDER_POOL, 4))
    add_frame(grid, frame)

    mask1 = random_shape(rng)
    mask2 = random_shape(rng)
    c_lane = rng.randint(2, w - max(mask1.shape[1], mask2.shape[1]) - 3)
    r1 = rng.randint(2, 4)
    r2_min = r1 + mask1.shape[0] + 2
    r2_max = h - mask2.shape[0] - 5
    if r2_min > r2_max:
        return None
    r2 = rng.randint(r2_min, r2_max)
    if not can_place_mask(grid, mask1, r1, c_lane):
        return None
    paint_mask(grid, mask1, r1, c_lane, GREY)
    if not can_place_mask(grid, mask2, r2, c_lane):
        return None
    paint_mask(grid, mask2, r2, c_lane, GREY)

    extra_count = rng.randint(1, 2)
    for _ in range(extra_count):
        placed = False
        for _ in range(120):
            mask = random_shape(rng)
            mode = rng.choice(("bottom", "left", "right", "free"))
            if mode == "bottom":
                r0 = rng.randint(max(2, h - mask.shape[0] - 4), h - mask.shape[0] - 2)
                c0 = rng.randint(2, w - mask.shape[1] - 3)
            elif mode == "left":
                r0 = rng.randint(2, h - mask.shape[0] - 3)
                c0 = rng.randint(2, 3)
            elif mode == "right":
                r0 = rng.randint(2, h - mask.shape[0] - 3)
                c0 = rng.randint(max(2, w - mask.shape[1] - 4), w - mask.shape[1] - 2)
            else:
                r0 = rng.randint(2, h - mask.shape[0] - 3)
                c0 = rng.randint(2, w - mask.shape[1] - 3)
            if can_place_mask(grid, mask, r0, c0):
                paint_mask(grid, mask, r0, c0, GREY)
                placed = True
                break
        if not placed:
            return None

    counts = vote_plan(rng)
    palette = {"up": frame[0], "down": frame[1], "left": frame[2], "right": frame[3]}
    for direction, count in counts.items():
        for _ in range(count):
            placed = False
            for _ in range(150):
                r = rng.randint(2, h - 3)
                c = rng.randint(2, w - 3)
                if can_place_vote(grid, r, c):
                    grid[r, c] = palette[direction]
                    placed = True
                    break
            if not placed:
                return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = generate_canonical_up(rng)
        if grid is None:
            continue
        k = rng.randrange(4)
        grid = rotate_grid(grid, k)
        if validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid voting-gravity input")
