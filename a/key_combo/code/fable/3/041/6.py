from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
RED = 2
POINTER_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in DIRS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_hollow_frame(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
        return None
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return None
    return r0, c0, r1, c1


def _pointer_direction(
    center: tuple[int, int], pointer: tuple[int, int]
) -> tuple[int, int]:
    cr, cc = center
    pr, pc = pointer
    if pr == cr and pc != cc:
        return 0, 1 if pc > cc else -1
    if pc == cc and pr != cr:
        return 1 if pr > cr else -1, 0
    raise AssertionError("pointer must be strictly above, below, left, or right of the frame center")


def _parse_frames(g: np.ndarray) -> list[dict]:
    assert g.ndim == 2
    assert BG in set(map(int, np.unique(g))), "expected a black background"
    colors = set(map(int, np.unique(g)))
    assert RED in colors, "expected red hollow frames"

    frames: list[dict] = []
    for coords in _components4(g == RED):
        box = _is_hollow_frame(coords)
        assert box is not None, "each red component must be one hollow odd-sized frame"
        r0, c0, r1, c1 = box
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        pointer_locs = np.argwhere((interior != BG) & (interior != RED))
        assert pointer_locs.shape == (1, 2), "each frame must contain exactly one non-red pointer pixel"
        lr, lc = map(int, pointer_locs[0])
        pr, pc = r0 + 1 + lr, c0 + 1 + lc
        pointer_color = int(g[pr, pc])
        assert pointer_color not in {BG, RED}
        direction = _pointer_direction((cr, cc), (pr, pc))

        frames.append(
            {
                "bbox": box,
                "coords": coords,
                "center": (cr, cc),
                "pointer": (pr, pc),
                "pointer_color": pointer_color,
                "direction": direction,
            }
        )

    assert len(frames) >= 2, "at least two frames are needed for pointing"
    pointer_colors = [frame["pointer_color"] for frame in frames]
    assert len(set(pointer_colors)) == len(pointer_colors), "pointer colors must be unique"

    pointers = {frame["pointer"] for frame in frames}
    non_frame_colored = {
        tuple(map(int, rc))
        for rc in np.argwhere((g != BG) & (g != RED))
    }
    assert non_frame_colored == pointers, "only pointer pixels may use non-red foreground colors"
    return frames


def _frame_id_grid(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    ids = np.full(shape, -1, dtype=int)
    for idx, frame in enumerate(frames):
        coords = frame["coords"]
        ids[coords[:, 0], coords[:, 1]] = idx
    return ids


def _ray_target(ids: np.ndarray, source_idx: int, frame: dict) -> int:
    h, w = ids.shape
    cr, cc = frame["center"]
    dr, dc = frame["direction"]
    r, c = cr + dr, cc + dc
    while 0 <= r < h and 0 <= c < w:
        hit = int(ids[r, c])
        if hit >= 0 and hit != int(source_idx):
            return hit
        r += dr
        c += dc
    raise AssertionError("each pointer ray must hit another frame")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every red hollow frame using the pointer color of the frame hit by its ray.

    Input structure:
    - A black background contains several separated red hollow odd-sized frames.
    - Each frame interior has exactly one non-red, non-black pointer pixel.
    - The pointer lies strictly above, below, left, or right of that frame's center, and all pointer colors are unique.

    Rule:
    - For each frame, shoot a ray from the frame center through its pointer direction.
    - The first other red frame hit by that ray is the target frame.
    - Fill the source frame's black interior cells with the target frame's pointer color, preserving red borders and all pointer pixels.

    Key insight:
    - A frame's own pointer color is not its fill color; pointer position identifies another frame whose pointer supplies the color.
    """
    g = np.asarray(grid, dtype=int)
    frames = _parse_frames(g)
    ids = _frame_id_grid(g.shape, frames)
    targets = [_ray_target(ids, idx, frame) for idx, frame in enumerate(frames)]

    out = g.copy()
    for frame, target_idx in zip(frames, targets):
        r0, c0, r1, c1 = frame["bbox"]
        fill_color = int(frames[int(target_idx)]["pointer_color"])
        black_interior = g[r0 + 1 : r1, c0 + 1 : c1] == BG
        view = out[r0 + 1 : r1, c0 + 1 : c1]
        view[black_interior] = fill_color
    return out


# --------------- generator (o5x) ---------------

GEN_FRAME_DIMS = ((5, 5), (5, 7), (7, 5), (7, 7))
GEN_PATTERNS = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
)
GEN_HALF_MAX = 3
GEN_MIN_CENTER_GAP = 8
GEN_ATTEMPTS = 500


def _gen_axis_coords(count: int, limit: int) -> list[int] | None:
    choices = list(range(GEN_HALF_MAX, int(limit) - GEN_HALF_MAX))
    for _ in range(300):
        coords = sorted(random.sample(choices, int(count)))
        if all(b - a >= GEN_MIN_CENTER_GAP for a, b in zip(coords, coords[1:])):
            return coords
    return None


def _gen_transform_pattern(points: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in points]
    if random.choice([False, True]):
        pts = [(c, r) for r, c in pts]
    max_r = max(r for r, _c in pts)
    max_c = max(c for _r, c in pts)
    if random.choice([False, True]):
        pts = [(max_r - r, c) for r, c in pts]
    if random.choice([False, True]):
        pts = [(r, max_c - c) for r, c in pts]
    min_r = min(r for r, _c in pts)
    min_c = min(c for _r, c in pts)
    return sorted({(r - min_r, c - min_c) for r, c in pts})


def _available_dirs(point: tuple[int, int], points: set[tuple[int, int]]) -> list[str]:
    r, c = point
    dirs = []
    if any(rr < r and cc == c for rr, cc in points):
        dirs.append("U")
    if any(rr > r and cc == c for rr, cc in points):
        dirs.append("D")
    if any(rr == r and cc < c for rr, cc in points):
        dirs.append("L")
    if any(rr == r and cc > c for rr, cc in points):
        dirs.append("R")
    return dirs


def _lattice_target(
    point: tuple[int, int], direction: str, points: set[tuple[int, int]]
) -> tuple[int, int]:
    r, c = point
    if direction == "U":
        return max((rr, cc) for rr, cc in points if cc == c and rr < r)
    if direction == "D":
        return min((rr, cc) for rr, cc in points if cc == c and rr > r)
    if direction == "L":
        return max((rr, cc) for rr, cc in points if rr == r and cc < c)
    return min((rr, cc) for rr, cc in points if rr == r and cc > c)


def _draw_frame(grid: np.ndarray, center: tuple[int, int], dims: tuple[int, int]) -> tuple[int, int, int, int]:
    cr, cc = center
    fh, fw = dims
    r0 = cr - fh // 2
    c0 = cc - fw // 2
    r1 = r0 + fh - 1
    c1 = c0 + fw - 1
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    return r0, c0, r1, c1


def _pointer_for(center: tuple[int, int], dims: tuple[int, int], direction: str) -> tuple[int, int]:
    cr, cc = center
    fh, fw = dims
    dr, dc = DIRS[direction]
    limit = (fh // 2 - 1) if dr else (fw // 2 - 1)
    step = random.randint(1, int(limit))
    return cr + dr * step, cc + dc * step


def _render_expected(inp: np.ndarray, specs: list[dict], targets: list[int]) -> np.ndarray:
    out = inp.copy()
    for idx, spec in enumerate(specs):
        r0, c0, r1, c1 = spec["bbox"]
        fill_color = int(specs[int(targets[idx])]["color"])
        mask = inp[r0 + 1 : r1, c0 + 1 : c1] == BG
        view = out[r0 + 1 : r1, c0 + 1 : c1]
        view[mask] = fill_color
    return out


def _has_non_nearest_target(specs: list[dict], targets: list[int]) -> bool:
    centers = [spec["center"] for spec in specs]
    for idx, target_idx in enumerate(targets):
        cr, cc = centers[idx]
        target = centers[int(target_idx)]
        target_dist = abs(cr - target[0]) + abs(cc - target[1])
        nearest = min(
            abs(cr - r) + abs(cc - c)
            for j, (r, c) in enumerate(centers)
            if j != idx
        )
        if target_dist > nearest:
            return True
    return False


def _gen_semantic_checks(specs: list[dict], targets: list[int]) -> bool:
    directions = {spec["direction"] for spec in specs}
    if len(directions) < 2:
        return False
    if len(set(targets)) < 2:
        return False
    if any(int(t) == i for i, t in enumerate(targets)):
        return False
    if not _has_non_nearest_target(specs, targets):
        return False

    fills_by_size: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for idx, spec in enumerate(specs):
        fills_by_size[spec["dims"]].add(int(specs[int(targets[idx])]["color"]))
    return any(len(colors) >= 2 for colors in fills_by_size.values())


def _gen_attempt() -> np.ndarray | None:
    lattice = _gen_transform_pattern(random.choice(GEN_PATTERNS))
    points = set(lattice)
    row_count = max(r for r, _c in lattice) + 1
    col_count = max(c for _r, c in lattice) + 1

    min_h = 2 * GEN_HALF_MAX + 1 + (row_count - 1) * GEN_MIN_CENTER_GAP
    min_w = 2 * GEN_HALF_MAX + 1 + (col_count - 1) * GEN_MIN_CENTER_GAP
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(max(15, min_h), 30)
    w = random.randint(max(15, min_w), 30)
    row_coords = _gen_axis_coords(row_count, h)
    col_coords = _gen_axis_coords(col_count, w)
    if row_coords is None or col_coords is None:
        return None

    point_to_idx = {point: idx for idx, point in enumerate(lattice)}
    directions: list[str] = []
    targets: list[int] = []
    for point in lattice:
        choices = _available_dirs(point, points)
        if not choices:
            return None
        direction = random.choice(choices)
        directions.append(direction)
        targets.append(point_to_idx[_lattice_target(point, direction, points)])

    n = len(lattice)
    colors = list(random.sample(POINTER_COLORS, n))
    dims = [random.choice(GEN_FRAME_DIMS) for _ in range(n)]
    if n >= 2:
        a, b = random.sample(range(n), 2)
        dims[a] = dims[b] = random.choice(GEN_FRAME_DIMS)

    inp = np.full((h, w), BG, dtype=int)
    specs: list[dict] = []
    for idx, (lr, lc) in enumerate(lattice):
        center = (int(row_coords[lr]), int(col_coords[lc]))
        box = _draw_frame(inp, center, dims[idx])
        pr, pc = _pointer_for(center, dims[idx], directions[idx])
        inp[pr, pc] = int(colors[idx])
        specs.append(
            {
                "center": center,
                "bbox": box,
                "dims": dims[idx],
                "direction": directions[idx],
                "pointer": (pr, pc),
                "color": int(colors[idx]),
            }
        )

    if not _gen_semantic_checks(specs, targets):
        return None

    expected = _render_expected(inp, specs, targets)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-ray frame input")
