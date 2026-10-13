import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
SPRITE_COLORS = (BLUE, RED, YELLOW, ORANGE)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

SPRITE_LIBRARY = [
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=int),
    np.array([[0, 0, 1, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=int),
]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_center(comp: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = comp["bbox"]
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _find_components(grid: np.ndarray, colors: tuple[int, ...] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    allow = None if colors is None else {int(c) for c in colors}
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            if allow is not None and color not in allow:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": int(coords.shape[0]),
                    "crop": _crop_bbox(g, bbox),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _trace_path(start: tuple[int, int], shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = map(int, shape)
    r, c = map(int, start)
    dc = 1
    path: list[tuple[int, int]] = []
    bounces: list[tuple[int, int]] = []

    while 0 <= r < H and 0 <= c < W:
        path.append((int(r), int(c)))
        nr = r - 1
        if nr < 0:
            break
        nc = c + dc
        if nc < 0 or nc >= W:
            bounces.append((int(r), int(c)))
            dc = -dc
            nc = c + dc
        r, c = nr, nc

    return path, bounces


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_components(g, colors=SPRITE_COLORS + (GREEN,))
    starts = [comp for comp in comps if comp["color"] == GREEN and comp["area"] == 1]
    assert len(starts) == 1, "expected one isolated green start pixel"

    sprites = [comp for comp in comps if comp["color"] in SPRITE_COLORS]
    assert 2 <= len(sprites) <= 4, "expected 2-4 sprite components"

    start = tuple(map(int, starts[0]["coords"][0]))
    return g, start, sprites


def _nearest_sprite_index(
    bounce: tuple[int, int], sprites: list[dict], chooser: str = "nearest"
) -> int:
    if chooser == "first":
        return 0
    if chooser == "last":
        return len(sprites) - 1

    br, bc = map(int, bounce)
    ranked = []
    for idx, sprite in enumerate(sprites):
        sr, sc = _component_center(sprite)
        dist = abs(br - sr) + abs(bc - sc)
        ranked.append((float(dist), idx, sprite["bbox"]))
    ranked.sort(key=lambda item: (item[0], item[2][0], item[2][1], item[1]))
    return int(ranked[0][1])


def _paint_stamp(
    out: np.ndarray,
    sprite_crop: np.ndarray,
    center: tuple[int, int],
    protect_mask: np.ndarray,
) -> None:
    h, w = map(int, sprite_crop.shape)
    top = int(center[0]) - h // 2
    left = int(center[1]) - w // 2
    H, W = out.shape

    for rr in range(h):
        for cc in range(w):
            color = int(sprite_crop[rr, cc])
            if color == BG:
                continue
            gr = top + rr
            gc = left + cc
            if 0 <= gr < H and 0 <= gc < W and not bool(protect_mask[gr, gc]):
                out[gr, gc] = color


def _solve_grid(grid: np.ndarray, mirror_mode: str = "wall", chooser: str = "nearest") -> np.ndarray:
    g, start, sprites = _parse_grid(grid)
    out = g.copy()
    path, bounces = _trace_path(start, g.shape)

    for r, c in path:
        out[r, c] = GREEN

    protect_mask = np.zeros(g.shape, dtype=bool)
    for sprite in sprites:
        rr = sprite["coords"][:, 0]
        cc = sprite["coords"][:, 1]
        protect_mask[rr, cc] = True

    for bounce in bounces:
        idx = _nearest_sprite_index(bounce, sprites, chooser=chooser)
        sprite_crop = sprites[idx]["crop"]
        if mirror_mode == "wall":
            patch = np.fliplr(sprite_crop) if bounce[1] == 0 else sprite_crop
        elif mirror_mode == "always_mirror":
            patch = np.fliplr(sprite_crop)
        elif mirror_mode == "none":
            patch = sprite_crop
        else:
            raise ValueError(f"unsupported mirror mode: {mirror_mode}")
        _paint_stamp(out, patch, bounce, protect_mask)

    return out


def _draw_path_only(grid: np.ndarray) -> np.ndarray:
    g, start, _sprites = _parse_grid(grid)
    out = g.copy()
    path, _bounces = _trace_path(start, g.shape)
    for r, c in path:
        out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the nearest sprite body onto each diagonal wall-bounce point, mirroring only on left-wall bounces.

    Input structure:
    - A black background contains one isolated green start pixel in the lower half of the grid.
    - The grid also contains 2-4 small monochrome sprite bodies in blue, red, yellow, or orange.
    - Sprite bodies are separated from each other and from the side walls; the bounce path does not pass through sprite pixels.

    Rule:
    - From the green pixel, trace a green ray along the up-right diagonal, reversing only its horizontal direction when it hits the left or right wall, and stop when it exits above the top edge.
    - Treat each wall cell where that horizontal reversal happens as a bounce point.
    - For every bounce point, choose the sprite whose bounding-box center has the smallest Manhattan distance to that bounce point.
    - Stamp that sprite centered on the bounce point; mirror it left-right for left-wall bounces and keep it unchanged for right-wall bounces.
    - Clip out-of-bounds stamp pixels, allow later stamps to overwrite earlier stamps, and never overwrite the original sprite bodies.

    Key insight:
    - The bounce path determines where to stamp, nearest-sprite assignment determines what to stamp, and the wall side alone determines whether the stamped body must be mirrored.
    """
    return _solve_grid(np.asarray(grid, dtype=int))


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask.astype(bool), int(color), BG).astype(int)


def _global_mask_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask.astype(bool))
    return [(int(top + r), int(left + c)) for r, c in coords]


def _component_gap(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> int:
    best = 10**9
    for ra, ca in coords_a:
        for rb, cb in coords_b:
            best = min(best, abs(int(ra) - int(rb)) + abs(int(ca) - int(cb)))
    return int(best)


def _placeable(
    mask: np.ndarray,
    top: int,
    left: int,
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    placed: list[list[tuple[int, int]]],
) -> bool:
    H, W = map(int, shape)
    h, w = map(int, mask.shape)
    if top < 2 or left < 2 or top + h > H - 2 or left + w > W - 2:
        return False

    coords = _global_mask_coords(mask, top, left)
    if any((r, c) in path_set for r, c in coords):
        return False
    for other in placed:
        if _component_gap(coords, other) < 6:
            return False
    return True


def _candidate_positions(
    mask: np.ndarray,
    shape: tuple[int, int],
    target_center: tuple[int, int],
    path_set: set[tuple[int, int]],
    placed: list[list[tuple[int, int]]],
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    H, W = map(int, shape)
    h, w = map(int, mask.shape)
    tr, tc = map(int, target_center)
    candidates = []

    for top in range(2, H - h - 1):
        for left in range(2, W - w - 1):
            if not _placeable(mask, top, left, shape, path_set, placed):
                continue
            center_r = top + h // 2
            center_c = left + w // 2
            score = abs(center_r - tr) + abs(center_c - tc)
            coords = _global_mask_coords(mask, top, left)
            candidates.append((int(score), int(top), int(left), coords))

    random.shuffle(candidates)
    candidates.sort(key=lambda item: item[0])
    return [(top, left, coords) for _score, top, left, coords in candidates]


def _distance_profile(
    bounces: list[tuple[int, int]], sprites: list[dict]
) -> tuple[list[int], bool]:
    chosen: list[int] = []
    unique = True
    for bounce in bounces:
        br, bc = map(int, bounce)
        dists = []
        for idx, sprite in enumerate(sprites):
            sr, sc = _component_center(sprite)
            dists.append(abs(br - sr) + abs(bc - sc))
        order = sorted((float(d), idx) for idx, d in enumerate(dists))
        if len(order) >= 2 and abs(order[0][0] - order[1][0]) < 1e-9:
            unique = False
        chosen.append(int(order[0][1]))
    return chosen, unique


def _semantic_checks(grid: np.ndarray) -> bool:
    g, start, sprites = _parse_grid(grid)
    path, bounces = _trace_path(start, g.shape)
    if len(bounces) < 2:
        return False
    if not any(c == 0 for _r, c in bounces):
        return False
    if not any(c == g.shape[1] - 1 for _r, c in bounces):
        return False

    path_set = set(path)
    for sprite in sprites:
        if any((int(r), int(c)) in path_set for r, c in sprite["coords"]):
            return False
        if np.array_equal(sprite["crop"], np.fliplr(sprite["crop"])):
            return False

    chosen, unique = _distance_profile(bounces, sprites)
    if not unique or len(set(chosen)) < 2:
        return False

    actual = main(g)
    no_mirror = _solve_grid(g, mirror_mode="none")
    always_mirror = _solve_grid(g, mirror_mode="always_mirror")
    first_choice = _solve_grid(g, chooser="first")
    path_only = _draw_path_only(g)

    if np.array_equal(actual, no_mirror):
        return False
    if np.array_equal(actual, always_mirror):
        return False
    if np.array_equal(actual, first_choice):
        return False
    if np.array_equal(actual, path_only):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(15, 22)
    W = random.randint(18, 24)

    start = None
    path = None
    bounces = None
    for _ in range(200):
        sr = random.randint(H // 2 + 1, H - 2)
        sc = random.randint(1, W - 2)
        trial_path, trial_bounces = _trace_path((sr, sc), (H, W))
        if len(trial_bounces) < 2:
            continue
        sides = {c for _r, c in trial_bounces}
        if sides != {0, W - 1}:
            continue
        start = (int(sr), int(sc))
        path = trial_path
        bounces = trial_bounces
        break
    if start is None or path is None or bounces is None:
        return None

    left_rows = [r for r, c in bounces if c == 0]
    right_rows = [r for r, c in bounces if c == W - 1]
    if not left_rows or not right_rows:
        return None

    sprite_count = random.randint(2, 4)
    masks = random.sample(SPRITE_LIBRARY, sprite_count)
    colors = random.sample(SPRITE_COLORS, sprite_count)
    path_set = set(path)
    placed_coords: list[list[tuple[int, int]]] = []
    sprites_data: list[tuple[np.ndarray, int, int, list[tuple[int, int]]]] = []

    target_specs = [
        (
            masks[0],
            colors[0],
            (
                max(4, min(H - 5, round(sum(left_rows) / len(left_rows)) + random.randint(-1, 1))),
                random.randint(4, max(4, W // 2 - 2)),
            ),
        ),
        (
            masks[1],
            colors[1],
            (
                max(4, min(H - 5, round(sum(right_rows) / len(right_rows)) + random.randint(-1, 1))),
                random.randint(min(W // 2 + 1, W - 5), W - 5),
            ),
        ),
    ]

    for idx in range(2, sprite_count):
        target_specs.append(
            (
                masks[idx],
                colors[idx],
                (
                    random.randint(4, H - 5),
                    random.randint(max(4, W // 3), min(W - 5, (2 * W) // 3)),
                ),
            )
        )

    for mask, color, target_center in target_specs:
        candidates = _candidate_positions(mask, (H, W), target_center, path_set, placed_coords)
        if not candidates:
            return None
        limit = min(8, len(candidates))
        top, left, coords = random.choice(candidates[:limit])
        placed_coords.append(coords)
        sprites_data.append((mask, int(color), int(top), int(left), coords))

    grid = np.zeros((H, W), dtype=int)
    grid[start] = GREEN
    for mask, color, top, left, _coords in sprites_data:
        patch = _mask_to_patch(mask, color)
        h, w = patch.shape
        region = grid[top : top + h, left : left + w]
        if np.any((patch != BG) & (region != BG)):
            return None
        region[patch != BG] = patch[patch != BG]

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 500 attempts")
