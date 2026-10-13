from __future__ import annotations

from collections import Counter, deque
import math
import random

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    assert mask.ndim == 2
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    neigh = _NEIGHBORS8 if int(connectivity) == 8 else _NEIGHBORS4
    comps: list[np.ndarray] = []

    starts = np.argwhere(mask)
    for sr, sc in starts:
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in neigh:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[int(top) : int(top) + int(h), int(left) : int(left) + int(w)]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _rotation_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    pts = set()
    rr = int(r)
    cc = int(c)
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = int(cc), int(n - 1 - rr)
    return pts


def _rotate_mask_cw(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    k = int(quarter_turns) % 4
    if k == 0:
        return mask.copy()

    h, w = map(int, mask.shape)
    if h == w:
        return np.rot90(mask, -k).copy()

    side = max(h, w)
    pad = np.zeros((side, side), dtype=bool)
    r0 = (side - h) // 2
    c0 = (side - w) // 2
    pad[r0 : r0 + h, c0 : c0 + w] = mask
    rot = np.rot90(pad, -k)
    return rot[r0 : r0 + h, c0 : c0 + w].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tight-crop the main board component, then rotate each board color layer according
    to the area of its same-color outside cue sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    fg_components = _connected_components(g != bg, connectivity=8)
    assert fg_components, "expected at least one foreground component"

    def comp_key(coords: np.ndarray) -> tuple[int, int, int]:
        bbox = _bbox_from_coords(coords)
        patch_colors = {int(g[r, c]) for r, c in coords}
        bbox_area = int((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1))
        return int(coords.shape[0]), len(patch_colors), bbox_area

    board_coords = max(fg_components, key=comp_key)
    r0, c0, r1, c1 = _bbox_from_coords(board_coords)

    board_mask = np.zeros_like(g, dtype=bool)
    board_mask[board_coords[:, 0], board_coords[:, 1]] = True

    board = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    rel = board_coords - np.array([r0, c0], dtype=int)
    board[rel[:, 0], rel[:, 1]] = g[board_coords[:, 0], board_coords[:, 1]]

    colors = [int(col) for col in np.unique(board) if int(col) != int(bg)]
    out = np.full_like(board, int(bg))

    for color in colors:
        layer = board == int(color)
        cue_components = _connected_components((g == int(color)) & (~board_mask), connectivity=8)
        cue_area = max((int(comp.shape[0]) for comp in cue_components), default=0)
        rot_k = cue_area % 4
        rotated = _rotate_mask_cw(layer, rot_k)
        out[rotated] = int(color)

    return out


GRID_MIN = 18
GRID_MAX = 26
BOARD_MIN = 9
BOARD_MAX = 15
MAX_GEN_ATTEMPTS = 300
MIN_BBOX_GAP = 2
COLOR_POOL = tuple(range(10))
STATE_OPTIONS = [None, 0, 1, 2, 3]
EMPTY = -1


def _zone_masks(n: int, band_widths: list[int]) -> list[np.ndarray]:
    rr, cc = np.indices((n, n))
    dist = np.minimum.reduce([rr, cc, n - 1 - rr, n - 1 - cc])
    zones: list[np.ndarray] = []
    start = 0
    for width in band_widths:
        zone = (dist >= start) & (dist < start + int(width))
        zones.append(zone)
        start += int(width)
    zones.append(dist >= start)
    return zones


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _sample_zone_layer(
    zone: np.ndarray,
    *,
    density_lo: float,
    density_hi: float,
    rot_k: int,
    require_full_bbox: bool = False,
) -> np.ndarray | None:
    n = int(zone.shape[0])
    coords = [(int(r), int(c)) for r, c in np.argwhere(zone)]
    if not coords:
        return None

    area = len(coords)
    rr, cc = np.indices(zone.shape)
    side_seeds: set[tuple[int, int]] = set()
    if require_full_bbox:
        tops = [(0, c) for c in range(n) if zone[0, c]]
        bottoms = [(n - 1, c) for c in range(n) if zone[n - 1, c]]
        lefts = [(r, 0) for r in range(n) if zone[r, 0]]
        rights = [(r, n - 1) for r in range(n) if zone[r, n - 1]]
        side_seeds = {
            random.choice(tops),
            random.choice(bottoms),
            random.choice(lefts),
            random.choice(rights),
        }

    orbit_candidates = [p for p in coords if len(_rotation_orbit(p[0], p[1], n)) == 4]
    if not orbit_candidates:
        orbit_candidates = coords

    for _ in range(120):
        phase = random.randrange(2)
        mask = zone & (((rr + cc + phase) % 2) == 0)

        anchor = random.choice(orbit_candidates)
        orbit = _rotation_orbit(anchor[0], anchor[1], n)
        for rr0, cc0 in orbit:
            mask[rr0, cc0] = False
        mask[anchor] = True

        for seed in side_seeds:
            mask[seed] = True

        base = int(mask.sum())
        lo = max(base, int(math.ceil(density_lo * area)))
        hi = int(math.floor(density_hi * area))
        if hi < lo:
            continue

        target = random.randint(lo, hi)
        free = [p for p in coords if not mask[p]]
        random.shuffle(free)
        for r, c in free[: max(0, target - base)]:
            mask[r, c] = True

        if require_full_bbox and not _touches_all_sides(mask):
            continue
        if rot_k and np.array_equal(mask, _rotate_mask_cw(mask, rot_k)):
            continue
        return mask
    return None


def _cue_area_for_rotation(rot_k: int) -> int:
    options = {
        0: [4, 8],
        1: [1, 5],
        2: [2, 6],
        3: [3, 7],
    }
    return int(random.choice(options[int(rot_k) % 4]))


def _grow_blob(area: int) -> np.ndarray:
    side = 3 if area <= 5 else 4
    center = side // 2
    pts = {(center, center)}
    while len(pts) < int(area):
        frontier = set()
        for r, c in pts:
            for dr, dc in _NEIGHBORS8:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in pts:
                    frontier.add((nr, nc))
        pts.add(random.choice(sorted(frontier)))

    arr = np.zeros((side, side), dtype=bool)
    for r, c in pts:
        arr[r, c] = True

    coords = np.argwhere(arr)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return int(max(dr, dc))


def _place_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    forbidden: list[tuple[int, int, int, int]],
    min_gap: int,
) -> tuple[int, int, int, int] | None:
    H, W = map(int, canvas.shape)
    h, w = map(int, patch.shape)
    mask = patch != EMPTY
    candidates = []

    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if np.any((canvas[r0 : r0 + h, c0 : c0 + w] != EMPTY) & mask):
                continue
            bb = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_bbox_gap(bb, other) < int(min_gap) for other in forbidden):
                continue
            candidates.append((r0, c0, bb))

    if not candidates:
        return None
    r0, c0, bb = random.choice(candidates)
    _paste(canvas, patch, r0, c0, transparent=EMPTY)
    return bb


def _choose_states(n_layers: int) -> list[int | None]:
    while True:
        states = [random.choice(STATE_OPTIONS) for _ in range(int(n_layers))]
        if any(state in (1, 2, 3) for state in states):
            return states


def _build_board() -> tuple[np.ndarray, np.ndarray, list[int | None], list[int], int] | None:
    n_layers = random.randint(2, 4)

    for _ in range(80):
        band_widths = [random.choice([1, 2]) for _ in range(n_layers - 1)]
        min_n = max(BOARD_MIN, 2 * sum(band_widths) + 3)
        if min_n > BOARD_MAX:
            continue
        n = random.randint(min_n, BOARD_MAX)
        zones = _zone_masks(n, band_widths)
        states = _choose_states(n_layers)
        colors = random.sample(list(COLOR_POOL), n_layers + 1)
        bg = colors[0]
        layer_colors = colors[1:]

        masks: list[np.ndarray] = []
        for i, (zone, state) in enumerate(zip(zones, states)):
            lo = 0.45 if i < n_layers - 1 else 0.35
            hi = 0.8 if i == 0 else 0.75
            if zone.sum() <= 6:
                lo = min(lo, 0.5)
                hi = 1.0
            mask = _sample_zone_layer(
                zone,
                density_lo=lo,
                density_hi=hi,
                rot_k=0 if state is None else int(state),
                require_full_bbox=(i == 0),
            )
            if mask is None:
                break
            masks.append(mask)
        if len(masks) != n_layers:
            continue

        out_board = np.full((n, n), bg, dtype=int)
        in_board = np.full((n, n), bg, dtype=int)

        ok = True
        for color, mask, state in zip(layer_colors, masks, states):
            out_board[mask] = int(color)
            back = _rotate_mask_cw(mask, 0 if state is None else -int(state))
            if np.any((in_board != int(bg)) & back):
                ok = False
                break
            in_board[back] = int(color)
        if not ok:
            continue

        comps = _connected_components(in_board != bg, connectivity=8)
        if len(comps) != 1:
            continue
        if _bbox_from_coords(comps[0]) != (0, 0, n - 1, n - 1):
            continue
        return np.array(in_board, dtype=int), np.array(out_board, dtype=int), states, [int(c) for c in layer_colors], int(bg)

    return None


def _gen_attempt() -> np.ndarray | None:
    built = _build_board()
    if built is None:
        return None
    in_board, out_board, states, layer_colors, bg = built
    n = int(in_board.shape[0])

    H = random.randint(max(GRID_MIN, n + 5), GRID_MAX)
    W = random.randint(max(GRID_MIN, n + 5), GRID_MAX)

    canvas = np.full((H, W), EMPTY, dtype=int)
    board_top = random.randint(0, H - n)
    board_left = random.randint(0, W - n)
    _paste(canvas, in_board, board_top, board_left, transparent=bg)
    forbidden = [(board_top, board_left, board_top + n - 1, board_left + n - 1)]

    cue_count = 0
    for color, state in zip(layer_colors, states):
        if state is None:
            continue
        cue_count += 1
        area = _cue_area_for_rotation(int(state))
        blob = _grow_blob(area)
        patch = np.full(blob.shape, EMPTY, dtype=int)
        patch[blob] = int(color)
        placed = _place_patch(canvas, patch, forbidden=forbidden, min_gap=MIN_BBOX_GAP)
        if placed is None:
            return None
        forbidden.append(placed)

    if cue_count == 0:
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[canvas != EMPTY] = canvas[canvas != EMPTY]

    fg_components = _connected_components(grid != bg, connectivity=8)
    if not fg_components:
        return None
    areas = [int(comp.shape[0]) for comp in fg_components]
    max_area = max(areas)
    if areas.count(max_area) != 1:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, out_board):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
