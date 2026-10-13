from __future__ import annotations

from collections import deque
import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_H_MIN, GRID_H_MAX = 18, 30
GRID_W_MIN, GRID_W_MAX = 18, 30
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _layer_distances(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    dist = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(mask):
        is_boundary = False
        for nr, nc in _neighbors4(int(r), int(c), h, w):
            if not mask[nr, nc]:
                is_boundary = True
                break
        if not is_boundary and (r == 0 or c == 0 or r == h - 1 or c == w - 1):
            is_boundary = True
        if is_boundary:
            dist[r, c] = 1
            q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and dist[nr, nc] == 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    assert np.all(dist[mask] > 0)
    return dist


def _has_hole(mask: np.ndarray) -> bool:
    inv = ~mask
    h, w = mask.shape
    for comp in _components4(inv):
        if not any(
            r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp.tolist()
        ):
            return True
    return False


def _make_shell(interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = interior.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = interior
    boundary = _dilate4(padded) & ~padded
    return padded, boundary


def _build_target_patch(
    interior: np.ndarray, seed_color: int, seed_rc: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    padded, boundary = _make_shell(interior)
    dist = _layer_distances(interior)
    patch_in = np.full(padded.shape, BLACK, dtype=int)
    patch_out = np.full(padded.shape, BLACK, dtype=int)
    patch_in[boundary] = GREY
    patch_out[boundary] = GREY

    for r, c in np.argwhere(interior):
        rr, cc = int(r) + 1, int(c) + 1
        if dist[r, c] % 2 == 1:
            patch_out[rr, cc] = seed_color

    sr, sc = seed_rc
    patch_in[sr + 1, sc + 1] = seed_color
    return patch_in, patch_out


def _make_random_tree(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                if mask[nr, nc]:
                    continue
                deg = sum(mask[rr, cc] for rr, cc in _neighbors4(nr, nc, h, w))
                if deg == 1:
                    frontier.add((nr, nc))
        if not frontier:
            break
        candidates = list(frontier)
        pick = candidates[int(rng.integers(0, len(candidates)))]
        mask[pick] = True
    return mask


def _make_interior_mask(rng: np.random.Generator, min_depth: int) -> np.ndarray | None:
    for _ in range(80):
        base_h = int(rng.integers(5, 10))
        base_w = int(rng.integers(5, 10))
        target = int(rng.integers(max(4, (base_h * base_w) // 10), max(8, (base_h * base_w) // 4)))
        base = _make_random_tree(rng, base_h, base_w, target)
        if int(base.sum()) < 4:
            continue

        interior = base.copy()
        rounds = 1 if rng.random() < 0.55 else 2
        for _ in range(rounds):
            interior = _dilate4(interior)

        if rng.random() < 0.35:
            bump_h = int(rng.integers(2, min(5, interior.shape[0] + 1)))
            bump_w = int(rng.integers(2, min(5, interior.shape[1] + 1)))
            rr = int(rng.integers(0, interior.shape[0] - bump_h + 1))
            cc = int(rng.integers(0, interior.shape[1] - bump_w + 1))
            bump = np.zeros_like(interior)
            bump[rr : rr + bump_h, cc : cc + bump_w] = True
            if np.any(interior & bump):
                interior |= bump

        interior = _crop_mask(interior)
        area = int(interior.sum())
        if area < 6 or area > 48:
            continue
        if len(_components4(interior)) != 1 or _has_hole(interior):
            continue

        dist = _layer_distances(interior)
        if int(dist[interior].max()) < min_depth:
            continue
        return interior
    return None


def _make_target_region(
    rng: np.random.Generator, desired_seed_parity: int, min_depth: int
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(120):
        interior = _make_interior_mask(rng, min_depth=min_depth)
        if interior is None:
            continue
        dist = _layer_distances(interior)
        coords = np.argwhere(interior)
        parity_mask = (dist[coords[:, 0], coords[:, 1]] % 2) == desired_seed_parity
        candidates = coords[parity_mask]
        if len(candidates) == 0:
            candidates = coords
        seed = candidates[int(rng.integers(0, len(candidates)))]
        seed_color = int(rng.choice(SEED_COLORS))
        return _build_target_patch(interior, seed_color, (int(seed[0]), int(seed[1])))
    return None


def _make_demo_shape(rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        h = int(rng.integers(4, 7))
        w = int(rng.integers(5, 8))
        return np.ones((h, w), dtype=bool)

    radius = int(rng.integers(2, 4))
    size = 2 * radius + 1
    shape = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if abs(r - radius) + abs(c - radius) <= radius:
                shape[r, c] = True
    return shape


def _make_demo_box(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(40):
        interior = _make_demo_shape(rng)
        dist = _layer_distances(interior)
        if int(dist[interior].max()) < 2:
            continue
        color = int(rng.choice(SEED_COLORS))
        padded, boundary = _make_shell(interior)
        core = np.full(padded.shape, BLACK, dtype=int)
        core[boundary] = GREY
        for r, c in np.argwhere(interior):
            rr, cc = int(r) + 1, int(c) + 1
            if dist[r, c] % 2 == 1:
                core[rr, cc] = color

        gap = int(rng.integers(1, 3))
        box = np.full((core.shape[0] + 2 + 2 * gap, core.shape[1] + 2 + 2 * gap), BLACK, dtype=int)
        box[0, :] = MAROON
        box[-1, :] = MAROON
        box[:, 0] = MAROON
        box[:, -1] = MAROON
        top = 1 + gap
        left = 1 + gap
        box[top : top + core.shape[0], left : left + core.shape[1]] = core
        return box.copy(), box.copy()
    raise RuntimeError("failed to build demo box")


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not np.any(reserved[r0:r1, c0:c1])


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    reserved[r0:r1, c0:c1] = True


def _stamp(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each grey-enclosed region with a single colored seed and paint
    alternating inward contour bands using that seed color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    free = g != GREY
    for comp in _components4(free):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp.tolist()):
            continue

        vals = g[comp[:, 0], comp[:, 1]]
        colored = vals != BLACK
        if int(np.count_nonzero(colored)) != 1:
            continue

        seed_color = int(vals[colored][0])
        comp_mask = np.zeros((h, w), dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True

        dist = np.zeros((h, w), dtype=int)
        q: deque[tuple[int, int]] = deque()
        for r, c in comp:
            rr, cc = int(r), int(c)
            if any(g[nr, nc] == GREY for nr, nc in _neighbors4(rr, cc, h, w)):
                dist[rr, cc] = 1
                q.append((rr, cc))

        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c, h, w):
                if comp_mask[nr, nc] and dist[nr, nc] == 0:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))

        for r, c in comp:
            rr, cc = int(r), int(c)
            out[rr, cc] = seed_color if dist[rr, cc] % 2 == 1 else BLACK

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(GRID_H_MIN, GRID_H_MAX + 1))
    w = int(rng.integers(GRID_W_MIN, GRID_W_MAX + 1))

    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    demo_in, demo_out = _make_demo_box(rng)
    dh, dw = demo_in.shape
    if dh > h or dw > w:
        return None

    demo_positions = [
        (0, 0),
        (0, w - dw),
        (h - dh, 0),
        (h - dh, w - dw),
        (0, max(0, (w - dw) // 2)),
        (h - dh, max(0, (w - dw) // 2)),
        (max(0, (h - dh) // 2), 0),
        (max(0, (h - dh) // 2), w - dw),
    ]
    demo_positions = [pos for pos in demo_positions if 0 <= pos[0] <= h - dh and 0 <= pos[1] <= w - dw]
    if not demo_positions:
        return None
    demo_top, demo_left = demo_positions[int(rng.integers(0, len(demo_positions)))]
    _stamp(inp, demo_in, demo_top, demo_left)
    _stamp(out, demo_out, demo_top, demo_left)
    _reserve(reserved, demo_top, demo_left, dh, dw, pad=1)

    region_count = int(rng.integers(2, 6))
    parity_targets = [int(rng.integers(0, 2)) for _ in range(region_count)]
    if region_count >= 2:
        parity_targets[0] = 0
        parity_targets[1] = 1
        rng.shuffle(parity_targets)
    depth_targets = [1] * region_count
    depth_targets[int(rng.integers(0, region_count))] = int(rng.choice([2, 3]))

    for desired_parity, min_depth in zip(parity_targets, depth_targets):
        placed = False
        for _ in range(160):
            patch = _make_target_region(rng, desired_seed_parity=desired_parity, min_depth=min_depth)
            if patch is None:
                continue
            patch_in, patch_out = patch
            ph, pw = patch_in.shape
            if ph > h or pw > w:
                continue
            for _ in range(120):
                top = int(rng.integers(0, h - ph + 1))
                left = int(rng.integers(0, w - pw + 1))
                if not _can_place(reserved, top, left, ph, pw, pad=1):
                    continue
                _stamp(inp, patch_in, top, left)
                _stamp(out, patch_out, top, left)
                _reserve(reserved, top, left, ph, pw, pad=1)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 4")
