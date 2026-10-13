import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PATTERN_COLORS = [RED, GREEN, YELLOW]
MAX_GEN_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_true_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[:0, :0].copy()
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        rr, cc = q.popleft()
        for dr, dc in CARDINALS:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return ~outside


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.astype(np.uint8).tobytes()


def _find_template_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == MAGENTA):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(patch[0, :] == MAGENTA)
            and np.all(patch[-1, :] == MAGENTA)
            and np.all(patch[:, 0] == MAGENTA)
            and np.all(patch[:, -1] == MAGENTA)
        ):
            continue
        if patch.shape[0] <= 2 or patch.shape[1] <= 2:
            continue
        interior = patch[1:-1, 1:-1]
        if np.any(interior == MAGENTA):
            continue
        if not np.any(interior != BLACK):
            continue
        candidates.append((r0, c0, r1, c1))
    assert candidates, "expected a magenta template frame"
    return min(candidates, key=lambda box: (box[0], box[1]))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    fr0, fc0, fr1, fc1 = _find_template_frame(g)

    interior = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    painted = interior != BLACK
    assert np.any(painted), "template pattern must contain colored cells"
    pr0, pc0, pr1, pc1 = _bbox_from_coords(np.argwhere(painted))
    pattern = interior[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    pattern_mask = pattern != BLACK

    outer_shape = _fill_holes(pattern_mask)
    assert np.any(outer_shape), "template outline must define a target shape"
    assert len(_components(outer_shape)) == 1, "template outline must resolve to one connected shape"
    target_key = _mask_key(outer_shape)

    out = g.copy()
    matched = 0
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        comp_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = comp - np.array([r0, c0], dtype=int)
        comp_mask[rel[:, 0], rel[:, 1]] = True
        if _mask_key(comp_mask) != target_key:
            continue
        matched += 1
        rr, cc = np.nonzero(comp_mask)
        out[r0 + rr, c0 + cc] = pattern[rr, cc]

    assert matched >= 1, "expected at least one matching grey target"
    return out


def _boundary_cells(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in CARDINALS:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _interior_cells(mask: np.ndarray) -> np.ndarray:
    return mask & ~_boundary_cells(mask)


def _random_outer_shape(rng: random.Random, need_interior: bool) -> np.ndarray | None:
    min_dim = 4 if need_interior else 3
    for _ in range(120):
        h = rng.randint(min_dim, 7)
        w = rng.randint(min_dim, 7)
        mask = np.ones((h, w), dtype=bool)
        for _ in range(rng.randint(0, 3)):
            hh, ww = mask.shape
            candidate = mask.copy()
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                if hh < 3 or ww < 2:
                    continue
                depth = rng.randint(1, hh - 2)
                span = rng.randint(1, ww - 1)
                start = rng.randint(0, ww - span)
                rows = slice(0, depth) if side == "top" else slice(hh - depth, hh)
                candidate[rows, start : start + span] = False
            else:
                if ww < 3 or hh < 2:
                    continue
                depth = rng.randint(1, ww - 2)
                span = rng.randint(1, hh - 1)
                start = rng.randint(0, hh - span)
                cols = slice(0, depth) if side == "left" else slice(ww - depth, ww)
                candidate[start : start + span, cols] = False
            cropped = _crop_true_bbox(candidate)
            if cropped.size == 0 or int(cropped.sum()) < 6:
                continue
            mask = cropped

        mask = _crop_true_bbox(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < 6:
            continue
        if len(_components(mask)) != 1:
            continue
        if not np.array_equal(_fill_holes(mask), mask):
            continue
        if need_interior and not np.any(_interior_cells(mask)):
            continue
        return mask
    return None


def _sample_holes(shape_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    interior = _interior_cells(shape_mask)
    holes = np.zeros_like(shape_mask, dtype=bool)
    coords = [tuple(x) for x in np.argwhere(interior)]
    if not coords:
        return holes

    rng.shuffle(coords)
    hole_count = rng.randint(1, min(4, len(coords)))
    holes[coords[0][0], coords[0][1]] = True

    frontier = [coords[0]]
    while int(holes.sum()) < hole_count:
        neighbors: list[tuple[int, int]] = []
        for rr, cc in frontier:
            for dr, dc in CARDINALS:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < shape_mask.shape[0] and 0 <= nc < shape_mask.shape[1] and interior[nr, nc] and not holes[nr, nc]:
                    neighbors.append((nr, nc))
        if not neighbors:
            remaining = [cell for cell in coords if not holes[cell[0], cell[1]]]
            if not remaining:
                break
            pick = rng.choice(remaining)
        else:
            pick = rng.choice(neighbors)
        holes[pick[0], pick[1]] = True
        frontier.append(pick)
    return holes


def _make_pattern(shape_mask: np.ndarray, with_holes: bool, rng: random.Random) -> np.ndarray:
    holes = _sample_holes(shape_mask, rng) if with_holes else np.zeros_like(shape_mask, dtype=bool)
    visible = shape_mask & ~holes
    coords = [tuple(x) for x in np.argwhere(visible)]
    assert coords

    color_count = rng.randint(1, min(3, len(coords)))
    chosen = rng.sample(PATTERN_COLORS, color_count)
    rng.shuffle(coords)

    pattern = np.zeros(shape_mask.shape, dtype=int)
    for idx, color in enumerate(chosen):
        rr, cc = coords[idx]
        pattern[rr, cc] = color
    for rr, cc in coords[color_count:]:
        pattern[rr, cc] = rng.choice(chosen)
    return pattern


def _make_template_patch(pattern: np.ndarray, rng: random.Random) -> np.ndarray:
    pad_top = rng.randint(0, 2)
    pad_bottom = rng.randint(0, 2)
    pad_left = rng.randint(0, 2)
    pad_right = rng.randint(0, 2)
    inner_h = pattern.shape[0] + pad_top + pad_bottom
    inner_w = pattern.shape[1] + pad_left + pad_right

    patch = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    patch[0, :] = MAGENTA
    patch[-1, :] = MAGENTA
    patch[:, 0] = MAGENTA
    patch[:, -1] = MAGENTA
    patch[1 + pad_top : 1 + pad_top + pattern.shape[0], 1 + pad_left : 1 + pad_left + pattern.shape[1]] = pattern
    return patch


def _make_shape_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return False

    for _ in range(240):
        r = rng.randint(1, h - ph - 1)
        c = rng.randint(1, w - pw - 1)
        if np.any(blocked[r - 1 : r + ph + 1, c - 1 : c + pw + 1]):
            continue
        window = grid[r : r + ph, c : c + pw]
        if np.any((patch != BLACK) & (window != BLACK)):
            continue
        window[patch != BLACK] = patch[patch != BLACK]
        blocked[r - 1 : r + ph + 1, c - 1 : c + pw + 1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    want_holes = bool(rng.randint(0, 1))
    outer_shape = _random_outer_shape(rng, need_interior=want_holes)
    if outer_shape is None:
        return None

    pattern = _make_pattern(outer_shape, with_holes=want_holes, rng=rng)
    template_patch = _make_template_patch(pattern, rng)

    target_count = rng.randint(1, 3)
    distractor_count = rng.randint(1, 4)

    object_patches: list[np.ndarray] = []
    for _ in range(target_count):
        object_patches.append(_make_shape_patch(outer_shape, GREY))

    target_key = _mask_key(outer_shape)
    for _ in range(distractor_count):
        distractor: np.ndarray | None = None
        for _ in range(80):
            candidate = _random_outer_shape(rng, need_interior=False)
            if candidate is None:
                continue
            if _mask_key(candidate) != target_key:
                distractor = candidate
                break
        if distractor is None:
            return None
        object_patches.append(_make_shape_patch(distractor, GREY))

    min_h = max(18, template_patch.shape[0] + 4, max(p.shape[0] for p in object_patches) + 4)
    min_w = max(18, template_patch.shape[1] + 4, max(p.shape[1] for p in object_patches) + 4)
    max_h = max(min_h, 30)
    max_w = max(min_w, 30)
    if min_h > max_h or min_w > max_w:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    tr = 1
    tc = 1
    th, tw = template_patch.shape
    grid[tr : tr + th, tc : tc + tw] = template_patch
    blocked[tr - 1 : tr + th + 1, tc - 1 : tc + tw + 1] = True

    rng.shuffle(object_patches)
    for patch in sorted(object_patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        if not _place_patch(grid, blocked, patch, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
