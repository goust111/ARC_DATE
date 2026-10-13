from __future__ import annotations

from collections import Counter, deque
import numpy as np


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

GRID_MIN = 16
GRID_MAX = 30
OBJECT_COUNT_RANGE = (1, 4)  # 1..3
PLACEMENT_PAD = 1
GEN_RETRIES = 400
PLACE_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _dominant_non_bg(g: np.ndarray, bg: int = BG) -> int | None:
    vals = g[g != bg]
    if vals.size == 0:
        return None
    uniq, cnt = np.unique(vals, return_counts=True)
    return int(uniq[int(cnt.argmax())])


def _boundary_and_cavity(filled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = filled.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                boundary[r, c] = True
                continue
            for nr, nc in _neighbors4(r, c):
                if not filled[nr, nc]:
                    boundary[r, c] = True
                    break
    cavity = filled & (~boundary)
    return boundary, cavity


def _distance_from_grey(region_mask: np.ndarray, grey_mask: np.ndarray) -> np.ndarray:
    h, w = region_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if not region_mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and grey_mask[nr, nc]:
                    dist[r, c] = 1
                    q.append((r, c))
                    break
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and region_mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _enclosed_regions(g: np.ndarray, grey: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    comps = _components(g != grey)
    regions: list[list[tuple[int, int]]] = []
    for comp in comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with an inward border whose color comes from its seeds and whose thickness comes from the seed count.

    Input structure:
    - Background is black.
    - One or more grey 1-pixel orthogonal polygon outlines create closed regions.
    - Inside each closed region, black interior cells contain 1 to 3 scattered seed pixels of one shared non-grey color.

    Rule:
    - For each closed region, find the seed color C and the number of seed pixels N.
    - Keep the grey outline unchanged.
    - Measure 4-neighbor distance from the grey outline inward through the enclosed region.
    - Recolor every region cell whose inward distance is at most N to color C; deeper seed cells remain C unchanged.

    Key insight:
    - The seed color encodes the fill color, while the number of same-colored seed pixels independently encodes the inward border thickness.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    grey = _dominant_non_bg(g, bg=BG)
    if grey is None:
        return g.copy()

    out = g.copy()
    grey_mask = g == grey
    for comp in _enclosed_regions(g, grey):
        region_mask = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            region_mask[r, c] = True

        seed_vals = [int(g[r, c]) for r, c in comp if int(g[r, c]) not in (BG, grey)]
        if not seed_vals:
            continue
        color_counts = Counter(seed_vals)
        seed_color, seed_count = color_counts.most_common(1)[0]
        seed_count = int(sum(1 for v in seed_vals if v == seed_color))

        dist = _distance_from_grey(region_mask, grey_mask)
        for r, c in comp:
            if 0 < dist[r, c] <= seed_count:
                out[r, c] = int(seed_color)
    return out


def _transform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = mask
    k = int(rng.integers(0, 4))
    if k:
        out = np.rot90(out, k)
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    if bool(rng.integers(0, 2)):
        out = np.flipud(out)
    return np.array(out, dtype=bool, copy=True)


def _rectangle_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(7, 13))
    w = int(rng.integers(7, 13))
    return np.ones((h, w), dtype=bool)


def _l_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(9, 15))
    w = int(rng.integers(9, 15))
    arm_w = int(rng.integers(5, w - 2))
    arm_h = int(rng.integers(5, h - 2))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :arm_w] = True
    mask[h - arm_h :, :] = True
    return _transform_mask(mask, rng)


def _t_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(10, 16))
    w = int(rng.integers(10, 16))
    bar_h = int(rng.integers(5, h - 2))
    stem_w = int(rng.integers(5, w - 2))
    stem_c0 = int(rng.integers(1, w - stem_w))
    mask = np.zeros((h, w), dtype=bool)
    mask[:bar_h, :] = True
    mask[:, stem_c0 : stem_c0 + stem_w] = True
    return _transform_mask(mask, rng)


def _plus_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(11, 17))
    w = int(rng.integers(11, 17))
    horiz_h = int(rng.integers(5, h - 2))
    vert_w = int(rng.integers(5, w - 2))
    bar_r0 = int(rng.integers(1, h - horiz_h))
    bar_c0 = int(rng.integers(1, w - vert_w))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, bar_c0 : bar_c0 + vert_w] = True
    mask[bar_r0 : bar_r0 + horiz_h, :] = True
    return mask


def _shape_candidates(rng: np.random.Generator) -> list[np.ndarray]:
    makers = (_rectangle_shape, _l_shape, _t_shape, _plus_shape)
    order = list(range(len(makers)))
    rng.shuffle(order)
    return [makers[i](rng) for i in order]


def _seed_candidates(cavity: np.ndarray, boundary: np.ndarray) -> list[tuple[int, int]]:
    dist = _distance_from_grey(cavity, boundary)
    h, w = cavity.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not cavity[r, c] or dist[r, c] < 2:
                continue
            safe = True
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    if boundary[nr, nc]:
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                cells.append((r, c))
    return cells


def _valid_single_region(boundary: np.ndarray, cavity: np.ndarray) -> bool:
    if not cavity.any():
        return False
    local = np.zeros((boundary.shape[0] + 2, boundary.shape[1] + 2), dtype=int)
    local[1:-1, 1:-1][boundary] = GREY
    return len(_enclosed_regions(local, GREY)) == 1 and len(_components(cavity)) == 1


def _make_object_patch(
    rng: np.random.Generator,
    seed_color: int,
    seed_count: int,
) -> np.ndarray | None:
    for filled in _shape_candidates(rng):
        boundary, cavity = _boundary_and_cavity(filled)
        if not _valid_single_region(boundary, cavity):
            continue
        candidates = _seed_candidates(cavity, boundary)
        if len(candidates) < seed_count:
            continue

        patch = np.zeros(filled.shape, dtype=int)
        patch[boundary] = GREY

        chosen_idx = rng.choice(len(candidates), size=seed_count, replace=False)
        for idx in np.atleast_1d(chosen_idx):
            r, c = candidates[int(idx)]
            patch[r, c] = int(seed_color)

        analyzed = _enclosed_regions(patch, GREY)
        if len(analyzed) != 1:
            continue
        seed_vals = [int(patch[r, c]) for r, c in analyzed[0] if patch[r, c] not in (BG, GREY)]
        if len(seed_vals) != seed_count or set(seed_vals) != {int(seed_color)}:
            continue
        if np.array_equal(main(patch), patch):
            continue
        return patch
    return None


def _reserve_box(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    if reserved[rr0:rr1, cc0:cc1].any():
        return False
    reserved[rr0:rr1, cc0:cc1] = True
    return True


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    if ph + 2 > gh or pw + 2 > gw:
        return False

    for _ in range(PLACE_RETRIES):
        r0 = int(rng.integers(1, gh - ph))
        c0 = int(rng.integers(1, gw - pw))
        if not _reserve_box(reserved, r0, c0, ph, pw, PLACEMENT_PAD):
            continue
        mask = patch != BG
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        sub[mask] = patch[mask]
        return True
    return False


def _generate_once(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    object_count = int(rng.integers(*OBJECT_COUNT_RANGE))
    seed_counts = [int(rng.integers(1, 4)) for _ in range(object_count)]
    colors = [int(v) for v in rng.choice(PALETTE, size=object_count, replace=False)]

    patches: list[np.ndarray] = []
    for color, count in zip(colors, seed_counts, strict=False):
        patch = _make_object_patch(rng, color, count)
        if patch is None:
            return None
        patches.append(patch)

    patches.sort(key=lambda p: p.shape[0] * p.shape[1], reverse=True)
    for patch in patches:
        if not _place_patch(grid, reserved, patch, rng):
            return None

    grey = _dominant_non_bg(grid, bg=BG)
    if grey != GREY:
        return None
    regions = _enclosed_regions(grid, GREY)
    if len(regions) != object_count:
        return None
    for comp in regions:
        vals = [int(grid[r, c]) for r, c in comp if int(grid[r, c]) not in (BG, GREY)]
        if not (1 <= len(vals) <= 3):
            return None
        if len(set(vals)) != 1:
            return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
