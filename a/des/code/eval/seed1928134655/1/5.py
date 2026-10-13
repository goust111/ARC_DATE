from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


GREY = 5
GREEN = 3
BG_CHOICES = [0, 1, 2, 4, 6, 7, 8]
KEY_COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]

GRID_H_RANGE = (24, 33)
GRID_W_RANGE = (24, 33)
INNER_H_RANGE = (3, 6)
INNER_W_RANGE = (3, 6)
PLACEMENT_PAD = 1
MAX_GENERATION_ATTEMPTS = 400

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    neigh = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or not bool(mask[r, c]):
                continue
            visited[r, c] = True
            stack = [(r, c)]
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in neigh:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and bool(mask[nr, nc]):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_color_components(
    grid: np.ndarray,
    excluded: set[int] | None = None,
    colors: set[int] | None = None,
    connectivity: int = 8,
) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    neigh = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    excluded = set() if excluded is None else set(map(int, excluded))
    wanted = None if colors is None else set(map(int, colors))
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in excluded:
                continue
            if wanted is not None and color not in wanted:
                continue

            visited[r, c] = True
            stack = [(r, c)]
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in neigh:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(len(coords)),
                }
            )
    return comps


def _trim_mask(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size > 0 and cols.size > 0
    r0 = int(rows[0])
    r1 = int(rows[-1])
    c0 = int(cols[0])
    c1 = int(cols[-1])
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), r0, c0


def _touches_bbox_boundary(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _extract_lock_data(grid: np.ndarray, bg: int) -> tuple[list[dict], set[tuple[int, int]]]:
    h, w = grid.shape
    grey_comps = _find_color_components(grid, colors={GREY}, connectivity=8)
    assert grey_comps, "expected at least one grey lock outline"

    locks = []
    indicator_positions: set[tuple[int, int]] = set()
    indicator_colors = set()

    for comp in grey_comps:
        r0, c0, r1, c1 = comp["bbox"]
        assert r1 + 1 < h and c1 + 1 < w, "indicator pixel must fit at lower-right exterior"

        indicator_pos = (int(r1 + 1), int(c1 + 1))
        indicator_color = int(grid[indicator_pos])
        assert indicator_color not in {bg, GREY}, "indicator pixel must be a distinct non-grey color"
        assert indicator_pos not in indicator_positions
        assert indicator_color not in indicator_colors

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        bg_components = _components_mask(patch == bg, connectivity=4)
        hole_candidates = [pts for pts in bg_components if not _touches_bbox_boundary(pts, patch.shape)]
        assert len(hole_candidates) == 1, "each grey lock should enclose exactly one cavity"

        hole_mask = np.zeros(patch.shape, dtype=bool)
        for rr, cc in hole_candidates[0]:
            hole_mask[rr, cc] = True
        inner_mask, inner_r0, inner_c0 = _trim_mask(hole_mask)
        inner_h, inner_w = inner_mask.shape
        assert bool(inner_mask.any())

        locks.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "indicator_pos": indicator_pos,
                "indicator_color": indicator_color,
                "inner_mask": inner_mask,
                "inner_origin": (int(r0 + inner_r0), int(c0 + inner_c0)),
                "inner_shape": (int(inner_h), int(inner_w)),
            }
        )
        indicator_positions.add(indicator_pos)
        indicator_colors.add(indicator_color)

    return locks, indicator_positions


def _mask_fits(inner_mask: np.ndarray, key_mask: np.ndarray) -> bool:
    kh, kw = key_mask.shape
    ih, iw = inner_mask.shape
    if kh > ih or kw > iw:
        return False
    return bool(np.all(inner_mask[:kh, :kw][key_mask]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match keys to lock cavities by color, rotate the key to the unique fitting orientation,
    and fill the remaining cavity cells with green.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    locks, indicator_positions = _extract_lock_data(g, bg)

    work = g.copy()
    for r, c in indicator_positions:
        work[r, c] = bg

    key_comps = _find_color_components(work, excluded={bg, GREY}, connectivity=8)
    keys_by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in key_comps:
        keys_by_color[int(comp["color"])].append(comp)

    out = g.copy()
    for lock in locks:
        color = int(lock["indicator_color"])
        assert len(keys_by_color[color]) == 1, "each indicator color should identify exactly one key"
        key = keys_by_color[color][0]

        fits: list[tuple[int, np.ndarray]] = []
        for k in range(4):
            rot_mask = np.rot90(key["mask"], k)
            if _mask_fits(lock["inner_mask"], rot_mask):
                fits.append((int(k), rot_mask))
        assert len(fits) == 1, "key rotation must be uniquely determined"

        key_color = int(key["color"])
        for rr, cc in key["coords"]:
            out[rr, cc] = bg

        inner_r0, inner_c0 = lock["inner_origin"]
        inner_h, inner_w = lock["inner_shape"]
        inner_view = out[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w]
        inner_view[lock["inner_mask"]] = GREEN

        rot_mask = fits[0][1]
        kh, kw = rot_mask.shape
        key_view = out[inner_r0 : inner_r0 + kh, inner_c0 : inner_c0 + kw]
        key_view[rot_mask] = key_color

    return out


def _has_hole(mask: np.ndarray) -> bool:
    false_comps = _components_mask(~mask, connectivity=4)
    for comp in false_comps:
        if not _touches_bbox_boundary(comp, mask.shape):
            return True
    return False


def _grow_connected_mask(allowed: np.ndarray, area: int) -> np.ndarray | None:
    cells = [tuple(map(int, x)) for x in np.argwhere(allowed)]
    if not cells or area < 1 or area > len(cells):
        return None

    start = random.choice(cells)
    mask = np.zeros_like(allowed, dtype=bool)
    mask[start] = True

    while int(mask.sum()) < area:
        frontier = set()
        for rr, cc in np.argwhere(mask):
            for dr, dc in NEIGHBORS4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and bool(allowed[nr, nc]) and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
        if not frontier:
            return None
        pick = random.choice(list(frontier))
        mask[pick] = True
    return mask


def _generate_inner_mask() -> np.ndarray | None:
    h = random.randint(*INNER_H_RANGE)
    w = random.randint(*INNER_W_RANGE)
    min_area = max(h + w - 1, 5)
    max_area = h * w
    if min_area > max_area:
        return None

    for _ in range(120):
        if random.random() < 0.2:
            area = h * w
        else:
            area = random.randint(min_area, max_area - 1 if max_area > min_area else max_area)
        mask = _grow_connected_mask(np.ones((h, w), dtype=bool), area)
        if mask is None:
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if _has_hole(mask):
            continue
        return mask
    return None


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _generate_key_fit_mask(inner_mask: np.ndarray) -> np.ndarray | None:
    ih, iw = inner_mask.shape
    inner_area = int(inner_mask.sum())

    size_candidates = []
    for kh in range(2, ih + 1):
        for kw in range(2, iw + 1):
            allowed = inner_mask[:kh, :kw]
            if _touches_all_sides(allowed):
                size_candidates.append((kh, kw))
    random.shuffle(size_candidates)

    for kh, kw in size_candidates:
        allowed = inner_mask[:kh, :kw]
        allowed_count = int(allowed.sum())
        max_area = min(allowed_count, inner_area - 1)
        if max_area < 2:
            continue
        min_area = min(max_area, max(2, kh + kw - 3))

        for _ in range(120):
            area = random.randint(min_area, max_area)
            mask = _grow_connected_mask(allowed, area)
            if mask is None:
                continue
            if not _touches_all_sides(mask):
                continue
            fit_count = 0
            for rot in range(4):
                if _mask_fits(inner_mask, np.rot90(mask, rot)):
                    fit_count += 1
            if fit_count == 1:
                return mask
    return None


def _make_lock_patch(inner_mask: np.ndarray, indicator_color: int, bg: int) -> np.ndarray:
    ih, iw = inner_mask.shape
    patch = np.full((ih + 3, iw + 3), bg, dtype=int)
    patch[: ih + 2, : iw + 2] = GREY
    interior = patch[1 : ih + 1, 1 : iw + 1]
    interior[inner_mask] = bg
    patch[ih + 2, iw + 2] = int(indicator_color)
    return patch


def _mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, pad: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    for rr, cc in np.argwhere(patch != bg):
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        r1 = max(0, gr - pad)
        r2 = min(h, gr + pad + 1)
        c1 = max(0, gc - pad)
        c2 = min(w, gc + pad + 1)
        if np.any(grid[r1:r2, c1:c2] != bg):
            return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != bg
    target = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    target[mask] = patch[mask]


def _random_place(grid: np.ndarray, patch: np.ndarray, bg: int, pad: int) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    starts = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(starts)
    for r0, c0 in starts:
        if _can_place(grid, patch, r0, c0, bg, pad):
            _place_patch(grid, patch, r0, c0, bg)
            return int(r0), int(c0)
    return None


def _construct_expected(
    inp: np.ndarray,
    placements: list[dict],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for item in placements:
        key_r0, key_c0 = item["key_pos"]
        key_patch = item["key_patch"]
        key_mask = key_patch != bg
        out[key_r0 : key_r0 + key_patch.shape[0], key_c0 : key_c0 + key_patch.shape[1]][key_mask] = bg

        inner_r0 = int(item["lock_pos"][0] + 1)
        inner_c0 = int(item["lock_pos"][1] + 1)
        inner_mask = item["inner_mask"]
        ih, iw = inner_mask.shape
        out[inner_r0 : inner_r0 + ih, inner_c0 : inner_c0 + iw][inner_mask] = GREEN

        placed_key = np.rot90(item["key_mask_input"], item["rotation_to_fit"])
        kh, kw = placed_key.shape
        out[inner_r0 : inner_r0 + kh, inner_c0 : inner_c0 + kw][placed_key] = int(item["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    key_colors = [c for c in KEY_COLOR_POOL if c not in {bg, GREY, GREEN}]
    pair_count = random.choices([1, 2, 3, 4], weights=[4, 3, 2, 1], k=1)[0]
    if pair_count > len(key_colors):
        return None
    colors = random.sample(key_colors, pair_count)

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), bg, dtype=int)
    placements: list[dict] = []

    for color in colors:
        inner_mask = _generate_inner_mask()
        if inner_mask is None:
            return None
        fit_mask = _generate_key_fit_mask(inner_mask)
        if fit_mask is None:
            return None

        rotation_to_fit = random.randint(0, 3)
        key_mask_input = np.rot90(fit_mask, -rotation_to_fit)
        lock_patch = _make_lock_patch(inner_mask, color, bg)
        key_patch = _mask_to_patch(key_mask_input, color, bg)

        lock_pos = _random_place(grid, lock_patch, bg, PLACEMENT_PAD)
        if lock_pos is None:
            return None
        key_pos = _random_place(grid, key_patch, bg, PLACEMENT_PAD)
        if key_pos is None:
            return None

        placements.append(
            {
                "color": int(color),
                "inner_mask": inner_mask,
                "rotation_to_fit": int(rotation_to_fit),
                "key_mask_input": key_mask_input,
                "lock_pos": lock_pos,
                "key_pos": key_pos,
                "key_patch": key_patch,
            }
        )

    if _get_bg(grid) != bg:
        return None

    expected = _construct_expected(grid, placements, bg)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
