from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
CONTOUR = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = comp_id
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _pattern_has_full_period(seed: np.ndarray) -> bool:
    h, w = seed.shape
    for ph in range(1, h):
        if h % ph == 0:
            proto = seed[:ph, :]
            if np.array_equal(seed, np.tile(proto, (h // ph, 1))):
                return False
    for pw in range(1, w):
        if w % pw == 0:
            proto = seed[:, :pw]
            if np.array_equal(seed, np.tile(proto, (1, w // pw))):
                return False
    return True


def _extract_seed(
    grid: np.ndarray,
    region_mask: np.ndarray,
    bg: int,
) -> tuple[int, int, np.ndarray]:
    coords = np.argwhere(region_mask)
    r0, c0, _r1, _c1 = _bbox_from_coords(coords)
    allowed = (grid != bg) & (grid != CONTOUR) & (grid != MAROON) & region_mask
    assert allowed[r0, c0], "region must start with its seed block"

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for sh in (2, 3):
        for sw in (2, 3):
            if r0 + sh > grid.shape[0] or c0 + sw > grid.shape[1]:
                continue
            sub_mask = region_mask[r0 : r0 + sh, c0 : c0 + sw]
            sub_vals = grid[r0 : r0 + sh, c0 : c0 + sw]
            if sub_mask.shape != (sh, sw):
                continue
            if np.all(sub_mask) and np.all((sub_vals != bg) & (sub_vals != CONTOUR) & (sub_vals != MAROON)):
                candidates.append((sh * sw, sh, sw, sub_vals.copy()))

    assert candidates, "failed to detect the top-left seed block"
    _area, sh, sw, seed = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return int(r0), int(c0), seed


def _find_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    bg_labels, n_bg = _label_mask(g == bg)
    assert n_bg >= 2, "expected enclosed and exterior background"
    bg_sizes = {cid: int(np.count_nonzero(bg_labels == cid)) for cid in range(1, n_bg + 1)}
    outside_bg_id = max(bg_sizes, key=bg_sizes.get)

    passable_labels, n_pass = _label_mask(g != CONTOUR)
    regions: list[dict] = []
    for cid in range(1, n_pass + 1):
        coords = np.argwhere(passable_labels == cid)
        if coords.size == 0 or _touches_border(coords, g.shape):
            continue
        bg_ids = {
            int(bg_labels[int(r), int(c)])
            for r, c in coords
            if int(g[int(r), int(c)]) == bg and int(bg_labels[int(r), int(c)]) != 0
        }
        if not bg_ids:
            continue
        assert outside_bg_id not in bg_ids, "enclosed region cannot include exterior background"
        assert len(bg_ids) == 1, "generator should keep one background component per enclosure"

        region_mask = passable_labels == cid
        seed_r, seed_c, seed = _extract_seed(g, region_mask, bg)
        regions.append(
            {
                "coords": coords,
                "mask": region_mask,
                "bbox": _bbox_from_coords(coords),
                "seed_r": seed_r,
                "seed_c": seed_c,
                "seed": seed,
                "bg_id": next(iter(bg_ids)),
            }
        )

    regions.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each enclosure's top-left seed motif across that enclosure while keeping maroon distractors untouched.

    Input structure:
    - Black background surrounds several gray 8-connected closed contour loops.
    - Each contour encloses one passable interior region containing a small top-left
      rectangular seed motif and a few maroon distractor pixels.
    - Extra non-background distractor pixels may also appear outside every contour.

    Rule:
    - Treat all non-background pixels as barriers to identify background components;
      the largest background component is the exterior and the others lie inside contours.
    - For each enclosed contour region, read the contiguous non-background, non-gray,
      non-maroon rectangle anchored at that region's top-left corner as the seed motif.
    - Starting from that same top-left anchor, tile the seed with its own height and
      width periods across the enclosed region only, clipping to the region boundary.
      Maroon distractor pixels are skipped and remain unchanged.

    Key insight:
    - This is not one global wallpaper: every enclosure has its own phase origin at
      its own top-left seed, and the gray contour strictly clips where the repeated
      motif may appear.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)

    for region in _find_regions(g):
        seed = region["seed"]
        sh, sw = seed.shape
        r0 = int(region["seed_r"])
        c0 = int(region["seed_c"])
        for r, c in region["coords"]:
            rr = int(r)
            cc = int(c)
            if int(g[rr, cc]) == MAROON:
                continue
            out[rr, cc] = int(seed[(rr - r0) % sh, (cc - c0) % sw])

    # Exterior content stays unchanged; only enclosed passable cells are rewritten.
    assert out.shape == g.shape
    assert _get_bg(out) == bg or np.count_nonzero(out == bg) < np.count_nonzero(g == bg)
    return out


def _connected(mask: np.ndarray) -> bool:
    labels, n = _label_mask(mask)
    return n == 1 and labels.any()


def _make_region_shape(seed_h: int, seed_w: int, force_irregular: bool) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(max(6, seed_h + 3), 9)
        w = random.randint(max(6, seed_w + 3), 9)
        mask = np.ones((h, w), dtype=bool)
        irregular = False

        notch_count = random.choice([0, 1, 1, 2, 2, 3])
        for _ in range(notch_count):
            side = random.choice(("bottom", "right", "corner"))
            if side == "bottom" and h >= seed_h + 3 and w >= 4:
                depth = random.randint(1, min(3, h - seed_h - 1))
                width = random.randint(1, w - 2)
                start = random.randint(0, w - width)
                if width < w:
                    mask[h - depth : h, start : start + width] = False
                    irregular = True
            elif side == "right" and w >= seed_w + 3 and h >= 4:
                width = random.randint(1, min(3, w - seed_w - 1))
                height = random.randint(1, h - 2)
                start = random.randint(0, h - height)
                if height < h:
                    mask[start : start + height, w - width : w] = False
                    irregular = True
            elif side == "corner" and h >= seed_h + 3 and w >= seed_w + 3:
                depth = random.randint(1, min(3, h - seed_h - 1))
                width = random.randint(1, min(3, w - seed_w - 1))
                mask[h - depth : h, w - width : w] = False
                irregular = True

        if force_irregular and not irregular:
            continue
        if not np.all(mask[:seed_h, :seed_w]):
            continue
        if not mask[0, 0]:
            continue
        if not _connected(mask):
            continue
        area = int(mask.sum())
        if not (25 <= area <= 60):
            continue
        if h * w < 4 * seed_h * seed_w:
            continue
        if h <= seed_h or w <= seed_w:
            continue
        return mask
    return None


def _make_seed_pattern(shape: tuple[int, int], forbidden: set[bytes]) -> np.ndarray | None:
    sh, sw = shape
    for _ in range(160):
        palette_size = 2 if sh * sw <= 6 else random.choice([2, 3])
        palette = random.sample(SEED_COLORS, palette_size)
        seed = np.array([[random.choice(palette) for _ in range(sw)] for _ in range(sh)], dtype=int)
        if len(set(map(int, seed.ravel()))) < 2:
            continue
        if np.all(seed == seed[0, 0]):
            continue
        if not np.any(seed[1:, :] != seed[:-1, :]):
            continue
        if not np.any(seed[:, 1:] != seed[:, :-1]):
            continue
        if not _pattern_has_full_period(seed):
            continue
        key = seed.tobytes()
        if key in forbidden:
            continue
        return seed
    return None


def _pick_maroon_cells(region_mask: np.ndarray, seed_h: int, seed_w: int) -> list[tuple[int, int]] | None:
    candidates = [
        (int(r), int(c))
        for r, c in np.argwhere(region_mask)
        if not (int(r) < seed_h and int(c) < seed_w)
    ]
    if len(candidates) < 3:
        return None
    count = random.randint(1, min(3, len(candidates)))
    random.shuffle(candidates)
    return candidates[:count]


def _single_interior_bg_component(local_region: np.ndarray, patch: np.ndarray) -> bool:
    bg_inside = local_region & (patch == BG)
    labels, n = _label_mask(bg_inside)
    if n != 1:
        return False
    coords = np.argwhere(bg_inside)
    return coords.size > 0 and not _touches_border(coords, local_region.shape)


def _local_region_patch(shape_mask: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    h, w = shape_mask.shape
    local_region = np.zeros((h + 2, w + 2), dtype=bool)
    local_region[1 : h + 1, 1 : w + 1] = shape_mask

    contour_mask = _dilate8(local_region) & ~local_region
    for _ in range(80):
        maroons = _pick_maroon_cells(shape_mask, seed.shape[0], seed.shape[1])
        if maroons is None:
            return None

        patch = np.full((h + 2, w + 2), BG, dtype=int)
        patch[contour_mask] = CONTOUR
        patch[1 : 1 + seed.shape[0], 1 : 1 + seed.shape[1]] = seed
        for r, c in maroons:
            patch[r + 1, c + 1] = MAROON
        if _single_interior_bg_component(local_region, patch):
            return patch, local_region, [(r + 1, c + 1) for r, c in maroons]
    return None


def _can_place(occupied: np.ndarray, footprint: np.ndarray, top: int, left: int) -> bool:
    h, w = footprint.shape
    if top < 1 or left < 1 or top + h >= occupied.shape[0] or left + w >= occupied.shape[1]:
        return False
    blocked = _dilate8(occupied)
    return not np.any(blocked[top : top + h, left : left + w] & footprint)


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    footprint: np.ndarray,
) -> tuple[int, int] | None:
    h, w = patch.shape
    for _ in range(200):
        top = random.randint(1, grid.shape[0] - h - 1)
        left = random.randint(1, grid.shape[1] - w - 1)
        if not _can_place(occupied, footprint, top, left):
            continue
        sub = grid[top : top + h, left : left + w]
        if np.any((patch != BG) & (sub != BG)):
            continue
        sub[patch != BG] = patch[patch != BG]
        occupied[top : top + h, left : left + w] |= footprint
        return top, left
    return None


def _build_expected(grid: np.ndarray, metas: list[dict]) -> np.ndarray:
    out = grid.copy()
    for meta in metas:
        seed = meta["seed"]
        sh, sw = seed.shape
        r0 = meta["seed_anchor"][0]
        c0 = meta["seed_anchor"][1]
        for r, c in meta["region_coords"]:
            if int(grid[r, c]) == MAROON:
                continue
            out[r, c] = int(seed[(r - r0) % sh, (c - c0) % sw])
    return out


def _add_outside_distractors(grid: np.ndarray, occupied: np.ndarray) -> bool:
    bg_labels, n_bg = _label_mask(grid == BG)
    if n_bg == 0:
        return False
    sizes = {cid: int(np.count_nonzero(bg_labels == cid)) for cid in range(1, n_bg + 1)}
    outside_id = max(sizes, key=sizes.get)
    outside_cells = [(int(r), int(c)) for r, c in np.argwhere(bg_labels == outside_id)]
    if len(outside_cells) < 8:
        return False

    for _ in range(random.randint(2, 6)):
        random.shuffle(outside_cells)
        placed = False
        for r, c in outside_cells:
            if grid[r, c] != BG:
                continue
            color = random.choice(SEED_COLORS + (MAROON,))
            grid[r, c] = color
            occupied[r, c] = True
            placed = True
            if random.random() < 0.35:
                nbrs = [
                    (nr, nc)
                    for nr, nc in _neighbors4(r, c)
                    if 0 <= nr < grid.shape[0]
                    and 0 <= nc < grid.shape[1]
                    and bg_labels[nr, nc] == outside_id
                    and grid[nr, nc] == BG
                ]
                if nbrs:
                    nr, nc = random.choice(nbrs)
                    grid[nr, nc] = color
                    occupied[nr, nc] = True
            break
        if not placed:
            return False
    return True


def _validate_semantics(grid: np.ndarray, expected: np.ndarray, metas: list[dict]) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    regions = _find_regions(grid)
    if len(regions) != len(metas) or not (2 <= len(regions) <= 3):
        return False

    irregular_count = 0
    for region, meta in zip(regions, sorted(metas, key=lambda item: (item["bbox"][0], item["bbox"][1]))):
        seed = region["seed"]
        if not np.array_equal(seed, meta["seed"]):
            return False
        if not _pattern_has_full_period(seed):
            return False
        if len(set(map(int, seed.ravel()))) < 2:
            return False
        r0, c0, r1, c1 = region["bbox"]
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if seed.shape[0] * seed.shape[1] > bbox_area // 4:
            return False
        region_area = int(region["coords"].shape[0])
        if not (25 <= region_area <= 60):
            return False
        maroon_count = sum(int(grid[int(r), int(c)] == MAROON) for r, c in region["coords"])
        if not (1 <= maroon_count <= 3):
            return False
        local_mask = region["mask"][r0 : r1 + 1, c0 : c1 + 1]
        if int(local_mask.sum()) != bbox_area:
            irregular_count += 1
        tiled_colors = {int(expected[int(r), int(c)]) for r, c in region["coords"] if int(grid[int(r), int(c)]) != MAROON}
        if len(tiled_colors) < 2:
            return False
        for r, c in region["coords"]:
            rr = int(r)
            cc = int(c)
            if int(grid[rr, cc]) == MAROON:
                if int(expected[rr, cc]) != MAROON:
                    return False
                continue
            want = int(seed[(rr - region["seed_r"]) % seed.shape[0], (cc - region["seed_c"]) % seed.shape[1]])
            if int(expected[rr, cc]) != want:
                return False

    if irregular_count == 0:
        return False

    enclosed_mask = np.zeros_like(grid, dtype=bool)
    for meta in metas:
        enclosed_mask[meta["region_coords"][:, 0], meta["region_coords"][:, 1]] = True
    outside_changed = (grid != expected) & ~enclosed_mask
    if np.any(outside_changed):
        return False
    if np.array_equal(grid, expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(22, 28)
    w = random.randint(22, 28)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    region_count = random.choice([2, 3])
    metas: list[dict] = []
    used_seed_bytes: set[bytes] = set()
    need_irregular_index = random.randrange(region_count)

    for idx in range(region_count):
        seed_shape = random.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
        shape_mask = _make_region_shape(seed_shape[0], seed_shape[1], force_irregular=(idx == need_irregular_index))
        if shape_mask is None:
            return None
        seed = _make_seed_pattern(seed_shape, used_seed_bytes)
        if seed is None:
            return None
        used_seed_bytes.add(seed.tobytes())
        local = _local_region_patch(shape_mask, seed)
        if local is None:
            return None
        patch, local_region, maroons = local
        footprint = patch != BG
        placed = _place_patch(grid, occupied, patch, footprint)
        if placed is None:
            return None
        top, left = placed

        region_coords = np.argwhere(local_region) + np.array([top, left])
        bbox = _bbox_from_coords(region_coords)
        metas.append(
            {
                "seed": seed,
                "seed_anchor": (top + 1, left + 1),
                "region_coords": region_coords,
                "bbox": bbox,
                "maroons": [(top + r, left + c) for r, c in maroons],
            }
        )

    if not _add_outside_distractors(grid, occupied):
        return None

    expected = _build_expected(grid, metas)
    if not _validate_semantics(grid, expected, metas):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-tiling ARC input")
