from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PAIR_COUNT_CHOICES = (2, 3, 4)
SCALE_CHOICES = (2, 3, 4)
GRID_MIN_SIDE = 18
GRID_MAX_SIDE = 30
MAX_GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    cur = mask
    for _ in range(4):
        for candidate in (cur, np.fliplr(cur)):
            key = (candidate.shape, candidate.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
        cur = np.rot90(cur)
    return variants


def _object_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    mask = patch != BG
    values, counts = np.unique(patch[mask], return_counts=True)
    main_color = int(values[counts.argmax()])
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "mask": mask,
        "colors": tuple(int(v) for v in values),
        "main_color": main_color,
    }


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = _components(grid != BG)
    return [_object_info(grid, coords) for coords in comps]


def _scale_match(small_mask: np.ndarray, large_mask: np.ndarray) -> int:
    sh, sw = small_mask.shape
    lh, lw = large_mask.shape
    if sh == 0 or sw == 0 or lh % sh != 0 or lw % sw != 0:
        return 0
    scale = lh // sh
    if scale <= 1 or lw // sw != scale:
        return 0
    expanded = np.kron(small_mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)
    if expanded.shape != large_mask.shape or not np.array_equal(expanded, large_mask):
        return 0
    return int(scale)


def _find_pairs(grid: np.ndarray) -> list[tuple[dict, dict, int]]:
    objects = _extract_objects(grid)
    smalls = [obj for obj in objects if len(obj["colors"]) >= 2]
    larges = [obj for obj in objects if len(obj["colors"]) == 1]
    assert smalls and len(smalls) == len(larges), "input must contain matching small/large pairs"

    pairs: list[tuple[dict, dict, int]] = []
    used_large: set[int] = set()
    for small in smalls:
        marker_mask = (small["patch"] != BG) & (small["patch"] != small["main_color"])
        assert np.any(marker_mask), "small template must contain marked pixels"

        candidates: list[tuple[int, int]] = []
        for idx, large in enumerate(larges):
            if large["main_color"] != small["main_color"]:
                continue
            scale = _scale_match(small["mask"], large["mask"])
            if scale:
                candidates.append((idx, scale))

        assert len(candidates) == 1, "each small template must match exactly one large shape"
        large_idx, scale = candidates[0]
        assert large_idx not in used_large, "large shapes must be paired uniquely"
        used_large.add(large_idx)
        pairs.append((small, larges[large_idx], scale))

    assert len(used_large) == len(larges), "all large shapes must be paired"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the large-shape regions selected by the marked pixels in each smaller template.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)

    for small, large, scale in _find_pairs(g):
        marker_mask = (small["patch"] != BG) & (small["patch"] != small["main_color"])
        keep_mask = np.kron(marker_mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)
        keep_mask &= large["mask"]

        r0, c0, r1, c1 = large["bbox"]
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[keep_mask] = large["main_color"]

    return out


def _random_shape(rng: random.Random, scale: int) -> np.ndarray | None:
    if scale == 4:
        h_range = (2, 3)
        w_range = (2, 3)
    elif scale == 3:
        h_range = (2, 4)
        w_range = (2, 4)
    else:
        h_range = (3, 5)
        w_range = (3, 5)

    for _ in range(SHAPE_ATTEMPTS):
        h = rng.randint(*h_range)
        w = rng.randint(*w_range)
        total = h * w
        min_area = max(5, total // 2)
        max_area = total - 1
        if min_area > max_area:
            continue

        target_area = rng.randint(min_area, max_area)
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells and (nr, nc) not in seen:
                        frontier.append((nr, nc))
                        seen.add((nr, nc))
            if not frontier:
                break

            if rng.random() < 0.7:
                best = max(sum((ar, ac) in cells for ar, ac in _neighbors4(r, c, h, w)) for r, c in frontier)
                dense = [pos for pos in frontier if sum((ar, ac) in cells for ar, ac in _neighbors4(pos[0], pos[1], h, w)) == best]
                cells.add(rng.choice(dense))
            else:
                cells.add(rng.choice(frontier))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)

        if int(mask.sum()) >= 7 and rng.random() < 0.6:
            boundary = [(r, c) for r, c in np.argwhere(mask) if any(not mask[nr, nc] for nr, nc in _neighbors4(r, c, *mask.shape))]
            rng.shuffle(boundary)
            for r, c in boundary:
                trial = mask.copy()
                trial[r, c] = False
                if _is_connected(trial):
                    trial = _crop_mask(trial)
                    if int(trial.sum()) >= 5 and int(trial.sum()) < trial.shape[0] * trial.shape[1]:
                        mask = trial
                        break

        mask = rng.choice(_mask_variants(mask))
        area = int(mask.sum())
        mh, mw = mask.shape
        if area < 5 or min(mh, mw) < 2:
            continue
        if area == mh * mw:
            continue
        if mh * scale > 12 or mw * scale > 12:
            continue
        return mask
    return None


def _marker_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    area = len(cells)
    upper = max(1, min((area - 1) // 2, area // 3 + 1))
    target = rng.randint(1, upper)

    chosen: set[tuple[int, int]] = set()
    if target == 1 or rng.random() < 0.5:
        chosen = set(rng.sample(cells, target))
    else:
        chosen.add(rng.choice(cells))
        allowed = set(cells)
        while len(chosen) < target:
            frontier = []
            seen = set()
            for r, c in chosen:
                for nr, nc in _neighbors4(r, c, *mask.shape):
                    if (nr, nc) in allowed and (nr, nc) not in chosen and (nr, nc) not in seen:
                        frontier.append((nr, nc))
                        seen.add((nr, nc))
            if frontier:
                chosen.add(rng.choice(frontier))
            else:
                remaining = [cell for cell in cells if cell not in chosen]
                chosen.add(rng.choice(remaining))

    marks = np.zeros_like(mask, dtype=bool)
    for r, c in chosen:
        marks[r, c] = True
    return marks


def _render_pair(mask: np.ndarray, main_color: int, marker_color: int, scale: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    marks = _marker_mask(mask, rng)

    small = np.full(mask.shape, BG, dtype=int)
    small[mask] = main_color
    small[marks] = marker_color

    large_mask = np.kron(mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)
    large = np.full(large_mask.shape, BG, dtype=int)
    large[large_mask] = main_color
    return small, large


def _place_patches(patches: list[np.ndarray], shape: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}

    order = list(range(len(patches)))
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph + 2 > h or pw + 2 > w:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(1, h - ph - 1)
            left = rng.randint(1, w - pw - 1)
            r0 = top - 1
            c0 = left - 1
            r1 = top + ph + 1
            c1 = left + pw + 1
            if reserved[r0:r1, c0:c1].any():
                continue
            placements[idx] = (top, left)
            reserved[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None

    for idx, patch in enumerate(patches):
        top, left = placements[idx]
        ph, pw = patch.shape
        target = grid[top : top + ph, left : left + pw]
        mask = patch != BG
        target[mask] = patch[mask]

    return grid


def _valid_example(grid: np.ndarray, expected_pairs: int) -> bool:
    try:
        pairs = _find_pairs(grid)
        out = main(grid)
    except AssertionError:
        return False
    if len(pairs) != expected_pairs:
        return False
    if out.shape != grid.shape or not np.any(out != BG):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    main_colors = rng.sample(list(range(1, 10)), pair_count)
    marker_pool = [c for c in range(1, 10) if c not in main_colors]

    patches: list[np.ndarray] = []
    for main_color in main_colors:
        scale = rng.choice(SCALE_CHOICES)
        mask = _random_shape(rng, scale)
        if mask is None:
            return None
        marker_color = rng.choice(marker_pool)
        small, large = _render_pair(mask, main_color, marker_color, scale, rng)
        patches.extend([small, large])

    padded_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in patches)
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)

    side_floor = max(
        GRID_MIN_SIDE,
        max_h + 3,
        max_w + 3,
        int(np.ceil(np.sqrt(padded_area * 1.4))),
    )
    if side_floor > GRID_MAX_SIDE:
        return None

    for _ in range(8):
        grid_h = rng.randint(side_floor, GRID_MAX_SIDE)
        grid_w = rng.randint(side_floor, GRID_MAX_SIDE)
        grid = _place_patches(patches, (grid_h, grid_w), rng)
        if grid is not None and _valid_example(grid, pair_count):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid masking-scale ARC input")
