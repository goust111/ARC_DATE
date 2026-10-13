from __future__ import annotations

from collections import deque
import hashlib
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3

_SEEN_OUTPUT_SIGNATURES: set[str] = set()
_SEEN_OUTPUT_KEYS: set[tuple[int, str]] = set()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _flood_exterior(passable: np.ndarray) -> np.ndarray:
    h, w = map(int, passable.shape)
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bool(passable[r, c]) and not bool(exterior[r, c]):
                exterior[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if bool(passable[r, c]) and not bool(exterior[r, c]):
                exterior[r, c] = True
                q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and bool(passable[nr, nc]) and not bool(exterior[nr, nc]):
                exterior[nr, nc] = True
                q.append((int(nr), int(nc)))
    return exterior


def _analyze_regions(grid: np.ndarray, *, require_seed: bool = True) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    red_cells = np.argwhere(g == RED)
    if require_seed:
        assert int(red_cells.shape[0]) == 1, "exactly one red seed is required"
    else:
        assert int(red_cells.shape[0]) in (0, 1), "at most one red seed is allowed"

    passable = g != GREEN
    exterior = _flood_exterior(passable)
    enclosed = passable & ~exterior
    regions: list[dict] = []
    seed_region = -1
    for coords in _component_coords(enclosed):
        has_seed = any(int(g[int(r), int(c)]) == RED for r, c in coords)
        boundary: set[tuple[int, int]] = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == GREEN:
                    boundary.add((int(nr), int(nc)))
        assert boundary, "every enclosed region must have a green boundary"
        boundary_coords = np.array(sorted(boundary), dtype=int)
        union = np.vstack([coords, boundary_coords])
        regions.append(
            {
                "coords": coords,
                "boundary": boundary_coords,
                "bbox": _bbox_from_coords(union),
                "area": int(coords.shape[0]),
                "has_seed": bool(has_seed),
            }
        )
        if has_seed:
            assert seed_region == -1, "seed must belong to exactly one enclosed region"
            seed_region = int(len(regions) - 1)

    if require_seed:
        assert seed_region != -1, "red seed must lie inside a closed region"
    return regions, seed_region


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the seed-marked closed region and tile it by the total number of enclosed regions.

    Input structure:
    - A black background contains several disjoint green closed boundary loops.
    - Each loop encloses one black interior region, and exactly one enclosed region contains a red seed pixel.
    - The green loops do not overlap, and the seed belongs to only one topologically closed region.

    Rule:
    - Treat green pixels as walls and flood fill from the outside through every non-green cell to separate exterior from enclosed regions.
    - Count the enclosed regions; let that count be N.
    - Find the enclosed region that contains the red seed, crop the minimal bounding box of that region together with its enclosing green boundary, and replace the red seed with black inside the crop.
    - Tile that cropped patch N times vertically and N times horizontally to form the output.

    Key insight:
    - One topological analysis provides two different facts at once: the total number of closed regions determines the tiling factor, while the seed identifies which specific closed region should be cropped as the tile source.
    """
    g = np.asarray(grid, dtype=int)
    regions, seed_region = _analyze_regions(g)
    n = int(len(regions))
    r0, c0, r1, c1 = map(int, regions[seed_region]["bbox"])
    tile = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    tile[tile == RED] = BLACK
    return np.tile(tile, (n, n))


def _rect_patch(h: int, w: int) -> np.ndarray:
    return np.ones((int(h), int(w)), dtype=bool)


def _apply_corner_cut(mask: np.ndarray, corner: int, style: int) -> None:
    if int(style) == 0:
        return
    h, w = map(int, mask.shape)
    if corner == 0:
        r0, c0, dr, dc = 0, 0, 1, 1
    elif corner == 1:
        r0, c0, dr, dc = 0, w - 1, 1, -1
    elif corner == 2:
        r0, c0, dr, dc = h - 1, 0, -1, 1
    else:
        r0, c0, dr, dc = h - 1, w - 1, -1, -1
    cells = [(r0, c0)]
    if int(style) >= 2:
        cells.extend([(r0 + dr, c0), (r0, c0 + dc)])
    if int(style) >= 3:
        cells.append((r0 + dr, c0 + dc))
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = False


def _boundary_from_filled(filled: np.ndarray) -> np.ndarray:
    h, w = map(int, filled.shape)
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(filled[r, c]):
                continue
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds((h, w), nr, nc) or not bool(filled[nr, nc]):
                    out[r, c] = True
                    break
    return out


def _shape_signature(arr: np.ndarray) -> str:
    return hashlib.md5(arr.astype(np.uint8).tobytes() + str(arr.shape).encode()).hexdigest()


def _is_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    return len(_component_coords(mask)) == 1


def _is_rect_boundary(boundary: np.ndarray) -> bool:
    h, w = map(int, boundary.shape)
    target = np.zeros((h, w), dtype=bool)
    target[0, :] = True
    target[-1, :] = True
    target[:, 0] = True
    target[:, -1] = True
    return bool(np.array_equal(boundary, target))


def _build_patch_library() -> list[dict]:
    patches: dict[str, dict] = {}
    for h in range(4, 7):
        for w in range(4, 7):
            if not (4 <= (h - 2) * (w - 2) <= 12):
                continue
            for styles in np.ndindex((4, 4, 4, 4)):
                filled = _rect_patch(h, w)
                for corner, style in enumerate(styles):
                    _apply_corner_cut(filled, int(corner), int(style))
                if not bool(filled.any()) or not _is_connected(filled):
                    continue
                boundary = _boundary_from_filled(filled)
                if not _is_connected(boundary):
                    continue
                patch = np.zeros((h, w), dtype=int)
                patch[boundary] = GREEN
                regions, seed_idx = _analyze_regions(patch, require_seed=False)
                if int(seed_idx) != -1:
                    continue
                if len(regions) != 1:
                    continue
                area = int(regions[0]["area"])
                if not (4 <= area <= 12):
                    continue
                sig = _shape_signature(patch)
                if sig in patches:
                    continue
                interior = np.zeros((h, w), dtype=bool)
                coords = regions[0]["coords"]
                interior[coords[:, 0], coords[:, 1]] = True
                patches[sig] = {
                    "patch": patch,
                    "interior": interior,
                    "bbox": regions[0]["bbox"],
                    "area": area,
                    "is_rect": _is_rect_boundary(boundary),
                    "sig": sig,
                }
    lib = list(patches.values())
    assert len(lib) >= 12, "need a diverse library of loop patches"
    return lib


_PATCH_LIBRARY = _build_patch_library()


def _green_components(grid: np.ndarray) -> list[np.ndarray]:
    return _component_coords(np.asarray(grid, dtype=int) == GREEN)


def _min_component_gap(comps: list[np.ndarray]) -> int:
    best = 10**9
    for i in range(len(comps)):
        a = comps[i]
        for j in range(i + 1, len(comps)):
            b = comps[j]
            for r1, c1 in a:
                for r2, c2 in b:
                    dist = max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2)))
                    best = min(best, int(dist))
    return 0 if best == 10**9 else int(best)


def _validate_input(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BLACK, RED, GREEN})

    regions, seed_region = _analyze_regions(g)
    assert 3 <= len(regions) <= 5

    green_comps = _green_components(g)
    assert len(green_comps) == len(regions), "each loop must correspond to one enclosed region"
    assert _min_component_gap(green_comps) >= 2, "green boundaries must keep at least one full cell gap"

    seed_count = 0
    shape_sigs: set[str] = set()
    has_non_rect = False
    for region in regions:
        assert 4 <= int(region["area"]) <= 12
        r0, c0, r1, c1 = map(int, region["bbox"])
        tile = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        tile[tile == RED] = BLACK
        sig = _shape_signature(tile)
        assert sig not in shape_sigs, "region crops must be pairwise distinct"
        shape_sigs.add(sig)
        if not _is_rect_boundary(tile == GREEN):
            has_non_rect = True
        if bool(region["has_seed"]):
            seed_count += 1

    assert seed_count == 1
    assert has_non_rect, "at least one region must be non-rectangular"

    out = main(g)
    assert out.ndim == 2 and max(out.shape) <= 30
    target_bbox = regions[seed_region]["bbox"]
    th = int(target_bbox[2] - target_bbox[0] + 1)
    tw = int(target_bbox[3] - target_bbox[1] + 1)
    assert out.shape == (len(regions) * th, len(regions) * tw)
    return regions, seed_region


def _random_patch_pool(count: int) -> list[dict]:
    tries = 300
    for _ in range(tries):
        pool = random.sample(_PATCH_LIBRARY, count)
        if any(not item["is_rect"] for item in pool):
            return pool
    raise RuntimeError("failed to sample a diverse patch pool")


def _choose_output_key() -> tuple[int, dict] | None:
    options = [
        (n_regions, item)
        for n_regions in range(3, 6)
        for item in _PATCH_LIBRARY
        if (n_regions, str(item["sig"])) not in _SEEN_OUTPUT_KEYS
    ]
    if not options:
        return None
    return random.choice(options)


def _placeable(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    bh, bw = blocked.shape
    rr0 = max(0, int(r) - 2)
    cc0 = max(0, int(c) - 2)
    rr1 = min(bh, int(r) + int(h) + 2)
    cc1 = min(bw, int(c) + int(w) + 2)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_blocked(blocked: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    rr0 = max(0, int(r) - 2)
    cc0 = max(0, int(c) - 2)
    rr1 = min(blocked.shape[0], int(r) + int(h) + 2)
    cc1 = min(blocked.shape[1], int(c) + int(w) + 2)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = map(int, patch.shape)
    gh, gw = map(int, grid.shape)
    if gh - h < 2 or gw - w < 2:
        return None
    for _ in range(200):
        r = random.randint(1, gh - h - 1)
        c = random.randint(1, gw - w - 1)
        if not _placeable(blocked, r, c, h, w):
            continue
        grid[r : r + h, c : c + w][patch == GREEN] = GREEN
        _mark_blocked(blocked, r, c, h, w)
        return int(r), int(c)
    return None


def _gen_attempt() -> np.ndarray | None:
    choice = _choose_output_key()
    if choice is None:
        return None
    n_regions, target_item = choice
    for _ in range(100):
        distractors = random.sample([item for item in _PATCH_LIBRARY if item["sig"] != target_item["sig"]], n_regions - 1)
        pool = [target_item, *distractors]
        random.shuffle(pool)
        if any(not item["is_rect"] for item in pool):
            break
    else:
        return None
    heights = [int(item["patch"].shape[0]) for item in pool]
    widths = [int(item["patch"].shape[1]) for item in pool]
    h = random.randint(max(12, max(heights) + 4), 20)
    w = random.randint(max(12, max(widths) + 4), 20)

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []
    for item in pool:
        loc = _place_patch(grid, blocked, item["patch"])
        if loc is None:
            return None
        r, c = loc
        placements.append({"item": item, "r": r, "c": c})

    seed_pick = next((idx for idx, item in enumerate(pool) if item["sig"] == target_item["sig"]), None)
    if seed_pick is None:
        return None
    target = placements[seed_pick]
    interior = np.argwhere(target["item"]["interior"])
    if int(interior.shape[0]) == 0:
        return None
    sr, sc = map(int, random.choice(interior.tolist()))
    grid[int(target["r"]) + sr, int(target["c"]) + sc] = RED

    try:
        regions, seed_region = _validate_input(grid)
    except AssertionError:
        return None

    out = main(grid)
    if len(regions) != n_regions or seed_region < 0:
        return None
    output_key = (int(n_regions), str(target_item["sig"]))
    if output_key in _SEEN_OUTPUT_KEYS:
        return None
    out_sig = _shape_signature(out)
    if out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_KEYS.add(output_key)
    _SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
