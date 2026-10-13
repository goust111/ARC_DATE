from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PAYLOAD_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_MAX_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _solid_cross(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    solid_rows = [
        (r, int(g[r, 0]))
        for r in range(h)
        if int(g[r, 0]) != BG and np.all(g[r, :] == int(g[r, 0]))
    ]
    solid_cols = [
        (c, int(g[0, c]))
        for c in range(w)
        if int(g[0, c]) != BG and np.all(g[:, c] == int(g[0, c]))
    ]
    matches = [
        (r, c, color)
        for r, color in solid_rows
        for c, col_color in solid_cols
        if color == col_color
    ]
    assert len(matches) == 1, "expected exactly one solid separator cross"
    return matches[0]


def _quadrants(grid: np.ndarray, cross_r: int, cross_c: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    parts = [
        g[:cross_r, :cross_c],
        g[:cross_r, cross_c + 1 :],
        g[cross_r + 1 :, :cross_c],
        g[cross_r + 1 :, cross_c + 1 :],
    ]
    shapes = {part.shape for part in parts}
    assert len(shapes) == 1, "expected four equal quadrants"
    shape = parts[0].shape
    assert shape[0] > 0 and shape[1] > 0 and shape[0] == shape[1], "expected nonempty square quadrants"
    return [part.copy() for part in parts]


def _candidate_components(region: np.ndarray, separator: int) -> list[dict]:
    mask = (region != BG) & (region != separator)
    comps = _components(mask)
    assert comps, "expected at least one foreground component per quadrant"

    candidates = []
    for coords in comps:
        canvas = np.zeros_like(region)
        canvas[coords[:, 0], coords[:, 1]] = region[coords[:, 0], coords[:, 1]]
        box = _bbox(canvas != BG)
        assert box is not None
        candidates.append(
            {
                "canvas": canvas,
                "coords": coords,
                "bbox": box,
                "area": int(coords.shape[0]),
            }
        )

    candidates.sort(key=lambda item: (-item["area"], item["bbox"]))
    return candidates


def _solve_details(grid: np.ndarray) -> dict:
    cross_r, cross_c, separator = _solid_cross(grid)
    quads = _quadrants(grid, cross_r, cross_c)
    candidate_lists = [_candidate_components(quad, separator) for quad in quads]

    keyed = []
    for candidates in candidate_lists:
        mapping = {}
        for item in candidates:
            mapping[item["canvas"].tobytes()] = item
        keyed.append(mapping)

    solutions = []
    for ul in candidate_lists[0]:
        expected_ur = np.rot90(ul["canvas"], -1)
        expected_ll = np.rot90(ul["canvas"], 1)
        expected_lr = np.rot90(ul["canvas"], 2)
        ur = keyed[1].get(expected_ur.tobytes())
        ll = keyed[2].get(expected_ll.tobytes())
        lr = keyed[3].get(expected_lr.tobytes())
        if ur is None or ll is None or lr is None:
            continue
        solutions.append({"ul": ul, "ur": ur, "ll": ll, "lr": lr})

    assert solutions, "expected at least one rotationally consistent component choice"
    max_area = max(item["ul"]["area"] for item in solutions)
    best = [item for item in solutions if item["ul"]["area"] == max_area]
    assert len(best) == 1, "expected a unique maximal rotationally consistent solution"
    return {
        "cross_r": cross_r,
        "cross_c": cross_c,
        "separator": separator,
        "quadrants": quads,
        "candidate_lists": candidate_lists,
        "solutions": solutions,
        "best": best[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Remove quadrant noise with rotational consistency and return the cropped source tile.

    Input structure:
    - A black background grid is split by one solid separator row and one solid separator column into four equal square quadrants.
    - Each quadrant contains several non-black, non-separator 4-connected components: one true multicolor pattern and some detached noise components.
    - The four true patterns form a rotational collage: upper-left is the source, upper-right is its clockwise rotation, lower-right is its 180-degree rotation, and lower-left is its counterclockwise rotation.

    Rule:
    - Find the unique solid cross to locate the four quadrants.
    - Within each quadrant, enumerate every 4-connected foreground component as a possible denoised pattern.
    - Select the unique maximal set of four components that satisfies the quadrant rotation relation, then crop the upper-left selected component to its bounding box.

    Key insight:
    - Local size alone is not reliable because a noise component can tie or beat the real pattern; the correct denoising is the one that makes all four quadrants agree under quarter-turn rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    details = _solve_details(g)
    source = details["best"]["ul"]["canvas"]
    box = _bbox(source != BG)
    assert box is not None
    return _crop_bbox(source, box)


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    for _ in range(120):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        box = _bbox(mask)
        if box != (0, 0, h - 1, w - 1):
            continue
        return mask
    return None


def _random_colored_tile(h: int, w: int) -> np.ndarray | None:
    area_min = max(4, (h * w) // 2)
    area_max = h * w - 1
    if area_min > area_max:
        area_min = area_max
    for _ in range(120):
        area = random.randint(area_min, area_max)
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue

        coords = np.argwhere(mask)
        palette_size = random.randint(2, min(4, area))
        palette = random.sample(PAYLOAD_COLORS, palette_size)
        values = palette[:] + [random.choice(palette) for _ in range(area - palette_size)]
        random.shuffle(values)

        tile = np.zeros((h, w), dtype=int)
        tile[coords[:, 0], coords[:, 1]] = values
        if h == w and np.array_equal(tile, np.rot90(tile, -1)):
            continue
        return tile
    return None


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    out = canvas.copy()
    h, w = patch.shape
    out[r0 : r0 + h, c0 : c0 + w] = patch
    return out


def _can_place_component(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    qh, qw = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > qh or c0 + w > qw:
        return False

    cells = set()
    for r in range(h):
        for c in range(w):
            if patch[r, c] != BG:
                rr, cc = r0 + r, c0 + c
                if canvas[rr, cc] != BG:
                    return False
                cells.add((rr, cc))

    for rr, cc in cells:
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < qh and 0 <= nc < qw and canvas[nr, nc] != BG and (nr, nc) not in cells:
                return False
    return True


def _sample_noise_patch(area: int, colors: list[int]) -> np.ndarray | None:
    dims = [(1, area)] + [(area, 1)]
    for h in range(2, min(4, area) + 1):
        for w in range(2, min(4, area) + 1):
            if h * w >= area:
                dims.append((h, w))
    random.shuffle(dims)

    for h, w in dims:
        if area > h * w:
            continue
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        patch = np.zeros((h, w), dtype=int)
        values = [random.choice(colors) for _ in range(coords.shape[0])]
        patch[coords[:, 0], coords[:, 1]] = values
        return patch
    return None


def _add_noise_component(
    canvas: np.ndarray,
    area: int,
    colors: list[int],
    forbidden_rotations: list[np.ndarray],
) -> np.ndarray | None:
    for _ in range(120):
        patch = _sample_noise_patch(area, colors)
        if patch is None:
            continue
        h, w = patch.shape
        q = canvas.shape[0]
        candidates = [
            (r0, c0)
            for r0 in range(q - h + 1)
            for c0 in range(q - w + 1)
            if _can_place_component(canvas, patch, r0, c0)
        ]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            placed = _place_patch(canvas, patch, r0, c0)
            if any(np.array_equal(placed, target) for target in forbidden_rotations):
                continue
            return placed
    return None


def _needs_global_disambiguation(details: dict) -> bool:
    best = details["best"]
    for name, idx in (("ul", 0), ("ur", 1), ("ll", 2), ("lr", 3)):
        chosen = best[name]
        candidates = details["candidate_lists"][idx]
        max_area = max(item["area"] for item in candidates)
        max_count = sum(item["area"] == max_area for item in candidates)
        if chosen["area"] < max_area or max_count > 1:
            return True
    return False


def _quadrant_palette(tile: np.ndarray, share_colors: bool) -> list[int]:
    tile_colors = sorted(set(int(v) for v in tile.flat if int(v) != BG))
    if share_colors and tile_colors:
        return tile_colors
    extra = [color for color in PAYLOAD_COLORS if color not in tile_colors]
    pool = tile_colors + extra
    return pool if pool else PAYLOAD_COLORS


def _build_case() -> tuple[np.ndarray, np.ndarray, bool] | None:
    q = random.randint(6, 8)
    tile_h = random.randint(3, min(4, q - 1))
    tile_w = random.randint(3, min(4, q - 1))
    tile = _random_colored_tile(tile_h, tile_w)
    if tile is None:
        return None

    source = np.zeros((q, q), dtype=int)
    top = random.randint(0, q - tile_h)
    left = random.randint(0, q - tile_w)
    source[top : top + tile_h, left : left + tile_w] = tile
    if np.array_equal(source, np.rot90(source, -1)):
        return None

    clean_quads = [
        source,
        np.rot90(source, -1),
        np.rot90(source, 1),
        np.rot90(source, 2),
    ]
    quadrants = [quad.copy() for quad in clean_quads]

    mode = random.choices(
        ["plain", "tie", "oversize"],
        weights=[2, 3, 3],
        k=1,
    )[0]
    share_colors = random.random() < 0.5
    tricky = mode != "plain"

    pattern_area = int(np.count_nonzero(tile))
    special_idx = random.randrange(4) if tricky else -1
    forbidden_rotations = [
        np.rot90(source, 0),
        np.rot90(source, -1),
        np.rot90(source, 1),
        np.rot90(source, 2),
    ]

    for idx in range(4):
        palette = _quadrant_palette(tile, share_colors and random.random() < 0.7)
        canvas = quadrants[idx]
        if idx == special_idx:
            target_area = pattern_area if mode == "tie" else pattern_area + 1
            canvas = _add_noise_component(canvas, target_area, palette, forbidden_rotations)
            if canvas is None:
                return None
        small_noise_count = random.randint(1, 3)
        for _ in range(small_noise_count):
            area = random.randint(1, min(3, max(1, pattern_area - 1)))
            updated = _add_noise_component(canvas, area, palette, forbidden_rotations)
            if updated is None:
                continue
            canvas = updated
        quadrants[idx] = canvas

    grid = np.zeros((2 * q + 1, 2 * q + 1), dtype=int)
    grid[q, :] = SEP
    grid[:, q] = SEP
    grid[:q, :q] = quadrants[0]
    grid[:q, q + 1 :] = quadrants[1]
    grid[q + 1 :, :q] = quadrants[2]
    grid[q + 1 :, q + 1 :] = quadrants[3]

    expected = tile.copy()
    return grid, expected, tricky


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, expected, tricky = built
        try:
            details = _solve_details(grid)
            if tricky and not _needs_global_disambiguation(details):
                continue
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, expected):
            return grid
    raise RuntimeError("failed to generate a valid noisy rotational-quadrant input")
