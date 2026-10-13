from __future__ import annotations

from collections import deque
from functools import lru_cache
import random

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _component_from_coords(coords: list[tuple[int, int]], color: int) -> dict:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": rel,
        "coord_set": set(rel),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "area": int(len(rel)),
    }


def _component_from_mask(mask: np.ndarray, color: int) -> dict:
    coords = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    assert coords
    return _component_from_coords(coords, color)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            components.append(_component_from_coords(coords, color))

    return components


def _all_rectangle_fits(comp_a: dict, comp_b: dict, bg: int) -> list[dict]:
    fits: list[dict] = []
    seen: set[tuple[int, int, bytes]] = set()

    ha, wa = int(comp_a["h"]), int(comp_a["w"])
    hb, wb = int(comp_b["h"]), int(comp_b["w"])
    total_area = int(comp_a["area"]) + int(comp_b["area"])
    aset = comp_a["coord_set"]

    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            min_r = min(0, dr)
            min_c = min(0, dc)
            max_r = max(ha - 1, dr + hb - 1)
            max_c = max(wa - 1, dc + wb - 1)
            out_h = int(max_r - min_r + 1)
            out_w = int(max_c - min_c + 1)
            if out_h * out_w != total_area:
                continue

            shifted_b: list[tuple[int, int]] = []
            overlap = False
            for r, c in comp_b["coords"]:
                cell = (int(r + dr), int(c + dc))
                if cell in aset:
                    overlap = True
                    break
                shifted_b.append(cell)
            if overlap:
                continue

            patch = np.full((out_h, out_w), int(bg), dtype=int)
            for r, c in comp_a["coords"]:
                patch[int(r - min_r), int(c - min_c)] = int(comp_a["color"])
            for r, c in shifted_b:
                patch[int(r - min_r), int(c - min_c)] = int(comp_b["color"])

            key = (out_h, out_w, patch.tobytes())
            if key in seen:
                continue
            seen.add(key)
            fits.append(
                {
                    "h": out_h,
                    "w": out_w,
                    "area": int(total_area),
                    "patch": patch,
                }
            )

    fits.sort(key=lambda fit: (int(fit["h"]), int(fit["w"]), fit["patch"].tobytes()))
    return fits


def _solve_matching(fits_by_pair: dict[tuple[int, int], dict], n: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    @lru_cache(maxsize=None)
    def rec(mask: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if mask == 0:
            return 1, ()

        i = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << i)
        total = 0
        best: tuple[tuple[int, int], ...] | None = None

        for j in range(i + 1, n):
            if not (rest & (1 << j)):
                continue
            pair = (i, j)
            if pair not in fits_by_pair:
                continue
            count, tail = rec(rest ^ (1 << j))
            if count == 0:
                continue
            if best is None:
                best = (pair,) + tail
            total = min(2, total + count)

        return total, (() if best is None else best)

    return rec((1 << n) - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered monochrome fragments by complementary shape, reconstruct each rectangle,
    sort the rectangles by area, and stack them vertically with one background row between.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)

    components = _find_components(g, bg)
    n = len(components)
    assert n > 0 and n % 2 == 0

    fits_by_pair: dict[tuple[int, int], dict] = {}
    for i in range(n):
        for j in range(i + 1, n):
            fits = _all_rectangle_fits(components[i], components[j], bg)
            if fits:
                fits_by_pair[(i, j)] = fits[0]

    match_count, matching = _solve_matching(fits_by_pair, n)
    assert match_count == 1, "fragments must have a unique complementary pairing"

    rectangles = [fits_by_pair[pair] for pair in matching]
    rectangles.sort(key=lambda fit: int(fit["area"]))

    out_h = sum(int(rect["h"]) for rect in rectangles) + len(rectangles) - 1
    out_w = max(int(rect["w"]) for rect in rectangles)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    r0 = 0
    for rect in rectangles:
        h, w = int(rect["h"]), int(rect["w"])
        out[r0 : r0 + h, :w] = rect["patch"]
        r0 += h + 1

    return out


BG = 0
COLOR_POOL = tuple(range(1, 10))
PAIR_COUNT_RANGE = (2, 4)
RECT_HEIGHT_RANGE = (2, 6)
RECT_WIDTH_RANGE = (2, 7)
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
MIN_PAIR_AREA = 6
MAX_PAIR_AREA = 24
PARTITION_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 40
GENERATION_ATTEMPTS = 300


def _cells_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _grow_partition(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    all_cells = {(r, c) for r in range(h) for c in range(w)}
    area = h * w
    lo = max(2, int(np.ceil(area * 0.25)))
    hi = min(area - 2, int(np.floor(area * 0.75)))
    if lo > hi:
        lo, hi = 2, area - 2
    if lo > hi:
        return None

    target = random.randint(lo, hi)
    start = random.choice(tuple(all_cells))
    cells_a = {start}
    frontier = {nbr for nbr in _neighbors4(start[0], start[1], h, w)}

    while len(cells_a) < target:
        choices = list(frontier - cells_a)
        random.shuffle(choices)
        picked = None
        for cell in choices:
            new_a = cells_a | {cell}
            if _cells_connected(all_cells - new_a):
                picked = cell
                cells_a = new_a
                frontier.update(_neighbors4(cell[0], cell[1], h, w))
                break
        if picked is None:
            return None

    cells_b = all_cells - cells_a
    if not _cells_connected(cells_a) or not _cells_connected(cells_b):
        return None

    mask_a = np.zeros((h, w), dtype=bool)
    mask_b = np.zeros((h, w), dtype=bool)
    for r, c in cells_a:
        mask_a[int(r), int(c)] = True
    for r, c in cells_b:
        mask_b[int(r), int(c)] = True
    return mask_a, mask_b


def _stair_partition(h: int, w: int, vertical: bool) -> tuple[np.ndarray, np.ndarray] | None:
    if vertical:
        cuts = [random.randint(1, w - 1)]
        for _ in range(1, h):
            nxt = cuts[-1] + random.choice((-1, 0, 1))
            cuts.append(max(1, min(w - 1, nxt)))
        mask_a = np.zeros((h, w), dtype=bool)
        for r, cut in enumerate(cuts):
            mask_a[r, :cut] = True
    else:
        cuts = [random.randint(1, h - 1)]
        for _ in range(1, w):
            nxt = cuts[-1] + random.choice((-1, 0, 1))
            cuts.append(max(1, min(h - 1, nxt)))
        mask_a = np.zeros((h, w), dtype=bool)
        for c, cut in enumerate(cuts):
            mask_a[:cut, c] = True

    if int(mask_a.sum()) < 2 or int((~mask_a).sum()) < 2:
        return None
    mask_b = ~mask_a
    if not _cells_connected({tuple(map(int, cell)) for cell in np.argwhere(mask_a)}):
        return None
    if not _cells_connected({tuple(map(int, cell)) for cell in np.argwhere(mask_b)}):
        return None
    return mask_a, mask_b


def _random_transform_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tagged = np.zeros(mask_a.shape, dtype=int)
    tagged[mask_a] = 1
    tagged[mask_b] = 2

    tagged = np.rot90(tagged, random.randrange(4))
    if random.random() < 0.5:
        tagged = np.fliplr(tagged)
    if random.random() < 0.5:
        tagged = np.flipud(tagged)

    return tagged == 1, tagged == 2


def _build_piece(mask: np.ndarray, color: int) -> dict:
    comp = _component_from_mask(mask, color)
    patch = np.full((int(comp["h"]), int(comp["w"])), BG, dtype=int)
    for r, c in comp["coords"]:
        patch[int(r), int(c)] = int(color)
    return {
        "component": comp,
        "mask": patch != BG,
        "patch": patch,
        "h": int(comp["h"]),
        "w": int(comp["w"]),
        "area": int(comp["area"]),
    }


def _expected_output(rectangles: list[dict]) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda rect: int(rect["area"]))
    out_h = sum(int(rect["h"]) for rect in ordered) + len(ordered) - 1
    out_w = max(int(rect["w"]) for rect in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    r0 = 0
    for rect in ordered:
        h, w = int(rect["h"]), int(rect["w"])
        out[r0 : r0 + h, :w] = rect["patch"]
        r0 += h + 1
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    return dilated


def _place_pieces(pieces: list[dict], h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    order = list(range(len(pieces)))
    random.shuffle(order)
    order.sort(key=lambda idx: (pieces[idx]["h"] * pieces[idx]["w"], pieces[idx]["area"]), reverse=True)

    for idx in order:
        piece = pieces[idx]
        ph, pw = int(piece["h"]), int(piece["w"])
        dilated = _dilate8(occupied)
        positions: list[tuple[int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                if bool((dilated[r0 : r0 + ph, c0 : c0 + pw] & piece["mask"]).any()):
                    continue
                positions.append((int(r0), int(c0)))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        view[piece["mask"]] = piece["patch"][piece["mask"]]
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= piece["mask"]

    return grid


def _matching_is_unique(components: list[dict]) -> bool:
    n = len(components)
    fits_by_pair: dict[tuple[int, int], dict] = {}
    degrees = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            fits = _all_rectangle_fits(components[i], components[j], BG)
            if not fits:
                continue
            fits_by_pair[(i, j)] = fits[0]
            degrees[i] += 1
            degrees[j] += 1
    if any(deg != 1 for deg in degrees):
        return False
    count, _matching = _solve_matching(fits_by_pair, n)
    return count == 1


def _sample_rectangle_dims(used_areas: set[int], pair_count: int) -> list[tuple[int, int]] | None:
    dims: list[tuple[int, int]] = []
    total_height = pair_count - 1
    for _ in range(pair_count):
        candidates = [
            (hh, ww)
            for hh in range(RECT_HEIGHT_RANGE[0], RECT_HEIGHT_RANGE[1] + 1)
            for ww in range(RECT_WIDTH_RANGE[0], RECT_WIDTH_RANGE[1] + 1)
            if MIN_PAIR_AREA <= hh * ww <= MAX_PAIR_AREA
            and hh * ww not in used_areas
            and total_height + hh <= 30
        ]
        if not candidates:
            return None
        h, w = random.choice(candidates)
        dims.append((int(h), int(w)))
        used_areas.add(int(h * w))
        total_height += int(h)
    return dims


def _build_pair(h: int, w: int, color_a: int, color_b: int) -> tuple[dict, dict, dict] | None:
    for _ in range(PARTITION_ATTEMPTS):
        mode = random.choice(("grow", "stair_v", "stair_h"))
        if mode == "grow":
            result = _grow_partition(h, w)
        elif mode == "stair_v":
            result = _stair_partition(h, w, vertical=True)
        else:
            result = _stair_partition(h, w, vertical=False)
        if result is None:
            continue

        mask_a, mask_b = result
        if random.random() < 0.5:
            mask_a, mask_b = mask_b, mask_a
        mask_a, mask_b = _random_transform_pair(mask_a, mask_b)

        comp_a = _component_from_mask(mask_a, color_a)
        comp_b = _component_from_mask(mask_b, color_b)
        fits = _all_rectangle_fits(comp_a, comp_b, BG)
        if len(fits) != 1:
            continue

        piece_a = _build_piece(mask_a, color_a)
        piece_b = _build_piece(mask_b, color_b)
        rect_patch = fits[0]["patch"]
        rectangle = {
            "h": int(fits[0]["h"]),
            "w": int(fits[0]["w"]),
            "area": int(fits[0]["area"]),
            "patch": rect_patch,
        }
        return piece_a, piece_b, rectangle
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    dims = _sample_rectangle_dims(set(), pair_count)
    if dims is None:
        return None

    colors = random.sample(COLOR_POOL, 2 * pair_count)
    pieces: list[dict] = []
    rectangles: list[dict] = []

    for i, (h, w) in enumerate(dims):
        built = _build_pair(h, w, colors[2 * i], colors[2 * i + 1])
        if built is None:
            return None
        piece_a, piece_b, rectangle = built
        pieces.extend([piece_a, piece_b])
        rectangles.append(rectangle)

    components = [piece["component"] for piece in pieces]
    if not _matching_is_unique(components):
        return None

    expected = _expected_output(rectangles)
    max_piece_h = max(piece["h"] for piece in pieces)
    max_piece_w = max(piece["w"] for piece in pieces)

    min_h = max(GRID_HEIGHT_RANGE[0], max_piece_h + 2)
    min_w = max(GRID_WIDTH_RANGE[0], max_piece_w + 2)
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        grid_h = random.randint(min_h, GRID_HEIGHT_RANGE[1])
        grid_w = random.randint(min_w, GRID_WIDTH_RANGE[1])
        grid = _place_pieces(pieces, grid_h, grid_w)
        if grid is None:
            continue
        if _bg_color(grid) != BG:
            continue
        if len(_find_components(grid, BG)) != len(pieces):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
