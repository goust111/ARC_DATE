import random
from collections import deque
from itertools import combinations

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TRANSFORMS = ("identity", "hflip", "vflip", "rot90")
CANONICAL_L = frozenset({(0, 0), (1, 0), (1, 1)})


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _transform_patch(mask: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return mask.copy()
    if transform == "hflip":
        return np.fliplr(mask)
    if transform == "vflip":
        return np.flipud(mask)
    if transform == "rot90":
        return np.rot90(mask, -1)
    raise ValueError(f"unknown transform: {transform}")


def _orientation_map() -> dict[frozenset[tuple[int, int]], str]:
    base = np.zeros((2, 2), dtype=bool)
    for r, c in CANONICAL_L:
        base[r, c] = True
    mapping = {}
    for transform in TRANSFORMS:
        rotated = _transform_patch(base, transform)
        mapping[frozenset(map(tuple, np.argwhere(rotated)))] = transform
    return mapping


L_ORIENTATION_TO_TRANSFORM = _orientation_map()


def _is_band_row(row: np.ndarray) -> bool:
    vals = np.unique(row)
    if len(vals) != 2:
        return False
    return bool(np.all(row[:-1] != row[1:]))


def _pair_key(values: np.ndarray | list[int]) -> tuple[int, int]:
    uniq = sorted({int(v) for v in values})
    assert len(uniq) == 2
    return uniq[0], uniq[1]


def _components_non_bg(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _components_same_color(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _classify_legend_component(
    legend: np.ndarray, coords: list[tuple[int, int]]
) -> tuple[str, object, tuple[int, int]] | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    center = (r0 + r1, c0 + c1)
    if (h, w) == (2, 2):
        rel = frozenset((r - r0, c - c0) for r, c in coords)
        colors = {int(legend[r, c]) for r, c in coords}
        if len(colors) == 1 and rel in L_ORIENTATION_TO_TRANSFORM:
            return "marker", L_ORIENTATION_TO_TRANSFORM[rel], center
        return None
    if (h, w) == (1, 3):
        colors = [int(legend[r0, c]) for c in range(c0, c1 + 1)]
        if len(set(colors)) == 2 and colors[0] == colors[2] and colors[0] != colors[1]:
            return "bar", _pair_key(colors), center
    return None


def _parse_legend(legend: np.ndarray) -> dict[tuple[int, int], str]:
    components = _components_non_bg(legend)
    markers = []
    bars = []
    for coords in components:
        classified = _classify_legend_component(legend, coords)
        if classified is None:
            continue
        kind, payload, center = classified
        if kind == "marker":
            markers.append({"transform": payload, "center": center})
        else:
            bars.append({"key": payload, "center": center})

    assert len(markers) == 4 and len(bars) == 4, "expected four legend markers and four legend bars"
    markers.sort(key=lambda item: (item["center"][1], item["center"][0]))
    bars.sort(key=lambda item: (item["center"][1], item["center"][0]))

    mapping: dict[tuple[int, int], str] = {}
    for marker, bar in zip(markers, bars):
        mapping[bar["key"]] = marker["transform"]
    assert len(mapping) == 4, "legend band keys must be unique"
    return mapping


def _transform_object(coords: list[tuple[int, int]], transform: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    patch = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True

    transformed = _transform_patch(patch, transform)
    new_h, new_w = transformed.shape
    shift_r = (h - new_h) // 2
    shift_c = (w - new_w) // 2
    return [(r0 + shift_r + int(r), c0 + shift_c + int(c)) for r, c in np.argwhere(transformed)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, _ = g.shape
    band_rows = [r for r in range(h) if _is_band_row(g[r])]
    assert len(band_rows) >= 1, "no instruction bands found"

    legend = g[: band_rows[0], :]
    mapping = _parse_legend(legend)

    out = g.copy()
    for idx, band_r in enumerate(band_rows):
        region_r0 = band_r + 1
        region_r1 = band_rows[idx + 1] - 1 if idx + 1 < len(band_rows) else h - 1
        if region_r0 > region_r1:
            continue
        key = _pair_key(g[band_r])
        assert key in mapping, "band key missing from legend"
        transform = mapping[key]

        region = g[region_r0 : region_r1 + 1, :]
        region_out = np.full_like(region, BG)
        for obj in _components_same_color(region):
            color = int(obj["color"])
            new_coords = _transform_object(obj["coords"], transform)
            for rr, cc in new_coords:
                assert 0 <= rr < region.shape[0] and 0 <= cc < region.shape[1], "object moved out of region"
                assert region_out[rr, cc] in {BG, color}, "objects overlap after transform"
                region_out[rr, cc] = color
        out[region_r0 : region_r1 + 1, :] = region_out

    return out


LEGEND_HEIGHT_RANGE = (4, 6)
REGION_HEIGHT_RANGE = (6, 10)
WIDTH_RANGE = (29, 37)
REGION_COUNT = 4
ITEM_WIDTH = 6
ITEM_GAP_MIN = 1
ITEM_MARGIN = 1
OBJECT_COUNT_RANGE = (1, 4)
MAX_ATTEMPTS = 300
ALL_NON_BG = list(range(1, 10))


def _split_slack(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _alternating_sequence(pair: tuple[int, int], length: int) -> list[int]:
    start = random.randrange(2)
    return [pair[(start + i) % 2] for i in range(length)]


def _legend_item_starts(width: int) -> list[int]:
    base = 2 * ITEM_MARGIN + REGION_COUNT * ITEM_WIDTH + (REGION_COUNT - 1) * ITEM_GAP_MIN
    assert width >= base
    extras = _split_slack(width - base, REGION_COUNT + 1)
    starts = []
    cur = ITEM_MARGIN + extras[0]
    for idx in range(REGION_COUNT):
        starts.append(cur)
        cur += ITEM_WIDTH
        if idx + 1 < REGION_COUNT:
            cur += ITEM_GAP_MIN + extras[idx + 1]
    return starts


def _draw_legend(
    grid: np.ndarray,
    legend_items: list[tuple[tuple[int, int], str]],
    marker_color: int,
    legend_height: int,
) -> None:
    starts = _legend_item_starts(grid.shape[1])
    for start, (pair, transform) in zip(starts, legend_items):
        top = random.randint(0, legend_height - 2)
        bar_row = top + random.randint(0, 1)

        marker = np.zeros((2, 2), dtype=bool)
        for r, c in CANONICAL_L:
            marker[r, c] = True
        marker = _transform_patch(marker, transform)
        for r, c in np.argwhere(marker):
            grid[top + int(r), start + int(c)] = marker_color

        bar_vals = _alternating_sequence(pair, 3)
        for offset, color in enumerate(bar_vals):
            grid[bar_row, start + 3 + offset] = color


def _draw_band_row(grid: np.ndarray, row: int, pair: tuple[int, int]) -> None:
    grid[row, :] = np.array(_alternating_sequence(pair, grid.shape[1]), dtype=int)


def _connect_points(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        moves = []
        if r < er:
            moves.append((1, 0))
        elif r > er:
            moves.append((-1, 0))
        if c < ec:
            moves.append((0, 1))
        elif c > ec:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        mask[r, c] = True


def _random_square_mask() -> np.ndarray:
    for _ in range(200):
        size = random.choice([2, 3, 4])
        if size == 2:
            mask = np.array([[1, 1], [1, 0]], dtype=bool)
        else:
            mask = np.zeros((size, size), dtype=bool)
            side_points = [
                (0, random.randrange(size)),
                (size - 1, random.randrange(size)),
                (random.randrange(size), 0),
                (random.randrange(size), size - 1),
            ]
            order = side_points[:]
            random.shuffle(order)
            for a, b in zip(order, order[1:]):
                _connect_points(mask, a, b)

            lo = int(mask.sum())
            hi = min(size * size - 1, lo + size)
            if lo > hi:
                continue
            target = random.randint(lo, hi)
            while int(mask.sum()) < target:
                frontier = set()
                for r, c in np.argwhere(mask):
                    for dr, dc in DIRS4:
                        nr, nc = int(r) + dr, int(c) + dc
                        if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                            frontier.add((nr, nc))
                if not frontier:
                    break
                rr, cc = random.choice(tuple(frontier))
                mask[rr, cc] = True

        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != mask.shape:
            continue
        keys = {_mask_key(_transform_patch(mask, transform)) for transform in TRANSFORMS}
        if len(keys) != 4:
            continue
        oriented = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            oriented = np.fliplr(oriented)
        return oriented.astype(bool).copy()
    raise RuntimeError("failed to build an asymmetric square mask")


def _bbox_mask(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> tuple[slice, slice]:
    return (
        slice(max(0, r0 - pad), min(h, r0 + mask.shape[0] + pad)),
        slice(max(0, c0 - pad), min(w, c0 + mask.shape[1] + pad)),
    )


def _place_region_objects(region_h: int, width: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(80):
        region = np.zeros((region_h, width), dtype=int)
        blocked = np.zeros((region_h, width), dtype=bool)
        count = random.randint(*OBJECT_COUNT_RANGE)
        chosen_colors = random.sample(colors, count) if count <= len(colors) else [random.choice(colors) for _ in range(count)]
        placed = 0
        order = chosen_colors[:]
        random.shuffle(order)
        for color in order:
            mask = _random_square_mask()
            size = int(mask.shape[0])
            margin = 1 if region_h - size >= 2 else 0
            candidates = []
            for r0 in range(margin, region_h - size - margin + 1):
                for c0 in range(margin, width - size - margin + 1):
                    rs, cs = _bbox_mask(mask, r0, c0, region_h, width, pad=1)
                    if blocked[rs, cs].any():
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                break
            r0, c0 = random.choice(candidates)
            for rr, cc in np.argwhere(mask):
                region[r0 + int(rr), c0 + int(cc)] = color
            rs, cs = _bbox_mask(mask, r0, c0, region_h, width, pad=1)
            blocked[rs, cs] = True
            placed += 1
        if placed == count:
            return region
    return None


def _random_mapping() -> tuple[dict[tuple[int, int], str], list[tuple[tuple[int, int], str]], int]:
    palette_size = random.choice([5, 6])
    band_palette = random.sample(ALL_NON_BG, palette_size)
    pair_choices = list(combinations(band_palette, 2))
    chosen_pairs = random.sample(pair_choices, REGION_COUNT)
    transforms = list(TRANSFORMS)
    random.shuffle(transforms)
    mapping = {tuple(sorted(pair)): transform for pair, transform in zip(chosen_pairs, transforms)}
    remaining = [color for color in ALL_NON_BG if color not in band_palette]
    marker_color = random.choice(remaining) if remaining else random.choice(ALL_NON_BG)
    legend_items = list(mapping.items())
    random.shuffle(legend_items)
    return mapping, legend_items, marker_color


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*WIDTH_RANGE)
    legend_height = random.randint(*LEGEND_HEIGHT_RANGE)
    region_heights = [random.randint(*REGION_HEIGHT_RANGE) for _ in range(REGION_COUNT)]
    total_height = legend_height + REGION_COUNT + sum(region_heights)

    mapping, legend_items, marker_color = _random_mapping()
    band_items = list(mapping.items())
    random.shuffle(band_items)
    object_colors = [color for color in ALL_NON_BG if color != marker_color]

    grid = np.zeros((total_height, width), dtype=int)
    _draw_legend(grid, legend_items, marker_color, legend_height)

    row = legend_height
    for idx, (pair, _transform) in enumerate(band_items):
        _draw_band_row(grid, row, pair)
        row += 1
        region_h = region_heights[idx]
        region = _place_region_objects(region_h, width, object_colors)
        if region is None:
            return None
        grid[row : row + region_h, :] = region
        row += region_h

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
