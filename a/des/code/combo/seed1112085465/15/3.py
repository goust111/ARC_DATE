from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 8
PALETTE = [c for c in range(10) if c != BG]
MAX_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    max_count = int(counts.max())
    winners = [int(v) for v, cnt in zip(values, counts) if int(cnt) == max_count]
    if BG in winners:
        return BG
    return int(winners[0])


def _find_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == bg:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _crop_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not bool(mask[nr, nc]):
                    out[r, c] = True
                    break
    return out


def _analyze_object(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    colors = [int(grid[int(r), int(c)]) for r, c in coords]
    counts = Counter(colors)
    max_count = max(counts.values())
    main_colors = [color for color, cnt in counts.items() if cnt == max_count]
    assert len(main_colors) == 1, "expected a unique main color"
    main_color = int(main_colors[0])

    connector_cells = [(int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) != main_color]
    assert len(connector_cells) == 1, "expected exactly one connector pixel"
    connector_r, connector_c = connector_cells[0]
    connector_color = int(grid[connector_r, connector_c])

    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    rel_r = connector_r - r0
    rel_c = connector_c - c0

    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    assert bool(_boundary_mask(mask)[rel_r, rel_c]), "connector pixel must lie on the object boundary"

    return {
        "main_color": main_color,
        "connector_color": connector_color,
        "patch": _crop_component(grid, coords, bg),
    }


def _pack_patches(patches: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = max(int(p.shape[0]) for p in patches)
    out_w = sum(int(p.shape[1]) for p in patches) + max(0, len(patches) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for patch in patches:
        h, w = map(int, patch.shape)
        out[:h, col : col + w] = patch
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the connector-color chain and lay the referenced objects out in one row.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    components = _find_components(g, bg)
    assert len(components) >= 2, "expected multiple foreground objects"

    objects = [_analyze_object(g, coords, bg) for coords in components]

    main_to_index: dict[int, int] = {}
    for idx, obj in enumerate(objects):
        color = int(obj["main_color"])
        assert color not in main_to_index, "main colors must be unique"
        main_to_index[color] = idx

    next_index: list[int | None] = []
    indegree = [0] * len(objects)
    tail_count = 0
    for obj in objects:
        nxt = main_to_index.get(int(obj["connector_color"]))
        next_index.append(nxt)
        if nxt is None:
            tail_count += 1
        else:
            indegree[nxt] += 1

    heads = [idx for idx, deg in enumerate(indegree) if deg == 0]
    assert len(heads) == 1, "expected a unique chain head"
    assert tail_count == 1, "expected a unique chain tail"

    order: list[int] = []
    seen: set[int] = set()
    cur: int | None = heads[0]
    while cur is not None:
        assert cur not in seen, "chain must be acyclic"
        seen.add(cur)
        order.append(cur)
        cur = next_index[cur]

    assert len(order) == len(objects), "chain must include every object"
    patches = [np.asarray(objects[idx]["patch"], dtype=int) for idx in order]
    return _pack_patches(patches, bg)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _grow_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
        nr, nc = random.choice(list(frontier))
        mask[nr, nc] = True

    return _crop_mask(mask)


def _sample_shape_mask() -> np.ndarray:
    mode = random.choice(("grow", "grow", "rect", "notch"))

    if mode == "rect":
        h = random.randint(2, 4)
        w = random.randint(2, 5)
        return np.ones((h, w), dtype=bool)

    if mode == "notch":
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        mask = np.ones((h, w), dtype=bool)
        notch_h = random.randint(1, min(2, h - 1))
        notch_w = random.randint(1, min(2, w - 1))
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            mask[:notch_h, :notch_w] = False
        elif corner == "tr":
            mask[:notch_h, w - notch_w :] = False
        elif corner == "bl":
            mask[h - notch_h :, :notch_w] = False
        else:
            mask[h - notch_h :, w - notch_w :] = False
        return _random_transform(mask)

    h = random.randint(2, 5)
    w = random.randint(2, 5)
    min_area = max(4, min(h * w, max(h, w)))
    area = random.randint(min_area, min(h * w, 12))
    return _random_transform(_grow_mask(h, w, area))


def _make_object_patch(main_color: int, connector_color: int) -> np.ndarray:
    mask = _sample_shape_mask()
    boundary = np.argwhere(_boundary_mask(mask))
    connector_r, connector_c = map(int, random.choice(boundary.tolist()))
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = main_color
    patch[connector_r, connector_c] = connector_color
    return patch


def _can_place(mask: np.ndarray, occupied: set[tuple[int, int]], top: int, left: int) -> bool:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            rr = top + r
            cc = left + c
            for nr in range(rr - 1, rr + 2):
                for nc in range(cc - 1, cc + 2):
                    if (nr, nc) in occupied:
                        return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, occupied: set[tuple[int, int]], top: int, left: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            value = int(patch[r, c])
            if value == BG:
                continue
            rr = top + r
            cc = left + c
            grid[rr, cc] = value
            occupied.add((rr, cc))


def _place_objects(patches: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()

    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: int(np.count_nonzero(patches[idx] != BG)), reverse=True)

    for idx in order:
        patch = patches[idx]
        h, w = map(int, patch.shape)
        row_range = range(1, max(2, height - h))
        col_range = range(1, max(2, width - w))
        candidates = [(r, c) for r in row_range for c in col_range]
        random.shuffle(candidates)

        placed = False
        mask = patch != BG
        for top, left in candidates:
            if _can_place(mask, occupied, top, left):
                _place_patch(grid, patch, occupied, top, left)
                placed = True
                break
        if not placed:
            return None

    return grid


def _background_is_dominant(grid: np.ndarray) -> bool:
    bg_pixels = int(np.count_nonzero(grid == BG))
    fg_pixels = int(grid.size - bg_pixels)
    return bg_pixels > fg_pixels


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(3, 6)
    colors = random.sample(PALETTE, n_objects + 1)
    main_colors = colors[:n_objects]
    tail_color = colors[-1]

    chain_patches = []
    for idx, main_color in enumerate(main_colors):
        connector_color = main_colors[idx + 1] if idx + 1 < n_objects else tail_color
        chain_patches.append(_make_object_patch(main_color, connector_color))

    max_h = max(int(patch.shape[0]) for patch in chain_patches)
    max_w = max(int(patch.shape[1]) for patch in chain_patches)
    total_fg = sum(int(np.count_nonzero(patch != BG)) for patch in chain_patches)

    min_h = max(12, max_h + 4)
    min_w = max(12, max_w + 4)
    slack = random.randint(4, 10)
    height_hi = min(28, min_h + slack)
    if min_h > height_hi:
        return None
    height = random.randint(min_h, height_hi)
    width_floor = max(min_w, int(np.ceil((total_fg * 3) / max(1, height - 2))) + 4)
    width_hi = min(28, width_floor + random.randint(4, 10))
    if width_floor > width_hi:
        return None
    width = random.randint(width_floor, width_hi)

    placement_order = list(range(n_objects))
    random.shuffle(placement_order)
    input_patches = [chain_patches[idx] for idx in placement_order]
    inp = _place_objects(input_patches, height, width)
    if inp is None or not _background_is_dominant(inp):
        return None

    if len(_find_components(inp, BG)) != n_objects:
        return None

    expected = _pack_patches(chain_patches, BG)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-layout ARC input")
