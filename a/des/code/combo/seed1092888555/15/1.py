import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_from_coords(coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "area": len(coords),
    }


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(_component_from_coords(coords))
    return components


def _placements_of_subset(model_mask: np.ndarray, key_mask: np.ndarray) -> list[tuple[int, int]]:
    mh, mw = model_mask.shape
    kh, kw = key_mask.shape
    placements = []
    for r0 in range(mh - kh + 1):
        for c0 in range(mw - kw + 1):
            window = model_mask[r0:r0 + kh, c0:c0 + kw]
            if np.all(window[key_mask]):
                placements.append((r0, c0))
    return placements


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = _find_components_of_color(g, color)
        assert len(comps) == 2, "each non-background color must form exactly two components"
        comps.sort(key=lambda comp: comp["area"])
        key, model = comps[0], comps[1]
        assert key["area"] < model["area"], "key must be strictly smaller than model"

        matches = _placements_of_subset(model["mask"], key["mask"])
        assert len(matches) == 1, "key shape must match the model in a unique location"
        off_r, off_c = matches[0]
        mr0, mc0, _mr1, _mc1 = model["bbox"]

        for rr in range(key["mask"].shape[0]):
            for cc in range(key["mask"].shape[1]):
                if key["mask"][rr, cc]:
                    out[mr0 + off_r + rr, mc0 + off_c + cc] = bg

        for r, c in key["coords"]:
            out[r, c] = bg

    return out


GRID_H_RANGE = (14, 25)
GRID_W_RANGE = (14, 25)
PAIR_COUNT_OPTIONS = [1, 2, 3, 4]
GEN_ATTEMPTS = 500


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in list(cells):
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            break
        pick = random.choice(frontier)
        mask[pick] = True
        cells.add(pick)
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0:r1 + 1, c0:c1 + 1].copy()


def _sample_connected_subset(model_mask: np.ndarray, size: int) -> np.ndarray | None:
    cells = [tuple(int(x) for x in p) for p in np.argwhere(model_mask)]
    if len(cells) < size:
        return None
    seed = random.choice(cells)
    chosen = {seed}
    while len(chosen) < size:
        frontier = []
        for r, c in chosen:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < model_mask.shape[0] and 0 <= nc < model_mask.shape[1] and model_mask[nr, nc] and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            return None
        chosen.add(random.choice(frontier))
    sub = np.zeros_like(model_mask, dtype=bool)
    for r, c in chosen:
        sub[r, c] = True
    return _normalize_mask(sub)


def _build_pair_masks() -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    for _ in range(200):
        h = random.randint(4, 8)
        w = random.randint(4, 8)
        area = random.randint(max(7, (h * w) // 3), min(h * w - 1, h * w - max(1, (h * w) // 5)))
        model_mask = _grow_connected_mask(h, w, area)
        model_mask = _normalize_mask(model_mask)
        model_area = int(model_mask.sum())
        if model_area < 7:
            continue

        size_hi = min(7, model_area - 2)
        if size_hi < 2:
            continue
        sizes = list(range(2, size_hi + 1))
        random.shuffle(sizes)
        for size in sizes:
            for _ in range(80):
                key_mask = _sample_connected_subset(model_mask, size)
                if key_mask is None:
                    continue
                placements = _placements_of_subset(model_mask, key_mask)
                if len(placements) != 1:
                    continue
                offset = placements[0]
                carved = model_mask.copy()
                kh, kw = key_mask.shape
                carved[offset[0]:offset[0] + kh, offset[1]:offset[1] + kw][key_mask] = False
                if carved.sum() == 0:
                    continue
                return model_mask, key_mask, offset
    return None


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, gap: int = 1) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(occupied.shape[0], r0 + h + gap)
    cc1 = min(occupied.shape[1], c0 + w + gap)
    region = occupied[rr0:rr1, cc0:cc1]
    if region.any():
        return False
    return True


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int):
    h, w = mask.shape
    patch = grid[r0:r0 + h, c0:c0 + w]
    patch[mask] = color


def _place_mask(occupied: np.ndarray, grid: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = mask.shape
    positions = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if _can_place(occupied, mask, r0, c0, gap=1):
            _stamp_mask(grid, mask, r0, c0, color)
            occupied[r0:r0 + h, c0:c0 + w] |= mask
            return True
    return False


def _expected_from_specs(shape: tuple[int, int], bg: int, specs: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in specs:
        model_mask = spec["model_mask"].copy()
        kr, kc = spec["offset"]
        key_mask = spec["key_mask"]
        kh, kw = key_mask.shape
        model_mask[kr:kr + kh, kc:kc + kw][key_mask] = False
        _stamp_mask(out, model_mask, spec["model_pos"][0], spec["model_pos"][1], spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)

    max_pairs = min(len(palette), random.choice(PAIR_COUNT_OPTIONS))
    pair_count = max_pairs
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    specs = []

    for color in palette[:pair_count]:
        built = _build_pair_masks()
        if built is None:
            return None
        model_mask, key_mask, offset = built

        if model_mask.shape[0] > h - 2 or model_mask.shape[1] > w - 2:
            return None
        if key_mask.shape[0] > h - 2 or key_mask.shape[1] > w - 2:
            return None

        if not _place_mask(occupied, grid, model_mask, color):
            return None
        model_components = _find_components_of_color(grid, color)
        model_pos = model_components[0]["bbox"][:2]

        if not _place_mask(occupied, grid, key_mask, color):
            return None
        comps = _find_components_of_color(grid, color)
        if len(comps) != 2:
            return None
        areas = sorted(comp["area"] for comp in comps)
        if areas[0] != int(key_mask.sum()) or areas[1] != int(model_mask.sum()):
            return None

        key_pos = min((comp["bbox"][:2] for comp in comps if comp["area"] == int(key_mask.sum())), default=None)
        if key_pos is None:
            return None

        specs.append(
            {
                "color": color,
                "model_mask": model_mask,
                "key_mask": key_mask,
                "offset": offset,
                "model_pos": model_pos,
                "key_pos": key_pos,
            }
        )

    expected = _expected_from_specs((h, w), bg, specs)

    try:
        found = main(grid)
    except Exception:
        return None
    if np.array_equal(found, grid):
        return None
    if not np.array_equal(found, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
