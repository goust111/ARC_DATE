from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
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


def _tab_count(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> int:
    pts = set(cells)
    total = 0
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        if degree == 1:
            total += 1
    return total


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _rotate_flip(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _base_shape() -> np.ndarray:
    kind = random.choice(["rect", "corner_cut", "step"])
    mask = np.zeros((8, 8), dtype=bool)
    if kind == "rect":
        h, w = random.choice([(2, 4), (2, 5), (3, 3), (3, 4), (4, 3)])
        r0 = random.randint(1, 7 - h)
        c0 = random.randint(1, 7 - w)
        mask[r0 : r0 + h, c0 : c0 + w] = True
    elif kind == "corner_cut":
        h, w = random.choice([(4, 4), (4, 5), (5, 4)])
        r0 = random.randint(1, 7 - h)
        c0 = random.randint(1, 7 - w)
        mask[r0 : r0 + h, c0 : c0 + w] = True
        cut_r = 0 if random.random() < 0.5 else h - 2
        cut_c = 0 if random.random() < 0.5 else w - 2
        mask[r0 + cut_r : r0 + cut_r + 2, c0 + cut_c : c0 + cut_c + 2] = False
    else:
        mask[1:4, 1:4] = True
        mask[3:6, 2:5] = True
    mask = _rotate_flip(mask)
    return _trim_mask(mask)


def _tab_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    padded = np.pad(mask, 1)
    for r in range(1, h + 1):
        for c in range(1, w + 1):
            if padded[r, c]:
                continue
            degree = sum(padded[r + dr, c + dc] for dr, dc in DIRS4)
            if degree == 1:
                candidates.append((r - 1, c - 1))
    return candidates


def _embed_for_tabs(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    canvas = np.zeros((h + 4, w + 4), dtype=bool)
    canvas[2 : 2 + h, 2 : 2 + w] = mask
    return canvas


def _make_object(desired_tabs: int) -> np.ndarray | None:
    for _ in range(120):
        core = _base_shape()
        if _tab_count(set(map(tuple, np.argwhere(core)))) != 0:
            continue
        canvas = _embed_for_tabs(core)
        candidates = _tab_candidates(canvas)
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for r, c in candidates:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == desired_tabs:
                    break
        if len(chosen) != desired_tabs:
            continue
        for r, c in chosen:
            canvas[r, c] = True
        trimmed = _trim_mask(canvas)
        cells = list(map(tuple, np.argwhere(trimmed)))
        area = len(cells)
        if not (8 <= area <= 20):
            continue
        if len(_components(trimmed)) != 1:
            continue
        if _tab_count(cells) != desired_tabs:
            continue
        return trimmed
    return None


def _analysis(grid: np.ndarray):
    object_comps = _components(grid == GREY)
    neutral_comps = _components(grid == ORANGE)
    labels = np.full(grid.shape, -1, dtype=int)
    tab_counts: list[int] = []
    for idx, comp in enumerate(object_comps):
        for r, c in comp:
            labels[r, c] = idx
        tab_counts.append(_tab_count(comp))
    touches: list[set[int]] = []
    for comp in neutral_comps:
        adjacent = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and labels[nr, nc] >= 0:
                    adjacent.add(int(labels[nr, nc]))
        touches.append(adjacent)
    return object_comps, neutral_comps, labels, tab_counts, touches


def main(grid: np.ndarray) -> np.ndarray:
    """Rank grey rectilinear objects by boundary-tab count, recolor the extremes, and propagate those colors into adjacent neutral masks.

    Input structure:
    - The grid uses black background, grey 4-connected foreground objects, and orange 4-connected neutral mask components.
    - There are several separated thick rectilinear grey objects, each with a distinct count of one-neighbor boundary tabs.
    - Every orange component is edge-adjacent to exactly one grey object.

    Rule:
    - For each grey object, count grey pixels that have exactly one grey 4-neighbor.
    - Recolor the object with the largest count blue and the object with the smallest count red; other grey objects stay grey.
    - For each orange neutral component, find the unique adjacent object and recolor the whole component to that object's post-ranking color.
    - Keep the background and all unrelated pixels unchanged.

    Key insight:
    - The visible tabs are not repaired locally; they are counted globally to rank objects, and that ranking determines the colors that later spread into neighboring neutral regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    object_comps, neutral_comps, labels, tab_counts, neutral_touches = _analysis(g)
    assert 3 <= len(object_comps) <= 5
    assert len(set(tab_counts)) == len(tab_counts)
    assert neutral_comps

    min_idx = min(range(len(tab_counts)), key=tab_counts.__getitem__)
    max_idx = max(range(len(tab_counts)), key=tab_counts.__getitem__)

    obj_colors: dict[int, int] = {}
    out = g.copy()
    for idx, comp in enumerate(object_comps):
        color = GREY
        if idx == min_idx:
            color = RED
        if idx == max_idx:
            color = BLUE
        obj_colors[idx] = color
        for r, c in comp:
            out[r, c] = color

    touched_objects = set()
    for comp, adjacent in zip(neutral_comps, neutral_touches):
        assert len(adjacent) == 1
        obj_idx = next(iter(adjacent))
        touched_objects.add(obj_idx)
        color = obj_colors[obj_idx]
        for r, c in comp:
            out[r, c] = color

    assert len(touched_objects) == len(object_comps)
    return out


def _within_pad(r: int, c: int, box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(shape[0] - 1, r1 + pad)
    cc1 = min(shape[1] - 1, c1 + pad)
    return rr0 <= r <= rr1 and cc0 <= c <= cc1


def _grow_neutral_region(
    sector_shape: tuple[int, int],
    object_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = sector_shape
    box = _bbox(object_cells)
    seeds = []
    for r, c in object_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in object_cells and (nr, nc) not in forbidden:
                if _within_pad(nr, nc, box, pad=3, shape=sector_shape):
                    seeds.append((nr, nc))
    random.shuffle(seeds)
    for seed in seeds:
        target_size = random.randint(2, 6)
        comp = {seed}
        frontier = set()
        while len(comp) < target_size:
            frontier.clear()
            for r, c in comp:
                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in comp or (nr, nc) in object_cells or (nr, nc) in forbidden:
                        continue
                    if not _within_pad(nr, nc, box, pad=3, shape=sector_shape):
                        continue
                    frontier.add((nr, nc))
            if not frontier:
                break
            comp.add(random.choice(tuple(frontier)))
        if len(comp) == target_size:
            return comp
    return None


def _sector_layout(count: int) -> tuple[int, int]:
    if count == 5:
        return 2, 3
    if count == 4:
        return 2, 2
    return random.choice([(2, 2), (1, 3)])


def _render_sector(
    grid: np.ndarray,
    sector_r0: int,
    sector_c0: int,
    sector_h: int,
    sector_w: int,
    obj_mask: np.ndarray,
) -> bool:
    obj_h, obj_w = obj_mask.shape
    if obj_h > sector_h - 2 or obj_w > sector_w - 2:
        return False
    top = random.randint(1, sector_h - obj_h - 1)
    left = random.randint(1, sector_w - obj_w - 1)

    object_cells_local = {
        (top + r, left + c)
        for r in range(obj_h)
        for c in range(obj_w)
        if obj_mask[r, c]
    }
    if not object_cells_local:
        return False

    regions: list[set[tuple[int, int]]] = []
    forbidden = set(object_cells_local)
    desired_regions = 1 + int(random.random() < 0.35)
    for region_idx in range(desired_regions):
        region = _grow_neutral_region((sector_h, sector_w), object_cells_local, forbidden)
        if region is None:
            if region_idx == 0:
                return False
            break
        regions.append(region)
        halo = set(region)
        for r, c in region:
            halo.update((nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < sector_h and 0 <= nc < sector_w)
        forbidden.update(halo)

    if not regions:
        return False

    for r, c in object_cells_local:
        grid[sector_r0 + r, sector_c0 + c] = GREY
    for region in regions:
        for r, c in region:
            grid[sector_r0 + r, sector_c0 + c] = ORANGE
    return True


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    rows, cols = _sector_layout(count)
    gutter = random.randint(1, 2)
    sector_h = random.randint(8, 10)
    sector_w = random.randint(7, 8)
    h = rows * sector_h + (rows + 1) * gutter
    w = cols * sector_w + (cols + 1) * gutter
    if max(h, w) > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    counts = random.sample([1, 2, 3, 4, 5], count)
    slot_ids = list(range(rows * cols))
    random.shuffle(slot_ids)

    for desired_tabs, slot in zip(counts, slot_ids[:count]):
        obj_mask = _make_object(desired_tabs)
        if obj_mask is None:
            return None
        sr = slot // cols
        sc = slot % cols
        sector_r0 = gutter + sr * (sector_h + gutter)
        sector_c0 = gutter + sc * (sector_w + gutter)
        if not _render_sector(grid, sector_r0, sector_c0, sector_h, sector_w, obj_mask):
            return None

    object_comps, neutral_comps, _labels, tab_counts, neutral_touches = _analysis(grid)
    if len(object_comps) != count or len(neutral_comps) < count:
        return None
    if len(set(tab_counts)) != count:
        return None
    if any(len(adjacent) != 1 for adjacent in neutral_touches):
        return None
    touched = set()
    for adjacent in neutral_touches:
        touched.update(adjacent)
    if len(touched) != count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    colors = set(map(int, out.ravel()))
    if BLUE not in colors or RED not in colors or GREY not in colors:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
