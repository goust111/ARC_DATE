from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
OBJ = 1
FILL_EVEN = 2
FILL_ODD = 5

MAX_DIM = 30
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return out


def _hole_components(crop: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = crop.shape
    holes: list[np.ndarray] = []
    for comp in _components(crop == bg, connectivity=4):
        rows = comp[:, 0]
        cols = comp[:, 1]
        if np.any(rows == 0) or np.any(rows == h - 1) or np.any(cols == 0) or np.any(cols == w - 1):
            continue
        holes.append(comp)
    return holes


def _fill_holes(crop: np.ndarray, bg: int) -> np.ndarray:
    out = crop.copy()
    for hole in _hole_components(crop, bg):
        by_col: dict[int, list[int]] = {}
        for r, c in hole:
            by_col.setdefault(int(c), []).append(int(r))
        for c, rows in by_col.items():
            top = min(rows)
            for r in rows:
                out[r, c] = FILL_EVEN if (r - top) % 2 == 0 else FILL_ODD
    return out


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert set(map(int, np.unique(g))) <= {bg, OBJ}

    objects = _components(g == OBJ, connectivity=8)
    assert 3 <= len(objects) <= 5

    infos = []
    for coords in objects:
        crop = _crop_component(g, coords, bg)
        holes = _hole_components(crop, bg)
        assert len(holes) == 1
        hole = holes[0]
        hr0, _hc0, hr1, _hc1 = _bbox_coords(hole)
        infos.append(
            {
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "crop": crop,
                "mask": crop == OBJ,
                "symmetric": bool(np.array_equal(crop == OBJ, np.fliplr(crop == OBJ))),
                "hole": hole,
                "hole_area": int(len(hole)),
                "hole_height": int(hr1 - hr0 + 1),
                "area": int(len(coords)),
                "width": int(crop.shape[1]),
                "height": int(crop.shape[0]),
            }
        )

    ordered = sorted(infos, key=lambda item: item["bbox"][0])
    for i in range(len(ordered) - 1):
        gap = ordered[i + 1]["bbox"][0] - ordered[i]["bbox"][2] - 1
        assert gap >= 2

    asymmetric = [info for info in infos if not info["symmetric"]]
    assert len(asymmetric) == 1
    target = asymmetric[0]
    assert target["hole_area"] >= 2
    assert target["hole_height"] >= 2
    for info in infos:
        assert info["hole_area"] >= 2

    target_width = target["width"]
    target_area = target["area"]
    has_decoy = any(
        info is not target and (info["width"] == target_width or info["area"] == target_area)
        for info in infos
    )
    assert has_decoy

    return {"bg": bg, "objects": infos, "target": target}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique left-right-asymmetric blue object and alternately fill its enclosed hole by column.

    Input structure:
    - A black background contains several separated blue 8-connected objects.
    - Each object has exactly one internal black hole inside its tight bounding box.
    - Exactly one object's foreground mask is not mirror-symmetric left-to-right.

    Rule:
    - Split the blue foreground into 8-connected objects and crop each one to its tight bbox.
    - Select the unique crop whose blue mask differs from its left-right reflection.
    - Find black 4-connected regions in that crop that do not touch the crop border.
    - For every hole column, start at the topmost hole cell in that column and fill hole cells downward with alternating red then gray parity by row offset within that column.

    Key insight:
    - The puzzle first uses asymmetry only to choose which sprite to crop; the actual output then depends on detecting the crop's enclosed hole and applying a column-local alternating fill pattern inside it.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    return _fill_holes(info["target"]["crop"], info["bg"])


def _tight_crop(mask: np.ndarray, value: int = 1) -> np.ndarray:
    coords = np.argwhere(mask == value)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_connected_mask(height: int, width: int, area_min: int, area_max: int) -> np.ndarray:
    target = random.randint(area_min, min(area_max, height * width))
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors(r, c, 4):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_symmetric_hole(inner_h: int, inner_w: int) -> np.ndarray:
    area_max = max(2, min(inner_h * inner_w, 6))
    for _ in range(200):
        half_w = (inner_w + 1) // 2
        seed = (random.randrange(inner_h), half_w - 1)
        half = np.zeros((inner_h, half_w), dtype=bool)
        half[seed] = True
        steps = random.randint(1, max(1, min(area_max, inner_h * inner_w) - 1))
        for _ in range(steps):
            frontier = set()
            for r, c in np.argwhere(half):
                for nr, nc in _neighbors(int(r), int(c), 4):
                    if 0 <= nr < inner_h and 0 <= nc < half_w and not half[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            half[random.choice(tuple(frontier))] = True

        full = np.zeros((inner_h, inner_w), dtype=bool)
        full[:, :half_w] |= half
        full[:, inner_w - half_w :] |= np.fliplr(half)
        if np.count_nonzero(full) < 2 or np.count_nonzero(full) > area_max:
            continue
        if len(_components(full, 4)) != 1:
            continue
        return full
    raise RuntimeError("failed to build symmetric hole")


def _random_asymmetric_hole(inner_h: int, inner_w: int, require_tall: bool) -> np.ndarray:
    area_max = max(2, min(inner_h * inner_w, 6))
    for _ in range(200):
        mask = _random_connected_mask(inner_h, inner_w, area_min=2, area_max=area_max)
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        coords = np.argwhere(mask)
        r0, _c0, r1, _c1 = _bbox_coords(coords)
        if require_tall and r1 - r0 + 1 < 2:
            continue
        return mask
    raise RuntimeError("failed to build asymmetric hole")


def _side_rows(height: int) -> list[int]:
    count = random.randint(1, min(2, height))
    rows = random.sample(range(height), count)
    rows.sort()
    return rows


def _make_object(symmetric: bool, allowed_heights: tuple[int, ...] = (4, 5)) -> np.ndarray:
    for _ in range(400):
        base_h = random.choice(allowed_heights)
        base_w = random.choice([4, 5, 6])
        inner_h = base_h - 2
        inner_w = base_w - 2
        if symmetric:
            hole = _random_symmetric_hole(inner_h, inner_w)
        else:
            hole_mode = random.choice(["hole", "tab"])
            hole = (
                _random_asymmetric_hole(inner_h, inner_w, require_tall=True)
                if hole_mode == "hole"
                else _random_symmetric_hole(inner_h, inner_w)
            )

        obj = np.ones((base_h, base_w), dtype=int)
        obj[1:-1, 1:-1][hole] = BG

        canvas = np.pad(obj, ((0, 0), (1, 1)), constant_values=BG)
        if symmetric:
            if random.random() < 0.75:
                for rr in _side_rows(base_h):
                    canvas[rr, 0] = OBJ
                    canvas[rr, -1] = OBJ
        else:
            if hole_mode == "tab" or random.random() < 0.55:
                side_col = 0 if random.random() < 0.5 else canvas.shape[1] - 1
                for rr in _side_rows(base_h):
                    canvas[rr, side_col] = OBJ
            elif random.random() < 0.35:
                for rr in _side_rows(base_h):
                    canvas[rr, 0] = OBJ
                    canvas[rr, -1] = OBJ

        cropped = _tight_crop(canvas, value=OBJ)
        mask = cropped == OBJ
        if len(_hole_components(cropped, BG)) != 1:
            continue
        is_sym = np.array_equal(mask, np.fliplr(mask))
        if is_sym != symmetric:
            continue
        hole_coords = _hole_components(cropped, BG)[0]
        hr0, _hc0, hr1, _hc1 = _bbox_coords(hole_coords)
        if len(hole_coords) < 2:
            continue
        if not symmetric and hr1 - hr0 + 1 < 2:
            continue
        return cropped
    raise RuntimeError("failed to build object")


def _place_objects(objects: list[np.ndarray]) -> np.ndarray:
    heights = [obj.shape[0] for obj in objects]
    widths = [obj.shape[1] for obj in objects]
    gap = 2
    base_h = sum(heights) + gap * (len(objects) - 1)
    grid_h = random.randint(base_h, min(MAX_DIM, base_h + 2))
    grid_w = random.randint(max(widths) + 3, min(MAX_DIM, max(widths) + 12))
    grid = np.zeros((grid_h, grid_w), dtype=int)

    extra_rows = grid_h - base_h
    prefix_gaps = [0] * (len(objects) + 1)
    for _ in range(extra_rows):
        prefix_gaps[random.randrange(len(prefix_gaps))] += 1

    row = prefix_gaps[0]
    for idx, obj in enumerate(objects):
        max_c = grid_w - obj.shape[1] - 1
        col = random.randint(1, max_c)
        grid[row : row + obj.shape[0], col : col + obj.shape[1]] = obj
        row += obj.shape[0]
        if idx < len(objects) - 1:
            row += gap + prefix_gaps[idx + 1]
    return grid


def _attempt_generate() -> np.ndarray | None:
    count = random.randint(3, 5)
    asym_index = random.randrange(count)
    allowed_heights = (4,) if count >= 4 else (4, 5)

    objects: list[np.ndarray] = []
    for i in range(count):
        objects.append(_make_object(symmetric=(i != asym_index), allowed_heights=allowed_heights))

    base_h = sum(obj.shape[0] for obj in objects) + 2 * (count - 1)
    if base_h > MAX_DIM:
        return None

    random.shuffle(objects)
    grid = _place_objects(objects)

    try:
        info = _analyze(grid)
    except AssertionError:
        return None

    target = info["target"]
    target_out = _fill_holes(target["crop"], info["bg"])
    if target_out.shape[0] > MAX_DIM or target_out.shape[1] > MAX_DIM:
        return None
    if np.count_nonzero(target_out == FILL_EVEN) == 0 or np.count_nonzero(target_out == FILL_ODD) == 0:
        return None
    if not np.array_equal(main(grid), target_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
