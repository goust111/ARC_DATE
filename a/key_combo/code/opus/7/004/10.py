from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BG = 0
TEMPLATE = 1
FRAME = 2
OBSTACLE = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_PATTERN_CACHE: dict[tuple[int, int], list[dict]] = {}


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = DIR4
    else:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _frame_components(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in components_mask(grid == FRAME, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        expected = 2 * (h + w) - 4
        if len(coords) != expected:
            continue
        if not np.all(patch[0, :] == FRAME):
            continue
        if not np.all(patch[-1, :] == FRAME):
            continue
        if not np.all(patch[:, 0] == FRAME):
            continue
        if not np.all(patch[:, -1] == FRAME):
            continue
        inner = patch[1:-1, 1:-1]
        if np.any(inner == FRAME):
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner": inner.copy(),
                "inner_shape": inner.shape,
            }
        )
    frames.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return frames


def _largest_zero_rect(inner: np.ndarray) -> tuple[int, int]:
    h, w = inner.shape
    blocked = (inner != BG).astype(int)
    pref = np.pad(blocked, ((1, 0), (1, 0)))
    pref = pref.cumsum(axis=0).cumsum(axis=1)
    best_area = -1
    best_count = 0
    for r0 in range(h):
        for r1 in range(r0 + 1, h):
            height = r1 - r0 + 1
            if height < 2:
                continue
            for c0 in range(w):
                for c1 in range(c0 + 1, w):
                    width = c1 - c0 + 1
                    if width < 2:
                        continue
                    total = pref[r1 + 1, c1 + 1] - pref[r0, c1 + 1] - pref[r1 + 1, c0] + pref[r0, c0]
                    if total != 0:
                        continue
                    area = height * width
                    if area > best_area:
                        best_area = area
                        best_count = 1
                    elif area == best_area:
                        best_count += 1
    if best_area < 0:
        raise AssertionError("expected at least one 2x2 all-background rectangle")
    return best_area, best_count


def _extract_template(grid: np.ndarray) -> np.ndarray:
    comps = components_mask(grid == TEMPLATE, connectivity=4)
    assert len(comps) == 1, "expected exactly one blue template component"
    coords = comps[0]
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = TEMPLATE
    return patch


def _build_output(template: np.ndarray, areas: list[int]) -> np.ndarray:
    tile_h, tile_w = template.shape
    rows = []
    max_w = 0
    for area in areas:
        row_w = area * tile_w + (area - 1)
        row = np.zeros((tile_h, row_w), dtype=int)
        for k in range(area):
            c0 = k * (tile_w + 1)
            mask = template != BG
            row[:, c0 : c0 + tile_w][mask] = TEMPLATE
        rows.append(row)
        max_w = max(max_w, row_w)

    out_h = len(rows) * tile_h + (len(rows) - 1)
    out = np.zeros((out_h, max_w), dtype=int)
    for i, row in enumerate(rows):
        r0 = i * (tile_h + 1)
        out[r0 : r0 + tile_h, : row.shape[1]] = row
    return out


def _spec(grid: np.ndarray) -> tuple[list[dict], np.ndarray, list[int], np.ndarray]:
    frames = _frame_components(grid)
    assert 2 <= len(frames) <= 3, "expected 2 or 3 valid frames"
    areas = []
    for frame in frames:
        area, count = _largest_zero_rect(frame["inner"])
        assert count == 1, "largest empty rectangle must be unique"
        areas.append(area)
    assert len(set(areas)) == len(areas), "frame areas must be distinct"
    template = _extract_template(grid)
    output = _build_output(template, areas)
    return frames, template, areas, output


def main(grid: np.ndarray) -> np.ndarray:
    """
    Search each hollow frame for its largest empty inner rectangle, then repeat the bottom template by that area.

    Input structure:
    - Black background contains 2 or 3 red one-cell-thick hollow rectangles and one blue template sprite near the bottom.
    - Each red frame encloses a black interior with 1 to 3 isolated grey obstacle pixels.
    - The blue template is a single 4-connected monochrome sprite whose tight bbox is at most 3x3.

    Rule:
    - Identify all red hollow rectangles and sort them by their bbox top-left column, then row.
    - For each frame interior, search all fully black subrectangles with height and width at least 2, and take the unique maximum-area one.
    - Crop the blue template to its tight bbox.
    - Produce one output row per sorted frame by repeating the template horizontally exactly that maximum area many times, with one black column between copies.
    - Stack those rows top to bottom with one black separator row between them.

    Key insight:
    - The grey obstacle layout inside each frame implicitly encodes a repetition count via the unique largest empty rectangle area, so the solver must perform an actual rectangle search rather than reading frame size or obstacle count directly.
    """
    g = np.asarray(grid, dtype=int)
    _frames, _template, _areas, out = _spec(g)
    return out


def _random_template() -> np.ndarray:
    target_cells = random.randint(3, 5)
    cells = {(random.randint(0, 2), random.randint(0, 2))}
    while len(cells) < target_cells:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = TEMPLATE
    return patch


def _isolated_pattern(pattern: tuple[tuple[int, int], ...]) -> bool:
    for i, (r0, c0) in enumerate(pattern):
        for r1, c1 in pattern[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _pattern_catalog(inner_h: int, inner_w: int) -> list[dict]:
    key = (inner_h, inner_w)
    if key in _PATTERN_CACHE:
        return _PATTERN_CACHE[key]

    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    catalog = []
    for k in (1, 2, 3):
        for comb in combinations(cells, k):
            if not _isolated_pattern(comb):
                continue
            inner = np.zeros((inner_h, inner_w), dtype=int)
            for r, c in comb:
                inner[r, c] = OBSTACLE
            area, count = _largest_zero_rect(inner)
            if count != 1:
                continue
            catalog.append(
                {
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "obstacles": tuple(sorted(comb)),
                    "obstacle_count": k,
                    "area": area,
                }
            )
    _PATTERN_CACHE[key] = catalog
    return catalog


def _choose_specs(n_frames: int, max_area: int) -> list[dict] | None:
    size_choices = [(h, w) for h in range(4, 7) for w in range(5, 8)]
    same_size_pairs = []
    all_specs = []
    for inner_h, inner_w in size_choices:
        specs = [x for x in _pattern_catalog(inner_h, inner_w) if 4 <= x["area"] <= max_area]
        if not specs:
            continue
        all_specs.extend(specs)
        by_count: dict[int, list[dict]] = {}
        for spec in specs:
            by_count.setdefault(spec["obstacle_count"], []).append(spec)
        for variants in by_count.values():
            by_area: dict[int, list[dict]] = {}
            for spec in variants:
                by_area.setdefault(spec["area"], []).append(spec)
            areas = sorted(by_area)
            if len(areas) >= 2:
                a0, a1 = random.sample(areas, 2)
                same_size_pairs.append((random.choice(by_area[a0]), random.choice(by_area[a1])))

    if not same_size_pairs or len(all_specs) < n_frames:
        return None

    random.shuffle(same_size_pairs)
    random.shuffle(all_specs)
    for first, second in same_size_pairs:
        chosen = [first, second]
        used_areas = {first["area"], second["area"]}
        if n_frames == 2:
            random.shuffle(chosen)
            return chosen
        candidates = [spec for spec in all_specs if spec["area"] not in used_areas]
        random.shuffle(candidates)
        for third in candidates:
            trio = chosen + [third]
            interior_areas = [spec["inner_h"] * spec["inner_w"] for spec in trio]
            rect_areas = [spec["area"] for spec in trio]
            if max(interior_areas) == min(interior_areas):
                continue
            if max(rect_areas) == min(rect_areas):
                continue
            if max(interior_areas) == interior_areas[np.argmax(rect_areas)] and min(interior_areas) == interior_areas[np.argmin(rect_areas)]:
                continue
            random.shuffle(trio)
            return trio
    return None


def _place_rectangles(h: int, w: int, sizes: list[tuple[int, int]], reserve_top: int) -> list[tuple[int, int]] | None:
    occ = np.zeros((h, w), dtype=bool)
    positions = []
    order = list(range(len(sizes)))
    order.sort(key=lambda idx: -(sizes[idx][0] * sizes[idx][1]))
    for idx in order:
        ph, pw = sizes[idx]
        candidates = []
        for r0 in range(1, reserve_top - ph):
            for c0 in range(1, w - pw):
                rr0 = max(0, r0 - 2)
                cc0 = max(0, c0 - 2)
                rr1 = min(h, r0 + ph + 2)
                cc1 = min(w, c0 + pw + 2)
                if occ[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions.append((idx, r0, c0))
        occ[r0 : r0 + ph, c0 : c0 + pw] = True
    pos_by_idx = [None] * len(sizes)
    for idx, r0, c0 in positions:
        pos_by_idx[idx] = (r0, c0)
    return pos_by_idx  # type: ignore[return-value]


def _paint_frame(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    inner_h = spec["inner_h"]
    inner_w = spec["inner_w"]
    h = inner_h + 2
    w = inner_w + 2
    grid[top, left : left + w] = FRAME
    grid[top + h - 1, left : left + w] = FRAME
    grid[top : top + h, left] = FRAME
    grid[top : top + h, left + w - 1] = FRAME
    for r, c in spec["obstacles"]:
        grid[top + 1 + r, left + 1 + c] = OBSTACLE


def _semantic_checks(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    frames, template, areas, output = _spec(grid)
    assert max(output.shape) <= 30
    assert 3 <= int((template == TEMPLATE).sum()) <= 5
    assert template.shape[0] <= 3 and template.shape[1] <= 3

    blue_coords = np.argwhere(grid == TEMPLATE)
    assert len(blue_coords) > 0
    assert blue_coords[:, 0].max() >= grid.shape[0] - 4, "template must sit near the bottom"

    specs = []
    for frame in frames:
        inner = frame["inner"]
        obstacle_coords = np.argwhere(inner == OBSTACLE)
        assert 1 <= len(obstacle_coords) <= 3
        for i, (r0, c0) in enumerate(obstacle_coords):
            for r1, c1 in obstacle_coords[i + 1 :]:
                assert max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1))) > 1
        area, count = _largest_zero_rect(inner)
        assert count == 1
        specs.append(
            {
                "inner_shape": inner.shape,
                "obstacle_count": len(obstacle_coords),
                "area": area,
            }
        )

    same_size_counterexample = False
    same_count_counterexample = False
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if specs[i]["inner_shape"] == specs[j]["inner_shape"] and specs[i]["area"] != specs[j]["area"]:
                same_size_counterexample = True
            if specs[i]["obstacle_count"] == specs[j]["obstacle_count"] and specs[i]["area"] != specs[j]["area"]:
                same_count_counterexample = True
    assert same_size_counterexample or same_count_counterexample

    actual = main(grid)
    assert np.array_equal(actual, output)


def _attempt() -> np.ndarray | None:
    template = _random_template()
    tile_h, tile_w = template.shape
    max_repeat = (31) // (tile_w + 1)
    max_repeat = min(max_repeat, 10)
    if max_repeat < 4:
        return None

    n_frames = random.choice([2, 3])
    specs = _choose_specs(n_frames, max_repeat)
    if specs is None:
        return None

    frame_sizes = [(spec["inner_h"] + 2, spec["inner_w"] + 2) for spec in specs]
    grid_h = random.randint(18, 24)
    grid_w = random.randint(20, 28)

    template_top = random.randint(grid_h - tile_h - 2, grid_h - tile_h - 1)
    reserve_top = template_top - 1
    if reserve_top < 9:
        return None

    positions = _place_rectangles(grid_h, grid_w, frame_sizes, reserve_top)
    if positions is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for (top, left), spec in zip(positions, specs):
        _paint_frame(grid, top, left, spec)

    template_left = random.randint(1, grid_w - tile_w - 2)
    grid[template_top : template_top + tile_h, template_left : template_left + tile_w][template != 0] = TEMPLATE

    try:
        _semantic_checks(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
