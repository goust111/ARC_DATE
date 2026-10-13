from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
SEED = 1
WALL = 5
RECT_COLORS = (2, 4, 7, 9)

GRID_MIN = 18
GRID_MAX = 24
RECT_COUNT_CHOICES = (2, 3)
RECT_SIZE_CHOICES = (5, 6, 7)
MAX_GENERATION_RETRIES = 400
MAX_PLACEMENT_RETRIES = 200


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def peel_layers(patch: np.ndarray) -> list[dict]:
    h, w = patch.shape
    layers: list[dict] = []
    inset = 0
    while True:
        r0, c0 = inset, inset
        r1, c1 = h - 1 - inset, w - 1 - inset
        if r0 > r1 or c0 > c1:
            break
        sub = patch[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros_like(sub, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        border_values = np.unique(sub[border_mask])
        assert len(border_values) == 1
        color = int(border_values[0])
        layers.append({"bbox_local": (r0, c0, r1, c1), "color": color})
        if np.all(sub == color):
            break
        inset += 1
    assert len(layers) >= 2
    return layers


def find_rectangles(grid: np.ndarray) -> list[dict]:
    rect_mask = ~np.isin(grid, [BG, SEED, WALL])
    rects: list[dict] = []
    for coords in component_coords(rect_mask):
        r0, c0, r1, c1 = bbox_coords(coords)
        assert rect_mask[r0 : r1 + 1, c0 : c1 + 1].all()
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        layers = peel_layers(patch)
        colors = [layer["color"] for layer in layers]
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "layers": [
                    {
                        "bbox": (
                            r0 + lr0,
                            c0 + lc0,
                            r0 + lr1,
                            c0 + lc1,
                        ),
                        "color": layer["color"],
                    }
                    for layer in layers
                    for (lr0, lc0, lr1, lc1) in [layer["bbox_local"]]
                ],
                "colors": colors,
            }
        )
    rects.sort(key=lambda rect: rect["bbox"])
    assert rects
    return rects


def flood_reachable(grid: np.ndarray) -> np.ndarray:
    starts = [tuple(map(int, rc)) for rc in np.argwhere(grid == SEED)]
    assert starts
    reachable = np.zeros_like(grid, dtype=bool)
    q = deque()
    for r, c in starts:
        reachable[r, c] = True
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc) or reachable[nr, nc]:
                continue
            if int(grid[nr, nc]) != BG:
                continue
            reachable[nr, nc] = True
            q.append((nr, nc))
    return reachable


def rectangle_touched(rect: dict, reachable: np.ndarray) -> bool:
    r0, c0, r1, c1 = rect["bbox"]
    for c in range(c0, c1 + 1):
        if _touches_reachable(r0, c, reachable) or _touches_reachable(r1, c, reachable):
            return True
    for r in range(r0 + 1, r1):
        if _touches_reachable(r, c0, reachable) or _touches_reachable(r, c1, reachable):
            return True
    return False


def _touches_reachable(r: int, c: int, reachable: np.ndarray) -> bool:
    for nr, nc in neighbors4(r, c):
        if in_bounds(reachable.shape, nr, nc) and reachable[nr, nc]:
            return True
    return False


def solve_with_details(grid: np.ndarray) -> tuple[np.ndarray, list[dict], set[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects = find_rectangles(g)
    reachable = flood_reachable(g)
    touched = {i for i, rect in enumerate(rects) if rectangle_touched(rect, reachable)}

    out = g.copy()
    for i, rect in enumerate(rects):
        if i not in touched:
            continue
        reversed_colors = list(reversed(rect["colors"]))
        for layer, new_color in zip(rect["layers"], reversed_colors):
            r0, c0, r1, c1 = layer["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = new_color
    out[out == SEED] = BG
    return out, rects, touched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the color order only for nested rectangles touched by seed-reachable flood regions.

    Input structure:
    - The background is black, blue cells are flood seeds, and grey cells are wall segments.
    - The foreground contains 2-3 separated solid rectangles made of 2-3 nested colored layers.
    - Rectangle pixels block movement, and the black background may be split into multiple 4-connected regions by walls.

    Rule:
    - Start from every blue seed and flood only through 4-connected black cells, treating grey walls and rectangle cells as blockers.
    - For each nested rectangle, test whether any outer-layer pixel is 4-adjacent to a seed or flood-reached black cell.
    - If touched, reverse that rectangle's layer-color order from outermost to innermost; otherwise leave it unchanged.
    - Erase all blue cells in the final output so the flood remains only as a hidden classification step.

    Key insight:
    - The visible change is not the flood itself but which rectangles get their nested color sequence reversed after the temporary reachability test.
    """
    out, _rects, _touched = solve_with_details(grid)
    return out


def fill_nested_rectangle(grid: np.ndarray, bbox: tuple[int, int, int, int], colors: list[int]) -> None:
    r0, c0, r1, c1 = bbox
    for inset, color in enumerate(colors):
        grid[r0 + inset : r1 - inset + 1, c0 + inset : c1 - inset + 1] = color


def draw_box_wall(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    gap: tuple[str, int, int] | None = None,
) -> None:
    r0, c0, r1, c1 = bbox
    for c in range(c0, c1 + 1):
        if not _is_gap_cell("top", r0, c, gap):
            grid[r0, c] = WALL
        if not _is_gap_cell("bottom", r1, c, gap):
            grid[r1, c] = WALL
    for r in range(r0, r1 + 1):
        if not _is_gap_cell("left", r, c0, gap):
            grid[r, c0] = WALL
        if not _is_gap_cell("right", r, c1, gap):
            grid[r, c1] = WALL


def _is_gap_cell(side: str, r: int, c: int, gap: tuple[str, int, int] | None) -> bool:
    if gap is None or gap[0] != side:
        return False
    return gap[1] <= (c if side in ("top", "bottom") else r) <= gap[2]


def black_component_labels(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = grid == BG
    labels = np.zeros(grid.shape, dtype=int)
    comps: list[np.ndarray] = []
    next_id = 1
    for coords in component_coords(mask):
        labels[coords[:, 0], coords[:, 1]] = next_id
        comps.append(coords)
        next_id += 1
    return labels, comps


def adjacent_black_components(labels: np.ndarray, rect: dict) -> set[int]:
    r0, c0, r1, c1 = rect["bbox"]
    ids: set[int] = set()
    for c in range(c0, c1 + 1):
        for rr, cc in ((r0, c), (r1, c)):
            for nr, nc in neighbors4(rr, cc):
                if in_bounds(labels.shape, nr, nc) and labels[nr, nc] > 0:
                    ids.add(int(labels[nr, nc]))
    for r in range(r0 + 1, r1):
        for rr, cc in ((r, c0), (r, c1)):
            for nr, nc in neighbors4(rr, cc):
                if in_bounds(labels.shape, nr, nc) and labels[nr, nc] > 0:
                    ids.add(int(labels[nr, nc]))
    return ids


def choose_seed_cell(grid: np.ndarray, coords: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    candidates = []
    for r, c in coords:
        degree = 0
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BG:
                degree += 1
        if degree >= 1:
            candidates.append((int(r), int(c), degree))
    if not candidates:
        return None
    max_degree = max(deg for _r, _c, deg in candidates)
    best = [(r, c) for r, c, deg in candidates if deg == max_degree]
    return best[int(rng.integers(len(best)))]


def add_random_segments(grid: np.ndarray, rects: list[dict], rng: np.random.Generator) -> None:
    protected = np.zeros_like(grid, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        pr0 = max(0, r0 - 2)
        pc0 = max(0, c0 - 2)
        pr1 = min(grid.shape[0] - 1, r1 + 2)
        pc1 = min(grid.shape[1] - 1, c1 + 2)
        protected[pr0 : pr1 + 1, pc0 : pc1 + 1] = True

    for _ in range(int(rng.integers(0, 4))):
        vertical = bool(rng.integers(2))
        length = int(rng.integers(2, 6))
        placed = False
        for _ in range(40):
            if vertical:
                r0 = int(rng.integers(0, grid.shape[0] - length + 1))
                c0 = int(rng.integers(0, grid.shape[1]))
                cells = [(r0 + k, c0) for k in range(length)]
            else:
                r0 = int(rng.integers(0, grid.shape[0]))
                c0 = int(rng.integers(0, grid.shape[1] - length + 1))
                cells = [(r0, c0 + k) for k in range(length)]
            if any(protected[r, c] or int(grid[r, c]) != BG for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = WALL
            placed = True
            break
        if not placed:
            continue


def reverse_all_rectangles(grid: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = grid.copy()
    for rect in rects:
        reversed_colors = list(reversed(rect["colors"]))
        for layer, new_color in zip(rect["layers"], reversed_colors):
            r0, c0, r1, c1 = layer["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = new_color
    out[out == SEED] = BG
    return out


def validate_generated_case(grid: np.ndarray, expected_touched: set[int]) -> None:
    out, rects, touched = solve_with_details(grid)
    assert touched == expected_touched
    assert 0 < len(touched) < len(rects)
    assert np.any(grid == SEED)
    assert np.array_equal(out[out != BG], main(grid)[out != BG])

    changed_rects = set()
    for i, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        if not np.array_equal(grid[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]):
            changed_rects.add(i)
    assert changed_rects == touched

    none_out = grid.copy()
    none_out[none_out == SEED] = BG
    all_out = reverse_all_rectangles(grid, rects)
    assert not np.array_equal(out, none_out)
    assert not np.array_equal(out, all_out)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-flood ARC input")


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.zeros((h, w), dtype=int)

    rect_count = int(rng.choice(RECT_COUNT_CHOICES))
    specs = []
    for _ in range(rect_count):
        layers = int(rng.choice((2, 3), p=(0.45, 0.55)))
        hh = int(rng.choice(RECT_SIZE_CHOICES))
        ww = int(rng.choice(RECT_SIZE_CHOICES))
        if min(hh, ww) < 2 * layers - 1:
            return None
        colors = list(map(int, rng.choice(RECT_COLORS, size=layers, replace=False)))
        specs.append({"h": hh, "w": ww, "layers": layers, "colors": colors})

    placed: list[dict] = []
    for spec in specs:
        success = False
        for _ in range(MAX_PLACEMENT_RETRIES):
            r0 = int(rng.integers(2, h - spec["h"] - 1))
            c0 = int(rng.integers(2, w - spec["w"] - 1))
            r1 = r0 + spec["h"] - 1
            c1 = c0 + spec["w"] - 1
            expanded = (r0 - 4, c0 - 4, r1 + 4, c1 + 4)
            if any(_bboxes_overlap(expanded, other["expanded"]) for other in placed):
                continue
            placed.append({"bbox": (r0, c0, r1, c1), "expanded": expanded, **spec})
            success = True
            break
        if not success:
            return None

    for spec in placed:
        fill_nested_rectangle(grid, spec["bbox"], spec["colors"])

    rect_count = len(placed)
    touched_count = int(rng.integers(1, rect_count))
    touched_indices = set(map(int, rng.choice(rect_count, size=touched_count, replace=False)))
    inner_seed_idx = None
    if rng.random() < 0.45:
        inner_seed_idx = int(rng.choice(sorted(touched_indices)))

    modes: list[str] = []
    for i in range(rect_count):
        if i not in touched_indices:
            modes.append("closed")
        elif i == inner_seed_idx:
            modes.append("closed")
        else:
            modes.append("open" if rng.random() < 0.55 else "none")

    for i, spec in enumerate(placed):
        r0, c0, r1, c1 = spec["bbox"]
        box = (r0 - 2, c0 - 2, r1 + 2, c1 + 2)
        if box[0] < 0 or box[1] < 0 or box[2] >= h or box[3] >= w:
            return None
        if modes[i] == "closed":
            gap = None
            if i in touched_indices and i != inner_seed_idx and rng.random() < 0.35:
                gap = _random_gap(box, rng)
            draw_box_wall(grid, box, gap=gap)
        elif modes[i] == "open":
            draw_box_wall(grid, box, gap=_random_gap(box, rng))

    rects = find_rectangles(grid)
    add_random_segments(grid, rects, rng)

    labels, comps = black_component_labels(grid)
    seeds: list[tuple[int, int]] = []

    outside_targets = [i for i in sorted(touched_indices) if i != inner_seed_idx]
    if outside_targets:
        common_ids = adjacent_black_components(labels, rects[outside_targets[0]])
        for i in outside_targets[1:]:
            common_ids &= adjacent_black_components(labels, rects[i])
        common_ids = {cid for cid in common_ids if len(comps[cid - 1]) >= 2}
        if not common_ids:
            return None
        cid = int(rng.choice(sorted(common_ids)))
        cell = choose_seed_cell(grid, comps[cid - 1], rng)
        if cell is None:
            return None
        seeds.append(cell)

    if inner_seed_idx is not None:
        inner_ids = adjacent_black_components(labels, rects[inner_seed_idx])
        inner_ids = {cid for cid in inner_ids if len(comps[cid - 1]) >= 2}
        if outside_targets and seeds:
            outside_id = labels[seeds[0]]
            inner_ids.discard(int(outside_id))
        if not inner_ids:
            return None
        cid = int(rng.choice(sorted(inner_ids)))
        cell = choose_seed_cell(grid, comps[cid - 1], rng)
        if cell is None:
            return None
        seeds.append(cell)

    if not seeds:
        return None
    if len({seed for seed in seeds}) != len(seeds):
        return None

    for r, c in seeds:
        if int(grid[r, c]) != BG:
            return None
        grid[r, c] = SEED

    try:
        validate_generated_case(grid, touched_indices)
    except AssertionError:
        return None
    return grid


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _random_gap(box: tuple[int, int, int, int], rng: np.random.Generator) -> tuple[str, int, int]:
    r0, c0, r1, c1 = box
    side = str(rng.choice(("top", "bottom", "left", "right")))
    if side in ("top", "bottom"):
        pos = int(rng.integers(c0 + 1, c1))
        width = 1 if c1 - c0 <= 6 else int(rng.integers(1, 3))
        return side, pos, min(c1 - 1, pos + width - 1)
    pos = int(rng.integers(r0 + 1, r1))
    width = 1 if r1 - r0 <= 6 else int(rng.integers(1, 3))
    return side, pos, min(r1 - 1, pos + width - 1)
