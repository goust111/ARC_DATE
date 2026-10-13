from __future__ import annotations

import random

import numpy as np


BG = 0
SOIL = 5
FRAME_COLOR = 1
COLOR_POOL = (2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250
BLOB_RETRIES = 80

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

TEMPLATES: dict[str, tuple[tuple[int, int], ...]] = {
    "cross3": tuple(sorted({(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)})),
    "tee3": tuple(sorted({(0, 0), (-1, -1), (-1, 0), (-1, 1), (1, 0)})),
    "el3": tuple(sorted({(0, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)})),
    "diamond5": tuple(
        sorted(
            {
                (dr, dc)
                for dr in range(-2, 3)
                for dc in range(-2, 3)
                if abs(dr) + abs(dc) <= 2
            }
        )
    ),
    "plus5": tuple(sorted({(dr, 0) for dr in range(-2, 3)} | {(0, dc) for dc in range(-2, 3)})),
    "hook5": tuple(
        sorted(
            {
                (-2, -2),
                (-1, -2),
                (0, -2),
                (1, -2),
                (2, -2),
                (2, -1),
                (2, 0),
                (2, 1),
                (2, 2),
                (0, -1),
                (0, 0),
                (1, 0),
                (0, 1),
            }
        )
    ),
}

SMALL_TEMPLATE_NAMES = ("cross3", "tee3", "el3")
LARGE_TEMPLATE_NAMES = ("diamond5", "plus5", "hook5")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if bool(seen[r, c]):
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((int(cr), int(cc)))
            for dr, dc in NEIGHBORS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _corner_bbox(shape: tuple[int, int], corner: str, height: int, width: int) -> tuple[int, int, int, int]:
    h, w = shape
    if corner == "tl":
        r0 = 0
        c0 = 0
    elif corner == "tr":
        r0 = 0
        c0 = w - width
    elif corner == "bl":
        r0 = h - height
        c0 = 0
    else:
        r0 = h - height
        c0 = w - width
    return int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)


def _scan_frame_extent(grid: np.ndarray, corner: str) -> tuple[int, int, int] | None:
    h, w = grid.shape
    if corner == "tl":
        rr, cc = 0, 0
        dr, dc = 1, 1
    elif corner == "tr":
        rr, cc = 0, w - 1
        dr, dc = 1, -1
    elif corner == "bl":
        rr, cc = h - 1, 0
        dr, dc = -1, 1
    else:
        rr, cc = h - 1, w - 1
        dr, dc = -1, -1
    color = int(grid[rr, cc])
    if color == BG:
        return None

    width = 0
    cur_c = cc
    while 0 <= cur_c < w and int(grid[rr, cur_c]) == color:
        width += 1
        cur_c += dc
    height = 0
    cur_r = rr
    while 0 <= cur_r < h and int(grid[cur_r, cc]) == color:
        height += 1
        cur_r += dr
    if height < 5 or width < 7:
        return None
    return color, height, width


def _find_legend(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for corner in ("tl", "tr", "bl", "br"):
        info = _scan_frame_extent(g, corner)
        if info is None:
            continue
        frame_color, height, width = info
        r0, c0, r1, c1 = _corner_bbox(g.shape, corner, height, width)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(sub[0, :] == frame_color)
            and np.all(sub[-1, :] == frame_color)
            and np.all(sub[:, 0] == frame_color)
            and np.all(sub[:, -1] == frame_color)
        ):
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0 or not bool(np.any(inner != BG)):
            continue
        mask = np.zeros_like(g, dtype=bool)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        candidates.append((int(frame_color), (int(r0), int(c0), int(r1), int(c1)), mask))
    assert len(candidates) == 1, "expected exactly one corner legend frame"
    return candidates[0]


def _parse_legend(grid: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[tuple[int, int], ...]]]:
    frame_color, (r0, c0, r1, c1), legend_mask = _find_legend(grid)
    inner = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    colors = sorted(int(v) for v in np.unique(inner) if int(v) not in (BG, frame_color))
    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    for color in colors:
        comps = _components(inner == color)
        assert len(comps) == 2, "each legend color must form one key pixel and one template"
        comps = sorted(comps, key=len)
        assert len(comps[0]) == 1 and len(comps[1]) > 1, "legend entry must contain a singleton key and a template"
        tr0, tc0, tr1, tc1 = _bbox(comps[1])
        assert (tr1 - tr0) % 2 == 0 and (tc1 - tc0) % 2 == 0, "template bbox must have odd size"
        cr = (tr0 + tr1) // 2
        cc = (tc0 + tc1) // 2
        mapping[int(color)] = tuple(sorted((int(r - cr), int(c - cc)) for r, c in comps[1]))
    assert mapping, "legend mapping must be non-empty"
    return legend_mask, mapping


def _grow_component(out: np.ndarray, coords: list[tuple[int, int]], templates: dict[int, tuple[tuple[int, int], ...]]) -> None:
    comp_set = set(coords)
    seed_cells: list[tuple[int, int, int]] = []
    for r, c in coords:
        val = int(out[r, c])
        if val != SOIL:
            seed_cells.append((int(r), int(c), int(val)))
    assert 1 <= len(seed_cells) <= 3, "each soil component must contain one to three seeds"
    assert len({color for _, _, color in seed_cells}) == len(seed_cells), "seed colors inside one soil must be distinct"

    frontiers: dict[int, set[tuple[int, int]]] = {}
    for r, c, color in seed_cells:
        assert color in templates, "every seed color must appear in the legend"
        frontiers.setdefault(int(color), set()).add((int(r), int(c)))

    blocked: set[tuple[int, int]] = set()
    while frontiers:
        claims: dict[tuple[int, int], set[int]] = {}
        for color, frontier in frontiers.items():
            offsets = templates[int(color)]
            for r, c in frontier:
                for dr, dc in offsets:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if (nr, nc) not in comp_set:
                        continue
                    if (nr, nc) in blocked or int(out[nr, nc]) != SOIL:
                        continue
                    claims.setdefault((nr, nc), set()).add(int(color))
        if not claims:
            break
        next_frontiers: dict[int, set[tuple[int, int]]] = {}
        for pos, colors in claims.items():
            if len(colors) == 1:
                color = next(iter(colors))
                out[pos] = int(color)
                next_frontiers.setdefault(int(color), set()).add(pos)
            else:
                blocked.add(pos)
        frontiers = next_frontiers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_mask, mapping = _parse_legend(g)
    scene_mask = (g != BG) & ~legend_mask
    soil_components = _components(scene_mask)
    assert soil_components, "expected at least one soil component"

    out = g.copy()
    for comp in soil_components:
        colors = {int(g[r, c]) for r, c in comp}
        assert colors <= {SOIL, *mapping.keys()}, "scene may only contain soil and legend-listed seed colors"
        _grow_component(out, comp, mapping)

    return out


def _legend_layout(entries: list[tuple[int, tuple[tuple[int, int], ...]]]) -> tuple[np.ndarray, tuple[int, int]]:
    count = len(entries)
    inner_h = count * 5 + (count - 1)
    inner_w = 8
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    for idx, (color, offsets) in enumerate(entries):
        row0 = 1 + idx * 6
        patch[row0 + 2, 2] = int(color)
        cr = row0 + 2
        cc = 6
        for dr, dc in offsets:
            patch[cr + dr, cc + dc] = int(color)
    return patch, patch.shape


def _place_patch(grid: np.ndarray, patch: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    ph, pw = patch.shape
    r0, c0, r1, c1 = _corner_bbox(grid.shape, corner, ph, pw)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return int(r0), int(c0), int(r1), int(c1)


def _weighted_choice(cells: list[tuple[int, int]], weights: list[int]) -> tuple[int, int]:
    total = sum(weights)
    pick = random.randrange(total)
    running = 0
    for cell, weight in zip(cells, weights):
        running += weight
        if pick < running:
            return cell
    return cells[-1]


def _grow_blob(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(1, height - 1), random.randrange(1, width - 1))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and not bool(mask[nr, nc]):
                frontier.add((int(nr), int(nc)))

    add_frontier(*start)
    while int(mask.sum()) < target_area and frontier:
        cells = list(frontier)
        weights = []
        for r, c in cells:
            touch = 0
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]):
                    touch += 1
            weights.append(1 + touch * touch)
        cell = _weighted_choice(cells, weights)
        frontier.remove(cell)
        mask[cell] = True
        add_frontier(*cell)

    for _ in range(2):
        fills: list[tuple[int, int]] = []
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if bool(mask[r, c]):
                    continue
                touch = sum(1 for dr, dc in NEIGHBORS4 if bool(mask[r + dr, c + dc]))
                if touch >= 3 and random.random() < 0.55:
                    fills.append((int(r), int(c)))
        for cell in fills:
            mask[cell] = True

    return _crop_mask(mask)


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            continue
        if all(bool(mask[r + dr, c + dc]) for dr, dc in NEIGHBORS4):
            cells.append((int(r), int(c)))
    return cells


def _pick_spaced(cells: list[tuple[int, int]], count: int, min_dist: int) -> list[tuple[int, int]] | None:
    if len(cells) < count:
        return None
    for _ in range(60):
        order = cells[:]
        random.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for cell in order:
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= min_dist for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _make_soil_patch(seed_count: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    min_dim = 6 + seed_count
    max_dim = 10 + seed_count
    min_area = 18 + 8 * seed_count
    for _ in range(BLOB_RETRIES):
        height = random.randint(min_dim, max_dim)
        width = random.randint(min_dim, max_dim)
        target = random.randint(min_area, max(min_area, int(height * width * 0.72)))
        mask = _grow_blob(height, width, target)
        if int(mask.sum()) < min_area:
            continue
        interior = _interior_cells(mask)
        if len(interior) < seed_count:
            continue
        min_dist = 3 if seed_count >= 2 else 1
        picks = _pick_spaced(interior, seed_count, min_dist)
        if picks is None and seed_count >= 2:
            picks = _pick_spaced(interior, seed_count, 2)
        if picks is None:
            continue
        return mask, picks
    raise RuntimeError("failed to build a soil patch with enough seed positions")


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _can_place(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not bool(np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]))


def _mark_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _choose_templates() -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    entry_count = random.choice((3, 4))
    if entry_count == 3:
        small_count = random.choice((1, 2))
        large_count = 3 - small_count
    else:
        small_count = 2
        large_count = 2
    names = random.sample(SMALL_TEMPLATE_NAMES, small_count) + random.sample(LARGE_TEMPLATE_NAMES, large_count)
    random.shuffle(names)
    colors = random.sample(COLOR_POOL, entry_count)
    return [(int(color), TEMPLATES[name]) for color, name in zip(colors, names)]


def _seed_count_plan(num_regions: int, num_colors: int) -> list[int]:
    for _ in range(200):
        counts = [random.choice((1, 2, 3)) for _ in range(num_regions)]
        if sum(counts) < num_colors:
            continue
        if max(counts) == 1 and random.random() < 0.75:
            continue
        return counts
    counts = [1] * num_regions
    idx = 0
    while sum(counts) < num_colors:
        counts[idx % num_regions] = min(3, counts[idx % num_regions] + 1)
        idx += 1
    return counts


def _assign_region_colors(region_seed_counts: list[int], legend_colors: list[int]) -> list[list[int]]:
    slots = [(idx, slot) for idx, count in enumerate(region_seed_counts) for slot in range(count)]
    random.shuffle(slots)
    assigned: list[list[int]] = [[] for _ in region_seed_counts]
    for color, (region_idx, _slot) in zip(random.sample(legend_colors, len(legend_colors)), slots):
        assigned[region_idx].append(int(color))
    for region_idx, need in enumerate(region_seed_counts):
        while len(assigned[region_idx]) < need:
            choices = [color for color in legend_colors if color not in assigned[region_idx]]
            assigned[region_idx].append(int(random.choice(choices)))
        random.shuffle(assigned[region_idx])
    return assigned


def _gen_attempt() -> np.ndarray:
    entries = _choose_templates()
    legend_colors = [color for color, _ in entries]
    corner = random.choice(("tl", "tr", "bl", "br"))
    legend_patch, (legend_h, legend_w) = _legend_layout(entries)

    height = random.randint(max(28, legend_h + 4), 38)
    width = random.randint(max(24, legend_w + 10), 36)
    grid = np.full((height, width), BG, dtype=int)

    legend_bbox = _place_patch(grid, legend_patch, corner)
    occupied = np.zeros_like(grid, dtype=bool)
    _mark_bbox(occupied, _expand_bbox(legend_bbox, 1, grid.shape))

    num_regions = random.choice((2, 3))
    seed_counts = _seed_count_plan(num_regions, len(entries))
    region_colors = _assign_region_colors(seed_counts, legend_colors)

    region_specs = []
    for idx in range(num_regions):
        mask, seeds = _make_soil_patch(seed_counts[idx])
        region_specs.append(
            {
                "mask": mask,
                "seeds": seeds,
                "colors": region_colors[idx],
                "area": int(mask.sum()),
            }
        )
    region_specs.sort(key=lambda item: int(item["area"]), reverse=True)

    for spec in region_specs:
        mask = spec["mask"]
        ph, pw = mask.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, height - ph - 1)
            c0 = random.randint(1, width - pw - 1)
            bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
            expanded = _expand_bbox(bbox, 1, grid.shape)
            if not _can_place(occupied, expanded):
                continue
            patch = grid[r0 : r0 + ph, c0 : c0 + pw]
            patch[mask] = SOIL
            for (sr, sc), color in zip(spec["seeds"], spec["colors"]):
                patch[sr, sc] = int(color)
            _mark_bbox(occupied, expanded)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place all soil regions")

    out = main(grid)
    legend_mask, mapping = _parse_legend(grid)
    assert set(mapping) == set(legend_colors)
    assert np.array_equal(out[legend_mask], grid[legend_mask]), "legend must stay unchanged"
    assert np.array_equal(out != BG, grid != BG), "growth may only recolor existing non-background cells"
    assert bool(np.any(out != grid)), "generated puzzle must require a non-trivial transformation"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
