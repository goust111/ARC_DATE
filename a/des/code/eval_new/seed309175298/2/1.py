import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

FILL_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE]
MAX_RETRIES = 300
MAX_MAIN_W = 20
MAX_MAIN_H = 29
LEGEND_INNER_H = 3
LEGEND_INNER_W = 6
LEGEND_W = LEGEND_INNER_W + 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed grey-bounded region by matching its maroon seed shape to the right-side legend.

    Input structure:
    - Black background with several irregular closed regions bounded by Grey lines.
    - Each enclosed region contains one small Maroon seed shape made of 2-4 pixels.
    - A right-side legend area, separated from the main scene, contains boxed examples: the same Maroon shapes plus an adjacent single fill-color pixel.

    Rule:
    - Remove the legend area from the output.
    - For every enclosed region, extract the exact Maroon seed shape up to translation only; rotation is significant and must match exactly.
    - Find the legend entry with the identical Maroon shape orientation and read its adjacent color.
    - Fill all Black pixels inside that enclosed region with the matched color, while keeping Grey walls and the Maroon seed unchanged.

    Key insight:
    - The seed shape is both the lookup key and the orientation cue, so shape identity alone is insufficient; the rotated pixel arrangement must match the legend exactly.
    """
    g = np.asarray(grid, dtype=int)
    sep_col = _find_separator_col(g)

    main_part = g[:, :sep_col].copy()
    legend_part = g[:, sep_col + 1 :]

    mapping = _parse_legend(legend_part)
    assert mapping, "legend mapping must be non-empty"

    for comp in _enclosed_components(main_part):
        colors = {int(main_part[r, c]) for r, c in comp}
        if MAROON not in colors:
            continue
        extras = colors - {BLACK, MAROON}
        assert not extras, "main enclosed regions should only contain black and maroon"
        seed = [(r, c) for r, c in comp if int(main_part[r, c]) == MAROON]
        sig = _shape_signature(seed)
        assert sig in mapping, "every seed shape must exist in the legend"
        fill = mapping[sig]
        for r, c in comp:
            if int(main_part[r, c]) == BLACK:
                main_part[r, c] = fill

    return main_part


def _find_separator_col(grid: np.ndarray) -> int:
    full_grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    candidates = []
    for c in full_grey_cols:
        if c == 0 or c == grid.shape[1] - 1:
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if left.size == 0 or right.size == 0:
            continue
        region_like = 0
        legend_like = 0
        for comp in _enclosed_components(left):
            vals = {int(left[r, cc]) for r, cc in comp}
            if MAROON not in vals:
                continue
            extras = vals - {BLACK, MAROON}
            if not extras:
                region_like += 1
        for comp in _enclosed_components(right):
            vals = {int(right[r, cc]) for r, cc in comp}
            if MAROON not in vals:
                continue
            extras = vals - {BLACK, MAROON}
            if len(extras) == 1:
                legend_like += 1
        if region_like >= 1 and legend_like >= 1:
            candidates.append(c)
    assert candidates, "could not locate the legend separator"
    return min(candidates, key=lambda c: (c, -grid.shape[1]))


def _parse_legend(legend: np.ndarray) -> dict[tuple[tuple[int, int], ...], int]:
    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    for comp in _enclosed_components(legend):
        vals = {int(legend[r, c]) for r, c in comp}
        if MAROON not in vals:
            continue
        extras = vals - {BLACK, MAROON}
        if len(extras) != 1:
            continue
        sig = _shape_signature([(r, c) for r, c in comp if int(legend[r, c]) == MAROON])
        fill = int(next(iter(extras)))
        assert sig not in mapping, "legend shapes must be unique"
        mapping[sig] = fill
    return mapping


def _enclosed_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = grid != GREY
    comps = []
    for comp in _components(mask):
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in comp):
            continue
        comps.append(comp)
    return comps


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _shape_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    assert coords, "shape must contain at least one pixel"
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return _shape_signature(coords)


def _rotate_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rotated = [(c, -r) for r, c in coords]
    return _normalize_coords(rotated)


def _shape_size(sig: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in sig) + 1, max(c for _, c in sig) + 1


def _build_shape_catalog():
    base_shapes = {
        "domino": [(0, 0), (0, 1)],
        "line3": [(0, 0), (0, 1), (0, 2)],
        "L3": [(0, 0), (1, 0), (1, 1)],
        "square": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "T4": [(0, 1), (1, 0), (1, 1), (1, 2)],
        "Z4": [(0, 0), (0, 1), (1, 1), (1, 2)],
        "L4": [(0, 0), (1, 0), (2, 0), (2, 1)],
    }
    catalog = []
    by_family: dict[str, list[dict[str, object]]] = {}
    for family, coords in base_shapes.items():
        sig = _normalize_coords(coords)
        seen = set()
        cur = sig
        for _ in range(4):
            if cur not in seen:
                seen.add(cur)
                h, w = _shape_size(cur)
                if h <= LEGEND_INNER_H and w <= LEGEND_INNER_H:
                    item = {"family": family, "sig": cur, "h": h, "w": w}
                    catalog.append(item)
                    by_family.setdefault(family, []).append(item)
            cur = _rotate_coords(cur)
    return catalog, by_family


SHAPE_CATALOG, SHAPES_BY_FAMILY = _build_shape_catalog()


def _sample_entries() -> list[dict[str, object]]:
    n_entries = random.randint(3, 5)
    entries: list[dict[str, object]] = []

    if random.random() < 0.75:
        multi_families = [family for family, items in SHAPES_BY_FAMILY.items() if len(items) >= 2 and family != "square"]
        family = random.choice(multi_families)
        family_items = SHAPES_BY_FAMILY[family]
        max_take = min(len(family_items), n_entries)
        if max_take >= 2:
            take = 2
            if max_take >= 3 and n_entries >= 4 and random.random() < 0.35:
                take = 3
            entries.extend(random.sample(family_items, take))

    chosen = {item["sig"] for item in entries}
    pool = [item for item in SHAPE_CATALOG if item["sig"] not in chosen]
    while len(entries) < n_entries:
        item = random.choice(pool)
        entries.append(item)
        pool = [cand for cand in pool if cand["sig"] != item["sig"]]

    random.shuffle(entries)
    colors = random.sample(FILL_COLORS, n_entries)
    result = []
    for item, color in zip(entries, colors):
        result.append(
            {
                "family": item["family"],
                "sig": item["sig"],
                "h": item["h"],
                "w": item["w"],
                "color": color,
            }
        )
    return result


def _sample_trimmed_rectangle(max_dim: int) -> np.ndarray:
    h = random.randint(2, max_dim)
    w = random.randint(2, max_dim)
    mask = np.ones((h, w), dtype=bool)
    removals = random.randint(0, min(3, h * w - 4))
    for _ in range(removals):
        candidates = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                if mask.sum() <= 4:
                    break
                if 0 < r < h - 1 and 0 < c < w - 1 and np.all(mask[r - 1 : r + 2, c - 1 : c + 2]):
                    continue
                trial = mask.copy()
                trial[r, c] = False
                if _is_connected(trial):
                    candidates.append((r, c))
            else:
                continue
            break
        if not candidates:
            continue
        rr, cc = random.choice(candidates)
        mask[rr, cc] = False
    return mask


def _sample_grown_mask(max_dim: int) -> np.ndarray:
    h = random.randint(2, max_dim)
    w = random.randint(2, max_dim)
    target = random.randint(max(4, h + w - 1), min(h * w, h * w - 1 if h * w > 4 else h * w))
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    while int(mask.sum()) < target:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            break
        mask[random.choice(frontier)] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    start = tuple(coords[0])
    q = deque([start])
    seen[start] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _render_region(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    canvas = np.full((2 * h + 1, 2 * w + 1), BLACK, dtype=int)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or not mask[r - 1, c]:
                canvas[2 * r, 2 * c : 2 * c + 3] = GREY
            if r == h - 1 or not mask[r + 1, c]:
                canvas[2 * r + 2, 2 * c : 2 * c + 3] = GREY
            if c == 0 or not mask[r, c - 1]:
                canvas[2 * r : 2 * r + 3, 2 * c] = GREY
            if c == w - 1 or not mask[r, c + 1]:
                canvas[2 * r : 2 * r + 3, 2 * c + 2] = GREY
    return canvas


def _seed_placements(region: np.ndarray, sig: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    sh, sw = _shape_size(sig)
    h, w = region.shape
    placements = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if all(int(region[r0 + dr, c0 + dc]) == BLACK for dr, dc in sig):
                placements.append((r0, c0))
    return placements


def _make_region(sig: tuple[tuple[int, int], ...], max_dim: int) -> np.ndarray | None:
    for _ in range(120):
        if random.random() < 0.55:
            coarse = _sample_trimmed_rectangle(max_dim)
        else:
            coarse = _sample_grown_mask(max_dim)
        if not _is_connected(coarse):
            continue
        region = _render_region(coarse)
        if int(np.count_nonzero(region == BLACK)) <= len(sig) + 2:
            continue
        placements = _seed_placements(region, sig)
        if not placements:
            continue
        out = region.copy()
        r0, c0 = random.choice(placements)
        for dr, dc in sig:
            out[r0 + dr, c0 + dc] = MAROON
        return out
    return None


def _pack_regions(regions: list[np.ndarray]) -> np.ndarray | None:
    indices = list(range(len(regions)))
    random.shuffle(indices)

    for use_two_cols in random.sample([False, True], 2):
        gap = random.randint(2, 3)
        margin = random.randint(1, 2)

        if not use_two_cols or len(regions) < 3:
            width_base = max(regions[i].shape[1] for i in indices) + 2 * margin
            height_base = sum(regions[i].shape[0] for i in indices) + gap * (len(indices) - 1) + 2 * margin
            if width_base > MAX_MAIN_W or height_base > MAX_MAIN_H:
                continue
            extra_w = random.randint(0, min(2, MAX_MAIN_W - width_base))
            extra_h = random.randint(0, min(2, MAX_MAIN_H - height_base))
            width = width_base + extra_w
            height = height_base + extra_h
            canvas = np.full((height, width), BLACK, dtype=int)
            y = margin
            for i in indices:
                reg = regions[i]
                x = margin + random.randint(0, width - width_base)
                rh, rw = reg.shape
                canvas[y : y + rh, x : x + rw] = np.where(reg != BLACK, reg, canvas[y : y + rh, x : x + rw])
                y += rh + gap
            return canvas

        columns = [[], []]
        col_heights = [0, 0]
        col_widths = [0, 0]
        for i in indices:
            rh, rw = regions[i].shape
            col = 0 if (col_heights[0], random.random()) <= (col_heights[1], 0.5) else 1
            columns[col].append(i)
            col_heights[col] += rh + (gap if columns[col][:-1] else 0)
            col_widths[col] = max(col_widths[col], rw)

        if not columns[0] or not columns[1]:
            continue

        width_base = 2 * margin + col_widths[0] + gap + col_widths[1]
        height_base = 2 * margin + max(col_heights)
        if width_base > MAX_MAIN_W or height_base > MAX_MAIN_H:
            continue
        extra_w = random.randint(0, min(2, MAX_MAIN_W - width_base))
        extra_h = random.randint(0, min(2, MAX_MAIN_H - height_base))
        width = width_base + extra_w
        height = height_base + extra_h
        canvas = np.full((height, width), BLACK, dtype=int)

        x0 = margin
        x1 = margin + col_widths[0] + gap
        for col_idx, xbase in enumerate((x0, x1)):
            y = margin
            for i in columns[col_idx]:
                reg = regions[i]
                rh, rw = reg.shape
                x = xbase + random.randint(0, col_widths[col_idx] - rw)
                canvas[y : y + rh, x : x + rw] = np.where(reg != BLACK, reg, canvas[y : y + rh, x : x + rw])
                y += rh + gap
        return canvas
    return None


def _build_legend(entries: list[dict[str, object]]) -> np.ndarray:
    n = len(entries)
    legend = np.full((n * (LEGEND_INNER_H + 1) + 1, LEGEND_W), BLACK, dtype=int)
    legend[:, 0] = GREY
    legend[:, -1] = GREY
    for k in range(n + 1):
        legend[k * (LEGEND_INNER_H + 1), :] = GREY

    order = entries[:]
    random.shuffle(order)
    for idx, entry in enumerate(order):
        sig = entry["sig"]
        fill = int(entry["color"])
        sh, sw = _shape_size(sig)
        top = idx * (LEGEND_INNER_H + 1) + 1
        r_off = random.randint(0, LEGEND_INNER_H - sh)
        c_off = random.randint(0, LEGEND_INNER_W - (sw + 1))
        for dr, dc in sig:
            legend[top + r_off + dr, 1 + c_off + dc] = MAROON
        edge_rows = [dr for dr, dc in sig if dc == sw - 1]
        color_row = random.choice(edge_rows)
        legend[top + r_off + color_row, 1 + c_off + sw] = fill
    return legend


def _assemble_grid(main_part: np.ndarray, legend: np.ndarray) -> np.ndarray | None:
    main_h, main_w = main_part.shape
    legend_h, legend_w = legend.shape
    max_dim = max(main_h, legend_h)
    if max_dim >= 30:
        return None

    extra_h_cap = min(4, 30 - max_dim)
    if extra_h_cap < 1:
        return None
    grid_h = max_dim + random.randint(1, extra_h_cap)

    min_w = main_w + 1 + legend_w
    if min_w > 30:
        return None
    extra_w_cap = min(2, 30 - min_w)
    gap = random.randint(0, min(1, 30 - min_w))
    right_pad = random.randint(0, extra_w_cap)
    grid_w = main_w + 1 + gap + legend_w + right_pad
    if grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    main_top = random.randint(0, grid_h - main_h)
    legend_top = random.randint(0, grid_h - legend_h)
    grid[main_top : main_top + main_h, :main_w] = main_part
    grid[:, main_w] = GREY
    legend_left = main_w + 1 + gap
    grid[legend_top : legend_top + legend_h, legend_left : legend_left + legend_w] = np.where(
        legend != BLACK,
        legend,
        grid[legend_top : legend_top + legend_h, legend_left : legend_left + legend_w],
    )
    return grid


def _gen_attempt() -> np.ndarray | None:
    entries = _sample_entries()
    used_count = random.randint(2, min(4, len(entries)))
    used_entries = random.sample(entries, used_count)
    n_regions = random.randint(used_count, min(used_count + 1, 4))

    assignments = used_entries[:]
    while len(assignments) < n_regions:
        assignments.append(random.choice(used_entries))
    random.shuffle(assignments)

    max_dim = 3 if n_regions >= 4 else random.choice([3, 4])
    regions = []
    for entry in assignments:
        reg = _make_region(entry["sig"], max_dim)
        if reg is None:
            return None
        regions.append(reg)

    main_part = _pack_regions(regions)
    if main_part is None:
        return None

    legend = _build_legend(entries)
    grid = _assemble_grid(main_part, legend)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape[0] != grid.shape[0] or out.shape[1] != main_part.shape[1]:
        return None
    if not np.any((out != BLACK) & (out != GREY) & (out != MAROON)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
