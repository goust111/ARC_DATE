import random
from collections import Counter
from functools import lru_cache

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_RETRIES = 400
ARC_MAX = 30
RECTANGLE_CHOICES = [
    (2, 3),
    (2, 4),
    (3, 3),
    (2, 5),
    (2, 7),
    (3, 5),
    (4, 4),
    (3, 6),
    (4, 5),
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, _r1, _c1 = _bbox(coords)
    cells = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    return cells, h, w


def _shape_to_mask(shape: tuple[tuple[int, int], ...], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[int(r), int(c)] = True
    return mask


def _is_top_hint_row(grid: np.ndarray, bg: int) -> bool:
    if grid.shape[0] < 2:
        return False
    cols = [c for c in range(grid.shape[1]) if int(grid[0, c]) != int(bg)]
    if len(cols) < 2:
        return False
    for c in cols:
        if int(grid[1, c]) != int(bg):
            return False
        if c > 0 and int(grid[0, c - 1]) != int(bg):
            return False
        if c + 1 < grid.shape[1] and int(grid[0, c + 1]) != int(bg):
            return False
    return True


def _is_left_hint_col(grid: np.ndarray, bg: int) -> bool:
    if grid.shape[1] < 2:
        return False
    rows = [r for r in range(grid.shape[0]) if int(grid[r, 0]) != int(bg)]
    if len(rows) < 2:
        return False
    for r in rows:
        if int(grid[r, 1]) != int(bg):
            return False
        if r > 0 and int(grid[r - 1, 0]) != int(bg):
            return False
        if r + 1 < grid.shape[0] and int(grid[r + 1, 0]) != int(bg):
            return False
    return True


def _strip_hint_strip(grid: np.ndarray, bg: int) -> np.ndarray:
    clean = np.array(grid, dtype=int, copy=True)
    if _is_top_hint_row(clean, bg):
        clean[0, :] = int(bg)
    if _is_left_hint_col(clean, bg):
        clean[:, 0] = int(bg)
    return clean


def _all_rectangle_dims(shapes: list[tuple[tuple[tuple[int, int], ...], int, int]]) -> list[tuple[int, int]]:
    total_area = sum(len(shape) for shape, _h, _w in shapes)
    min_h = max(h for _shape, h, _w in shapes)
    min_w = max(w for _shape, _h, w in shapes)
    candidates = set()

    for h in range(1, int(total_area**0.5) + 1):
        if total_area % h != 0:
            continue
        w = total_area // h
        if h >= min_h and w >= min_w and _can_tile_rectangle(shapes, h, w):
            candidates.add((int(h), int(w)))
        if w >= min_h and h >= min_w and _can_tile_rectangle(shapes, w, h):
            candidates.add((int(w), int(h)))

    return sorted(candidates)


def _can_tile_rectangle(shapes: list[tuple[tuple[tuple[int, int], ...], int, int]], out_h: int, out_w: int) -> bool:
    n = len(shapes)
    full_mask = (1 << (out_h * out_w)) - 1
    placement_maps: list[list[list[int]]] = []

    for shape, sh, sw in shapes:
        if sh > out_h or sw > out_w:
            return False
        per_cell = [[] for _ in range(out_h * out_w)]
        for dr in range(out_h - sh + 1):
            for dc in range(out_w - sw + 1):
                bits = 0
                for r, c in shape:
                    idx = (int(r) + int(dr)) * out_w + int(c) + int(dc)
                    bits |= 1 << idx
                for idx in range(out_h * out_w):
                    if (bits >> idx) & 1:
                        per_cell[idx].append(bits)
        placement_maps.append(per_cell)

    @lru_cache(maxsize=None)
    def search(used: int, filled: int) -> bool:
        if filled == full_mask:
            return used == (1 << n) - 1

        first_empty = 0
        while (filled >> first_empty) & 1:
            first_empty += 1

        for i in range(n):
            if (used >> i) & 1:
                continue
            for bits in placement_maps[i][first_empty]:
                if bits & filled:
                    continue
                if search(used | (1 << i), filled | bits):
                    return True
        return False

    return search(0, 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered same-color fragments into rectangles, then stack them by area.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_get_bg(g))
    g = _strip_hint_strip(g, bg)

    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    groups = []

    for color in colors:
        coords = np.argwhere(g == int(color))
        assert coords.size > 0
        fragments = _find_components_of_color(g, int(color))
        shapes = [_normalize_shape(comp) for comp in fragments]
        dims = _all_rectangle_dims(shapes)
        assert dims, f"color {color} cannot form a rectangle"
        rect_h, rect_w = dims[0]
        group_box = _bbox(coords)
        groups.append(
            {
                "color": int(color),
                "area": int(coords.shape[0]),
                "h": int(rect_h),
                "w": int(rect_w),
                "bbox": group_box,
            }
        )

    groups.sort(key=lambda item: (-int(item["area"]), int(item["bbox"][0]), int(item["bbox"][1])))

    out_h = sum(int(item["h"]) for item in groups) + max(0, len(groups) - 1)
    out_w = max(int(item["w"]) for item in groups)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    r0 = 0
    for item in groups:
        h = int(item["h"])
        w = int(item["w"])
        c0 = (out_w - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = int(item["color"])
        r0 += h + 1

    return out


def _random_partition_sizes(area: int, n_parts: int) -> list[int]:
    sizes = [2] * n_parts
    for _ in range(area - 2 * n_parts):
        sizes[random.randrange(n_parts)] += 1
    random.shuffle(sizes)
    return sizes


def _partition_rectangle(h: int, w: int, n_parts: int) -> list[tuple[tuple[int, int], ...]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    sizes = _random_partition_sizes(h * w, n_parts)
    seeds = random.sample(cells, n_parts)
    parts = [{seed} for seed in seeds]
    owners = {seed: i for i, seed in enumerate(seeds)}

    while len(owners) < h * w:
        choices = []
        for i, part in enumerate(parts):
            if len(part) >= sizes[i]:
                continue
            frontier = set()
            for r, c in part:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in owners:
                        frontier.add((nr, nc))
            if frontier:
                choices.append((i, list(frontier)))
        if not choices:
            return None
        i, frontier = random.choice(choices)
        cell = random.choice(frontier)
        parts[i].add(cell)
        owners[cell] = i

    if sorted(len(part) for part in parts) != sorted(sizes):
        return None
    return [tuple(sorted(part)) for part in parts]


def _generate_group(color: int, shape_choice: tuple[int, int]) -> dict | None:
    target_h, target_w = shape_choice
    area = target_h * target_w
    max_parts = min(4, area // 2)
    if max_parts < 2:
        return None

    for _ in range(200):
        n_parts = random.randint(2, max_parts)
        parts = _partition_rectangle(target_h, target_w, n_parts)
        if parts is None:
            continue

        shapes = []
        patches = []
        non_rectangular = 0
        for part in parts:
            coords = np.array(part, dtype=int)
            shape, ph, pw = _normalize_shape(coords)
            mask = _shape_to_mask(shape, ph, pw)
            patch = np.full((ph, pw), BG, dtype=int)
            patch[mask] = int(color)
            shapes.append((shape, ph, pw))
            patches.append({"shape": shape, "patch": patch, "mask": mask})
            if len(shape) != ph * pw:
                non_rectangular += 1

        dims = _all_rectangle_dims(shapes)
        if dims != [(int(target_h), int(target_w))]:
            continue
        if len(patches) > 2 and non_rectangular == 0:
            continue

        return {
            "color": int(color),
            "h": int(target_h),
            "w": int(target_w),
            "area": int(area),
            "patches": patches,
        }
    return None


def _mark_blocked(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    for r, c in np.argwhere(mask):
        rr = int(r0 + r)
        cc = int(c0 + c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
                    blocked[nr, nc] = True


def _can_place_mask(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > blocked.shape[0] or c0 + w > blocked.shape[1]:
        return False
    for r, c in np.argwhere(mask):
        if blocked[int(r0 + r), int(c0 + c)]:
            return False
    return True


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]
    _mark_blocked(blocked, mask, r0, c0)


def _scatter_fragments(groups: list[dict], hint_axis: str, out_shape: tuple[int, int]) -> np.ndarray | None:
    min_h = max(14, out_shape[0] + 6)
    min_w = max(16, out_shape[1] + 10)
    if min_h > ARC_MAX or min_w > ARC_MAX:
        return None
    canvas_h = random.randint(min_h, min(ARC_MAX, min_h + 8))
    canvas_w = random.randint(min_w, min(ARC_MAX, min_w + 10))

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)

    row_min = 2 if hint_axis == "top" else 1
    col_min = 1 if hint_axis == "top" else 2
    row_max = canvas_h - 1
    col_max = canvas_w - 1

    pieces = []
    for group in groups:
        for patch_info in group["patches"]:
            pieces.append({"color": group["color"], **patch_info})
    random.shuffle(pieces)
    pieces.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)

    for item in pieces:
        patch = item["patch"]
        mask = item["mask"]
        ph, pw = patch.shape
        coords = [
            (r0, c0)
            for r0 in range(row_min, row_max - ph + 1)
            for c0 in range(col_min, col_max - pw + 1)
            if _can_place_mask(blocked, mask, r0, c0)
        ]
        if not coords:
            return None
        r0, c0 = random.choice(coords)
        _place_patch(grid, blocked, patch, mask, r0, c0)

    return grid


def _add_hint_strip(grid: np.ndarray, groups: list[dict], hint_axis: str) -> bool:
    colors_by_area = [int(item["color"]) for item in sorted(groups, key=lambda item: -int(item["area"]))]
    hint_order = colors_by_area[:]
    attempts = 0
    while hint_order == colors_by_area and attempts < 20:
        random.shuffle(hint_order)
        attempts += 1
    if hint_order == colors_by_area:
        hint_order = list(reversed(hint_order))

    def choose_isolated(candidates: list[int], count: int) -> list[int] | None:
        random.shuffle(candidates)
        chosen: list[int] = []
        for pos in candidates:
            if all(abs(int(pos) - int(other)) > 1 for other in chosen):
                chosen.append(int(pos))
            if len(chosen) == count:
                return sorted(chosen)
        return None

    if hint_axis == "top":
        positions = list(range(1, grid.shape[1] - 1))
        if len(positions) < len(hint_order) * 2 - 1:
            return False
        valid = [p for p in positions if int(grid[1, p]) == BG]
        if len(valid) < len(hint_order):
            return False
        chosen = choose_isolated(valid, len(hint_order))
        if chosen is None:
            return False
        for c, color in zip(chosen, hint_order):
            if int(grid[0, c]) != BG or int(grid[1, c]) != BG:
                return False
            grid[0, c] = int(color)
        return True

    positions = list(range(1, grid.shape[0] - 1))
    if len(positions) < len(hint_order) * 2 - 1:
        return False
    valid = [p for p in positions if int(grid[p, 1]) == BG]
    if len(valid) < len(hint_order):
        return False
    chosen = choose_isolated(valid, len(hint_order))
    if chosen is None:
        return False
    for r, color in zip(chosen, hint_order):
        if int(grid[r, 0]) != BG or int(grid[r, 1]) != BG:
            return False
        grid[r, 0] = int(color)
    return True


def _build_expected(groups: list[dict]) -> np.ndarray:
    ordered = sorted(groups, key=lambda item: -int(item["area"]))
    out_h = sum(int(item["h"]) for item in ordered) + len(ordered) - 1
    out_w = max(int(item["w"]) for item in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)
    r0 = 0
    for item in ordered:
        h = int(item["h"])
        w = int(item["w"])
        c0 = (out_w - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = int(item["color"])
        r0 += h + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.choice([3, 4])
    colors = random.sample(PALETTE, n_groups)

    choices = RECTANGLE_CHOICES[:]
    random.shuffle(choices)
    selected = []
    used_areas = set()
    for base_h, base_w in choices:
        h, w = (base_h, base_w) if random.random() < 0.5 else (base_w, base_h)
        if h * w in used_areas:
            continue
        selected.append((h, w))
        used_areas.add(h * w)
        if len(selected) == n_groups:
            break
    if len(selected) != n_groups:
        return None

    groups = []
    for color, dims in zip(colors, selected):
        group = _generate_group(int(color), dims)
        if group is None:
            return None
        groups.append(group)

    expected = _build_expected(groups)
    hint_axis = random.choice(["top", "left"])
    grid = _scatter_fragments(groups, hint_axis, expected.shape)
    if grid is None:
        return None
    if not _add_hint_strip(grid, groups, hint_axis):
        return None

    if int(_get_bg(grid)) != BG:
        return None
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a random valid input for the scattered-fragments rectangle puzzle.
    """
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_RETRIES} attempts")
