from __future__ import annotations

import itertools
import math
import random
from functools import lru_cache

import numpy as np


Coord = tuple[int, int]
Shape = tuple[Coord, ...]


NEIGHBORS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
AREA_CHOICES = (6, 8, 9, 10, 12)
PAIR_COUNT_RANGE = (2, 4)
WIDTH_RANGE = (7, 20)
HEIGHT_RANGE = (12, 24)
MAX_GENERATION_ATTEMPTS = 300


def _normalize_cells(cells: set[Coord] | list[Coord] | tuple[Coord, ...]) -> Shape:
    pts = tuple(sorted((int(r), int(c)) for r, c in cells))
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0 = min(rs)
    c0 = min(cs)
    return tuple((r - r0, c - c0) for r, c in pts)


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _shape_mask(shape: Shape) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[int(r), int(c)] = True
    return mask


def _shape_patch(shape: Shape, color: int) -> np.ndarray:
    patch = np.zeros(_shape_dims(shape), dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _is_connected(cells: set[Coord] | Shape) -> bool:
    pts = set(cells)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


@lru_cache(maxsize=None)
def _assembly_options(shape_a: Shape, shape_b: Shape) -> tuple[tuple[int, int, Shape], ...]:
    area = len(shape_a) + len(shape_b)
    ha, wa = _shape_dims(shape_a)
    opts: list[tuple[int, int, Shape]] = []

    for out_h in range(1, area + 1):
        if area % out_h != 0:
            continue
        out_w = area // out_h
        if ha > out_h or wa > out_w:
            continue
        full = {(r, c) for r in range(out_h) for c in range(out_w)}
        for top in range(out_h - ha + 1):
            for left in range(out_w - wa + 1):
                placed_a = {(r + top, c + left) for r, c in shape_a}
                rem = full - placed_a
                if len(rem) != len(shape_b):
                    continue
                if _normalize_cells(rem) == shape_b:
                    opts.append((out_h, out_w, tuple(sorted(placed_a))))
    return tuple(opts)


def _assemble_pair(shape_a: Shape, color_a: int, shape_b: Shape, color_b: int) -> np.ndarray | None:
    opts = _assembly_options(shape_a, shape_b)
    if not opts:
        return None
    out_h, out_w, placed_a = min(opts, key=lambda item: (item[0], item[1], item[2]))
    patch = np.full((out_h, out_w), int(color_b), dtype=int)
    for r, c in placed_a:
        patch[int(r), int(c)] = int(color_a)
    return patch


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == 0:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[Coord] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != 0:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            colors = {int(g[r, c]) for r, c in coords}
            assert len(colors) == 1, "each fragment must be monochromatic"
            color = next(iter(colors))
            shape = _normalize_cells(coords)
            rs = [r for r, _c in coords]
            cs = [c for _r, c in coords]
            comps.append(
                {
                    "color": color,
                    "shape": shape,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return comps


def _unique_pairing(comps: list[dict], pair_patches: dict[tuple[int, int], np.ndarray]) -> list[tuple[int, int]]:
    n = len(comps)
    used = [False] * n
    current: list[tuple[int, int]] = []
    solutions: list[list[tuple[int, int]]] = []

    def dfs() -> None:
        if len(solutions) >= 2:
            return
        remaining = [i for i in range(n) if not used[i]]
        if not remaining:
            solutions.append(current.copy())
            return

        def partners(idx: int) -> list[int]:
            return [
                j
                for j in remaining
                if j != idx and (min(idx, j), max(idx, j)) in pair_patches
            ]

        choice = min(remaining, key=lambda idx: len(partners(idx)))
        options = partners(choice)
        if not options:
            return
        for other in options:
            a, b = min(choice, other), max(choice, other)
            used[a] = True
            used[b] = True
            current.append((a, b))
            dfs()
            current.pop()
            used[a] = False
            used[b] = False

    dfs()
    assert len(solutions) == 1, "fragment pairing must be unique"
    return solutions[0]


def _pack_rectangles(patches: list[np.ndarray], width: int) -> np.ndarray:
    positions: list[tuple[int, int, np.ndarray]] = []
    row_y = 0
    row_h = 0
    cursor = 0

    for patch in patches:
        h, w = patch.shape
        assert w <= width, "output rectangle must fit the canvas width"
        needed = w if cursor == 0 else 1 + w
        if cursor > 0 and cursor + needed > width:
            row_y += row_h + 1
            row_h = 0
            cursor = 0
            needed = w
        x = cursor if cursor == 0 else cursor + 1
        positions.append((row_y, x, patch))
        cursor = x + w
        row_h = max(row_h, h)

    out_h = row_y + row_h
    out = np.zeros((out_h, width), dtype=int)
    for top, left, patch in positions:
        h, w = patch.shape
        out[top : top + h, left : left + w] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered monochrome fragments by complementary shape, assemble each pair
    into a solid rectangle, then pack the rectangles by increasing area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_components(g)
    assert comps and len(comps) % 2 == 0

    pair_patches: dict[tuple[int, int], np.ndarray] = {}
    for i, comp_a in enumerate(comps):
        for j in range(i + 1, len(comps)):
            comp_b = comps[j]
            patch = _assemble_pair(comp_a["shape"], comp_a["color"], comp_b["shape"], comp_b["color"])
            if patch is not None:
                pair_patches[(i, j)] = patch

    matching = _unique_pairing(comps, pair_patches)
    rectangles = [
        pair_patches[(i, j)]
        for i, j in matching
    ]
    rectangles.sort(key=lambda patch: patch.shape[0] * patch.shape[1])
    return _pack_rectangles(rectangles, g.shape[1])


def _build_partition_library() -> dict[int, list[dict]]:
    by_area: dict[int, list[dict]] = {}
    for rect_h in range(1, 13):
        for rect_w in range(1, 13):
            area = rect_h * rect_w
            if area not in AREA_CHOICES:
                continue
            cells = [(r, c) for r in range(rect_h) for c in range(rect_w)]
            for area_a in range(2, min(6, area - 2) + 1):
                for combo in itertools.combinations(cells, area_a):
                    part_a = set(combo)
                    part_b = set(cells) - part_a
                    if not (2 <= len(part_b) <= 6):
                        continue
                    if not _is_connected(part_a) or not _is_connected(part_b):
                        continue
                    shape_a = _normalize_cells(part_a)
                    shape_b = _normalize_cells(part_b)
                    opts = _assembly_options(shape_a, shape_b)
                    if len(opts) != 1:
                        continue
                    only_h, only_w, placed_a = opts[0]
                    full_a = tuple(sorted(part_a))
                    if only_h != rect_h or only_w != rect_w or placed_a != full_a:
                        continue
                    by_area.setdefault(area, []).append(
                        {
                            "area": area,
                            "rect_h": rect_h,
                            "rect_w": rect_w,
                            "full_a": full_a,
                            "full_b": tuple(sorted(part_b)),
                        }
                    )
    assert set(AREA_CHOICES).issubset(set(by_area))
    return by_area


PARTITIONS_BY_AREA = _build_partition_library()


def _transform_partition(entry: dict) -> dict:
    patch = np.zeros((entry["rect_h"], entry["rect_w"]), dtype=int)
    for r, c in entry["full_a"]:
        patch[int(r), int(c)] = 1
    for r, c in entry["full_b"]:
        patch[int(r), int(c)] = 2

    patch = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        patch = np.fliplr(patch)

    full_a = tuple(sorted(map(tuple, np.argwhere(patch == 1))))
    full_b = tuple(sorted(map(tuple, np.argwhere(patch == 2))))
    shape_a = _normalize_cells(full_a)
    shape_b = _normalize_cells(full_b)
    opts = _assembly_options(shape_a, shape_b)
    assert len(opts) == 1 and opts[0][0] == patch.shape[0] and opts[0][1] == patch.shape[1]
    assert opts[0][2] == full_a
    return {
        "area": entry["area"],
        "rect_h": int(patch.shape[0]),
        "rect_w": int(patch.shape[1]),
        "full_a": full_a,
        "full_b": full_b,
        "shape_a": shape_a,
        "shape_b": shape_b,
    }


def _assembled_patch(entry: dict, color_a: int, color_b: int) -> np.ndarray:
    patch = np.zeros((entry["rect_h"], entry["rect_w"]), dtype=int)
    for r, c in entry["full_a"]:
        patch[int(r), int(c)] = int(color_a)
    for r, c in entry["full_b"]:
        patch[int(r), int(c)] = int(color_b)
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _place_fragments(height: int, width: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        mask = patch != 0
        ph, pw = patch.shape
        blocked = _dilate8(occupied)
        candidates: list[tuple[int, int]] = []
        for top in range(height - ph + 1):
            for left in range(width - pw + 1):
                if not (blocked[top : top + ph, left : left + pw] & mask).any():
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        view = grid[top : top + ph, left : left + pw]
        view[mask] = patch[mask]
        occupied[top : top + ph, left : left + pw] |= mask
    return grid


def _choose_width(rectangles: list[np.ndarray]) -> int | None:
    widths = [patch.shape[1] for patch in rectangles]
    max_w = max(widths)
    total = sum(widths) + len(widths) - 1
    force_wrap = random.random() < 0.5

    lo, hi = WIDTH_RANGE
    candidates = []
    if force_wrap:
        candidates = [w for w in range(max(lo, max_w), hi + 1) if w < total]
    else:
        candidates = [w for w in range(max(lo, max_w, total), hi + 1)]
        if not candidates:
            candidates = [w for w in range(max(lo, max_w), hi + 1)]
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    areas = random.sample(AREA_CHOICES, n_pairs)

    entries = [_transform_partition(random.choice(PARTITIONS_BY_AREA[area])) for area in areas]
    colors = random.sample(range(1, 10), 2 * n_pairs)

    rectangles: list[np.ndarray] = []
    fragments: list[np.ndarray] = []
    for pair_idx, entry in enumerate(entries):
        color_a = colors[2 * pair_idx]
        color_b = colors[2 * pair_idx + 1]
        if random.random() < 0.5:
            color_a, color_b = color_b, color_a
        rectangles.append(_assembled_patch(entry, color_a, color_b))
        fragments.append(_shape_patch(entry["shape_a"], color_a))
        fragments.append(_shape_patch(entry["shape_b"], color_b))

    rectangles.sort(key=lambda patch: patch.shape[0] * patch.shape[1])
    width = _choose_width(rectangles)
    if width is None:
        return None

    expected = _pack_rectangles(rectangles, width)
    bbox_area = sum(patch.shape[0] * patch.shape[1] for patch in fragments)
    min_height = max(
        HEIGHT_RANGE[0],
        expected.shape[0] + 1,
        math.ceil((bbox_area + 8 * len(fragments)) / width),
    )
    if min_height > HEIGHT_RANGE[1]:
        return None
    height = random.randint(min_height, HEIGHT_RANGE[1])

    inp = _place_fragments(height, width, fragments)
    if inp is None:
        return None

    comps = _find_components(inp)
    if len(comps) != 2 * n_pairs:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
