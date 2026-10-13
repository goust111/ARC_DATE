import math
from collections import Counter

import numpy as np


BG = 0
SEPARATOR = 5
POINTER_COLORS = (1, 2, 3)  # select 1st / 2nd / 3rd boundary-rank color
POINTER_TO_RANK = {color: rank for rank, color in enumerate(POINTER_COLORS)}
TEMPLATE_COLORS = (4, 6, 7, 8, 9)
MAX_RETRIES = 400
RNG = np.random.default_rng()

SHAPE_LIBRARY = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def find_separator_col(grid: np.ndarray, bg: int) -> int:
    cands = []
    for c in range(grid.shape[1]):
        col_vals = np.unique(grid[:, c])
        if len(col_vals) == 1 and int(col_vals[0]) != bg:
            cands.append(int(c))
    assert len(cands) == 1, "expected exactly one full-height separator column"
    return cands[0]


def boundary_positions(
    hole_coords: list[tuple[int, int]], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    hole_set = set(hole_coords)
    seen = set()
    out: list[tuple[int, int]] = []
    for r, c in hole_coords:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in hole_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                out.append((int(nr), int(nc)))
    return out


def ranked_boundary_colors(
    grid: np.ndarray, hole_coords: list[tuple[int, int]], bg: int
) -> list[int]:
    boundary = [p for p in boundary_positions(hole_coords, grid.shape) if int(grid[p]) != bg]
    counts = Counter(int(grid[r, c]) for r, c in boundary)
    assert len(counts) >= 2, "each hole must touch at least two colors"
    boundary_sorted = sorted(boundary)
    first_pos: dict[int, int] = {}
    for idx, (r, c) in enumerate(boundary_sorted):
        first_pos.setdefault(int(grid[r, c]), idx)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], first_pos[item[0]]))
    return [int(color) for color, _count in ordered]


def transform_shape(
    coords: list[tuple[int, int]], rot: int, flip: bool
) -> list[tuple[int, int]]:
    pts = np.array(coords, dtype=int)
    if flip:
        pts[:, 1] *= -1
    for _ in range(rot % 4):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    pts[:, 0] -= np.min(pts[:, 0])
    pts[:, 1] -= np.min(pts[:, 1])
    return [(int(r), int(c)) for r, c in pts.tolist()]


def chebyshev_dilate(coords: set[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((int(r + dr), int(c + dc)))
    return out


def inside_rect(cell: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = rect
    return r0 <= r < r1 and c0 <= c < c1


def distinct_quotas(total: int) -> list[int]:
    assert total >= 6
    for _ in range(200):
        cuts = sorted(RNG.choice(np.arange(1, total), size=2, replace=False).tolist())
        parts = [cuts[0], cuts[1] - cuts[0], total - cuts[1]]
        if min(parts) > 0 and len(set(parts)) == 3:
            return sorted([int(x) for x in parts], reverse=True)
    raise RuntimeError("failed to split boundary into three distinct quotas")


def order_boundary_cycle(
    hole_coords: list[tuple[int, int]], boundary: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    center_r = float(sum(r for r, _ in hole_coords)) / len(hole_coords)
    center_c = float(sum(c for _, c in hole_coords)) / len(hole_coords)
    ordered = sorted(
        boundary,
        key=lambda p: (math.atan2(p[0] - center_r, p[1] - center_c), p[0], p[1]),
    )
    shift = int(RNG.integers(0, len(ordered)))
    return ordered[shift:] + ordered[:shift]


def render_shape_patch(coords: list[tuple[int, int]], color: int) -> np.ndarray:
    key = normalize_coords(coords)
    h = max(r for r, _ in key) + 1
    w = max(c for _, c in key) + 1
    patch = np.full((h, w), BG, dtype=int)
    for r, c in key:
        patch[r, c] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble torn holes in a multicolor template by matching fragment shapes, then fill each hole via boundary-color ranking.

    Input structure:
    - bg = most common color.
    - One full-height grey separator column splits the grid into a larger left template area and a smaller right fragment area.
    - The left area contains one large connected multicolor figure with several enclosed bg holes.
    - The right area contains isolated monochrome fragments whose exact shapes match the holes by translation only.

    Rule:
    - Find each enclosed hole in the left area and each fragment in the right area.
    - Match fragment to hole by exact normalized shape, without rotation or reflection.
    - For the matched hole, collect the distinct non-bg colors on its 4-neighbor boundary and rank them by how many boundary cells of each color touch the hole.
    - The fragment's dedicated pointer color selects the 1st, 2nd, or 3rd ranked boundary color; fill the whole hole with that selected color.
    - Return only the repaired left area, excluding the separator and the fragment area.

    Key insight:
    - Shape identifies where a fragment belongs, but fragment color does not paint directly; it indexes a rank inside the hole's visible boundary-color distribution.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sep_c = find_separator_col(g, bg)

    left = g[:, :sep_c].copy()
    right = g[:, sep_c + 1 :]

    holes = [
        comp
        for comp in components_mask(left == bg)
        if not touches_border(comp, left.shape)
    ]
    fragments = []
    for comp in components_mask(right != bg):
        colors = {int(right[r, c]) for r, c in comp}
        assert len(colors) == 1, "each fragment must be monochrome"
        pointer_color = colors.pop()
        assert pointer_color in POINTER_TO_RANK, "unexpected pointer color"
        fragments.append((normalize_coords(comp), int(pointer_color)))

    assert holes and len(holes) == len(fragments), "holes and fragments must pair up"
    hole_by_shape: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for hole in holes:
        key = normalize_coords(hole)
        assert key not in hole_by_shape, "hole shapes must be unique"
        hole_by_shape[key] = hole

    for shape_key, pointer_color in fragments:
        assert shape_key in hole_by_shape, "fragment shape must match a hole"
        hole = hole_by_shape[shape_key]
        ordered_colors = ranked_boundary_colors(left, hole, bg)
        rank = POINTER_TO_RANK[pointer_color]
        assert rank < len(ordered_colors), "pointer rank exceeds boundary-color count"
        fill_color = ordered_colors[rank]
        for r, c in hole:
            left[r, c] = int(fill_color)

    return left


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(20, 27))
    left_w = int(RNG.integers(14, 19))
    right_w = int(RNG.integers(6, 9))
    total_w = left_w + 1 + right_w
    if total_w > 30:
        return None

    left_out = np.full((h, left_w), BG, dtype=int)
    top_margin = int(RNG.integers(2, 4))
    bottom_margin = int(RNG.integers(2, 4))
    left_margin = int(RNG.integers(2, 4))
    right_margin = int(RNG.integers(2, 4))
    shape_rect = (
        top_margin,
        left_margin,
        h - bottom_margin,
        left_w - right_margin,
    )
    r0, c0, r1, c1 = shape_rect
    shape_h = r1 - r0
    shape_w = c1 - c0
    if shape_h < 12 or shape_w < 9:
        return None

    base_color = int(RNG.choice(TEMPLATE_COLORS))
    left_out[r0:r1, c0:c1] = base_color

    hole_count = 3
    selected_holes = []
    used_shape_keys = set()
    attempts = 0
    while len(selected_holes) < hole_count and attempts < 200:
        attempts += 1
        base_shape = SHAPE_LIBRARY[int(RNG.integers(0, len(SHAPE_LIBRARY)))]
        coords = transform_shape(
            base_shape,
            rot=int(RNG.integers(0, 4)),
            flip=bool(int(RNG.integers(0, 2))),
        )
        key = normalize_coords(coords)
        if key in used_shape_keys:
            continue
        used_shape_keys.add(key)
        selected_holes.append(coords)
    if len(selected_holes) != hole_count:
        return None

    placement_order = RNG.permutation(np.arange(hole_count)).tolist()
    rank_order = RNG.permutation(np.arange(hole_count)).tolist()
    ordered_holes = [selected_holes[i] for i in placement_order]
    ordered_ranks = [rank_order[i] for i in placement_order]
    hole_specs = []
    hole_heights = [max(r for r, _ in normalize_coords(coords)) + 1 for coords in ordered_holes]
    current_top = r0 + 2

    for hole_idx, coords in enumerate(ordered_holes):
        key = normalize_coords(coords)
        hole_h = hole_heights[hole_idx]
        hole_w = max(c for _, c in key) + 1

        remaining_heights = sum(hole_heights[hole_idx + 1 :])
        remaining_gaps = 2 * (hole_count - hole_idx - 1)
        latest_top = r1 - 2 - hole_h - remaining_heights - remaining_gaps
        if latest_top < current_top:
            return None
        pr = int(RNG.integers(current_top, latest_top + 1))
        min_pc = c0 + 2
        max_pc = c1 - hole_w - 1
        if max_pc < min_pc:
            return None
        pc = int(RNG.integers(min_pc, max_pc + 1))
        hole_cells = {(pr + rr, pc + cc) for rr, cc in key}
        boundary = boundary_positions(list(hole_cells), left_out.shape)
        if any(not inside_rect(p, shape_rect) for p in boundary):
            return None
        current_top = pr + hole_h + 2

        boundary = boundary_positions(list(hole_cells), left_out.shape)
        if len(boundary) < 6:
            return None
        cycle = order_boundary_cycle(list(hole_cells), boundary)
        quotas = distinct_quotas(len(cycle))
        boundary_colors = [int(x) for x in RNG.choice(TEMPLATE_COLORS, size=3, replace=False)]
        start = 0
        for color, quota in zip(boundary_colors, quotas):
            for cell in cycle[start : start + quota]:
                left_out[cell] = int(color)
            start += quota

        target_rank = int(ordered_ranks[hole_idx])
        fill_color = boundary_colors[target_rank]
        for hr, hc in hole_cells:
            left_out[hr, hc] = int(fill_color)

        counts = Counter(int(left_out[br, bc]) for br, bc in boundary)
        if len(counts) != 3 or len(set(counts.values())) != 3:
            return None

        hole_specs.append(
            {
                "hole_cells": sorted((int(r), int(c)) for r, c in hole_cells),
                "shape_key": key,
                "pointer_color": POINTER_COLORS[target_rank],
            }
        )

    if len(components_mask(left_out != BG)) != 1:
        return None

    left_in = left_out.copy()
    for spec in hole_specs:
        for r, c in spec["hole_cells"]:
            left_in[r, c] = BG

    full = np.full((h, total_w), BG, dtype=int)
    full[:, :left_w] = left_in
    full[:, left_w] = SEPARATOR

    right = full[:, left_w + 1 :]
    fragment_order = RNG.permutation(np.arange(len(hole_specs))).tolist()
    ordered_specs = [hole_specs[i] for i in fragment_order]
    frag_heights = []
    for spec in ordered_specs:
        patch = render_shape_patch(list(spec["shape_key"]), int(spec["pointer_color"]))
        frag_heights.append(int(patch.shape[0]))

    current_top = 0
    for idx, spec in enumerate(ordered_specs):
        patch = render_shape_patch(list(spec["shape_key"]), int(spec["pointer_color"]))
        ph, pw = patch.shape
        remaining_heights = sum(frag_heights[idx + 1 :])
        remaining_gaps = len(ordered_specs) - idx - 1
        latest_top = h - ph - remaining_heights - remaining_gaps
        if latest_top < current_top:
            return None
        rr = int(RNG.integers(current_top, latest_top + 1))
        cc = int(RNG.integers(0, right_w - pw + 1))
        for pr, pc in np.argwhere(patch != BG):
            right[rr + pr, cc + pc] = int(spec["pointer_color"])
        current_top = rr + ph + 1

    if get_bg(full) != BG:
        return None

    expected = left_out
    actual = main(full)
    if not np.array_equal(actual, expected):
        return None

    sig_holes = [normalize_coords(spec["hole_cells"]) for spec in hole_specs]
    if len(set(sig_holes)) != hole_count:
        return None

    return full


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for seed720058768/2")
