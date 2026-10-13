import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
FRAGMENT_COLORS = (2, 3, 4, 6, 7, 8, 9)
GRID_SIDE_RANGE = (18, 24)
HALF_HEIGHT_RANGE = (5, 8)
HALF_WIDTH_RANGE = (3, 5)
HALF_AREA_RANGE = (8, 15)
FRAGMENT_COUNT_RANGE = (2, 4)
FRAGMENT_SIZE_RANGE = (2, 5)
GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords) -> tuple[int, int, int, int]:
    pts = list(coords)
    rs = [int(r) for r, _c in pts]
    cs = [int(c) for _r, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _shift(coords, dr: int, dc: int):
    return frozenset((int(r) + int(dr), int(c) + int(dc)) for r, c in coords)


def _normalize(coords):
    r0, c0, _r1, _c1 = _bbox(coords)
    return frozenset((int(r) - r0, int(c) - c0) for r, c in coords)


def _components_from_mask(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(frozenset(cells))
    return comps


def _find_components(grid: np.ndarray, colors):
    comps = []
    for color in colors:
        for coords in _components_from_mask(grid == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _reconstruct_template(blue_coords, grid_width: int):
    r0, c0, r1, c1 = _bbox(blue_coords)
    center_c = (int(grid_width) - 1) / 2.0
    d_left = abs(c0 - center_c)
    d_right = abs(c1 - center_c)
    assert d_left != d_right, "template axis must be uniquely determined"
    axis = int(c0 if d_left < d_right else c1)
    shown_side = "right" if axis == c0 else "left"
    if shown_side == "left":
        assert all(int(c) <= axis for _r, c in blue_coords)
    else:
        assert all(int(c) >= axis for _r, c in blue_coords)

    full = set((int(r), int(c)) for r, c in blue_coords)
    for r, c in blue_coords:
        mc = 2 * axis - int(c)
        assert 0 <= mc < int(grid_width)
        full.add((int(r), int(mc)))
    return frozenset(full), axis, shown_side


def _matching_shifts(fragment_coords, template_coords, shape: tuple[int, int]):
    template = frozenset((int(r), int(c)) for r, c in template_coords)
    fragment = frozenset((int(r), int(c)) for r, c in fragment_coords)
    h, w = map(int, shape)
    first_r, first_c = next(iter(fragment))
    shifts = []
    for tr, tc in template:
        dr = first_r - int(tr)
        dc = first_c - int(tc)
        placed = _shift(template, dr, dc)
        if any(r < 0 or c < 0 or r >= h or c >= w for r, c in placed):
            continue
        if fragment <= placed:
            shifts.append((int(dr), int(dc)))
    return sorted(set(shifts))


def _shape_occurrences(shape_coords, template_coords):
    shape = frozenset((int(r), int(c)) for r, c in shape_coords)
    template = frozenset((int(r), int(c)) for r, c in template_coords)
    anchor_r, anchor_c = next(iter(shape))
    out = []
    for tr, tc in template:
        dr = int(tr) - anchor_r
        dc = int(tc) - anchor_c
        placed = _shift(shape, dr, dc)
        if placed <= template:
            out.append(placed)
    return out


def _connected_subsets(coords, max_size: int):
    cells = sorted((int(r), int(c)) for r, c in coords)
    cell_set = set(cells)
    seen = set()
    for start in cells:
        subset = frozenset([start])
        if subset in seen:
            continue
        seen.add(subset)
        stack = [subset]
        while stack:
            cur = stack.pop()
            if 2 <= len(cur) <= max_size:
                yield cur
            if len(cur) == max_size:
                continue
            frontier = set()
            for r, c in cur:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cell_set and (nr, nc) not in cur:
                        frontier.add((nr, nc))
            for nxt in sorted(frontier):
                grown = frozenset(set(cur) | {nxt})
                if grown not in seen:
                    seen.add(grown)
                    stack.append(grown)


def _random_half_template():
    for _ in range(200):
        h = random.randint(*HALF_HEIGHT_RANGE)
        half_w = random.randint(*HALF_WIDTH_RANGE)
        area = random.randint(max(HALF_AREA_RANGE[0], h + 1), min(HALF_AREA_RANGE[1], h * half_w))
        axis_c = half_w - 1
        start = (random.randrange(h), axis_c)
        cells = {start}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            nbrs = [
                (nr, nc)
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < half_w and (nr, nc) not in cells
            ]
            if not nbrs:
                continue
            cells.add(random.choice(nbrs))
        if not any(c == 0 for _r, c in cells):
            continue
        if not any(c == axis_c for _r, c in cells):
            continue
        if not any(r == 0 for r, _c in cells):
            continue
        if not any(r == h - 1 for r, _c in cells):
            continue
        left_half = frozenset((int(r), int(c)) for r, c in cells)
        full = set(left_half)
        for r, c in left_half:
            full.add((int(r), 2 * axis_c - int(c)))
        if len(_components_from_mask(_coords_to_mask(full))) != 1:
            continue
        return left_half, frozenset(full), h, half_w
    return None


def _coords_to_mask(coords):
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _eligible_fragments(full_local, visible_local, hidden_local):
    eligible = []
    for subset in _connected_subsets(full_local, FRAGMENT_SIZE_RANGE[1]):
        if len(subset) < FRAGMENT_SIZE_RANGE[0]:
            continue
        if not (subset & hidden_local):
            continue
        shape = _normalize(subset)
        if len(_shape_occurrences(shape, full_local)) != 1:
            continue
        if len(_shape_occurrences(shape, visible_local)) != 0:
            continue
        eligible.append(subset)
    eligible.sort(key=lambda s: (len(s), sorted(s)))
    return eligible


def _boxes_overlap(a, b, gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    )


def _placeable_positions(shape: tuple[int, int], template_size: tuple[int, int], count: int, forbidden_bboxes):
    h, w = map(int, shape)
    th, tw = map(int, template_size)
    candidates = [(r, c) for r in range(1, h - th) for c in range(1, w - tw)]
    random.shuffle(candidates)
    chosen = []
    occupied = list(forbidden_bboxes)
    for r0, c0 in candidates:
        bbox = (r0, c0, r0 + th - 1, c0 + tw - 1)
        if any(_boxes_overlap(bbox, other, gap=1) for other in occupied):
            continue
        chosen.append((int(r0), int(c0)))
        occupied.append(bbox)
        if len(chosen) == count:
            return chosen
    return None


def _paint(coords, color: int, grid: np.ndarray):
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden half of the blue symmetric template, then use it to complete every colored fragment into a translated full copy.

    Input structure:
    - A black background contains one blue 4-connected half-template and 2 to 4 smaller same-color fragments in distinct non-blue colors.
    - The blue object is exactly one left or right half of a larger horizontally symmetric template and includes the symmetry-axis column.
    - Every fragment is a connected subset of that full template at some translated location, and at least one fragment cell comes from the template half hidden in the blue input object.

    Rule:
    - Find the blue object's bounding box and take the vertical bbox edge closer to the grid center as the symmetry axis.
    - Reflect the blue half across that axis and transparently union the reflection with the original blue cells to reconstruct the full blue template.
    - For each non-blue fragment, find the unique translation whose full template copy contains all observed fragment cells, then fill the missing template cells at that location with the fragment's color.
    - Leave the background black, replace the blue half-template by the full blue template, and keep each fragment at its matched translated template position.

    Key insight:
    - The blue shape is deliberately incomplete; only after mirroring it back to the full symmetric template do the scattered fragment anchors admit unique translated matches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == BLACK

    blue_comps = _find_components(g, [BLUE])
    assert len(blue_comps) == 1, "expected one blue half-template"
    blue = blue_comps[0]
    full_template, _axis, _side = _reconstruct_template(blue["coords"], g.shape[1])

    fragment_colors = [int(c) for c in np.unique(g) if int(c) not in (bg, BLUE)]
    assert 2 <= len(fragment_colors) <= 4
    fragments = _find_components(g, fragment_colors)
    assert len(fragments) == len(fragment_colors), "each fragment color must appear in exactly one component"
    fragments.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    out = np.full_like(g, bg)
    _paint(full_template, BLUE, out)
    occupied = set(full_template)

    for comp in fragments:
        shifts = _matching_shifts(comp["coords"], full_template, g.shape)
        assert len(shifts) == 1, "fragment placement must be unique"
        dr, dc = shifts[0]
        placed = _shift(full_template, dr, dc)
        assert occupied.isdisjoint(placed), "completed objects must not overlap"
        _paint(placed, comp["color"], out)
        occupied |= set(placed)

    return out


def _build_instance():
    template_info = _random_half_template()
    if template_info is None:
        return None
    left_half, full_local, template_h, half_w = template_info
    axis_local = half_w - 1
    shown_left = random.choice([True, False])
    if shown_left:
        visible_local = frozenset((r, c) for r, c in full_local if c <= axis_local)
        hidden_local = frozenset((r, c) for r, c in full_local if c > axis_local)
    else:
        visible_local = frozenset((r, c) for r, c in full_local if c >= axis_local)
        hidden_local = frozenset((r, c) for r, c in full_local if c < axis_local)
    eligible = _eligible_fragments(full_local, visible_local, hidden_local)
    if len(eligible) < 3:
        return None

    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    full_w = 2 * half_w - 1
    if h <= template_h + 3 or w <= full_w + 3:
        return None

    center_c = (w - 1) / 2.0
    blue_positions = []
    for r0 in range(1, h - template_h):
        for c0 in range(1, w - full_w):
            axis_global = c0 + axis_local
            if shown_left and axis_global < center_c:
                blue_positions.append((r0, c0))
            if (not shown_left) and axis_global > center_c:
                blue_positions.append((r0, c0))
    if not blue_positions:
        return None
    blue_r0, blue_c0 = random.choice(blue_positions)
    blue_full = _shift(full_local, blue_r0, blue_c0)
    blue_visible = _shift(visible_local, blue_r0, blue_c0)
    blue_bbox = (blue_r0, blue_c0, blue_r0 + template_h - 1, blue_c0 + full_w - 1)

    frag_count = random.randint(*FRAGMENT_COUNT_RANGE)
    placements = _placeable_positions((h, w), (template_h, full_w), frag_count, [blue_bbox])
    if placements is None:
        return None

    colors = random.sample(FRAGMENT_COLORS, frag_count)
    grid = np.zeros((h, w), dtype=int)
    _paint(blue_visible, BLUE, grid)

    expected = np.zeros_like(grid)
    _paint(blue_full, BLUE, expected)

    fragments = []
    used_output = set(blue_full)
    for color, (r0, c0) in zip(colors, placements):
        full_copy = _shift(full_local, r0, c0)
        if used_output & set(full_copy):
            return None
        subset_local = random.choice(eligible)
        subset_global = _shift(subset_local, r0, c0)
        _paint(subset_global, color, grid)
        _paint(full_copy, color, expected)
        used_output |= set(full_copy)
        fragments.append(
            {
                "color": int(color),
                "subset_local": subset_local,
                "subset_global": subset_global,
                "full_copy": full_copy,
            }
        )

    meta = {
        "full_local": full_local,
        "visible_local": visible_local,
        "hidden_local": hidden_local,
        "blue_full": blue_full,
        "blue_visible": blue_visible,
        "fragments": fragments,
        "expected": expected,
        "shown_left": shown_left,
    }
    if not _validate_instance(grid, meta):
        return None
    return grid


def _validate_instance(grid: np.ndarray, meta) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if get_bg(grid) != BLACK:
        return False

    blue_comps = _find_components(grid, [BLUE])
    if len(blue_comps) != 1:
        return False
    rebuilt, _axis, _side = _reconstruct_template(blue_comps[0]["coords"], grid.shape[1])
    if rebuilt != meta["blue_full"]:
        return False

    visible = _normalize(meta["visible_local"])
    full = _normalize(meta["full_local"])
    hidden = meta["hidden_local"]
    fragment_colors = [frag["color"] for frag in meta["fragments"]]
    comps = _find_components(grid, fragment_colors)
    if len(comps) != len(fragment_colors):
        return False

    occupied = set(meta["blue_full"])
    for frag in meta["fragments"]:
        if not (frag["subset_local"] & hidden):
            return False
        shape = _normalize(frag["subset_local"])
        if len(_shape_occurrences(shape, full)) != 1:
            return False
        if len(_shape_occurrences(shape, visible)) != 0:
            return False
        shifts = _matching_shifts(frag["subset_global"], meta["blue_full"], grid.shape)
        if shifts != [(
            min(r for r, _c in frag["full_copy"]) - min(r for r, _c in meta["blue_full"]),
            min(c for _r, c in frag["full_copy"]) - min(c for _r, c in meta["blue_full"]),
        )]:
            return False
        if occupied & set(frag["full_copy"]):
            return False
        occupied |= set(frag["full_copy"])

    return np.array_equal(main(grid), meta["expected"])


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-template completion input")
