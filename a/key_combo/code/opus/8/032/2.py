from __future__ import annotations

from collections import Counter, deque

import numpy as np


BACKGROUND = 0
TEMPLATE_COLOR = 5
PALETTE = tuple(color for color in range(10) if color not in (BACKGROUND, TEMPLATE_COLOR))
RING_COUNT = 3
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _component_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _components(np.asarray(grid == int(color), dtype=bool))


def _flood_from_border(open_mask: np.ndarray) -> np.ndarray:
    h, w = open_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    return visited


def _ring_enclosed_cells(grid: np.ndarray, boundary_coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(boundary_coords)
    r0 = max(0, r0 - 1)
    c0 = max(0, c0 - 1)
    r1 = min(grid.shape[0] - 1, r1 + 1)
    c1 = min(grid.shape[1] - 1, c1 + 1)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    wall = np.zeros_like(sub, dtype=bool)
    for rr, cc in boundary_coords:
        wall[int(rr) - r0, int(cc) - c0] = True
    exterior = _flood_from_border(~wall)
    inside = (~wall) & (~exterior)
    local = np.argwhere(inside)
    if local.size == 0:
        return np.empty((0, 2), dtype=int)
    local[:, 0] += r0
    local[:, 1] += c0
    return local.astype(int)


def _template_mask(grid: np.ndarray) -> np.ndarray:
    components = _component_of_color(grid, TEMPLATE_COLOR)
    assert len(components) == 1, "expected exactly one grey template"
    coords = components[0]
    box = _bbox_from_coords(coords)
    crop = _crop_bbox(grid, box)
    mask = crop == TEMPLATE_COLOR
    assert mask.any(), "template crop must be nonempty"
    return mask


def _extract_ring_infos(grid: np.ndarray) -> list[tuple[int, int]]:
    infos: list[tuple[int, int]] = []
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BACKGROUND, TEMPLATE_COLOR)):
        for coords in _component_of_color(grid, color):
            if coords.shape[0] <= 1:
                continue
            enclosed = _ring_enclosed_cells(grid, coords)
            if enclosed.shape[0] == 0:
                continue
            assert enclosed.shape[0] == 1, "each valid ring must enclose exactly one cell"
            rr, cc = map(int, enclosed[0])
            inner_color = int(grid[rr, cc])
            assert inner_color not in (BACKGROUND, color), "ring source must be a non-boundary color"
            top = int(coords[:, 0].min())
            infos.append((top, inner_color))
    assert len(infos) == RING_COUNT, "expected exactly three enclosing loops"
    infos.sort(key=lambda item: item[0])
    assert len({top for top, _ in infos}) == len(infos), "ring top rows must be distinct"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the colors hidden inside closed loops and stamp the cropped grey template in that top-to-bottom order.

    Input structure:
    - A black background contains one grey multi-pixel template on the left.
    - The right side contains three small monochrome closed loops, each enclosing
      exactly one pixel whose color differs from the loop color.
    - Extra isolated single pixels may appear as distractors, but they are not
      enclosed by any loop.

    Rule:
    - Crop the grey template to its tight bounding box and keep only that shape as a stencil.
    - For each closed loop, use topology: ignore the loop color itself and read the single pixel
      trapped inside it as the recolor source.
    - Sort the loops by their topmost boundary row from top to bottom.
    - Recolor one copy of the cropped template for each source color, then stack the copies vertically.

    Key insight:
    - The visible loop color is a decoy; the answer comes from the pixel enclosed by the loop,
      and only the loop positions determine the stamping order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = _template_mask(g)
    ring_infos = _extract_ring_infos(g)

    th, tw = mask.shape
    out = np.full((th * len(ring_infos), tw), BACKGROUND, dtype=int)
    for idx, (_top, color) in enumerate(ring_infos):
        patch = np.full((th, tw), BACKGROUND, dtype=int)
        patch[mask] = int(color)
        out[idx * th : (idx + 1) * th, :] = patch
    return out


def _transform_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in (mask, np.fliplr(mask), np.flipud(mask), np.flipud(np.fliplr(mask))):
        for k in range(4):
            candidate = np.rot90(base, k)
            key = candidate.tobytes() + bytes(candidate.shape)
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
    return variants


def _valid_template(mask: np.ndarray) -> bool:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not (4 <= int(mask.sum()) <= 7):
        return False
    if len(_components(mask)) != 1:
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return (r1 - r0 + 1, c1 - c0 + 1) == mask.shape


def _valid_ring(mask: np.ndarray) -> bool:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if len(_components(mask)) != 1:
        return False
    if np.all(mask):
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != mask.shape:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.array_equal(mask, border):
        return False
    enclosed = (~mask) & (~_flood_from_border(~mask))
    return int(enclosed.sum()) == 1


def _build_template_library() -> list[np.ndarray]:
    bases = [
        np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1]], dtype=bool),
    ]
    library: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in bases:
        for variant in _transform_variants(base):
            if _valid_template(variant):
                key = variant.tobytes() + bytes(variant.shape)
                if key not in seen:
                    seen.add(key)
                    library.append(variant)
    assert library, "template library must be nonempty"
    return library


def _build_ring_library() -> list[np.ndarray]:
    bases = [
        np.array([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1]], dtype=bool),
    ]
    library: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in bases:
        for variant in _transform_variants(base):
            if _valid_ring(variant):
                key = variant.tobytes() + bytes(variant.shape)
                if key not in seen:
                    seen.add(key)
                    library.append(variant)
    assert library, "ring library must be nonempty"
    return library


TEMPLATE_LIBRARY = _build_template_library()
RING_LIBRARY = _build_ring_library()


def _ring_inner_coord(mask: np.ndarray) -> tuple[int, int]:
    enclosed = (~mask) & (~_flood_from_border(~mask))
    coords = np.argwhere(enclosed)
    assert coords.shape == (1, 2)
    return int(coords[0, 0]), int(coords[0, 1])


def _paste_mask(dst: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    dst[top : top + h, left : left + w][mask] = int(color)


def _occupied_with_margin(mask: np.ndarray, top: int, left: int, h: int, w: int, margin: int = 1) -> bool:
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(mask.shape[0], top + h + margin)
    c1 = min(mask.shape[1], left + w + margin)
    return bool(mask[r0:r1, c0:c1].any())


def _singleton_rows(coords: list[tuple[int, int]]) -> list[int]:
    return sorted(r for r, _c in coords)


def _expected_from_parts(template_mask: np.ndarray, ordered_inner_colors: list[int]) -> np.ndarray:
    th, tw = template_mask.shape
    out = np.full((th * len(ordered_inner_colors), tw), BACKGROUND, dtype=int)
    for idx, color in enumerate(ordered_inner_colors):
        out[idx * th : (idx + 1) * th, :][template_mask] = int(color)
    return out


def _wrong_boundary_output(template_mask: np.ndarray, ordered_boundary_colors: list[int]) -> np.ndarray:
    return _expected_from_parts(template_mask, ordered_boundary_colors)


def _count_colors(grid: np.ndarray) -> Counter[int]:
    return Counter(int(x) for x in grid.ravel())


def _semantic_ok(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    counts = _count_colors(grid)
    if counts[BACKGROUND] <= sum(v for k, v in counts.items() if k != BACKGROUND):
        return False

    template_components = _component_of_color(grid, TEMPLATE_COLOR)
    if len(template_components) != 1:
        return False
    template_box = _bbox_from_coords(template_components[0])
    template_mask = _crop_bbox(grid, template_box) == TEMPLATE_COLOR
    if not _valid_template(template_mask):
        return False

    ring_infos: list[tuple[int, int, int]] = []
    for color in range(10):
        if color in (BACKGROUND, TEMPLATE_COLOR) or counts[color] == 0:
            continue
        comps = _component_of_color(grid, color)
        for comp in comps:
            if comp.shape[0] == 1:
                continue
            enclosed = _ring_enclosed_cells(grid, comp)
            if enclosed.shape[0] == 0:
                continue
            if enclosed.shape[0] != 1:
                return False
            rr, cc = map(int, enclosed[0])
            inner_color = int(grid[rr, cc])
            if inner_color in (BACKGROUND, color):
                return False
            ring_infos.append((int(comp[:, 0].min()), color, inner_color))

    if len(ring_infos) != RING_COUNT:
        return False
    ring_infos.sort(key=lambda item: item[0])
    if len({top for top, _boundary, _inner in ring_infos}) != RING_COUNT:
        return False

    inner_colors = [inner for _top, _boundary, inner in ring_infos]
    if len(set(inner_colors)) != RING_COUNT:
        return False
    if any(counts[color] != 1 for color in inner_colors):
        return False

    boundary_colors = [boundary for _top, boundary, _inner in ring_infos]
    if len(set(boundary_colors)) != RING_COUNT:
        return False
    if set(boundary_colors) & set(inner_colors):
        return False

    if np.array_equal(expected, _wrong_boundary_output(template_mask, boundary_colors)):
        return False

    singleton_coords = [
        tuple(map(int, coord))
        for color, count in counts.items()
        if color not in (BACKGROUND, TEMPLATE_COLOR) and count == 1
        for coord in np.argwhere(grid == color)
    ]
    if len(singleton_coords) <= RING_COUNT:
        return False
    true_singleton_rows = sorted(int(np.argwhere(grid == color)[0, 0]) for color in inner_colors)
    all_singleton_rows = _singleton_rows(singleton_coords)
    if all_singleton_rows[:RING_COUNT] == true_singleton_rows:
        return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    template_mask = TEMPLATE_LIBRARY[int(rng.integers(0, len(TEMPLATE_LIBRARY)))]
    ring_masks = [RING_LIBRARY[int(rng.integers(0, len(RING_LIBRARY)))] for _ in range(RING_COUNT)]
    template_h, template_w = map(int, template_mask.shape)
    ring_sizes = [tuple(map(int, mask.shape)) for mask in ring_masks]

    h = int(rng.integers(14, 19))
    w = int(rng.integers(18, 24))
    if max(h, w) > 30:
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    left_zone_w = max(6, w // 3)
    if template_w + 3 > left_zone_w:
        return None
    template_top = int(rng.integers(1, h - template_h - 1))
    template_left = int(rng.integers(1, left_zone_w - template_w))
    _paste_mask(grid, template_mask, template_top, template_left, TEMPLATE_COLOR)
    occupied[template_top : template_top + template_h, template_left : template_left + template_w] |= template_mask

    colors = list(PALETTE)
    rng.shuffle(colors)
    boundary_colors = [int(colors[i]) for i in range(RING_COUNT)]
    inner_colors = [int(colors[i + RING_COUNT]) for i in range(RING_COUNT)]
    distractor_colors = [int(color) for color in colors[2 * RING_COUNT :]]
    distractor_count = int(rng.integers(1, len(distractor_colors) + 1))
    distractor_colors = distractor_colors[:distractor_count]

    max_ring_h = max(rh for rh, _rw in ring_sizes)
    top_candidates = list(range(1, h - max_ring_h - 1))
    rng.shuffle(top_candidates)
    ring_tops: list[int] = []
    for candidate in top_candidates:
        if all(abs(candidate - existing) >= max_ring_h + 1 for existing in ring_tops):
            ring_tops.append(int(candidate))
            if len(ring_tops) == RING_COUNT:
                break
    if len(ring_tops) != RING_COUNT:
        return None
    ring_tops.sort()

    right_min = left_zone_w + 2
    if right_min >= w - 4:
        return None

    ring_lefts: list[int] = []
    placed_rings: list[tuple[int, int, np.ndarray, int, int]] = []
    for idx, (mask, top, boundary_color, inner_color) in enumerate(
        zip(ring_masks, ring_tops, boundary_colors, inner_colors)
    ):
        rh, rw = map(int, mask.shape)
        options = list(range(right_min, w - rw - 1))
        rng.shuffle(options)
        chosen_left = None
        for left in options:
            if _occupied_with_margin(occupied, top, left, rh, rw, margin=1):
                continue
            chosen_left = int(left)
            break
        if chosen_left is None:
            return None
        ring_lefts.append(chosen_left)
        placed_rings.append((top, chosen_left, mask, boundary_color, inner_color))
        occupied[top : top + rh, chosen_left : chosen_left + rw] |= mask

    if ring_lefts == sorted(ring_lefts):
        return None

    for top, left, mask, boundary_color, inner_color in placed_rings:
        _paste_mask(grid, mask, top, left, boundary_color)
        inner_r, inner_c = _ring_inner_coord(mask)
        if occupied[top + inner_r, left + inner_c]:
            return None
        grid[top + inner_r, left + inner_c] = inner_color
        occupied[top + inner_r, left + inner_c] = True

    ring_inner_rows = sorted(top + _ring_inner_coord(mask)[0] for top, _left, mask, _b, _i in placed_rings)
    free_cells = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(right_min, w - 1)
        if not _occupied_with_margin(occupied, r, c, 1, 1, margin=1)
    ]
    rng.shuffle(free_cells)
    distractor_coords: list[tuple[int, int]] = []
    for color in distractor_colors:
        chosen = None
        for idx, (r, c) in enumerate(free_cells):
            trial_rows = sorted(ring_inner_rows + [r])
            if trial_rows[:RING_COUNT] == ring_inner_rows:
                continue
            chosen = (r, c)
            free_cells.pop(idx)
            break
        if chosen is None:
            return None
        r, c = chosen
        grid[r, c] = color
        occupied[r, c] = True
        distractor_coords.append(chosen)

    ordered_inner_colors = [inner for _top, _left, _mask, _boundary, inner in sorted(placed_rings, key=lambda item: item[0])]
    expected = _expected_from_parts(template_mask, ordered_inner_colors)

    if max(expected.shape) > 30:
        return None
    if not distractor_coords:
        return None
    return grid if _semantic_ok(grid, expected) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
