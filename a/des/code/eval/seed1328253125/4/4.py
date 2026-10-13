from __future__ import annotations

import numpy as np


Coord = tuple[int, int]
DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


_RNG = np.random.default_rng()

GEN_ATTEMPTS = 400
PARTITION_ATTEMPTS = 200
PAIR_COUNT_RANGE = (2, 5)  # [2, 4]
RECT_H_RANGE = (2, 7)      # [2, 6]
RECT_W_RANGE = (2, 7)      # [2, 6]
GRID_H_RANGE = (14, 27)    # [14, 26]
GRID_W_RANGE = (14, 31)    # [14, 30]
PALETTE = tuple(range(10))
HALO = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(cell: Coord, shape: tuple[int, int]) -> list[Coord]:
    r, c = cell
    h, w = shape
    out: list[Coord] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors((rr, cc), mask.shape):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(_components(mask)) == 1


def _is_full_rectangle(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    return int(mask.sum()) == int(mask.size)


def _valid_rectangle_translations(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int, int, int]]:
    a_coords = [tuple(map(int, coord)) for coord in np.argwhere(mask_a)]
    b_coords = [tuple(map(int, coord)) for coord in np.argwhere(mask_b)]
    a_set = set(a_coords)
    total_area = len(a_coords) + len(b_coords)
    limit = total_area

    valid: list[tuple[int, int, int, int]] = []
    for dr in range(-limit, limit + 1):
        for dc in range(-limit, limit + 1):
            shifted_b = {(r + dr, c + dc) for r, c in b_coords}
            if a_set & shifted_b:
                continue
            union = a_set | shifted_b
            rows = [r for r, _c in union]
            cols = [c for _r, c in union]
            height = max(rows) - min(rows) + 1
            width = max(cols) - min(cols) + 1
            if height * width == total_area:
                valid.append((dr, dc, int(height), int(width)))
    return valid


def _rectangle_dims_from_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    valid = _valid_rectangle_translations(mask_a, mask_b)
    assert valid, "pair does not reconstruct any rectangle"
    dims = {(h, w) for _dr, _dc, h, w in valid}
    assert len(dims) == 1, "pair reconstructs multiple rectangle sizes"
    return next(iter(dims))


def _render_output(rectangles: list[tuple[int, int, int]], bg: int) -> np.ndarray:
    assert rectangles
    out_h = max(h for h, _w, _color in rectangles)
    out_w = sum(w for _h, w, _color in rectangles) + max(0, len(rectangles) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    col = 0
    for h, w, color in rectangles:
        out[:h, col : col + w] = int(color)
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments, reassemble each pair into a solid rectangle,
    sort the rectangles by area, then pack them left-to-right top-aligned.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    rectangles: list[tuple[int, int, int, tuple[int, int]]] = []

    for color in [int(v) for v in np.unique(g) if int(v) != int(bg)]:
        comps = _components(g == color)
        assert len(comps) == 2, "each foreground color must appear in exactly two components"
        masks = [_coords_to_mask(comp) for comp in comps]
        height, width = _rectangle_dims_from_pair(masks[0], masks[1])
        anchor = min((int(comp[:, 0].min()), int(comp[:, 1].min())) for comp in comps)
        rectangles.append((int(height), int(width), int(color), anchor))

    assert rectangles, "expected at least one color pair"
    rectangles.sort(key=lambda item: (item[0] * item[1], item[0], item[1], item[3][0], item[3][1]))
    return _render_output([(h, w, color) for h, w, color, _anchor in rectangles], bg)


def _sample_partition(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    total = h * w
    cells = [(r, c) for r in range(h) for c in range(w)]
    min_piece_area = 2 if total <= 8 else 3

    for _ in range(PARTITION_ATTEMPTS):
        seed_indices = rng.choice(len(cells), size=2, replace=False)
        seed_a = cells[int(seed_indices[0])]
        seed_b = cells[int(seed_indices[1])]
        target_a = int(rng.integers(min_piece_area, total - min_piece_area + 1))
        target_sizes = [target_a, total - target_a]

        regions = [{seed_a}, {seed_b}]
        assigned = {seed_a, seed_b}
        frontiers = [set(), set()]
        for idx, seed in enumerate((seed_a, seed_b)):
            for nbr in _neighbors(seed, (h, w)):
                if nbr not in assigned:
                    frontiers[idx].add(nbr)

        failed = False
        while len(assigned) < total:
            for frontier in frontiers:
                frontier.difference_update(assigned)
            choices = [idx for idx in (0, 1) if frontiers[idx]]
            if not choices:
                failed = True
                break

            if len(choices) == 1:
                chosen = choices[0]
            else:
                remaining = np.array(
                    [max(1, target_sizes[idx] - len(regions[idx])) for idx in choices],
                    dtype=float,
                )
                remaining /= remaining.sum()
                pick = int(rng.choice(len(choices), p=remaining))
                chosen = choices[pick]

            frontier_list = list(frontiers[chosen])
            cell = frontier_list[int(rng.integers(len(frontier_list)))]
            regions[chosen].add(cell)
            assigned.add(cell)
            frontiers[chosen].discard(cell)
            for nbr in _neighbors(cell, (h, w)):
                if nbr not in assigned:
                    frontiers[chosen].add(nbr)

        if failed:
            continue

        full_a = np.zeros((h, w), dtype=bool)
        full_b = np.zeros((h, w), dtype=bool)
        for cell in regions[0]:
            full_a[cell] = True
        for cell in regions[1]:
            full_b[cell] = True

        if not (_is_connected(full_a) and _is_connected(full_b)):
            continue

        mask_a = _trim_mask(full_a)
        mask_b = _trim_mask(full_b)
        if _is_full_rectangle(mask_a) and _is_full_rectangle(mask_b):
            continue

        valid = _valid_rectangle_translations(mask_a, mask_b)
        if len(valid) != 1:
            continue
        _dr, _dc, out_h, out_w = valid[0]
        if (int(out_h), int(out_w)) != (int(h), int(w)):
            continue
        return mask_a, mask_b
    return None


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, int(r) - HALO)
        r1 = min(h, int(r) + HALO + 1)
        c0 = max(0, int(c) - HALO)
        c1 = min(w, int(c) + HALO + 1)
        blocked[r0:r1, c0:c1] = True


def _free_locations(mask: np.ndarray, blocked: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    occ = np.argwhere(mask)
    grid_h, grid_w = shape
    h, w = mask.shape
    locations: list[tuple[int, int]] = []
    for top in range(grid_h - h + 1):
        for left in range(grid_w - w + 1):
            if not bool(blocked[top + occ[:, 0], left + occ[:, 1]].any()):
                locations.append((int(top), int(left)))
    return locations


def _place_pieces(
    pieces: list[tuple[np.ndarray, int]],
    grid_shape: tuple[int, int],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    grid = np.full(grid_shape, int(bg), dtype=int)
    blocked = np.zeros(grid_shape, dtype=bool)

    order = sorted(range(len(pieces)), key=lambda idx: int(pieces[idx][0].sum()), reverse=True)
    for idx in order:
        mask, color = pieces[idx]
        locations = _free_locations(mask, blocked, grid_shape)
        if not locations:
            return None
        top, left = locations[int(rng.integers(len(locations)))]
        occ = np.argwhere(mask)
        placed = occ + np.array([top, left], dtype=int)
        grid[placed[:, 0], placed[:, 1]] = int(color)
        _mark_blocked(blocked, placed)
    return grid


def _sample_pairs(
    pair_count: int,
    bg: int,
    rng: np.random.Generator,
) -> list[dict[str, object]] | None:
    colors = [color for color in PALETTE if color != bg]
    rng.shuffle(colors)
    colors = colors[:pair_count]

    pairs: list[dict[str, object]] = []
    used_areas: set[int] = set()
    for color in colors:
        built = False
        for _ in range(PARTITION_ATTEMPTS):
            rect_h = int(rng.integers(*RECT_H_RANGE))
            rect_w = int(rng.integers(*RECT_W_RANGE))
            rect_area = rect_h * rect_w
            if rect_area in used_areas or rect_area < 6:
                continue
            partition = _sample_partition(rect_h, rect_w, rng)
            if partition is None:
                continue
            piece_a, piece_b = partition
            pairs.append(
                {
                    "color": int(color),
                    "dims": (rect_h, rect_w),
                    "pieces": [piece_a, piece_b],
                }
            )
            used_areas.add(rect_area)
            built = True
            break
        if not built:
            return None
    return pairs


def _expected_from_pairs(pairs: list[dict[str, object]], bg: int) -> np.ndarray:
    rectangles = [
        (int(pair["dims"][0]), int(pair["dims"][1]), int(pair["color"]))  # type: ignore[index]
        for pair in pairs
    ]
    rectangles.sort(key=lambda item: item[0] * item[1])
    return _render_output(rectangles, bg)


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    bg = int(rng.integers(10))
    pair_count = int(rng.integers(*PAIR_COUNT_RANGE))
    pairs = _sample_pairs(pair_count, bg, rng)
    if pairs is None:
        return None

    pieces: list[tuple[np.ndarray, int]] = []
    total_fg = 0
    max_rect_area = 0
    max_piece_h = 0
    max_piece_w = 0
    for pair in pairs:
        color = int(pair["color"])  # type: ignore[index]
        rect_h, rect_w = pair["dims"]  # type: ignore[index]
        total_fg += int(rect_h) * int(rect_w)
        max_rect_area = max(max_rect_area, int(rect_h) * int(rect_w))
        for mask in pair["pieces"]:  # type: ignore[index]
            mask = np.asarray(mask, dtype=bool)
            max_piece_h = max(max_piece_h, int(mask.shape[0]))
            max_piece_w = max(max_piece_w, int(mask.shape[1]))
            pieces.append((mask, color))

    min_h = max(GRID_H_RANGE[0], max_piece_h + 3)
    min_w = max(GRID_W_RANGE[0], max_piece_w + 3)
    if min_h >= GRID_H_RANGE[1] or min_w >= GRID_W_RANGE[1]:
        return None

    for _ in range(40):
        grid_h = int(rng.integers(min_h, GRID_H_RANGE[1]))
        grid_w = int(rng.integers(min_w, GRID_W_RANGE[1]))
        bg_count = grid_h * grid_w - total_fg
        if bg_count <= max_rect_area:
            continue
        grid = _place_pieces(pieces, (grid_h, grid_w), bg, rng)
        if grid is None:
            continue

        expected = _expected_from_pairs(pairs, bg)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
