import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(int(x) for x in mask.shape), mask.tobytes()


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = _mask_key(rotated)
        if key in seen:
            continue
        seen.add(key)
        out.append(rotated)
    return out


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _extract_pieces(grid: np.ndarray, sep: int) -> dict[int, np.ndarray]:
    top = grid[:sep, :]
    comps = _components((top != BLACK) & (top != GREY))
    pieces: dict[int, np.ndarray] = {}
    for coords in comps:
        colors = {int(top[r, c]) for r, c in coords}
        assert len(colors) == 1, "top fragments must be monochrome"
        color = colors.pop()
        assert color not in pieces, "fragment colors must be unique"
        pieces[color] = _crop_mask_from_coords(coords)
    assert pieces, "expected at least one top fragment"
    return pieces


def _extract_containers(grid: np.ndarray, sep: int) -> list[dict]:
    bottom = grid[sep + 1 :, :]
    comps = _components(bottom == GREY)
    containers = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        shape = np.zeros((h, w), dtype=bool)
        shape[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not np.array_equal(shape, border):
            continue
        interior = bottom[r0 + 1 : r1, c0 + 1 : c1]
        colors = {int(v) for v in np.unique(interior) if int(v) not in {BLACK, GREY}}
        assert len(colors) == 1, "each container must contain exactly one marker color"
        color = colors.pop()
        marker = interior == color
        marker_parts = _components(marker)
        assert len(marker_parts) == 1, "marker must be 4-connected"
        area = int(marker.sum())
        assert area in (2, 3), "marker must have size 2 or 3"
        containers.append(
            {
                "color": color,
                "top": sep + 1 + r0,
                "left": c0,
                "outer_shape": (h, w),
                "interior_shape": marker.shape,
                "marker": marker,
            }
        )
    assert containers, "expected at least one container"
    return containers


def _candidate_occupancies(piece_mask: np.ndarray, marker_mask: np.ndarray, interior_shape: tuple[int, int]) -> list[np.ndarray]:
    ih, iw = (int(interior_shape[0]), int(interior_shape[1]))
    marker_mask = np.asarray(marker_mask, dtype=bool)
    assert marker_mask.shape == (ih, iw)
    occupancies: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for rotated in _unique_rotations(piece_mask):
        ph, pw = rotated.shape
        if ph > ih or pw > iw:
            continue
        for top in range(ih - ph + 1):
            for left in range(iw - pw + 1):
                canvas = np.zeros((ih, iw), dtype=bool)
                canvas[top : top + ph, left : left + pw] = rotated
                if np.all(canvas[marker_mask]):
                    occupancies[_mask_key(canvas)] = canvas
    return list(occupancies.values())


def _solve_piece_placement(piece_mask: np.ndarray, marker_mask: np.ndarray, interior_shape: tuple[int, int]) -> np.ndarray:
    occupancies = _candidate_occupancies(piece_mask, marker_mask, interior_shape)
    assert len(occupancies) == 1, "piece/container match must be unique"
    return occupancies[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each top fragment to the same-colored container marker, rotate the
    fragment until the marker is a subset of it, then place the full fragment
    into the container and erase the source fragment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    sep = _find_separator_row(g)
    pieces = _extract_pieces(g, sep)
    containers = _extract_containers(g, sep)

    container_colors = {int(c["color"]) for c in containers}
    assert set(pieces) == container_colors, "piece colors and container colors must match exactly"

    out = g.copy()
    top = out[:sep, :]
    top[top != GREY] = BLACK

    for container in containers:
        color = int(container["color"])
        placement = _solve_piece_placement(pieces[color], container["marker"], container["interior_shape"])
        r0 = int(container["top"])
        c0 = int(container["left"])
        h, w = container["outer_shape"]
        interior = out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1]
        interior[:] = BLACK
        interior[placement] = color

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    cells = _normalize_cells(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_piece(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        target_area = rng.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target_area:
            r, c = rng.choice(tuple(cells))
            dr, dc = rng.choice(NEIGHBORS4)
            cells.add((r + dr, c + dc))
        mask = _cells_to_mask(cells)
        h, w = mask.shape
        area = int(mask.sum())
        if h < 2 or w < 2:
            continue
        if area == h * w:
            continue
        return mask
    raise RuntimeError("failed to sample a valid fragment")


def _is_connected_subset(cells: tuple[tuple[int, int], ...]) -> bool:
    remaining = set(cells)
    start = next(iter(remaining))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in remaining and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == remaining


def _distance_to_wall(cell: tuple[int, int], h: int, w: int) -> int:
    r, c = cell
    return min(r, c, h - 1 - r, w - 1 - c)


def _design_container(piece_mask: np.ndarray, rng: random.Random) -> dict | None:
    area = int(piece_mask.sum())
    candidate_sizes = [size for size in (2, 3) if size < area]
    if not candidate_sizes:
        return None

    for _ in range(120):
        target_k = rng.randrange(4)
        placed_piece = np.rot90(piece_mask, target_k)
        ph, pw = placed_piece.shape
        pad_h = rng.randint(1, 3)
        pad_w = rng.randint(1, 3)
        ih = ph + pad_h
        iw = pw + pad_w

        mode = rng.randrange(3)
        if mode == 0:
            top = rng.choice([0, pad_h])
            left = rng.choice([0, pad_w])
        elif mode == 1:
            top = rng.choice([0, pad_h])
            left = rng.randint(0, pad_w)
        else:
            top = rng.randint(0, pad_h)
            left = rng.choice([0, pad_w])

        occupancy = np.zeros((ih, iw), dtype=bool)
        occupancy[top : top + ph, left : left + pw] = placed_piece
        occupied_cells = [tuple(map(int, cell)) for cell in np.argwhere(occupancy)]
        near_wall = [cell for cell in occupied_cells if _distance_to_wall(cell, ih, iw) <= 1]
        if len(near_wall) < min(candidate_sizes):
            continue

        markers_by_size: dict[int, list[np.ndarray]] = {2: [], 3: []}
        for size in candidate_sizes:
            for subset in combinations(near_wall, size):
                if not _is_connected_subset(subset):
                    continue
                marker = np.zeros((ih, iw), dtype=bool)
                for r, c in subset:
                    marker[r, c] = True
                solutions = _candidate_occupancies(piece_mask, marker, (ih, iw))
                if len(solutions) != 1:
                    continue
                if not np.array_equal(solutions[0], occupancy):
                    continue
                markers_by_size[size].append(marker)

        available_sizes = [size for size in candidate_sizes if markers_by_size[size]]
        if not available_sizes:
            continue
        marker_size = rng.choice(available_sizes)
        marker = rng.choice(markers_by_size[marker_size])
        return {
            "piece_mask": piece_mask,
            "target_k": target_k,
            "occupancy": occupancy,
            "marker": marker,
            "interior_shape": (ih, iw),
            "outer_shape": (ih + 2, iw + 2),
        }

    return None


def _arrange_containers(outer_shapes: list[tuple[int, int]], rng: random.Random) -> tuple[int, int, dict[int, tuple[int, int]]] | None:
    n = len(outer_shapes)
    order = list(range(n))
    rng.shuffle(order)
    if n >= 3 and rng.random() < 0.5:
        split = rng.randint(1, n - 1)
        rows = [order[:split], order[split:]]
    else:
        rows = [order]

    row_gap = rng.randint(2, 4)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)

    row_specs = []
    min_inner_w = 0
    total_h = top_margin + bottom_margin + row_gap * (len(rows) - 1)
    for row in rows:
        gaps = [rng.randint(2, 4) for _ in range(max(0, len(row) - 1))]
        row_w = sum(int(outer_shapes[idx][1]) for idx in row) + sum(gaps)
        row_h = max(int(outer_shapes[idx][0]) for idx in row)
        row_specs.append((row, gaps, row_w, row_h))
        min_inner_w = max(min_inner_w, row_w)
        total_h += row_h

    final_w = left_margin + min_inner_w + right_margin + rng.randint(0, 6)
    positions: dict[int, tuple[int, int]] = {}
    y = top_margin
    for row, gaps, row_w, row_h in row_specs:
        slack = final_w - left_margin - right_margin - row_w
        if slack < 0:
            return None
        x = left_margin + rng.randint(0, slack)
        for i, idx in enumerate(row):
            h, w = outer_shapes[idx]
            dy = rng.randint(0, row_h - h)
            positions[idx] = (y + dy, x)
            x += w
            if i < len(gaps):
                x += gaps[i]
        y += row_h + row_gap

    return final_w, total_h, positions


def _place_top_pieces(top_masks: list[np.ndarray], width: int, rng: random.Random) -> tuple[int, list[tuple[int, int]]] | None:
    min_h = max(int(mask.shape[0]) for mask in top_masks) + 2
    max_h = min_h + 5
    for _ in range(40):
        top_h = rng.randint(min_h, max_h)
        occupied = np.zeros((top_h, width), dtype=bool)
        positions: list[tuple[int, int] | None] = [None] * len(top_masks)
        order = sorted(
            range(len(top_masks)),
            key=lambda idx: int(top_masks[idx].shape[0] * top_masks[idx].shape[1]),
            reverse=True,
        )
        ok = True
        for idx in order:
            h, w = top_masks[idx].shape
            if w > width or h >= top_h:
                ok = False
                break
            found = None
            for _ in range(120):
                r = rng.randint(0, top_h - h - 1)
                c = rng.randint(0, width - w)
                r0 = max(0, r - 1)
                r1 = min(top_h, r + h + 1)
                c0 = max(0, c - 1)
                c1 = min(width, c + w + 1)
                if occupied[r0:r1, c0:c1].any():
                    continue
                found = (r, c, r0, r1, c0, c1)
                break
            if found is None:
                ok = False
                break
            r, c, r0, r1, c0, c1 = found
            positions[idx] = (r, c)
            occupied[r0:r1, c0:c1] = True
        if ok:
            return top_h, [pos for pos in positions if pos is not None]
    return None


def _draw_border(grid: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    grid[top, left : left + w] = GREY
    grid[top + h - 1, left : left + w] = GREY
    grid[top : top + h, left] = GREY
    grid[top : top + h, left + w - 1] = GREY


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, count)

    specs = []
    for color in colors:
        piece_mask = _random_piece(rng)
        container = _design_container(piece_mask, rng)
        if container is None:
            return None
        container["color"] = color
        specs.append(container)

    top_masks = []
    for spec in specs:
        display_k = rng.randrange(4)
        spec["display_k"] = display_k
        top_masks.append(np.rot90(spec["piece_mask"], display_k))

    changeable = []
    for idx, spec in enumerate(specs):
        target_view = np.rot90(spec["piece_mask"], spec["target_k"])
        choices = [k for k in range(4) if not np.array_equal(np.rot90(spec["piece_mask"], k), target_view)]
        if choices:
            changeable.append((idx, choices))
    if changeable and all(
        np.array_equal(top_masks[idx], np.rot90(spec["piece_mask"], spec["target_k"]))
        for idx, spec in enumerate(specs)
    ):
        idx, choices = rng.choice(changeable)
        specs[idx]["display_k"] = rng.choice(choices)
        top_masks[idx] = np.rot90(specs[idx]["piece_mask"], specs[idx]["display_k"])

    outer_shapes = [tuple(int(x) for x in spec["outer_shape"]) for spec in specs]
    arranged = _arrange_containers(outer_shapes, rng)
    if arranged is None:
        return None
    bottom_w, bottom_h, container_positions = arranged

    min_w = max(bottom_w, 14, max(int(mask.shape[1]) for mask in top_masks) + 2)
    width = min_w + rng.randint(0, 4)
    if width < bottom_w:
        width = bottom_w

    if width != bottom_w:
        slack = width - bottom_w
        shift = rng.randint(0, slack)
        container_positions = {idx: (r, c + shift) for idx, (r, c) in container_positions.items()}

    top_placed = _place_top_pieces(top_masks, width, rng)
    if top_placed is None:
        return None
    top_h, top_positions = top_placed

    total_h = top_h + 1 + bottom_h
    sep = top_h
    inp = np.zeros((total_h, width), dtype=int)
    out = np.zeros((total_h, width), dtype=int)
    inp[sep, :] = GREY
    out[sep, :] = GREY

    for idx, spec in enumerate(specs):
        br, bc = container_positions[idx]
        r0 = sep + 1 + br
        c0 = bc
        oh, ow = spec["outer_shape"]
        _draw_border(inp, r0, c0, oh, ow)
        _draw_border(out, r0, c0, oh, ow)

        marker = spec["marker"]
        occupancy = spec["occupancy"]
        color = int(spec["color"])
        inp[r0 + 1 : r0 + oh - 1, c0 + 1 : c0 + ow - 1][marker] = color
        out[r0 + 1 : r0 + oh - 1, c0 + 1 : c0 + ow - 1][occupancy] = color

    for spec, mask, (r, c) in zip(specs, top_masks, top_positions):
        _draw_mask(inp, mask, r, c, int(spec["color"]))

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
