import random
from collections import Counter, deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch
    return dst


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(mask.shape, r, c):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray, aux: np.ndarray | None = None):
    box = bbox(mask)
    assert box is not None
    cropped = crop_bbox(mask, box).astype(bool)
    if aux is None:
        return cropped
    return cropped, crop_bbox(aux, box).astype(bool)


def _random_transform(mask: np.ndarray, aux: np.ndarray | None = None):
    turns = random.randrange(4)
    flip = random.random() < 0.5
    for _ in range(turns):
        mask = np.rot90(mask)
        if aux is not None:
            aux = np.rot90(aux)
    if flip:
        mask = np.fliplr(mask)
        if aux is not None:
            aux = np.fliplr(aux)
    return _crop_mask(mask, aux)


def _shape_coords(mask: np.ndarray):
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _shape_ok(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(mask.sum())
    if not (4 <= h <= 7 and 4 <= w <= 7):
        return False
    if not (8 <= area <= 24):
        return False
    if not _is_connected(mask):
        return False
    if area == h * w:
        return False
    row_sums = {int(x) for x in mask.sum(axis=1)}
    col_sums = {int(x) for x in mask.sum(axis=0)}
    if len(row_sums) == 1 and len(col_sums) == 1:
        return False
    return True


def _gen_blob_shape():
    h = random.randint(5, 7)
    w = random.randint(5, 7)
    target = random.randint(max(8, (h * w) // 3), min(h * w - 2, (2 * h * w) // 3))
    mask = np.zeros((h, w), dtype=bool)
    sr = random.randrange(h)
    sc = random.randrange(w)
    mask[sr, sc] = True

    while int(mask.sum()) < target:
        frontier = set()
        for r, c in _shape_coords(mask):
            for nr, nc in _neighbors4(mask.shape, r, c):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None, None
        ranked = sorted(
            frontier,
            key=lambda rc: -sum(mask[nr, nc] for nr, nc in _neighbors4(mask.shape, rc[0], rc[1])),
        )
        cell = random.choice(ranked[: max(1, len(ranked) // 2)])
        mask[cell] = True

    if int(mask.sum()) >= 12 and random.random() < 0.45:
        removable = []
        for r, c in _shape_coords(mask):
            if sum(mask[nr, nc] for nr, nc in _neighbors4(mask.shape, r, c)) < 2:
                continue
            trial = mask.copy()
            trial[r, c] = False
            if _is_connected(trial):
                removable.append((r, c))
        if removable:
            mask[random.choice(removable)] = False

    mask = _crop_mask(mask)
    return mask, np.zeros_like(mask, dtype=bool)


def _gen_barbell_shape():
    h = random.randint(5, 7)
    w = random.randint(6, 7)
    mask = np.zeros((h, w), dtype=bool)
    left_h = random.randint(2, 3)
    left_w = random.randint(2, 3)
    right_h = random.randint(2, 3)
    right_w = random.randint(2, 3)
    mask[:left_h, :left_w] = True
    rr0 = h - right_h
    rc0 = w - right_w
    mask[rr0:, rc0:] = True
    before = mask.copy()

    start = (random.randrange(left_h), left_w - 1)
    end = (rr0 + random.randrange(right_h), rc0)
    if random.random() < 0.5:
        r = start[0]
        c0, c1 = sorted((start[1], end[1]))
        mask[r, c0 : c1 + 1] = True
        r0, r1 = sorted((r, end[0]))
        mask[r0 : r1 + 1, end[1]] = True
    else:
        c = start[1]
        r0, r1 = sorted((start[0], end[0]))
        mask[r0 : r1 + 1, c] = True
        c0, c1 = sorted((c, end[1]))
        mask[end[0], c0 : c1 + 1] = True

    bridge = mask & ~before
    extras = []
    for r, c in _shape_coords(bridge):
        for nr, nc in _neighbors4(mask.shape, r, c):
            if not mask[nr, nc]:
                extras.append((nr, nc))
    random.shuffle(extras)
    for r, c in extras[: random.randint(0, 2)]:
        mask[r, c] = True

    return mask, bridge


def _gen_frame_shape():
    h = random.randint(5, 7)
    w = random.randint(5, 7)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    side = random.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"} and w >= 5:
        start = random.randint(1, w - 3)
        length = random.randint(1, min(2, w - 1 - start))
        row = 0 if side == "top" else h - 1
        mask[row, start : start + length] = False
    elif side in {"left", "right"} and h >= 5:
        start = random.randint(1, h - 3)
        length = random.randint(1, min(2, h - 1 - start))
        col = 0 if side == "left" else w - 1
        mask[start : start + length, col] = False

    spur_side = random.choice(["top", "bottom", "left", "right"])
    if spur_side in {"top", "bottom"}:
        c = random.randint(1, w - 2)
        depth = random.randint(1, h - 3)
        if spur_side == "top":
            mask[1 : depth + 2, c] = True
        else:
            mask[h - depth - 2 : h - 1, c] = True
    else:
        r = random.randint(1, h - 2)
        depth = random.randint(1, w - 3)
        if spur_side == "left":
            mask[r, 1 : depth + 2] = True
        else:
            mask[r, w - depth - 2 : w - 1] = True

    return mask, np.zeros_like(mask, dtype=bool)


def _generate_shape():
    families = [
        _gen_blob_shape,
        _gen_blob_shape,
        _gen_barbell_shape,
        _gen_frame_shape,
    ]
    for _ in range(200):
        mask, aux = random.choice(families)()
        if mask is None:
            continue
        mask, aux = _random_transform(mask, aux)
        if _shape_ok(mask):
            return mask, aux
    raise RuntimeError("failed to generate prototype shape")


def _sample_counts(total: int, parts: int, min_each: int, max_each: int):
    if parts * min_each > total:
        return None
    for _ in range(200):
        counts = [min_each] * parts
        remaining = total - parts * min_each
        while remaining > 0:
            choices = [i for i, count in enumerate(counts) if count < max_each]
            if not choices:
                break
            counts[random.choice(choices)] += 1
            remaining -= 1
        if remaining == 0:
            random.shuffle(counts)
            return counts
    return None


def _chunk_order(order, counts):
    owners = []
    start = 0
    for count in counts:
        owners.append(order[start : start + count])
        start += count
    return owners


def _walk_order(mask: np.ndarray):
    start = random.choice(_shape_coords(mask))
    order = []
    seen = set()
    stack = [start]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        order.append((r, c))
        nbrs = [(nr, nc) for nr, nc in _neighbors4(mask.shape, r, c) if mask[nr, nc] and (nr, nc) not in seen]
        random.shuffle(nbrs)
        stack.extend(nbrs)
    assert len(order) == int(mask.sum())
    return order


def _partition_regions(mask: np.ndarray, parts: int, min_owner: int, max_owner: int):
    cells = _shape_coords(mask)
    for _ in range(200):
        seeds = random.sample(cells, parts)
        owners = [[] for _ in range(parts)]
        for r, c in cells:
            best = min(
                range(parts),
                key=lambda idx: (abs(r - seeds[idx][0]) + abs(c - seeds[idx][1]), idx),
            )
            owners[best].append((r, c))
        sizes = [len(owner) for owner in owners]
        if min(sizes) >= min_owner and max(sizes) <= max_owner:
            return owners
    return None


def _partition_bands(mask: np.ndarray, counts):
    direction = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
    a, b = direction
    order = sorted(
        _shape_coords(mask),
        key=lambda rc: (a * rc[0] + b * rc[1], b * rc[0] - a * rc[1], rc[0], rc[1]),
    )
    if random.random() < 0.5:
        order.reverse()
    return _chunk_order(order, counts)


def _partition_walk(mask: np.ndarray, counts):
    return _chunk_order(_walk_order(mask), counts)


def _owners_to_grid(mask: np.ndarray, owners):
    owner_grid = np.full(mask.shape, -1, dtype=int)
    for idx, owner in enumerate(owners):
        for r, c in owner:
            owner_grid[r, c] = idx
    assert np.all(owner_grid[mask] >= 0)
    return owner_grid


def _owner_counts(owner_grid: np.ndarray, parts: int):
    return [int(np.count_nonzero(owner_grid == idx)) for idx in range(parts)]


def _try_force_cells(owner_grid: np.ndarray, parts: int, cells, max_owner: int):
    if not cells:
        return owner_grid
    counts = _owner_counts(owner_grid, parts)
    targets = list(range(parts))
    random.shuffle(targets)
    for target in targets:
        trial = owner_grid.copy()
        for r, c in cells:
            trial[r, c] = target
        if max(_owner_counts(trial, parts)) <= max_owner:
            return trial
    return owner_grid


def _generate_owner_grid(mask: np.ndarray, bridge_mask: np.ndarray, parts: int):
    area = int(mask.sum())
    min_owner = 1 if area < 12 else 2
    min_present = max(4, area // 3)
    max_owner = area - min_present
    if max_owner < min_owner:
        return None

    for _ in range(200):
        mode = random.random()
        owners = None
        if mode < 0.4:
            owners = _partition_regions(mask, parts, min_owner, max_owner)
        else:
            counts = _sample_counts(area, parts, min_owner, max_owner)
            if counts is None:
                continue
            if mode < 0.72:
                owners = _partition_bands(mask, counts)
            else:
                owners = _partition_walk(mask, counts)
        if owners is None:
            continue

        owner_grid = _owners_to_grid(mask, owners)

        if random.random() < 0.55:
            h, w = mask.shape
            edge_sets = [
                [(0, c) for c in range(w) if mask[0, c]],
                [(h - 1, c) for c in range(w) if mask[h - 1, c]],
                [(r, 0) for r in range(h) if mask[r, 0]],
                [(r, w - 1) for r in range(h) if mask[r, w - 1]],
            ]
            owner_grid = _try_force_cells(owner_grid, parts, random.choice(edge_sets), max_owner)

        bridge_cells = _shape_coords(bridge_mask)
        if bridge_cells and random.random() < 0.65:
            owner_grid = _try_force_cells(owner_grid, parts, bridge_cells, max_owner)

        counts = _owner_counts(owner_grid, parts)
        if min(counts) < min_owner or max(counts) > max_owner:
            continue
        if any(area - count < min_present for count in counts):
            continue
        return owner_grid
    return None


def _candidate_missing_bits(proto_mask: np.ndarray, obs: np.ndarray, index_by_coord, full_shape: tuple[int, int]):
    h, w = proto_mask.shape
    gh, gw = full_shape
    proto_cells = _shape_coords(proto_mask)
    first_r, first_c = map(int, obs[0])
    candidates = {}
    for pr, pc in proto_cells:
        top = first_r - pr
        left = first_c - pc
        if top < 0 or left < 0 or top + h > gh or left + w > gw:
            continue
        local = obs - np.array([[top, left]])
        if np.any(local < 0):
            continue
        if np.any(local[:, 0] >= h) or np.any(local[:, 1] >= w):
            continue
        if not np.all(proto_mask[local[:, 0], local[:, 1]]):
            continue
        present = {tuple(map(int, rc)) for rc in local}
        if len(present) != len(obs):
            continue
        bits = 0
        for coord in proto_cells:
            if coord not in present:
                bits |= 1 << index_by_coord[coord]
        if bits != 0:
            candidates[bits] = True
    return sorted(candidates, key=int.bit_count)


def _solve_assignment(proto_mask: np.ndarray, copies: dict[int, np.ndarray], grid_shape: tuple[int, int]):
    proto_cells = _shape_coords(proto_mask)
    index_by_coord = {coord: idx for idx, coord in enumerate(proto_cells)}
    full_bits = (1 << len(proto_cells)) - 1
    candidates_by_color = {}
    for color, obs in copies.items():
        bits_list = _candidate_missing_bits(proto_mask, obs, index_by_coord, grid_shape)
        assert bits_list, f"no valid alignment for copy color {color}"
        candidates_by_color[color] = bits_list

    order = sorted(candidates_by_color, key=lambda color: len(candidates_by_color[color]))
    suffix_union = [0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        current_union = 0
        for bits in candidates_by_color[order[i]]:
            current_union |= bits
        suffix_union[i] = suffix_union[i + 1] | current_union

    solutions = []
    assignment = {}

    def search(idx: int, used_bits: int):
        if len(solutions) > 1:
            return
        if idx == len(order):
            if used_bits == full_bits:
                solutions.append(assignment.copy())
            return
        if (used_bits | suffix_union[idx]) != full_bits:
            return

        color = order[idx]
        rest_union = suffix_union[idx + 1]
        for bits in candidates_by_color[color]:
            if bits & used_bits:
                continue
            new_used = used_bits | bits
            if (new_used | rest_union) != full_bits:
                continue
            assignment[color] = bits
            search(idx + 1, new_used)
            assignment.pop(color, None)

    search(0, 0)
    assert len(solutions) == 1, "expected a unique exact partition of missing cells"
    return solutions[0], proto_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the prototype into a missing-ownership map colored by the defective copy that omits each cell.

    Input structure:
    - A black-background grid contains one complete teal prototype sprite and several translated single-color copies.
    - Each non-teal copy uses one unique color and occupies a subset of the prototype shape after alignment.
    - The copies may miss interior or boundary pixels, so their visible bounding boxes can shrink.

    Rule:
    - Find the teal prototype and use its bounding box as the local coordinate system.
    - For every other color, align that defective copy to the prototype by translation and recover which local cells are missing.
    - The missing-cell sets form an exact partition of the prototype, so color each prototype cell by the copy that is missing it and keep non-shape cells black.

    Key insight:
    - The defective copies are evidence, not targets for repair: the union of their holes is the full prototype, and the output is the prototype recolored by hole ownership.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    proto_mask_global = g == Color.TEAL
    proto_box = bbox(proto_mask_global)
    assert proto_box is not None, "missing teal prototype"
    proto_mask = crop_bbox(proto_mask_global, proto_box).astype(bool)

    copy_colors = [int(color) for color in np.unique(g) if color not in {bg, Color.TEAL}]
    assert len(copy_colors) >= 2, "expected several defective copies"
    copies = {color: np.argwhere(g == color) for color in copy_colors}

    assignment, proto_cells = _solve_assignment(proto_mask, copies, g.shape)
    out = np.zeros(proto_mask.shape, dtype=int)
    for color, bits in assignment.items():
        for idx, (r, c) in enumerate(proto_cells):
            if (bits >> idx) & 1:
                out[r, c] = color
    assert np.all(out[proto_mask] != Color.BLACK), "every prototype cell must be owned by one copy"
    return out


_COPY_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.MAROON,
]
_GEN_RETRY_LIMIT = 400


def _pick_copy_count(area: int) -> int:
    options = []
    for parts in range(2, 5):
        min_owner = 1 if area < 12 else 2
        min_present = max(4, area // 3)
        max_owner = area - min_present
        if parts * min_owner <= area and max_owner >= min_owner:
            options.append(parts)
    assert options
    weighted = []
    for parts in options:
        repeats = 3 if parts == 3 else 2
        weighted.extend([parts] * repeats)
    return random.choice(weighted)


def _render_input(mask: np.ndarray, owner_grid: np.ndarray, colors):
    h, w = mask.shape
    prototype = np.zeros((h, w), dtype=int)
    prototype[mask] = Color.TEAL

    patches = [prototype]
    for idx, color in enumerate(colors):
        patch = np.zeros((h, w), dtype=int)
        patch[mask & (owner_grid != idx)] = color
        patches.append(patch)

    count = len(patches)
    layouts = []
    for cols in range(1, min(3, count) + 1):
        rows = (count + cols - 1) // cols
        for gap in (2, 3):
            base_h = rows * h + (rows - 1) * gap
            base_w = cols * w + (cols - 1) * gap
            if base_h <= 28 and base_w <= 28:
                layouts.append((rows, cols, gap, base_h, base_w))
    assert layouts
    rows, cols, gap, base_h, base_w = random.choice(layouts)

    pad_top = random.randint(0, min(3, 30 - base_h))
    pad_bottom = random.randint(0, min(3, 30 - base_h - pad_top))
    pad_left = random.randint(0, min(3, 30 - base_w))
    pad_right = random.randint(0, min(3, 30 - base_w - pad_left))
    grid = np.zeros((base_h + pad_top + pad_bottom, base_w + pad_left + pad_right), dtype=int)

    slots = [(r * (h + gap), c * (w + gap)) for r in range(rows) for c in range(cols)]
    chosen = random.sample(slots, count)
    random.shuffle(patches)
    for patch, (r0, c0) in zip(patches, chosen):
        paste(grid, patch, pad_top + r0, pad_left + c0)
    return grid


def _generate_pair():
    mask, bridge_mask = _generate_shape()
    parts = _pick_copy_count(int(mask.sum()))
    owner_grid = _generate_owner_grid(mask, bridge_mask, parts)
    if owner_grid is None:
        return None, None

    colors = random.sample(_COPY_COLORS, parts)
    expected = np.zeros(mask.shape, dtype=int)
    for idx, color in enumerate(colors):
        expected[owner_grid == idx] = color
    inp = _render_input(mask, owner_grid, colors)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        inp, expected = _generate_pair()
        if inp is None:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ownership-reassembly input")
