import random
from collections import deque

import numpy as np


BG = 5
RED = 2
GREEN = 3
PIECE_COLORS = [0, 1, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _inside(r, c, h, w):
    return 0 <= r < h and 0 <= c < w


def _outside_reachable(red_mask):
    h, w = red_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not red_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not red_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _inside(nr, nc, h, w) and not red_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _interior_from_red(red_mask):
    outside = _outside_reachable(red_mask)
    return (~red_mask) & (~outside)


def _trim_mask(mask):
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return mask[:0, :0].copy()
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _coords_to_mask(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _rotate_mask(mask, clockwise_quarters):
    return np.rot90(mask, (-clockwise_quarters) % 4).copy()


def _extract_piece_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces = []
    special = {BG, RED, GREEN}

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if seen[sr, sc] or color in special:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if (
                        _inside(nr, nc, h, w)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            side_votes = []
            for r, c in cells:
                checks = [
                    (r - 1, c, 0),
                    (r, c + 1, 1),
                    (r + 1, c, 2),
                    (r, c - 1, 3),
                ]
                for nr, nc, side in checks:
                    if _inside(nr, nc, h, w) and int(grid[nr, nc]) == GREEN:
                        side_votes.append(side)
            clockwise = side_votes[0] if side_votes else 0
            mask = _rotate_mask(_coords_to_mask(cells), clockwise)
            pieces.append(
                {
                    "color": color,
                    "mask": mask,
                    "area": int(mask.sum()),
                    "pos": (min(r for r, _ in cells), min(c for _, c in cells)),
                }
            )

    pieces.sort(key=lambda p: (-p["area"], p["pos"][0], p["pos"][1]))
    return pieces


def _placements_for_piece(interior, mask):
    h, w = interior.shape
    ph, pw = mask.shape
    offsets = [(int(r), int(c)) for r, c in zip(*np.nonzero(mask))]
    cell_to_idx = {
        (int(r), int(c)): i for i, (r, c) in enumerate(zip(*np.nonzero(interior)))
    }
    placements = []
    if ph == 0 or pw == 0 or ph > h or pw > w:
        return placements
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            bits = 0
            coords = []
            ok = True
            for dr, dc in offsets:
                rc = (r0 + dr, c0 + dc)
                idx = cell_to_idx.get(rc)
                if idx is None:
                    ok = False
                    break
                bits |= 1 << idx
                coords.append(rc)
            if ok:
                placements.append((bits, tuple(coords), (r0, c0)))
    return placements


def _solve_tiling(interior, pieces):
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(interior))]
    total_area = len(cells)
    if sum(piece["area"] for piece in pieces) != total_area:
        return None
    if total_area == 0:
        return []

    all_bits = (1 << total_area) - 1
    placements = []
    for piece in pieces:
        piece_placements = _placements_for_piece(interior, piece["mask"])
        if not piece_placements:
            return None
        placements.append(piece_placements)

    failed = set()

    def search(covered, remaining):
        key = (covered, remaining)
        if key in failed:
            return None
        if not remaining:
            return [] if covered == all_bits else None

        uncovered = all_bits & ~covered
        best_options = None
        scan = uncovered
        while scan:
            cell_bit = scan & -scan
            options = []
            for pi in remaining:
                for placement_index, (bits, _, _) in enumerate(placements[pi]):
                    if bits & cell_bit and not (bits & covered):
                        options.append((pi, placement_index))
            if not options:
                failed.add(key)
                return None
            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break
            scan -= cell_bit

        best_options.sort(key=lambda x: (x[0], placements[x[0]][x[1]][2]))
        for pi, placement_index in best_options:
            bits, coords, _ = placements[pi][placement_index]
            if bits & covered:
                continue
            next_remaining = tuple(x for x in remaining if x != pi)
            result = search(covered | bits, next_remaining)
            if result is not None:
                return [(pi, coords)] + result

        failed.add(key)
        return None

    return search(0, tuple(range(len(pieces))))


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the red outline with the scattered pieces after applying marker-coded rotations.

    Input structure:
    The background is grey, a single-pixel red closed outline encloses the target area, and separate
    single-color fragments sit outside it.  Each fragment has exactly one adjacent green marker on
    one of its four sides.

    Rule:
    A green marker above a fragment means rotate it 0 degrees, on the right means rotate it 90
    degrees clockwise, below means rotate it 180 degrees, and on the left means rotate it 270 degrees
    clockwise.  Remove the green markers, translate the rotated fragments into the red outline's
    interior, and tile that interior without gaps or overlaps.  Keep only the red outline, the filled
    fragments, and the grey background.

    Key insight:
    The marker side supplies each fragment's orientation before the spatial exact-cover match can
    place the rotated shapes inside the outline.
    """
    arr = np.asarray(grid)
    red_mask = arr == RED
    interior = _interior_from_red(red_mask)
    pieces = _extract_piece_components(arr)
    solution = _solve_tiling(interior, pieces)

    out = np.full(arr.shape, BG, dtype=arr.dtype)
    out[red_mask] = RED
    if solution is None:
        return out
    for piece_index, coords in solution:
        color = pieces[piece_index]["color"]
        for r, c in coords:
            out[r, c] = color
    return out


def _boundary_coords_for_mask(mask):
    h, w = mask.shape
    red = set()
    for r, c in zip(*np.nonzero(mask)):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not _inside(nr, nc, h, w) or not mask[nr, nc]:
                    red.add((nr + 1, nc + 1))
    return red


def _boundary_is_valid(mask):
    h, w = mask.shape
    red_mask = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in _boundary_coords_for_mask(mask):
        red_mask[r, c] = True
    expected = np.zeros_like(red_mask)
    expected[1 : h + 1, 1 : w + 1] = mask
    return np.array_equal(_interior_from_red(red_mask), expected)


def _random_connected_mask(h, w, area, rng):
    start = (rng.randrange(h), rng.randrange(w))
    mask = np.zeros((h, w), dtype=bool)
    mask[start] = True
    frontier = set()
    for dr, dc in DIRS4:
        nr, nc = start[0] + dr, start[1] + dc
        if _inside(nr, nc, h, w):
            frontier.add((nr, nc))

    while int(mask.sum()) < area and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        r, c = cell
        if mask[r, c]:
            continue
        mask[r, c] = True
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _inside(nr, nc, h, w) and not mask[nr, nc]:
                frontier.add((nr, nc))
    if int(mask.sum()) != area:
        return None
    return _trim_mask(mask)


def _partition_mask(mask, n, rng):
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(mask))]
    if len(cells) < n:
        return None
    seeds = rng.sample(cells, n)
    labels = np.full(mask.shape, -1, dtype=int)
    sizes = [0] * n
    for i, cell in enumerate(seeds):
        labels[cell] = i
        sizes[i] = 1
    unassigned = set(cells) - set(seeds)
    h, w = mask.shape

    while unassigned:
        choices = []
        for label in range(n):
            frontier = []
            for r, c in unassigned:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if _inside(nr, nc, h, w) and labels[nr, nc] == label:
                        frontier.append((r, c))
                        break
            if frontier:
                choices.append((sizes[label], label, frontier))
        if not choices:
            return None
        smallest = min(size for size, _, _ in choices)
        small_choices = [item for item in choices if item[0] <= smallest + 1]
        _, label, frontier = rng.choice(small_choices)
        cell = rng.choice(frontier)
        labels[cell] = label
        sizes[label] += 1
        unassigned.remove(cell)

    if min(sizes) < 3:
        return None
    return labels


def _stamp_with_marker(mask, clockwise, rng):
    h, w = mask.shape
    if clockwise == 0:
        candidates = [(0, int(c)) for c in np.nonzero(mask[0])[0]]
        r, c = rng.choice(candidates)
        marker = (r - 1, c)
    elif clockwise == 1:
        candidates = [(int(r), w - 1) for r in np.nonzero(mask[:, w - 1])[0]]
        r, c = rng.choice(candidates)
        marker = (r, c + 1)
    elif clockwise == 2:
        candidates = [(h - 1, int(c)) for c in np.nonzero(mask[h - 1])[0]]
        r, c = rng.choice(candidates)
        marker = (r + 1, c)
    else:
        candidates = [(int(r), 0) for r in np.nonzero(mask[:, 0])[0]]
        r, c = rng.choice(candidates)
        marker = (r, c - 1)

    piece = [(int(r), int(c)) for r, c in zip(*np.nonzero(mask))]
    all_cells = piece + [marker]
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)
    norm_piece = [(r - min_r, c - min_c) for r, c in piece]
    norm_marker = (marker[0] - min_r, marker[1] - min_c)
    return norm_piece, norm_marker, (max_r - min_r + 1, max_c - min_c + 1)


def _reserve_cells(reserved, cells):
    h, w = reserved.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _inside(nr, nc, h, w):
                    reserved[nr, nc] = True


def _place_stamps(grid, stamps, right_start, rng):
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)
    _reserve_cells(reserved, [(int(r), int(c)) for r, c in zip(*np.nonzero(grid != BG))])

    order = sorted(range(len(stamps)), key=lambda i: -len(stamps[i]["piece"]))
    for idx in order:
        stamp = stamps[idx]
        sh, sw = stamp["shape"]
        if sh > h - 2 or sw > w - right_start:
            return False
        positions = []
        r_values = list(range(1, max(1, h - sh)))
        c_values = list(range(right_start, max(right_start, w - sw)))
        for r0 in r_values:
            for c0 in c_values:
                all_cells = (
                    [(r0 + r, c0 + c) for r, c in stamp["piece"]]
                    + [(r0 + stamp["marker"][0], c0 + stamp["marker"][1])]
                )
                if all(not reserved[r, c] for r, c in all_cells):
                    positions.append((r0, c0, all_cells))
        if not positions:
            return False
        r0, c0, all_cells = rng.choice(positions)
        for r, c in [(r0 + r, c0 + c) for r, c in stamp["piece"]]:
            grid[r, c] = stamp["color"]
        marker_r, marker_c = stamp["marker"]
        grid[r0 + marker_r, c0 + marker_c] = GREEN
        _reserve_cells(reserved, all_cells)
    return True


def _rotation_sequence(n, rng):
    rotations = []
    while len(rotations) < n:
        batch = [0, 1, 2, 3]
        rng.shuffle(batch)
        rotations.extend(batch)
    rotations = rotations[:n]
    rng.shuffle(rotations)
    return rotations


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        box_h = rng.randint(5, 10)
        box_w = rng.randint(5, 8)
        area = rng.randint(max(14, (box_h * box_w) // 2), box_h * box_w)
        target = _random_connected_mask(box_h, box_w, area, rng)
        if target is None or target.shape[0] < 4 or target.shape[1] < 4:
            continue
        if not _boundary_is_valid(target):
            continue

        area = int(target.sum())
        max_pieces = min(6, len(PIECE_COLORS), area // 3)
        if max_pieces < 3:
            continue
        piece_count = rng.randint(3, max_pieces)
        labels = _partition_mask(target, piece_count, rng)
        if labels is None:
            continue

        th, tw = target.shape
        rows = rng.randint(max(18, th + 4), 30)
        target_left = rng.randint(1, 3)
        right_start = target_left + tw + 4
        min_cols = max(24, right_start + 9)
        if min_cols > 30:
            continue
        cols = rng.randint(min_cols, 30)
        target_top = rng.randint(1, rows - th - 2)

        grid = np.full((rows, cols), BG, dtype=int)
        for r, c in _boundary_coords_for_mask(target):
            grid[target_top + r, target_left + c] = RED

        colors = PIECE_COLORS[:]
        rng.shuffle(colors)
        rotations = _rotation_sequence(piece_count, rng)
        label_order = list(range(piece_count))
        rng.shuffle(label_order)
        stamps = []
        for out_index, label in enumerate(label_order):
            final_mask = _trim_mask(labels == label)
            input_mask = _rotate_mask(final_mask, (-rotations[out_index]) % 4)
            piece_cells, marker, shape = _stamp_with_marker(
                input_mask, rotations[out_index], rng
            )
            stamps.append(
                {
                    "piece": piece_cells,
                    "marker": marker,
                    "shape": shape,
                    "color": colors[out_index],
                }
            )
        rng.shuffle(stamps)
        if _place_stamps(grid, stamps, right_start, rng):
            return grid

    return _fallback_input(rng)


def _fallback_input(rng):
    rows, cols = rng.randint(20, 26), rng.randint(25, 30)
    target = np.ones((6, 6), dtype=bool)
    labels = np.zeros((6, 6), dtype=int)
    labels[:, 2:4] = 1
    labels[:, 4:] = 2
    target_top, target_left = 2, 1
    right_start = 12
    grid = np.full((rows, cols), BG, dtype=int)
    for r, c in _boundary_coords_for_mask(target):
        grid[target_top + r, target_left + c] = RED

    colors = PIECE_COLORS[:]
    rng.shuffle(colors)
    rotations = _rotation_sequence(3, rng)
    stamps = []
    for label in range(3):
        final_mask = _trim_mask(labels == label)
        input_mask = _rotate_mask(final_mask, (-rotations[label]) % 4)
        piece_cells, marker, shape = _stamp_with_marker(input_mask, rotations[label], rng)
        stamps.append(
            {
                "piece": piece_cells,
                "marker": marker,
                "shape": shape,
                "color": colors[label],
            }
        )
    _place_stamps(grid, stamps, right_start, rng)
    return grid
