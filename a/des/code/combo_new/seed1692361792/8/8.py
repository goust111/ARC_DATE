from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
TOWER = 4
RECEIVER_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

_NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
_DIRECTIONS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


RECEIVER_TEMPLATES = (
    _mask(("##.", "#..", ".##")),
    _mask(("##.", ".##", "..#")),
    _mask(("#..", "###", ".#.")),
    _mask(("##...", ".#...", ".##..", "..##.", "..###")),
    _mask(("###..", "..#..", ".###.", "...#.", "...##")),
    _mask(("##...", ".#...", ".###.", "...##", "....#")),
    _mask(("##...", "..#..", ".###.", "...##", "....#")),
)


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    start = tuple(map(int, points[0]))
    seen = {start}
    queue = deque([start])
    height, width = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in _NEIGHBORS8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(points.shape[0])


def _is_asymmetric(mask: np.ndarray) -> bool:
    variants = {np.rot90(mask, k).tobytes() for k in range(4)}
    return len(variants) == 4


for _template in RECEIVER_TEMPLATES:
    assert _template.shape[0] == _template.shape[1] and _template.shape[0] % 2 == 1
    assert _is_connected(_template)
    assert _is_asymmetric(_template)


def _rotate_offset(dr: int, dc: int, turns: int) -> tuple[int, int]:
    turns %= 4
    if turns == 0:
        return dr, dc
    if turns == 1:
        return dc, -dr
    if turns == 2:
        return -dr, -dc
    return -dc, dr


def _rotate_coords(
    coords: list[tuple[int, int]], center: tuple[int, int], turns: int
) -> list[tuple[int, int]]:
    cr, cc = center
    rotated = []
    for r, c in coords:
        dr = int(r - cr)
        dc = int(c - cc)
        rr, cc2 = _rotate_offset(dr, dc, turns)
        rotated.append((int(cr + rr), int(cc + cc2)))
    return rotated


def _find_receivers(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    receivers: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color in (BG, TOWER) or bool(visited[r, c]):
                continue
            visited[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS8:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if bool(visited[nr, nc]) or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            assert (r1 - r0 + 1) % 2 == 1 and (c1 - c0 + 1) % 2 == 1
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            receivers.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "center": center,
                }
            )

    receivers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return receivers


def _owner_grid(shape: tuple[int, int], receivers: list[dict]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for rid, receiver in enumerate(receivers):
        for r, c in receiver["coords"]:
            owner[int(r), int(c)] = int(rid)
    return owner


def _ray_cells(
    shape: tuple[int, int], owner: np.ndarray, rid: int, center: tuple[int, int]
) -> dict[str, list[tuple[int, int]]]:
    height, width = shape
    cr, cc = center
    rays: dict[str, list[tuple[int, int]]] = {}
    for name, (dr, dc) in _DIRECTIONS.items():
        cells: list[tuple[int, int]] = []
        r = int(cr + dr)
        c = int(cc + dc)
        while 0 <= r < height and 0 <= c < width:
            cell_owner = int(owner[r, c])
            if cell_owner >= 0 and cell_owner != rid:
                break
            if cell_owner < 0:
                cells.append((int(r), int(c)))
            r += dr
            c += dc
        rays[name] = cells
    return rays


def _visible_tower_count(grid: np.ndarray, owner: np.ndarray, rid: int, center: tuple[int, int]) -> int:
    count = 0
    for cells in _ray_cells(grid.shape, owner, rid, center).values():
        for r, c in cells:
            if int(grid[r, c]) == TOWER:
                count += 1
    return int(count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each monochrome receiver by the number of visible yellow towers on its orthogonal sight lines.

    Input structure:
    - The background is black.
    - Yellow single pixels are signal towers.
    - Every other non-black color forms one monochrome receiver component, with a unique color per receiver.
    - Receivers are separated from one another and from towers.

    Rule:
    - For each receiver, take the center pixel of its bounding box as the sight origin.
    - Cast rays upward, rightward, downward, and leftward from that center through empty cells and the receiver's own pixels.
    - A ray stops at the grid boundary or just before the first pixel belonging to a different receiver.
    - Count the yellow towers lying on the four rays; if the total is N, rotate the receiver clockwise by (N mod 4) * 90 degrees around that same center.
    - Keep all towers in place and redraw every receiver in its rotated orientation.

    Key insight:
    - The rotation cue is not a global tower count: each receiver only uses towers on its own four line-of-sight rays, and other receivers can block those rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    receivers = _find_receivers(g)
    owner = _owner_grid(g.shape, receivers)

    out = np.full_like(g, BG)
    out[g == TOWER] = TOWER

    for rid, receiver in enumerate(receivers):
        turns = _visible_tower_count(g, owner, rid, receiver["center"]) % 4
        for r, c in _rotate_coords(receiver["coords"], receiver["center"], turns):
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(out[r, c]) != TOWER:
                out[r, c] = int(receiver["color"])

    return out


def _slot_centers(limit: int) -> list[int]:
    centers = []
    pos = 3
    while pos <= limit - 3:
        centers.append(int(pos))
        pos += 6
    return centers


def _mask_to_coords(mask: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int]]:
    half = mask.shape[0] // 2
    cr, cc = center
    coords = []
    for r, c in np.argwhere(mask):
        coords.append((int(cr - half + r), int(cc - half + c)))
    return coords


def _receiver_clear(receivers: list[dict]) -> bool:
    for i, receiver_a in enumerate(receivers):
        for receiver_b in receivers[i + 1 :]:
            for r1, c1 in receiver_a["coords"]:
                for r2, c2 in receiver_b["coords"]:
                    if max(abs(int(r1 - r2)), abs(int(c1 - c2))) <= 1:
                        return False
    return True


def _tower_forbidden(shape: tuple[int, int], receivers: list[dict]) -> np.ndarray:
    height, width = shape
    forbidden = np.zeros((height, width), dtype=bool)
    for receiver in receivers:
        for r, c in receiver["coords"]:
            for rr in range(max(0, int(r - 1)), min(height, int(r + 2))):
                for cc in range(max(0, int(c - 1)), min(width, int(c + 2))):
                    forbidden[rr, cc] = True
    return forbidden


def _towers_nonadjacent(existing: list[tuple[int, int]], cell: tuple[int, int]) -> bool:
    r, c = cell
    for rr, cc in existing:
        if max(abs(int(r - rr)), abs(int(c - cc))) <= 1:
            return False
    return True


def _build_expected(
    shape: tuple[int, int],
    receivers: list[dict],
    tower_cells: list[tuple[int, int]],
    counts: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in tower_cells:
        out[int(r), int(c)] = TOWER
    occupied = set(tower_cells)
    for rid, receiver in enumerate(receivers):
        for r, c in _rotate_coords(receiver["coords"], receiver["center"], counts[rid] % 4):
            if (int(r), int(c)) in occupied:
                raise ValueError("rotated receiver overlaps a tower")
            out[int(r), int(c)] = int(receiver["color"])
    return out


def _sample_layout() -> tuple[int, int, list[tuple[int, int]], tuple | None] | None:
    height = random.randint(24, 30)
    width = random.randint(24, 30)
    row_slots = _slot_centers(height)
    col_slots = _slot_centers(width)

    options = []
    if min(len(row_slots), len(col_slots)) >= 3:
        options.append("independent")
    if len(row_slots) >= 2 and len(col_slots) >= 3:
        options.append("row_block")
    if len(row_slots) >= 3 and len(col_slots) >= 2:
        options.append("col_block")
    if not options:
        return None

    layout = random.choice(options)
    if layout == "independent":
        max_n = min(5, len(row_slots), len(col_slots))
        n = random.randint(3, max_n)
        rows = random.sample(row_slots, n)
        cols = random.sample(col_slots, n)
        random.shuffle(cols)
        return height, width, list(zip(rows, cols)), None

    if layout == "row_block":
        max_n = min(5, 2 + min(len(row_slots) - 1, len(col_slots) - 2))
        n = random.randint(3, max_n)
        shared_row = random.choice(row_slots)
        pair_cols = sorted(random.sample(col_slots, 2))
        centers = [(shared_row, pair_cols[0]), (shared_row, pair_cols[1])]
        extra = n - 2
        rows = random.sample([r for r in row_slots if r != shared_row], extra)
        cols = random.sample([c for c in col_slots if c not in pair_cols], extra)
        random.shuffle(cols)
        centers.extend(zip(rows, cols))
        return height, width, centers, ("row", 0, 1)

    max_n = min(5, 2 + min(len(row_slots) - 2, len(col_slots) - 1))
    n = random.randint(3, max_n)
    shared_col = random.choice(col_slots)
    pair_rows = sorted(random.sample(row_slots, 2))
    centers = [(pair_rows[0], shared_col), (pair_rows[1], shared_col)]
    extra = n - 2
    rows = random.sample([r for r in row_slots if r not in pair_rows], extra)
    cols = random.sample([c for c in col_slots if c != shared_col], extra)
    random.shuffle(cols)
    centers.extend(zip(rows, cols))
    return height, width, centers, ("col", 0, 1)


def _choose_turns(n: int) -> list[int]:
    base = [0, 1, 2, 3]
    random.shuffle(base)
    turns = base[: min(n, 4)]
    while len(turns) < n:
        turns.append(random.choice(base))
    random.shuffle(turns)
    return turns


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    height, width, centers, blocking_info = layout
    n_receivers = len(centers)

    colors = random.sample(RECEIVER_COLORS, n_receivers)
    receivers: list[dict] = []
    for color, center in zip(colors, centers):
        template = random.choice(RECEIVER_TEMPLATES)
        initial_turns = random.randint(0, 3)
        mask = np.rot90(template, -initial_turns)
        receivers.append(
            {
                "color": int(color),
                "center": (int(center[0]), int(center[1])),
                "coords": _mask_to_coords(mask, center),
            }
        )

    if not _receiver_clear(receivers):
        return None

    owner = _owner_grid((height, width), receivers)
    forbidden = _tower_forbidden((height, width), receivers)

    rays_by_receiver: list[dict[str, list[tuple[int, int]]]] = []
    visible_map: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for rid, receiver in enumerate(receivers):
        rays = _ray_cells((height, width), owner, rid, receiver["center"])
        rays_by_receiver.append(rays)
        for cells in rays.values():
            for cell in cells:
                if not bool(forbidden[cell]):
                    visible_map[cell].add(int(rid))

    exclusive_by_dir: list[dict[str, list[tuple[int, int]]]] = []
    exclusive_all: list[list[tuple[int, int]]] = []
    for rid, rays in enumerate(rays_by_receiver):
        per_dir: dict[str, list[tuple[int, int]]] = {}
        flat: list[tuple[int, int]] = []
        for name, cells in rays.items():
            good = [
                (int(r), int(c))
                for r, c in cells
                if not bool(forbidden[r, c]) and visible_map.get((int(r), int(c))) == {rid}
            ]
            per_dir[name] = good
            flat.extend(good)
        exclusive_by_dir.append(per_dir)
        exclusive_all.append(flat)

    if any(len(cells) < 3 for cells in exclusive_all):
        return None

    turns = _choose_turns(n_receivers)

    forced: tuple[int, str] | None = None
    if blocking_info is not None:
        axis, first, second = blocking_info
        candidates: list[tuple[int, str]] = []
        if axis == "row":
            left, right = sorted(
                [first, second],
                key=lambda idx: receivers[idx]["center"][1],
            )
            if len(exclusive_by_dir[left]["left"]) >= (4 if turns[left] == 0 else max(1, turns[left])):
                candidates.append((left, "left"))
            if len(exclusive_by_dir[right]["right"]) >= (4 if turns[right] == 0 else max(1, turns[right])):
                candidates.append((right, "right"))
        else:
            top, bottom = sorted(
                [first, second],
                key=lambda idx: receivers[idx]["center"][0],
            )
            if len(exclusive_by_dir[top]["up"]) >= (4 if turns[top] == 0 else max(1, turns[top])):
                candidates.append((top, "up"))
            if len(exclusive_by_dir[bottom]["down"]) >= (4 if turns[bottom] == 0 else max(1, turns[bottom])):
                candidates.append((bottom, "down"))
        if candidates:
            forced = random.choice(candidates)

    counts: list[int] = []
    for rid, turn in enumerate(turns):
        cap = min(6, len(exclusive_all[rid]))
        options = [value for value in (turn, turn + 4) if value <= cap]
        if rid == (forced[0] if forced is not None else -1):
            options = [value for value in options if value > 0]
        if not options:
            return None
        if len(options) == 2 and random.random() < 0.4:
            counts.append(int(options[1]))
        else:
            counts.append(int(options[0]))

    tower_cells: list[tuple[int, int]] = []
    order = sorted(range(n_receivers), key=lambda rid: (-counts[rid], rid))
    if forced is not None and forced[0] in order:
        order.remove(forced[0])
        order.insert(0, forced[0])

    for rid in order:
        need = counts[rid]
        if need == 0:
            continue
        chosen: list[tuple[int, int]] = []
        if forced is not None and rid == forced[0]:
            pool = exclusive_by_dir[rid][forced[1]].copy()
            random.shuffle(pool)
            for cell in pool:
                if _towers_nonadjacent(tower_cells, cell):
                    chosen.append(cell)
                    break
            if not chosen:
                return None
        pool = exclusive_all[rid].copy()
        random.shuffle(pool)
        for cell in pool:
            if cell in chosen:
                continue
            if not _towers_nonadjacent(tower_cells + chosen, cell):
                continue
            chosen.append(cell)
            if len(chosen) == need:
                break
        if len(chosen) != need:
            return None
        tower_cells.extend(chosen)

    grid = np.full((height, width), BG, dtype=int)
    for receiver in receivers:
        for r, c in receiver["coords"]:
            grid[int(r), int(c)] = int(receiver["color"])
    for r, c in tower_cells:
        grid[int(r), int(c)] = TOWER

    actual_receivers = _find_receivers(grid)
    if len(actual_receivers) != n_receivers:
        return None
    actual_owner = _owner_grid(grid.shape, actual_receivers)
    actual_counts = [
        _visible_tower_count(grid, actual_owner, rid, receiver["center"])
        for rid, receiver in enumerate(actual_receivers)
    ]
    expected_counts = [
        next(counts[idx] for idx, receiver in enumerate(receivers) if receiver["color"] == actual_receiver["color"])
        for actual_receiver in actual_receivers
    ]
    if actual_counts != expected_counts:
        return None

    try:
        expected = _build_expected(grid.shape, actual_receivers, tower_cells, actual_counts)
    except ValueError:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower receiver sample")
