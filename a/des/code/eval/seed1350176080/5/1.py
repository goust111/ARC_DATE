import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
MAROON = 9
COLOR_POOL = (1, 3, 4, 6, 7, 8)

GEN_RETRIES = 300

LEGEND_ROW_H = 7
LEGEND_ROW_W = 11
LEGEND_LEFT_W = 5
LEGEND_SEP_COL = 6
LEGEND_GAP = 1

ROOM_COUNT_RANGE = (3, 4)
GRID_H_RANGE = (26, 34)
GRID_W_RANGE = (34, 42)

TOPOLOGIES = ("H", "V", "L")
DOMINO_SHAPES = ((1, 2), (2, 1))

STAMP_TEMPLATES = (
    np.array([[0, 1, 2], [1, 2, 1], [3, 1, 2]], dtype=int),
    np.array([[0, 1, 0], [2, 3, 2], [1, 2, 3]], dtype=int),
    np.array([[0, 2, 1], [3, 1, 2], [2, 3, 1]], dtype=int),
    np.array([[0, 1, 2], [2, 0, 1], [1, 3, 2]], dtype=int),
    np.array([[0, 2, 1], [1, 3, 0], [2, 1, 3]], dtype=int),
    np.array([[0, 1, 2], [3, 2, 1], [1, 0, 3]], dtype=int),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _cluster(values: list[float], tol: float = 0.75) -> tuple[list[int], int]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    group_ids = [0] * len(values)
    group_count = 0
    last_value = None
    for idx in order:
        v = values[idx]
        if last_value is None or v - last_value > tol:
            group_count += 1
        group_ids[idx] = group_count - 1
        last_value = v
    return group_ids, group_count


def _signature_from_area(area: np.ndarray) -> str:
    mask = (area != BG) & (area != GREY) & (area != RED)
    comps = _components(mask)
    assert 2 <= len(comps) <= 3, f"expected 2-3 disconnected blocks, got {len(comps)}"

    centers = []
    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        centers.append(((r0 + r1) / 2.0, (c0 + c1) / 2.0))

    row_ids, row_count = _cluster([r for r, _ in centers])
    col_ids, col_count = _cluster([c for _, c in centers])

    if len(comps) == 2:
        if row_count == 1 and col_count >= 2:
            return "H"
        if col_count == 1 and row_count >= 2:
            return "V"
    else:
        if row_count == 1:
            return "H"
        if col_count == 1:
            return "V"
        coords = {(row_ids[i], col_ids[i]) for i in range(len(comps))}
        if row_count == 2 and col_count == 2 and len(coords) == 3:
            return "L"

    row_span = max(r for r, _ in centers) - min(r for r, _ in centers)
    col_span = max(c for _, c in centers) - min(c for _, c in centers)
    if len(comps) == 3 and row_count >= 2 and col_count >= 2:
        return "L"
    return "H" if col_span >= row_span else "V"


def _crop_non_bg(area: np.ndarray) -> np.ndarray:
    rows = np.any(area != BG, axis=1)
    cols = np.any(area != BG, axis=0)
    assert rows.any() and cols.any(), "expected non-background pixels"
    return area[np.ix_(rows, cols)]


def _grey_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if (
            np.all(sub[0, :] == GREY)
            and np.all(sub[-1, :] == GREY)
            and np.all(sub[:, 0] == GREY)
            and np.all(sub[:, -1] == GREY)
        ):
            boxes.append((r0, c0, r1, c1))
    return sorted(boxes)


def _adjacent_reds(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    reds: list[tuple[int, int]] = []
    if r0 - 1 >= 0:
        reds.extend([(r0 - 1, c) for c in range(c0, c1 + 1) if grid[r0 - 1, c] == RED])
    if r1 + 1 < h:
        reds.extend([(r1 + 1, c) for c in range(c0, c1 + 1) if grid[r1 + 1, c] == RED])
    if c0 - 1 >= 0:
        reds.extend([(r, c0 - 1) for r in range(r0, r1 + 1) if grid[r, c0 - 1] == RED])
    if c1 + 1 < w:
        reds.extend([(r, c1 + 1) for r in range(r0, r1 + 1) if grid[r, c1 + 1] == RED])
    return sorted(set(reds))


def _separator_cols(sub: np.ndarray) -> list[int]:
    return [c for c in range(1, sub.shape[1] - 1) if np.all(sub[:, c] == GREY)]


def _parse_legend_row(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, np.ndarray]:
    r0, c0, r1, c1 = box
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    seps = _separator_cols(sub)
    assert seps, "legend row must contain an interior grey separator"
    sep = seps[0]
    left = sub[1:-1, 1:sep]
    right = sub[1:-1, sep + 1 : -1]
    sig = _signature_from_area(left)
    stamp = _crop_non_bg(right)
    assert stamp.shape == (3, 3), f"expected 3x3 stamp, got {stamp.shape}"
    return sig, stamp


def _parse_room(grid: np.ndarray, box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = box
    area = grid[r0 + 1 : r1, c0 + 1 : c1]
    return _signature_from_area(area)


def _parse_layout(grid: np.ndarray) -> tuple[list[tuple[tuple[int, int, int, int], tuple[int, int]]], dict[str, np.ndarray]]:
    rooms: list[tuple[tuple[int, int, int, int], tuple[int, int]]] = []
    legend_rows: list[tuple[int, int, int, int]] = []
    for box in _grey_boxes(grid):
        reds = _adjacent_reds(grid, box)
        sub = grid[box[0] : box[2] + 1, box[1] : box[3] + 1]
        if reds:
            rooms.append((box, reds[0]))
        elif _separator_cols(sub):
            legend_rows.append(box)

    mapping: dict[str, np.ndarray] = {}
    for box in legend_rows:
        sig, stamp = _parse_legend_row(grid, box)
        mapping[sig] = stamp
    return rooms, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each room's internal block topology with the legend row showing the same arrangement,
    then stamp the corresponding 3x3 pattern at the room's red anchor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rooms, mapping = _parse_layout(g)
    assert rooms, "expected at least one room"
    assert set(mapping) == set(TOPOLOGIES), f"expected legend for H/V/L, got {sorted(mapping)}"

    h, w = g.shape
    for box, anchor in rooms:
        sig = _parse_room(g, box)
        stamp = mapping[sig]
        ar, ac = anchor
        for sr in range(stamp.shape[0]):
            for sc in range(stamp.shape[1]):
                rr, cc = ar + sr, ac + sc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if stamp[sr, sc] != BG and out[rr, cc] == BG:
                    out[rr, cc] = int(stamp[sr, sc])
    return out


def _colorize_template(template: np.ndarray) -> np.ndarray:
    ids = sorted(int(v) for v in np.unique(template) if v != 0)
    colors = random.sample(COLOR_POOL, len(ids))
    remap = {src: dst for src, dst in zip(ids, colors)}
    out = np.zeros_like(template)
    for src, dst in remap.items():
        out[template == src] = dst
    return out


def _sample_pattern(h: int, w: int, topology: str, colors: list[int]) -> np.ndarray | None:
    target_blocks = 2 if topology in {"H", "V"} else 3
    fill_colors = [colors[0]] * target_blocks if len(colors) == 1 else list(colors[:target_blocks])

    def cells_for(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
        sh, sw = shape
        return [(r + dr, c + dc) for dr in range(sh) for dc in range(sw)]

    for _ in range(1200):
        area = np.zeros((h, w), dtype=int)
        occupied: set[tuple[int, int]] = set()
        ok = True
        for idx in range(target_blocks):
            shape = random.choice(DOMINO_SHAPES)
            sh, sw = shape
            choices = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
            random.shuffle(choices)
            placed = False
            for r, c in choices:
                pts = cells_for(r, c, shape)
                neighbors = set(pts)
                for pr, pc in pts:
                    neighbors.update({(pr + 1, pc), (pr - 1, pc), (pr, pc + 1), (pr, pc - 1)})
                if any(pt in occupied for pt in neighbors):
                    continue
                for pr, pc in pts:
                    area[pr, pc] = fill_colors[idx]
                occupied.update(pts)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and len(_components(area != BG)) == target_blocks and _signature_from_area(area) == topology:
            return area
    return None


def _make_legend_row(topology: str, stamp: np.ndarray) -> np.ndarray:
    row = np.zeros((LEGEND_ROW_H, LEGEND_ROW_W), dtype=int)
    row[0, :] = GREY
    row[-1, :] = GREY
    row[:, 0] = GREY
    row[:, -1] = GREY
    row[:, LEGEND_SEP_COL] = GREY

    left = _sample_pattern(5, LEGEND_LEFT_W, topology, [MAROON])
    assert left is not None
    row[1:6, 1:1 + LEGEND_LEFT_W] = left
    row[2:5, 7:10] = stamp
    return row


def _make_room(topology: str) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(300):
        if topology == "L":
            inner_h = random.randint(5, 6)
            inner_w = random.randint(5, 6)
        else:
            inner_h = random.randint(4, 6)
            inner_w = random.randint(4, 6)
        n_blocks = 2 if topology in {"H", "V"} else 3
        colors = random.sample(COLOR_POOL, n_blocks)
        inner = _sample_pattern(inner_h, inner_w, topology, colors)
        if inner is None:
            continue

        room = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
        room[0, :] = GREY
        room[-1, :] = GREY
        room[:, 0] = GREY
        room[:, -1] = GREY
        room[1:-1, 1:-1] = inner

        if random.random() < 0.5:
            anchor = (random.randint(0, room.shape[0] - 1), room.shape[1])
        else:
            anchor = (room.shape[0], random.randint(0, room.shape[1] - 1))
        return room, anchor
    raise RuntimeError(f"failed to make room for topology {topology}")


def _footprint(room_shape: tuple[int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    room_h, room_w = room_shape
    ar, ac = anchor
    return max(room_h, ar + 3), max(room_w, ac + 3)


def _mark(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    occ[r0 : r1 + 1, c0 : c1 + 1] = True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    legend_total_h = 3 * LEGEND_ROW_H + 2 * LEGEND_GAP
    if h <= legend_total_h + 2 or w <= LEGEND_ROW_W + 14:
        return None

    legend_top = random.randint(1, h - legend_total_h - 1)
    legend_left = random.randint(w - LEGEND_ROW_W - 3, w - LEGEND_ROW_W - 1)

    chosen_templates = random.sample(STAMP_TEMPLATES, 3)
    stamps = {topo: _colorize_template(tpl.copy()) for topo, tpl in zip(TOPOLOGIES, chosen_templates)}
    legend_order = list(TOPOLOGIES)
    random.shuffle(legend_order)
    for idx, topo in enumerate(legend_order):
        row_patch = _make_legend_row(topo, stamps[topo])
        r0 = legend_top + idx * (LEGEND_ROW_H + LEGEND_GAP)
        c0 = legend_left
        grid[r0 : r0 + LEGEND_ROW_H, c0 : c0 + LEGEND_ROW_W] = row_patch
        _mark(
            occ,
            max(0, r0 - 1),
            max(0, c0 - 1),
            min(h - 1, r0 + LEGEND_ROW_H),
            min(w - 1, c0 + LEGEND_ROW_W),
        )

    room_count = random.randint(*ROOM_COUNT_RANGE)
    topologies = list(TOPOLOGIES)
    if room_count == 4:
        topologies.append(random.choice(TOPOLOGIES))
    random.shuffle(topologies)

    for topo in topologies:
        placed = False
        room, anchor = _make_room(topo)
        room_h, room_w = room.shape
        foot_h, foot_w = _footprint(room.shape, anchor)
        max_top = h - foot_h - 1
        max_left = legend_left - foot_w - 2
        if max_top < 1 or max_left < 1:
            return None

        for _ in range(300):
            r0 = random.randint(1, max_top)
            c0 = random.randint(1, max_left)
            r1 = r0 + foot_h - 1
            c1 = c0 + foot_w - 1
            pr0, pc0 = max(0, r0 - 1), max(0, c0 - 1)
            pr1, pc1 = min(h - 1, r1 + 1), min(w - 1, c1 + 1)
            if occ[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue

            grid[r0 : r0 + room_h, c0 : c0 + room_w] = room
            ar, ac = anchor
            grid[r0 + ar, c0 + ac] = RED
            _mark(occ, pr0, pc0, pr1, pc1)
            placed = True
            break
        if not placed:
            return None

    rooms, mapping = _parse_layout(grid)
    if len(rooms) != room_count or set(mapping) != set(TOPOLOGIES):
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
