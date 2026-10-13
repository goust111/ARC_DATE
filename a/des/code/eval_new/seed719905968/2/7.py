from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
DIRS = ("up", "down", "left", "right")
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_CANVAS = 30
GEN_ATTEMPTS = 500
GEN_SYSTEM_COUNT = (1, 3)
GEN_DISTRACTOR_FRAMES = (0, 3)


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _border_coords(offset: int, side: int) -> list[tuple[int, int]]:
    end = offset + side - 1
    coords = [(offset, c) for c in range(offset, end + 1)]
    coords += [(r, end) for r in range(offset + 1, end)]
    coords += [(end, c) for c in range(end, offset - 1, -1)]
    coords += [(r, offset) for r in range(end - 1, offset, -1)]
    return coords


def _flip_bbox_h(bbox: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0, width - 1 - c1, r1, width - 1 - c0


def _flip_bbox_v(bbox: tuple[int, int, int, int], height: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return height - 1 - r1, c0, height - 1 - r0, c1


def _shift_bbox(
    bbox: tuple[int, int, int, int], dr: int, dc: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _inner_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _sort_bboxes_along(
    boxes: list[tuple[int, int, int, int]], direction: str
) -> list[tuple[int, int, int, int]]:
    if direction == "right":
        return sorted(boxes, key=lambda b: (b[1] + b[3], b[0] + b[2]))
    if direction == "left":
        return sorted(boxes, key=lambda b: (-(b[1] + b[3]), b[0] + b[2]))
    if direction == "down":
        return sorted(boxes, key=lambda b: (b[0] + b[2], b[1] + b[3]))
    if direction == "up":
        return sorted(boxes, key=lambda b: (-(b[0] + b[2]), b[1] + b[3]))
    raise AssertionError(direction)


def _shell_info(
    sub: np.ndarray, offset: int, side: int
) -> tuple[int, str, int] | None:
    border = _border_coords(offset, side)
    values = [int(sub[r, c]) for r, c in border]
    non_bg = {value for value in values if value != BG}
    if len(non_bg) != 1:
        return None
    color = non_bg.pop()
    if color == GREY:
        return None

    holes = [(r, c) for r, c in border if int(sub[r, c]) == BG]
    if len(holes) != 1:
        return None

    hr, hc = holes[0]
    end = offset + side - 1
    if hr == offset and offset < hc < end:
        direction = "up"
        axis = hc
    elif hr == end and offset < hc < end:
        direction = "down"
        axis = hc
    elif hc == offset and offset < hr < end:
        direction = "left"
        axis = hr
    elif hc == end and offset < hr < end:
        direction = "right"
        axis = hr
    else:
        return None

    if any(value not in (BG, color) for value in values):
        return None
    return color, direction, axis


def _parse_receivers(grid: np.ndarray) -> list[dict[str, object]]:
    receivers: list[dict[str, object]] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        side = r1 - r0 + 1
        if side != c1 - c0 + 1 or side < 3:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(sub[0, :] == GREY)
            and np.all(sub[-1, :] == GREY)
            and np.all(sub[:, 0] == GREY)
            and np.all(sub[:, -1] == GREY)
        ):
            continue
        if not np.all(sub[1:-1, 1:-1] == BG):
            continue
        if len(coords) != 4 * side - 4:
            continue
        receivers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "shape": (side - 2, side - 2),
                "row2": r0 + r1,
                "col2": c0 + c1,
            }
        )
    return receivers


def _parse_ring_structure(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side >= 3

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    layers = 0
    direction = None
    axis = None
    ring_cells = np.zeros_like(sub, dtype=bool)

    while side - 2 * layers >= 3:
        shell_side = side - 2 * layers
        info = _shell_info(sub, layers, shell_side)
        if info is None:
            break
        _color, shell_dir, shell_axis = info
        if direction is None:
            direction = shell_dir
            axis = shell_axis
        elif direction != shell_dir or axis != shell_axis:
            break

        border = _border_coords(layers, shell_side)
        gap = {
            "up": (layers, axis),
            "down": (layers + shell_side - 1, axis),
            "left": (axis, layers),
            "right": (axis, layers + shell_side - 1),
        }[direction]
        for cell in border:
            if cell != gap:
                ring_cells[cell] = True
        layers += 1

    payload_side = side - 2 * layers
    assert 1 <= layers <= 3
    assert 1 <= payload_side <= 3
    assert direction is not None and axis is not None

    payload = sub[layers : side - layers, layers : side - layers].copy()
    assert payload.shape == (payload_side, payload_side)
    assert np.all((payload != BG) & (payload != GREY))

    expected_mask = ring_cells.copy()
    expected_mask[layers : side - layers, layers : side - layers] = True
    actual_mask = (sub != BG) & (sub != GREY)
    assert np.array_equal(actual_mask, expected_mask)

    return {
        "bbox": (r0, c0, r1, c1),
        "row2": r0 + r1,
        "col2": c0 + c1,
        "direction": direction,
        "layers": layers,
        "payload": payload,
    }


def _target_receiver(
    structure: dict[str, object], receivers: list[dict[str, object]]
) -> dict[str, object]:
    direction = str(structure["direction"])
    row2 = int(structure["row2"])
    col2 = int(structure["col2"])

    if direction == "right":
        candidates = [rec for rec in receivers if int(rec["row2"]) == row2 and int(rec["col2"]) > col2]
        candidates.sort(key=lambda rec: (int(rec["col2"]) - col2, int(rec["row2"])))
    elif direction == "left":
        candidates = [rec for rec in receivers if int(rec["row2"]) == row2 and int(rec["col2"]) < col2]
        candidates.sort(key=lambda rec: (col2 - int(rec["col2"]), int(rec["row2"])))
    elif direction == "down":
        candidates = [rec for rec in receivers if int(rec["col2"]) == col2 and int(rec["row2"]) > row2]
        candidates.sort(key=lambda rec: (int(rec["row2"]) - row2, int(rec["col2"])))
    else:
        candidates = [rec for rec in receivers if int(rec["col2"]) == col2 and int(rec["row2"]) < row2]
        candidates.sort(key=lambda rec: (row2 - int(rec["row2"]), int(rec["col2"])))

    index = int(structure["layers"]) - 1
    assert len(candidates) > index
    return candidates[index]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each ring payload into the receiver frame selected by its opening direction and layer count.

    Input structure:
    - The background is black.
    - Each non-grey object is a connected concentric square-ring structure with 1 to 3 layers, one aligned 1-pixel gap on a single side, and a fully colored 1x1 to 3x3 payload at the center.
    - Grey objects are hollow square receiver frames whose interiors are black and sized to hold one payload.

    Rule:
    - For every ring structure, detect the gap direction and count how many nested ring layers it has.
    - From the structure center, look only along that direction and order the aligned receiver frames by distance from nearest to farthest.
    - Copy the central payload into the interior of the L-th receiver frame, where L is the number of ring layers, leaving the source rings and the grey frame borders unchanged.

    Key insight:
    - Solving the task requires chaining three structural cues: the shared gap identifies a ray, the nesting depth gives which receiver on that ray to select, and the remaining center square is the payload to stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    receivers = _parse_receivers(g)
    assert receivers

    structures = [
        _parse_ring_structure(g, coords)
        for coords in _components((g != BG) & (g != GREY))
    ]
    assert structures

    out = g.copy()
    planned: dict[tuple[int, int], int] = {}

    for structure in structures:
        target = _target_receiver(structure, receivers)
        payload = np.asarray(structure["payload"], dtype=int)
        inner_r0, inner_c0, inner_r1, inner_c1 = target["inner_bbox"]
        assert payload.shape == tuple(target["shape"])
        assert inner_r1 - inner_r0 + 1 == payload.shape[0]
        assert inner_c1 - inner_c0 + 1 == payload.shape[1]

        for rr in range(payload.shape[0]):
            for cc in range(payload.shape[1]):
                pos = (inner_r0 + rr, inner_c0 + cc)
                value = int(payload[rr, cc])
                if pos in planned:
                    assert planned[pos] == value
                planned[pos] = value

    for (r, c), value in planned.items():
        out[r, c] = value
    return out


def _sample_payload(size: int, forbidden: set[int]) -> np.ndarray:
    colors = [color for color in PALETTE if color not in forbidden]
    use = random.sample(colors, random.randint(1, min(3, len(colors))))
    mode = random.choice(("uniform", "checker", "split", "random"))
    payload = np.empty((size, size), dtype=int)

    if size == 1 or mode == "uniform" or len(use) == 1:
        payload[:, :] = use[0]
        return payload

    if mode == "checker":
        for r in range(size):
            for c in range(size):
                payload[r, c] = use[(r + c) % len(use)]
        return payload

    if mode == "split":
        payload[:, :] = use[0]
        if random.random() < 0.5:
            cut = random.randint(1, size - 1)
            for idx in range(cut, size):
                payload[idx, :] = use[(idx - cut + 1) % len(use)]
        else:
            cut = random.randint(1, size - 1)
            for idx in range(cut, size):
                payload[:, idx] = use[(idx - cut + 1) % len(use)]
        return payload

    for r in range(size):
        for c in range(size):
            payload[r, c] = random.choice(use)
    if len(set(map(int, payload.flat))) == 1:
        payload[0, -1] = use[1]
    return payload


def _build_ring(payload: np.ndarray, layers: int, direction: str, ring_colors: list[int]) -> np.ndarray:
    payload_size = payload.shape[0]
    side = payload_size + 2 * layers
    patch = np.zeros((side, side), dtype=int)
    patch[layers : layers + payload_size, layers : layers + payload_size] = payload
    axis = (side - 1) // 2

    for depth, color in enumerate(ring_colors):
        end = side - depth - 1
        patch[depth, depth : end + 1] = color
        patch[end, depth : end + 1] = color
        patch[depth : end + 1, depth] = color
        patch[depth : end + 1, end] = color
        if direction == "up":
            patch[depth, axis] = BG
        elif direction == "down":
            patch[end, axis] = BG
        elif direction == "left":
            patch[axis, depth] = BG
        else:
            patch[axis, end] = BG

    return patch


def _receiver_patch(payload_size: int) -> np.ndarray:
    side = payload_size + 2
    patch = np.zeros((side, side), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    return patch


def _random_gaps(count: int, slack: int) -> list[int]:
    gaps = [1] * count
    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        if slack <= 0:
            break
        extra = random.randint(0, min(2, slack))
        gaps[idx] += extra
        slack -= extra
    return gaps


def _build_horizontal_system(direction: str, payload_size: int, layers: int) -> dict[str, object] | None:
    src_side = payload_size + 2 * layers
    rec_side = payload_size + 2
    max_receivers = min(5, (GEN_CANVAS - src_side) // (rec_side + 1))
    if max_receivers < layers:
        return None

    receiver_count = random.randint(layers, min(layers + 2, max_receivers))
    min_width = src_side + receiver_count * rec_side + receiver_count
    gaps = _random_gaps(receiver_count, GEN_CANVAS - min_width)

    ring_colors = random.sample(PALETTE, layers)
    payload = _sample_payload(payload_size, set(ring_colors) | {GREY, BG})
    source = _build_ring(payload, layers, "right", ring_colors)
    receiver = _receiver_patch(payload_size)

    recv_top = layers - 1
    lefts: list[int] = []
    cursor = src_side
    for gap in gaps:
        cursor += gap
        lefts.append(cursor)
        cursor += rec_side

    patch = np.zeros((src_side, cursor), dtype=int)
    patch[:src_side, :src_side] = source
    receiver_boxes: list[tuple[int, int, int, int]] = []
    for left in lefts:
        patch[recv_top : recv_top + rec_side, left : left + rec_side] = receiver
        receiver_boxes.append((recv_top, left, recv_top + rec_side - 1, left + rec_side - 1))

    source_bbox = (0, 0, src_side - 1, src_side - 1)
    if direction == "left":
        patch = np.fliplr(patch)
        width = patch.shape[1]
        source_bbox = _flip_bbox_h(source_bbox, width)
        receiver_boxes = [_flip_bbox_h(box, width) for box in receiver_boxes]

    ordered = _sort_bboxes_along(receiver_boxes, direction)
    return {
        "patch": patch,
        "direction": direction,
        "layers": layers,
        "payload": payload,
        "source_bbox": source_bbox,
        "receiver_boxes": ordered,
        "target_box": ordered[layers - 1],
    }


def _build_vertical_system(direction: str, payload_size: int, layers: int) -> dict[str, object] | None:
    src_side = payload_size + 2 * layers
    rec_side = payload_size + 2
    max_receivers = min(5, (GEN_CANVAS - src_side) // (rec_side + 1))
    if max_receivers < layers:
        return None

    receiver_count = random.randint(layers, min(layers + 2, max_receivers))
    min_height = src_side + receiver_count * rec_side + receiver_count
    gaps = _random_gaps(receiver_count, GEN_CANVAS - min_height)

    ring_colors = random.sample(PALETTE, layers)
    payload = _sample_payload(payload_size, set(ring_colors) | {GREY, BG})
    source = _build_ring(payload, layers, "down", ring_colors)
    receiver = _receiver_patch(payload_size)

    recv_left = layers - 1
    tops: list[int] = []
    cursor = src_side
    for gap in gaps:
        cursor += gap
        tops.append(cursor)
        cursor += rec_side

    patch = np.zeros((cursor, src_side), dtype=int)
    patch[:src_side, :src_side] = source
    receiver_boxes: list[tuple[int, int, int, int]] = []
    for top in tops:
        patch[top : top + rec_side, recv_left : recv_left + rec_side] = receiver
        receiver_boxes.append((top, recv_left, top + rec_side - 1, recv_left + rec_side - 1))

    source_bbox = (0, 0, src_side - 1, src_side - 1)
    if direction == "up":
        patch = np.flipud(patch)
        height = patch.shape[0]
        source_bbox = _flip_bbox_v(source_bbox, height)
        receiver_boxes = [_flip_bbox_v(box, height) for box in receiver_boxes]

    ordered = _sort_bboxes_along(receiver_boxes, direction)
    return {
        "patch": patch,
        "direction": direction,
        "layers": layers,
        "payload": payload,
        "source_bbox": source_bbox,
        "receiver_boxes": ordered,
        "target_box": ordered[layers - 1],
    }


def _build_system(direction: str) -> dict[str, object] | None:
    layers = random.randint(1, 3)
    payload_size = random.randint(1, 3)
    if direction in ("left", "right"):
        return _build_horizontal_system(direction, payload_size, layers)
    return _build_vertical_system(direction, payload_size, layers)


def _system_positions(
    occ: np.ndarray,
    system: dict[str, object],
    used_row2s: set[int],
    used_col2s: set[int],
) -> list[tuple[int, int]]:
    patch = np.asarray(system["patch"], dtype=int)
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []

    for r0 in range(1, GEN_CANVAS - ph):
        for c0 in range(1, GEN_CANVAS - pw):
            if occ[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue

            rows = set()
            cols = set()
            for box in system["receiver_boxes"]:
                gr0, gc0, gr1, gc1 = _shift_bbox(box, r0, c0)
                rows.add(gr0 + gr1)
                cols.add(gc0 + gc1)

            direction = str(system["direction"])
            if direction in ("left", "right"):
                if len(rows) != 1:
                    continue
                if next(iter(rows)) in used_row2s:
                    continue
                if cols & used_col2s:
                    continue
            else:
                if len(cols) != 1:
                    continue
                if next(iter(cols)) in used_col2s:
                    continue
                if rows & used_row2s:
                    continue

            positions.append((r0, c0))
    return positions


def _place_system(
    grid: np.ndarray,
    expected: np.ndarray,
    occ: np.ndarray,
    system: dict[str, object],
    top: int,
    left: int,
    used_row2s: set[int],
    used_col2s: set[int],
) -> None:
    patch = np.asarray(system["patch"], dtype=int)
    ph, pw = patch.shape
    region = grid[top : top + ph, left : left + pw]
    region[patch != BG] = patch[patch != BG]
    expected[top : top + ph, left : left + pw][patch != BG] = patch[patch != BG]
    occ[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True

    for box in system["receiver_boxes"]:
        gr0, gc0, gr1, gc1 = _shift_bbox(box, top, left)
        used_row2s.add(gr0 + gr1)
        used_col2s.add(gc0 + gc1)

    tr0, tc0, tr1, tc1 = _inner_bbox(_shift_bbox(system["target_box"], top, left))
    payload = np.asarray(system["payload"], dtype=int)
    expected[tr0 : tr1 + 1, tc0 : tc1 + 1] = payload


def _frame_positions(
    occ: np.ndarray,
    size: int,
    used_row2s: set[int],
    used_col2s: set[int],
) -> list[tuple[int, int]]:
    side = size + 2
    positions: list[tuple[int, int]] = []
    for r0 in range(1, GEN_CANVAS - side):
        for c0 in range(1, GEN_CANVAS - side):
            if occ[r0 - 1 : r0 + side + 1, c0 - 1 : c0 + side + 1].any():
                continue
            row2 = 2 * r0 + side - 1
            col2 = 2 * c0 + side - 1
            if row2 in used_row2s or col2 in used_col2s:
                continue
            positions.append((r0, c0))
    return positions


def _place_frame(
    grid: np.ndarray,
    occ: np.ndarray,
    size: int,
    top: int,
    left: int,
    used_row2s: set[int],
    used_col2s: set[int],
) -> None:
    patch = _receiver_patch(size)
    side = patch.shape[0]
    grid[top : top + side, left : left + side] = patch
    occ[top - 1 : top + side + 1, left - 1 : left + side + 1] = True
    used_row2s.add(2 * top + side - 1)
    used_col2s.add(2 * left + side - 1)


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_extra = random.randint(0, int(r0))
    left_extra = random.randint(0, int(c0))
    bottom_extra = random.randint(0, GEN_CANVAS - 1 - int(r1))
    right_extra = random.randint(0, GEN_CANVAS - 1 - int(c1))
    rs = slice(int(r0) - top_extra, int(r1) + bottom_extra + 1)
    cs = slice(int(c0) - left_extra, int(c1) + right_extra + 1)
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _gen_attempt() -> np.ndarray | None:
    canvas = np.zeros((GEN_CANVAS, GEN_CANVAS), dtype=int)
    expected = canvas.copy()
    occ = np.zeros_like(canvas, dtype=bool)
    used_row2s: set[int] = set()
    used_col2s: set[int] = set()

    n_systems = random.randint(*GEN_SYSTEM_COUNT)
    if random.random() < 0.65 and n_systems <= len(DIRS):
        directions = random.sample(DIRS, n_systems)
    else:
        directions = [random.choice(DIRS) for _ in range(n_systems)]

    for direction in directions:
        system = _build_system(direction)
        if system is None:
            return None
        positions = _system_positions(occ, system, used_row2s, used_col2s)
        if not positions:
            return None
        top, left = random.choice(positions)
        _place_system(canvas, expected, occ, system, top, left, used_row2s, used_col2s)

    for _ in range(random.randint(*GEN_DISTRACTOR_FRAMES)):
        size = random.randint(1, 3)
        positions = _frame_positions(occ, size, used_row2s, used_col2s)
        if not positions:
            break
        top, left = random.choice(positions)
        _place_frame(canvas, occ, size, top, left, used_row2s, used_col2s)

    inp, out = _crop_pair(canvas, expected)
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
