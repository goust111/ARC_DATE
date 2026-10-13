from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_ROT = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
LEFT_TURN = {
    (-1, 0): (0, -1),
    (0, -1): (1, 0),
    (1, 0): (0, 1),
    (0, 1): (-1, 0),
}
RIGHT_TURN = {v: k for k, v in LEFT_TURN.items()}
INWARD_FROM_SIDE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
SIDES = tuple(INWARD_FROM_SIDE)
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BLACK) -> None:
    h, w = patch.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(dst.shape[0], r0 + h)
    cc1 = min(dst.shape[1], c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - r0
    pc0 = cc0 - c0
    patch_view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
    mask = patch_view != transparent
    dst[rr0:rr1, cc0:cc1][mask] = patch_view[mask]


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comp = np.zeros_like(mask, dtype=bool)
        for cr, cc in coords:
            comp[cr, cc] = True
        components.append(comp)
    return components


def _is_connected(mask: np.ndarray) -> bool:
    return np.count_nonzero(mask) > 0 and len(_component_masks(mask)) == 1


def _boundary_cells(shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r in range(h):
        out.add((r, 0))
        out.add((r, w - 1))
    for c in range(w):
        out.add((0, c))
        out.add((h - 1, c))
    return out


def _trace_open_polyline(mask: np.ndarray) -> list[tuple[int, int]]:
    assert mask.ndim == 2
    pts = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert pts
    degrees = {}
    for r, c in pts:
        neigh = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in pts]
        assert len(neigh) <= 2
        degrees[(r, c)] = neigh
    endpoints = [pt for pt, neigh in degrees.items() if len(neigh) == 1]
    assert len(endpoints) == 2
    boundary = _boundary_cells(mask.shape)
    boundary_endpoints = [pt for pt in endpoints if pt in boundary]
    assert len(boundary_endpoints) == 1

    start = boundary_endpoints[0]
    path = [start]
    prev = None
    cur = start
    while True:
        nxts = degrees[cur]
        if prev is not None:
            nxts = [pt for pt in nxts if pt != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    assert len(path) == len(pts)
    assert len(set(path)) == len(path)
    return path


def _turn_sign(incoming: tuple[int, int], outgoing: tuple[int, int]) -> str:
    if outgoing == incoming:
        return "straight"
    if outgoing == LEFT_TURN[incoming]:
        return "left"
    if outgoing == RIGHT_TURN[incoming]:
        return "right"
    raise AssertionError((incoming, outgoing))


def _extract_template(grid: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int, int, int]]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GRAY))
    assert len(colors) == 1
    color = colors[0]
    bbox = _bbox(grid == color)
    assert bbox is not None
    template = _crop_bbox(grid, bbox)
    assert set(map(int, np.unique(template))).issubset({BLACK, color})
    return template, color, bbox


def _place_stamp_at_bend(out: np.ndarray, template: np.ndarray, bend: tuple[int, int], incoming: tuple[int, int], turn: str) -> None:
    stamp = np.rot90(template, k=DIR_TO_ROT[incoming])
    if turn == "right":
        stamp = np.fliplr(stamp)
    r, c = bend
    top = r - stamp.shape[0] // 2
    left = c - stamp.shape[1] // 2
    _paste(out, stamp, top, left, transparent=BLACK)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the template onto every bend of the gray open polyline, orienting each copy by the incoming direction and turn sign.

    Input structure:
    - A black grid contains one gray one-pixel-wide open orthogonal polyline whose pixels all have degree at most 2.
    - Exactly one endpoint of that polyline lies on the grid boundary.
    - Away from the polyline there is one monochrome stamp template with odd bbox height and width.

    Rule:
    - Start from the boundary endpoint and trace the polyline in order.
    - At each bend pixel, center the template bbox on that bend.
    - Rotate the template so its original pose corresponds to entering the bend while moving downward, then keep it as-is for a left turn or apply a left-right mirror for a right turn.
    - Overlay only the non-black stamp pixels, erase the original template from the scene, and keep the polyline itself.

    Key insight:
    - The stamp pose is controlled by two independent clues at every bend: the incoming segment chooses the rotation, and the outgoing turn direction chooses whether a mirror is applied afterward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    line_mask = g == GRAY
    path = _trace_open_polyline(line_mask)
    template, stamp_color, bbox = _extract_template(g)
    assert stamp_color != BLACK and stamp_color != GRAY
    assert template.shape[0] % 2 == 1 and template.shape[1] % 2 == 1

    out = g.copy()
    out[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = BLACK
    out[line_mask] = GRAY

    for i in range(1, len(path) - 1):
        prev_pt = path[i - 1]
        bend = path[i]
        next_pt = path[i + 1]
        incoming = (bend[0] - prev_pt[0], bend[1] - prev_pt[1])
        outgoing = (next_pt[0] - bend[0], next_pt[1] - bend[1])
        turn = _turn_sign(incoming, outgoing)
        if turn == "straight":
            continue
        _place_stamp_at_bend(out, template, bend, incoming, turn)

    out[line_mask] = GRAY
    return out


def _template_has_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _template_asymmetry_ok(mask: np.ndarray) -> bool:
    if not _template_has_full_bbox(mask):
        return False
    for k in (1, 2, 3):
        if mask.shape == np.rot90(mask, k).shape and np.array_equal(mask, np.rot90(mask, k)):
            return False
    if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
        return False
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            return False
        if np.array_equal(mask, np.fliplr(mask).T):
            return False
    return True


def _random_template() -> np.ndarray:
    while True:
        h = random.choice((3, 5))
        w = random.choice((3, 5))
        density = random.uniform(0.35, 0.72)
        mask = np.random.rand(h, w) < density
        if np.count_nonzero(mask) < 4 or np.count_nonzero(mask) > h * w - 2:
            continue
        if not _is_connected(mask):
            continue
        if not _template_asymmetry_ok(mask):
            continue
        color = random.choice(STAMP_COLORS)
        return np.where(mask, color, BLACK).astype(int)


def _segment_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = start, end
    assert r0 == r1 or c0 == c1
    cells = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.append((r0, c))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.append((r, c0))
    return cells


def _build_path_from_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path = [vertices[0]]
    for i in range(1, len(vertices)):
        seg = _segment_cells(vertices[i - 1], vertices[i])
        path.extend(seg[1:])
    return path


def _polyline_grid(shape: tuple[int, int], path: list[tuple[int, int]]) -> np.ndarray:
    g = np.full(shape, BLACK, dtype=int)
    for r, c in path:
        g[r, c] = GRAY
    return g


def _candidate_lengths(
    shape: tuple[int, int],
    cur: tuple[int, int],
    direction: tuple[int, int],
    min_len: int,
    max_len: int,
    safe_margin: int,
    is_bend_end: bool,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int]]]:
    h, w = shape
    candidates = []
    for length in range(min_len, max_len + 1):
        end = (cur[0] + direction[0] * length, cur[1] + direction[1] * length)
        er, ec = end
        if not _in_bounds(shape, er, ec):
            continue
        if is_bend_end:
            if not (safe_margin <= er < h - safe_margin and safe_margin <= ec < w - safe_margin):
                continue
        else:
            if er in (0, h - 1) or ec in (0, w - 1):
                continue
        seg = _segment_cells(cur, end)[1:]
        if any(cell in occupied for cell in seg):
            continue
        if any(cell[0] in (0, h - 1) or cell[1] in (0, w - 1) for cell in seg[:-1]):
            continue
        valid = True
        chain = [cur] + seg
        chain_set = set(chain)
        for idx, cell in enumerate(seg, start=1):
            allowed = {chain[idx - 1]}
            if idx + 1 < len(chain):
                allowed.add(chain[idx + 1])
            for nbr in _neighbors4(*cell):
                if nbr in occupied and nbr not in allowed:
                    valid = False
                    break
                if nbr in chain_set and nbr not in allowed and nbr != cell:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        candidates.append((length, end))
    random.shuffle(candidates)
    return candidates


def _random_turn_sequence(num_bends: int) -> list[str]:
    while True:
        seq = [random.choice(("left", "right")) for _ in range(num_bends)]
        if "left" in seq and "right" in seq:
            return seq


def _generate_vertices(shape: tuple[int, int], template: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = shape
    safe_margin = max(template.shape) // 2
    num_bends = random.randint(3, 5)
    num_segments = num_bends + 1
    min_seg = max(template.shape)
    max_seg = min(8, max(6, min(h, w) - 2))
    turns = _random_turn_sequence(num_bends)
    side = random.choice(SIDES)
    direction = INWARD_FROM_SIDE[side]

    if side in ("top", "bottom"):
        c = random.randint(safe_margin, w - 1 - safe_margin)
        start = (0, c) if side == "top" else (h - 1, c)
    else:
        r = random.randint(safe_margin, h - 1 - safe_margin)
        start = (r, 0) if side == "left" else (r, w - 1)

    def rec(
        seg_index: int,
        cur: tuple[int, int],
        cur_dir: tuple[int, int],
        occupied: set[tuple[int, int]],
        vertices: list[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        is_bend_end = seg_index < num_segments - 1
        lengths = _candidate_lengths(
            shape,
            cur,
            cur_dir,
            min_seg if seg_index > 0 else safe_margin + 1,
            max_seg,
            safe_margin,
            is_bend_end,
            occupied,
        )
        for _length, end in lengths:
            seg_cells = _segment_cells(cur, end)[1:]
            new_occupied = occupied | set(seg_cells)
            new_vertices = vertices + [end]
            if seg_index == num_segments - 1:
                return new_vertices
            next_dir = LEFT_TURN[cur_dir] if turns[seg_index] == "left" else RIGHT_TURN[cur_dir]
            result = rec(seg_index + 1, end, next_dir, new_occupied, new_vertices)
            if result is not None:
                return result
        return None

    vertices = rec(0, start, direction, {start}, [start])
    if vertices is None:
        return None

    path = _build_path_from_vertices(vertices)
    traced = _trace_open_polyline(_polyline_grid(shape, path) == GRAY)
    if traced != path:
        return None
    incoming_dirs = set()
    turn_kinds = set()
    for i in range(1, len(path) - 1):
        incoming = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        outgoing = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        turn = _turn_sign(incoming, outgoing)
        if turn != "straight":
            incoming_dirs.add(incoming)
            turn_kinds.add(turn)
    if len(turn_kinds) < 2 or len(incoming_dirs) < 2:
        return None
    return vertices


def _stamp_box(center: tuple[int, int], stamp_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    h, w = stamp_shape
    return r - h // 2, c - w // 2, r + h // 2, c + w // 2


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _stamp_boxes_ok(shape: tuple[int, int], path: list[tuple[int, int]], template: np.ndarray) -> bool:
    boxes = []
    for i in range(1, len(path) - 1):
        bend = path[i]
        incoming = (bend[0] - path[i - 1][0], bend[1] - path[i - 1][1])
        outgoing = (path[i + 1][0] - bend[0], path[i + 1][1] - bend[1])
        turn = _turn_sign(incoming, outgoing)
        if turn == "straight":
            continue
        stamp = np.rot90(template, k=DIR_TO_ROT[incoming])
        if turn == "right":
            stamp = np.fliplr(stamp)
        box = _stamp_box(bend, stamp.shape)
        if not (0 <= box[0] and 0 <= box[1] and box[2] < shape[0] and box[3] < shape[1]):
            return False
        if any(_boxes_overlap(box, prev) for prev in boxes):
            return False
        boxes.append(box)
    return len(boxes) >= 3


def _template_location_ok(
    shape: tuple[int, int],
    top: int,
    left: int,
    template: np.ndarray,
    line_mask: np.ndarray,
) -> bool:
    h, w = template.shape
    if top < 0 or left < 0 or top + h > shape[0] or left + w > shape[1]:
        return False
    for r in range(h):
        for c in range(w):
            if template[r, c] == BLACK:
                continue
            gr = top + r
            gc = left + c
            if line_mask[gr, gc]:
                return False
            for nr in range(gr - 1, gr + 2):
                for nc in range(gc - 1, gc + 2):
                    if _in_bounds(shape, nr, nc) and line_mask[nr, nc]:
                        return False
    return True


def _place_template(shape: tuple[int, int], template: np.ndarray, line_mask: np.ndarray) -> tuple[int, int] | None:
    h, w = template.shape
    options = []
    for top in range(shape[0] - h + 1):
        for left in range(shape[1] - w + 1):
            if _template_location_ok(shape, top, left, template, line_mask):
                options.append((top, left))
    if not options:
        return None
    return random.choice(options)


def _construct_expected(inp: np.ndarray, template: np.ndarray, template_bbox: tuple[int, int, int, int], path: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    r0, c0, r1, c1 = template_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for i in range(1, len(path) - 1):
        bend = path[i]
        incoming = (bend[0] - path[i - 1][0], bend[1] - path[i - 1][1])
        outgoing = (path[i + 1][0] - bend[0], path[i + 1][1] - bend[1])
        turn = _turn_sign(incoming, outgoing)
        if turn == "straight":
            continue
        _place_stamp_at_bend(out, template, bend, incoming, turn)
    for r, c in path:
        out[r, c] = GRAY
    return out


def _validate_generated(inp: np.ndarray, path: list[tuple[int, int]], template: np.ndarray, template_bbox: tuple[int, int, int, int]) -> None:
    assert inp.ndim == 2 and max(inp.shape) <= 30
    line_mask = inp == GRAY
    traced = _trace_open_polyline(line_mask)
    assert traced == path
    assert _stamp_boxes_ok(inp.shape, path, template)
    assert template.shape[0] % 2 == 1 and template.shape[1] % 2 == 1
    assert _template_asymmetry_ok(template != BLACK)
    template_color = int(template[template != BLACK][0])
    assert np.array_equal(_crop_bbox(inp, template_bbox), template)
    expected = _construct_expected(inp, template, template_bbox, path)
    actual = main(inp)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, inp)
    assert np.count_nonzero(actual == template_color) >= np.count_nonzero(template != BLACK)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        template = _random_template()
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        shape = (h, w)
        vertices = _generate_vertices(shape, template)
        if vertices is None:
            continue
        path = _build_path_from_vertices(vertices)
        if not _stamp_boxes_ok(shape, path, template):
            continue

        line_mask = _polyline_grid(shape, path) == GRAY
        loc = _place_template(shape, template, line_mask)
        if loc is None:
            continue

        inp = _polyline_grid(shape, path)
        _paste(inp, template, loc[0], loc[1], transparent=BLACK)
        template_bbox = (loc[0], loc[1], loc[0] + template.shape[0] - 1, loc[1] + template.shape[1] - 1)

        try:
            _validate_generated(inp, path, template, template_bbox)
        except Exception:
            continue
        return inp
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
