from __future__ import annotations

import itertools
import math
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GUIDE_COLOR_POOL = [3, 4, 6, 7, 8, 9]
CENTER_COLOR_POOL = [3, 4, 5, 6, 7, 8, 9]
FRAME_SIZE = 5
VECTOR_BOUND = 8
GEN_ATTEMPTS = 400
VECTOR_SAMPLE_TRIES = 20000
CONNECT_TRIES = 60


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _clock_angle(center: tuple[float, float], ref: tuple[float, float]) -> float:
    dr = center[0] - ref[0]
    dc = center[1] - ref[1]
    return (math.atan2(dc, -dr) + 2 * math.pi) % (2 * math.pi)


def _angle_delta(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def _extract_center_sprite(grid: np.ndarray) -> dict:
    mask = (grid != BG) & (grid != BLUE) & (grid != RED)
    parts = _components(mask)
    assert parts
    ref = ((grid.shape[0] - 1) / 2.0, (grid.shape[1] - 1) / 2.0)

    def score(cells: list[tuple[int, int]]) -> tuple[float, int]:
        r0, c0, r1, c1 = _bbox(cells)
        center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        dist2 = (center[0] - ref[0]) ** 2 + (center[1] - ref[1]) ** 2
        return dist2, -len(cells)

    cells = min(parts, key=score)
    r0, c0, r1, c1 = _bbox(cells)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    rel = [(r - r0, c - c0) for r, c in cells]
    colors = {(r - r0, c - c0): int(grid[r, c]) for r, c in cells}
    return {
        "bbox": (r0, c0, r1, c1),
        "coords_abs": cells,
        "coords_rel": rel,
        "coord_set_abs": set(cells),
        "patch": patch,
        "colors": colors,
        "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
    }


def _extract_frames(grid: np.ndarray) -> tuple[list[dict], list[list[tuple[int, int]]]]:
    red_parts = _components(grid == RED)
    assert 3 <= len(red_parts) <= 4
    frames = []
    for cells in red_parts:
        r0, c0, r1, c1 = _bbox(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h >= 5
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert int(border.sum()) == len(cells)
        assert np.all(patch[border] == RED)
        assert np.all(patch[~border] != RED)
        interior = patch[1:-1, 1:-1]
        guide_colors = sorted({int(v) for v in np.unique(interior)} - {BG, BLUE, RED})
        assert len(guide_colors) == 1
        color = guide_colors[0]
        coords = np.argwhere(interior == color)
        assert len(coords) >= 2
        guide_coords = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in coords]
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "color": color,
                "guide_coords": guide_coords,
            }
        )
    return frames, red_parts


def _blue_cycle_edges(grid: np.ndarray, red_parts: list[list[tuple[int, int]]]) -> set[tuple[int, int]]:
    owner = {}
    for idx, cells in enumerate(red_parts):
        for cell in cells:
            owner[cell] = idx

    edges: set[tuple[int, int]] = set()
    for cells in _components(grid == BLUE):
        touched = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                idx = owner.get((nr, nc))
                if idx is not None:
                    touched.add(idx)
        assert len(touched) == 2
        a, b = sorted(touched)
        edges.add((a, b))

    assert len(edges) == len(red_parts)
    for idx in range(len(red_parts)):
        assert sum(idx in edge for edge in edges) == 2
    return edges


def _recover_vector(
    center_coords: list[tuple[int, int]],
    guide_coords: list[tuple[int, int]],
    frame_center: tuple[float, float],
    ref_center: tuple[float, float],
) -> tuple[int, int]:
    center_set = set(center_coords)
    approx_angle = _clock_angle(frame_center, ref_center)
    candidates = set()
    for gr, gc in guide_coords:
        for cr, cc in center_coords:
            dr = gr - cr
            dc = gc - cc
            if dr == 0 and dc == 0:
                continue
            if all((rr - dr, cc2 - dc) in center_set for rr, cc2 in guide_coords):
                candidates.add((dr, dc))
    assert candidates
    ranked = sorted(
        candidates,
        key=lambda v: (
            _angle_delta(_clock_angle((v[0], v[1]), (0.0, 0.0)), approx_angle),
            -(v[0] * v[0] + v[1] * v[1]),
        ),
    )
    return ranked[0]


def _paint_rays(
    base: np.ndarray,
    center_coords: list[tuple[int, int]],
    sequence: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    out = base.copy()
    immutable = (base == RED) | (base == BLUE)
    h, w = base.shape
    for (dr, dc), color in sequence:
        k = 1
        while True:
            any_in = False
            for r, c in center_coords:
                rr = r + k * dr
                cc = c + k * dc
                if 0 <= rr < h and 0 <= cc < w:
                    any_in = True
                    if not immutable[rr, cc]:
                        out[rr, cc] = color
            if not any_in:
                break
            k += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the central sprite along guide directions while rotating ray colors by the clockwise frame ring.

    Input structure:
    - The grid has a black background, one small multicolor central sprite, and 3-4 hollow red square frames.
    - Blue segments connect the red frames into one simple cycle, and each frame interior contains one monochrome partial copy of the central sprite.
    - The partial-copy colors are distinct and the frame centers occupy distinct directions around the grid center.

    Rule:
    - Align each frame's partial copy with the central sprite to recover a translation vector.
    - Order the frames by center angle clockwise, starting from the 12 o'clock direction.
    - For frame i, draw repeated translated copies of the central sprite along its recovered vector, but recolor the whole ray with frame (i+1) mod N's guide color.
    - Keep the central sprite, red frames, blue connectors, and guide partial copies unchanged; later clockwise rays overwrite earlier rays where they overlap.

    Key insight:
    - The frame interiors still encode ray directions through subset alignment, but the ray colors are decoupled and come from the next frame in clockwise ring order instead of the guide's own color.
    """
    g = np.asarray(grid, dtype=int)
    center = _extract_center_sprite(g)
    frames, red_parts = _extract_frames(g)
    edges = _blue_cycle_edges(g, red_parts)
    ref_center = ((g.shape[0] - 1) / 2.0, (g.shape[1] - 1) / 2.0)
    order = sorted(range(len(frames)), key=lambda i: _clock_angle(frames[i]["center"], ref_center))
    for a, b in zip(order, order[1:] + order[:1]):
        assert tuple(sorted((a, b))) in edges

    vectors = {
        i: _recover_vector(center["coords_abs"], frames[i]["guide_coords"], frames[i]["center"], ref_center)
        for i in range(len(frames))
    }
    sequence = []
    for idx, frame_id in enumerate(order):
        next_id = order[(idx + 1) % len(order)]
        sequence.append((vectors[frame_id], frames[next_id]["color"]))
    return _paint_rays(g, center["coords_abs"], sequence)


def _is_connected(cells: tuple[tuple[int, int], ...]) -> bool:
    s = set(cells)
    q = deque([next(iter(s))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in s and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(s)


def _enumerate_shapes() -> list[dict]:
    shapes = []
    cells = [(r, c) for r in range(3) for c in range(3)]
    for bits in range(1, 1 << 9):
        coords = tuple(cells[i] for i in range(9) if (bits >> i) & 1)
        if not (4 <= len(coords) <= 6):
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        if min(rows) != 0 or max(rows) != 2 or min(cols) != 0 or max(cols) != 2:
            continue
        if not _is_connected(coords):
            continue

        coord_set = set(coords)
        subsets = []
        for k in range(2, len(coords)):
            for sub in itertools.combinations(coords, k):
                if not _is_connected(sub):
                    continue
                valid = set()
                for sr, sc in sub:
                    for cr, cc in coords:
                        dr = sr - cr
                        dc = sc - cc
                        if all((r - dr, c - dc) in coord_set for r, c in sub):
                            valid.add((dr, dc))
                if valid == {(0, 0)}:
                    subsets.append(tuple(sorted(sub)))
        subsets = sorted(set(subsets))
        if len(subsets) >= 5:
            shapes.append({"coords": coords, "subsets": subsets})
    return shapes


def _vector_candidates() -> list[tuple[int, int]]:
    out = []
    for dr in range(-VECTOR_BOUND, VECTOR_BOUND + 1):
        for dc in range(-VECTOR_BOUND, VECTOR_BOUND + 1):
            if dr == 0 and dc == 0:
                continue
            if max(abs(dr), abs(dc)) < 4:
                continue
            frame_box = (dr - 1, dc - 1, dr + 3, dc + 3)
            center_box = (0, 0, 2, 2)
            if not (
                frame_box[2] < center_box[0]
                or center_box[2] < frame_box[0]
                or frame_box[3] < center_box[1]
                or center_box[3] < frame_box[1]
            ):
                continue
            out.append((dr, dc))
    return out


SHAPES = _enumerate_shapes()
VECTORS = _vector_candidates()


def _frame_boxes_disjoint(vectors: tuple[tuple[int, int], ...]) -> bool:
    boxes = []
    for dr, dc in vectors:
        box = (dr - 1, dc - 1, dr + 3, dc + 3)
        for other in boxes:
            if not (
                box[2] + 1 < other[0]
                or other[2] + 1 < box[0]
                or box[3] + 1 < other[1]
                or other[3] + 1 < box[1]
            ):
                return False
        boxes.append(box)
    return True


def _ray_union(shape_coords: tuple[tuple[int, int], ...], vector: tuple[int, int], bound: int = 14) -> set[tuple[int, int]]:
    cells = {(r - 1, c - 1) for r, c in shape_coords}
    out = set()
    k = 1
    while True:
        pts = {
            (r + k * vector[0], c + k * vector[1])
            for r, c in cells
            if -bound <= r + k * vector[0] <= bound and -bound <= c + k * vector[1] <= bound
        }
        if not pts:
            break
        out |= pts
        k += 1
    return out


def _choose_vectors(shape: dict, n_frames: int) -> tuple[tuple[int, int], ...] | None:
    for _ in range(VECTOR_SAMPLE_TRIES):
        combo = tuple(random.sample(VECTORS, n_frames))
        if not _frame_boxes_disjoint(combo):
            continue
        return tuple(sorted(combo, key=lambda v: _clock_angle((v[0], v[1]), (0.0, 0.0))))
    return None


def _assign_center_colors(guide_colors: list[int], shape_coords: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], int]:
    center_palette = [c for c in CENTER_COLOR_POOL if c not in guide_colors]
    n_colors = random.choice([2, 3]) if len(shape_coords) >= 5 else 2
    picked = [5]
    for color in center_palette:
        if color not in picked:
            picked.append(color)
        if len(picked) == n_colors:
            break
    picked = picked[:n_colors]
    if len(picked) < 2:
        picked = [5, center_palette[0]]

    cells = list(shape_coords)
    random.shuffle(cells)
    colors = {}
    for idx, cell in enumerate(cells):
        colors[cell] = picked[idx % len(picked)]
    if len({colors[cell] for cell in cells}) < 2:
        colors[cells[0]] = picked[0]
        colors[cells[1]] = picked[1]
    return colors


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], occupied: set[tuple[int, int]]):
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    for c in range(c0, c1 + 1):
        occupied.add((r0, c))
        occupied.add((r1, c))
    for r in range(r0, r1 + 1):
        occupied.add((r, c0))
        occupied.add((r, c1))


def _axis_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    if sr == er:
        step = 1 if ec >= sc else -1
        return [(sr, c) for c in range(sc, ec + step, step)]
    step = 1 if er >= sr else -1
    return [(r, sc) for r in range(sr, er + step, step)]


def _one_turn_path(start: tuple[int, int], corner: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    return _axis_path(start, corner) + _axis_path(corner, end)[1:]


def _two_turn_vhv(start: tuple[int, int], mid_r: int, end: tuple[int, int]) -> list[tuple[int, int]]:
    _, sc = start
    _, ec = end
    c1 = (mid_r, sc)
    c2 = (mid_r, ec)
    return _axis_path(start, c1) + _axis_path(c1, c2)[1:] + _axis_path(c2, end)[1:]


def _two_turn_hvh(start: tuple[int, int], mid_c: int, end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, _ = start
    er, _ = end
    c1 = (sr, mid_c)
    c2 = (er, mid_c)
    return _axis_path(start, c1) + _axis_path(c1, c2)[1:] + _axis_path(c2, end)[1:]


def _find_path(
    start: tuple[int, int],
    end: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev = {start: None}
    while q:
        cell = q.popleft()
        if cell == end:
            path = []
            cur = end
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        neighbors = list(_neighbors4(*cell))
        random.shuffle(neighbors)
        for nr, nc in neighbors:
            nxt = (nr, nc)
            if not _in_bounds(shape, nr, nc):
                continue
            if nxt in occupied or nxt in prev:
                continue
            prev[nxt] = cell
            q.append(nxt)
    return None


def _best_side(frame_bbox: tuple[int, int, int, int], target_center: tuple[float, float]) -> str:
    r0, c0, r1, c1 = frame_bbox
    fr = (r0 + r1) / 2.0
    fc = (c0 + c1) / 2.0
    dr = target_center[0] - fr
    dc = target_center[1] - fc
    if abs(dr) > abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _pick_attachment_candidates(
    frame_bbox: tuple[int, int, int, int],
    side: str,
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    out: list[tuple[int, int]] = []
    if side in {"top", "bottom"}:
        cols = list(range(c0 + 1, c1))
        random.shuffle(cols)
        row = r0 if side == "top" else r1
        outside_row = row - 1 if side == "top" else row + 1
        for c in cols:
            outside = (outside_row, c)
            if _in_bounds(shape, *outside) and outside not in occupied:
                out.append(outside)
    else:
        rows = list(range(r0 + 1, r1))
        random.shuffle(rows)
        col = c0 if side == "left" else c1
        outside_col = col - 1 if side == "left" else col + 1
        for r in rows:
            outside = (r, outside_col)
            if _in_bounds(shape, *outside) and outside not in occupied:
                out.append(outside)
    return out


def _all_attachment_candidates(
    frame_bbox: tuple[int, int, int, int],
    target_center: tuple[float, float],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    preferred = _best_side(frame_bbox, target_center)
    sides = [preferred] + [s for s in ("top", "right", "bottom", "left") if s != preferred]
    out: list[tuple[int, int]] = []
    for side in sides:
        out.extend(_pick_attachment_candidates(frame_bbox, side, occupied, shape))
    return out


def _connect_cycle(grid: np.ndarray, frames: list[dict], order: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    vector_to_frame = {frame["vector"]: frame for frame in frames}
    blue_cells: set[tuple[int, int]] = set()
    border_owner = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        for c in range(c0, c1 + 1):
            border_owner[(r0, c)] = idx
            border_owner[(r1, c)] = idx
        for r in range(r0, r1 + 1):
            border_owner[(r, c0)] = idx
            border_owner[(r, c1)] = idx

    for idx, vector in enumerate(order):
        current = vector_to_frame[vector]
        nxt = vector_to_frame[order[(idx + 1) % len(order)]]
        current_id = frames.index(current)
        next_id = frames.index(nxt)
        starts = _all_attachment_candidates(current["bbox"], nxt["center"], occupied, grid.shape)
        ends = _all_attachment_candidates(nxt["bbox"], current["center"], occupied, grid.shape)
        pairs = [(s, e) for s in starts[:8] for e in ends[:8]]
        random.shuffle(pairs)
        best_path = None
        for start, end in pairs[:CONNECT_TRIES]:
            path = _find_path(start, end, occupied, grid.shape)
            if path is None:
                continue
            if any((nr, nc) in blue_cells for r, c in path for nr, nc in _neighbors4(r, c)):
                continue
            touched = set()
            for r, c in path:
                for nr, nc in _neighbors4(r, c):
                    owner = border_owner.get((nr, nc))
                    if owner is not None:
                        touched.add(owner)
            if touched != {current_id, next_id}:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
        if best_path is None:
            return False
        for r, c in best_path:
            grid[r, c] = BLUE
            occupied.add((r, c))
            blue_cells.add((r, c))
    return True


def _render_expected(
    grid: np.ndarray,
    center_coords: list[tuple[int, int]],
    order: tuple[tuple[int, int], ...],
    color_by_vector: dict[tuple[int, int], int],
) -> np.ndarray:
    sequence = [
        (vector, color_by_vector[order[(idx + 1) % len(order)]])
        for idx, vector in enumerate(order)
    ]
    return _paint_rays(grid, center_coords, sequence)


def _render_self_color(
    grid: np.ndarray,
    center_coords: list[tuple[int, int]],
    order: tuple[tuple[int, int], ...],
    color_by_vector: dict[tuple[int, int], int],
) -> np.ndarray:
    return _paint_rays(grid, center_coords, [(v, color_by_vector[v]) for v in order])


def _render_prev_color(
    grid: np.ndarray,
    center_coords: list[tuple[int, int]],
    order: tuple[tuple[int, int], ...],
    color_by_vector: dict[tuple[int, int], int],
) -> np.ndarray:
    sequence = [
        (vector, color_by_vector[order[(idx - 1) % len(order)]])
        for idx, vector in enumerate(order)
    ]
    return _paint_rays(grid, center_coords, sequence)


def _render_rotated_draw_order(
    grid: np.ndarray,
    center_coords: list[tuple[int, int]],
    order: tuple[tuple[int, int], ...],
    color_by_vector: dict[tuple[int, int], int],
) -> np.ndarray:
    sequence = [
        (vector, color_by_vector[order[(idx + 1) % len(order)]])
        for idx, vector in enumerate(order)
    ]
    sequence = sequence[1:] + sequence[:1]
    return _paint_rays(grid, center_coords, sequence)


def _gen_attempt() -> np.ndarray | None:
    size = random.choice([21, 23, 25, 27])
    grid = np.zeros((size, size), dtype=int)
    center_top = size // 2 - 1
    center_left = size // 2 - 1
    shape = random.choice(SHAPES)
    n_frames = random.choice([3, 4])
    vectors = _choose_vectors(shape, n_frames)
    if vectors is None:
        return None

    guide_subsets = random.sample(shape["subsets"], n_frames)
    guide_colors = random.sample(GUIDE_COLOR_POOL, n_frames)
    center_colors = _assign_center_colors(guide_colors, shape["coords"])

    occupied: set[tuple[int, int]] = set()
    center_coords_abs: list[tuple[int, int]] = []
    for r, c in shape["coords"]:
        rr = center_top + r
        cc = center_left + c
        grid[rr, cc] = center_colors[(r, c)]
        occupied.add((rr, cc))
        center_coords_abs.append((rr, cc))

    frames = []
    for vector, subset, color in zip(vectors, guide_subsets, guide_colors):
        dr, dc = vector
        r0 = center_top + dr - 1
        c0 = center_left + dc - 1
        r1 = r0 + FRAME_SIZE - 1
        c1 = c0 + FRAME_SIZE - 1
        if not (0 <= r0 and 0 <= c0 and r1 < size and c1 < size):
            return None
        bbox = (r0, c0, r1, c1)
        _draw_frame(grid, bbox, occupied)
        for sr, sc in subset:
            rr = center_top + dr + sr
            cc = center_left + dc + sc
            grid[rr, cc] = color
            occupied.add((rr, cc))
        frames.append(
            {
                "bbox": bbox,
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "vector": vector,
                "color": color,
            }
        )

    if not _connect_cycle(grid, frames, vectors, occupied):
        return None

    color_by_vector = {frame["vector"]: frame["color"] for frame in frames}
    expected = _render_expected(grid, center_coords_abs, vectors, color_by_vector)
    if np.array_equal(expected, _render_self_color(grid, center_coords_abs, vectors, color_by_vector)):
        return None
    if np.array_equal(expected, _render_prev_color(grid, center_coords_abs, vectors, color_by_vector)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
