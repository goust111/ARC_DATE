import math
import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
PALETTE = [3, 4, 6, 7, 8, 9]

FRAME_COUNT_CHOICES = [3, 4]
GRID_RANGE_BY_COUNT = {
    3: ((22, 28), (22, 28)),
    4: ((25, 29), (25, 29)),
}
FRAME_SIZE_CHOICES = [5, 6, 7]
GEN_ATTEMPTS = 400

BASE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_shape(cells) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _transform_shape(cells, rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = [(r, c) for r, c in cells]
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize_shape(pts)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _extract_frames(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == RED)
    frames = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and 5 <= h <= 7
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(crop, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(crop[border] == RED)
        assert not np.any(crop[~border] == RED)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "interior_h": h - 2,
                "interior_w": w - 2,
            }
        )
    assert len(frames) in (3, 4)
    return frames


def _frame_owner_map(frames: list[dict]) -> dict[tuple[int, int], int]:
    owner = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        for c in range(c0, c1 + 1):
            owner[(r0, c)] = idx
            owner[(r1, c)] = idx
        for r in range(r0, r1 + 1):
            owner[(r, c0)] = idx
            owner[(r, c1)] = idx
    return owner


def _blue_cycle_edges(grid: np.ndarray, frames: list[dict]) -> set[tuple[int, int]]:
    owner = _frame_owner_map(frames)
    edges: set[tuple[int, int]] = set()
    for coords in _components(grid == BLUE):
        comp_set = {tuple(map(int, cell)) for cell in coords}
        touched = set()
        endpoints = 0
        for r, c in comp_set:
            deg = sum((nr, nc) in comp_set for nr, nc in _neighbors4(r, c))
            assert deg <= 2
            if deg == 1:
                endpoints += 1
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in owner:
                    touched.add(owner[(nr, nc)])
        assert len(touched) == 2
        assert endpoints == 2
        a, b = sorted(touched)
        edges.add((a, b))

    n = len(frames)
    assert len(edges) == n
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    for i in range(n):
        assert len(adj[i]) == 2
    seen = set()
    stack = [0]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj[cur] - seen)
    assert len(seen) == n
    return edges


def _clockwise_order(frames: list[dict], edges: set[tuple[int, int]]) -> list[int]:
    mean_r = sum(frame["center"][0] for frame in frames) / len(frames)
    mean_c = sum(frame["center"][1] for frame in frames) / len(frames)
    angles = [
        math.atan2(frame["center"][0] - mean_r, frame["center"][1] - mean_c)
        for frame in frames
    ]
    assert len({round(angle, 12) for angle in angles}) == len(angles)
    order = sorted(range(len(frames)), key=lambda i: angles[i])
    for a, b in zip(order, order[1:] + order[:1]):
        assert tuple(sorted((a, b))) in edges
    return order


def _extract_frame_content(grid: np.ndarray, frame: dict) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    ih, iw = interior.shape

    markers = []
    for rr in range(ih):
        left = int(interior[rr, 0])
        right = int(interior[rr, -1])
        if left != BG and left == right:
            markers.append((rr, left))
    assert len(markers) == 1
    marker_row_rel, marker_color = markers[0]
    assert np.count_nonzero(interior == marker_color) == 2

    glyph_mask = (interior != BG) & (interior != marker_color)
    comps = _components(glyph_mask)
    assert len(comps) == 1
    coords = comps[0]
    assert 3 <= len(coords) <= 5
    gr0, gc0, gr1, gc1 = _bbox(coords)
    shape = tuple(sorted((int(r - gr0), int(c - gc0)) for r, c in coords))
    return {
        **frame,
        "marker_row_rel": int(marker_row_rel),
        "marker_color": int(marker_color),
        "shape": shape,
        "shape_h": int(gr1 - gr0 + 1),
        "shape_w": int(gc1 - gc0 + 1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each frame's glyph source clockwise around the blue cycle, but keep the destination frame's own anchor row and marker color.

    Input structure
    - The grid has a black background with 3 or 4 hollow red square frames connected into one blue cycle.
    - Inside each frame there is one small 4-connected glyph in a non-black/non-red/non-blue color.
    - Each frame also contains exactly one same-row endpoint pair: two same-colored pixels in the leftmost and rightmost interior columns.

    Rule
    - Detect the red frames and the blue connector cycle, then sort the frame centers in clockwise order.
    - For each frame X, erase its interior content, take the glyph shape from the next clockwise frame Y, center that shape horizontally inside X, and align the first glyph row to X's own endpoint row.
    - Draw the transferred shape using X's own endpoint color, then redraw X's endpoint pair; keep the red borders, blue connectors, and background unchanged.

    Key insight
    - The final content of a frame is split across three sources: the next frame supplies the shape, while the current frame still supplies both the color and the anchor row.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    edges = _blue_cycle_edges(g, frames)
    order = _clockwise_order(frames, edges)
    info = [_extract_frame_content(g, frame) for frame in frames]

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG

    for pos, idx in enumerate(order):
        donor_idx = order[(pos + 1) % len(order)]
        dst = info[idx]
        src = info[donor_idx]
        r0, c0, r1, c1 = dst["bbox"]
        ih = dst["interior_h"]
        iw = dst["interior_w"]
        top = dst["marker_row_rel"]
        left = (iw - src["shape_w"]) // 2
        assert 0 <= left <= iw - src["shape_w"]
        assert 0 <= top <= ih - src["shape_h"]
        for dr, dc in src["shape"]:
            out[r0 + 1 + top + dr, c0 + 1 + left + dc] = dst["marker_color"]
        out[r0 + 1 + dst["marker_row_rel"], c0 + 1] = dst["marker_color"]
        out[r0 + 1 + dst["marker_row_rel"], c1 - 1] = dst["marker_color"]

    return out


def _axis_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    sr, sc = start
    er, ec = end
    if sr != er and sc != ec:
        return None
    if sr == er:
        step = 1 if ec >= sc else -1
        return [(sr, c) for c in range(sc, ec + step, step)]
    step = 1 if er >= sr else -1
    return [(r, sc) for r in range(sr, er + step, step)]


def _concat_paths(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for part in parts:
        if not part:
            continue
        if out and out[-1] == part[0]:
            out.extend(part[1:])
        else:
            out.extend(part)
    return out


def _one_turn_path(start: tuple[int, int], corner: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    first = _axis_path(start, corner)
    second = _axis_path(corner, end)
    if first is None or second is None:
        return None
    return _concat_paths(first, second)


def _two_turn_vhv(start: tuple[int, int], mid_r: int, end: tuple[int, int]) -> list[tuple[int, int]] | None:
    sr, sc = start
    er, ec = end
    p1 = _axis_path(start, (mid_r, sc))
    p2 = _axis_path((mid_r, sc), (mid_r, ec))
    p3 = _axis_path((mid_r, ec), end)
    if p1 is None or p2 is None or p3 is None:
        return None
    return _concat_paths(p1, p2, p3)


def _two_turn_hvh(start: tuple[int, int], mid_c: int, end: tuple[int, int]) -> list[tuple[int, int]] | None:
    sr, sc = start
    er, ec = end
    p1 = _axis_path(start, (sr, mid_c))
    p2 = _axis_path((sr, mid_c), (er, mid_c))
    p3 = _axis_path((er, mid_c), end)
    if p1 is None or p2 is None or p3 is None:
        return None
    return _concat_paths(p1, p2, p3)


def _frame_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2) -> bool:
    r0a, c0a, r1a, c1a = a
    r0b, c0b, r1b, c1b = b
    return (
        r1a + pad < r0b
        or r1b + pad < r0a
        or c1a + pad < c0b
        or c1b + pad < c0a
    )


def _sample_frame_bboxes(n_frames: int, sizes: list[int], grid_h: int, grid_w: int):
    center_r = (grid_h - 1) / 2.0
    center_c = (grid_w - 1) / 2.0
    base = [2 * math.pi * k / n_frames for k in range(n_frames)]
    angles = sorted(angle + random.uniform(-0.28, 0.28) for angle in base)
    rad_r = random.uniform(grid_h * 0.27, grid_h * 0.34)
    rad_c = random.uniform(grid_w * 0.27, grid_w * 0.34)

    boxes = []
    for angle, size in zip(angles, sizes):
        rr = center_r + rad_r * math.sin(angle)
        cc = center_c + rad_c * math.cos(angle)
        r0 = int(round(rr - size / 2))
        c0 = int(round(cc - size / 2))
        r0 = max(1, min(grid_h - size - 2, r0))
        c0 = max(1, min(grid_w - size - 2, c0))
        boxes.append((r0, c0, r0 + size - 1, c0 + size - 1))

    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            if not _frame_gap_ok(boxes[i], boxes[j]):
                return None
    return boxes


def _best_side(frame_bbox: tuple[int, int, int, int], target_r: float, target_c: float) -> str:
    r0, c0, r1, c1 = frame_bbox
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    dr = target_r - cr
    dc = target_c - cc
    if abs(dr) > abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _pick_attachment(
    bbox: tuple[int, int, int, int],
    side: str,
    blocked: set[tuple[int, int]],
    grid_h: int,
    grid_w: int,
):
    r0, c0, r1, c1 = bbox
    if side == "top":
        cells = [(r0, c) for c in range(c0 + 1, c1)]
        delta = (-1, 0)
    elif side == "bottom":
        cells = [(r1, c) for c in range(c0 + 1, c1)]
        delta = (1, 0)
    elif side == "left":
        cells = [(r, c0) for r in range(r0 + 1, r1)]
        delta = (0, -1)
    else:
        cells = [(r, c1) for r in range(r0 + 1, r1)]
        delta = (0, 1)
    random.shuffle(cells)
    for br, bc in cells:
        sr = br + delta[0]
        sc = bc + delta[1]
        if 0 <= sr < grid_h and 0 <= sc < grid_w and (sr, sc) not in blocked:
            return (br, bc), (sr, sc)
    return None, None


def _path_valid(
    path: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
    blocked: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
    frame_owner: dict[tuple[int, int], int],
    src_idx: int,
    dst_idx: int,
) -> bool:
    if path is None or len(path) < 2:
        return False
    if len(path) != len(set(path)):
        return False
    path_set = set(path)
    for pos, (r, c) in enumerate(path):
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            return False
        if (r, c) in blocked:
            return False
        if any((nr, nc) in blue_cells for nr, nc in _neighbors4(r, c)):
            return False
        touched = {frame_owner[(nr, nc)] for nr, nc in _neighbors4(r, c) if (nr, nc) in frame_owner}
        if pos == 0:
            if touched != {src_idx}:
                return False
        elif pos == len(path) - 1:
            if touched != {dst_idx}:
                return False
        elif touched:
            return False
        deg = sum((nr, nc) in path_set for nr, nc in _neighbors4(r, c))
        if pos in (0, len(path) - 1):
            if deg != 1:
                return False
        elif deg != 2:
            return False
    return True


def _find_connector_path(
    start: tuple[int, int],
    end: tuple[int, int],
    grid_h: int,
    grid_w: int,
    blocked: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
    frame_owner: dict[tuple[int, int], int],
    src_idx: int,
    dst_idx: int,
):
    sr, sc = start
    er, ec = end
    candidates = []
    straight = _axis_path(start, end)
    if straight is not None:
        candidates.append(straight)
    for corner in [(sr, ec), (er, sc)]:
        cand = _one_turn_path(start, corner, end)
        if cand is not None:
            candidates.append(cand)
    for _ in range(18):
        if sr != er:
            mid_r = random.randint(min(sr, er), max(sr, er))
            cand = _two_turn_vhv(start, mid_r, end)
            if cand is not None:
                candidates.append(cand)
        if sc != ec:
            mid_c = random.randint(min(sc, ec), max(sc, ec))
            cand = _two_turn_hvh(start, mid_c, end)
            if cand is not None:
                candidates.append(cand)
    random.shuffle(candidates)
    for path in candidates:
        if 2 <= len(path) <= 24 and _path_valid(
            path, grid_h, grid_w, blocked, blue_cells, frame_owner, src_idx, dst_idx
        ):
            return path
    return None


def _sample_shapes(n_frames: int) -> list[tuple[tuple[int, int], ...]]:
    shapes = []
    seen = set()
    for _ in range(200):
        base = random.choice(BASE_SHAPES)
        shape = _transform_shape(base, random.randrange(4), bool(random.randrange(2)))
        if shape in seen:
            continue
        seen.add(shape)
        shapes.append(shape)
        if len(shapes) == n_frames:
            return shapes
    raise RuntimeError("shape sampling failed")


def _top_row_avoids_markers(shape, interior_w: int) -> bool:
    _, shape_w = _shape_dims(shape)
    left = (interior_w - shape_w) // 2
    return all(left + dc not in (0, interior_w - 1) for dr, dc in shape if dr == 0)


def _sample_marker_and_input_rows(
    frames: list[dict], order: list[int], shapes: list[tuple[tuple[int, int], ...]]
):
    marker_rows = [None] * len(frames)
    input_rows = [None] * len(frames)
    for pos, idx in enumerate(order):
        donor = order[(pos + 1) % len(order)]
        ih = frames[idx]["interior_h"]
        iw = frames[idx]["interior_w"]
        donor_h, _ = _shape_dims(shapes[donor])
        if not _top_row_avoids_markers(shapes[donor], iw):
            return None
        options = list(range(0, ih - donor_h + 1))
        if not options:
            return None
        marker_rows[idx] = random.choice(options)

    for idx, frame in enumerate(frames):
        ih = frame["interior_h"]
        iw = frame["interior_w"]
        own_h, own_w = _shape_dims(shapes[idx])
        left = (iw - own_w) // 2
        choices = []
        for top in range(0, ih - own_h + 1):
            occupied = {(top + dr, left + dc) for dr, dc in shapes[idx]}
            if (marker_rows[idx], 0) in occupied or (marker_rows[idx], iw - 1) in occupied:
                continue
            choices.append(top)
        if not choices:
            return None
        input_rows[idx] = random.choice(choices)
    return marker_rows, input_rows


def _render_input(
    grid_h: int,
    grid_w: int,
    boxes: list[tuple[int, int, int, int]],
    order: list[int],
    shapes: list[tuple[tuple[int, int], ...]],
    marker_rows: list[int],
    input_rows: list[int],
):
    grid = np.zeros((grid_h, grid_w), dtype=int)
    frames = []
    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        grid[r0, c0 : c1 + 1] = RED
        grid[r1, c0 : c1 + 1] = RED
        grid[r0 : r1 + 1, c0] = RED
        grid[r0 : r1 + 1, c1] = RED
        frames.append(
            {
                "bbox": bbox,
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "interior_h": r1 - r0 - 1,
                "interior_w": c1 - c0 - 1,
            }
        )

    frame_owner = _frame_owner_map(frames)
    blocked = set(frame_owner)
    blue_cells: set[tuple[int, int]] = set()

    for pos, idx in enumerate(order):
        jdx = order[(pos + 1) % len(order)]
        dst_center = frames[jdx]["center"]
        src_center = frames[idx]["center"]
        side_i = _best_side(frames[idx]["bbox"], *dst_center)
        side_j = _best_side(frames[jdx]["bbox"], *src_center)
        _, start_i = _pick_attachment(frames[idx]["bbox"], side_i, blocked, grid_h, grid_w)
        _, start_j = _pick_attachment(frames[jdx]["bbox"], side_j, blocked, grid_h, grid_w)
        if start_i is None or start_j is None:
            return None
        path = _find_connector_path(
            start_i, start_j, grid_h, grid_w, blocked, blue_cells, frame_owner, idx, jdx
        )
        if path is None:
            return None
        for r, c in path:
            grid[r, c] = BLUE
            blocked.add((r, c))
            blue_cells.add((r, c))

    marker_colors = random.sample(PALETTE, len(frames))
    glyph_colors = []
    for idx in range(len(frames)):
        choices = [color for color in PALETTE if color != marker_colors[idx]]
        glyph_colors.append(random.choice(choices))

    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        ih = frame["interior_h"]
        iw = frame["interior_w"]
        shape = shapes[idx]
        top = input_rows[idx]
        shape_h, shape_w = _shape_dims(shape)
        left = (iw - shape_w) // 2
        for dr, dc in shape:
            grid[r0 + 1 + top + dr, c0 + 1 + left + dc] = glyph_colors[idx]
        rr = r0 + 1 + marker_rows[idx]
        grid[rr, c0 + 1] = marker_colors[idx]
        grid[rr, c1 - 1] = marker_colors[idx]

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        frames = _extract_frames(grid)
        edges = _blue_cycle_edges(grid, frames)
        order = _clockwise_order(frames, edges)
        info = [_extract_frame_content(grid, frame) for frame in frames]
        if len({item["shape"] for item in info}) != len(info):
            return False
        if len({item["marker_color"] for item in info}) != len(info):
            return False

        out = main(grid)
        changed = 0
        for pos, idx in enumerate(order):
            donor = order[(pos + 1) % len(order)]
            dst = info[idx]
            src = info[donor]
            r0, c0, r1, c1 = dst["bbox"]
            interior_out = out[r0 + 1 : r1, c0 + 1 : c1]
            expected = np.zeros_like(interior_out)
            left = (dst["interior_w"] - src["shape_w"]) // 2
            top = dst["marker_row_rel"]
            for dr, dc in src["shape"]:
                expected[top + dr, left + dc] = dst["marker_color"]
            expected[dst["marker_row_rel"], 0] = dst["marker_color"]
            expected[dst["marker_row_rel"], -1] = dst["marker_color"]
            if not np.array_equal(interior_out, expected):
                return False
            interior_in = grid[r0 + 1 : r1, c0 + 1 : c1]
            if not np.array_equal(interior_in, interior_out):
                changed += 1
        return changed == len(frames)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    (h0, h1), (w0, w1) = GRID_RANGE_BY_COUNT[n_frames]
    grid_h = random.randint(h0, h1)
    grid_w = random.randint(w0, w1)
    sizes = [random.choice(FRAME_SIZE_CHOICES) for _ in range(n_frames)]

    boxes = None
    for _ in range(80):
        boxes = _sample_frame_bboxes(n_frames, sizes, grid_h, grid_w)
        if boxes is not None:
            break
    if boxes is None:
        return None

    frames = [
        {
            "bbox": bbox,
            "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            "interior_h": bbox[2] - bbox[0] - 1,
            "interior_w": bbox[3] - bbox[1] - 1,
        }
        for bbox in boxes
    ]
    mean_r = sum(frame["center"][0] for frame in frames) / n_frames
    mean_c = sum(frame["center"][1] for frame in frames) / n_frames
    angles = [math.atan2(frame["center"][0] - mean_r, frame["center"][1] - mean_c) for frame in frames]
    if len({round(angle, 12) for angle in angles}) != n_frames:
        return None
    order = sorted(range(n_frames), key=lambda i: angles[i])

    shapes = _sample_shapes(n_frames)
    rows = _sample_marker_and_input_rows(frames, order, shapes)
    if rows is None:
        return None
    marker_rows, input_rows = rows
    grid = _render_input(grid_h, grid_w, boxes, order, shapes, marker_rows, input_rows)
    if grid is None or not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid

    raise RuntimeError("failed to generate valid input")
