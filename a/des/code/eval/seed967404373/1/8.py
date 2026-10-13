import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
FRAME_COUNT_CHOICES = [3, 4, 5]
FRAME_HEIGHT_OPTIONS = {
    3: [2, 4],
    4: [5],
    5: [2, 3],
}
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 200
SHAPE_ATTEMPTS = 200


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _component_count(mask: np.ndarray, neighbors: list[tuple[int, int]]) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _transform_cells(cells: list[tuple[int, int]], mode: int) -> list[tuple[int, int]]:
    pts = []
    for r, c in cells:
        if mode == 0:
            rr, cc = r, c
        elif mode == 1:
            rr, cc = c, -r
        elif mode == 2:
            rr, cc = -r, -c
        elif mode == 3:
            rr, cc = -c, r
        elif mode == 4:
            rr, cc = r, -c
        elif mode == 5:
            rr, cc = -r, c
        elif mode == 6:
            rr, cc = c, r
        else:
            rr, cc = -c, -r
        pts.append((rr, cc))
    return _normalize_cells(set(pts))


def _shape_bbox(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)


def _choose_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color == bg:
            continue
        if not np.all(grid[:, c] == color):
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if left.size == 0 or right.size == 0:
            continue
        if not np.any((left != bg) & (left != color)):
            continue
        row_count = int(np.count_nonzero(np.all(right == color, axis=1)))
        if row_count < 2:
            continue
        if not np.any((right != bg) & (right != color)):
            continue
        candidates.append((row_count, right.shape[1], -c, c, color))
    assert candidates, "failed to find divider column"
    candidates.sort(reverse=True)
    _, _, _, sep_c, grey = candidates[0]
    return int(sep_c), int(grey)


def _extract_frames(timeline: np.ndarray, grey: int) -> list[np.ndarray]:
    sep_rows = np.all(timeline == grey, axis=1)
    frames = []
    r = 0
    while r < timeline.shape[0]:
        while r < timeline.shape[0] and sep_rows[r]:
            r += 1
        start = r
        while r < timeline.shape[0] and not sep_rows[r]:
            r += 1
        if start < r:
            frames.append(timeline[start:r].copy())
    assert 3 <= len(frames) <= 5, "expected 3-5 timeline frames"
    heights = {frame.shape[0] for frame in frames}
    widths = {frame.shape[1] for frame in frames}
    assert len(heights) == 1 and len(widths) == 1, "timeline frames must share a size"
    return frames


def _frame_points(frame: np.ndarray, bg: int, grey: int) -> dict[int, tuple[int, int]]:
    pts = {}
    for color in np.unique(frame):
        color = int(color)
        if color in (bg, grey):
            continue
        coords = np.argwhere(frame == color)
        assert coords.shape[0] == 1, "each frame color should appear exactly once"
        pts[color] = tuple(map(int, coords[0]))
    assert pts, "frame must contain colored points"
    return pts


def _collect_objects(main_area: np.ndarray, colors: set[int]) -> dict[int, dict[str, object]]:
    objects = {}
    for color in colors:
        mask = main_area == color
        coords = np.argwhere(mask)
        assert coords.size > 0, "missing main-area object"
        assert _component_count(mask, NEIGHBORS8) == 1, "each color should form one object"
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        objects[int(color)] = {"top": (r0, c0), "rel": rel}
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the timeline snapshots, move each colored object by the per-frame scaled
    displacement, and accumulate every visited position on the main canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(_most_common_color(g))
    sep_c, grey = _choose_separator(g, bg)
    main_area = g[:, :sep_c]
    timeline = g[:, sep_c + 1 :]
    assert main_area.size > 0 and timeline.size > 0

    frames = _extract_frames(timeline, grey)
    frame_h, frame_w = frames[0].shape
    out_h, out_w = main_area.shape
    assert out_h % frame_h == 0 and out_w % frame_w == 0
    scale_h = out_h // frame_h
    scale_w = out_w // frame_w
    assert scale_h == scale_w, "expected square scaling blocks"
    scale = int(scale_h)

    frame_maps = [_frame_points(frame, bg, grey) for frame in frames]
    colors = set(frame_maps[0])
    for pts in frame_maps[1:]:
        assert set(pts) == colors, "every frame should describe the same object colors"

    objects = _collect_objects(main_area, colors)
    assert set(objects) == colors, "timeline colors and object colors must match"

    draw_order = [
        color
        for color, _ in sorted(frame_maps[0].items(), key=lambda item: (item[1][0], item[1][1]))
    ]
    first_frame = frame_maps[0]
    out = np.full(main_area.shape, bg, dtype=int)
    for pts in frame_maps:
        for color in draw_order:
            start_r, start_c = first_frame[color]
            now_r, now_c = pts[color]
            dr = int(now_r - start_r) * scale
            dc = int(now_c - start_c) * scale
            base_r = int(objects[color]["top"][0]) + dr
            base_c = int(objects[color]["top"][1]) + dc
            for rr, cc in objects[color]["rel"]:
                tr = base_r + rr
                tc = base_c + cc
                assert 0 <= tr < out_h and 0 <= tc < out_w, "object trail left the main area"
                out[tr, tc] = int(color)
    return out


def _random_shape(rng: np.random.Generator, scale: int) -> list[tuple[int, int]]:
    max_h = min(4, max(2, scale - 1))
    max_w = min(4, max(2, scale - 1))
    max_area = min(6, max_h * max_w)
    for _ in range(SHAPE_ATTEMPTS):
        area = int(rng.integers(2, max_area + 1))
        start = (int(rng.integers(max_h)), int(rng.integers(max_w)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != area:
            continue
        shape = _normalize_cells(cells)
        shape = _transform_cells(shape, int(rng.integers(8)))
        bbox_h, bbox_w = _shape_bbox(shape)
        if bbox_h < scale and bbox_w < scale:
            return shape
    raise RuntimeError("failed to sample a shape")


def _sample_paths(
    rng: np.random.Generator,
    frame_count: int,
    frame_h: int,
    frame_w: int,
    n_obj: int,
) -> list[list[tuple[int, int]]]:
    all_cells = [(r, c) for r in range(frame_h) for c in range(frame_w)]
    step_pool = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (2, 0),
        (-2, 0),
        (0, 2),
        (0, -2),
    ]
    for _ in range(PATH_ATTEMPTS):
        starts_idx = rng.choice(len(all_cells), size=n_obj, replace=False)
        paths = [[all_cells[int(idx)]] for idx in starts_idx]
        moved = False
        stationary = False
        valid = True
        for _frame in range(1, frame_count):
            curr = [None] * n_obj
            occupied = set()
            order = list(rng.permutation(n_obj))
            for idx in order:
                prev_r, prev_c = paths[idx][-1]
                candidates = []
                for step_idx in rng.permutation(len(step_pool)):
                    dr, dc = step_pool[int(step_idx)]
                    nr = prev_r + dr
                    nc = prev_c + dc
                    if not (0 <= nr < frame_h and 0 <= nc < frame_w):
                        continue
                    if (nr, nc) in occupied:
                        continue
                    candidates.append((nr, nc))
                if not candidates:
                    valid = False
                    break
                prefer_move = [pos for pos in candidates if pos != (prev_r, prev_c)]
                if prefer_move and rng.random() < 0.75:
                    pick = prefer_move[int(rng.integers(len(prefer_move)))]
                else:
                    pick = candidates[int(rng.integers(len(candidates)))]
                occupied.add(pick)
                curr[idx] = pick
                moved = moved or pick != (prev_r, prev_c)
                stationary = stationary or pick == (prev_r, prev_c)
            if not valid:
                break
            for idx in range(n_obj):
                paths[idx].append(curr[idx])
        if valid and moved and stationary:
            return paths
    raise RuntimeError("failed to sample object paths")


def _render_spec(
    h: int,
    main_w: int,
    frame_h: int,
    frame_w: int,
    grey: int,
    colors: list[int],
    objects: list[dict[str, object]],
    paths: list[list[tuple[int, int]]],
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((h, main_w + 1 + frame_w), BG, dtype=int)
    grid[:, main_w] = grey

    timeline_left = main_w + 1
    for row in range(0, h, frame_h + 1):
        grid[row, timeline_left:] = grey
    grid[h - 1, timeline_left:] = grey

    for t in range(len(paths[0])):
        start_r = 1 + t * (frame_h + 1)
        for color, path in zip(colors, paths):
            pr, pc = path[t]
            grid[start_r + pr, timeline_left + pc] = int(color)

    main_area = np.full((h, main_w), BG, dtype=int)
    for color, obj, path in zip(colors, objects, paths):
        start_block_r, start_block_c = path[0]
        off_r, off_c = obj["offset"]
        top_r = start_block_r * scale + off_r
        top_c = start_block_c * scale + off_c
        for rr, cc in obj["shape"]:
            main_area[top_r + rr, top_c + cc] = int(color)
    grid[:, :main_w] = main_area

    out = np.full_like(main_area, BG)
    for t in range(len(paths[0])):
        for color, obj, path in zip(colors, objects, paths):
            pr0, pc0 = path[0]
            pr, pc = path[t]
            top_r = obj["start_top"][0] + (pr - pr0) * scale
            top_c = obj["start_top"][1] + (pc - pc0) * scale
            for rr, cc in obj["shape"]:
                out[top_r + rr, top_c + cc] = int(color)
    return grid, out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.choice(FRAME_COUNT_CHOICES))
    frame_h = int(rng.choice(FRAME_HEIGHT_OPTIONS[frame_count]))
    scale = frame_count + (frame_count + 1) // frame_h
    h = frame_count * frame_h + (frame_count + 1)
    assert h == frame_h * scale

    max_frame_w = min(6, max(3, 30 // scale))
    if max_frame_w < 3:
        return None
    frame_w = int(rng.integers(3, max_frame_w + 1))
    main_w = frame_w * scale

    grey = int(rng.choice(COLOR_POOL))
    palette = [color for color in COLOR_POOL if color != grey]
    max_objects = min(5, frame_h * frame_w - 1, len(palette))
    if max_objects < 2:
        return None
    n_obj = int(rng.integers(2, max_objects + 1))
    colors = [int(color) for color in rng.choice(palette, size=n_obj, replace=False)]

    try:
        paths = _sample_paths(rng, frame_count, frame_h, frame_w, n_obj)
    except RuntimeError:
        return None

    objects = []
    occupied_start = np.zeros((h, main_w), dtype=bool)
    for path in paths:
        shape = _random_shape(rng, scale)
        bbox_h, bbox_w = _shape_bbox(shape)
        slack_r = scale - bbox_h
        slack_c = scale - bbox_w
        if slack_r < 0 or slack_c < 0:
            return None
        offset_r = int(rng.integers(slack_r + 1))
        offset_c = int(rng.integers(slack_c + 1))
        top_r = path[0][0] * scale + offset_r
        top_c = path[0][1] * scale + offset_c
        touches = False
        for rr, cc in shape:
            tr = top_r + rr
            tc = top_c + cc
            if occupied_start[tr, tc]:
                touches = True
                break
        if touches:
            return None
        for rr, cc in shape:
            occupied_start[top_r + rr, top_c + cc] = True
        objects.append(
            {
                "shape": shape,
                "offset": (offset_r, offset_c),
                "start_top": (top_r, top_c),
            }
        )

    grid, expected = _render_spec(
        h=h,
        main_w=main_w,
        frame_h=frame_h,
        frame_w=frame_w,
        grey=grey,
        colors=colors,
        objects=objects,
        paths=paths,
        scale=scale,
    )

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid timeline-trail ARC input")
