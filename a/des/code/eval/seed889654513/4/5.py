from __future__ import annotations

import random

import numpy as np


BG = 0
PAYLOAD_COLOR = 4
FRAME_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)

MIN_GROUPS = 3
MAX_GROUPS = 4
MAX_CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 300


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _bbox_from_coords(coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.any()) == 0 or len(_find_components(mask)) == 1


def _rect_frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(local, expected):
        return None
    return r0, c0, r1, c1


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _transform_mask(mask: np.ndarray, depth: int) -> np.ndarray:
    mode = (depth - 1) % 3
    if mode == 0:
        return np.flipud(mask)
    if mode == 1:
        return np.fliplr(mask)
    return np.flipud(np.fliplr(mask))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _detect_frames(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    frame_mask = np.zeros_like(g, dtype=bool)
    frames: list[dict] = []

    for color in sorted(set(map(int, g.ravel())) - {BG}):
        for coords in _find_components(g == color):
            bbox = _rect_frame_bbox(coords)
            if bbox is None:
                continue
            frame_mask[coords[:, 0], coords[:, 1]] = True
            frames.append({"bbox": bbox, "color": int(color), "coords": coords})

    assert frames, "expected at least one rectangular frame"
    return frames, frame_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested rectangular containers by spatial containment and transform each
    innermost payload according to the nesting depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames, frame_mask = _detect_frames(g)
    bboxes = [frame["bbox"] for frame in frames]
    depths = [1 + sum(_contains(other, bbox) for other in bboxes) for bbox in bboxes]

    out = g.copy()
    for frame, depth in zip(frames, depths):
        bbox = frame["bbox"]
        if any(_contains(bbox, other) for other in bboxes if other != bbox):
            continue

        r0, c0, r1, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_frame_mask = frame_mask[r0 + 1 : r1, c0 + 1 : c1]
        payload_mask = (inner != BG) & ~inner_frame_mask
        assert payload_mask.any(), "each innermost frame must contain a payload"

        payload_colors = np.unique(inner[payload_mask])
        assert payload_colors.size == 1, "payload should use one visible color"
        payload_color = int(payload_colors[0])

        pr0, pc0, pr1, pc1 = _bbox_from_mask(payload_mask)
        sprite = payload_mask[pr0 : pr1 + 1, pc0 : pc1 + 1]
        transformed = _transform_mask(sprite, depth)

        target = out[r0 + 1 + pr0 : r0 + 2 + pr1, c0 + 1 + pc0 : c0 + 2 + pc1]
        target[sprite] = BG
        target[transformed] = payload_color

    return out


def _sample_payload() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 5)
        target_cells = random.randint(3, h * w - 1)

        mask = np.zeros((h, w), dtype=bool)
        cells = {(random.randrange(h), random.randrange(w))}
        mask[next(iter(cells))] = True

        while len(cells) < target_cells:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            cells.add(cell)
            mask[cell] = True

        if not _is_connected(mask):
            continue
        mask = _crop_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(np.fliplr(mask))):
            continue
        return mask
    raise RuntimeError("failed to sample a non-symmetric payload")


def _make_group(depth: int) -> tuple[np.ndarray, np.ndarray]:
    payload = _sample_payload()
    ph, pw = payload.shape

    inner_h = ph + random.randint(2, 4 if depth <= 2 else 3)
    inner_w = pw + random.randint(2, 4 if depth <= 2 else 3)
    spare_h = inner_h - ph
    spare_w = inner_w - pw
    assert spare_h >= 2 and spare_w >= 2

    top = random.randint(1, spare_h - 1)
    left = random.randint(1, spare_w - 1)

    between_layers = [random.randint(0, 1) + 1 for _ in range(depth - 1)]
    base_h = inner_h + 2
    base_w = inner_w + 2
    total_pad = sum(between_layers)
    total_h = base_h + 2 * total_pad
    total_w = base_w + 2 * total_pad

    group_inp = np.full((total_h, total_w), BG, dtype=int)
    frame_bboxes_inner_to_outer: list[tuple[int, int, int, int]] = []
    for i in range(depth):
        offset = sum(between_layers[i:]) if i < depth - 1 else 0
        expand = sum(between_layers[:i])
        h = base_h + 2 * expand
        w = base_w + 2 * expand
        frame_bboxes_inner_to_outer.append((offset, offset, offset + h - 1, offset + w - 1))

    frame_bboxes_outer_to_inner = list(reversed(frame_bboxes_inner_to_outer))
    frame_colors = random.sample(FRAME_COLORS, depth)
    for bbox, color in zip(frame_bboxes_outer_to_inner, frame_colors):
        _draw_frame(group_inp, bbox, color)

    ir0, ic0, _ir1, _ic1 = frame_bboxes_inner_to_outer[0]
    pr0 = ir0 + 1 + top
    pc0 = ic0 + 1 + left
    group_inp[pr0 : pr0 + ph, pc0 : pc0 + pw][payload] = PAYLOAD_COLOR

    group_out = group_inp.copy()
    transformed = _transform_mask(payload, depth)
    region = group_out[pr0 : pr0 + ph, pc0 : pc0 + pw]
    region[payload] = BG
    region[transformed] = PAYLOAD_COLOR
    return group_inp, group_out


def _compose_groups(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        order = groups[:]
        random.shuffle(order)
        row_count = random.choice((1, 2))
        if row_count == 1:
            rows = [order]
        else:
            cut = random.randint(1, len(order) - 1)
            rows = [order[:cut], order[cut:]]

        row_widths: list[int] = []
        row_heights: list[int] = []
        x_positions: list[list[int]] = []
        for row in rows:
            xs: list[int] = []
            x = 0
            for i, (inp, _out) in enumerate(row):
                if i > 0:
                    x += random.randint(2, 5)
                xs.append(x)
                x += int(inp.shape[1])
            row_widths.append(x)
            row_heights.append(max(int(inp.shape[0]) for inp, _out in row))
            x_positions.append(xs)

        y_positions: list[int] = []
        y = 0
        for i, height in enumerate(row_heights):
            if i > 0:
                y += random.randint(2, 5)
            y_positions.append(y)
            y += height

        top = random.randint(0, 3)
        bottom = random.randint(0, 3)
        left = random.randint(0, 3)
        right = random.randint(0, 3)
        canvas_h = top + y + bottom
        canvas_w = left + max(row_widths) + right
        if canvas_h > MAX_CANVAS_SIZE or canvas_w > MAX_CANVAS_SIZE:
            continue

        inp = np.full((canvas_h, canvas_w), BG, dtype=int)
        out = np.full((canvas_h, canvas_w), BG, dtype=int)
        for row_index, row in enumerate(rows):
            row_y = top + y_positions[row_index]
            row_h = row_heights[row_index]
            for col_index, (group_inp, group_out) in enumerate(row):
                gh, gw = group_inp.shape
                dy = random.randint(0, row_h - gh)
                x = left + x_positions[row_index][col_index]
                y0 = row_y + dy
                inp[y0 : y0 + gh, x : x + gw] = group_inp
                out[y0 : y0 + gh, x : x + gw] = group_out
        return inp, out
    return None


def _sample_depths() -> list[int]:
    depths = [random.choice((1, 4)), 2, 3]
    extra_count = random.randint(0, MAX_GROUPS - MIN_GROUPS)
    for _ in range(extra_count):
        depths.append(random.randint(1, 4))
    random.shuffle(depths)
    return depths


def _gen_attempt() -> np.ndarray | None:
    depths = _sample_depths()
    groups = [_make_group(depth) for depth in depths]
    composed = _compose_groups(groups)
    if composed is None:
        return None
    inp, expected = composed

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-container sample")
