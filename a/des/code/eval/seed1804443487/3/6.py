from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return tuple(out)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            r0 = int(coord_arr[:, 0].min())
            c0 = int(coord_arr[:, 1].min())
            r1 = int(coord_arr[:, 0].max())
            c1 = int(coord_arr[:, 1].max())
            components.append(
                {
                    "color": int(color),
                    "coords": coord_arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )
    return components


def _is_frame_component(component: dict) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(component["area"]) != 2 * (h + w) - 4:
        return False

    local = np.zeros((h, w), dtype=bool)
    coords = component["coords"]
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not local[0, :].all():
        return False
    if not local[-1, :].all():
        return False
    if not local[:, 0].all():
        return False
    if not local[:, -1].all():
        return False
    return not local[1:-1, 1:-1].any()


def _strictly_contains(outer_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = map(int, outer_bbox)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested rectangular containers, then flip each container's contents
    from the inside out: odd depths flip vertically, even depths horizontally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frames: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for component in _find_color_components(g, color):
            if _is_frame_component(component):
                frames.append(component)

    if not frames:
        return g.copy()

    parents: list[int | None] = [None] * len(frames)
    for i, inner in enumerate(frames):
        inner_bbox = inner["bbox"]
        containing: list[tuple[int, int]] = []
        for j, outer in enumerate(frames):
            if i == j:
                continue
            if _strictly_contains(outer["bbox"], inner_bbox):
                r0, c0, r1, c1 = map(int, outer["bbox"])
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                containing.append((area, j))
        if containing:
            parents[i] = min(containing)[1]

    depths = [0] * len(frames)

    def _depth(idx: int) -> int:
        if depths[idx]:
            return depths[idx]
        parent = parents[idx]
        depths[idx] = 1 if parent is None else _depth(parent) + 1
        return depths[idx]

    order = sorted(range(len(frames)), key=_depth, reverse=True)
    out = g.copy()
    for idx in order:
        r0, c0, r1, c1 = map(int, frames[idx]["bbox"])
        interior = out[r0 + 1 : r1, c0 + 1 : c1].copy()
        if _depth(idx) % 2 == 1:
            interior = np.flipud(interior)
        else:
            interior = np.fliplr(interior)
        out[r0 + 1 : r1, c0 + 1 : c1] = interior
    return out


# --------------- generator ---------------

GRID_MIN_SIDE = 12
GRID_MAX_SIDE = 30
MAX_ATTEMPTS = 400
LAYOUT_GAP_RANGE = (2, 4)
OUTER_PADDING_RANGE = (1, 4)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _grow_polyomino(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, height, width):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _tight_crop(mask)


def _make_pattern_patch(color: int, bg: int) -> np.ndarray | None:
    for _ in range(100):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        max_area = min(5, height * width - 1)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _grow_polyomino(height, width, area)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if np.array_equal(mask, np.flipud(mask)) and np.array_equal(mask, np.fliplr(mask)):
            continue
        patch = np.full(mask.shape, int(bg), dtype=int)
        patch[mask] = int(color)
        return patch
    return None


def _sample_padding(depth: int) -> tuple[int, int, int, int]:
    asym = random.choice([(1, 2), (2, 1)])
    if depth % 2 == 1:
        top, bottom = asym
        left = random.randint(1, 2)
        right = random.randint(1, 2)
    else:
        left, right = asym
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
    return top, bottom, left, right


def _wrap_frame(
    child_input: np.ndarray,
    child_output: np.ndarray,
    border_color: int,
    bg: int,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    top, bottom, left, right = _sample_padding(depth)
    child_h, child_w = map(int, child_input.shape)
    inner_h = child_h + top + bottom
    inner_w = child_w + left + right

    interior_input = np.full((inner_h, inner_w), int(bg), dtype=int)
    interior_input[top : top + child_h, left : left + child_w] = child_input

    interior_output = np.full((inner_h, inner_w), int(bg), dtype=int)
    interior_output[top : top + child_h, left : left + child_w] = child_output
    if depth % 2 == 1:
        interior_output = np.flipud(interior_output)
    else:
        interior_output = np.fliplr(interior_output)

    patch_h = inner_h + 2
    patch_w = inner_w + 2
    patch_input = np.full((patch_h, patch_w), int(bg), dtype=int)
    patch_output = np.full((patch_h, patch_w), int(bg), dtype=int)

    patch_input[0, :] = int(border_color)
    patch_input[-1, :] = int(border_color)
    patch_input[:, 0] = int(border_color)
    patch_input[:, -1] = int(border_color)
    patch_output[0, :] = int(border_color)
    patch_output[-1, :] = int(border_color)
    patch_output[:, 0] = int(border_color)
    patch_output[:, -1] = int(border_color)

    patch_input[1:-1, 1:-1] = interior_input
    patch_output[1:-1, 1:-1] = interior_output
    return patch_input, patch_output


def _build_chain(depth: int, frame_colors: list[int], pattern_color: int, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    pattern = _make_pattern_patch(pattern_color, bg)
    if pattern is None:
        return None

    child_input = pattern
    child_output = pattern
    for level in range(depth, 0, -1):
        child_input, child_output = _wrap_frame(
            child_input,
            child_output,
            border_color=frame_colors[level - 1],
            bg=bg,
            depth=level,
        )
    return child_input, child_output


def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, src.shape)
    dst[top : top + h, left : left + w] = src


def _sample_depths() -> list[int]:
    if random.random() < 0.55:
        return [random.randint(1, 4)]
    pairs = [(a, b) for a in range(1, 5) for b in range(1, 5) if a + b + 2 <= 9]
    return list(random.choice(pairs))


def _compose_roots(
    roots_in: list[np.ndarray],
    roots_out: list[np.ndarray],
    bg: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    count = len(roots_in)
    order = list(range(count))
    random.shuffle(order)
    roots_in = [roots_in[i] for i in order]
    roots_out = [roots_out[i] for i in order]

    top_pad = random.randint(*OUTER_PADDING_RANGE)
    bottom_pad = random.randint(*OUTER_PADDING_RANGE)
    left_pad = random.randint(*OUTER_PADDING_RANGE)
    right_pad = random.randint(*OUTER_PADDING_RANGE)
    gap = random.randint(*LAYOUT_GAP_RANGE)
    horizontal = True if count == 1 else random.choice([True, False])

    if horizontal:
        body_h = max(int(root.shape[0]) for root in roots_in)
        body_w = sum(int(root.shape[1]) for root in roots_in) + gap * (count - 1)
        grid_h = body_h + top_pad + bottom_pad
        grid_w = body_w + left_pad + right_pad
        if not (GRID_MIN_SIDE <= grid_h <= GRID_MAX_SIDE and GRID_MIN_SIDE <= grid_w <= GRID_MAX_SIDE):
            return None

        inp = np.full((grid_h, grid_w), int(bg), dtype=int)
        out = np.full((grid_h, grid_w), int(bg), dtype=int)
        cursor = left_pad
        for patch_in, patch_out in zip(roots_in, roots_out):
            h = int(patch_in.shape[0])
            top = top_pad + random.randint(0, body_h - h)
            _paste(inp, patch_in, top, cursor)
            _paste(out, patch_out, top, cursor)
            cursor += int(patch_in.shape[1]) + gap
        return inp, out

    body_h = sum(int(root.shape[0]) for root in roots_in) + gap * (count - 1)
    body_w = max(int(root.shape[1]) for root in roots_in)
    grid_h = body_h + top_pad + bottom_pad
    grid_w = body_w + left_pad + right_pad
    if not (GRID_MIN_SIDE <= grid_h <= GRID_MAX_SIDE and GRID_MIN_SIDE <= grid_w <= GRID_MAX_SIDE):
        return None

    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    out = np.full((grid_h, grid_w), int(bg), dtype=int)
    cursor = top_pad
    for patch_in, patch_out in zip(roots_in, roots_out):
        w = int(patch_in.shape[1])
        left = left_pad + random.randint(0, body_w - w)
        _paste(inp, patch_in, cursor, left)
        _paste(out, patch_out, cursor, left)
        cursor += int(patch_in.shape[0]) + gap
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    depths = _sample_depths()
    total_frame_colors = sum(depths)
    total_needed = total_frame_colors + len(depths)
    palette = [color for color in range(10) if color != bg]
    if total_needed > len(palette):
        return None

    chosen = random.sample(palette, total_needed)
    frame_colors_flat = chosen[:total_frame_colors]
    pattern_colors = chosen[total_frame_colors:]

    roots_in: list[np.ndarray] = []
    roots_out: list[np.ndarray] = []
    cursor = 0
    for depth, pattern_color in zip(depths, pattern_colors):
        colors = frame_colors_flat[cursor : cursor + depth]
        cursor += depth
        built = _build_chain(depth, colors, pattern_color, bg)
        if built is None:
            return None
        patch_in, patch_out = built
        roots_in.append(patch_in)
        roots_out.append(patch_out)

    composed = _compose_roots(roots_in, roots_out, bg)
    if composed is None:
        return None
    inp, expected = composed
    if _most_common_color(inp) != bg:
        return None
    if np.array_equal(inp, expected):
        return None
    if Counter(map(int, inp.ravel())).most_common(1)[0][0] != bg:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
