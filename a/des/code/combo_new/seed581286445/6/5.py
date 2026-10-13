from __future__ import annotations

import numpy as np


_NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_MASTER_SIZE = 30
_MIN_SIDE = 3
_MAX_SIDE = 7
_GEN_ATTEMPTS = 400
_PLACEMENT_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in _NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _frame_from_component(coords: np.ndarray, color: int) -> dict | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected_area = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected_area:
        return None

    coord_set = {tuple(map(int, p)) for p in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "coords": coords.astype(int, copy=False),
    }


def _find_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _background_color(grid)
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _component_coords(grid == color):
            frame = _frame_from_component(coords, color)
            if frame is not None:
                frames.append(frame)
    return bg, frames


def _neighbor_sets(shape: tuple[int, int], frames: list[dict]) -> list[set[int]]:
    h, w = shape
    owner = np.full((h, w), -1, dtype=int)
    for idx, frame in enumerate(frames):
        coords = frame["coords"]
        owner[coords[:, 0], coords[:, 1]] = idx

    neighbors = [set() for _ in frames]
    for idx, frame in enumerate(frames):
        for r, c in frame["coords"]:
            r = int(r)
            c = int(c)
            for dr, dc in _NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(owner[nr, nc])
                    if other >= 0 and other != idx:
                        neighbors[idx].add(other)
    return neighbors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete isolated hollow rectangles, keep singly connected ones, and solidify highly connected ones.

    Input structure:
    - The grid has a dominant black background and several foreground objects.
    - Each foreground object is a monochrome one-pixel-thick hollow rectangle.
    - Some rectangles are 8-neighbor-adjacent to other rectangles, while others are isolated.

    Rule:
    - Identify each hollow rectangular frame and count how many other frames touch it through 8-neighbor adjacency.
    - If a frame has 0 neighbors, remove it entirely to background.
    - If a frame has exactly 1 neighbor, keep it unchanged.
    - If a frame has 2 or more neighbors, fill its interior with its own color so the whole bbox becomes solid.

    Key insight:
    - The transformation depends on each frame's degree in the adjacency graph, not on its size, position, or color.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames = _find_frames(g)
    if not frames:
        return np.full_like(g, bg)

    neighbor_sets = _neighbor_sets(g.shape, frames)
    out = np.full_like(g, bg)
    for frame, neighbors in zip(frames, neighbor_sets):
        degree = len(neighbors)
        if degree == 0:
            continue

        color = int(frame["color"])
        if degree == 1:
            coords = frame["coords"]
            out[coords[:, 0], coords[:, 1]] = color
            continue

        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _rand_side(rng: np.random.Generator) -> int:
    return int(rng.integers(_MIN_SIDE, _MAX_SIDE + 1))


def _frame_patch(height: int, width: int, color: int, bg: int = 0) -> np.ndarray:
    patch = np.full((int(height), int(width)), int(bg), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _draw_frame(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[top, left : left + width] = color
    canvas[top + height - 1, left : left + width] = color
    canvas[top : top + height, left] = color
    canvas[top : top + height, left + width - 1] = color


def _rects_to_patch(rects: list[tuple[int, int, int, int, int]], bg: int = 0) -> np.ndarray:
    min_r = min(r for r, _, _, _, _ in rects)
    min_c = min(c for _, c, _, _, _ in rects)
    max_r = max(r + h - 1 for r, _, h, _, _ in rects)
    max_c = max(c + w - 1 for _, c, _, w, _ in rects)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    for r, c, h, w, color in rects:
        _draw_frame(patch, r - min_r, c - min_c, h, w, color)
    return patch


def _isolated_patch(rng: np.random.Generator, color: int, bg: int = 0) -> np.ndarray:
    return _frame_patch(_rand_side(rng), _rand_side(rng), color, bg)


def _pair_patch(rng: np.random.Generator, colors: list[int], bg: int = 0) -> np.ndarray:
    h1, w1 = _rand_side(rng), _rand_side(rng)
    h2, w2 = _rand_side(rng), _rand_side(rng)
    mode = str(rng.choice(["horizontal", "vertical", "diagonal"]))
    if mode == "horizontal":
        top2 = int(rng.integers(-h2 + 1, h1))
        rects = [(0, 0, h1, w1, colors[0]), (top2, w1, h2, w2, colors[1])]
    elif mode == "vertical":
        left2 = int(rng.integers(-w2 + 1, w1))
        rects = [(0, 0, h1, w1, colors[0]), (h1, left2, h2, w2, colors[1])]
    else:
        dr = h1 if int(rng.integers(0, 2)) else -h2
        dc = w1 if int(rng.integers(0, 2)) else -w2
        rects = [(0, 0, h1, w1, colors[0]), (dr, dc, h2, w2, colors[1])]
    return _rects_to_patch(rects, bg)


def _chain_patch(rng: np.random.Generator, colors: list[int], bg: int = 0) -> np.ndarray:
    n = len(colors)
    dims = [(_rand_side(rng), _rand_side(rng)) for _ in range(n)]
    mode = str(rng.choice(["horizontal", "vertical"]))
    rects: list[tuple[int, int, int, int, int]] = []
    if mode == "horizontal":
        top = 0
        left = 0
        prev_h, prev_w = dims[0]
        rects.append((top, left, prev_h, prev_w, colors[0]))
        for idx in range(1, n):
            h, w = dims[idx]
            top = int(rng.integers(top - h + 1, top + prev_h))
            left += prev_w
            rects.append((top, left, h, w, colors[idx]))
            prev_h, prev_w = h, w
    else:
        top = 0
        left = 0
        prev_h, prev_w = dims[0]
        rects.append((top, left, prev_h, prev_w, colors[0]))
        for idx in range(1, n):
            h, w = dims[idx]
            left = int(rng.integers(left - w + 1, left + prev_w))
            top += prev_h
            rects.append((top, left, h, w, colors[idx]))
            prev_h, prev_w = h, w
    return _rects_to_patch(rects, bg)


def _rect_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 < br0 - pad
        or br1 < ar0 - pad
        or ac1 < bc0 - pad
        or bc1 < ac0 - pad
    )


def _place_patches(rng: np.random.Generator, patches: list[np.ndarray], bg: int = 0) -> np.ndarray | None:
    canvas = np.full((_MASTER_SIZE, _MASTER_SIZE), bg, dtype=int)
    placed_rects: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(_PLACEMENT_ATTEMPTS):
            top = int(rng.integers(0, _MASTER_SIZE - ph + 1))
            left = int(rng.integers(0, _MASTER_SIZE - pw + 1))
            rect = (top, left, top + ph - 1, left + pw - 1)
            if any(_rect_too_close(rect, other) for other in placed_rects):
                continue
            region = canvas[top : top + ph, left : left + pw]
            mask = patch != bg
            if np.any(region[mask] != bg):
                continue
            region[mask] = patch[mask]
            placed_rects.append(rect)
            placed = True
            break
        if not placed:
            return None
    return canvas


def _crop_with_margin(rng: np.random.Generator, canvas: np.ndarray, bg: int = 0) -> np.ndarray:
    coords = np.argwhere(canvas != bg)
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    top_margin = int(rng.integers(0, min(4, int(r0)) + 1))
    left_margin = int(rng.integers(0, min(4, int(c0)) + 1))
    bottom_margin = int(rng.integers(0, min(4, int(canvas.shape[0] - 1 - r1)) + 1))
    right_margin = int(rng.integers(0, min(4, int(canvas.shape[1] - 1 - c1)) + 1))
    return canvas[
        int(r0) - top_margin : int(r1) + 1 + bottom_margin,
        int(c0) - left_margin : int(c1) + 1 + right_margin,
    ]


def _degrees(grid: np.ndarray) -> list[int]:
    _bg, frames = _find_frames(grid)
    return [len(neigh) for neigh in _neighbor_sets(grid.shape, frames)]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = 0
    chain_len = int(rng.integers(3, 5))
    n_isolates = int(rng.integers(1, 3))
    include_pair = bool(int(rng.integers(0, 2)))
    n_objects = chain_len + n_isolates + (2 if include_pair else 0)
    if n_objects > 9:
        return None

    colors = [int(c) for c in rng.choice(np.arange(1, 10), size=n_objects, replace=False)]
    idx = 0
    patches = [_chain_patch(rng, colors[idx : idx + chain_len], bg)]
    idx += chain_len
    for _ in range(n_isolates):
        patches.append(_isolated_patch(rng, colors[idx], bg))
        idx += 1
    if include_pair:
        patches.append(_pair_patch(rng, colors[idx : idx + 2], bg))

    rng.shuffle(patches)
    canvas = _place_patches(rng, patches, bg)
    if canvas is None:
        return None

    grid = _crop_with_margin(rng, canvas, bg)
    if grid.ndim != 2 or max(grid.shape) > 30:
        return None

    bg_color, frames = _find_frames(grid)
    if bg_color != bg or len(frames) != n_objects:
        return None

    degrees = _degrees(grid)
    if not any(d == 0 for d in degrees):
        return None
    if not any(d == 1 for d in degrees):
        return None
    if not any(d >= 2 for d in degrees):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_ATTEMPTS} attempts")
