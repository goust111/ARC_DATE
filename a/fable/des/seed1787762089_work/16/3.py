from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
PALETTE = (2, 3, 4, 5, 6, 7, 8, 9)
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample(seq, k: int) -> list[int]:
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [int(seq[int(i)]) for i in idx]


def _bbox(points) -> tuple[int, int, int, int] | None:
    pts = np.asarray(list(points), dtype=int)
    if pts.size == 0:
        return None
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _normalize_points(points) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    pts = np.asarray(list(points), dtype=int)
    if pts.size == 0:
        return tuple(), (0, 0)
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    h = int(pts[:, 0].max()) - r0 + 1
    w = int(pts[:, 1].max()) - c0 + 1
    return norm, (h, w)


def _mask_from_norm(norm: tuple[tuple[int, int], ...]) -> np.ndarray:
    if not norm:
        return np.zeros((0, 0), dtype=bool)
    h = max(int(r) for r, _ in norm) + 1
    w = max(int(c) for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[int(r), int(c)] = True
    return mask


def _component_count(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            count += 1
            stack = [(int(r), int(c))]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
    return int(count)


def _best_vertical_axis(points, width: int) -> tuple[int, int, list[int]]:
    pts = {(int(r), int(c)) for r, c in points}
    best_axes: list[int] = []
    best_score = -1
    for axis in range(int(width)):
        score = sum(1 for r, c in pts if (int(r), int(2 * axis - c)) in pts)
        if score > best_score:
            best_score = int(score)
            best_axes = [int(axis)]
        elif score == best_score:
            best_axes.append(int(axis))
    assert best_axes
    return int(best_axes[0]), int(best_score), best_axes


def _extract_key_shape(grid: np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, list[tuple[int, int]], list[int]]:
    pts = np.argwhere(grid == BLUE)
    assert pts.size > 0, "grid must contain blue pixels"
    axis, _score, best_axes = _best_vertical_axis(pts, int(grid.shape[1]))
    assert len(best_axes) == 1, "mirror axis must be unique"
    blue = {(int(r), int(c)) for r, c in pts}
    unmatched = sorted((int(r), int(c)) for r, c in blue if (int(r), int(2 * axis - c)) not in blue)
    assert unmatched, "there must be at least one unmatched blue pixel"
    norm, _dims = _normalize_points(unmatched)
    return norm, int(axis), unmatched, best_axes


def _frame_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, BLUE)]
    candidates = []
    for color in colors:
        pts = np.argwhere(g == int(color))
        box = _bbox(pts)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not bool(np.all(sub[border] == int(color))):
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0:
            continue
        if bool(np.any((inner != BG) & (inner != int(color)))):
            continue
        pattern = np.argwhere(inner == int(color))
        if pattern.size == 0:
            continue
        shape, _dims = _normalize_points(pattern)
        candidates.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "interior": inner.copy(),
                "shape": shape,
            }
        )
    candidates.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1]), int(item["color"])))
    return candidates


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    key_shape, _axis, _unmatched, _axes = _extract_key_shape(g)
    candidates = _frame_candidates(g)
    assert len(candidates) == 3, "expected exactly three frame candidates"
    matches = [cand for cand in candidates if cand["shape"] == key_shape]
    assert len(matches) == 1, "key shape must match exactly one frame"
    return np.asarray(matches[0]["interior"], dtype=int).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract unmatched blue pixels under the best vertical mirror axis, then use
    that shape to select the matching frame and output its interior.
    """
    return _solve(grid)


def _grow_connected(allowed: np.ndarray, size: int, start: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = map(int, allowed.shape)
    sr, sc = map(int, start)
    if not (0 <= sr < h and 0 <= sc < w) or not bool(allowed[sr, sc]):
        return None
    cells = {(int(sr), int(sc))}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and bool(allowed[nr, nc]) and (nr, nc) not in cells:
                frontier.add((int(nr), int(nc)))

    add_frontier(sr, sc)
    while len(cells) < int(size):
        frontier_list = [cell for cell in frontier if cell not in cells]
        if not frontier_list:
            return None
        weights = np.asarray(
            [sum((cell[0] + dr, cell[1] + dc) in cells for dr, dc in DIRS) for cell in frontier_list],
            dtype=float,
        )
        weights /= weights.sum()
        cell = frontier_list[int(_RNG.choice(len(frontier_list), p=weights))]
        frontier.discard(cell)
        cells.add((int(cell[0]), int(cell[1])))
        add_frontier(int(cell[0]), int(cell[1]))
    return cells


def _make_symmetric_body() -> tuple[np.ndarray, int] | None:
    for _ in range(120):
        body_h = _randint(8, 11)
        half_w = _randint(4, 5)
        allowed = np.ones((body_h, half_w), dtype=bool)
        start = (_randint(0, body_h - 1), half_w - 1)
        size = _randint(max(body_h + 1, 12), min(body_h * half_w, 22))
        half = _grow_connected(allowed, size, start)
        if half is None:
            continue
        body_w = 2 * half_w - 1
        axis = half_w - 1
        mask = np.zeros((body_h, body_w), dtype=bool)
        for r, c in half:
            rr, cc = int(r), int(c)
            mask[rr, cc] = True
            mask[rr, 2 * axis - cc] = True
        area = int(mask.sum())
        if 26 <= area <= 44 and int(mask.shape[1]) >= 7:
            return mask, int(axis)
    return None


def _add_noise(body: np.ndarray, axis: int) -> tuple[np.ndarray, tuple[tuple[int, int], ...]] | None:
    body_h, body_w = map(int, body.shape)
    pad_r = 2
    pad_c = 6
    canvas = np.zeros((body_h + 2 * pad_r, body_w + 2 * pad_c), dtype=bool)
    canvas[pad_r : pad_r + body_h, pad_c : pad_c + body_w] = body
    axis_col = int(axis + pad_c)
    side = _choice(["left", "right"])
    target = _randint(4, 7)

    for _ in range(120):
        occupied = {(int(r), int(c)) for r, c in np.argwhere(canvas)}
        anchors = []
        for r, c in occupied:
            for dr, dc in DIRS:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1]):
                    continue
                if (nr, nc) in occupied:
                    continue
                if side == "left" and nc >= axis_col:
                    continue
                if side == "right" and nc <= axis_col:
                    continue
                mc = int(2 * axis_col - nc)
                if 0 <= mc < canvas.shape[1] and bool(canvas[nr, mc]):
                    continue
                anchors.append((int(nr), int(nc)))
        if not anchors:
            return None

        noise = {anchors[_randint(0, len(anchors) - 1)]}
        while len(noise) < target:
            frontier = []
            for r, c in noise:
                for dr, dc in DIRS:
                    nr, nc = int(r + dr), int(c + dc)
                    if not (0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1]):
                        continue
                    if (nr, nc) in occupied or (nr, nc) in noise:
                        continue
                    if side == "left" and nc >= axis_col:
                        continue
                    if side == "right" and nc <= axis_col:
                        continue
                    mc = int(2 * axis_col - nc)
                    if 0 <= mc < canvas.shape[1] and (bool(canvas[nr, mc]) or (nr, mc) in noise):
                        continue
                    frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            uniq = sorted(set(frontier))
            weights = np.asarray(
                [sum((cell[0] + dr, cell[1] + dc) in noise for dr, dc in DIRS) for cell in uniq],
                dtype=float,
            )
            weights /= weights.sum()
            cell = uniq[int(_RNG.choice(len(uniq), p=weights))]
            noise.add((int(cell[0]), int(cell[1])))
        if len(noise) != target:
            continue
        full = canvas.copy()
        for r, c in noise:
            full[int(r), int(c)] = True
        if _component_count(full) != 1:
            continue
        box = _bbox(np.argwhere(full))
        assert box is not None
        r0, c0, r1, c1 = box
        cropped = full[r0 : r1 + 1, c0 : c1 + 1]
        rel_noise = sorted((int(r - r0), int(c - c0)) for r, c in noise)
        norm, _dims = _normalize_points(rel_noise)
        return cropped, norm
    return None


def _random_shape(min_cells: int, max_cells: int) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]] | None:
    for _ in range(120):
        h = _randint(2, 5)
        w = _randint(2, 5)
        area = _randint(int(min_cells), min(int(max_cells), int(h * w)))
        shape = _grow_connected(np.ones((h, w), dtype=bool), area, (_randint(0, h - 1), _randint(0, w - 1)))
        if shape is None:
            continue
        norm, dims = _normalize_points(shape)
        if norm:
            return norm, dims
    return None


def _frame_interior_size(shape_dims: tuple[int, int]) -> tuple[int, int]:
    sh, sw = map(int, shape_dims)
    return sh + _randint(2, 4), sw + _randint(2, 4)


def _inner_offset(space: int) -> int:
    if space <= 0:
        return 0
    if space >= 2:
        return _randint(1, space - 1)
    return _randint(0, space)


def _build_frame(shape_norm: tuple[tuple[int, int], ...], color: int) -> tuple[np.ndarray, np.ndarray]:
    shape_mask = _mask_from_norm(shape_norm)
    sh, sw = map(int, shape_mask.shape)
    inner_h, inner_w = _frame_interior_size((sh, sw))
    inner = np.zeros((inner_h, inner_w), dtype=int)
    off_r = _inner_offset(inner_h - sh)
    off_c = _inner_offset(inner_w - sw)
    coords = np.argwhere(shape_mask)
    for r, c in coords:
        inner[int(off_r + r), int(off_c + c)] = int(color)
    frame = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    frame[0, :] = int(color)
    frame[-1, :] = int(color)
    frame[:, 0] = int(color)
    frame[:, -1] = int(color)
    frame[1:-1, 1:-1] = inner
    return frame, inner


def _gen_attempt() -> np.ndarray | None:
    body_info = _make_symmetric_body()
    if body_info is None:
        return None
    body_mask, axis = body_info
    noise_info = _add_noise(body_mask, axis)
    if noise_info is None:
        return None
    sprite_mask, key_shape = noise_info

    blue_pixels = int(sprite_mask.sum())
    if not (30 <= blue_pixels <= 52):
        return None

    key_mask = _mask_from_norm(key_shape)
    key_dims = tuple(map(int, key_mask.shape))
    if len(key_shape) < 4 or len(key_shape) > 7:
        return None

    colors = _sample(PALETTE, 3)
    match_idx = _randint(0, 2)
    used_shapes = {key_shape}
    frame_patches = []
    expected_output = None

    for i in range(3):
        if i == match_idx:
            shape = key_shape
        else:
            shape = None
            for _ in range(120):
                other = _random_shape(3, 7)
                if other is None:
                    continue
                other_shape, _other_dims = other
                if other_shape not in used_shapes:
                    shape = other_shape
                    break
            if shape is None:
                return None
            used_shapes.add(shape)
        patch, inner = _build_frame(shape, int(colors[i]))
        frame_patches.append((patch, inner))
        if i == match_idx:
            expected_output = inner.copy()

    assert expected_output is not None
    max_frame_w = max(int(patch.shape[1]) for patch, _inner in frame_patches)
    frame_heights = [int(patch.shape[0]) for patch, _inner in frame_patches]
    gaps_y = [_randint(1, 2), _randint(1, 2)]
    total_frames_h = frame_heights[0] + frame_heights[1] + frame_heights[2] + gaps_y[0] + gaps_y[1]

    sprite_h, sprite_w = map(int, sprite_mask.shape)
    left_margin = _randint(1, 2)
    gap_x = _randint(4, 6)
    right_margin = _randint(1, 2)
    frame_x_base = left_margin + sprite_w + gap_x
    width = frame_x_base + max_frame_w + right_margin
    if width < 22:
        width += 22 - width
    if width > 30:
        return None

    height = max(22, total_frames_h + 2 + _randint(0, 4), sprite_h + 2 + _randint(0, 8))
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    sprite_top = _randint(1, height - sprite_h - 1)
    sprite_left = left_margin
    grid[sprite_top : sprite_top + sprite_h, sprite_left : sprite_left + sprite_w][sprite_mask] = BLUE

    frame_top = _randint(1, height - total_frames_h - 1)
    cursor = frame_top
    for idx, (patch, _inner) in enumerate(frame_patches):
        ph, pw = map(int, patch.shape)
        shift = _randint(0, max_frame_w - pw)
        x = frame_x_base + shift
        grid[cursor : cursor + ph, x : x + pw] = patch
        if idx < 2:
            cursor += ph + gaps_y[idx]

    if _component_count(grid == BLUE) != 1:
        return None

    try:
        key_norm, _axis, unmatched, best_axes = _extract_key_shape(grid)
        candidates = _frame_candidates(grid)
        solved = _solve(grid)
    except AssertionError:
        return None

    if len(best_axes) != 1 or key_norm != key_shape:
        return None
    if len(unmatched) != len(key_shape):
        return None
    if len(candidates) != 3:
        return None
    if not np.array_equal(solved, expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
