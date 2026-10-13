import random

import numpy as np


BG = 0
FRAME_COLOR = 5
ALL_COLORS = tuple(range(10))
MARKER_POOL = tuple(color for color in ALL_COLORS if color not in {BG, FRAME_COLOR})
LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
STAMP_SIZE_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dominant_color(values: np.ndarray) -> int:
    flat = values.astype(int).ravel()
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _get_bg(grid: np.ndarray) -> int:
    return _dominant_color(grid)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    candidates: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if coords.shape[0] != int(border.sum()):
            continue
        if not np.all(sub[border] != bg):
            continue
        if not np.all(sub[~border] == bg):
            continue
        candidates.append(((r1 - r0 + 1) * (c1 - c0 + 1), (r0, c0, r1, c1), coords))
    assert candidates, "expected a hollow rectangular frame"
    _, bbox, coords = max(candidates, key=lambda item: item[0])
    return bbox, coords


def _uniform_spacing(positions: list[int]) -> int | None:
    if len(positions) < 2:
        return None
    deltas = [b - a for a, b in zip(positions, positions[1:])]
    if len(set(deltas)) != 1:
        raise AssertionError("frame markers must be equally spaced")
    return deltas[0]


def _parse_stamp(
    grid: np.ndarray,
    coords: np.ndarray,
    bg: int,
    stamp_size: int,
) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = stamp_size + 1
    assert sub.shape == (expected, expected), "stamp component must be (s+1)x(s+1)"

    row_counts = np.count_nonzero(sub != bg, axis=1)
    col_counts = np.count_nonzero(sub != bg, axis=0)
    sparse_rows = np.flatnonzero(row_counts == 1)
    sparse_cols = np.flatnonzero(col_counts == 1)
    assert len(sparse_rows) == 1 and len(sparse_cols) == 1, "expected one row marker and one column marker"

    sparse_r = int(sparse_rows[0])
    sparse_c = int(sparse_cols[0])
    assert sparse_r in (0, stamp_size) and sparse_c in (0, stamp_size), "markers must sit on outer sides"

    row_offset = 1 if sparse_r == 0 else 0
    col_offset = 1 if sparse_c == 0 else 0
    patch = sub[row_offset : row_offset + stamp_size, col_offset : col_offset + stamp_size].copy()
    assert patch.shape == (stamp_size, stamp_size)
    assert np.all(patch != bg), "generated stamps are solid rectangular patches"

    row_marker_cols = np.flatnonzero(sub[sparse_r] != bg)
    col_marker_rows = np.flatnonzero(sub[:, sparse_c] != bg)
    assert len(row_marker_cols) == 1 and len(col_marker_rows) == 1
    row_color = int(sub[sparse_r, int(row_marker_cols[0])])
    col_color = int(sub[int(col_marker_rows[0]), sparse_c])
    return patch, row_color, col_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange scattered stamp patches into the album slots indexed by colored border markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    (fr0, fc0, fr1, fc1), frame_coords = _find_frame(g, bg)
    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[frame_coords[:, 0], frame_coords[:, 1]] = True
    frame_color = _dominant_color(g[frame_mask])

    top_markers = [(c, int(g[fr0, c])) for c in range(fc0 + 1, fc1) if int(g[fr0, c]) != frame_color]
    left_markers = [(r, int(g[r, fc0])) for r in range(fr0 + 1, fr1) if int(g[r, fc0]) != frame_color]
    assert top_markers and left_markers, "frame must contain row and column markers"

    col_positions = [pos for pos, _color in top_markers]
    row_positions = [pos for pos, _color in left_markers]
    stamp_size_values: list[int] = []
    top_spacing = _uniform_spacing(col_positions)
    left_spacing = _uniform_spacing(row_positions)
    if top_spacing is not None:
        stamp_size_values.append(top_spacing - 1)
    if left_spacing is not None:
        stamp_size_values.append(left_spacing - 1)

    non_frame_mask = (g != bg) & (~frame_mask)
    stamp_components = _components(non_frame_mask)
    assert stamp_components, "expected scattered stamps outside the frame"
    if not stamp_size_values:
        stamp_size_values.append(_bbox(stamp_components[0])[2] - _bbox(stamp_components[0])[0])
    stamp_size = stamp_size_values[0]
    assert all(size == stamp_size for size in stamp_size_values), "row and column spacings must agree"

    row_map = {int(color): int(pos) for pos, color in left_markers}
    col_map = {int(color): int(pos) for pos, color in top_markers}
    assert len(row_map) == len(left_markers) and len(col_map) == len(top_markers), "marker colors must be unique per edge"

    out = g.copy()
    for coords in stamp_components:
        out[coords[:, 0], coords[:, 1]] = bg

    for coords in stamp_components:
        patch, row_color, col_color = _parse_stamp(g, coords, bg, stamp_size)
        dest_r = row_map[row_color]
        dest_c = col_map[col_color]
        out[dest_r : dest_r + stamp_size, dest_c : dest_c + stamp_size] = patch

    return out


def _random_patch(rng: random.Random, size: int, palette: list[int]) -> np.ndarray:
    for _ in range(100):
        used = rng.sample(palette, rng.randint(2, min(4, len(palette))))
        mode = rng.randrange(5)
        patch = np.empty((size, size), dtype=int)

        if mode == 0:
            for r in range(size):
                for c in range(size):
                    patch[r, c] = rng.choice(used)
        elif mode == 1:
            for r in range(size):
                patch[r, :] = rng.choice(used)
        elif mode == 2:
            for c in range(size):
                patch[:, c] = rng.choice(used)
        elif mode == 3:
            for r in range(size):
                for c in range(size):
                    patch[r, c] = used[(r + c) % len(used)]
        else:
            split_r = rng.randint(1, size - 1)
            split_c = rng.randint(1, size - 1)
            block_colors = [rng.choice(used) for _ in range(4)]
            patch[:split_r, :split_c] = block_colors[0]
            patch[:split_r, split_c:] = block_colors[1]
            patch[split_r:, :split_c] = block_colors[2]
            patch[split_r:, split_c:] = block_colors[3]

        if len(set(map(int, patch.ravel()))) < 2:
            patch[rng.randrange(size), rng.randrange(size)] = used[0]
            patch[(rng.randrange(size) + 1) % size, (rng.randrange(size) + 1) % size] = used[-1]
        if len(set(map(int, patch.ravel()))) >= 2:
            return patch
    raise RuntimeError("failed to build a non-uniform stamp patch")


def _make_stamp_component(
    rng: random.Random,
    patch: np.ndarray,
    row_color: int,
    col_color: int,
) -> np.ndarray:
    size = int(patch.shape[0])
    comp = np.full((size + 1, size + 1), BG, dtype=int)
    row_on_top = bool(rng.randrange(2))
    col_on_left = bool(rng.randrange(2))
    patch_r0 = 1 if row_on_top else 0
    patch_c0 = 1 if col_on_left else 0
    comp[patch_r0 : patch_r0 + size, patch_c0 : patch_c0 + size] = patch

    row_marker_r = 0 if row_on_top else size
    row_marker_c = patch_c0 + rng.randrange(size)
    col_marker_r = patch_r0 + rng.randrange(size)
    col_marker_c = 0 if col_on_left else size
    comp[row_marker_r, row_marker_c] = row_color
    comp[col_marker_r, col_marker_c] = col_color
    return comp


def _rect_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    stamp_size: int,
    row_colors: list[int],
    col_colors: list[int],
) -> tuple[list[int], list[int]]:
    n_rows = len(row_colors)
    n_cols = len(col_colors)
    outer_h = n_rows * (stamp_size + 1) + 1
    outer_w = n_cols * (stamp_size + 1) + 1
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR

    row_starts = [top + 1 + idx * (stamp_size + 1) for idx in range(n_rows)]
    col_starts = [left + 1 + idx * (stamp_size + 1) for idx in range(n_cols)]
    for pos, color in zip(col_starts, col_colors):
        grid[top, pos] = color
    for pos, color in zip(row_starts, row_colors):
        grid[pos, left] = color
    return row_starts, col_starts


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    stamp_size = rng.randint(*STAMP_SIZE_RANGE)
    n_rows, n_cols = rng.choice(LAYOUT_OPTIONS)
    pattern_palette_size = len(MARKER_POOL) - n_rows - n_cols
    assert pattern_palette_size >= 2

    chosen = rng.sample(list(MARKER_POOL), n_rows + n_cols + pattern_palette_size)
    row_colors = chosen[:n_rows]
    col_colors = chosen[n_rows : n_rows + n_cols]
    pattern_palette = chosen[n_rows + n_cols :]

    all_slots = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    n_stamps = rng.randint(2, len(all_slots))
    rng.shuffle(all_slots)
    used_slots = all_slots[:n_stamps]

    outer_h = n_rows * (stamp_size + 1) + 1
    outer_w = n_cols * (stamp_size + 1) + 1
    comp_size = stamp_size + 1

    min_h = max(16, outer_h + comp_size + 3)
    min_w = max(16, outer_w + comp_size + 3)
    if min_h > 30 or min_w > 30:
        raise RuntimeError("layout exceeds ARC-sized grid")
    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)

    frame_top = rng.randint(1, grid_h - outer_h - 1)
    frame_left = rng.randint(1, grid_w - outer_w - 1)
    frame_bbox = (frame_top, frame_left, frame_top + outer_h - 1, frame_left + outer_w - 1)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)
    row_starts, col_starts = _draw_frame(grid, frame_top, frame_left, stamp_size, row_colors, col_colors)
    _draw_frame(expected, frame_top, frame_left, stamp_size, row_colors, col_colors)

    patch_bytes: set[bytes] = set()
    slot_to_patch: dict[tuple[int, int], np.ndarray] = {}
    for slot in used_slots:
        for _ in range(100):
            patch = _random_patch(rng, stamp_size, pattern_palette)
            key = patch.tobytes()
            if key not in patch_bytes:
                patch_bytes.add(key)
                slot_to_patch[slot] = patch
                break
        else:
            raise RuntimeError("failed to build unique stamp patches")

    candidates = [
        (r, c)
        for r in range(grid_h - comp_size + 1)
        for c in range(grid_w - comp_size + 1)
    ]
    rng.shuffle(candidates)
    occupied = [frame_bbox]
    placements: list[tuple[int, int, np.ndarray]] = []

    for slot in used_slots:
        patch = slot_to_patch[slot]
        row_idx, col_idx = slot
        comp = _make_stamp_component(rng, patch, row_colors[row_idx], col_colors[col_idx])
        found = None
        for r, c in candidates:
            bbox = (r, c, r + comp_size - 1, c + comp_size - 1)
            if all(_rect_far_enough(bbox, other, gap=1) for other in occupied):
                found = (r, c)
                occupied.append(bbox)
                break
        if found is None:
            raise RuntimeError("failed to place all scattered stamps")
        placements.append((found[0], found[1], comp))
        expected[
            row_starts[row_idx] : row_starts[row_idx] + stamp_size,
            col_starts[col_idx] : col_starts[col_idx] + stamp_size,
        ] = patch

    for r, c, comp in placements:
        grid[r : r + comp_size, c : c + comp_size] = comp

    return grid, expected


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _build_case(rng)
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate valid stamp-album case")


def generate_input() -> np.ndarray:
    grid, _expected = _generate_case()
    return grid
