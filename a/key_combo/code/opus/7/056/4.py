import numpy as np


BG = 0
FRAME = 3
SEPARATOR = 7
PALETTE = np.array([1, 2, 4, 5, 6, 8, 9], dtype=int)
MAX_ATTEMPTS = 400


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
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEPARATOR)]
    assert len(rows) == 1, "expected exactly one full-width separator row"
    return rows[0]


def _parse_frame(grid: np.ndarray, coords: np.ndarray) -> tuple[tuple[int, int], dict[tuple[int, int], int]]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    line_rows = [r for r in range(sub.shape[0]) if np.all(sub[r] == FRAME)]
    line_cols = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == FRAME)]
    assert len(line_rows) >= 3 and len(line_cols) >= 3

    slot_rows = len(line_rows) - 1
    slot_cols = len(line_cols) - 1
    slot_colors: dict[tuple[int, int], int] = {}
    for sr in range(slot_rows):
        for sc in range(slot_cols):
            block = sub[line_rows[sr] + 1 : line_rows[sr + 1], line_cols[sc] + 1 : line_cols[sc + 1]]
            assert block.size > 0
            colors = set(map(int, np.unique(block)))
            assert FRAME not in colors
            if colors == {BG}:
                continue
            assert len(colors) == 1, "slot interiors must be uniform"
            slot_colors[(sr, sc)] = next(iter(colors))
    return (slot_rows, slot_cols), slot_colors


def _infer_seed(top: np.ndarray) -> np.ndarray:
    frames = _components(top == FRAME)
    assert len(frames) >= 2

    layout: tuple[int, int] | None = None
    seen: dict[tuple[int, int], int] = {}
    for frame in frames:
        cur_layout, slot_colors = _parse_frame(top, frame)
        if layout is None:
            layout = cur_layout
        assert cur_layout == layout
        for pos, color in slot_colors.items():
            prev = seen.get(pos)
            if prev is None:
                seen[pos] = color
            else:
                assert prev == color

    assert layout is not None
    seed = np.zeros(layout, dtype=int)
    for sr in range(layout[0]):
        for sc in range(layout[1]):
            assert (sr, sc) in seen, "every logical slot must be visible in at least one frame"
            seed[sr, sc] = seen[(sr, sc)]
    return seed


def _find_cavity(bottom: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for coords in _components(bottom == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = bottom[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(sub[0] == FRAME):
            continue
        if not np.all(sub[-1] == FRAME):
            continue
        if not np.all(sub[:, 0] == FRAME):
            continue
        if not np.all(sub[:, -1] == FRAME):
            continue
        if np.any(sub[1:-1, 1:-1] == FRAME):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one rectangular cavity wall"
    return candidates[0]


def _tile_into(shape: tuple[int, int], seed: np.ndarray) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None] % seed.shape[0]
    cc = np.arange(w)[None, :] % seed.shape[1]
    return seed[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the complete top lattice pattern and use it as the tile for the lower cavity.

    Input structure:
    - A black background is split by one full-width orange separator row.
    - Above the separator are several green one-pixel lattice frames with the same logical slot layout; some slots are colored and others remain black.
    - Below the separator is one green rectangular wall enclosing a black cavity.

    Rule:
    - Read each top lattice frame, identify its logical slot grid, and collect the color shown for every slot position across all frames.
    - Assemble those per-position colors into one seed matrix indexed by slot row and slot column.
    - Remove the separator and everything above it, then tile the seed matrix from the cavity interior's top-left corner until the lower cavity is filled.

    Key insight:
    - The top lattices are fragmented storage for one shared pattern, and that recovered pattern is not restored in place but reused as the periodic seed for a different region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :].copy()

    seed = _infer_seed(top)
    r0, c0, r1, c1 = _find_cavity(bottom)
    inner = _tile_into((r1 - r0 - 1, c1 - c0 - 1), seed)
    bottom[r0 + 1 : r1, c0 + 1 : c1] = inner
    return bottom


def _sample_seed(rng: np.random.Generator) -> np.ndarray:
    while True:
        rows, cols = ((2, 2), (2, 3), (3, 2), (3, 3))[int(rng.integers(0, 4))]
        palette_size = int(rng.integers(3, min(6, len(PALETTE)) + 1))
        palette = rng.choice(PALETTE, size=palette_size, replace=False)
        seed = palette[rng.integers(0, palette_size, size=(rows, cols))]
        unique_count = len(set(map(int, seed.ravel())))
        min_unique = 3 if rows * cols >= 6 else 2
        if unique_count < min_unique:
            continue
        if rows > 1 and np.all(seed[0] == seed[1:]):
            continue
        if cols > 1 and np.all(seed[:, 0:1] == seed[:, 1:]):
            continue
        return seed.astype(int)


def _sample_frame_spec(rng: np.random.Generator, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    heights = rng.integers(1, 3, size=rows)
    widths = rng.integers(1, 3, size=cols)
    return heights.astype(int), widths.astype(int)


def _render_frame(
    seed: np.ndarray,
    heights: np.ndarray,
    widths: np.ndarray,
    shown: set[tuple[int, int]],
) -> np.ndarray:
    rows, cols = seed.shape
    h = int(heights.sum() + rows + 1)
    w = int(widths.sum() + cols + 1)
    frame = np.zeros((h, w), dtype=int)

    line_rows = [0]
    for size in heights:
        line_rows.append(line_rows[-1] + int(size) + 1)
    line_cols = [0]
    for size in widths:
        line_cols.append(line_cols[-1] + int(size) + 1)

    frame[line_rows, :] = FRAME
    frame[:, line_cols] = FRAME
    for sr in range(rows):
        for sc in range(cols):
            if (sr, sc) not in shown:
                continue
            frame[
                line_rows[sr] + 1 : line_rows[sr + 1],
                line_cols[sc] + 1 : line_cols[sc + 1],
            ] = seed[sr, sc]
    return frame


def _distribute_hints(
    rng: np.random.Generator, rows: int, cols: int, n_frames: int
) -> list[set[tuple[int, int]]]:
    positions = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(positions)
    shown = [set() for _ in range(n_frames)]
    for idx, pos in enumerate(positions):
        shown[idx % n_frames].add(pos)

    extra_count = int(rng.integers(0, len(positions) // 2 + 1))
    extras = positions[:]
    rng.shuffle(extras)
    for pos in extras[:extra_count]:
        frame_ids = [i for i in range(n_frames) if pos not in shown[i]]
        if not frame_ids:
            continue
        target = int(frame_ids[int(rng.integers(0, len(frame_ids)))])
        shown[target].add(pos)

    total = len(positions)
    for frame_id in range(n_frames):
        if len(shown[frame_id]) == total:
            removable = [pos for pos in shown[frame_id] if sum(pos in s for s in shown) >= 2]
            assert removable
            drop = removable[int(rng.integers(0, len(removable)))]
            shown[frame_id].remove(drop)
        assert shown[frame_id]
        assert len(shown[frame_id]) < total
    return shown


def _expected_bottom(
    bottom: np.ndarray, cavity_bbox: tuple[int, int, int, int], seed: np.ndarray
) -> np.ndarray:
    out = bottom.copy()
    r0, c0, r1, c1 = cavity_bbox
    out[r0 + 1 : r1, c0 + 1 : c1] = _tile_into((r1 - r0 - 1, c1 - c0 - 1), seed)
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    seed = _sample_seed(rng)
    seed_rows, seed_cols = seed.shape
    n_frames = int(rng.integers(2, 4))
    shown = _distribute_hints(rng, seed_rows, seed_cols, n_frames)

    frame_specs = [_sample_frame_spec(rng, seed_rows, seed_cols) for _ in range(n_frames)]
    frame_imgs = [_render_frame(seed, hs, ws, shown[i]) for i, (hs, ws) in enumerate(frame_specs)]
    frame_heights = [img.shape[0] for img in frame_imgs]
    frame_widths = [img.shape[1] for img in frame_imgs]

    gaps = [int(rng.integers(1, 3)) for _ in range(max(0, n_frames - 1))]
    top_used_w = 1 + sum(frame_widths) + sum(gaps) + 1
    top_h = max(frame_heights) + 2

    rep_r = int(rng.integers(2, 6))
    rep_c = int(rng.integers(2, 6))
    cavity_inner_h = seed_rows * rep_r
    cavity_inner_w = seed_cols * rep_c
    cavity_outer_h = cavity_inner_h + 2
    cavity_outer_w = cavity_inner_w + 2

    bottom_h = int(rng.integers(1, 3)) + cavity_outer_h + int(rng.integers(1, 3))
    min_bottom_w = cavity_outer_w + 2
    base_w = max(top_used_w, min_bottom_w)
    if top_h + 1 + bottom_h > 30 or base_w > 30:
        return None

    total_w = int(base_w + rng.integers(0, min(30 - base_w, 3) + 1))
    if total_w > 30:
        return None

    grid = np.zeros((top_h + 1 + bottom_h, total_w), dtype=int)

    row = 1
    col = 1
    for i, img in enumerate(frame_imgs):
        h, w = img.shape
        grid[row : row + h, col : col + w] = img
        col += w
        if i < len(gaps):
            col += gaps[i]

    sep = top_h
    grid[sep, :] = SEPARATOR

    bottom = grid[sep + 1 :]
    cavity_top = int(rng.integers(1, bottom_h - cavity_outer_h))
    cavity_left = int(rng.integers(1, total_w - cavity_outer_w))
    r0, c0 = cavity_top, cavity_left
    r1, c1 = r0 + cavity_outer_h - 1, c0 + cavity_outer_w - 1
    bottom[r0, c0 : c1 + 1] = FRAME
    bottom[r1, c0 : c1 + 1] = FRAME
    bottom[r0 : r1 + 1, c0] = FRAME
    bottom[r0 : r1 + 1, c1] = FRAME

    expected = _expected_bottom(bottom, (r0, c0, r1, c1), seed)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
