from __future__ import annotations

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _run_starts(grid: np.ndarray, axis: int) -> list[int]:
    starts = [0]
    for i in range(1, grid.shape[axis]):
        prev = grid[i - 1, :] if axis == 0 else grid[:, i - 1]
        cur = grid[i, :] if axis == 0 else grid[:, i]
        if not np.array_equal(prev, cur):
            starts.append(i)
    return starts


def _compress_panel(panel: np.ndarray) -> np.ndarray:
    row_starts = _run_starts(panel, axis=0)
    col_starts = _run_starts(panel, axis=1)
    assert len(row_starts) < panel.shape[0], "panel must contain repeated row runs"
    assert len(col_starts) < panel.shape[1], "panel must contain repeated column runs"
    return panel[np.ix_(row_starts, col_starts)].copy()


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2-D grid"
    bg = _most_common_color(g)
    separators = [
        r
        for r in range(g.shape[0])
        if np.all(g[r, :] == int(g[r, 0])) and int(g[r, 0]) != bg
    ]
    assert len(separators) == 3, "expected exactly three full-width separator rows"
    panel_h = separators[0]
    assert panel_h > 0, "panels must be non-empty"
    expected = [panel_h + k * (panel_h + 1) for k in range(3)]
    assert separators == expected, "separator rows must split four equal-height panels"
    assert g.shape[0] == 4 * panel_h + 3, "unexpected stacked-panel height"
    panels = []
    start = 0
    for sep in separators:
        panels.append(g[start:sep, :].copy())
        start = sep + 1
    panels.append(g[start:, :].copy())
    assert len({panel.shape for panel in panels}) == 1, "observed panels must share a shape"
    return panels, bg


def _top_left_mask(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "mask must contain at least one cell"
    r0, c0 = coords.min(axis=0)
    return int(r0), int(c0)


def _top_left(frame: np.ndarray, bg: int) -> tuple[int, int]:
    coords = np.argwhere(frame != bg)
    assert len(coords) > 0, "each panel must contain a foreground sprite"
    return _top_left_mask(frame != bg)


def _sprite_signature(frame: np.ndarray, color: int) -> tuple[tuple[int, int], ...]:
    r0, c0 = _top_left_mask(frame == color)
    coords = np.argwhere(frame == color)
    return tuple(
        sorted((int(r) - r0, int(c) - c0) for r, c in coords)
    )


def _shift_sprite(frame: np.ndarray, color: int, dr: int, dc: int, bg: int) -> np.ndarray:
    out = frame.copy()
    out[out == color] = bg
    h, w = frame.shape
    for r, c in np.argwhere(frame == color):
        nr = int(r) + int(dr)
        nc = int(c) + int(dc)
        assert 0 <= nr < h and 0 <= nc < w, "predicted sprite must stay in-bounds"
        assert out[nr, nc] == bg, "sprite must move into empty cells"
        out[nr, nc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each stretched panel, extrapolate the moving sprite's constant motion, and return the compressed fifth frame.

    Input structure
    - The input is a vertical strip of four equal-height panels separated by full-width solid rows.
    - Each panel has a black background, a fixed set of grey guide marks, and one small colored 4-connected sprite.
    - Within each panel, some adjacent rows are exact duplicates and some adjacent columns are exact duplicates because that frame was independently stretched.

    Rule
    - Split the strip into its four panels using the separator rows.
    - For each panel, keep the first row from every consecutive run of identical rows and the first column from every consecutive run of identical columns; this recovers the compressed frame.
    - The grey guide marks stay fixed, so isolate the non-grey moving sprite in the compressed frames, measure its bbox top-left shift between consecutive frames, and continue that constant offset once more.
    - Output only the fifth compressed frame, keeping the grey guides unchanged and without re-stretching it.

    Key insight
    - The visible stretched panels do not show a consistent raw offset; only after run-deduplicating each panel does the moving sprite reveal a stable translation against the fixed guide pattern.
    """
    panels, bg = _split_panels(np.asarray(grid, dtype=int))
    compressed = [_compress_panel(panel) for panel in panels]
    assert len({frame.shape for frame in compressed}) == 1, "compressed panels must match in size"

    colors = set(map(int, np.unique(compressed[0]))) - {bg, SEPARATOR}
    assert len(colors) == 1, "expected exactly one moving sprite color"
    sprite_color = next(iter(colors))
    for frame in compressed[1:]:
        assert set(map(int, np.unique(frame))) - {bg, SEPARATOR} == {sprite_color}

    signatures = [_sprite_signature(frame, sprite_color) for frame in compressed]
    assert len(set(signatures)) == 1, "compressed frames must contain the same translated sprite"

    tops = [_top_left_mask(frame == sprite_color) for frame in compressed]
    deltas = [
        (tops[i + 1][0] - tops[i][0], tops[i + 1][1] - tops[i][1])
        for i in range(3)
    ]
    assert len(set(deltas)) == 1, "compressed sprite motion must be a constant offset"
    dr, dc = deltas[0]
    assert (dr, dc) != (0, 0), "sprite must actually move"
    return _shift_sprite(compressed[-1], sprite_color, dr, dc, bg)


def _paint_shape(shape: tuple[int, int], coords: tuple[tuple[int, int], ...], color: int, r0: int, c0: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for dr, dc in coords:
        out[r0 + dr, c0 + dc] = int(color)
    return out


def _connected_shape(
    rng: np.random.Generator,
    canvas_h: int,
    canvas_w: int,
    cells: int,
) -> tuple[tuple[int, int], ...]:
    coords = {(0, 0)}
    while len(coords) < cells:
        base = list(coords)[int(rng.integers(len(coords)))]
        r, c = base
        neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        rng.shuffle(neighbors)
        for nr, nc in neighbors:
            if 0 <= nr < canvas_h and 0 <= nc < canvas_w:
                coords.add((nr, nc))
                break

    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    norm = sorted((r - r0, c - c0) for r, c in coords)
    return tuple((int(r), int(c)) for r, c in norm)


def _shape_bbox(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _start_bounds(limit: int, size: int, delta: int) -> tuple[int, int] | None:
    offsets = [t * delta for t in range(5)]
    low = max(0, max(-off for off in offsets))
    high = min(limit - size, min(limit - size - off for off in offsets))
    if low > high:
        return None
    return int(low), int(high)


def _bounded_composition(
    total: int,
    parts: int,
    max_part: int,
    rng: np.random.Generator,
) -> tuple[int, ...] | None:
    if total < parts or total > parts * max_part:
        return None
    values = [1] * parts
    remaining = total - parts
    while remaining > 0:
        choices = [i for i, v in enumerate(values) if v < max_part]
        if not choices:
            return None
        idx = int(choices[int(rng.integers(len(choices)))])
        values[idx] += 1
        remaining -= 1
    if max(values) == 1:
        return None
    return tuple(values)


def _expand_panel(panel: np.ndarray, row_runs: tuple[int, ...], col_runs: tuple[int, ...]) -> np.ndarray:
    return np.repeat(np.repeat(panel, row_runs, axis=0), col_runs, axis=1)


def _guide_panel(h: int, w: int) -> np.ndarray:
    panel = np.zeros((h, w), dtype=int)
    panel[0, ::2] = SEPARATOR
    panel[h - 1, 1::2] = SEPARATOR
    if h >= 3:
        panel[1, 0] = SEPARATOR
        panel[h - 2, w - 1] = SEPARATOR
    return panel


def _colored_coords(panel: np.ndarray, bg: int) -> set[tuple[int, int, int]]:
    return {
        (int(r), int(c), int(panel[r, c]))
        for r, c in np.argwhere(panel != bg)
    }


def _translation_delta(a: np.ndarray, b: np.ndarray, bg: int) -> tuple[int, int] | None:
    coords_a = _colored_coords(a, bg)
    coords_b = _colored_coords(b, bg)
    if len(coords_a) != len(coords_b) or not coords_a:
        return None
    a_top = _top_left(a, bg)
    b_top = _top_left(b, bg)
    dr = b_top[0] - a_top[0]
    dc = b_top[1] - a_top[1]
    shifted = {(r + dr, c + dc, color) for r, c, color in coords_a}
    if shifted == coords_b:
        return int(dr), int(dc)
    return None


def _raw_motion_is_too_easy(panels: list[np.ndarray], bg: int, color: int) -> bool:
    sprite_only = []
    for panel in panels:
        frame = np.where(panel == color, color, bg)
        sprite_only.append(frame)
    deltas = [_translation_delta(sprite_only[i], sprite_only[i + 1], bg) for i in range(3)]
    return deltas[0] is not None and deltas[0] == deltas[1] == deltas[2]


def _build_compressed_frames(
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], int, int, int, int, int] | None:
    color = int(rng.choice(SPRITE_COLORS))
    hc = int(rng.integers(4, 6))
    shape_h = hc - 2
    shape_w = int(rng.integers(2, 5))
    wc = int(rng.integers(shape_w + 6, shape_w + 9))
    max_cells = min(6, shape_h * shape_w)
    cells = int(rng.integers(max(3, shape_w), max_cells + 1))

    for _ in range(200):
        coords = _connected_shape(rng, shape_h, shape_w, cells)
        sh, sw = _shape_bbox(coords)
        if sh != shape_h or sw != shape_w:
            continue
        sprite_mask = _paint_shape((shape_h, shape_w), coords, 1, 0, 0)
        if any(np.all(sprite_mask[r, :] == 0) for r in range(shape_h)):
            continue
        if any(np.all(sprite_mask[:, c] == 0) for c in range(shape_w)):
            continue
        break
    else:
        return None

    dr = 0
    dc = -1 if bool(rng.integers(2)) else 1
    r0 = 1
    cb = _start_bounds(wc, sw, dc)
    if cb is None:
        return None
    base = _guide_panel(hc, wc)
    candidates = []
    for c0 in range(cb[0], cb[1] + 1):
        ok = True
        for t in range(5):
            sprite_frame = _paint_shape((hc, wc), coords, color, r0, c0 + t * dc)
            if np.any((sprite_frame == color) & (base != 0)):
                ok = False
                break
        if ok:
            candidates.append(c0)
    if not candidates:
        return None
    c0 = int(candidates[int(rng.integers(len(candidates)))])

    frames = []
    for t in range(5):
        frame = base.copy()
        rr = r0 + t * dr
        cc = c0 + t * dc
        sprite_frame = _paint_shape((hc, wc), coords, color, rr, cc)
        if np.any((sprite_frame == color) & (base != 0)):
            return None
        frame[sprite_frame == color] = color
        frames.append(frame)
    return frames, dr, dc, hc, wc, color


def _sample_run_vectors(
    rng: np.random.Generator,
    parts: int,
    total: int,
    max_part: int,
    count: int,
) -> list[tuple[int, ...]] | None:
    vectors: list[tuple[int, ...]] = []
    seen = set()
    for _ in range(200):
        if len(vectors) == count:
            return vectors
        vec = _bounded_composition(total, parts, max_part, rng)
        if vec is None or vec in seen:
            continue
        seen.add(vec)
        vectors.append(vec)
    return None


def _build_observed_panels(
    rng: np.random.Generator,
    compressed_frames: list[np.ndarray],
    dr: int,
    dc: int,
    hc: int,
    wc: int,
    color: int,
) -> tuple[list[np.ndarray], list[tuple[int, ...]], list[tuple[int, ...]]] | None:
    hs = int(rng.integers(hc + 1, 7))
    ws = int(rng.integers(wc + 1, min(15, wc + 5)))

    row_runs = _sample_run_vectors(rng, hc, hs, 3, 4)
    col_runs = _sample_run_vectors(rng, wc, ws, 3, 4)
    if row_runs is None or col_runs is None:
        return None

    for _ in range(200):
        pair_indices = rng.permutation(4)
        panels = [
            _expand_panel(compressed_frames[i], row_runs[i], col_runs[pair_indices[i]])
            for i in range(4)
        ]
        if len({panel.shape for panel in panels}) != 1:
            continue
        raw_tops = [_top_left_mask(panel == color) for panel in panels]
        raw_steps = [
            (raw_tops[i + 1][0] - raw_tops[i][0], raw_tops[i + 1][1] - raw_tops[i][1])
            for i in range(3)
        ]
        if dr != 0 and len({step[0] for step in raw_steps}) == 1:
            continue
        if dc != 0 and len({step[1] for step in raw_steps}) == 1:
            continue
        if _raw_motion_is_too_easy(panels, BACKGROUND, color):
            continue
        return panels, row_runs, [col_runs[idx] for idx in pair_indices]
    return None


def _stack_panels(panels: list[np.ndarray]) -> np.ndarray:
    sep = np.full((1, panels[0].shape[1]), SEPARATOR, dtype=int)
    rows = []
    for panel in panels[:-1]:
        rows.append(panel)
        rows.append(sep)
    rows.append(panels[-1])
    return np.vstack(rows)


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    dr: int,
    dc: int,
    color: int,
) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False

    panels, bg = _split_panels(inp)
    compressed = [_compress_panel(panel) for panel in panels]
    if len({frame.shape for frame in compressed}) != 1:
        return False
    if not np.array_equal(_shift_sprite(compressed[-1], color, dr, dc, bg), expected):
        return False
    if _raw_motion_is_too_easy(panels, bg, color):
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    built = _build_compressed_frames(rng)
    if built is None:
        return None
    compressed_frames, dr, dc, hc, wc, color = built

    observed = _build_observed_panels(rng, compressed_frames[:4], dr, dc, hc, wc, color)
    if observed is None:
        return None
    panels, _row_runs, _col_runs = observed

    inp = _stack_panels(panels)
    expected = compressed_frames[4]
    if inp.shape[0] > 30 or inp.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not _semantic_check(inp, expected, dr, dc, color):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stretched-panel sequence")
