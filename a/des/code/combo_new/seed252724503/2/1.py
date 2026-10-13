import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
ORIENTATIONS = ("bottom", "right")
MAX_GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _is_alternating(segment: np.ndarray) -> bool:
    vals = [int(v) for v in segment.tolist()]
    if len(vals) < 2 or vals[0] == vals[1]:
        return False
    for idx, val in enumerate(vals):
        if val != vals[idx % 2]:
            return False
    return True


def _parse_segmented_legend(key_line: np.ndarray, texture_line: np.ndarray) -> tuple[dict[int, tuple[int, ...]], int] | None:
    length = int(len(key_line))
    for seg_len in range(2, length + 1):
        if length % seg_len != 0 or length // seg_len < 2:
            continue
        mapping: dict[int, tuple[int, ...]] = {}
        ok = True
        for start in range(0, length, seg_len):
            key_seg = key_line[start : start + seg_len]
            tex_seg = texture_line[start : start + seg_len]
            if len(set(int(v) for v in key_seg.tolist())) != 1:
                ok = False
                break
            key = int(key_seg[0])
            if key == BG or key in mapping or not _is_alternating(tex_seg):
                ok = False
                break
            mapping[key] = tuple(int(v) for v in tex_seg.tolist())
        if ok:
            return mapping, seg_len
    return None


def _detect_legend(grid: np.ndarray) -> tuple[str, dict[int, tuple[int, ...]]]:
    h, w = grid.shape
    candidates: list[tuple[str, dict[int, tuple[int, ...]]]] = []

    if h >= 4:
        bottom_mask = (grid[-2] != BG) | (grid[-1] != BG)
        for c0, c1 in _find_runs(bottom_mask):
            parsed = _parse_segmented_legend(grid[-2, c0 : c1 + 1], grid[-1, c0 : c1 + 1])
            if parsed is not None:
                candidates.append(("bottom", parsed[0]))

    if w >= 4:
        right_mask = (grid[:, -2] != BG) | (grid[:, -1] != BG)
        for r0, r1 in _find_runs(right_mask):
            parsed = _parse_segmented_legend(grid[r0 : r1 + 1, -2], grid[r0 : r1 + 1, -1])
            if parsed is not None:
                candidates.append(("right", parsed[0]))

    assert len(candidates) == 1, "expected exactly one edge legend"
    return candidates[0]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _expected_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_frames(core: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color in (int(v) for v in np.unique(core) if int(v) != BG):
        for comp in _components4(core == color):
            if len(comp) <= 1:
                continue

            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            border_cells = set(comp)
            if border_cells != _expected_border_cells(r0, c0, r1, c1):
                continue

            interior = core[r0 + 1 : r1, c0 + 1 : c1]
            marker_pts = np.argwhere(interior != BG)
            if marker_pts.shape[0] != 1:
                continue
            mr, mc = marker_pts[0]
            marker = int(interior[int(mr), int(mc)])
            frames.append((r0, c0, r1, c1, marker))

    assert frames, "expected at least one rectangular frame"
    return frames


def _tile_pattern(pattern: tuple[int, ...], height: int, width: int) -> np.ndarray:
    pattern_arr = np.array(pattern, dtype=int)
    idx = (np.arange(height)[:, None] + np.arange(width)[None, :]) % len(pattern_arr)
    return pattern_arr[idx]


def _fill_frame_interiors(core: np.ndarray, frames: list[tuple[int, int, int, int, int]], mapping: dict[int, tuple[int, ...]]) -> np.ndarray:
    out = core.copy()
    for r0, c0, r1, c1, marker in frames:
        assert marker in mapping, "frame marker must appear in the legend"
        fill = _tile_pattern(mapping[marker], r1 - r0 - 1, c1 - c0 - 1)
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed frame with the edge-legend texture selected by its interior marker color.

    Input structure:
    - Black background surrounds several one-pixel-thick rectangular frames.
    - Every frame encloses background cells and exactly one single-pixel marker whose color is unique among frames.
    - A two-cell-thick legend strip touches the bottom edge or the right edge; each equal-sized segment has a key-color line and an alternating two-color texture line.

    Rule:
    - Detect the bottom or right legend and read it as a marker-color -> texture-strip mapping.
    - Remove the attached legend strip from the output.
    - Keep every rectangular border unchanged, then replace each frame interior, including the marker pixel, by tiling the matched texture from the interior top-left corner.

    Key insight:
    - The singleton marker inside each frame is not part of the picture to preserve; it is a lookup key that chooses which alternating edge texture must be tiled into that enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    orientation, mapping = _detect_legend(g)
    core = g[:-2].copy() if orientation == "bottom" else g[:, :-2].copy()
    frames = _find_frames(core)
    return _fill_frame_interiors(core, frames, mapping)


def _choose_palette(rng: random.Random, n_keys: int) -> tuple[int, list[int], list[int]]:
    colors = list(COLORS)
    rng.shuffle(colors)
    border = colors[0]
    keys = colors[1 : 1 + n_keys]
    texture_pool = colors[1 + n_keys :]
    return border, keys, texture_pool


def _make_mapping(rng: random.Random, keys: list[int], seg_len: int, texture_pool: list[int]) -> dict[int, tuple[int, ...]]:
    mapping: dict[int, tuple[int, ...]] = {}
    for key in keys:
        a, b = rng.sample(texture_pool, 2)
        phase = rng.randint(0, 1)
        mapping[key] = tuple(a if (idx + phase) % 2 == 0 else b for idx in range(seg_len))
    return mapping


def _try_place_frames(
    rng: random.Random,
    body_h: int,
    body_w: int,
    border: int,
    keys: list[int],
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]] | None:
    if body_h < 8 or body_w < 8:
        return None

    core = np.full((body_h, body_w), BG, dtype=int)
    occupied = np.zeros((body_h, body_w), dtype=bool)
    frames: list[tuple[int, int, int, int, int]] = []
    key_order = keys[:]
    rng.shuffle(key_order)

    max_h = min(8, body_h - 2)
    max_w = min(8, body_w - 2)
    if max_h < 4 or max_w < 4:
        return None

    for key in key_order:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            outer_h = rng.randint(4, max_h)
            outer_w = rng.randint(4, max_w)
            max_r0 = body_h - outer_h - 1
            max_c0 = body_w - outer_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue
            r0 = rng.randint(1, max_r0)
            c0 = rng.randint(1, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue

            core[r0, c0 : c1 + 1] = border
            core[r1, c0 : c1 + 1] = border
            core[r0 : r1 + 1, c0] = border
            core[r0 : r1 + 1, c1] = border

            mr = rng.randint(r0 + 1, r1 - 1)
            mc = rng.randint(c0 + 1, c1 - 1)
            core[mr, mc] = key

            occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            frames.append((r0, c0, r1, c1, key))
            placed = True
            break

        if not placed:
            return None

    return core, frames


def _render_with_bottom_legend(
    rng: random.Random,
    core: np.ndarray,
    keys: list[int],
    mapping: dict[int, tuple[int, ...]],
    seg_len: int,
) -> np.ndarray:
    body_h, body_w = core.shape
    legend_w = len(keys) * seg_len
    col0 = rng.randint(0, body_w - legend_w)
    order = keys[:]
    rng.shuffle(order)

    grid = np.full((body_h + 2, body_w), BG, dtype=int)
    grid[:body_h, :] = core
    for idx, key in enumerate(order):
        start = col0 + idx * seg_len
        end = start + seg_len
        grid[-2, start:end] = key
        grid[-1, start:end] = np.array(mapping[key], dtype=int)
    return grid


def _render_with_right_legend(
    rng: random.Random,
    core: np.ndarray,
    keys: list[int],
    mapping: dict[int, tuple[int, ...]],
    seg_len: int,
) -> np.ndarray:
    body_h, body_w = core.shape
    legend_h = len(keys) * seg_len
    row0 = rng.randint(0, body_h - legend_h)
    order = keys[:]
    rng.shuffle(order)

    grid = np.full((body_h, body_w + 2), BG, dtype=int)
    grid[:, :body_w] = core
    for idx, key in enumerate(order):
        start = row0 + idx * seg_len
        end = start + seg_len
        grid[start:end, -2] = key
        grid[start:end, -1] = np.array(mapping[key], dtype=int)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    orientation = rng.choice(ORIENTATIONS)
    n_keys = rng.randint(2, 4)
    seg_len = rng.randint(2, 5)

    if orientation == "bottom":
        body_h = rng.randint(10, 24)
        body_w = rng.randint(max(12, n_keys * seg_len), 28)
    else:
        body_h = rng.randint(max(12, n_keys * seg_len), 28)
        body_w = rng.randint(10, 24)

    border, keys, texture_pool = _choose_palette(rng, n_keys)
    mapping = _make_mapping(rng, keys, seg_len, texture_pool)
    placed = _try_place_frames(rng, body_h, body_w, border, keys)
    if placed is None:
        return None
    core, frames = placed

    if orientation == "bottom":
        grid = _render_with_bottom_legend(rng, core, keys, mapping, seg_len)
    else:
        grid = _render_with_right_legend(rng, core, keys, mapping, seg_len)

    expected = _fill_frame_interiors(core, frames, mapping)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid ARC sample")
