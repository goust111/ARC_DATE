import random
from collections import Counter

import numpy as np


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rect_frame(coords: np.ndarray) -> bool:
    if coords.shape[0] < 8:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if coords.shape[0] != expected:
        return False
    coord_set = {(int(r), int(c)) for r, c in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_perimeter = r in (r0, r1) or c in (c0, c1)
            if on_perimeter != ((r, c) in coord_set):
                return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        for comp in _components(grid == color):
            if not _is_rect_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "interior": grid[r0 + 1 : r1, c0 + 1 : c1].copy(),
                }
            )
    assert len(frames) >= 2, "expected at least two frames"
    colors = [frame["color"] for frame in frames]
    assert len(colors) == len(set(colors)), "frame colors must be unique"
    inner_shapes = {frame["interior"].shape for frame in frames}
    assert len(inner_shapes) == 1, "all frames must share one interior size"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    frames = _find_frames(g, bg)
    frame_colors = {frame["color"] for frame in frames}
    color_to_idx = {frame["color"]: idx for idx, frame in enumerate(frames)}

    successors: dict[int, int] = {}
    indegree = [0] * len(frames)
    patterns: list[np.ndarray] = []

    for idx, frame in enumerate(frames):
        interior = frame["interior"]
        pointer_cells = np.argwhere(np.isin(interior, list(frame_colors - {frame["color"]})))
        assert pointer_cells.shape[0] <= 1, "each frame may contain at most one pointer"

        pattern = interior.copy()
        if pointer_cells.shape[0] == 1:
            pr, pc = map(int, pointer_cells[0])
            target_color = int(interior[pr, pc])
            assert target_color in color_to_idx, "pointer must target a frame color"
            target_idx = color_to_idx[target_color]
            assert idx not in successors, "frame must have at most one outgoing pointer"
            successors[idx] = target_idx
            indegree[target_idx] += 1
            pattern[pr, pc] = bg
        patterns.append(pattern)

    sources = [idx for idx, deg in enumerate(indegree) if deg == 0]
    sinks = [idx for idx in range(len(frames)) if idx not in successors]
    assert len(sources) == 1, "expected exactly one source frame"
    assert len(sinks) == 1, "expected exactly one sink frame"

    order = []
    seen = set()
    cur = sources[0]
    while True:
        assert cur not in seen, "pointer chain must be acyclic"
        seen.add(cur)
        order.append(cur)
        if cur not in successors:
            break
        cur = successors[cur]
    assert len(order) == len(frames), "pointer chain must visit every frame"
    assert all(deg <= 1 for deg in indegree), "chain must have indegree at most one"

    out = g.copy()
    cumulative = patterns[order[0]].copy()
    for pos, idx in enumerate(order):
        frame = frames[idx]
        r0, c0, r1, c1 = frame["bbox"]
        interior = frame["interior"].copy()
        if pos > 0:
            fill_mask = (interior == bg) & (cumulative != bg)
            interior[fill_mask] = cumulative[fill_mask]
        out[r0 + 1 : r1, c0 + 1 : c1] = interior

        current_pattern = patterns[idx].copy()
        if pos > 0:
            carry_mask = (current_pattern == bg) & (cumulative != bg)
            current_pattern[carry_mask] = cumulative[carry_mask]
        cumulative = current_pattern

    return out


_BG = 0
_FRAME_COUNT_RANGE = (3, 5)
_INNER_H_RANGE = (4, 6)
_INNER_W_RANGE = (4, 6)
_GRID_H_RANGE = (22, 30)
_GRID_W_RANGE = (22, 30)
_MIN_FRAME_GAP = 2
_MAX_GEN_ATTEMPTS = 400


def _make_pattern(inner_h: int, inner_w: int, palette: list[int], blocked: tuple[int, int] | None) -> np.ndarray:
    pattern = np.zeros((inner_h, inner_w), dtype=int)
    cell_count = random.randint(max(2, (inner_h * inner_w) // 6), max(3, (inner_h * inner_w) // 3))
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) != blocked]
    random.shuffle(cells)
    chosen = cells[:cell_count]
    color_count = random.randint(1, min(3, len(palette)))
    colors = random.sample(palette, color_count)
    for r, c in chosen:
        pattern[r, c] = random.choice(colors)
    return pattern


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    return not occ[rr0:rr1, cc0:cc1].any()


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*_GRID_H_RANGE)
    grid_w = random.randint(*_GRID_W_RANGE)
    inner_h = random.randint(*_INNER_H_RANGE)
    inner_w = random.randint(*_INNER_W_RANGE)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    max_rows = max(1, grid_h // (outer_h + _MIN_FRAME_GAP))
    max_cols = max(1, grid_w // (outer_w + _MIN_FRAME_GAP))
    max_frames = min(_FRAME_COUNT_RANGE[1], max_rows * max_cols + 1)
    if max_frames < _FRAME_COUNT_RANGE[0]:
        return None
    frame_count = random.randint(_FRAME_COUNT_RANGE[0], max_frames)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_colors = palette[:frame_count]
    pattern_palette = palette[frame_count:]
    if len(pattern_palette) < 3:
        return None

    order = list(range(frame_count))
    random.shuffle(order)
    successor_of = {order[i]: order[i + 1] for i in range(frame_count - 1)}

    positions = []
    occ = np.zeros((grid_h, grid_w), dtype=bool)
    for _idx in range(frame_count):
        placed = False
        for _ in range(200):
            r0 = random.randint(_MIN_FRAME_GAP, grid_h - outer_h - _MIN_FRAME_GAP)
            c0 = random.randint(_MIN_FRAME_GAP, grid_w - outer_w - _MIN_FRAME_GAP)
            if not _can_place(occ, r0, c0, outer_h, outer_w, _MIN_FRAME_GAP):
                continue
            occ[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
            positions.append((r0, c0))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    for idx, (r0, c0) in enumerate(positions):
        color = frame_colors[idx]
        grid[r0, c0 : c0 + outer_w] = color
        grid[r0 + outer_h - 1, c0 : c0 + outer_w] = color
        grid[r0 : r0 + outer_h, c0] = color
        grid[r0 : r0 + outer_h, c0 + outer_w - 1] = color

    for idx, (r0, c0) in enumerate(positions):
        target = successor_of.get(idx)
        pointer_pos = None
        if target is not None:
            pointer_pos = (random.randrange(inner_h), random.randrange(inner_w))
        pattern = _make_pattern(inner_h, inner_w, pattern_palette, pointer_pos)
        if np.count_nonzero(pattern) == 0:
            return None
        interior = pattern.copy()
        if target is not None:
            pr, pc = pointer_pos
            interior[pr, pc] = frame_colors[target]
        if np.count_nonzero(interior == _BG) < 2:
            return None
        grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = interior

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
