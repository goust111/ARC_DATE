from __future__ import annotations

import numpy as np


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RNG = np.random.default_rng()
_BG = 0
_FG_COLORS = tuple(range(1, 10))
_MAX_GENERATION_ATTEMPTS = 300


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _DIRS]


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = grid != _BG
    pts = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert pts, "expected a non-empty foreground path"

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for cell in pts:
        deg[cell] = sum(nbr in pset for nbr in _neighbors(cell))

    assert all(d in (1, 2) for d in deg.values()), "path must be a simple 4-connected chain"
    ends = [cell for cell, d in deg.items() if d == 1]
    assert len(ends) == 2, "path must have exactly two endpoints"

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = min(ends)
    while True:
        order.append(cur)
        nbrs = [nbr for nbr in _neighbors(cur) if nbr in pset and nbr != prev]
        if not nbrs:
            break
        assert len(nbrs) == 1, "path traversal must stay unambiguous"
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts), "foreground must form one connected path"
    return order


def _split_segments(
    order: list[tuple[int, int]], colors: list[int]
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(order) + 1):
        if idx == len(order) or colors[idx] != colors[start]:
            segments.append((start, idx))
            start = idx
    return segments


def _rotate_cw(rel: tuple[int, int]) -> tuple[int, int]:
    dr, dc = rel
    return dc, -dr


def _transform_path(
    order: list[tuple[int, int]], colors: list[int]
) -> list[tuple[tuple[int, int], int]]:
    segments = _split_segments(order, colors)
    anchor = order[0]
    transformed: list[tuple[tuple[int, int], int]] = []

    for seg_idx, (start, stop) in enumerate(segments):
        base = order[start]
        rel_cells = [(r - base[0], c - base[1]) for r, c in order[start:stop]]
        rotated_cells = [_rotate_cw(rel) for rel in rel_cells]
        placed = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in rotated_cells]
        color = int(colors[start])
        transformed.extend((cell, color) for cell in placed)

        if seg_idx + 1 < len(segments):
            step = (
                order[stop][0] - order[stop - 1][0],
                order[stop][1] - order[stop - 1][1],
            )
            anchor = (placed[-1][0] + step[0], placed[-1][1] + step[1])

    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a multicolor snake into same-color segments and rotate each segment 90 degrees clockwise.

    The path is traced from the top-left endpoint. Each maximal same-color segment is rotated
    around its own first pixel, while later segments are translated so that the color-change
    links stay attached to the previous rotated segment. Pixels outside the original canvas are
    clipped.
    """
    g = np.asarray(grid, dtype=int)
    order = _trace_path(g)
    colors = [int(g[r, c]) for r, c in order]

    out = np.full_like(g, _BG)
    for (r, c), color in _transform_path(order, colors):
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            out[r, c] = color
    return out


def _random_partition(total: int, parts: int, min_len: int, max_len: int) -> list[int] | None:
    if parts * min_len > total or parts * max_len < total:
        return None
    vals = [min_len] * parts
    remaining = total - parts * min_len
    while remaining:
        choices = [idx for idx, val in enumerate(vals) if val < max_len]
        if not choices:
            return None
        vals[int(_RNG.choice(choices))] += 1
        remaining -= 1
    _RNG.shuffle(vals)
    return [int(v) for v in vals]


def _valid_extension(
    current: tuple[int, int],
    candidate: tuple[int, int],
    visited: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    r, c = candidate
    if not (0 <= r < height and 0 <= c < width):
        return False
    if candidate in visited:
        return False

    touched = [nbr for nbr in _neighbors(candidate) if nbr in visited]
    return len(touched) == 1 and touched[0] == current


def _build_path(height: int, width: int, target_len: int) -> list[tuple[int, int]] | None:
    for _ in range(20):
        start = (int(_RNG.integers(height)), int(_RNG.integers(width)))
        path = [start]
        visited = {start}

        def rec() -> bool:
            if len(path) == target_len:
                return True

            cur = path[-1]
            prev = path[-2] if len(path) >= 2 else None
            candidates = []
            for dr, dc in _DIRS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if _valid_extension(cur, nxt, visited, height, width):
                    candidates.append(nxt)

            _RNG.shuffle(candidates)
            if prev is not None:
                straight = (cur[0] - prev[0], cur[1] - prev[1])
                candidates.sort(
                    key=lambda nxt: (
                        (nxt[0] - cur[0], nxt[1] - cur[1]) != straight,
                        float(_RNG.random()),
                    )
                )

            for nxt in candidates:
                path.append(nxt)
                visited.add(nxt)
                if rec():
                    return True
                visited.remove(nxt)
                path.pop()
            return False

        if rec():
            if path[-1] < path[0]:
                path.reverse()
            return path

    return None


def _sample_segment_colors(segment_count: int) -> list[int]:
    palette_size = int(_RNG.integers(2, min(5, segment_count) + 1))
    palette = list(map(int, _RNG.choice(_FG_COLORS, size=palette_size, replace=False)))
    unused = palette.copy()
    _RNG.shuffle(unused)

    colors: list[int] = []
    for _ in range(segment_count):
        source = unused if unused else palette
        prev = colors[-1] if colors else None
        choices = [color for color in source if color != prev]
        if not choices:
            choices = [color for color in palette if color != prev]
        color = int(_RNG.choice(choices))
        colors.append(color)
        if color in unused:
            unused.remove(color)

    assert len(set(colors)) >= 2
    return colors


def _gen_attempt(require_clip: bool) -> np.ndarray | None:
    height = int(_RNG.integers(10, 19))
    width = int(_RNG.integers(10, 19))
    max_len = min(32, (height * width) // 3)
    if max_len < 8:
        return None
    target_len = int(_RNG.integers(8, max_len + 1))

    path = _build_path(height, width, target_len)
    if path is None:
        return None

    segment_count = int(_RNG.integers(2, min(6, target_len) + 1))
    segment_lengths = _random_partition(
        total=target_len,
        parts=segment_count,
        min_len=1,
        max_len=max(3, min(9, target_len - segment_count + 1)),
    )
    if segment_lengths is None or max(segment_lengths) < 2:
        return None

    segment_colors = _sample_segment_colors(segment_count)
    colors = [color for seg_len, color in zip(segment_lengths, segment_colors) for _ in range(seg_len)]

    transformed = _transform_path(path, colors)
    in_bounds = [(r, c) for (r, c), _ in transformed if 0 <= r < height and 0 <= c < width]
    clipped = len(in_bounds) != len(transformed)
    if clipped != require_clip:
        return None

    unique_visible = len(set(in_bounds))
    if unique_visible < max(6, target_len // 2):
        return None

    grid = np.zeros((height, width), dtype=int)
    for (r, c), color in zip(path, colors):
        grid[r, c] = int(color)

    # Keep the generator honest by verifying the structural assumptions and its own branch target.
    order = _trace_path(grid)
    if order != path:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    """Generate a random multicolor snake path with balanced clipped/non-clipped outputs."""
    require_clip = bool(int(_RNG.integers(2)))
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(require_clip=require_clip)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the new ARC task")
