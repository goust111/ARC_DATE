import random
from itertools import permutations

import numpy as np


BG = 0
GEN_ATTEMPTS = 500
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _spiral_path(height: int, width: int) -> list[tuple[int, int]]:
    """Generate a clockwise spiral by walking forward until blocked, then turning right."""
    path: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))

    def can_step(rr: int, cc: int, prev: tuple[int, int]) -> bool:
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        if (rr, cc) in occupied:
            return False
        for dr, dc in dirs:
            nbr = (rr + dr, cc + dc)
            if nbr in occupied and nbr != prev:
                return False
        return True

    r, c = 0, 0
    dir_idx = 0
    while True:
        cur = (r, c)
        occupied.add(cur)
        path.append(cur)

        dr, dc = dirs[dir_idx]
        if can_step(r + dr, c + dc, cur):
            r += dr
            c += dc
            continue

        next_dir = (dir_idx + 1) % 4
        dr, dc = dirs[next_dir]
        if can_step(r + dr, c + dc, cur):
            dir_idx = next_dir
            r += dr
            c += dc
            continue

        break

    return path


def _is_boundary_cell(cell: tuple[int, int], height: int, width: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == height - 1 or c == width - 1


def _follow_path(mask: np.ndarray) -> list[tuple[int, int]]:
    """Recover the outer-to-inner order of the spiral path from its 4-connected mask."""
    height, width = mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    coord_set = set(coords)
    assert coords, "expected a non-empty path"

    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        cur_nbrs: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if (rr, cc) in coord_set:
                cur_nbrs.append((rr, cc))
        nbrs[(r, c)] = cur_nbrs

    endpoints = [cell for cell, ns in nbrs.items() if len(ns) == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    start = min(
        endpoints,
        key=lambda cell: (
            0 if _is_boundary_cell(cell, height, width) else 1,
            cell[0],
            cell[1],
        ),
    )

    order = [start]
    visited = {start}
    prev = None
    cur = start
    while True:
        next_cells = [cell for cell in nbrs[cur] if cell != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path must not branch"
        nxt = next_cells[0]
        assert nxt not in visited, "path must not self-intersect"
        order.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    assert len(order) == len(coords), "path traversal missed cells"
    return order


def _build_sequence(length: int, colors: tuple[int, int, int], seg_lengths: tuple[int, int, int]) -> np.ndarray:
    seq: list[int] = []
    idx = 0
    while len(seq) < length:
        run_len = seg_lengths[idx % 3]
        seq.extend([int(colors[idx % 3])] * run_len)
        idx += 1
    return np.array(seq[:length], dtype=int)


def _segment_start_mask(seq: np.ndarray) -> np.ndarray:
    starts = np.zeros(len(seq), dtype=bool)
    if len(seq):
        starts[0] = True
        starts[1:] = seq[1:] != seq[:-1]
    return starts


def _infer_base_sequence(obs_seq: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int], int]:
    colors = tuple(sorted({int(x) for x in obs_seq}))
    assert len(colors) == 3, "expected exactly three path colors"
    first_color = int(obs_seq[0])
    assert first_color in colors

    n = len(obs_seq)
    max_seg_len = min(8, max(4, n // 3))
    length_choices = list(range(3, max_seg_len + 1))
    assert len(length_choices) >= 3

    best = None
    tie_count = 0

    for color_order in permutations(colors):
        if color_order[0] != first_color:
            continue
        for seg_lengths in permutations(length_choices, 3):
            cand = _build_sequence(n, color_order, seg_lengths)
            mismatch_idx = np.flatnonzero(cand != obs_seq)
            start_mask = _segment_start_mask(cand)
            feasible = True
            for idx in mismatch_idx:
                marker_color = int(obs_seq[idx])
                if not any(int(pos) > int(idx) and start_mask[pos] and int(cand[pos]) == marker_color for pos in range(idx + 1, n)):
                    feasible = False
                    break
            if not feasible:
                continue
            score = (
                int(mismatch_idx.size),
                int(start_mask[mismatch_idx].sum()) if mismatch_idx.size else 0,
                int(np.sum(np.diff(mismatch_idx) == 1)) if mismatch_idx.size > 1 else 0,
            )

            if best is None or score < best["score"]:
                best = {
                    "seq": cand,
                    "colors": tuple(map(int, color_order)),
                    "lengths": tuple(map(int, seg_lengths)),
                    "score": score,
                }
                tie_count = 1
            elif score == best["score"]:
                tie_count += 1
                if tuple(map(int, color_order + seg_lengths)) < tuple(map(int, best["colors"] + best["lengths"])):
                    best = {
                        "seq": cand,
                        "colors": tuple(map(int, color_order)),
                        "lengths": tuple(map(int, seg_lengths)),
                        "score": score,
                    }

    assert best is not None, "failed to infer the spiral color pattern"
    return best["seq"], best["colors"], best["lengths"], tie_count


def _render_path(shape: tuple[int, int], path: list[tuple[int, int]], seq: np.ndarray) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for (r, c), color in zip(path, seq):
        out[r, c] = int(color)
    return out


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    path_mask = g != BG
    path = _follow_path(path_mask)
    obs_seq = np.array([int(g[r, c]) for r, c in path], dtype=int)

    base_seq, colors, seg_lengths, tie_count = _infer_base_sequence(obs_seq)
    out_seq = base_seq.copy()

    start_mask = _segment_start_mask(base_seq)
    marker_idx = np.flatnonzero(obs_seq != base_seq)
    for idx in marker_idx:
        marker_color = int(obs_seq[idx])
        target = None
        for pos in np.flatnonzero(start_mask):
            if int(pos) > int(idx) and int(base_seq[pos]) == marker_color:
                target = int(pos)
                break
        assert target is not None, "each marker must have a future same-color segment"
        out_seq[target] = marker_color

    out = _render_path(g.shape, path, out_seq)
    meta = {
        "path": path,
        "observed": obs_seq,
        "base": base_seq,
        "colors": colors,
        "lengths": seg_lengths,
        "markers": marker_idx,
        "tie_count": tie_count,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a three-color spiral path whose segment pattern is disturbed by single-pixel markers.

    Input structure:
    - Black is background.
    - A one-pixel-wide clockwise spiral path runs from the outer boundary inward.
    - The path is colored by a repeating three-segment pattern: the same three colors
      appear cyclically, and each color always keeps the same segment length.
    - Some path pixels are overwritten by single-pixel markers. A marker reuses one of
      the three path colors, but it is placed on top of a differently colored segment.

    Rule:
    - Reconstruct the hidden path colors by fitting the global three-color / three-length
      pattern along the spiral order.
    - Treat each mismatch pixel as a marker and conceptually slide it forward along the
      spiral until the first future segment whose color matches the marker.
    - Because markers use the same visible color as their destination segment, the final
      rendered grid is the fully restored spiral path.
    """
    out, _meta = _solve(grid)
    return out


def _choose_marker_positions(
    rng: random.Random,
    base_seq: np.ndarray,
    colors: tuple[int, int, int],
) -> list[tuple[int, int]]:
    n = len(base_seq)
    starts = _segment_start_mask(base_seq)

    candidates: list[tuple[int, list[int]]] = []
    for idx in range(1, n - 1):
        if starts[idx]:
            continue
        if int(base_seq[idx - 1]) != int(base_seq[idx]) or int(base_seq[idx + 1]) != int(base_seq[idx]):
            continue

        future_colors = []
        for color in colors:
            if int(color) == int(base_seq[idx]):
                continue
            for pos in range(idx + 1, n):
                if starts[pos] and int(base_seq[pos]) == int(color):
                    future_colors.append(int(color))
                    break
        if future_colors:
            candidates.append((idx, future_colors))

    rng.shuffle(candidates)
    if not candidates:
        return []

    max_markers = min(6, max(2, len(candidates) // 4))
    target_count = rng.randint(2, max_markers)
    chosen: list[tuple[int, int]] = []
    used_idx: set[int] = set()

    for idx, future_colors in candidates:
        if any(abs(idx - prev_idx) <= 1 for prev_idx in used_idx):
            continue
        marker_color = int(rng.choice(future_colors))
        chosen.append((int(idx), marker_color))
        used_idx.add(int(idx))
        if len(chosen) >= target_count:
            break

    return chosen


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(9, 17)
    width = rng.randint(9, 17)
    path = _spiral_path(height, width)
    if len(path) < 36:
        return None

    colors = tuple(rng.sample(range(1, 10), 3))
    seg_lengths = tuple(rng.sample(range(3, 7), 3))
    base_seq = _build_sequence(len(path), colors, seg_lengths)
    base_grid = _render_path((height, width), path, base_seq)

    markers = _choose_marker_positions(rng, base_seq, colors)
    if len(markers) < 2:
        return None

    inp = base_grid.copy()
    for idx, marker_color in markers:
        r, c = path[idx]
        inp[r, c] = int(marker_color)

    solved, meta = _solve(inp)
    if meta["tie_count"] != 1:
        return None
    if not np.array_equal(solved, base_grid):
        return None

    boundary_endpoints = [
        cell
        for cell in (path[0], path[-1])
        if _is_boundary_cell(cell, height, width)
    ]
    if len(boundary_endpoints) != 1 or boundary_endpoints[0] != path[0]:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-marker sample")
