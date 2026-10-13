from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
BACKSLASH = "\\"
SLASH = "/"
_RNG = np.random.default_rng()

# --------------- shared helpers ---------------


def _in_bounds(r: int, c: int, height: int, width: int) -> bool:
    return 0 <= r < height and 0 <= c < width


def _boundary_direction(r: int, c: int, height: int, width: int) -> int:
    if r == 0 and 0 < c < width - 1:
        return 2
    if r == height - 1 and 0 < c < width - 1:
        return 0
    if c == 0 and 0 < r < height - 1:
        return 1
    if c == width - 1 and 0 < r < height - 1:
        return 3
    raise AssertionError("source must lie on exactly one non-corner edge")


def _reflect_dir(dir_idx: int, mirror_kind: str) -> int:
    if mirror_kind == BACKSLASH:
        return [3, 2, 1, 0][dir_idx]
    if mirror_kind == SLASH:
        return [1, 0, 3, 2][dir_idx]
    raise AssertionError("unknown mirror kind")


def _cells_ahead(pos: tuple[int, int], dir_idx: int, height: int, width: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    r, c = pos
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(r, c, height, width):
            return cells
        cells.append((int(r), int(c)))


def _parse_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    height, width = map(int, grid.shape)
    assert int(grid[height - 1, width - 1]) not in (BLACK, GREY), "legend must end at bottom-right corner"

    start = int(width - 1)
    while start - 1 >= 0 and int(grid[height - 1, start - 1]) not in (BLACK, GREY):
        start -= 1
    colors = [int(v) for v in grid[height - 1, start:width]]
    assert len(colors) >= 3, "legend must contain at least three colors"
    assert len(set(colors)) == len(colors), "legend colors must be distinct"
    if start - 1 >= 0:
        assert int(grid[height - 1, start - 1]) == BLACK, "legend must be separated from other bottom-edge objects"
    return int(start), colors


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((int(r + dr), int(c + dc)))
    return out


def _extract_mirror_map(grid: np.ndarray) -> dict[tuple[int, int], str]:
    height, width = map(int, grid.shape)
    grey_mask = grid == GREY
    visited = np.zeros_like(grey_mask, dtype=bool)
    mirror_map: dict[tuple[int, int], str] = {}

    for r in range(height):
        for c in range(width):
            if not bool(grey_mask[r, c]) or bool(visited[r, c]):
                continue

            stack = [(int(r), int(c))]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors8(rr, cc):
                    if not _in_bounds(nr, nc, height, width):
                        continue
                    if not bool(grey_mask[nr, nc]) or bool(visited[nr, nc]):
                        continue
                    visited[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            coords.sort()
            assert len(coords) in (2, 3), "each mirror segment must have length 2 or 3"
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            assert len(set(rows)) == len(coords), "mirror segment rows must be unique"
            row_steps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
            col_steps = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
            assert all(step == 1 for step in row_steps), "mirror pixels must advance one row per step"
            if all(step == 1 for step in col_steps):
                kind = BACKSLASH
            else:
                assert all(step == -1 for step in col_steps), "mirror pixels must form a diagonal segment"
                kind = SLASH

            for cell in coords:
                mirror_map[cell] = kind

    return mirror_map


def _list_sources(grid: np.ndarray, legend_start: int) -> list[tuple[int, int, int]]:
    height, width = map(int, grid.shape)
    sources: list[tuple[int, int, int]] = []
    for r in range(height):
        for c in range(width):
            if not (r in (0, height - 1) or c in (0, width - 1)):
                continue
            if r == height - 1 and c >= legend_start:
                continue
            value = int(grid[r, c])
            if value in (BLACK, GREY):
                continue
            sources.append((int(r), int(c), value))
    return sources


def _trace_beams(
    grid: np.ndarray,
    sources: list[tuple[int, int, int]],
    legend_colors: list[int],
    mirror_map: dict[tuple[int, int], str],
) -> np.ndarray:
    height, width = map(int, grid.shape)
    next_color = {
        int(legend_colors[i]): int(legend_colors[(i + 1) % len(legend_colors)])
        for i in range(len(legend_colors))
    }
    out = np.array(grid, dtype=int, copy=True)
    base = np.array(grid, dtype=int, copy=False)

    for r0, c0, color0 in sorted(sources, key=lambda item: (item[0], item[1])):
        assert color0 in next_color, "source color must appear in the legend"
        dir_idx = _boundary_direction(r0, c0, height, width)
        r, c = int(r0), int(c0)
        color = int(color0)
        seen: set[tuple[int, int, int, int]] = set()

        while True:
            state = (int(r), int(c), int(dir_idx), int(color))
            if state in seen:
                break
            seen.add(state)

            dr, dc = DIRS[dir_idx]
            nr, nc = int(r + dr), int(c + dc)
            if not _in_bounds(nr, nc, height, width):
                break

            cell = int(base[nr, nc])
            if cell == BLACK:
                r, c = int(nr), int(nc)
                out[r, c] = int(color)
                continue

            if cell == GREY:
                assert (int(nr), int(nc)) in mirror_map, "every grey pixel must belong to a mirror segment"
                r, c = int(nr), int(nc)
                dir_idx = _reflect_dir(dir_idx, mirror_map[(r, c)])
                color = next_color[color]
                continue

            break

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored edge beams through grey diagonal mirrors and rotate each beam color after every reflection.

    Input structure:
    - The background is black.
    - Edge cells contain several single-pixel colored sources; a source's edge determines its initial direction into the grid.
    - Interior grey pixels form separate diagonal mirror segments of length 2 or 3, each oriented either as `\\` or `/`.
    - The bottom-right corner contains a one-row color-cycle legend: distinct non-grey colors arranged left-to-right.

    Rule:
    - Keep the grid size and all non-black structural objects unchanged.
    - For each source, move one cell at a time in its current cardinal direction and paint every visited black cell with the beam's current color.
    - When the beam enters a grey mirror cell, reflect its direction by the mirror orientation and advance its color to the next legend color cyclically.
    - Continue until the beam would leave the grid; if multiple beams paint the same black cell, later beams overwrite earlier ones.

    Key insight:
    - The task combines two independent cues: diagonal grey segments encode geometric reflection, while the bottom-right legend encodes the cyclic color change triggered by each reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    legend_start, legend_colors = _parse_legend(g)
    mirror_map = _extract_mirror_map(g)
    sources = _list_sources(g, legend_start)
    return _trace_beams(g, sources, legend_colors, mirror_map)


# --------------- generator ---------------

MIN_SIZE = 12
MAX_SIZE = 24
MAX_GEN_ATTEMPTS = 600
MIN_TOTAL_PAINTED = 8


def _mirror_variants(hit: tuple[int, int], kind: str, height: int, width: int) -> list[list[tuple[int, int]]]:
    r, c = hit
    if kind == BACKSLASH:
        candidates = [
            [(r, c), (r + 1, c + 1)],
            [(r - 1, c - 1), (r, c)],
            [(r - 1, c - 1), (r, c), (r + 1, c + 1)],
        ]
    else:
        candidates = [
            [(r, c), (r + 1, c - 1)],
            [(r - 1, c + 1), (r, c)],
            [(r - 1, c + 1), (r, c), (r + 1, c - 1)],
        ]

    out: list[list[tuple[int, int]]] = []
    for seg in candidates:
        if all(_in_bounds(rr, cc, height, width) and 0 < rr < height - 1 and 0 < cc < width - 1 for rr, cc in seg):
            out.append([(int(rr), int(cc)) for rr, cc in seg])
    return out


def _mirror_segment_separate(
    segment: list[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
) -> bool:
    for r, c in segment:
        for mr, mc in mirror_cells:
            if max(abs(int(r) - int(mr)), abs(int(c) - int(mc))) <= 1:
                return False
    return True


def _plan_beam(
    start: tuple[int, int],
    dir_idx: int,
    signs: list[str],
    occupied: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
    height: int,
    width: int,
) -> tuple[list[tuple[int, int]], list[tuple[str, list[tuple[int, int]]]]] | None:
    def rec(
        pos: tuple[int, int],
        direction: int,
        bounce_idx: int,
        occ: set[tuple[int, int]],
        mirrors_now: set[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[tuple[str, list[tuple[int, int]]]]] | None:
        ahead = _cells_ahead(pos, direction, height, width)

        if bounce_idx == len(signs):
            if len(ahead) < 1:
                return None
            final_cells: list[tuple[int, int]] = []
            for cell in ahead:
                if cell in occ:
                    return None
                final_cells.append(cell)
            return final_cells, []

        sign = signs[bounce_idx]
        candidates: list[tuple[list[tuple[int, int]], list[tuple[int, int]], int, str]] = []
        for dist, hit in enumerate(ahead, start=1):
            if hit in occ:
                break
            if dist < 2:
                continue
            segment_variants = _mirror_variants(hit, sign, height, width)
            _RNG.shuffle(segment_variants)
            next_dir = _reflect_dir(direction, sign)
            next_step = (int(hit[0] + DIRS[next_dir][0]), int(hit[1] + DIRS[next_dir][1]))
            if not _in_bounds(next_step[0], next_step[1], height, width):
                continue
            straight_cells = list(ahead[: dist - 1])
            for segment in segment_variants:
                seg_set = set(segment)
                if any(cell in occ for cell in seg_set):
                    continue
                if not _mirror_segment_separate(segment, mirrors_now):
                    continue
                candidates.append((straight_cells, segment, next_dir, sign))

        _RNG.shuffle(candidates)
        for straight_cells, segment, next_dir, sign_used in candidates:
            occ2 = set(occ)
            occ2.update(straight_cells)
            occ2.update(segment)
            mirrors2 = set(mirrors_now)
            mirrors2.update(segment)
            # The recursive start position is the hit cell, which is the path cell shared by all variants.
            hit_cell = ahead[len(straight_cells)]
            result = rec(hit_cell, next_dir, bounce_idx + 1, occ2, mirrors2)
            if result is None:
                continue
            tail_cells, tail_segments = result
            return straight_cells + tail_cells, [(sign_used, segment)] + tail_segments

        return None

    return rec(start, dir_idx, 0, set(occupied), set(mirror_cells))


def _sample_sources(height: int, width: int, legend_start: int, count: int) -> list[tuple[int, int]]:
    edge_names = ["top", "bottom", "left", "right"]
    chosen_edges = list(_RNG.choice(edge_names, size=count, replace=False))
    positions: list[tuple[int, int]] = []
    for edge in chosen_edges:
        if edge == "top":
            c = int(_RNG.integers(1, width - 1))
            positions.append((0, c))
        elif edge == "bottom":
            c = int(_RNG.integers(1, legend_start - 1))
            positions.append((height - 1, c))
        elif edge == "left":
            r = int(_RNG.integers(1, height - 1))
            positions.append((r, 0))
        else:
            r = int(_RNG.integers(1, height - 1))
            positions.append((r, width - 1))
    return positions


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(MIN_SIZE, MAX_SIZE + 1))
    width = int(_RNG.integers(MIN_SIZE, MAX_SIZE + 1))

    legend_len = int(_RNG.integers(3, 6))
    if width < legend_len + 4:
        return None
    legend_start = int(width - legend_len)

    legend_colors = [int(v) for v in _RNG.choice(COLOR_POOL, size=legend_len, replace=False)]
    source_count = int(_RNG.integers(2, min(4, legend_len) + 1))
    source_positions = _sample_sources(height, width, legend_start, source_count)
    source_colors = [int(v) for v in _RNG.choice(legend_colors, size=source_count, replace=False)]

    featured = int(_RNG.integers(source_count))
    bounce_counts = np.zeros(source_count, dtype=int)
    bounce_counts[featured] = int(_RNG.integers(2, 4))
    extra_reflections = int(_RNG.integers(0, source_count + 1))
    for _ in range(extra_reflections):
        choices = np.where(bounce_counts < 3)[0]
        idx = int(_RNG.choice(choices))
        bounce_counts[idx] += 1

    total_reflections = int(bounce_counts.sum())
    if total_reflections < 2:
        return None

    sign_sequence = [BACKSLASH, SLASH]
    sign_sequence.extend(
        BACKSLASH if int(bit) == 0 else SLASH
        for bit in _RNG.integers(0, 2, size=total_reflections - 2)
    )
    _RNG.shuffle(sign_sequence)

    beam_signs: list[list[str]] = []
    cursor = 0
    for count in bounce_counts:
        beam_signs.append(sign_sequence[cursor : cursor + int(count)])
        cursor += int(count)

    plan_order = sorted(range(source_count), key=lambda idx: (-int(bounce_counts[idx]), idx))
    occupied: set[tuple[int, int]] = set()
    mirror_cells: set[tuple[int, int]] = set()

    for c in range(legend_start, width):
        occupied.add((height - 1, int(c)))
    for pos in source_positions:
        occupied.add((int(pos[0]), int(pos[1])))

    beam_segments: list[list[tuple[str, list[tuple[int, int]]]]] = [[] for _ in range(source_count)]

    for idx in plan_order:
        start = source_positions[idx]
        direction = _boundary_direction(start[0], start[1], height, width)
        result = _plan_beam(start, direction, beam_signs[idx], occupied, mirror_cells, height, width)
        if result is None:
            return None
        path_cells, segments = result
        beam_segments[idx] = segments
        occupied.update(path_cells)
        for _, segment in segments:
            occupied.update(segment)
            mirror_cells.update(segment)

    grid = np.full((height, width), BLACK, dtype=int)
    for c, color in enumerate(legend_colors, start=legend_start):
        grid[height - 1, int(c)] = int(color)

    for pos, color in zip(source_positions, source_colors):
        grid[int(pos[0]), int(pos[1])] = int(color)

    for segments in beam_segments:
        for _, segment in segments:
            for r, c in segment:
                grid[int(r), int(c)] = GREY

    if len(mirror_cells) < 4:
        return None

    out = main(grid)
    painted = int(np.count_nonzero((grid == BLACK) & (out != BLACK)))
    if painted < MIN_TOTAL_PAINTED:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
