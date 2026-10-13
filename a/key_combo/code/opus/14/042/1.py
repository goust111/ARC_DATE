import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in steps:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _legend_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components_from_mask(grid != bg, connectivity=4):
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if w != 2 or h < 3:
            continue
        if coords.shape[0] != 2 * h:
            continue
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(block == bg):
            continue
        if len(set(map(int, block.ravel()))) < 3:
            continue
        candidates.append((r0, c0, r1, c1))
    return candidates


def _parse_instance(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    candidates = _legend_candidates(g, bg)
    assert len(candidates) == 1, "expected one 2-column legend"
    legend_bbox = candidates[0]
    r0, c0, r1, c1 = legend_bbox
    pairs = [(int(g[r, c0]), int(g[r, c1])) for r in range(r0, r1 + 1)]

    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    seeds = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != bg) & (~legend_mask))]
    assert 2 <= len(seeds) <= 3, "expected 2-3 seed pixels"

    src_colors = {src for src, _dst in pairs}
    for r, c, color in seeds:
        assert color in src_colors, "seed color must appear in the legend left column"
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) == (r, c) or not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                    continue
                assert not ((g[nr, nc] != bg) and (not legend_mask[nr, nc])), "seeds must be isolated"
                assert not legend_mask[nr, nc], "seed must not touch the legend"
    return bg, legend_bbox, pairs, seeds


def _apply_chain_trace(color: int, pairs: list[tuple[int, int]]) -> tuple[int, list[int], int | None]:
    current = int(color)
    used_rows: list[int] = []
    first_match_dst: int | None = None
    for idx, (src, dst) in enumerate(pairs):
        if current == src:
            current = int(dst)
            used_rows.append(idx)
            if first_match_dst is None:
                first_match_dst = int(dst)
    return current, used_rows, first_match_dst


def _apply_chain(color: int, pairs: list[tuple[int, int]]) -> int:
    return _apply_chain_trace(color, pairs)[0]


def _color_sequence(start: int, pairs: list[tuple[int, int]]) -> list[int]:
    seq = [int(start)]
    seen = {int(start)}
    current = int(start)
    for _ in range(20):
        current = _apply_chain(current, pairs)
        seq.append(int(current))
        if current in seen:
            return seq
        seen.add(int(current))
    raise AssertionError("replacement chain failed to repeat quickly")


def _chebyshev_distance(r0: int, c0: int, r1: int, c1: int) -> int:
    return max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1)))


def _point_bbox_chebyshev_distance(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return max(dr, dc)


def _expected_output_from_parts(
    shape: tuple[int, int],
    bg: int,
    legend_bbox: tuple[int, int, int, int],
    pairs: list[tuple[int, int]],
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    r0, c0, r1, c1 = legend_bbox
    for rr, (src, dst) in enumerate(pairs, start=r0):
        out[rr, c0] = int(src)
        out[rr, c1] = int(dst)

    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    rows, cols = np.indices((h, w))
    for sr, sc, color in sorted(seeds, key=lambda item: (item[0], item[1])):
        seq = _color_sequence(color, pairs)
        distance = np.maximum(np.abs(rows - sr), np.abs(cols - sc))
        for dist, ring_color in enumerate(seq):
            mask = (distance == dist) & (~legend_mask) & (out == bg)
            out[mask] = int(ring_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each seed into Chebyshev rings whose colors come from iterative ordered legend replacement.

    Input structure:
    - The background is black.
    - One isolated 2-column vertical legend contains 3-5 color pairs, read top to bottom.
    - Elsewhere there are 2-3 isolated seed pixels whose colors all appear in the legend's left column.

    Rule:
    - Read each legend row as an ordered substitution from left color to right color.
    - For every seed, ring 0 is the seed itself; ring d uses the color obtained by running one full top-to-bottom substitution scan on ring d-1's color.
    - Draw the first repeated color ring as well, then stop that seed's expansion; when different seeds cover the same cell, the seed with smaller row, then smaller column, wins. The legend stays unchanged.

    Key insight:
    - The legend is not a static distance palette: it is a state machine whose ordered substitutions are reapplied once per ring, and the first repeated state is the stopping condition.
    """
    g = np.asarray(grid, dtype=int)
    bg, legend_bbox, pairs, seeds = _parse_instance(g)
    return _expected_output_from_parts(g.shape, bg, legend_bbox, pairs, seeds)


def _build_pairs() -> tuple[list[tuple[int, int]], list[int]]:
    n_rows = random.randint(3, 5)
    colors = random.sample(range(1, 10), n_rows)
    a, b, c = colors[:3]
    pairs = [(a, b), (c, a), (b, c)]
    if n_rows >= 4:
        d = colors[3]
        pairs.append((d, a))
    if n_rows == 5:
        e = colors[4]
        pairs.append((e, d))
    return pairs, colors


def _choose_seed_colors(pairs: list[tuple[int, int]]) -> list[int]:
    src = [s for s, _ in pairs]
    if len(pairs) == 3:
        return [src[0], src[2], src[1]]
    if len(pairs) == 4:
        if random.random() < 0.5:
            return [src[3], src[0]]
        return [src[3], src[2], src[1]]
    if random.random() < 0.5:
        return [src[4], src[0]]
    return [src[4], src[2], src[1]]


def _seed_specs(seed_colors: list[int], pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    specs = [(int(color), len(_color_sequence(color, pairs)) - 1) for color in seed_colors]
    random.shuffle(specs)
    return specs


def _legend_position(height: int, width: int, n_rows: int) -> tuple[int, int, int, int]:
    r0 = random.randint(1, 2)
    if random.random() < 0.5:
        c0 = random.randint(1, 3)
    else:
        c0 = random.randint(width - 5, width - 3)
    return r0, c0, r0 + n_rows - 1, c0 + 1


def _margin_to_edge(r: int, c: int, height: int, width: int) -> int:
    return min(r, c, height - 1 - r, width - 1 - c)


def _seed_square_intersects(sr0: int, sc0: int, rad0: int, sr1: int, sc1: int, rad1: int) -> bool:
    return abs(sr0 - sr1) <= rad0 + rad1 and abs(sc0 - sc1) <= rad0 + rad1


def _choose_seed_positions(
    height: int,
    width: int,
    legend_bbox: tuple[int, int, int, int],
    specs: list[tuple[int, int]],
    need_overlap: bool,
    need_clipped: bool,
) -> list[tuple[int, int, int]]:
    clip_index = random.randrange(len(specs)) if need_clipped else None
    order = sorted(range(len(specs)), key=lambda idx: (-specs[idx][1], idx))
    for _ in range(1000):
        placed: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in order:
            _color, radius = specs[idx]
            candidates: list[tuple[int, int]] = []
            for r in range(legend_bbox[2] + 3, height):
                for c in range(width):
                    if _point_bbox_chebyshev_distance(r, c, legend_bbox) <= 1:
                        continue
                    if _point_bbox_chebyshev_distance(r, c, legend_bbox) <= radius:
                        continue
                    margin = _margin_to_edge(r, c, height, width)
                    if idx == clip_index:
                        if margin >= radius:
                            continue
                    else:
                        if margin < radius:
                            continue
                    if any(_chebyshev_distance(r, c, pr, pc) <= 1 for pr, pc in placed.values()):
                        continue
                    candidates.append((r, c))
            random.shuffle(candidates)
            if not candidates:
                ok = False
                break
            placed[idx] = candidates[0]
        if not ok:
            continue

        seeds = [(placed[idx][0], placed[idx][1], specs[idx][0]) for idx in range(len(specs))]
        radii = [specs[idx][1] for idx in range(len(specs))]

        if any(
            _chebyshev_distance(seeds[i][0], seeds[i][1], seeds[j][0], seeds[j][1]) <= max(radii[i], radii[j])
            for i in range(len(seeds))
            for j in range(i + 1, len(seeds))
        ):
            continue

        overlap = any(
            _seed_square_intersects(seeds[i][0], seeds[i][1], radii[i], seeds[j][0], seeds[j][1], radii[j])
            for i in range(len(seeds))
            for j in range(i + 1, len(seeds))
        )
        if overlap != need_overlap:
            continue
        return seeds
    raise RuntimeError("failed to place seeds")


def _rows_used_in_seed_sequence(start: int, pairs: list[tuple[int, int]]) -> set[int]:
    used: set[int] = set()
    seen = {int(start)}
    current = int(start)
    for _ in range(20):
        nxt, rows_used, _first = _apply_chain_trace(current, pairs)
        used.update(rows_used)
        if nxt in seen:
            return used
        seen.add(int(nxt))
        current = int(nxt)
    raise AssertionError("sequence did not terminate")


def _validate_generated_instance(
    grid: np.ndarray,
    expected_pairs: list[tuple[int, int]],
    expected_seeds: list[tuple[int, int, int]],
    expect_overlap: bool,
    expect_clipped: bool,
) -> None:
    g = np.asarray(grid, dtype=int)
    bg, legend_bbox, parsed_pairs, parsed_seeds = _parse_instance(g)
    assert parsed_pairs == expected_pairs
    assert sorted(parsed_seeds) == sorted(expected_seeds)

    sequences = {color: _color_sequence(color, parsed_pairs) for _r, _c, color in parsed_seeds}
    assert len({tuple(seq) for seq in sequences.values()}) >= 2
    assert any(seq[-1] != seq[-2] for seq in sequences.values())

    row_usage = set()
    multi_step_needed = False
    for _r, _c, color in parsed_seeds:
        row_usage.update(_rows_used_in_seed_sequence(color, parsed_pairs))
        final_color, used_rows, first_match = _apply_chain_trace(color, parsed_pairs)
        if len(used_rows) >= 2 and first_match is not None and int(final_color) != int(first_match):
            multi_step_needed = True
    assert row_usage == set(range(len(parsed_pairs)))
    assert multi_step_needed

    clipped = any(
        _margin_to_edge(r, c, g.shape[0], g.shape[1]) < len(sequences[color]) - 1
        for r, c, color in parsed_seeds
    )
    assert clipped == expect_clipped

    out = main(g)
    assert not np.array_equal(out, g)
    assert np.count_nonzero(out != bg) > np.count_nonzero(g != bg)

    r0, c0, r1, c1 = legend_bbox
    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    overlap = False
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if legend_mask[r, c]:
                assert out[r, c] == g[r, c]
                continue
            coverers: list[tuple[int, int, int, int]] = []
            for sr, sc, color in parsed_seeds:
                seq = sequences[color]
                dist = _chebyshev_distance(r, c, sr, sc)
                if dist <= len(seq) - 1:
                    coverers.append((sr, sc, color, seq[dist]))
            if len(coverers) >= 2:
                overlap = True
            if not coverers:
                assert out[r, c] == bg
            else:
                winner = min(coverers, key=lambda item: (item[0], item[1]))
                assert out[r, c] == winner[3]
    assert overlap == expect_overlap


def _generate_attempt() -> np.ndarray:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    pairs, _colors = _build_pairs()
    seed_colors = _choose_seed_colors(pairs)
    specs = _seed_specs(seed_colors, pairs)
    legend_bbox = _legend_position(height, width, len(pairs))
    need_overlap = random.random() < 0.55
    need_clipped = random.random() < 0.45
    seeds = _choose_seed_positions(height, width, legend_bbox, specs, need_overlap, need_clipped)

    grid = np.zeros((height, width), dtype=int)
    r0, c0, _r1, _c1 = legend_bbox
    for offset, (src, dst) in enumerate(pairs):
        grid[r0 + offset, c0] = int(src)
        grid[r0 + offset, c0 + 1] = int(dst)
    for r, c, color in seeds:
        grid[r, c] = int(color)

    _validate_generated_instance(grid, pairs, seeds, need_overlap, need_clipped)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _generate_attempt()
        except AssertionError:
            continue
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid chain-ring puzzle")
