from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
DEFAULT_SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}): "right",
    frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}): "left",
    frozenset({(0, 1), (1, 1), (2, 1), (0, 0), (0, 2)}): "up",
    frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}): "down",
}

DRAW_PATTERNS = {
    "right": ((1, 0), (1, 1), (1, 2), (0, 2), (2, 2)),
    "left": ((1, 0), (1, 1), (1, 2), (0, 0), (2, 0)),
    "up": ((0, 1), (1, 1), (2, 1), (0, 0), (0, 2)),
    "down": ((0, 1), (1, 1), (2, 1), (2, 0), (2, 2)),
}


RegionId = tuple[int, int]


def _infer_separator_color(g: np.ndarray) -> int:
    candidates: dict[int, int] = {}
    for color in map(int, np.unique(g)):
        if color == BACKGROUND:
            continue
        row_hits = int(np.sum(np.all(g == color, axis=1)))
        col_hits = int(np.sum(np.all(g == color, axis=0)))
        total = row_hits + col_hits
        if total > 0:
            candidates[color] = total
    assert candidates, "expected a full-row or full-column separator color"
    best = max(candidates.values())
    winners = [color for color, score in candidates.items() if score == best]
    assert len(winners) == 1, "separator color must be structurally unique"
    return winners[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    cuts = [-1] + sorted(separators) + [length]
    spans: list[tuple[int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        start = left + 1
        end = right - 1
        assert start <= end, "separator lines must leave non-empty regions"
        spans.append((start, end))
    return spans


def _classify_region(sub: np.ndarray, separator: int) -> dict[str, int | str]:
    assert not np.any(sub == separator), "separator color cannot appear inside a region"
    coords = np.argwhere(sub != BACKGROUND)
    if len(coords) == 0:
        return {"kind": "empty"}
    colors = {int(sub[r, c]) for r, c in coords}
    assert len(colors) == 1, "each region must contain a single object color"
    color = next(iter(colors))
    if len(coords) == 1:
        return {"kind": "source", "color": color}
    assert len(coords) == 5, "foreground region must be a source or a 5-cell arrow"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    assert r1 - r0 == 2 and c1 - c0 == 2, "arrow must fit in a 3x3 box"
    rel = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    direction = ARROW_PATTERNS.get(rel)
    assert direction is not None, "unknown arrow shape"
    return {"kind": "arrow", "color": color, "direction": direction}


def _parse_regions(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], dict[RegionId, dict[str, object]]]:
    separator = _infer_separator_color(g)
    row_separators = [int(r) for r in np.where(np.all(g == separator, axis=1))[0]]
    col_separators = [int(c) for c in np.where(np.all(g == separator, axis=0))[0]]
    assert row_separators or col_separators, "expected at least one separator line"

    row_spans = _spans(g.shape[0], row_separators)
    col_spans = _spans(g.shape[1], col_separators)

    regions: dict[RegionId, dict[str, object]] = {}
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            info = _classify_region(g[r0 : r1 + 1, c0 : c1 + 1], separator)
            info["bbox"] = (r0, r1, c0, c1)
            regions[(rr, cc)] = info
    return separator, row_spans, col_spans, regions


def _neighbor(region: RegionId, direction: str, shape: tuple[int, int]) -> RegionId | None:
    dr, dc = DIRS[direction]
    nr = region[0] + dr
    nc = region[1] + dc
    if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
        return (nr, nc)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify colored arrow symbols, follow their directed relay chains between wall-separated
    regions, and fill each terminal empty region with the color of the source pixel that started
    the longest chain reaching it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _separator, row_spans, col_spans, regions = _parse_regions(g)
    shape = (len(row_spans), len(col_spans))

    arrows: dict[int, RegionId] = {}
    sources: list[tuple[RegionId, int]] = []
    for region, info in regions.items():
        kind = info["kind"]
        if kind == "arrow":
            color = int(info["color"])
            assert color not in arrows, "arrow colors must be unique"
            arrows[color] = region
        elif kind == "source":
            sources.append((region, int(info["color"])))

    assert arrows and sources, "expected at least one arrow and one source"

    winners: dict[RegionId, tuple[int, int]] = {}
    for _source_region, color in sources:
        start = arrows.get(color)
        assert start is not None, "each source color must match an arrow color"
        current = start
        seen: set[RegionId] = set()
        steps = 0
        while True:
            assert current not in seen, "relay chains must not cycle"
            seen.add(current)
            steps += 1
            current_info = regions[current]
            nxt = _neighbor(current, str(current_info["direction"]), shape)
            assert nxt is not None, "arrow must point to an adjacent region"
            nxt_info = regions[nxt]
            if nxt_info["kind"] == "arrow":
                current = nxt
                continue
            assert nxt_info["kind"] == "empty", "relay chains must end in empty regions"
            previous = winners.get(nxt)
            if previous is None or steps > previous[0]:
                winners[nxt] = (steps, color)
            elif steps == previous[0]:
                assert previous[1] == color, "equal-length conflicts are ambiguous"
            break

    out = g.copy()
    for region, (_steps, color) in winners.items():
        r0, r1, c0, c1 = regions[region]["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == BACKGROUND), "terminal region must be empty before filling"
        patch[:] = color
    return out


def _all_regions(shape: tuple[int, int]) -> list[RegionId]:
    return [(r, c) for r in range(shape[0]) for c in range(shape[1])]


def _region_neighbors(region: RegionId, shape: tuple[int, int]) -> list[RegionId]:
    neighbors = []
    for dr, dc in DIRS.values():
        nr = region[0] + dr
        nc = region[1] + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            neighbors.append((nr, nc))
    return neighbors


def _sample_backward_path(
    end: RegionId,
    arrow_count: int,
    forbidden: set[RegionId],
    shape: tuple[int, int],
    banned_terminal_neighbors: set[RegionId] | None = None,
) -> list[RegionId] | None:
    used0 = set(forbidden)
    used0.add(end)

    def dfs(front: RegionId, remaining: int, used: set[RegionId], path: list[RegionId]) -> list[RegionId] | None:
        if remaining == 0:
            return path
        choices = _region_neighbors(front, shape)
        random.shuffle(choices)
        if len(path) == 1 and banned_terminal_neighbors:
            choices = [cell for cell in choices if cell not in banned_terminal_neighbors]
        for cell in choices:
            if cell in used:
                continue
            result = dfs(cell, remaining - 1, used | {cell}, [cell] + path)
            if result is not None:
                return result
        return None

    return dfs(end, arrow_count, used0, [end])


def _place_unique_chain(length: int, reserved: set[RegionId], shape: tuple[int, int]) -> list[RegionId] | None:
    terminals = _all_regions(shape)
    random.shuffle(terminals)
    for terminal in terminals:
        if terminal in reserved:
            continue
        path = _sample_backward_path(terminal, length, reserved, shape)
        if path is not None:
            return path
    return None


def _layout_non_conflict(shape: tuple[int, int]) -> tuple[list[list[RegionId]], list[RegionId]] | None:
    region_count = shape[0] * shape[1]
    for _ in range(100):
        chain_count = random.randint(1, 3)
        lengths = [random.randint(1, 4) for _ in range(chain_count)]
        if sum(lengths) > len(PALETTE):
            continue
        if sum(lengths) + 2 * chain_count > region_count:
            continue

        chains: list[list[RegionId]] = []
        reserved: set[RegionId] = set()
        ok = True
        for length in sorted(lengths, reverse=True):
            path = _place_unique_chain(length, reserved, shape)
            if path is None:
                ok = False
                break
            chains.append(path)
            reserved.update(path)
        if not ok:
            continue

        unused = [cell for cell in _all_regions(shape) if cell not in reserved]
        if len(unused) < len(chains):
            continue
        sources = random.sample(unused, len(chains))
        random.shuffle(chains)
        return chains, sources
    return None


def _layout_conflict(shape: tuple[int, int]) -> tuple[list[list[RegionId]], list[RegionId]] | None:
    region_count = shape[0] * shape[1]
    all_cells = _all_regions(shape)
    shared_candidates = [cell for cell in all_cells if len(_region_neighbors(cell, shape)) >= 2]
    if not shared_candidates:
        return None

    for _ in range(100):
        long_len, short_len = sorted(random.sample(range(1, 5), 2), reverse=True)
        lengths = [long_len, short_len]
        extra_len: int | None = None
        if random.random() < 0.5:
            extra_max = min(3, len(PALETTE) - sum(lengths))
            if extra_max >= 1:
                extra_len = random.randint(1, extra_max)
                lengths.append(extra_len)
        terminal_count = 1 + (1 if extra_len is not None else 0)
        if sum(lengths) + terminal_count + len(lengths) > region_count:
            continue

        random.shuffle(shared_candidates)
        for shared_terminal in shared_candidates:
            first = _sample_backward_path(shared_terminal, long_len, set(), shape)
            if first is None:
                continue
            reserved = set(first)
            second = _sample_backward_path(
                shared_terminal,
                short_len,
                reserved,
                shape,
                banned_terminal_neighbors={first[-2]},
            )
            if second is None:
                continue

            chains = [first, second]
            reserved.update(second[:-1])
            reserved.add(shared_terminal)

            if extra_len is not None:
                extra = _place_unique_chain(extra_len, reserved, shape)
                if extra is None:
                    continue
                chains.append(extra)
                reserved.update(extra)

            unused = [cell for cell in all_cells if cell not in reserved]
            if len(unused) < len(chains):
                continue
            sources = random.sample(unused, len(chains))
            random.shuffle(chains)
            return chains, sources
    return None


def _build_partition() -> tuple[np.ndarray, dict[RegionId, tuple[int, int, int, int]], tuple[int, int]]:
    region_rows = random.randint(4, 5)
    region_cols = random.randint(4, 5)
    row_sizes = [random.randint(4, 6) for _ in range(region_rows)]
    col_sizes = [random.randint(4, 6) for _ in range(region_cols)]

    height = sum(row_sizes) + region_rows - 1
    width = sum(col_sizes) + region_cols - 1
    grid = np.full((height, width), BACKGROUND, dtype=int)

    row_cursor = 0
    row_starts: list[tuple[int, int]] = []
    for i, size in enumerate(row_sizes):
        row_starts.append((row_cursor, row_cursor + size - 1))
        row_cursor += size
        if i + 1 < region_rows:
            grid[row_cursor, :] = DEFAULT_SEPARATOR
            row_cursor += 1

    col_cursor = 0
    col_starts: list[tuple[int, int]] = []
    for i, size in enumerate(col_sizes):
        col_starts.append((col_cursor, col_cursor + size - 1))
        col_cursor += size
        if i + 1 < region_cols:
            grid[:, col_cursor] = DEFAULT_SEPARATOR
            col_cursor += 1

    regions: dict[RegionId, tuple[int, int, int, int]] = {}
    for rr, (r0, r1) in enumerate(row_starts):
        for cc, (c0, c1) in enumerate(col_starts):
            regions[(rr, cc)] = (r0, r1, c0, c1)
    return grid, regions, (region_rows, region_cols)


def _direction_between(a: RegionId, b: RegionId) -> str:
    delta = (b[0] - a[0], b[1] - a[1])
    for name, vec in DIRS.items():
        if vec == delta:
            return name
    raise AssertionError("regions must be adjacent")


def _draw_arrow(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, direction: str) -> None:
    r0, r1, c0, c1 = bbox
    top = random.randint(r0, r1 - 2)
    left = random.randint(c0, c1 - 2)
    for dr, dc in DRAW_PATTERNS[direction]:
        grid[top + dr, left + dc] = color


def _draw_source(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bbox
    grid[random.randint(r0, r1), random.randint(c0, c1)] = color


def _apply_transform(grid: np.ndarray, op: int) -> np.ndarray:
    if op == 0:
        return np.ascontiguousarray(grid)
    if op == 1:
        return np.ascontiguousarray(np.rot90(grid, 1))
    if op == 2:
        return np.ascontiguousarray(np.rot90(grid, 2))
    if op == 3:
        return np.ascontiguousarray(np.rot90(grid, 3))
    if op == 4:
        return np.ascontiguousarray(np.flipud(grid))
    if op == 5:
        return np.ascontiguousarray(np.fliplr(grid))
    if op == 6:
        return np.ascontiguousarray(grid.T)
    return np.ascontiguousarray(np.fliplr(grid).T)


def _render_scene(
    base: np.ndarray,
    regions: dict[RegionId, tuple[int, int, int, int]],
    chains: list[list[RegionId]],
    sources: list[RegionId],
) -> tuple[np.ndarray, np.ndarray]:
    arrow_regions = [region for chain in chains for region in chain[:-1]]
    assert len(arrow_regions) == len(set(arrow_regions)), "arrow regions must be unique"
    assert len(arrow_regions) <= len(PALETTE), "not enough distinct colors for arrows"

    arrow_colors = random.sample(list(PALETTE), len(arrow_regions))
    arrow_color_by_region = dict(zip(arrow_regions, arrow_colors))
    arrow_dir_by_region: dict[RegionId, str] = {}
    for chain in chains:
        for cur, nxt in zip(chain[:-1], chain[1:]):
            arrow_dir_by_region[cur] = _direction_between(cur, nxt)

    input_grid = base.copy()
    for region, color in arrow_color_by_region.items():
        _draw_arrow(input_grid, regions[region], color, arrow_dir_by_region[region])

    source_color_by_region: dict[RegionId, int] = {}
    for chain, source_region in zip(chains, sources):
        source_color = arrow_color_by_region[chain[0]]
        source_color_by_region[source_region] = source_color
        _draw_source(input_grid, regions[source_region], source_color)

    expected = input_grid.copy()
    winners: dict[RegionId, tuple[int, int]] = {}
    for chain in chains:
        terminal = chain[-1]
        steps = len(chain) - 1
        color = arrow_color_by_region[chain[0]]
        previous = winners.get(terminal)
        if previous is None or steps > previous[0]:
            winners[terminal] = (steps, color)
        elif steps == previous[0]:
            raise AssertionError("generator must avoid equal-length terminal conflicts")

    for region, (_steps, color) in winners.items():
        r0, r1, c0, c1 = regions[region]
        expected[r0 : r1 + 1, c0 : c1 + 1] = color

    return input_grid, expected


def _gen_attempt() -> np.ndarray | None:
    base, regions, shape = _build_partition()

    if random.random() < 0.5:
        layout = _layout_conflict(shape)
        if layout is None:
            layout = _layout_non_conflict(shape)
    else:
        layout = _layout_non_conflict(shape)
        if layout is None:
            layout = _layout_conflict(shape)
    if layout is None:
        return None

    chains, sources = layout
    if len(chains) != len(sources):
        return None

    try:
        inp, expected = _render_scene(base, regions, chains, sources)
    except AssertionError:
        return None

    op = random.randrange(8)
    inp = _apply_transform(inp, op)
    expected = _apply_transform(expected, op)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate a valid input after {GEN_RETRIES} attempts")
