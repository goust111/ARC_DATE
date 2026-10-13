from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 8
ARROW = 9
PAYLOAD_COLORS = (1, 2, 3, 4, 5, 6, 7)
MAX_GENERATION_ATTEMPTS = 500

DIR_VECTORS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_SHAPES: dict[str, frozenset[tuple[int, int]]] = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]


def _spans_from_separators(length: int, seps: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in seps:
        spans.append((start, sep))
        start = sep + 1
    spans.append((start, length))
    return spans


def _detect_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    row_seps = _separator_indices(grid, axis=0)
    col_seps = _separator_indices(grid, axis=1)

    row_spans = _spans_from_separators(grid.shape[0], row_seps)
    col_spans = _spans_from_separators(grid.shape[1], col_seps)

    assert len(row_spans) == len(col_spans)
    assert len(row_spans) in (2, 3)
    assert all(r1 > r0 for r0, r1 in row_spans)
    assert all(c1 > c0 for c0, c1 in col_spans)
    assert len({r1 - r0 for r0, r1 in row_spans}) == 1
    assert len({c1 - c0 for c0, c1 in col_spans}) == 1
    return row_spans, col_spans, row_seps, col_seps


def _arrow_direction(region: np.ndarray) -> tuple[int, int]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(region == ARROW)]
    assert len(pts) == 5
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    rel = frozenset((r - min_r, c - min_c) for r, c in pts)
    for name, shape in ARROW_SHAPES.items():
        if rel == shape:
            return DIR_VECTORS[name]
    raise AssertionError(f"unrecognized arrow shape: {sorted(rel)}")


def _paint_separators(out: np.ndarray, row_seps: list[int], col_seps: list[int]) -> None:
    for r in row_seps:
        out[r, :] = SEP
    for c in col_seps:
        out[:, c] = SEP


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each region's payload into the neighboring region indicated by its maroon arrow.

    Input structure:
    - Black background with one-cell-thick azure separator rows and columns that partition the canvas into a uniform 2x2 or 3x3 grid of equal rectangular regions.
    - Every region contains exactly one maroon arrow symbol and one colored payload pattern made of non-background pixels.

    Rule:
    - Detect the arrow direction inside each region.
    - Remove the arrow and translate every non-arrow payload pixel by exactly one region step in that direction, preserving its local coordinates within the region.
    - Clear the source region; if the translation leaves the partition, the payload disappears.
    - If multiple payloads land in the same destination region, process source regions in row-major order so later pushes overwrite earlier colors on overlapping cells.

    Key insight:
    - The azure cross defines a repeated local coordinate frame, so solving the puzzle reduces to reading one arrow per region and copying that region's non-arrow pixels into the adjacent region at the same local offsets.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans, row_seps, col_seps = _detect_layout(g)

    out = np.full(g.shape, BG, dtype=int)
    _paint_separators(out, row_seps, col_seps)

    for src_ri, (src_r0, src_r1) in enumerate(row_spans):
        for src_ci, (src_c0, src_c1) in enumerate(col_spans):
            region = g[src_r0:src_r1, src_c0:src_c1]
            dr, dc = _arrow_direction(region)
            dst_ri = src_ri + dr
            dst_ci = src_ci + dc
            if not (0 <= dst_ri < len(row_spans) and 0 <= dst_ci < len(col_spans)):
                continue

            dst_r0, dst_r1 = row_spans[dst_ri]
            dst_c0, dst_c1 = col_spans[dst_ci]
            assert dst_r1 - dst_r0 == region.shape[0]
            assert dst_c1 - dst_c0 == region.shape[1]

            mask = (region != BG) & (region != ARROW)
            for local_r, local_c in np.argwhere(mask):
                out[dst_r0 + int(local_r), dst_c0 + int(local_c)] = int(region[local_r, local_c])

    return out


def _region_separators(count: int, size: int) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    seps: list[int] = []
    start = 0
    for idx in range(count):
        end = start + size
        spans.append((start, end))
        if idx != count - 1:
            seps.append(end)
            start = end + 1
    return spans, seps


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _grow_connected_cells(
    h: int,
    w: int,
    blocked: np.ndarray,
    size: int,
    allowed_cells: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    if allowed_cells is None:
        allowed = {(r, c) for r in range(h) for c in range(w) if not blocked[r, c]}
    else:
        allowed = {(r, c) for r, c in allowed_cells if not blocked[r, c]}
    if len(allowed) < size:
        return None

    start = random.choice(sorted(allowed))
    cells = {start}
    frontier = {nb for nb in _neighbors4(start[0], start[1], h, w) if nb in allowed}

    while len(cells) < size and frontier:
        cell = random.choice(sorted(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], h, w):
            if nb in allowed and nb not in cells:
                frontier.add(nb)

    if len(cells) != size:
        return None
    return sorted(cells)


def _random_payload_values(coords: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    assert coords
    if len(coords) == 1 or random.random() < 0.6:
        color_count = 1
    else:
        color_count = random.randint(2, min(3, len(coords), len(PAYLOAD_COLORS)))
    colors = random.sample(PAYLOAD_COLORS, color_count)
    shuffled = coords[:]
    random.shuffle(shuffled)

    values: dict[tuple[int, int], int] = {}
    for idx, coord in enumerate(shuffled):
        if idx < color_count:
            values[coord] = colors[idx]
        else:
            values[coord] = random.choice(colors)
    return values


def _arrow_cells(top: int, left: int, direction: str) -> set[tuple[int, int]]:
    return {(top + dr, left + dc) for dr, dc in ARROW_SHAPES[direction]}


def _expanded_bbox_mask(h: int, w: int, top: int, left: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r0 = max(0, top - 1)
    r1 = min(h, top + 4)
    c0 = max(0, left - 1)
    c1 = min(w, left + 4)
    mask[r0:r1, c0:c1] = True
    return mask


def _build_region(
    region_h: int,
    region_w: int,
    direction: str,
    forced_payload: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, dict[tuple[int, int], int]] | None:
    for _ in range(120):
        top = random.randint(0, region_h - 3)
        left = random.randint(0, region_w - 3)
        blocked = _expanded_bbox_mask(region_h, region_w, top, left)
        arrow_cells = _arrow_cells(top, left, direction)

        if forced_payload is not None:
            if any(blocked[r, c] for r, c in forced_payload):
                continue
            payload = dict(forced_payload)
        else:
            free_count = int(np.size(blocked) - np.count_nonzero(blocked))
            if free_count < 2:
                continue
            size = random.randint(2, min(6, free_count))
            payload_cells = _grow_connected_cells(region_h, region_w, blocked, size)
            if payload_cells is None:
                continue
            payload = _random_payload_values(payload_cells)

        region = np.full((region_h, region_w), BG, dtype=int)
        for (r, c), value in payload.items():
            region[r, c] = value
        for r, c in arrow_cells:
            region[r, c] = ARROW
        return region, payload
    return None


def _shared_payload(region_h: int, region_w: int) -> list[tuple[int, int]] | None:
    inner_cells = [
        (r, c)
        for r in range(1, region_h - 1)
        for c in range(1, region_w - 1)
    ]
    if len(inner_cells) < 2:
        inner_cells = [(r, c) for r in range(region_h) for c in range(region_w)]
    size = random.randint(2, min(5, len(inner_cells)))
    return _grow_connected_cells(
        region_h,
        region_w,
        np.zeros((region_h, region_w), dtype=bool),
        size,
        allowed_cells=inner_cells,
    )


def _toward(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for name, vec in DIR_VECTORS.items():
        if vec == (dr, dc):
            return name
    raise AssertionError("source and destination are not adjacent")


def _outward_choices(pos: tuple[int, int], count: int) -> list[str]:
    r, c = pos
    choices: list[str] = []
    if r == 0:
        choices.append("up")
    if r == count - 1:
        choices.append("down")
    if c == 0:
        choices.append("left")
    if c == count - 1:
        choices.append("right")
    return choices


def _inbound_choices(pos: tuple[int, int], count: int) -> list[str]:
    choices: list[str] = []
    for name, (dr, dc) in DIR_VECTORS.items():
        nr = pos[0] + dr
        nc = pos[1] + dc
        if 0 <= nr < count and 0 <= nc < count:
            choices.append(name)
    return choices


def _force_outward(
    directions: dict[tuple[int, int], str],
    count: int,
    protected: set[tuple[int, int]],
) -> None:
    if any(
        not (0 <= pos[0] + DIR_VECTORS[name][0] < count and 0 <= pos[1] + DIR_VECTORS[name][1] < count)
        for pos, name in directions.items()
    ):
        return

    edge_positions = [pos for pos in directions if _outward_choices(pos, count)]
    preferred = [pos for pos in edge_positions if pos not in protected]
    source = random.choice(preferred or edge_positions)
    directions[source] = random.choice(_outward_choices(source, count))


def _force_inbound(directions: dict[tuple[int, int], str], count: int) -> None:
    if any(
        0 <= pos[0] + DIR_VECTORS[name][0] < count and 0 <= pos[1] + DIR_VECTORS[name][1] < count
        for pos, name in directions.items()
    ):
        return
    source = random.choice(sorted(directions))
    directions[source] = random.choice(_inbound_choices(source, count))


def _render_from_metadata(
    count: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    row_seps: list[int],
    col_seps: list[int],
    regions: dict[tuple[int, int], np.ndarray],
    payloads: dict[tuple[int, int], dict[tuple[int, int], int]],
    directions: dict[tuple[int, int], str],
) -> tuple[np.ndarray, np.ndarray]:
    height = row_spans[-1][1]
    width = col_spans[-1][1]

    inp = np.full((height, width), BG, dtype=int)
    _paint_separators(inp, row_seps, col_seps)
    for (ri, ci), region in regions.items():
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        inp[r0:r1, c0:c1] = region

    out = np.full((height, width), BG, dtype=int)
    _paint_separators(out, row_seps, col_seps)
    for ri in range(count):
        for ci in range(count):
            direction = directions[(ri, ci)]
            dr, dc = DIR_VECTORS[direction]
            dst = (ri + dr, ci + dc)
            if not (0 <= dst[0] < count and 0 <= dst[1] < count):
                continue
            dst_r0, _ = row_spans[dst[0]]
            dst_c0, _ = col_spans[dst[1]]
            for (lr, lc), value in payloads[(ri, ci)].items():
                out[dst_r0 + lr, dst_c0 + lc] = value
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((2, 3))
    if count == 2:
        region_h = random.randint(7, 12)
        region_w = random.randint(7, 12)
    else:
        region_h = random.randint(5, 9)
        region_w = random.randint(5, 9)

    row_spans, row_seps = _region_separators(count, region_h)
    col_spans, col_seps = _region_separators(count, region_w)
    if row_spans[-1][1] > 30 or col_spans[-1][1] > 30:
        return None

    positions = [(ri, ci) for ri in range(count) for ci in range(count)]
    directions = {pos: random.choice(tuple(DIR_VECTORS)) for pos in positions}
    regions: dict[tuple[int, int], np.ndarray] = {}
    payloads: dict[tuple[int, int], dict[tuple[int, int], int]] = {}

    protected_sources: set[tuple[int, int]] = set()
    mode = random.choice(("free", "focus"))
    forced_payloads: dict[tuple[int, int], dict[tuple[int, int], int]] = {}

    if mode == "focus":
        candidates = [pos for pos in positions if len(_inbound_choices(pos, count)) >= 2]
        target = random.choice(candidates)
        neighbors: list[tuple[tuple[int, int], str]] = []
        for name, (dr, dc) in DIR_VECTORS.items():
            src = (target[0] - dr, target[1] - dc)
            if 0 <= src[0] < count and 0 <= src[1] < count:
                neighbors.append((src, name))
        random.shuffle(neighbors)
        chosen = neighbors[: random.randint(2, len(neighbors))]
        shared_cells = _shared_payload(region_h, region_w)
        if shared_cells is None:
            return None
        for src, direction in chosen:
            directions[src] = direction
            protected_sources.add(src)
            payload = _random_payload_values(shared_cells)
            while any(forced_payloads.get(other) == payload for other, _ in chosen if other != src):
                payload = _random_payload_values(shared_cells)
            forced_payloads[src] = payload

    _force_outward(directions, count, protected_sources)
    _force_inbound(directions, count)

    for pos in positions:
        built = _build_region(region_h, region_w, directions[pos], forced_payload=forced_payloads.get(pos))
        if built is None:
            return None
        region, payload = built
        regions[pos] = region
        payloads[pos] = payload

    inp, expected = _render_from_metadata(
        count=count,
        row_spans=row_spans,
        col_spans=col_spans,
        row_seps=row_seps,
        col_seps=col_seps,
        regions=regions,
        payloads=payloads,
        directions=directions,
    )

    if not np.array_equal(main(inp), expected):
        return None

    survivors = int(np.count_nonzero((expected != BG) & (expected != SEP)))
    if survivors == 0:
        return None

    if mode == "focus":
        incoming: dict[tuple[int, int], int] = {}
        for pos, name in directions.items():
            dr, dc = DIR_VECTORS[name]
            dst = (pos[0] + dr, pos[1] + dc)
            if 0 <= dst[0] < count and 0 <= dst[1] < count:
                incoming[dst] = incoming.get(dst, 0) + 1
        if max(incoming.values(), default=0) < 2:
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
