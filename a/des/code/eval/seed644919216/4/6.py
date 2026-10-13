from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PIPE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 600
ASSIGN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_of_color(g: np.ndarray, color: int) -> list[np.ndarray]:
    mask = g == color
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _find_cargoes(g: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = g.shape
    cargoes: list[dict] = []
    cargo_mask = np.zeros((h, w), dtype=bool)
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components_of_color(g, color):
            r0, c0, r1, c1 = _bbox(coords)
            size_r = r1 - r0 + 1
            size_c = c1 - c0 + 1
            if size_r != size_c or size_r not in {2, 3}:
                continue
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch == color):
                continue
            cargo_mask[r0 : r1 + 1, c0 : c1 + 1] = True
            cargoes.append(
                {
                    "color": int(color),
                    "size": int(size_r),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                }
            )
    assert cargoes, "expected at least one cargo square"
    return cargoes, cargo_mask


def _find_pipe_color(g: np.ndarray, bg: int, cargo_mask: np.ndarray) -> int:
    remaining = g[(g != bg) & (~cargo_mask)]
    assert remaining.size > 0
    counts = Counter(map(int, remaining.tolist()))
    return int(counts.most_common(1)[0][0])


def _square_border_mask(size: int) -> np.ndarray:
    border = np.ones((size, size), dtype=bool)
    if size == 3:
        border[1, 1] = False
    return border


def _find_frames(
    g: np.ndarray,
    bg: int,
    pipe_color: int,
    cargo_mask: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    h, w = g.shape
    frames: list[dict] = []
    frame_mask = np.zeros((h, w), dtype=bool)
    occupied = np.zeros((h, w), dtype=bool)

    for size in (3, 2):
        border = _square_border_mask(size)
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                if occupied[r0 : r0 + size, c0 : c0 + size].any():
                    continue
                if cargo_mask[r0 : r0 + size, c0 : c0 + size].any():
                    continue

                patch = g[r0 : r0 + size, c0 : c0 + size]
                if not np.all(patch[border] == pipe_color):
                    continue
                if size == 3 and int(patch[1, 1]) != bg:
                    continue

                contacts: set[tuple[int, int]] = set()
                for rr in range(r0, r0 + size):
                    for cc in range(c0, c0 + size):
                        if not border[rr - r0, cc - c0]:
                            continue
                        for dr, dc in DIRS4:
                            nr = rr + dr
                            nc = cc + dc
                            if not (0 <= nr < h and 0 <= nc < w):
                                continue
                            if r0 <= nr < r0 + size and c0 <= nc < c0 + size:
                                continue
                            if cargo_mask[nr, nc]:
                                continue
                            if int(g[nr, nc]) != bg:
                                contacts.add((int(nr), int(nc)))

                if len(contacts) != 1:
                    continue

                attach = next(iter(contacts))
                frames.append(
                    {
                        "size": int(size),
                        "bbox": (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)),
                        "attach": attach,
                    }
                )
                frame_mask[r0 : r0 + size, c0 : c0 + size] = True
                occupied[r0 : r0 + size, c0 : c0 + size] = True

    assert frames, "expected at least one exit frame"
    return frames, frame_mask


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            out.append((int(nr), int(nc)))
    return out


def _is_cross(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
            return False
    return True


def _trace_cargo(
    g: np.ndarray,
    pipe_mask: np.ndarray,
    pipe_color: int,
    start_endpoint: tuple[int, int],
    cargo_color: int,
) -> tuple[tuple[int, int], dict]:
    nbrs = _neighbors(pipe_mask, *start_endpoint)
    assert len(nbrs) == 1, "entrance must touch a pipe endpoint"

    current = (int(start_endpoint[0]), int(start_endpoint[1]))
    direction = (int(nbrs[0][0] - current[0]), int(nbrs[0][1] - current[1]))
    seen: set[tuple[int, int, int, int]] = set()
    stats = {"match": 0, "mismatch": 0, "straight": 0}

    while True:
        if current != start_endpoint and len(_neighbors(pipe_mask, *current)) == 1:
            return current, stats

        state = (current[0], current[1], direction[0], direction[1])
        assert state not in seen, "route must not loop"
        seen.add(state)

        if _is_cross(pipe_mask, *current):
            if int(g[current]) == pipe_color:
                stats["straight"] += 1
            else:
                if int(g[current]) == cargo_color:
                    direction = _turn_right(direction)
                    stats["match"] += 1
                else:
                    direction = _turn_left(direction)
                    stats["mismatch"] += 1

        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]
        assert pipe_mask[nr, nc], "cargo must stay on the pipe network"
        current = (int(nr), int(nc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transport each solid cargo square through the grey pipe network.

    - Cargo squares are solid 2x2 or 3x3 colored blocks adjacent to pipe endpoints.
    - Exit zones are same-sized grey frames attached to other endpoints.
    - At grey crossings cargo goes straight.
    - At colored crossings it turns right if the marker matches the cargo color,
      otherwise it turns left.
    - The input cargo blocks disappear and the corresponding exit frames are
      filled with the cargo color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 8 and w >= 8

    bg = int(Counter(map(int, g.ravel().tolist())).most_common(1)[0][0])
    cargoes, cargo_mask = _find_cargoes(g, bg)
    pipe_color = _find_pipe_color(g, bg, cargo_mask)
    frames, frame_mask = _find_frames(g, bg, pipe_color, cargo_mask)

    pipe_mask = (g != bg) & (~cargo_mask) & (~frame_mask)
    assert bool(pipe_mask.any())

    frame_by_attach: dict[tuple[int, int], dict] = {}
    for frame in frames:
        attach = tuple(frame["attach"])
        assert attach not in frame_by_attach
        frame_by_attach[attach] = frame

    out = g.copy()
    for cargo in cargoes:
        coords = np.asarray(cargo["coords"], dtype=int)
        adjacent: set[tuple[int, int]] = set()
        for rr, cc in coords:
            for dr, dc in DIRS4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and pipe_mask[nr, nc]:
                    adjacent.add((nr, nc))
        assert len(adjacent) == 1, "cargo must touch exactly one entrance endpoint"
        start_endpoint = next(iter(adjacent))
        dest_endpoint, _stats = _trace_cargo(
            g,
            pipe_mask,
            pipe_color,
            start_endpoint,
            int(cargo["color"]),
        )
        assert dest_endpoint in frame_by_attach, "destination endpoint must have a frame"
        frame = frame_by_attach[dest_endpoint]
        assert int(frame["size"]) == int(cargo["size"])

        out[coords[:, 0], coords[:, 1]] = bg
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(cargo["color"])

    return out


def _sample_spaced_positions(
    rng: np.random.Generator,
    low: int,
    high: int,
    count: int,
    min_gap: int,
) -> list[int] | None:
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        chosen = sorted(int(x) for x in rng.choice(values, size=count, replace=False))
        if all(b - a >= min_gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _rect_for_endpoint(r: int, c: int, side: str, size: int) -> tuple[int, int, int, int]:
    anchor = size // 2
    if side == "left":
        return r - anchor, c - size, r - anchor + size - 1, c - 1
    if side == "right":
        return r - anchor, c + 1, r - anchor + size - 1, c + size
    if side == "top":
        return r - size, c - anchor, r - 1, c - anchor + size - 1
    assert side == "bottom"
    return r + 1, c - anchor, r + size, c - anchor + size - 1


def _paint_block(g: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= g.shape[0] or c1 >= g.shape[1]:
        return False
    if np.any(g[r0 : r1 + 1, c0 : c1 + 1] != BG):
        return False
    g[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return True


def _paint_frame(g: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= g.shape[0] or c1 >= g.shape[1]:
        return False
    if np.any(g[r0 : r1 + 1, c0 : c1 + 1] != BG):
        return False
    g[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    if r1 - r0 + 1 == 3:
        g[r0 + 1, c0 + 1] = BG
    return True


def _endpoint_direction(side: str) -> tuple[int, int]:
    if side == "left":
        return 0, 1
    if side == "right":
        return 0, -1
    if side == "top":
        return 1, 0
    assert side == "bottom"
    return -1, 0


def _simulate_network(
    rows: list[int],
    cols: list[int],
    left_c: int,
    right_c: int,
    top_r: int,
    bottom_r: int,
    markers: dict[tuple[int, int], int],
    start_endpoint: dict,
    cargo_color: int,
) -> tuple[tuple[int, int], dict] | None:
    intersections = {(int(r), int(c)) for r in rows for c in cols}
    endpoints = {
        (int(ep["r"]), int(ep["c"])): ep
        for ep in (
            [{"r": r, "c": left_c, "side": "left"} for r in rows]
            + [{"r": r, "c": right_c, "side": "right"} for r in rows]
            + [{"r": top_r, "c": c, "side": "top"} for c in cols]
            + [{"r": bottom_r, "c": c, "side": "bottom"} for c in cols]
        )
    }
    row_nodes = [int(left_c)] + [int(c) for c in cols] + [int(right_c)]
    col_nodes = [int(top_r)] + [int(r) for r in rows] + [int(bottom_r)]

    current = (int(start_endpoint["r"]), int(start_endpoint["c"]))
    direction = _endpoint_direction(str(start_endpoint["side"]))
    start_coord = current
    seen: set[tuple[int, int, int, int]] = set()
    stats = {"match": 0, "mismatch": 0, "straight": 0}

    while True:
        if current != start_coord and current in endpoints:
            return current, stats
        state = (current[0], current[1], direction[0], direction[1])
        if state in seen:
            return None
        seen.add(state)

        if current in intersections:
            if current in markers:
                if int(markers[current]) == int(cargo_color):
                    direction = _turn_right(direction)
                    stats["match"] += 1
                else:
                    direction = _turn_left(direction)
                    stats["mismatch"] += 1
            else:
                stats["straight"] += 1

        dr, dc = direction
        if dr == 0:
            r = current[0]
            c = current[1]
            idx = row_nodes.index(c)
            next_idx = idx + 1 if dc > 0 else idx - 1
            if not (0 <= next_idx < len(row_nodes)):
                return None
            current = (int(r), int(row_nodes[next_idx]))
        else:
            r = current[0]
            c = current[1]
            idx = col_nodes.index(r)
            next_idx = idx + 1 if dr > 0 else idx - 1
            if not (0 <= next_idx < len(col_nodes)):
                return None
            current = (int(col_nodes[next_idx]), int(c))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    size = int(rng.choice([2, 3]))
    h = int(rng.integers(18, 27))
    w = int(rng.integers(18, 27))

    top_r = size
    bottom_r = h - size - 1
    left_c = size
    right_c = w - size - 1
    anchor = size // 2
    row_low = top_r + anchor
    row_high = bottom_r - (size - 1 - anchor)
    col_low = left_c + anchor
    col_high = right_c - (size - 1 - anchor)
    min_gap = size + 1

    shape_options: list[tuple[int, int, list[int], list[int]]] = []
    for row_count, col_count in ((2, 2), (2, 3), (3, 2)):
        rows = _sample_spaced_positions(rng, row_low, row_high, row_count, min_gap)
        cols = _sample_spaced_positions(rng, col_low, col_high, col_count, min_gap)
        if rows is not None and cols is not None:
            shape_options.append((row_count, col_count, rows, cols))
    if not shape_options:
        return None

    _row_count, _col_count, rows, cols = random.choice(shape_options)
    rows = sorted(rows)
    cols = sorted(cols)
    cargo_count = len(rows) + len(cols)
    cargo_colors = [int(x) for x in rng.choice(PALETTE, size=cargo_count, replace=False)]

    intersections = [(int(r), int(c)) for r in rows for c in cols]
    endpoints = (
        [{"r": int(r), "c": int(left_c), "side": "left"} for r in rows]
        + [{"r": int(r), "c": int(right_c), "side": "right"} for r in rows]
        + [{"r": int(top_r), "c": int(c), "side": "top"} for c in cols]
        + [{"r": int(bottom_r), "c": int(c), "side": "bottom"} for c in cols]
    )
    endpoint_coords = {(int(ep["r"]), int(ep["c"])) for ep in endpoints}

    for _ in range(ASSIGN_ATTEMPTS):
        active_count = int(rng.integers(2, len(intersections) + 1))
        active_nodes = rng.choice(len(intersections), size=active_count, replace=False)
        markers = {
            intersections[int(idx)]: int(rng.choice(cargo_colors))
            for idx in active_nodes
        }

        entrance_indices = rng.choice(len(endpoints), size=cargo_count, replace=False)
        entrances = [endpoints[int(i)] for i in entrance_indices]
        entrance_coords = {(int(ep["r"]), int(ep["c"])) for ep in entrances}
        permuted_colors = cargo_colors.copy()
        rng.shuffle(permuted_colors)

        traces: list[dict] = []
        destinations: set[tuple[int, int]] = set()
        total_match = 0
        total_mismatch = 0
        total_straight = 0
        ok = True
        for entrance, color in zip(entrances, permuted_colors):
            result = _simulate_network(
                rows,
                cols,
                left_c,
                right_c,
                top_r,
                bottom_r,
                markers,
                entrance,
                int(color),
            )
            if result is None:
                ok = False
                break
            dest, stats = result
            if dest in entrance_coords or dest in destinations or dest not in endpoint_coords:
                ok = False
                break
            destinations.add(dest)
            total_match += int(stats["match"])
            total_mismatch += int(stats["mismatch"])
            total_straight += int(stats["straight"])
            traces.append(
                {
                    "entrance": entrance,
                    "color": int(color),
                    "destination": dest,
                }
            )

        if not ok:
            continue
        if destinations != endpoint_coords - entrance_coords:
            continue
        if total_match == 0 or total_mismatch == 0 or total_straight == 0:
            continue

        grid = np.full((h, w), BG, dtype=int)
        for r in rows:
            grid[int(r), left_c : right_c + 1] = PIPE
        for c in cols:
            grid[top_r : bottom_r + 1, int(c)] = PIPE
        for (r, c), color in markers.items():
            grid[int(r), int(c)] = int(color)

        used_entrance_coords: set[tuple[int, int]] = set()
        for trace in traces:
            ep = trace["entrance"]
            bbox = _rect_for_endpoint(int(ep["r"]), int(ep["c"]), str(ep["side"]), size)
            if not _paint_block(grid, bbox, int(trace["color"])):
                ok = False
                break
            used_entrance_coords.add((int(ep["r"]), int(ep["c"])))
        if not ok:
            continue

        for ep in endpoints:
            coord = (int(ep["r"]), int(ep["c"]))
            if coord in used_entrance_coords:
                continue
            bbox = _rect_for_endpoint(int(ep["r"]), int(ep["c"]), str(ep["side"]), size)
            if not _paint_frame(grid, bbox, PIPE):
                ok = False
                break
        if not ok:
            continue

        try:
            solved = main(grid)
        except AssertionError:
            continue

        if solved.shape != grid.shape:
            continue
        if np.array_equal(solved, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-routing ARC input")
