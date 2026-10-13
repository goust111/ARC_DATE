import numpy as np


GREEN = 3
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FRAME_BORDER = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)
COLOR_NAMES = ("A", "B", "C", "D")
GEN_RETRIES = 200


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _frame_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }


def _draw_frame(grid: np.ndarray, center: tuple[int, int], fill: int) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2][FRAME_BORDER] = GREEN
    grid[r, c] = int(fill)


def _apply_symmetry(arr: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return arr.copy()
    if code == 1:
        return np.rot90(arr, 1).copy()
    if code == 2:
        return np.rot90(arr, 2).copy()
    if code == 3:
        return np.rot90(arr, 3).copy()
    if code == 4:
        return np.fliplr(arr).copy()
    if code == 5:
        return np.flipud(arr).copy()
    if code == 6:
        return np.rot90(np.fliplr(arr), 1).copy()
    return np.rot90(np.flipud(arr), 1).copy()


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            sub = g[r - 1 : r + 2, c - 1 : c + 2]
            if not np.all(sub[FRAME_BORDER] == GREEN):
                continue
            if int(sub[1, 1]) == GREEN:
                continue

            side_contacts = [
                ((r - 2, c), (r - 1, c)),
                ((r + 2, c), (r + 1, c)),
                ((r, c - 2), (r, c - 1)),
                ((r, c + 2), (r, c + 1)),
            ]
            contacts = []
            for endpoint, border_touch in side_contacts:
                er, ec = endpoint
                if not _in_bounds(g.shape, er, ec):
                    continue
                val = int(g[er, ec])
                if val == bg or val == GREEN:
                    continue
                contacts.append((endpoint, border_touch))
            frames.append(
                {
                    "center": (r, c),
                    "fill": int(g[r, c]),
                    "is_source": int(g[r, c]) != bg,
                    "contacts": contacts,
                }
            )
    return frames


def _trace_route(
    grid: np.ndarray,
    start: tuple[int, int],
    prev: tuple[int, int],
    bg: int,
) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    route: list[tuple[int, int]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    base_color = int(g[start])
    cur = start
    while True:
        state = (prev, cur)
        assert state not in seen, "route tracing entered a loop"
        seen.add(state)
        route.append(cur)

        candidates = []
        for nr, nc in _neighbors4(*cur):
            if not _in_bounds(g.shape, nr, nc) or (nr, nc) == prev:
                continue
            val = int(g[nr, nc])
            if val == bg or val == GREEN:
                continue
            candidates.append((nr, nc))

        if not candidates:
            break

        dr = cur[0] - prev[0]
        dc = cur[1] - prev[1]
        straight = (cur[0] + dr, cur[1] + dc)
        same_color = [cell for cell in candidates if int(g[cell]) == base_color]

        if int(g[cur]) == base_color and len(same_color) == 1:
            nxt = same_color[0]
        elif int(g[cur]) == base_color and len(same_color) == 2 and straight in same_color:
            nxt = straight
        elif straight in candidates:
            nxt = straight
        elif len(candidates) == 1:
            nxt = candidates[0]
        elif len(same_color) == 1:
            nxt = same_color[0]
        else:
            unseen = [cell for cell in candidates if cell not in route]
            assert len(unseen) == 1, "ambiguous path continuation"
            nxt = unseen[0]
        prev, cur = cur, nxt
    return route


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each source signal through single-pixel paths and fill the reached receiver center
    with the color of the final path.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    sources = [frame for frame in frames if frame["is_source"]]
    receivers = [frame for frame in frames if not frame["is_source"]]
    assert sources and receivers and len(sources) == len(receivers)

    routes: list[dict] = []
    for src in sources:
        assert len(src["contacts"]) == 1, "expected each source frame to expose one path endpoint"
        endpoint, border_touch = src["contacts"][0]
        route = _trace_route(g, endpoint, border_touch, bg)
        end = route[-1]
        receiver_matches = [idx for idx, frame in enumerate(receivers) if any(contact[0] == end for contact in frame["contacts"])]
        assert len(receiver_matches) == 1, "each source path must terminate at exactly one receiver"
        routes.append(
            {
                "cells": route,
                "color": int(g[endpoint]),
                "receiver_id": receiver_matches[0],
            }
        )

    cell_to_routes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for route_id, route in enumerate(routes):
        for idx, cell in enumerate(route["cells"]):
            cell_to_routes.setdefault(cell, []).append((route_id, idx))

    out = g.copy()
    for source_route in range(len(routes)):
        route_id = source_route
        idx = 0
        seen_states: set[tuple[int, int]] = set()
        while True:
            state = (route_id, idx)
            assert state not in seen_states, "signal routing entered a loop"
            seen_states.add(state)

            route = routes[route_id]
            if idx >= len(route["cells"]) - 1:
                receiver = receivers[route["receiver_id"]]
                rr, rc = receiver["center"]
                out[rr, rc] = int(route["color"])
                break

            cell = route["cells"][idx]
            memberships = cell_to_routes.get(cell, [])
            others = [(rid, pos) for rid, pos in memberships if rid != route_id]
            if others and int(g[cell]) != int(route["color"]):
                assert len(others) == 1, "expected pairwise path intersections"
                other_route, other_idx = others[0]
                route_id = other_route
                idx = other_idx + 1
            else:
                idx += 1

    return out


def _make_local_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    bg = int(rng.choice([c for c in range(10) if c != GREEN]))
    pool = [c for c in range(10) if c not in (bg, GREEN)]
    rng.shuffle(pool)
    path_colors = {name: pool[i] for i, name in enumerate(COLOR_NAMES)}
    source_colors = {name: pool[i + 4] for i, name in enumerate(COLOR_NAMES)}

    r_top = 3
    c_left = 3
    r_c = int(rng.integers(7, 10))
    r_a = r_c + int(rng.integers(2, 4))
    r_b = r_a + int(rng.integers(3, 5))
    r_d = r_b + int(rng.integers(1, 3))
    r_c_src = r_d + int(rng.integers(2, 4))

    c_c = int(rng.integers(4, 7))
    c_b = c_c + int(rng.integers(5, 8))
    c_c_end = c_b + int(rng.integers(4, 7))
    c_d = c_c_end + int(rng.integers(4, 7))
    c_a_end = c_d + int(rng.integers(4, 7))
    c_b_end = c_d + int(rng.integers(5, 8))

    h = r_c_src + 4
    w = max(c_a_end, c_b_end, c_c_end, c_d) + 4
    grid = np.full((h, w), bg, dtype=int)

    grid[r_a, c_left : c_a_end + 1] = path_colors["A"]
    grid[r_top : r_b + 1, c_b] = path_colors["B"]
    grid[r_b, c_b : c_b_end + 1] = path_colors["B"]
    grid[r_c : r_c_src + 1, c_c] = path_colors["C"]
    grid[r_c, c_c : c_c_end + 1] = path_colors["C"]
    grid[r_top : r_d + 1, c_d] = path_colors["D"]
    grid[r_d, c_left : c_d + 1] = path_colors["D"]

    grid[r_a, c_c] = path_colors["A"]
    grid[r_a, c_b] = path_colors["B"]
    grid[r_c, c_b] = path_colors["C"]
    grid[r_a, c_d] = path_colors["A"]
    grid[r_b, c_d] = path_colors["B"]
    grid[r_d, c_c] = path_colors["D"]

    source_centers = {
        "A": (r_a, c_left - 2),
        "B": (r_top - 2, c_b),
        "C": (r_c_src + 2, c_c),
        "D": (r_top - 2, c_d),
    }
    receiver_centers = {
        "A": (r_a, c_a_end + 2),
        "B": (r_b, c_b_end + 2),
        "C": (r_c, c_c_end + 2),
        "D": (r_d, c_left - 2),
    }

    for name, center in source_centers.items():
        _draw_frame(grid, center, source_colors[name])
    for center in receiver_centers.values():
        _draw_frame(grid, center, bg)

    expected = grid.copy()
    for name, center in receiver_centers.items():
        expected[center] = path_colors[name]
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        local_inp, local_out = _make_local_case(rng)
        code = int(rng.integers(0, 8))
        local_inp = _apply_symmetry(local_inp, code)
        local_out = _apply_symmetry(local_out, code)

        pad_top = int(rng.integers(0, 5))
        pad_bottom = int(rng.integers(0, 5))
        pad_left = int(rng.integers(0, 6))
        pad_right = int(rng.integers(0, 6))
        h, w = local_inp.shape
        bg = _most_common_color(local_inp)

        canvas_inp = np.full((h + pad_top + pad_bottom, w + pad_left + pad_right), bg, dtype=int)
        canvas_out = np.full_like(canvas_inp, bg)
        canvas_inp[pad_top : pad_top + h, pad_left : pad_left + w] = local_inp
        canvas_out[pad_top : pad_top + h, pad_left : pad_left + w] = local_out

        if np.array_equal(main(canvas_inp), canvas_out):
            return canvas_inp
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
