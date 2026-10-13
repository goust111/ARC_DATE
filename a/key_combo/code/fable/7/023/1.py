from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox, find_components, in_bounds, neighbors4


BACKGROUND = Color.BLACK
GREY = Color.GREY
SIDES = ("top", "bottom", "left", "right")


def _step_for_side(side: str) -> tuple[int, int]:
    if side == "top":
        return 1, 0
    if side == "bottom":
        return -1, 0
    if side == "left":
        return 0, 1
    if side == "right":
        return 0, -1
    raise ValueError(f"unknown side: {side}")


def _source_coord(rect: tuple[int, int, int, int], side: str, lane: int, distance: int) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return r0 - distance - 1, lane
    if side == "bottom":
        return r1 + distance + 1, lane
    if side == "left":
        return lane, c0 - distance - 1
    if side == "right":
        return lane, c1 + distance + 1
    raise ValueError(f"unknown side: {side}")


def _classify_against_rect(cell: tuple[int, int], rect: tuple[int, int, int, int]) -> tuple[str, int, tuple[int, int]]:
    r, c = cell
    r0, c0, r1, c1 = rect
    matches: list[tuple[str, int, tuple[int, int]]] = []
    if c0 <= c <= c1 and r < r0:
        matches.append(("top", r0 - r - 1, (1, 0)))
    if c0 <= c <= c1 and r > r1:
        matches.append(("bottom", r - r1 - 1, (-1, 0)))
    if r0 <= r <= r1 and c < c0:
        matches.append(("left", c0 - c - 1, (0, 1)))
    if r0 <= r <= r1 and c > c1:
        matches.append(("right", c - c1 - 1, (0, -1)))
    assert len(matches) == 1, "each source must have a unique nearest cardinal direction to the grey rectangle"
    side, distance, step = matches[0]
    assert distance > 0
    return side, int(distance), step


def _grey_rect(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    grey_mask = grid == GREY
    comps = find_components(grid, bg=BACKGROUND, connectivity=4, monochromatic=True, colors=[GREY], return_format="dict")
    assert len(comps) == 1, "expected exactly one grey object"
    comp = comps[0]
    rect = tuple(map(int, comp["bbox"]))
    area = int(comp["area"])
    r0, c0, r1, c1 = rect
    assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "grey object must be a solid rectangle"
    assert bbox(grey_mask) == rect
    return grey_mask, rect


def _tail_label_map(grid: np.ndarray, tail_color: int) -> tuple[list[dict], np.ndarray]:
    tail_comps = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        colors=[tail_color],
        return_format="dict",
    )
    assert tail_comps, "expected at least one tail component"
    label_map = np.full(grid.shape, -1, dtype=int)
    for idx, comp in enumerate(tail_comps):
        coords = comp["coords"]
        label_map[coords[:, 0], coords[:, 1]] = idx
    return tail_comps, label_map


def _parse_case(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9

    grey_mask, rect = _grey_rect(g)
    other_colors = sorted(int(c) for c in np.unique(g) if c not in {BACKGROUND, GREY})
    tail_candidates = [c for c in other_colors if int(np.count_nonzero(g == c)) > 1]
    assert len(tail_candidates) == 1, "expected exactly one shared tail color"
    tail_color = int(tail_candidates[0])

    tail_comps, tail_labels = _tail_label_map(g, tail_color)
    source_colors = [c for c in other_colors if c != tail_color]
    assert len(source_colors) >= 2, "expected several source pixels"

    movers = []
    used_tail_ids = set()
    distances = []

    for color in source_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2), "each source color must appear exactly once as a single pixel"
        r, c = map(int, coords[0])
        adjacent_tail_ids = {
            int(tail_labels[nr, nc])
            for nr, nc in neighbors4(r, c)
            if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == tail_color
        }
        assert len(adjacent_tail_ids) == 1, "each source must touch exactly one tail"
        tail_id = adjacent_tail_ids.pop()
        assert tail_id not in used_tail_ids, "each tail must belong to one source"
        used_tail_ids.add(tail_id)

        comp = tail_comps[tail_id]
        tail_coords = {tuple(map(int, xy)) for xy in comp["coords"]}
        touching_source_cells = {
            cell
            for cell in tail_coords
            if abs(cell[0] - r) + abs(cell[1] - c) == 1
        }
        assert len(touching_source_cells) == 1, "tail must meet the source at exactly one end"

        side, distance, step = _classify_against_rect((r, c), rect)
        distances.append(distance)
        movers.append(
            {
                "color": int(color),
                "source": (r, c),
                "side": side,
                "distance": int(distance),
                "step": step,
                "tail_len": int(comp["area"]),
                "tail_id": int(tail_id),
            }
        )

    assert len(used_tail_ids) == len(tail_comps), "every tail component must have a source"
    assert len(set(distances)) == len(distances), "source-to-magnet distances must be unique"
    movers.sort(key=lambda m: (m["distance"], m["source"][0], m["source"][1]))
    return {
        "grid": g,
        "rect": rect,
        "grey_mask": grey_mask,
        "tail_color": tail_color,
        "movers": movers,
    }


def _solve_from_parsed(parsed: dict) -> np.ndarray:
    g = parsed["grid"]
    out = np.full_like(g, BACKGROUND)
    out[parsed["grey_mask"]] = GREY

    for mover in parsed["movers"]:
        r, c = mover["source"]
        dr, dc = mover["step"]
        pos_r, pos_c = r, c
        for _ in range(mover["tail_len"]):
            nr = pos_r + dr
            nc = pos_c + dc
            assert in_bounds(out.shape, nr, nc), "movement should stay in bounds"
            if int(out[nr, nc]) != BACKGROUND:
                break
            pos_r, pos_c = nr, nc

        parked_color = mover["color"]
        if any(
            in_bounds(out.shape, ar, ac) and int(out[ar, ac]) == GREY
            for ar, ac in neighbors4(pos_r, pos_c)
        ):
            parked_color = GREY
        out[pos_r, pos_c] = int(parked_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored source toward the grey magnet by exactly its tail length, with earlier parked pixels blocking later ones.

    Input structure:
    - The background is black and contains exactly one solid grey rectangle acting as a magnet.
    - Several non-grey source colors each appear as a single pixel outside the rectangle.
    - Every source touches one 4-connected tail of one shared tail color; the tail can bend arbitrarily and is not aligned with the magnet.

    Rule:
    - For each source, ignore the tail shape except for its pixel count; erase all tails in the output.
    - Move the source in the unique cardinal direction that most directly approaches the grey rectangle, taking exactly as many steps as its tail length.
    - Process sources from smaller rectangle-gap distance to larger; if the next step would hit the grey rectangle or any earlier parked pixel, stop immediately before the collision.
    - After stopping, recolor the source grey iff it is 4-neighbor adjacent to the current grey mass; otherwise keep its original color.

    Key insight:
    - The tail is an odometer, not a path: direction comes from the rectangle, travel distance comes from tail length, and earlier parked pixels can shorten later moves.
    """
    parsed = _parse_case(np.asarray(grid, dtype=int))
    return _solve_from_parsed(parsed)


def _expanded_rect_cells(shape: tuple[int, int], rect: tuple[int, int, int, int], pad: int) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = rect
    cells = set()
    for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            cells.add((r, c))
    return cells


def _tail_is_bent(path: list[tuple[int, int]]) -> bool:
    if len(path) < 3:
        return False
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    return len(set(dirs)) > 1


def _build_tail(
    shape: tuple[int, int],
    source: tuple[int, int],
    move_step: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    all_sources: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    sr, sc = source
    outward = (-move_step[0], -move_step[1])
    first_moves = []
    for nr, nc in neighbors4(sr, sc):
        if not in_bounds(shape, nr, nc):
            continue
        step = (nr - sr, nc - sc)
        if step == move_step:
            continue
        first_moves.append((nr, nc, step == outward))
    random.shuffle(first_moves)
    first_moves.sort(key=lambda item: 0 if item[2] else 1)

    def valid(cell: tuple[int, int], path: list[tuple[int, int]], prev: tuple[int, int] | None) -> bool:
        r, c = cell
        if not in_bounds(shape, r, c):
            return False
        if cell in forbidden or cell in occupied:
            return False
        if cell in path:
            return False
        if prev is None:
            if abs(r - sr) + abs(c - sc) != 1:
                return False
        else:
            if abs(r - prev[0]) + abs(c - prev[1]) != 1:
                return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in all_sources and (nr, nc) != source:
                return False
            if (nr, nc) == source and prev is not None:
                return False
            if (nr, nc) in occupied and (nr, nc) not in {prev, source}:
                return False
            if (nr, nc) in path and (nr, nc) != prev:
                return False
        return True

    def dfs(path: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            return path.copy()
        last = path[-1]
        moves = list(neighbors4(*last))
        random.shuffle(moves)
        scored = []
        for nxt in moves:
            if not valid(nxt, path, last):
                continue
            dr = nxt[0] - last[0]
            dc = nxt[1] - last[1]
            turn_penalty = 0 if len(path) == 1 or (dr, dc) != (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]) else 1
            scored.append((turn_penalty, random.random(), nxt))
        scored.sort()
        for _, _, nxt in scored:
            path.append(nxt)
            found = dfs(path)
            if found is not None:
                return found
            path.pop()
        return None

    for nr, nc, _preferred in first_moves:
        first = (nr, nc)
        if not valid(first, [], None):
            continue
        if length == 1:
            return [first]
        found = dfs([first])
        if found is not None:
            return found
    return None


def _build_expected_from_plan(shape: tuple[int, int], rect: tuple[int, int, int, int], movers: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for mover in sorted(movers, key=lambda m: (m["distance"], m["source"][0], m["source"][1])):
        r, c = mover["final"]
        out[r, c] = int(mover["final_color"])
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(22, 28)
    w = random.randint(22, 28)
    rect_h = random.randint(4, 6)
    rect_w = random.randint(4, 6)
    rect_r0 = random.randint(7, h - rect_h - 8)
    rect_c0 = random.randint(7, w - rect_w - 8)
    rect = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    side_a, side_b = random.sample(SIDES, 2)
    max_dist = {
        "top": r0 - 1,
        "bottom": h - r1 - 2,
        "left": c0 - 1,
        "right": w - c1 - 2,
    }
    max_dist = {side: min(val, 7) for side, val in max_dist.items()}
    if max_dist[side_a] < 4 or max_dist[side_b] < 3:
        return None

    for _ in range(200):
        dc1 = random.randint(3, max_dist[side_a] - 1)
        dc2 = random.randint(dc1 + 1, max_dist[side_a])
        dg1 = random.randint(2, max_dist[side_b] - 1)
        dg2 = random.randint(dg1 + 1, max_dist[side_b])
        if len({dc1, dc2, dg1, dg2}) == 4:
            break
    else:
        return None

    lane_a = random.randint(c0, c1) if side_a in {"top", "bottom"} else random.randint(r0, r1)
    lane_b = random.randint(c0, c1) if side_b in {"top", "bottom"} else random.randint(r0, r1)
    if side_a in {"top", "bottom"} and side_b in {"top", "bottom"} and rect_w >= 2:
        while lane_b == lane_a:
            lane_b = random.randint(c0, c1)
    if side_a in {"left", "right"} and side_b in {"left", "right"} and rect_h >= 2:
        while lane_b == lane_a:
            lane_b = random.randint(r0, r1)

    tail_color = random.choice([c for c in range(1, 10) if c != GREY])
    source_palette = [c for c in range(1, 10) if c not in {GREY, tail_color}]
    source_colors = random.sample(source_palette, 4)

    g = random.randint(2, min(4, dc1 - 1))
    lc1 = dc1 - g
    lc2 = random.randint(dc2 - g, dc2 + 2)
    lg1 = dg1
    lg2 = random.randint(dg2, dg2 + 2)

    movers = [
        {
            "role": "colored_blocker",
            "side": side_a,
            "lane": lane_a,
            "distance": dc1,
            "tail_len": lc1,
            "source_color": source_colors[0],
        },
        {
            "role": "blocked_by_color",
            "side": side_a,
            "lane": lane_a,
            "distance": dc2,
            "tail_len": lc2,
            "source_color": source_colors[1],
        },
        {
            "role": "grey_maker",
            "side": side_b,
            "lane": lane_b,
            "distance": dg1,
            "tail_len": lg1,
            "source_color": source_colors[2],
        },
        {
            "role": "blocked_by_grey",
            "side": side_b,
            "lane": lane_b,
            "distance": dg2,
            "tail_len": lg2,
            "source_color": source_colors[3],
        },
    ]

    occupied = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    source_cells = set()
    for mover in movers:
        mover["step"] = _step_for_side(mover["side"])
        mover["source"] = _source_coord(rect, mover["side"], mover["lane"], mover["distance"])
        if mover["source"] in occupied:
            return None
        if not in_bounds(grid.shape, *mover["source"]):
            return None
        for other in source_cells:
            if abs(other[0] - mover["source"][0]) + abs(other[1] - mover["source"][1]) <= 1:
                return None
        source_cells.add(mover["source"])
        occupied.add(mover["source"])
        sr, sc = mover["source"]
        grid[sr, sc] = mover["source_color"]

    forbidden = _expanded_rect_cells(grid.shape, rect, 1)
    all_tail_cells: set[tuple[int, int]] = set()
    bent_count = 0
    for mover in movers:
        path = _build_tail(
            grid.shape,
            mover["source"],
            mover["step"],
            mover["tail_len"],
            occupied=occupied,
            all_sources=source_cells,
            forbidden=forbidden,
        )
        if path is None:
            return None
        mover["tail_path"] = path
        bent_count += int(_tail_is_bent(path))
        for cell in path:
            if cell in all_tail_cells:
                return None
            all_tail_cells.add(cell)
            occupied.add(cell)
            grid[cell] = tail_color

    if bent_count == 0:
        return None

    for mover in movers:
        sr, sc = mover["source"]
        dr, dc = mover["step"]
        if mover["role"] == "colored_blocker":
            mover["final"] = (sr + dr * mover["tail_len"], sc + dc * mover["tail_len"])
            mover["final_color"] = mover["source_color"]
        elif mover["role"] == "blocked_by_color":
            partner = next(m for m in movers if m["role"] == "colored_blocker")
            mover["final"] = (partner["final"][0] - dr, partner["final"][1] - dc)
            mover["final_color"] = mover["source_color"]
        elif mover["role"] == "grey_maker":
            mover["final"] = (sr + dr * mover["tail_len"], sc + dc * mover["tail_len"])
            mover["final_color"] = GREY
        else:
            partner = next(m for m in movers if m["role"] == "grey_maker")
            mover["final"] = (partner["final"][0] - dr, partner["final"][1] - dc)
            mover["final_color"] = GREY

    final_cells = [m["final"] for m in movers]
    if len(set(final_cells)) != len(final_cells):
        return None
    for cell in final_cells:
        if not in_bounds(grid.shape, *cell):
            return None

    expected = _build_expected_from_plan(grid.shape, rect, movers)

    try:
        parsed = _parse_case(grid)
        actual = _solve_from_parsed(parsed)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    colors_after = {m["role"]: int(expected[m["final"]]) for m in movers}
    if colors_after["colored_blocker"] == GREY or colors_after["blocked_by_color"] == GREY:
        return None
    if colors_after["grey_maker"] != GREY or colors_after["blocked_by_grey"] != GREY:
        return None

    parsed_roles = {m["color"]: m for m in parsed["movers"]}
    if not any(m["tail_len"] < m["distance"] for m in parsed["movers"]):
        return None
    if not any(m["tail_len"] == m["distance"] for m in parsed["movers"]):
        return None
    if len(parsed_roles) != 4:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _generate_case()
        if case is not None:
            grid, _expected = case
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
