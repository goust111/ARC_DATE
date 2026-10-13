import random
from collections import deque

import numpy as np


BG = 7
GREY = 5
SQUARE_COLORS = [1, 2, 3, 4, 6, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# For a square at top-left (r0, c0), the attached grey endpoint is at
# (r0 + off_r, c0 + off_c), and the launch direction points from the square
# through that endpoint and along the chain.
ATTACH_OFFSETS = {
    (-1, -1): (-1, -1),
    (-1, 1): (-1, 2),
    (1, -1): (2, -1),
    (1, 1): (2, 2),
}
SQUARE_FROM_ENDPOINT = {
    (-1, -1): (1, 1),
    (-1, 1): (1, -2),
    (1, -1): (-2, 1),
    (1, 1): (-2, -2),
}
BAR_DIRECTIONS = {
    (-1, -1): [(-1, 0), (0, -1)],
    (-1, 1): [(-1, 0), (0, 1)],
    (1, -1): [(1, 0), (0, -1)],
    (1, 1): [(1, 0), (0, 1)],
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    return DIR4 if connectivity == 4 else DIR8


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in _neighbors(rr, cc, connectivity):
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_monochrome_components(grid: np.ndarray, bg: int, connectivity: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color, connectivity):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _square_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _classify_grey_region(coords: np.ndarray) -> dict:
    pts = {tuple(map(int, rc)) for rc in coords}
    region = {"cells": pts, "kind": "solid"}
    if len(pts) < 2:
        return region

    for r, c in pts:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in pts:
                return region

    diffs = {r - c for r, c in pts}
    if len(diffs) == 1:
        ordered = sorted(pts)
        if all(ordered[i + 1] == (ordered[i][0] + 1, ordered[i][1] + 1) for i in range(len(ordered) - 1)):
            return {
                "cells": pts,
                "kind": "chain",
                "endpoints": (ordered[0], ordered[-1]),
                "axis_step": (1, 1),
            }

    sums = {r + c for r, c in pts}
    if len(sums) == 1:
        ordered = sorted(pts, key=lambda rc: (rc[0], -rc[1]))
        if all(ordered[i + 1] == (ordered[i][0] + 1, ordered[i][1] - 1) for i in range(len(ordered) - 1)):
            return {
                "cells": pts,
                "kind": "chain",
                "endpoints": (ordered[0], ordered[-1]),
                "axis_step": (1, -1),
            }

    return region


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    grey_components = _components(grid == GREY, connectivity=8)
    grey_regions: list[dict] = []
    grey_owner: dict[tuple[int, int], int] = {}
    for idx, coords in enumerate(grey_components):
        info = _classify_grey_region(coords)
        info["index"] = idx
        grey_regions.append(info)
        for cell in info["cells"]:
            grey_owner[cell] = idx

    movers: list[dict] = []
    stayers: list[dict] = []
    for comp in _find_monochrome_components(grid, BG, connectivity=4):
        if comp["color"] == GREY:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["area"] == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "foreground objects must be solid 2x2 squares"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == comp["color"]), "colored objects must be solid 2x2 squares"

        attachments: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for launch_dir, (dr, dc) in ATTACH_OFFSETS.items():
            endpoint = (r0 + dr, c0 + dc)
            if _in_bounds(grid.shape, *endpoint) and grid[endpoint] == GREY:
                attachments.append((endpoint, launch_dir))
        assert len(attachments) == 1, "each square must diagonally touch exactly one grey endpoint"
        endpoint, launch_dir = attachments[0]
        region = grey_regions[grey_owner[endpoint]]

        info = {
            "color": int(comp["color"]),
            "top_left": (r0, c0),
            "endpoint": endpoint,
            "launch_dir": launch_dir,
            "region_cells": set(region["cells"]),
        }
        if region["kind"] == "chain":
            dr, dc = launch_dir
            if (
                endpoint in region["endpoints"]
                and (endpoint[0] + dr, endpoint[1] + dc) in region["cells"]
                and (endpoint[0] - dr, endpoint[1] - dc) not in region["cells"]
            ):
                movers.append(info)
                continue
        stayers.append(info)

    return movers, stayers


def _advance_square(
    shape: tuple[int, int],
    start: tuple[int, int],
    launch_dir: tuple[int, int],
    static_blockers: np.ndarray,
    dynamic_blockers: set[tuple[int, int]],
    transparent_region: set[tuple[int, int]],
) -> tuple[int, int]:
    r0, c0 = start
    dr, dc = launch_dir
    while True:
        nr0, nc0 = r0 + dr, c0 + dc
        next_cells = _square_cells((nr0, nc0))
        if any(not _in_bounds(shape, rr, cc) for rr, cc in next_cells):
            return r0, c0
        if any((rr, cc) in dynamic_blockers for rr, cc in next_cells):
            return r0, c0
        if any(static_blockers[rr, cc] and (rr, cc) not in transparent_region for rr, cc in next_cells):
            return r0, c0
        r0, c0 = nr0, nc0


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    movers, _ = _analyze_grid(g)
    out = g.copy()
    for mover in movers:
        for rr, cc in _square_cells(mover["top_left"]):
            out[rr, cc] = BG

    placed: set[tuple[int, int]] = set()
    for mover in sorted(movers, key=lambda item: item["top_left"]):
        static_blockers = out != BG
        final_top_left = _advance_square(
            g.shape,
            mover["top_left"],
            mover["launch_dir"],
            static_blockers,
            placed,
            mover["region_cells"],
        )
        mover["final_top_left"] = final_top_left
        cells = _square_cells(final_top_left)
        assert all(cell not in placed for cell in cells), "moving squares must end at distinct cells"
        for rr, cc in cells:
            out[rr, cc] = mover["color"]
            placed.add((rr, cc))

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Launch each 2x2 colored square only when it is attached to the end of a pure diagonal grey chain.

    Input structure:
    - The background is orange.
    - Grey regions include both straight single-pixel diagonal chains and
      ordinary 4-connected bars or blocks.
    - Every non-grey foreground object is a solid 2x2 colored square that
      diagonally touches exactly one grey pixel at one corner.

    Rule:
    - Classify each grey 8-connected region by its internal connectivity.
    - If a square touches an endpoint of a straight grey diagonal chain, move
      the whole square step by step along that chain's diagonal direction, then
      keep going in the same direction until the next step would leave the grid
      or overlap any non-background pixel other than its own chain.
    - If the touched grey region has any 4-connected edge join, the square
      stays where it is. Grey regions themselves never change.

    Key insight:
    - The decisive clue is not the grey region's size but whether it is a
      corner-only diagonal rail; that rail determines both which squares move
      and the diagonal launch direction.
    """
    return _solve(np.asarray(input_grid, dtype=int))


def _max_steps_to_edge(shape: tuple[int, int], square_top_left: tuple[int, int], launch_dir: tuple[int, int]) -> int:
    r0, c0 = square_top_left
    dr, dc = launch_dir
    row_limit = (shape[0] - 2 - r0) if dr > 0 else r0
    col_limit = (shape[1] - 2 - c0) if dc > 0 else c0
    return min(row_limit, col_limit)


def _endpoint_from_square(square_top_left: tuple[int, int], launch_dir: tuple[int, int]) -> tuple[int, int]:
    off_r, off_c = ATTACH_OFFSETS[launch_dir]
    return square_top_left[0] + off_r, square_top_left[1] + off_c


def _square_from_endpoint(endpoint: tuple[int, int], launch_dir: tuple[int, int]) -> tuple[int, int]:
    off_r, off_c = SQUARE_FROM_ENDPOINT[launch_dir]
    return endpoint[0] + off_r, endpoint[1] + off_c


def _halo(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _can_add_assembly(shape: tuple[int, int], cells: set[tuple[int, int]], occupied_halo: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, r, c) and (r, c) not in occupied_halo for r, c in cells)


def _place_assembly(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    colored_cells: list[tuple[set[tuple[int, int]], int]],
) -> None:
    all_cells: set[tuple[int, int]] = set()
    for cells, _color in colored_cells:
        all_cells |= cells
    assert _can_add_assembly(grid.shape, all_cells, occupied_halo)
    for cells, color in colored_cells:
        _paint_cells(grid, cells, color)
    occupied |= all_cells
    occupied_halo |= _halo(all_cells)


def _make_moving_chain_assembly(
    shape: tuple[int, int], stop_kind: str
) -> tuple[list[tuple[set[tuple[int, int]], int]], dict] | None:
    for _ in range(200):
        launch_dir = random.choice(DIAGS)
        square_color = random.choice(SQUARE_COLORS)
        square_top_left = (random.randint(0, shape[0] - 2), random.randint(0, shape[1] - 2))
        max_steps = _max_steps_to_edge(shape, square_top_left, launch_dir)
        if max_steps < 4:
            continue

        if stop_kind == "edge":
            travel_steps = max_steps
            max_chain_len = min(6, travel_steps - 2)
        else:
            if max_steps < 6:
                continue
            max_chain_len = min(6, max_steps - 4)
        if max_chain_len < 2:
            continue
        chain_len = random.randint(2, max_chain_len)
        if stop_kind == "edge":
            travel_steps = max_steps
        else:
            travel_steps = random.randint(chain_len + 2, max_steps - 2)

        endpoint = _endpoint_from_square(square_top_left, launch_dir)
        chain_cells = {
            (endpoint[0] + i * launch_dir[0], endpoint[1] + i * launch_dir[1])
            for i in range(chain_len)
        }
        square_cells = set(_square_cells(square_top_left))
        if any(not _in_bounds(shape, r, c) for r, c in chain_cells):
            continue

        colored_cells: list[tuple[set[tuple[int, int]], int]] = [(chain_cells, GREY), (square_cells, square_color)]
        stop_payload = None
        if stop_kind == "grey":
            blocker_top_left = (
                square_top_left[0] + (travel_steps + 2) * launch_dir[0],
                square_top_left[1] + (travel_steps + 2) * launch_dir[1],
            )
            blocker_cells = set(_square_cells(blocker_top_left))
            if any(not _in_bounds(shape, r, c) for r, c in blocker_cells):
                continue
            colored_cells.append((blocker_cells, GREY))
            stop_payload = {"grey_blocker_cells": blocker_cells}

        all_cells = set().union(*(cells for cells, _ in colored_cells))
        if len(all_cells) != sum(len(cells) for cells, _ in colored_cells):
            continue
        return (
            colored_cells,
            {
                "kind": "moving",
                "square_top_left": square_top_left,
                "square_color": square_color,
                "launch_dir": launch_dir,
                "chain_cells": chain_cells,
                "travel_steps": travel_steps,
                "stop_kind": stop_kind,
                **(stop_payload or {}),
            },
        )
    return None


def _make_stationary_square_assembly(
    shape: tuple[int, int], forced_square_top_left: tuple[int, int] | None = None
) -> tuple[list[tuple[set[tuple[int, int]], int]], dict] | None:
    for _ in range(240):
        square_color = random.choice(SQUARE_COLORS)
        square_top_left = forced_square_top_left or (
            random.randint(0, shape[0] - 2),
            random.randint(0, shape[1] - 2),
        )
        square_cells = set(_square_cells(square_top_left))
        for launch_dir in random.sample(DIAGS, len(DIAGS)):
            endpoint = _endpoint_from_square(square_top_left, launch_dir)
            if not _in_bounds(shape, *endpoint):
                continue
            for bar_dir in random.sample(BAR_DIRECTIONS[launch_dir], len(BAR_DIRECTIONS[launch_dir])):
                bar_len = random.randint(3, 5)
                bar_cells = {
                    (endpoint[0] + i * bar_dir[0], endpoint[1] + i * bar_dir[1])
                    for i in range(bar_len)
                }
                if any(not _in_bounds(shape, r, c) for r, c in bar_cells):
                    continue
                all_cells = square_cells | bar_cells
                if len(all_cells) != len(square_cells) + len(bar_cells):
                    continue
                return (
                    [(bar_cells, GREY), (square_cells, square_color)],
                    {
                        "kind": "stationary",
                        "square_top_left": square_top_left,
                        "square_color": square_color,
                        "endpoint": endpoint,
                        "bar_cells": bar_cells,
                    },
                )
        if forced_square_top_left is not None:
            break
    return None


def _make_extra_grey_region(shape: tuple[int, int]) -> tuple[list[tuple[set[tuple[int, int]], int]], dict] | None:
    for _ in range(120):
        kind = random.choice(["block", "bar", "chain"])
        if kind == "block":
            h = random.randint(2, 3)
            w = random.randint(2, 3)
            r0 = random.randint(0, shape[0] - h)
            c0 = random.randint(0, shape[1] - w)
            cells = {(r0 + dr, c0 + dc) for dr in range(h) for dc in range(w)}
        elif kind == "bar":
            length = random.randint(3, 5)
            vertical = random.random() < 0.5
            if vertical:
                r0 = random.randint(0, shape[0] - length)
                c0 = random.randint(0, shape[1] - 1)
                cells = {(r0 + i, c0) for i in range(length)}
            else:
                r0 = random.randint(0, shape[0] - 1)
                c0 = random.randint(0, shape[1] - length)
                cells = {(r0, c0 + i) for i in range(length)}
        else:
            launch_dir = random.choice(DIAGS)
            chain_len = random.randint(3, 5)
            endpoint = (random.randint(0, shape[0] - 1), random.randint(0, shape[1] - 1))
            cells = {
                (endpoint[0] + i * launch_dir[0], endpoint[1] + i * launch_dir[1])
                for i in range(chain_len)
            }
            if any(not _in_bounds(shape, r, c) for r, c in cells):
                continue
        return ([(cells, GREY)], {"kind": "grey_extra", "cells": cells})
    return None


def _construct_expected_output(grid: np.ndarray, metadata: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in metadata:
        if item["kind"] == "moving":
            _paint_cells(out, _square_cells(item["square_top_left"]), BG)
    for item in metadata:
        if item["kind"] == "moving":
            final_top_left = (
                item["square_top_left"][0] + item["travel_steps"] * item["launch_dir"][0],
                item["square_top_left"][1] + item["travel_steps"] * item["launch_dir"][1],
            )
            _paint_cells(out, _square_cells(final_top_left), item["square_color"])
    return out


def _validate_case(grid: np.ndarray, metadata: list[dict]) -> bool:
    movers, stayers = _analyze_grid(grid)
    moving_meta = [item for item in metadata if item["kind"] == "moving"]
    stationary_meta = [item for item in metadata if item["kind"] == "stationary"]
    if len(movers) != len(moving_meta) or len(stayers) != len(stationary_meta):
        return False
    if not movers or not stayers:
        return False

    expected = _construct_expected_output(grid, metadata)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False

    for item in moving_meta:
        start = item["square_top_left"]
        final_top_left = (
            start[0] + item["travel_steps"] * item["launch_dir"][0],
            start[1] + item["travel_steps"] * item["launch_dir"][1],
        )
        next_top_left = (final_top_left[0] + item["launch_dir"][0], final_top_left[1] + item["launch_dir"][1])
        next_cells = _square_cells(next_top_left)
        blocked_by_edge = any(not _in_bounds(grid.shape, r, c) for r, c in next_cells)
        static = grid.copy()
        _paint_cells(static, _square_cells(start), BG)
        blocked_by_object = any(
            _in_bounds(grid.shape, r, c)
            and static[r, c] != BG
            and (r, c) not in item["chain_cells"]
            for r, c in next_cells
        )
        if not (blocked_by_edge or blocked_by_object):
            return False

    if np.array_equal(grid, actual):
        return False
    return True


def _generate_case(return_metadata: bool = False) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    for _ in range(400):
        shape = (random.randint(18, 26), random.randint(18, 26))
        grid = np.full(shape, BG, dtype=int)
        occupied: set[tuple[int, int]] = set()
        occupied_halo: set[tuple[int, int]] = set()
        metadata: list[dict] = []

        moving_modes = [random.choice(["edge", "grey"])]
        if random.random() < 0.6:
            moving_modes.append("square")
        if random.random() < 0.4:
            moving_modes.append(random.choice(["edge", "grey"]))

        ok = True
        pending_blocker_square: tuple[int, int] | None = None
        for mode in moving_modes:
            if mode in {"edge", "grey"}:
                built = _make_moving_chain_assembly(shape, mode)
                if built is None:
                    ok = False
                    break
                colored_cells, meta = built
                all_cells = set().union(*(cells for cells, _ in colored_cells))
                if not _can_add_assembly(shape, all_cells, occupied_halo):
                    ok = False
                    break
                _place_assembly(grid, occupied, occupied_halo, colored_cells)
                metadata.append(meta)
            else:
                built = _make_moving_chain_assembly(shape, "edge")
                if built is None:
                    ok = False
                    break
                colored_cells, meta = built
                max_steps = _max_steps_to_edge(shape, meta["square_top_left"], meta["launch_dir"])
                min_steps = len(meta["chain_cells"]) + 2
                if max_steps - 2 < min_steps:
                    ok = False
                    break
                travel_steps = random.randint(min_steps, max_steps - 2)
                meta["travel_steps"] = travel_steps
                final_top_left = (
                    meta["square_top_left"][0] + travel_steps * meta["launch_dir"][0],
                    meta["square_top_left"][1] + travel_steps * meta["launch_dir"][1],
                )
                pending_blocker_square = (
                    final_top_left[0] + 2 * meta["launch_dir"][0],
                    final_top_left[1] + 2 * meta["launch_dir"][1],
                )
                meta["stop_kind"] = "square"
                all_cells = set().union(*(cells for cells, _ in colored_cells))
                if not _can_add_assembly(shape, all_cells, occupied_halo):
                    ok = False
                    break
                _place_assembly(grid, occupied, occupied_halo, colored_cells)
                metadata.append(meta)

        if not ok:
            continue

        if pending_blocker_square is not None:
            blocker = _make_stationary_square_assembly(shape, forced_square_top_left=pending_blocker_square)
            if blocker is None:
                continue
            colored_cells, meta = blocker
            all_cells = set().union(*(cells for cells, _ in colored_cells))
            if not _can_add_assembly(shape, all_cells, occupied_halo):
                continue
            _place_assembly(grid, occupied, occupied_halo, colored_cells)
            metadata.append(meta)

        if len([item for item in metadata if item["kind"] == "stationary"]) == 0:
            stationary = _make_stationary_square_assembly(shape)
            if stationary is None:
                continue
            colored_cells, meta = stationary
            all_cells = set().union(*(cells for cells, _ in colored_cells))
            if not _can_add_assembly(shape, all_cells, occupied_halo):
                continue
            _place_assembly(grid, occupied, occupied_halo, colored_cells)
            metadata.append(meta)

        for _extra in range(random.randint(0, 2)):
            extra = _make_extra_grey_region(shape)
            if extra is None:
                continue
            colored_cells, meta = extra
            all_cells = set().union(*(cells for cells, _ in colored_cells))
            if not _can_add_assembly(shape, all_cells, occupied_halo):
                continue
            _place_assembly(grid, occupied, occupied_halo, colored_cells)
            metadata.append(meta)

        if _validate_case(grid, metadata):
            if return_metadata:
                return grid, metadata
            return grid
    raise RuntimeError("failed to generate a valid 斜链弹射轨道 instance")


def generate_input() -> np.ndarray:
    return _generate_case(return_metadata=False)
