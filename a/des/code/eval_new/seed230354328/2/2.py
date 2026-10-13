from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
RIGHT_OF = {
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
}
PALETTE = tuple(range(1, 10))

GEN_RETRIES = 500
CHAIN_RETRIES = 180
GRID_H_RANGE = (14, 25)
GRID_W_RANGE = (14, 25)
CHAIN_COUNT_RANGE = (2, 4)
CHAIN_LEN_RANGE = (1, 4)
RELAY_GAP_RANGE = (2, 6)
BOUNDARY_STEPS_RANGE = (2, 7)
OBSTACLE_GAP_RANGE = (2, 6)


Coord = tuple[int, int]
Arrow = dict[str, object]


def _add(a: Coord, b: Coord, k: int = 1) -> Coord:
    return int(a[0] + b[0] * k), int(a[1] + b[1] * k)


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    return 0 <= int(cell[0]) < int(shape[0]) and 0 <= int(cell[1]) < int(shape[1])


def _neighbors4(cell: Coord) -> tuple[Coord, Coord, Coord, Coord]:
    r, c = map(int, cell)
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _dilate4(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    out: set[Coord] = set()
    for cell in cells:
        if _in_bounds(shape, cell):
            out.add(cell)
        for nbr in _neighbors4(cell):
            if _in_bounds(shape, nbr):
                out.add(nbr)
    return out


def _components_by_color(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue: deque[Coord] = deque([(r, c)])
            cells: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4((int(rr), int(cc))):
                    if not _in_bounds(grid.shape, (nr, nc)):
                        continue
                    if bool(seen[nr, nc]) or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((int(nr), int(nc)))
            cells.sort()
            comps.append({"color": color, "cells": tuple(cells)})

    return comps


def _arrow_cells(receiver: Coord, direction: Coord) -> tuple[Coord, Coord, Coord]:
    right = RIGHT_OF[direction]
    return (
        tuple(map(int, receiver)),
        _add(receiver, direction),
        _add(receiver, right),
    )


def _arrow_geometry(receiver: Coord, direction: Coord) -> dict[str, object]:
    head = _add(receiver, direction)
    return {
        "receiver": tuple(map(int, receiver)),
        "dir": tuple(map(int, direction)),
        "cells": tuple(sorted(_arrow_cells(receiver, direction))),
        "slot": _add(receiver, direction, -1),
        "emit": _add(head, direction),
        "head": head,
        "side": _add(receiver, RIGHT_OF[direction]),
    }


def _parse_arrow(cells: tuple[Coord, ...], color: int) -> Arrow | None:
    if len(cells) != 3:
        return None
    cell_set = set(cells)
    for receiver in cells:
        for direction in DIRS4:
            geom = _arrow_geometry(receiver, direction)
            if set(geom["cells"]) == cell_set:
                out = dict(geom)
                out["color"] = int(color)
                return out
    return None


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    arrows: list[Arrow] = []
    arrow_cells: set[Coord] = set()
    other_cells: set[Coord] = set()

    for comp in _components_by_color(g):
        parsed = _parse_arrow(comp["cells"], int(comp["color"]))
        if parsed is None:
            other_cells.update(comp["cells"])
            continue
        parsed["id"] = len(arrows)
        arrows.append(parsed)
        arrow_cells.update(parsed["cells"])

    slot_to_arrow: dict[Coord, int] = {}
    for arrow in arrows:
        slot = tuple(map(int, arrow["slot"]))
        assert slot not in slot_to_arrow, "arrow slots must be unique"
        slot_to_arrow[slot] = int(arrow["id"])

    payloads: list[dict[str, object]] = []
    obstacle_cells: set[Coord] = set()
    for cell in sorted(other_cells):
        if cell in slot_to_arrow:
            payloads.append(
                {
                    "cell": tuple(map(int, cell)),
                    "color": int(g[cell]),
                    "arrow_id": int(slot_to_arrow[cell]),
                }
            )
        else:
            obstacle_cells.add(cell)

    payload_map = {tuple(item["cell"]): int(item["color"]) for item in payloads}
    return {
        "grid": g,
        "arrows": arrows,
        "slot_to_arrow": slot_to_arrow,
        "arrow_cells": arrow_cells,
        "payloads": payloads,
        "payload_map": payload_map,
        "obstacle_cells": obstacle_cells,
    }


def _follow_payload(
    base_grid: np.ndarray,
    arrows: list[Arrow],
    slot_to_arrow: dict[Coord, int],
    start_arrow_id: int,
) -> dict[str, object]:
    current_id = int(start_arrow_id)
    current_slot = tuple(map(int, arrows[current_id]["slot"]))
    seen_arrows: set[int] = set()
    relay_count = 0

    while True:
        if current_id in seen_arrows:
            return {
                "final": current_slot,
                "reason": "cycle",
                "relay_count": relay_count,
            }
        seen_arrows.add(current_id)

        arrow = arrows[current_id]
        dr, dc = map(int, arrow["dir"])
        r, c = map(int, arrow["emit"])
        last_black = current_slot

        while True:
            pos = (int(r), int(c))
            if not _in_bounds(base_grid.shape, pos):
                return {
                    "final": tuple(map(int, last_black)),
                    "reason": "boundary",
                    "relay_count": relay_count,
                }
            if pos in slot_to_arrow:
                current_id = int(slot_to_arrow[pos])
                current_slot = pos
                relay_count += 1
                break
            if int(base_grid[pos]) == BG:
                last_black = pos
                r += dr
                c += dc
                continue
            return {
                "final": tuple(map(int, last_black)),
                "reason": "obstacle",
                "relay_count": relay_count,
            }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each payload through a network of tiny L-shaped arrows and deposit it at the final stop.

    Input structure:
    - bg = black.
    - Foreground contains several monochrome 3-cell L shapes that act as arrows.
    - Each arrow is decoded geometrically: the degree-2 corner is its receiver, one branch is a 2-cell shaft, and the other branch is a 1-cell side flag that fixes the arrow's orientation.
    - Some arrows have a single non-arrow payload pixel on the cell immediately behind the receiver; other non-arrow foreground pixels are inert obstacles.

    Rule:
    - For every payload, remove it from its starting slot and launch it from the cell just beyond that arrow's 2-cell shaft.
    - Move the payload straight through black cells only.
    - If it reaches the behind-the-receiver slot of another arrow, that arrow relays the payload and launches it along the new direction.
    - If the next step would leave the grid or hit any non-background cell that is not a relay slot, deposit the payload on the last black cell reached.
    - Arrow pixels never move and the traversed path leaves no trace.

    Key insight:
    - Solving requires two layers: first decode each 3-cell elbow into a directed launcher, then simulate the resulting multi-segment relay path until the payload can no longer continue.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    g = np.asarray(scene["grid"], dtype=int)
    out = g.copy()
    for item in scene["payloads"]:
        out[tuple(item["cell"])] = BG

    for item in scene["payloads"]:
        trace = _follow_payload(
            out,
            scene["arrows"],
            scene["slot_to_arrow"],
            int(item["arrow_id"]),
        )
        out[tuple(trace["final"])] = int(item["color"])

    return out


def _ray_cells(start: Coord, direction: Coord, steps: int) -> list[Coord]:
    return [_add(start, direction, k) for k in range(int(steps))]


def _valid_arrow_for_slot(shape: tuple[int, int], slot: Coord, direction: Coord) -> Arrow | None:
    receiver = _add(slot, direction)
    geom = _arrow_geometry(receiver, direction)
    cells = set(geom["cells"])
    cells.add(tuple(map(int, slot)))
    cells.add(tuple(map(int, geom["emit"])))
    if not all(_in_bounds(shape, cell) for cell in cells):
        return None
    return geom


def _build_chain(
    shape: tuple[int, int],
    chain_len: int,
    terminal_mode: str,
    scene_nonbg: set[Coord],
    scene_halo: set[Coord],
    scene_slots: set[Coord],
    scene_reserved: set[Coord],
) -> dict[str, object] | None:
    h, w = map(int, shape)
    for _ in range(CHAIN_RETRIES):
        arrows: list[Arrow] = []
        slots: list[Coord] = []
        reserved: set[Coord] = set()
        object_cells: set[Coord] = set()
        halo = set()
        obstacle: Coord | None = None

        first_dir = random.choice(DIRS4)
        candidates: list[Coord] = []
        for r in range(h):
            for c in range(w):
                slot = (r, c)
                geom = _valid_arrow_for_slot(shape, slot, first_dir)
                if geom is None:
                    continue
                payload_cell = slot
                arrow_cells = set(geom["cells"])
                if payload_cell in arrow_cells:
                    continue
                if slot in scene_halo or slot in scene_slots or slot in scene_reserved:
                    continue
                if any(cell in scene_halo or cell in scene_reserved for cell in arrow_cells):
                    continue
                candidates.append(slot)
        if not candidates:
            continue

        slot = random.choice(candidates)
        geom = _valid_arrow_for_slot(shape, slot, first_dir)
        assert geom is not None
        object_cells.update(geom["cells"])
        object_cells.add(slot)
        halo = _dilate4(object_cells, shape)
        arrows.append(geom)
        slots.append(slot)
        ok = True

        for idx in range(chain_len - 1):
            prev = arrows[idx]
            prev_dir = tuple(map(int, prev["dir"]))
            gap = random.randint(*RELAY_GAP_RANGE)
            path_cells = _ray_cells(tuple(prev["emit"]), prev_dir, gap)
            next_slot = _add(tuple(prev["emit"]), prev_dir, gap)
            if not path_cells or not _in_bounds(shape, next_slot):
                ok = False
                break
            if any(
                cell in reserved
                or cell in object_cells
                or cell in scene_nonbg
                or cell in scene_reserved
                or cell in scene_slots
                for cell in path_cells
            ):
                ok = False
                break
            if next_slot in reserved or next_slot in scene_reserved or next_slot in scene_slots:
                ok = False
                break
            if next_slot in halo or next_slot in scene_halo:
                ok = False
                break

            direction_choices = [
                direction
                for direction in DIRS4
                if direction != (-prev_dir[0], -prev_dir[1])
            ]
            random.shuffle(direction_choices)
            placed = None
            for direction in direction_choices:
                geom = _valid_arrow_for_slot(shape, next_slot, direction)
                if geom is None:
                    continue
                arrow_cells = set(geom["cells"])
                if any(
                    cell in halo
                    or cell in scene_halo
                    or cell in reserved
                    or cell in scene_reserved
                    or cell in scene_slots
                    for cell in arrow_cells
                ):
                    continue
                placed = geom
                break
            if placed is None:
                ok = False
                break

            reserved.update(path_cells)
            object_cells.update(placed["cells"])
            halo = _dilate4(object_cells, shape)
            arrows.append(placed)
            slots.append(next_slot)

        if not ok:
            continue

        last = arrows[-1]
        last_dir = tuple(map(int, last["dir"]))
        emit = tuple(map(int, last["emit"]))

        if terminal_mode == "boundary":
            max_steps = 0
            probe = emit
            while _in_bounds(shape, probe):
                max_steps += 1
                probe = _add(probe, last_dir)
            if max_steps < BOUNDARY_STEPS_RANGE[0]:
                continue
            travel_steps = random.randint(
                BOUNDARY_STEPS_RANGE[0],
                min(BOUNDARY_STEPS_RANGE[1], max_steps),
            )
            path_cells = _ray_cells(emit, last_dir, travel_steps)
            if any(
                cell in reserved
                or cell in object_cells
                or cell in scene_nonbg
                or cell in scene_reserved
                or cell in scene_slots
                for cell in path_cells
            ):
                continue
            reserved.update(path_cells)
        elif terminal_mode == "obstacle":
            gap = random.randint(*OBSTACLE_GAP_RANGE)
            path_cells = _ray_cells(emit, last_dir, gap)
            obstacle = _add(emit, last_dir, gap)
            if not path_cells or not _in_bounds(shape, obstacle):
                continue
            if any(
                cell in reserved
                or cell in object_cells
                or cell in scene_nonbg
                or cell in scene_reserved
                or cell in scene_slots
                for cell in path_cells
            ):
                continue
            if obstacle in halo or obstacle in scene_halo or obstacle in scene_slots or obstacle in scene_reserved:
                continue
            reserved.update(path_cells)
            object_cells.add(obstacle)
            halo = _dilate4(object_cells, shape)
        else:
            raise AssertionError(f"unknown terminal mode: {terminal_mode}")

        if object_cells & scene_nonbg:
            continue
        if reserved & scene_reserved:
            continue
        if reserved & scene_slots:
            continue
        if set(slots) & scene_reserved:
            continue

        return {
            "arrows": arrows,
            "slots": slots,
            "payload_slot": slots[0],
            "obstacle": obstacle,
            "reserved": reserved,
            "terminal_mode": terminal_mode,
        }

    return None


def _paint_structure(shape: tuple[int, int], chains: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for chain in chains:
        arrow_colors = [random.choice(PALETTE) for _ in chain["arrows"]]
        payload_color = random.choice(PALETTE)
        if payload_color == arrow_colors[0]:
            payload_color = random.choice([c for c in PALETTE if c != arrow_colors[0]])

        for color, arrow in zip(arrow_colors, chain["arrows"], strict=True):
            for cell in arrow["cells"]:
                grid[cell] = int(color)

        slot = tuple(chain["payload_slot"])
        grid[slot] = int(payload_color)

        obstacle = chain["obstacle"]
        if obstacle is not None:
            obstacle_color = random.choice(PALETTE)
            grid[tuple(obstacle)] = int(obstacle_color)

    return grid


def _scene_metrics(grid: np.ndarray) -> dict[str, object] | None:
    try:
        scene = _extract_scene(grid)
    except AssertionError:
        return None
    if not scene["arrows"] or not scene["payloads"]:
        return None

    base = np.asarray(grid, dtype=int).copy()
    for payload in scene["payloads"]:
        base[tuple(payload["cell"])] = BG

    traces = []
    for payload in scene["payloads"]:
        traces.append(
            _follow_payload(
                base,
                scene["arrows"],
                scene["slot_to_arrow"],
                int(payload["arrow_id"]),
            )
        )

    return {
        "arrow_count": len(scene["arrows"]),
        "payload_count": len(scene["payloads"]),
        "reasons": [str(trace["reason"]) for trace in traces],
        "relay_counts": [int(trace["relay_count"]) for trace in traces],
        "directions": {tuple(map(int, arrow["dir"])) for arrow in scene["arrows"]},
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        shape = (h, w)

        chain_count = random.randint(*CHAIN_COUNT_RANGE)
        terminals = ["boundary", "obstacle"]
        while len(terminals) < chain_count:
            terminals.append(random.choice(("boundary", "obstacle")))
        random.shuffle(terminals)

        lengths = [random.randint(*CHAIN_LEN_RANGE) for _ in range(chain_count)]
        lengths[0] = max(2, lengths[0])
        if max(lengths) < 2:
            lengths[random.randrange(chain_count)] = 2

        scene_nonbg: set[Coord] = set()
        scene_slots: set[Coord] = set()
        scene_reserved: set[Coord] = set()
        scene_halo: set[Coord] = set()
        chains: list[dict[str, object]] = []

        ok = True
        for terminal_mode, chain_len in zip(terminals, lengths, strict=True):
            chain = _build_chain(
                shape,
                int(chain_len),
                str(terminal_mode),
                scene_nonbg,
                scene_halo,
                scene_slots,
                scene_reserved,
            )
            if chain is None:
                ok = False
                break

            chain_nonbg: set[Coord] = set()
            for arrow in chain["arrows"]:
                chain_nonbg.update(arrow["cells"])
            chain_nonbg.add(tuple(chain["payload_slot"]))
            if chain["obstacle"] is not None:
                chain_nonbg.add(tuple(chain["obstacle"]))

            scene_nonbg.update(chain_nonbg)
            scene_slots.update(tuple(slot) for slot in chain["slots"])
            scene_reserved.update(tuple(cell) for cell in chain["reserved"])
            scene_halo = _dilate4(scene_nonbg, shape)
            chains.append(chain)

        if not ok:
            continue

        grid = _paint_structure(shape, chains)
        metrics = _scene_metrics(grid)
        if metrics is None:
            continue
        if metrics["payload_count"] != chain_count:
            continue
        if "boundary" not in metrics["reasons"] or "obstacle" not in metrics["reasons"]:
            continue
        if max(metrics["relay_counts"]) < 1:
            continue
        if len(metrics["directions"]) < 2:
            continue

        out = main(grid)
        if np.array_equal(grid, out):
            continue

        return grid

    raise RuntimeError("failed to generate a valid arrow-relay scene")
