from __future__ import annotations

from collections import deque
import random

import numpy as np


# ARC colors with fixed visible roles in this puzzle.
BLACK = 0
GREY = 5
ORANGE = 7
SOURCE_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], p: Coord) -> bool:
    r, c = p
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[set[Coord]]:
    """Return 4-connected components using the common ARC (row, col) convention."""
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result: list[set[Coord]] = []
    while unseen:
        seed = unseen.pop()
        comp = {seed}
        queue = [seed]
        while queue:
            r, c = queue.pop()
            for dr, dc in CARDINALS:
                nxt = (r + dr, c + dc)
                if nxt in unseen:
                    unseen.remove(nxt)
                    comp.add(nxt)
                    queue.append(nxt)
        result.append(comp)
    return result


def _parse(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert ORANGE in g and GREY in g and BLACK in g
    assert set(map(int, np.unique(g))) <= {BLACK, GREY, ORANGE, *SOURCE_COLORS}

    wire = {tuple(map(int, p)) for p in np.argwhere(g == GREY)}
    lamp_components = _components(g == BLACK)
    lamps: list[set[Coord]] = []
    for comp in lamp_components:
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        assert r1 == r0 + 1 and c1 == c0 + 1 and comp == expected, (
            "every black object must be a solid 2x2 lamp"
        )
        lamps.append(comp)

    source_cells = [
        (int(r), int(c))
        for r, c in np.argwhere(~np.isin(g, (BLACK, GREY, ORANGE)))
    ]
    assert 1 <= len(source_cells) <= 2, "expected one or two singleton source pixels"
    assert all(
        max(abs(r - rr), abs(c - cc)) > 1
        for i, (r, c) in enumerate(source_cells)
        for rr, cc in source_cells[i + 1 :]
    )

    entities: list[dict] = []
    for p in source_cells:
        entities.append(
            {"kind": "source", "cells": {p}, "color": int(g[p]), "ports": []}
        )
    for comp in lamps:
        entities.append({"kind": "lamp", "cells": comp, "color": None, "ports": []})

    # A port records (entity corner, first grey cell, direction entity->wire).
    for eid, entity in enumerate(entities):
        if entity["kind"] == "source":
            r, c = next(iter(entity["cells"]))
            candidates = [
                ((r, c), (r + dr, c + dc), (dr, dc))
                for dr, dc in DIAGONALS
                if (r + dr, c + dc) in wire
            ]
        else:
            rows = [r for r, _c in entity["cells"]]
            cols = [c for _r, c in entity["cells"]]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            candidates = [
                ((r0, c0), (r0 - 1, c0 - 1), (-1, -1)),
                ((r0, c1), (r0 - 1, c1 + 1), (-1, 1)),
                ((r1, c0), (r1 + 1, c0 - 1), (1, -1)),
                ((r1, c1), (r1 + 1, c1 + 1), (1, 1)),
            ]
            candidates = [port for port in candidates if port[1] in wire]

        assert candidates, f"entity {eid} must touch at least one diagonal wire endpoint"
        entity["ports"] = candidates

    ports_at_wire: dict[Coord, list[tuple[int, int, Coord, Coord]]] = {}
    for eid, entity in enumerate(entities):
        for pidx, (corner, first, direction) in enumerate(entity["ports"]):
            ports_at_wire.setdefault(first, []).append((eid, pidx, corner, direction))

    return {
        "grid": g,
        "wire": wire,
        "entities": entities,
        "ports_at_wire": ports_at_wire,
    }


def _trace(parsed: dict, origin_eid: int, origin_pidx: int) -> dict:
    """Follow one oriented wire, preserving direction through X crossings."""
    wire: set[Coord] = parsed["wire"]
    entities: list[dict] = parsed["entities"]
    ports_at_wire = parsed["ports_at_wire"]
    corner, current, incoming = entities[origin_eid]["ports"][origin_pidx]
    previous = corner
    path: list[Coord] = []
    seen: set[tuple[Coord, Coord]] = set()

    while True:
        state = (current, incoming)
        if state in seen:
            return {"target": None, "path": path, "status": "cycle"}
        seen.add(state)
        path.append(current)

        straight_cell = (current[0] + incoming[0], current[1] + incoming[1])
        if straight_cell in wire:
            previous, current = current, straight_cell
            continue

        straight_terminals = []
        for eid, pidx, terminal_corner, _port_dir in ports_at_wire.get(current, []):
            if (eid, pidx) == (origin_eid, origin_pidx):
                continue
            if terminal_corner == straight_cell:
                straight_terminals.append(eid)
        assert len(straight_terminals) <= 1, "a wire endpoint cannot name two entities"
        if straight_terminals:
            target = straight_terminals[0]
            status = "lamp" if entities[target]["kind"] == "lamp" else "source"
            return {"target": target if status == "lamp" else None, "path": path, "status": status}

        turns: list[tuple[str, Coord, int | None]] = []
        for dr, dc in DIAGONALS:
            nxt = (current[0] + dr, current[1] + dc)
            if nxt in wire and nxt != previous:
                turns.append(("wire", (dr, dc), None))

        backward = (-incoming[0], -incoming[1])
        for eid, pidx, terminal_corner, _port_dir in ports_at_wire.get(current, []):
            if (eid, pidx) == (origin_eid, origin_pidx):
                continue
            outgoing = (terminal_corner[0] - current[0], terminal_corner[1] - current[1])
            if outgoing != backward:
                turns.append(("entity", outgoing, eid))

        # Duplicate geometric continuations would make the visible clue ambiguous.
        unique_turns = {(kind, direction, eid) for kind, direction, eid in turns}
        assert len(unique_turns) == len(turns)
        if len(turns) >= 2:
            return {"target": None, "path": path, "status": "branch"}
        if not turns:
            return {"target": None, "path": path, "status": "dead"}

        kind, direction, eid = turns[0]
        if kind == "entity":
            assert eid is not None
            status = "lamp" if entities[eid]["kind"] == "lamp" else "source"
            return {"target": eid if status == "lamp" else None, "path": path, "status": status}

        previous = current
        current = (current[0] + direction[0], current[1] + direction[1])
        incoming = direction


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    parsed = _parse(grid)
    entities: list[dict] = parsed["entities"]
    queue: deque[tuple[int, int]] = deque()
    lit: dict[int, int] = {}
    for eid, entity in enumerate(entities):
        if entity["kind"] == "source":
            queue.append((eid, int(entity["color"])))

    painted: dict[Coord, int] = {}
    processed: set[tuple[int, int, int]] = set()
    records: list[dict] = []

    while queue:
        eid, color = queue.popleft()
        for pidx in range(len(entities[eid]["ports"])):
            key = (eid, pidx, color)
            if key in processed:
                continue
            processed.add(key)
            result = _trace(parsed, eid, pidx)
            record = {
                **result,
                "origin": eid,
                "origin_kind": entities[eid]["kind"],
                "color": color,
                "became_lit": False,
            }
            target = result["target"]
            if result["status"] != "lamp" or target is None:
                records.append(record)
                continue

            if target in lit:
                assert lit[target] == color, "two source colors cannot claim the same lamp"
            else:
                lit[target] = color
                queue.append((target, color))
                record["became_lit"] = True

            for cell in result["path"]:
                assert cell not in painted or painted[cell] == color, (
                    "shared powered crossing has conflicting colors"
                )
                painted[cell] = color
            for cell in entities[target]["cells"]:
                assert cell not in painted or painted[cell] == color
                painted[cell] = color
            records.append(record)

    out = parsed["grid"].copy()
    for (r, c), color in painted.items():
        out[r, c] = color
    return out, {"parsed": parsed, "lit": lit, "painted": painted, "records": records}


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate each colored source through directional grey diagonal wires and light every reachable relay lamp.

    Input structure:
    - An orange background contains one-cell-wide grey diagonal wires, one or two
      singleton colored source pixels, and several solid black 2x2 lamp blocks.
    - Wire endpoints touch a source pixel or an outward corner of a lamp diagonally;
      two wires may share one grey cell as a visible X crossing.

    Rule:
    - Trace from every source while retaining the incoming diagonal direction.  A
      straight continuation wins at an X; only when straight is absent may the
      trace take its unique turn.  Two or more turns form a true branch and stop it.
    - A successful trace recolors its grey path and terminal 2x2 lamp with the
      source color.  Each newly lit lamp immediately repeats the rule from all its
      other ports, so power continues through any number of relay stages.

    Key insight:
    - Connectivity alone is insufficient: direction separates crossing wires,
      while lamp blocks change from targets into new sources.  Missing either the
      straight-through crossing rule or relay propagation changes downstream lamps.
    """
    out, _details = _solve_details(grid)
    return out


def _lamp(top: int, left: int) -> set[Coord]:
    return {(top + dr, left + dc) for dr in (0, 1) for dc in (0, 1)}


def _has_turn(path: list[Coord]) -> bool:
    directions = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    return any(a != b for a, b in zip(directions, directions[1:]))


def _is_x_cell(wire: set[Coord], p: Coord) -> bool:
    r, c = p
    return all((r + dr, c + dc) in wire for dr, dc in DIAGONALS)


def _transform(grid: np.ndarray, turns: int, mirror: bool) -> np.ndarray:
    result = np.rot90(grid, turns)
    if mirror:
        result = np.fliplr(result)
    return result.copy()


def _build_candidate() -> tuple[np.ndarray, np.ndarray]:
    # Parameters alter lengths, crossing position, source count, palette, margins,
    # and global orientation while preserving the same visible causal grammar.
    extension = random.randint(0, 1)
    crossing_shift = random.randint(0, 2)
    first_source = (random.choice((1, 2)),) * 2
    two_sources = random.random() < 0.5
    colors = random.sample(SOURCE_COLORS, 2 if two_sources else 1)
    primary_color = colors[0]

    source_a = first_source
    path_a = {(t, t) for t in range(first_source[0] + 1, 6)} | {(6, 4), (7, 3)}
    lamp_a = _lamp(8, 1)

    path_b = {(10 + t, 3 + t) for t in range(8 + extension)}
    lamp_b = _lamp(18 + extension, 11 + extension)

    xr, xc = 13 + crossing_shift, 6 + crossing_shift
    crossing = (xr, xc)
    decoy = {(xr + t, xc - t) for t in range(-3, 4)}
    upper_terminal = (xr - 4, xc + 4)
    lower_terminal = (xr + 4, xc - 4)
    lamp_cross_1 = _lamp(upper_terminal[0] - 1, upper_terminal[1])
    lamp_cross_2 = _lamp(lower_terminal[0], lower_terminal[1] - 1)

    branch = {
        *((19 + extension + t, 12 + extension + t) for t in range(1, 5)),
        (22 + extension, 17 + extension),
        (21 + extension, 18 + extension),
        (20 + extension, 19 + extension),
        (24 + extension, 15 + extension),
        (25 + extension, 14 + extension),
    }
    branch_node = (23 + extension, 16 + extension)
    lamp_branch_1 = _lamp(18 + extension, 20 + extension)
    lamp_branch_2 = _lamp(26 + extension, 12 + extension)

    sources: list[tuple[Coord, int]] = [(source_a, primary_color)]
    lamps = [
        lamp_a,
        lamp_b,
        lamp_cross_1,
        lamp_cross_2,
        lamp_branch_1,
        lamp_branch_2,
    ]
    powered_wire: list[tuple[set[Coord], int]] = [
        (path_a, primary_color),
        (path_b, primary_color),
    ]
    powered_lamps: list[tuple[set[Coord], int]] = [
        (lamp_a, primary_color),
        (lamp_b, primary_color),
    ]

    extra_path: set[Coord] = set()
    if two_sources:
        source_b = (2, 27)
        extra_path = {(3, 26), (4, 25), (5, 24), (6, 23)}
        lamp_extra = _lamp(7, 21)
        sources.append((source_b, colors[1]))
        lamps.append(lamp_extra)
        powered_wire.append((extra_path, colors[1]))
        powered_lamps.append((lamp_extra, colors[1]))

    all_wire = path_a | path_b | decoy | branch | extra_path
    all_lamp_cells = set().union(*lamps)
    source_cells = {p for p, _color in sources}
    assert crossing in path_b and crossing in decoy
    assert _is_x_cell(all_wire, crossing)
    assert branch_node in branch
    assert (branch_node[0] + 1, branch_node[1] + 1) not in all_wire
    assert {
        (branch_node[0] - 1, branch_node[1] + 1),
        (branch_node[0] + 1, branch_node[1] - 1),
    } <= all_wire
    assert not (all_lamp_cells & all_wire)
    assert not (source_cells & (all_wire | all_lamp_cells))
    assert sum(len(lamp) for lamp in lamps) == len(all_lamp_cells)

    occupied = all_wire | all_lamp_cells | source_cells
    min_r = min(r for r, _c in occupied)
    max_r = max(r for r, _c in occupied)
    min_c = min(c for _r, c in occupied)
    max_c = max(c for _r, c in occupied)
    body_height = max_r - min_r + 1
    body_width = max_c - min_c + 1
    vertical_margins = [
        (top, bottom)
        for top in (1, 2)
        for bottom in (1, 2)
        if body_height + top + bottom <= 30
    ]
    horizontal_margins = [
        (left, right)
        for left in (1, 2)
        for right in (1, 2)
        if body_width + left + right <= 30
    ]
    assert vertical_margins and horizontal_margins
    top, bottom = random.choice(vertical_margins)
    left, right = random.choice(horizontal_margins)
    height = body_height + top + bottom
    width = body_width + left + right

    grid = np.full((height, width), ORANGE, dtype=int)
    expected = grid.copy()

    def local(p: Coord) -> Coord:
        return p[0] - min_r + top, p[1] - min_c + left

    for p in all_wire:
        grid[local(p)] = GREY
        expected[local(p)] = GREY
    for lamp in lamps:
        for p in lamp:
            grid[local(p)] = BLACK
            expected[local(p)] = BLACK
    for p, color in sources:
        grid[local(p)] = color
        expected[local(p)] = color
    for cells, color in powered_wire + powered_lamps:
        for p in cells:
            expected[local(p)] = color

    # Metadata-level semantic checks exclude the three tempting shortcuts.
    assert expected[local(next(iter(lamp_b)))] == primary_color  # requires relay A
    assert expected[local(crossing)] == primary_color  # straight through the X
    assert expected[local(next(iter(lamp_branch_1)))] == BLACK  # branch blocks
    assert expected[local(next(iter(lamp_branch_2)))] == BLACK
    assert all(expected[local(p)] == GREY for p in branch)
    assert any(expected[local(p)] == GREY for p in decoy - {crossing})

    turns = random.randrange(4)
    mirror = bool(random.getrandbits(1))
    grid = _transform(grid, turns, mirror)
    expected = _transform(expected, turns, mirror)
    return grid, expected


def _semantic_accepts(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual, details = _solve_details(grid)
    except (AssertionError, IndexError, ValueError):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return False

    parsed = details["parsed"]
    entities = parsed["entities"]
    source_count = sum(entity["kind"] == "source" for entity in entities)
    lamp_count = sum(entity["kind"] == "lamp" for entity in entities)
    if source_count not in (1, 2) or lamp_count != 5 + source_count:
        return False
    if len(details["lit"]) != source_count + 1:
        return False

    records = details["records"]
    # At least one lamp is reached only after another lamp becomes a source.
    if not any(r["origin_kind"] == "lamp" and r["became_lit"] for r in records):
        return False
    # The successful relay path must pass a true four-arm X and keep direction.
    if not any(
        r["status"] == "lamp"
        and any(_is_x_cell(parsed["wire"], p) for p in r["path"])
        for r in records
    ):
        return False
    # A different lit-lamp port must actually terminate at a true branch.
    if not any(r["origin_kind"] == "lamp" and r["status"] == "branch" for r in records):
        return False
    # The unique-turn rule is also exercised by a successful bent path.
    if not any(r["status"] == "lamp" and _has_turn(r["path"]) for r in records):
        return False

    unlit_lamps = [
        eid
        for eid, entity in enumerate(entities)
        if entity["kind"] == "lamp" and eid not in details["lit"]
    ]
    if len(unlit_lamps) < 4:
        return False
    if any(actual[cell] != BLACK for eid in unlit_lamps for cell in entities[eid]["cells"]):
        return False
    return True


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked diagonal relay circuit."""
    for _ in range(100):
        grid, expected = _build_candidate()
        if _semantic_accepts(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid diagonal relay puzzle")
