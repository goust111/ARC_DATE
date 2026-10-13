from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


BG = int(Color.BLACK)
TAIL = int(Color.GREY)
PALETTE = [int(c) for c in Color.NOT_BLACK if int(c) != TAIL]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


Coord = tuple[int, int]


@dataclass(frozen=True)
class Mover:
    pos: Coord
    color: int
    direction: Coord


@dataclass(frozen=True)
class ChainPlan:
    head: Coord
    direction: Coord
    start_color: int
    tail: tuple[Coord, Coord]
    singletons: tuple[tuple[Coord, int], ...]
    dominoes: tuple[tuple[Coord, Coord, int], ...]
    event_types: tuple[str, ...]
    used_cells: frozenset[Coord]


def _add(pos: Coord, direction: Coord, steps: int = 1) -> Coord:
    return (int(pos[0] + direction[0] * steps), int(pos[1] + direction[1] * steps))


def _is_straight_domino(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 2:
        return False
    (r0, c0), (r1, c1) = coords.tolist()
    return abs(int(r0 - r1)) + abs(int(c0 - c1)) == 1


def _sorted_domino(coords: np.ndarray) -> tuple[Coord, Coord]:
    a, b = [tuple(map(int, rc)) for rc in coords.tolist()]
    return tuple(sorted((a, b)))


def _neighbors8(pos: Coord, shape: tuple[int, int]) -> set[Coord]:
    out: set[Coord] = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = pos[0] + dr
            cc = pos[1] + dc
            if in_bounds(shape, rr, cc):
                out.add((rr, cc))
    return out


def _expand_margin(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    out: set[Coord] = set()
    for cell in cells:
        out |= _neighbors8(cell, shape)
    return out


def _perpendiculars(direction: Coord) -> list[Coord]:
    dr, dc = direction
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _parse_input(grid: np.ndarray) -> tuple[list[Mover], dict[Coord, int], dict[Coord, Coord]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    movers: list[Mover] = []
    mover_heads: set[Coord] = set()

    tail_comps = find_components(
        g, bg=BG, connectivity=4, monochromatic=True, colors=[TAIL]
    )
    for comp in tail_comps:
        coords = comp["coords"]
        assert _is_straight_domino(coords), "gray trail must be a straight 2-cell bar"
        a, b = _sorted_domino(coords)
        candidates: list[tuple[Coord, Coord]] = []
        for end, other in ((a, b), (b, a)):
            direction = (end[0] - other[0], end[1] - other[1])
            head = _add(end, direction)
            if in_bounds(g.shape, head[0], head[1]) and int(g[head]) not in (BG, TAIL):
                candidates.append((head, direction))
        assert len(candidates) == 1, "each gray trail must point to one colored head"
        head, direction = candidates[0]
        movers.append(Mover(head, int(g[head]), direction))
        mover_heads.add(head)

    singletons: dict[Coord, int] = {}
    domino_partner: dict[Coord, Coord] = {}
    for comp in find_components(g, bg=BG, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color in (BG, TAIL):
            continue
        coords = comp["coords"]
        area = int(comp["area"])
        if area == 1:
            pos = tuple(map(int, coords[0]))
            if pos not in mover_heads:
                singletons[pos] = color
            continue
        assert area == 2 and _is_straight_domino(coords), "non-mover objects must be singleton or domino"
        a, b = _sorted_domino(coords)
        domino_partner[a] = b
        domino_partner[b] = a

    assert movers, "expected at least one mover"
    return movers, singletons, domino_partner


def _simulate(
    shape: tuple[int, int],
    movers: list[Mover],
    singletons: dict[Coord, int],
    domino_partner: dict[Coord, Coord],
    base: np.ndarray | None = None,
) -> np.ndarray:
    if base is None:
        out = np.full(shape, BG, dtype=int)
        for a, b in domino_partner.items():
            out[a] = out[b] = BG
        for pos, color in singletons.items():
            out[pos] = int(color)
    else:
        out = np.asarray(base, dtype=int).copy()
        out[out == TAIL] = BG

    pending = dict(singletons)
    for mover in sorted(movers, key=lambda m: (m.pos[0], m.pos[1], m.color, m.direction)):
        r, c = mover.pos
        color = int(mover.color)
        dr, dc = mover.direction
        out[r, c] = color
        step_budget = int(np.prod(shape)) * 8
        while step_budget > 0:
            step_budget -= 1
            nr, nc = r + dr, c + dc
            if not in_bounds(shape, nr, nc):
                break
            nxt = (nr, nc)
            if nxt in pending:
                color = int(pending.pop(nxt))
                r, c = nxt
                out[r, c] = color
                continue
            if nxt in domino_partner:
                pr, pc = domino_partner[nxt]
                dr, dc = int(pr - nr), int(pc - nc)
                assert abs(dr) + abs(dc) == 1
                continue
            r, c = nxt
            out[r, c] = color
        assert step_budget > 0, "simulation loop did not terminate"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Trace each gray-tailed mover through baton-pass singletons and turning dominoes until the chain reaches the border.

    Input structure:
    - Black background contains one or more movers, each drawn as one colored head pixel with a straight 2-cell gray tail directly behind it.
    - Other colored objects are either isolated single pixels or 2-cell horizontal/vertical domino bars.
    - Collision chains are spatially disjoint, so each mover has an unambiguous route.

    Rule:
    - A mover slides in the direction opposite its gray tail, painting every visited free cell with its own color.
    - When the next cell is an isolated single pixel, the current mover stops just before it and that pixel becomes the new mover, continuing in the same direction with its own color.
    - When the next cell is one end of a domino, the mover does not enter the domino; instead it turns so that its new direction is from the hit domino cell toward the other domino cell, while the domino stays in place.
    - Gray tails are erased in the output, dominoes remain, and each chain keeps reacting until the last active mover reaches an edge cell.

    Key insight:
    - Singleton pixels are momentum handoff targets while dominoes are fixed direction selectors, so the output is the union of piecewise-straight painted trails produced by repeated pass-through and turn events.
    """
    g = np.asarray(grid, dtype=int)
    movers, singletons, domino_partner = _parse_input(g)
    return _simulate(g.shape, movers, singletons, domino_partner, base=g)


def _pick_distinct_colors(count: int) -> list[int]:
    assert count <= len(PALETTE)
    return random.sample(PALETTE, count)


def _choice_excluding(*forbidden: int) -> int:
    options = [c for c in PALETTE if c not in set(map(int, forbidden))]
    assert options
    return int(random.choice(options))


def _plan_chain(
    shape: tuple[int, int],
    blocked: set[Coord],
    event_types: tuple[str, ...],
    carrier_colors: list[int],
) -> ChainPlan | None:
    height, width = shape
    assert 1 + event_types.count("handoff") == len(carrier_colors)

    start_cells = [
        (r, c)
        for r in range(2, height - 2)
        for c in range(2, width - 2)
    ]
    random.shuffle(start_cells)

    for head in start_cells:
        for direction in random.sample(DIRS, k=len(DIRS)):
            tail1 = _add(head, direction, -1)
            tail2 = _add(head, direction, -2)
            initial = {head, tail1, tail2}
            if any(not in_bounds(shape, r, c) for r, c in initial):
                continue
            if initial & blocked:
                continue

            local_used = set(initial)
            singletons: list[tuple[Coord, int]] = []
            dominoes: list[tuple[Coord, Coord, int]] = []
            current = head
            current_color = int(carrier_colors[0])
            current_direction = direction
            carrier_idx = 1
            ok = True

            for event in event_types:
                lengths = list(range(1, 9))
                random.shuffle(lengths)
                placed = False
                for run_len in lengths:
                    trail = [_add(current, current_direction, k) for k in range(1, run_len + 1)]
                    if any(not in_bounds(shape, r, c) for r, c in trail):
                        continue
                    if set(trail) & (blocked | local_used):
                        continue
                    stop = trail[-1]

                    if event == "handoff":
                        obstacle = _add(stop, current_direction)
                        if not in_bounds(shape, obstacle[0], obstacle[1]):
                            continue
                        if obstacle in blocked or obstacle in local_used:
                            continue
                        singletons.append((obstacle, int(carrier_colors[carrier_idx])))
                        carrier_idx += 1
                        local_used.update(trail)
                        local_used.add(obstacle)
                        current = obstacle
                        current_color = singletons[-1][1]
                        placed = True
                        break

                    for new_direction in random.sample(_perpendiculars(current_direction), k=2):
                        impact = _add(stop, current_direction)
                        partner = _add(impact, new_direction)
                        exit_cell = _add(stop, new_direction)
                        needed = {impact, partner, exit_cell}
                        if any(not in_bounds(shape, r, c) for r, c in needed):
                            continue
                        if needed & (blocked | local_used):
                            continue
                        if len({impact, partner, exit_cell, stop}) != 4:
                            continue
                        dom_color = _choice_excluding(current_color)
                        dominoes.append((impact, partner, dom_color))
                        local_used.update(trail)
                        local_used.add(impact)
                        local_used.add(partner)
                        current = stop
                        current_direction = new_direction
                        placed = True
                        break
                    if placed:
                        break

                if not placed:
                    ok = False
                    break

            if not ok:
                continue

            final_cells: list[Coord] = []
            cursor = current
            while True:
                nxt = _add(cursor, current_direction)
                if not in_bounds(shape, nxt[0], nxt[1]):
                    break
                if nxt in blocked or nxt in local_used:
                    ok = False
                    break
                final_cells.append(nxt)
                cursor = nxt
            if not ok or not final_cells:
                continue
            if cursor[0] not in (0, height - 1) and cursor[1] not in (0, width - 1):
                continue
            local_used.update(final_cells)

            return ChainPlan(
                head=head,
                direction=direction,
                start_color=int(carrier_colors[0]),
                tail=(tail1, tail2),
                singletons=tuple(singletons),
                dominoes=tuple(dominoes),
                event_types=event_types,
                used_cells=frozenset(local_used),
            )

    return None


def _render_input(shape: tuple[int, int], chains: list[ChainPlan]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for chain in chains:
        grid[chain.head] = int(chain.start_color)
        for tail in chain.tail:
            grid[tail] = TAIL
        for pos, color in chain.singletons:
            grid[pos] = int(color)
        for a, b, color in chain.dominoes:
            grid[a] = int(color)
            grid[b] = int(color)
    return grid


def _expected_from_plans(shape: tuple[int, int], chains: list[ChainPlan]) -> np.ndarray:
    movers = [Mover(chain.head, chain.start_color, chain.direction) for chain in chains]
    singletons: dict[Coord, int] = {}
    domino_partner: dict[Coord, Coord] = {}
    out = np.full(shape, BG, dtype=int)
    for chain in chains:
        for pos, color in chain.singletons:
            singletons[pos] = int(color)
            out[pos] = int(color)
        for a, b, color in chain.dominoes:
            domino_partner[a] = b
            domino_partner[b] = a
            out[a] = int(color)
            out[b] = int(color)
    return _simulate(shape, movers, singletons, domino_partner, base=out)


def _sample_event_templates() -> list[tuple[str, ...]]:
    handoff_templates = [("handoff",), ("handoff", "deflect"), ("handoff", "handoff")]
    deflect_templates = [("deflect",), ("deflect", "handoff"), ("deflect", "deflect")]
    chains = [random.choice(handoff_templates), random.choice(deflect_templates)]
    if random.random() < 0.45:
        chains.append(random.choice([tuple(), ("handoff",), ("deflect",), ("handoff", "deflect")]))
    random.shuffle(chains)
    return chains


def _build_sample() -> tuple[np.ndarray, np.ndarray, list[ChainPlan]]:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(18, 30)
        width = random.randint(18, 30)
        shape = (height, width)
        blocked: set[Coord] = set()
        chains: list[ChainPlan] = []
        ok = True

        for event_types in _sample_event_templates():
            carrier_count = 1 + event_types.count("handoff")
            carrier_colors = _pick_distinct_colors(carrier_count)
            plan = _plan_chain(shape, blocked, event_types, carrier_colors)
            if plan is None:
                ok = False
                break
            chains.append(plan)
            blocked |= _expand_margin(set(plan.used_cells), shape)

        if not ok:
            continue

        inp = _render_input(shape, chains)
        expected = _expected_from_plans(shape, chains)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue
        if int(np.count_nonzero(inp == TAIL)) != 2 * len(chains):
            continue
        if sum(event == "handoff" for chain in chains for event in chain.event_types) < 1:
            continue
        if sum(event == "deflect" for chain in chains for event in chain.event_types) < 1:
            continue
        if np.array_equal(inp, expected):
            continue
        return inp, expected, chains

    raise RuntimeError("failed to generate a valid ARC sample")


def generate_input() -> np.ndarray:
    return _build_sample()[0]
