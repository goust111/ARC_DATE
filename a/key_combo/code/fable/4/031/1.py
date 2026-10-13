from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random

import numpy as np


# This file is deliberately self-contained so SourceFileLoader can load it from
# any working directory.  It follows arc_utils' canonical conventions:
# coordinates are (row, column), bboxes are inclusive, and connectivity is 4-way.
_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class _Entity:
    coords: np.ndarray
    values: np.ndarray
    members: frozenset[int]
    closed: bool

    def copy(self) -> "_Entity":
        return _Entity(self.coords.copy(), self.values.copy(), self.members, self.closed)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        lo = self.coords.min(axis=0)
        hi = self.coords.max(axis=0)
        return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])

    def shifted(self, dr: int) -> "_Entity":
        moved = self.copy()
        moved.coords[:, 0] += int(dr)
        return moved


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N,2) arrays of (r,c)."""
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result: list[np.ndarray] = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        cells = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in _N4:
                q = (r + dr, c + dc)
                if 0 <= q[0] < h and 0 <= q[1] < w and q in unseen:
                    unseen.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(np.asarray(cells, dtype=int))
    return result


def _has_hole(coords: np.ndarray) -> bool:
    """Test for bbox-enclosed background using the object's own coordinate frame."""
    lo = coords.min(axis=0)
    rel = coords - lo
    hi = rel.max(axis=0)
    h, w = int(hi[0]) + 1, int(hi[1]) + 1
    solid = np.zeros((h, w), dtype=bool)
    solid[rel[:, 0], rel[:, 1]] = True

    exterior = np.zeros_like(solid)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not solid[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not solid[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any((~solid) & (~exterior)))


def _extract(grid: np.ndarray, bg: int) -> list[_Entity]:
    entities: list[_Entity] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            values = np.full(len(coords), color, dtype=int)
            entities.append(_Entity(coords, values, frozenset({color}), _has_hole(coords)))
    if not entities:
        raise ValueError("expected foreground objects")
    return entities


def _occupied(entities: list[_Entity]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for entity in entities for r, c in entity.coords}


def _can_shift(entity: _Entity, dr: int, occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in entity.coords:
        q = (int(r) + dr, int(c))
        if not (0 <= q[0] < h and 0 <= q[1] < w) or q in occupied:
            return False
    return True


def _pack_one(entity: _Entity, occupied: set[tuple[int, int]], shape: tuple[int, int], upward: bool) -> _Entity:
    r0, _c0, r1, _c1 = entity.bbox
    height = r1 - r0 + 1
    tops = range(0, shape[0] - height + 1) if upward else range(shape[0] - height, -1, -1)
    for top in tops:
        dr = top - r0
        if _can_shift(entity, dr, occupied, shape):
            return entity.shifted(dr)
    raise ValueError("objects cannot be settled without overlap")


def _move_up_stepwise(entity: _Entity, others: list[_Entity], shape: tuple[int, int]) -> tuple[_Entity, int]:
    occupied = _occupied(others)
    moved = entity.copy()
    distance = 0
    while _can_shift(moved, -1, occupied, shape):
        moved.coords[:, 0] -= 1
        distance += 1
    return moved, distance


def _touch(a: _Entity, b: _Entity) -> bool:
    bset = {(int(r), int(c)) for r, c in b.coords}
    for r, c in a.coords:
        rr, cc = int(r), int(c)
        if any((rr + dr, cc + dc) in bset for dr, dc in _N4):
            return True
    return False


def _contact_groups(entities: list[_Entity]) -> list[list[int]]:
    """Connected components of the entity-level 4-touch graph."""
    n = len(entities)
    graph = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _touch(entities[i], entities[j]):
                graph[i].add(j)
                graph[j].add(i)
    unseen = set(range(n))
    groups: list[list[int]] = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        group = [seed]
        while stack:
            i = stack.pop()
            for j in graph[i] & unseen:
                unseen.remove(j)
                stack.append(j)
                group.append(j)
        groups.append(sorted(group))
    return groups


def _is_connected_group(group: list[_Entity]) -> bool:
    if not group:
        return False
    reached = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(len(group)):
            if j not in reached and _touch(group[i], group[j]):
                reached.add(j)
                stack.append(j)
    return len(reached) == len(group)


def _minimal_hole_groups(open_entities: list[_Entity]) -> list[list[_Entity]]:
    """Find inclusion-minimal connected subsets whose union encloses a hole.

    A non-contributing object merely touching a newly formed frame must not be
    swallowed by it; otherwise a genuine later cascade would be impossible.
    Intended inputs make these minimal groups disjoint, which gives a unique
    simultaneous reaction without using color ids as priorities.
    """
    candidates: list[tuple[frozenset[int], list[_Entity]]] = []
    for component_indices in _contact_groups(open_entities):
        component = [open_entities[i] for i in component_indices]
        n = len(component)
        hole_index_sets: list[frozenset[int]] = []
        for size in range(2, n + 1):
            # n is small in ARC scenes; explicit subsets make minimality exact.
            for indices in combinations(range(n), size):
                index_set = frozenset(indices)
                if any(previous < index_set for previous in hole_index_sets):
                    continue
                group = [component[i] for i in indices]
                if not _is_connected_group(group):
                    continue
                coords = np.concatenate([e.coords for e in group], axis=0)
                if _has_hole(coords):
                    hole_index_sets.append(index_set)
                    candidates.append((frozenset(id(e) for e in group), group))

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if candidates[i][0] & candidates[j][0]:
                raise ValueError("ambiguous overlapping minimal hole-forming groups")
    return [group for _ids, group in candidates]


def _merge(group: list[_Entity]) -> _Entity:
    coords = np.concatenate([e.coords for e in group], axis=0)
    values = np.concatenate([e.values for e in group], axis=0)
    members = frozenset().union(*(e.members for e in group))
    return _Entity(coords, values, members, True)


def _render(shape: tuple[int, int], bg: int, entities: list[_Entity], dtype: np.dtype) -> np.ndarray:
    out = np.full(shape, bg, dtype=dtype)
    for entity in entities:
        out[entity.coords[:, 0], entity.coords[:, 1]] = entity.values
    return out


def _initial_settle(entities: list[_Entity], shape: tuple[int, int], trace: dict | None = None) -> list[_Entity]:
    closed = sorted(
        (e.copy() for e in entities if e.closed),
        key=lambda e: (e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3]),
    )
    opened = sorted(
        (e.copy() for e in entities if not e.closed),
        key=lambda e: (-e.bbox[2], e.bbox[1], -e.bbox[0], e.bbox[3]),
    )
    settled: list[_Entity] = []
    for entity in closed:
        old_top = entity.bbox[0]
        entity = _pack_one(entity, _occupied(settled), shape, upward=True)
        settled.append(entity)
        if trace is not None:
            trace["initial_moves"].append(("closed", entity.members, entity.bbox[0] - old_top))
    for entity in opened:
        old_top = entity.bbox[0]
        entity = _pack_one(entity, _occupied(settled), shape, upward=False)
        settled.append(entity)
        if trace is not None:
            trace["initial_moves"].append(("open", entity.members, entity.bbox[0] - old_top))
    return settled


def _solve(grid: np.ndarray, with_trace: bool = False):
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    bg = _background(g)
    trace = {"initial_moves": [], "merges": [], "emergent_rises": [], "rounds": 0}
    entities = _initial_settle(_extract(g, bg), g.shape, trace)

    # Each successful round reduces the number of entities, so the loop is finite.
    while True:
        open_entities = [e for e in entities if not e.closed]
        merge_groups = _minimal_hole_groups(open_entities)
        if not merge_groups:
            break

        trace["rounds"] += 1
        consumed = {id(e) for group in merge_groups for e in group}
        entities = [e for e in entities if id(e) not in consumed]
        emerged = [_merge(group) for group in merge_groups]
        for entity in emerged:
            trace["merges"].append(entity.members)
        entities.extend(emerged)

        # A reaction starts a new settling pass: existing/new closed bodies pack
        # upward in their current vertical order, then all remaining open bodies
        # repack downward.  Separating the two mover classes is the same visible
        # reference frame as the initial rule and lets vacated space cause a later
        # (not prematurely simultaneous) closure.
        old_tops = {entity.members: entity.bbox[0] for entity in emerged}
        entities = _initial_settle(entities, g.shape)
        for emerged_entity in emerged:
            moved = next(e for e in entities if e.members == emerged_entity.members)
            trace["emergent_rises"].append((moved.members, old_tops[moved.members] - moved.bbox[0]))

    out = _render(g.shape, bg, entities, g.dtype)
    return (out, trace, entities) if with_trace else out


def main(grid: np.ndarray) -> np.ndarray:
    """Settle hole-bearing objects upward and holeless objects downward, promoting any stack that creates a new hole.

    Input structure:
    - A black background contains several differently colored monochromatic 4-connected objects.
    - An object's class is measured in its own inclusive bounding box: some objects already enclose background holes,
      while every other object is individually holeless and initially separated from the others.

    Rule:
    - Preserve every pixel's color, shape, and column. Pack initially closed objects upward in their original vertical
      order, then pack open objects downward in their original vertical order. After each settling, find the uniquely
      determined inclusion-minimal 4-connected groups whose unions newly enclose background (touching non-contributors
      stay separate). Merge each such group, repack closed bodies upward and remaining open bodies downward, and repeat.

    Key insight:
    - Closed/open is a dynamic geometric state: downward stacking can create a hole, and that newly closed composite
      must reverse direction as one rigid body, so a one-time classification cannot produce the answer.
    """
    return _solve(grid, with_trace=False)


# ------------------------------- generator -------------------------------

_GEN_CALLS = 0
_SEEN_OUTPUTS: dict[bytes, bytes] = {}


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def _u_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = mask[:, -1] = True
    mask[-1, :] = True
    return mask


def _misc_open_mask() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    mask = np.zeros((h, w), dtype=bool)
    if random.random() < 0.5:
        mask[:, 0] = True
        mask[-1, :] = True
    else:
        mask[0, :] = True
        mask[:, random.randrange(w)] = True
    return mask


def _paint_object(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[top + rr, left + cc] = int(color)


def _make_expected(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        _paint_object(out, spec["mask"], spec["out_top"], spec["left"], spec["color"])
    return out


def _one_pass(grid: np.ndarray) -> np.ndarray:
    bg = _background(grid)
    entities = _initial_settle(_extract(grid, bg), grid.shape)
    return _render(grid.shape, bg, entities, grid.dtype)


def _valid_initial_objects(grid: np.ndarray, expected_count: int) -> bool:
    entities = _extract(grid, 0)
    if len(entities) != expected_count:
        return False
    colors = [next(iter(e.members)) for e in entities]
    if len(colors) != len(set(colors)):
        return False
    # No two distinct input objects are already touching; all contacts are caused by settling.
    return all(not _touch(entities[i], entities[j]) for i in range(len(entities)) for j in range(i + 1, len(entities)))


def _validate_attempt(
    grid: np.ndarray,
    expected: np.ndarray,
    specs: list[dict],
    expected_merges: int,
    expected_rounds: int = 1,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30 or np.array_equal(grid, expected):
        return False
    if int(np.count_nonzero(grid == 0)) <= int(np.max(np.bincount(grid.ravel())[1:], initial=0)):
        return False
    if not _valid_initial_objects(grid, len(specs)):
        return False

    try:
        actual, trace, final_entities = _solve(grid, with_trace=True)
    except (ValueError, AssertionError):
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(trace["merges"]) != expected_merges or trace["rounds"] != expected_rounds:
        return False
    if any(len(members) != 2 for members in trace["merges"]):
        return False
    if not trace["emergent_rises"] or any(distance <= 0 for _members, distance in trace["emergent_rises"]):
        return False
    initial_moves = trace["initial_moves"]
    if not any(kind == "closed" and dr < 0 for kind, _members, dr in initial_moves):
        return False
    if not all(dr > 0 for kind, _members, dr in initial_moves if kind == "open"):
        return False

    # Semantic anti-shortcut: stopping after static initial classification is observably wrong.
    if np.array_equal(_one_pass(grid), actual):
        return False

    # Conservation and the prescribed column reference frame are checked per visible color.
    for color in np.unique(grid):
        color = int(color)
        if color == 0:
            continue
        in_coords = np.argwhere(grid == color)
        out_coords = np.argwhere(actual == color)
        if len(in_coords) != len(out_coords):
            return False
        if sorted(map(int, in_coords[:, 1])) != sorted(map(int, out_coords[:, 1])):
            return False

    # Convergence means no remaining maximal open contact group encloses a hole.
    remaining = [e for e in final_entities if not e.closed]
    if _minimal_hole_groups(remaining):
        return False
    return True


def _gen_cascade_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    """Build a two-round chain: removing frame A lets the pieces of frame B meet."""
    height = random.randint(26, 30)
    chain_width = 5
    frame_a_width = 3
    misc_width = random.randint(2, 4)
    gap = random.randint(2, 3)
    minimum_width = chain_width + misc_width + gap + 2
    width = random.randint(minimum_width, min(18, minimum_width + 4))
    if random.random() < 0.5:
        chain_left = random.randint(1, width - minimum_width + 1)
        misc_left = chain_left + chain_width + gap
    else:
        misc_left = random.randint(1, width - minimum_width + 1)
        chain_left = misc_left + misc_width + gap

    colors = iter(random.sample(range(1, 10), 6))
    frame_color, cap_a_color, cup_a_color, cap_b_color, cup_b_color, misc_color = colors
    closed_h = random.randint(3, 4)
    cup_a_h = random.randint(2, 3)
    cup_b_h = random.randint(2, 3)
    closed_top = random.randint(1, 2)

    # Input vertical order, top to bottom: cap B, cap A, cup A, cup B.
    # After the first downward pack, only cap A + cup A is a minimal frame.
    cap_b_top = closed_top + closed_h + 2
    cap_a_top = cap_b_top + 2
    cup_a_top = cap_a_top + 2
    cup_b_top = cup_a_top + cup_a_h + 2
    if cup_b_top + cup_b_h > height - 2:
        return None

    frame_a_left = chain_left + random.choice((0, chain_width - frame_a_width))
    frame = _frame_mask(closed_h, chain_width)
    cap_a = np.ones((1, frame_a_width), dtype=bool)
    cup_a = _u_mask(cup_a_h, frame_a_width)
    cap_b = np.ones((1, chain_width), dtype=bool)
    cup_b = _u_mask(cup_b_h, chain_width)
    misc = _misc_open_mask()
    mh, mw = misc.shape
    if mw > misc_width:
        return None
    misc_col = misc_left + random.randint(0, misc_width - mw)
    misc_top = random.randint(6, height - mh - 3)

    grid = np.zeros((height, width), dtype=int)
    _paint_object(grid, frame, closed_top, chain_left, frame_color)
    _paint_object(grid, cap_b, cap_b_top, chain_left, cap_b_color)
    _paint_object(grid, cap_a, cap_a_top, frame_a_left, cap_a_color)
    _paint_object(grid, cup_a, cup_a_top, frame_a_left, cup_a_color)
    _paint_object(grid, cup_b, cup_b_top, chain_left, cup_b_color)
    _paint_object(grid, misc, misc_top, misc_col, misc_color)

    # A settles directly below the original frame.  B is created only after A
    # has left the lower stack, then settles directly below A in round two.
    specs = [
        {"mask": frame, "left": chain_left, "color": frame_color, "out_top": 0},
        {"mask": cap_a, "left": frame_a_left, "color": cap_a_color, "out_top": closed_h},
        {"mask": cup_a, "left": frame_a_left, "color": cup_a_color, "out_top": closed_h + 1},
        {"mask": cap_b, "left": chain_left, "color": cap_b_color, "out_top": closed_h + cup_a_h + 1},
        {"mask": cup_b, "left": chain_left, "color": cup_b_color, "out_top": closed_h + cup_a_h + 2},
        {"mask": misc, "left": misc_col, "color": misc_color, "out_top": height - mh},
    ]
    expected = _make_expected(grid.shape, specs)
    if not _validate_attempt(grid, expected, specs, expected_merges=2, expected_rounds=2):
        return None
    return grid, expected


def _gen_attempt(variant: int) -> tuple[np.ndarray, np.ndarray] | None:
    if variant == 3:
        return _gen_cascade_attempt()
    n_chains = 1 if variant % 2 == 0 else 2
    include_misc_closed = variant >= 2
    height = random.randint(20, 27)

    lanes: list[dict] = []
    for _ in range(n_chains):
        width = random.randint(3, 5)
        lanes.append({"kind": "chain", "width": width})
    misc_width = random.randint(3, 5)
    lanes.append({"kind": "misc", "width": misc_width})
    random.shuffle(lanes)

    gaps = [random.randint(2, 3) for _ in range(len(lanes) - 1)]
    minimum_width = sum(lane["width"] for lane in lanes) + sum(gaps) + 2
    width = random.randint(minimum_width, min(30, minimum_width + 5))
    slack = width - (sum(lane["width"] for lane in lanes) + sum(gaps))
    left = random.randint(1, slack - 1)
    for i, lane in enumerate(lanes):
        lane["left"] = left
        left += lane["width"] + (gaps[i] if i < len(gaps) else 0)

    color_count = 3 * n_chains + 1 + int(include_misc_closed)
    colors = random.sample(range(1, 10), color_count)
    random.shuffle(colors)
    color_iter = iter(colors)
    grid = np.zeros((height, width), dtype=int)
    specs: list[dict] = []

    for lane in lanes:
        c = int(lane["left"])
        w = int(lane["width"])
        if lane["kind"] == "chain":
            closed_h = random.randint(3, 4)
            cup_h = random.randint(2, 4)
            closed_top = random.randint(1, 3)
            cap_low = closed_top + closed_h + 2
            cap_high = min(10, height - cup_h - 8)
            if cap_low > cap_high:
                return None
            cap_top = random.randint(cap_low, cap_high)
            cup_top = random.randint(cap_top + 3, height - cup_h - 2)

            frame_color, cap_color, cup_color = next(color_iter), next(color_iter), next(color_iter)
            frame = _frame_mask(closed_h, w)
            cap = np.ones((1, w), dtype=bool)
            cup = _u_mask(cup_h, w)
            _paint_object(grid, frame, closed_top, c, frame_color)
            _paint_object(grid, cap, cap_top, c, cap_color)
            _paint_object(grid, cup, cup_top, c, cup_color)
            specs.extend(
                [
                    {"mask": frame, "left": c, "color": frame_color, "out_top": 0},
                    {"mask": cap, "left": c, "color": cap_color, "out_top": closed_h},
                    {"mask": cup, "left": c, "color": cup_color, "out_top": closed_h + 1},
                ]
            )
        else:
            if include_misc_closed:
                closed_h = random.randint(3, 4)
                closed_w = random.randint(3, w)
                closed_left = c + random.randint(0, w - closed_w)
                closed_top = random.randint(1, 3)
                closed_color = next(color_iter)
                frame = _frame_mask(closed_h, closed_w)
                _paint_object(grid, frame, closed_top, closed_left, closed_color)
                specs.append({"mask": frame, "left": closed_left, "color": closed_color, "out_top": 0})
                open_min_top = closed_top + closed_h + 3
            else:
                open_min_top = 5

            open_mask = _misc_open_mask()
            oh, ow = open_mask.shape
            if ow > w:
                return None
            open_left = c + random.randint(0, w - ow)
            open_top = random.randint(open_min_top, height - oh - 3)
            open_color = next(color_iter)
            _paint_object(grid, open_mask, open_top, open_left, open_color)
            specs.append({"mask": open_mask, "left": open_left, "color": open_color, "out_top": height - oh})

    expected = _make_expected(grid.shape, specs)
    if not _validate_attempt(grid, expected, specs, expected_merges=n_chains):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked examples with mandatory emergent-hole reversal."""
    global _GEN_CALLS
    variant = _GEN_CALLS % 4  # covers 1/2 chains and 1/multiple initially closed objects in every four calls
    for _ in range(300):
        made = _gen_attempt(variant)
        if made is None:
            continue
        grid, expected = made
        in_sig = grid.tobytes() + bytes(grid.shape)
        out_sig = expected.tobytes() + bytes(expected.shape)
        previous = _SEEN_OUTPUTS.get(out_sig)
        if previous is not None and previous != in_sig:
            continue
        _SEEN_OUTPUTS[out_sig] = in_sig
        _GEN_CALLS += 1
        return grid
    raise RuntimeError("generate_input: failed to construct a non-degenerate chain-reaction example")
