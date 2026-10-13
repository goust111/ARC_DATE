from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class _Slot:
    box: tuple[int, int, int, int]
    marker: tuple[int, int] | None
    direction: tuple[int, int] | None
    color: int | None


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    answer = []
    while unseen:
        start = unseen.pop()
        cells = [start]
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    cells.append((nr, nc))
                    queue.append((nr, nc))
        answer.append(np.asarray(cells, dtype=int))
    return answer


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corners(box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = box
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _corner_direction(box: tuple[int, int, int, int], marker: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    mapping = {
        (r0, c0): (-1, -1),
        (r0, c1): (-1, 1),
        (r1, c0): (1, -1),
        (r1, c1): (1, 1),
    }
    assert marker in mapping, "the gray gap must occupy a slot corner"
    return mapping[marker]


def _parse_slots(g: np.ndarray) -> list[_Slot]:
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, GRAY, *PALETTE})
    slots: list[_Slot] = []
    for coords in _components4(g != GRAY):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        assert r1 > r0 and c1 > c0, "slots must be at least 2 by 2"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        values = set(map(int, patch[patch != GRAY]))
        if values == {BLACK}:
            gaps = [corner for corner in _corners(box) if int(g[corner]) == GRAY]
            assert len(gaps) == 1, "an empty slot must have exactly one gray corner gap"
            assert int(np.count_nonzero(patch == GRAY)) == 1
            marker = gaps[0]
            slots.append(_Slot(box, marker, _corner_direction(box, marker), None))
        else:
            assert len(values) == 1 and BLACK not in values and not np.any(patch == GRAY)
            color = next(iter(values))
            assert np.all(patch == color), "a prefilled slot must be a solid rectangle"
            slots.append(_Slot(box, None, None, color))
    assert slots and any(slot.color is None for slot in slots)
    assert len({slot.color for slot in slots if slot.color is not None}) >= 2
    return slots


def _inside(box: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _ray_target(slot_index: int, slots: list[_Slot], shape: tuple[int, int]) -> int:
    slot = slots[slot_index]
    assert slot.marker is not None and slot.direction is not None
    h, w = shape
    r, c = slot.marker
    dr, dc = slot.direction
    r, c = r + dr, c + dc
    while 0 <= r < h and 0 <= c < w:
        hits = [j for j, other in enumerate(slots) if j != slot_index and _inside(other.box, (r, c))]
        assert len(hits) <= 1, "a pointer ray must have an unambiguous first slot"
        if hits:
            return hits[0]
        r, c = r + dr, c + dc
    raise AssertionError("every empty slot pointer must hit another slot")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every black slot by relaying colors backward through its gray-corner pointer chain.

    Input structure:
    - Gray support surrounds separated rectangular slots of varying sizes.
    - An empty slot is black except for exactly one gray corner gap; a prefilled slot is a solid non-gray, non-black color and has no gap.
    - At least two differently colored prefilled slots seed an acyclic pointer network.

    Rule:
    - The gap corner selects the outward 45-degree diagonal ray; the first slot bbox entered by that ray is the pointer target.
    - In synchronous rounds, an empty slot whose target is colored fills all of its black cells with that color, while its gray gap remains unchanged.
    - Repeat until all slots are colored; preserve the gray support and every prefilled slot.

    Key insight:
    - Slot geometry is a distraction: corner gaps encode directed edges, and colors must propagate through the resulting multi-round acyclic graph.
    """
    g = np.asarray(grid, dtype=int)
    slots = _parse_slots(g)
    targets = {
        i: _ray_target(i, slots, g.shape)
        for i, slot in enumerate(slots)
        if slot.color is None
    }

    out = g.copy()
    colors = [slot.color for slot in slots]
    remaining = set(targets)
    while remaining:
        snapshot = colors[:]
        ready = [(i, snapshot[targets[i]]) for i in remaining if snapshot[targets[i]] is not None]
        assert ready, "pointer relation must be acyclic and reach a prefilled slot"
        for i, color in ready:
            assert color is not None
            r0, c0, r1, c1 = slots[i].box
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[patch == BLACK] = int(color)
            colors[i] = int(color)
            remaining.remove(i)
    return out


# ------------------------------ generator ------------------------------


@dataclass
class _GenNode:
    box: tuple[int, int, int, int]
    parent: int | None
    direction: tuple[int, int] | None
    depth: int
    root: int
    color: int


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _box_from_corner(marker: tuple[int, int], direction: tuple[int, int], h: int, w: int):
    r, c = marker
    dr, dc = direction
    r0 = r if dr < 0 else r - h + 1
    c0 = c if dc < 0 else c - w + 1
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _marker_for_box(box: tuple[int, int, int, int], direction: tuple[int, int]):
    r0, c0, r1, c1 = box
    dr, dc = direction
    return (r0 if dr < 0 else r1, c0 if dc < 0 else c1)


def _ray_first_box(marker, direction, boxes, shape, skip: int | None = None) -> int | None:
    h, w = shape
    r, c = marker
    dr, dc = direction
    r, c = r + dr, c + dc
    while 0 <= r < h and 0 <= c < w:
        hits = [i for i, box in enumerate(boxes) if i != skip and _inside(box, (r, c))]
        if hits:
            return hits[0] if len(hits) == 1 else -1
        r, c = r + dr, c + dc
    return None


def _random_root_box(shape, boxes):
    H, W = shape
    for _ in range(200):
        h, w = random.randint(2, 4), random.randint(2, 5)
        r0 = random.randint(5, H - h - 5)
        c0 = random.randint(5, W - w - 5)
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if all(not _expanded_overlap(box, old, 2) for old in boxes):
            return box
    return None


def _place_child(nodes: list[_GenNode], parent: int, direction, shape):
    H, W = shape
    boxes = [node.box for node in nodes]
    target = nodes[parent].box
    target_cells = [(r, c) for r in range(target[0], target[2] + 1) for c in range(target[1], target[3] + 1)]
    random.shuffle(target_cells)
    sizes = [(h, w) for h in range(2, 5) for w in range(2, 6)]
    random.shuffle(sizes)
    gaps = list(range(2, 8))
    random.shuffle(gaps)
    for hit in target_cells:
        for gap in gaps:
            marker = (hit[0] - gap * direction[0], hit[1] - gap * direction[1])
            for h, w in sizes:
                box = _box_from_corner(marker, direction, h, w)
                r0, c0, r1, c1 = box
                if r0 < 1 or c0 < 1 or r1 >= H - 1 or c1 >= W - 1:
                    continue
                if any(_expanded_overlap(box, old, 1) for old in boxes):
                    continue
                if _ray_first_box(marker, direction, boxes, shape) != parent:
                    continue
                # A later box may not cut in front of an already established pointer.
                trial_boxes = boxes + [box]
                valid = True
                for i, node in enumerate(nodes):
                    if node.parent is None:
                        continue
                    old_marker = _marker_for_box(node.box, node.direction)
                    if _ray_first_box(old_marker, node.direction, trial_boxes, shape, skip=i) != node.parent:
                        valid = False
                        break
                if valid:
                    return box
    return None


def _render(nodes: list[_GenNode], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, GRAY, dtype=int)
    expected = inp.copy()
    for node in nodes:
        r0, c0, r1, c1 = node.box
        expected[r0 : r1 + 1, c0 : c1 + 1] = node.color
        if node.parent is None:
            inp[r0 : r1 + 1, c0 : c1 + 1] = node.color
        else:
            inp[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            inp[_marker_for_box(node.box, node.direction)] = GRAY
            expected[_marker_for_box(node.box, node.direction)] = GRAY
    return inp, expected


def _nearest_is_not_always_parent(nodes: list[_GenNode]) -> bool:
    centers = [((n.box[0] + n.box[2]) / 2, (n.box[1] + n.box[3]) / 2) for n in nodes]
    for i, node in enumerate(nodes):
        if node.parent is None:
            continue
        r, c = _marker_for_box(node.box, node.direction)
        distances = [(max(abs(r - rr), abs(c - cc)), j) for j, (rr, cc) in enumerate(centers) if j != i]
        nearest = min(distances)[1]
        if nearest != node.parent:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(24, 30), random.randint(24, 30))
    root_count = random.choice((2, 2, 3))
    colors = random.sample(PALETTE, root_count)
    nodes: list[_GenNode] = []
    for root, color in enumerate(colors):
        box = _random_root_box(shape, [node.box for node in nodes])
        if box is None:
            return None
        nodes.append(_GenNode(box, None, None, 0, root, color))

    total = random.randint(max(8, root_count + 5), 11)
    directions = list(DIRECTIONS)
    random.shuffle(directions)
    direction_plan = (directions * 3)[: total - root_count]
    first_children: list[int] = []
    for k in range(total - root_count):
        if k < root_count:
            parent = k
        elif k == root_count:
            parent = first_children[0]  # force a genuinely delayed, second-round relay
        else:
            eligible = [i for i, node in enumerate(nodes) if node.depth < 3]
            parent = random.choice(eligible)
        direction = direction_plan[k]
        box = _place_child(nodes, parent, direction, shape)
        if box is None:
            return None
        target = nodes[parent]
        nodes.append(_GenNode(box, parent, direction, target.depth + 1, target.root, target.color))
        if k < root_count:
            first_children.append(len(nodes) - 1)

    if len({(n.box[2] - n.box[0] + 1, n.box[3] - n.box[1] + 1) for n in nodes}) < 3:
        return None
    if len({n.direction for n in nodes if n.direction is not None}) < 4:
        return None
    if max(n.depth for n in nodes) < 2 or not _nearest_is_not_always_parent(nodes):
        return None
    if {n.root for n in nodes if n.parent is not None} != set(range(root_count)):
        return None

    inp, expected = _render(nodes, shape)
    try:
        parsed = _parse_slots(inp)
        actual_targets = {i: _ray_target(i, parsed, shape) for i, s in enumerate(parsed) if s.color is None}
        solved = main(inp)
    except (AssertionError, ValueError):
        return None
    if len(actual_targets) != total - root_count or not np.array_equal(solved, expected):
        return None
    intended_parent_box = {node.box: nodes[node.parent].box for node in nodes if node.parent is not None}
    for source, target in actual_targets.items():
        if parsed[target].box != intended_parent_box.get(parsed[source].box):
            return None
    if np.array_equal(inp, expected) or len(set(map(int, expected[expected != GRAY]))) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, ambiguity-checked multi-source pointer relay puzzle."""
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain input")
