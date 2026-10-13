from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


Coord = tuple[int, int]
Shape = tuple[Coord, ...]


# Small, visibly distinct 4-connected silhouettes.  Sprites are translated but
# never rotated, so these normalized masks are their exact family identities.
SHAPES: tuple[Shape, ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
)

# Canonical undirected orientations.  Opposite signs describe the same queue
# read in reverse, so only one representative of each is needed.
DIRECTIONS: tuple[Coord, ...] = ((0, 1), (1, 0), (1, 1), (1, -1))
GENERATION_ATTEMPTS = 600


@dataclass(frozen=True)
class _Component:
    color: int
    coords: tuple[Coord, ...]
    anchor: Coord
    shape: Shape


@dataclass(frozen=True)
class _FamilyPlan:
    color: int
    shape: Shape
    direction: Coord
    spacing: int
    anchors: tuple[Coord, ...]
    missing_index: int

    @property
    def missing_anchor(self) -> Coord:
        return self.anchors[self.missing_index]


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best = int(counts.max())
    winners = values[counts == best]
    assert len(winners) == 1, "background must be uniquely dominant"
    return int(winners[0])


def _components(grid: np.ndarray, bg: int) -> list[_Component]:
    """Return monochromatic 4-connected components using (row, column) coords."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[_Component] = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == bg or seen[sr, sc]:
                continue
            seen[sr, sc] = True
            todo: deque[Coord] = deque([(sr, sc)])
            cells: list[Coord] = []
            while todo:
                r, c = todo.popleft()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= rr < h
                        and 0 <= cc < w
                        and not seen[rr, cc]
                        and int(grid[rr, cc]) == color
                    ):
                        seen[rr, cc] = True
                        todo.append((rr, cc))

            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            shape = tuple(sorted((r - r0, c - c0) for r, c in cells))
            result.append(_Component(color, tuple(sorted(cells)), (r0, c0), shape))
    return result


def _group_components(components: list[_Component]) -> dict[tuple[int, Shape], list[_Component]]:
    groups: dict[tuple[int, Shape], list[_Component]] = defaultdict(list)
    for component in components:
        groups[(component.color, component.shape)].append(component)
    return dict(groups)


def _completion_is_arithmetic(anchors: set[Coord], candidate: Coord) -> bool:
    """Check that adding candidate makes one exact finite 2-D arithmetic train."""
    points = anchors | {candidate}
    count = len(points)
    if count != len(anchors) + 1 or count < 4:
        return False

    pairs = [(a, b) for a in points for b in points if a < b]
    p0, p1 = max(
        pairs,
        key=lambda pair: (pair[1][0] - pair[0][0]) ** 2 + (pair[1][1] - pair[0][1]) ** 2,
    )
    dr_total = p1[0] - p0[0]
    dc_total = p1[1] - p0[1]
    intervals = count - 1
    if dr_total % intervals or dc_total % intervals:
        return False
    step = (dr_total // intervals, dc_total // intervals)
    if step == (0, 0):
        return False

    expected = {(p0[0] + i * step[0], p0[1] + i * step[1]) for i in range(count)}
    if points != expected:
        return False
    # The absent member must be internal; absent endpoints are not observable as
    # the doubled internal gap required by the puzzle.
    return candidate not in {p0, p1}


def _missing_anchor(anchors: list[Coord]) -> Coord:
    """Find the unique integer midpoint whose insertion completes the train."""
    anchor_set = set(anchors)
    assert len(anchor_set) == len(anchors) >= 3, "family anchors must be distinct"
    candidates: set[Coord] = set()
    for i, a in enumerate(anchors):
        for b in anchors[i + 1 :]:
            if (a[0] + b[0]) % 2 or (a[1] + b[1]) % 2:
                continue
            midpoint = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            if midpoint not in anchor_set and _completion_is_arithmetic(anchor_set, midpoint):
                candidates.add(midpoint)
    assert len(candidates) == 1, "each repeated family must have one uniquely determined internal absence"
    return next(iter(candidates))


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    assert coords.size, "at least one missing sprite is required"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble every never-drawn member missing from repeated equidistant sprite trains, then tight-crop it.

    Input structure:
    A uniquely dominant background contains several monochromatic 4-connected
    sprites.  Sprites with the same translation-normalized mask and color form
    a straight equal-step train with one internal member absent; singleton
    shape-color sprites are distractors.

    Rule:
    Group components by both normalized shape and color.  For every group with
    at least three visible members, insert the unique internal anchor that turns
    its anchors into a complete two-dimensional arithmetic progression (the
    visible gap there is twice the standard step), stamp the family's full mask
    at that anchor in its own color, combine all such stamps, and return their
    minimum bounding-box crop.

    Key insight:
    Nothing already drawn is copied: the output is the composite shadow of the
    absent family members, while singleton lookalikes and same-color shapes must
    be ignored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2-D ARC grid"
    bg = _background(g)
    groups = _group_components(_components(g, bg))

    missing: list[tuple[int, Shape, Coord]] = []
    for (color, shape), members in groups.items():
        if len(members) == 1:
            continue
        assert len(members) >= 3, "non-singleton families must show at least three members"
        anchor = _missing_anchor([member.anchor for member in members])
        missing.append((color, shape, anchor))
    assert len(missing) >= 2, "the puzzle requires multiple repeated families"

    full = np.full(g.shape, bg, dtype=int)
    h, w = g.shape
    for color, shape, (r0, c0) in missing:
        for dr, dc in shape:
            r, c = r0 + dr, c0 + dc
            assert 0 <= r < h and 0 <= c < w, "missing sprite must fit completely inside the input"
            assert int(g[r, c]) == bg, "a missing member must never be drawn in the input"
            assert int(full[r, c]) == bg, "missing members of different families must not overlap"
            full[r, c] = color
    return _tight_crop(full, bg)


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _expanded_bbox_cells(shape: Shape, anchor: Coord, grid_shape: tuple[int, int], pad: int = 1) -> set[Coord]:
    h, w = grid_shape
    sh, sw = _shape_size(shape)
    r0, c0 = anchor
    return {
        (r, c)
        for r in range(max(0, r0 - pad), min(h, r0 + sh + pad))
        for c in range(max(0, c0 - pad), min(w, c0 + sw + pad))
    }


def _candidate_origins(
    grid_shape: tuple[int, int], shape: Shape, step: Coord, slots: int, margin: int = 1
) -> list[Coord]:
    h, w = grid_shape
    sh, sw = _shape_size(shape)
    row_offsets = [i * step[0] for i in range(slots)]
    col_offsets = [i * step[1] for i in range(slots)]
    r_low = margin - min(row_offsets)
    r_high = h - margin - sh - max(row_offsets)
    c_low = margin - min(col_offsets)
    c_high = w - margin - sw - max(col_offsets)
    if r_low > r_high or c_low > c_high:
        return []
    return [(r, c) for r in range(r_low, r_high + 1) for c in range(c_low, c_high + 1)]


def _place_family(
    grid: np.ndarray,
    blocked: set[Coord],
    color: int,
    shape: Shape,
    direction: Coord,
    spacing: int,
    slots: int,
    missing_index: int,
) -> _FamilyPlan | None:
    step = (direction[0] * spacing, direction[1] * spacing)
    origins = _candidate_origins(grid.shape, shape, step, slots)
    random.shuffle(origins)
    for origin in origins:
        anchors = tuple((origin[0] + i * step[0], origin[1] + i * step[1]) for i in range(slots))
        reservation: set[Coord] = set()
        for anchor in anchors:  # Reserve the invisible gap too, keeping its evidence clean.
            reservation.update(_expanded_bbox_cells(shape, anchor, grid.shape, pad=1))
        if not blocked.isdisjoint(reservation):
            continue
        for i, (r0, c0) in enumerate(anchors):
            if i == missing_index:
                continue
            for dr, dc in shape:
                grid[r0 + dr, c0 + dc] = color
        blocked.update(reservation)
        return _FamilyPlan(color, shape, direction, spacing, anchors, missing_index)
    return None


def _place_distractor(grid: np.ndarray, blocked: set[Coord], color: int, shape: Shape) -> Coord | None:
    h, w = grid.shape
    sh, sw = _shape_size(shape)
    candidates = [(r, c) for r in range(1, h - sh) for c in range(1, w - sw)]
    random.shuffle(candidates)
    for anchor in candidates:
        reservation = _expanded_bbox_cells(shape, anchor, grid.shape, pad=1)
        if not blocked.isdisjoint(reservation):
            continue
        for dr, dc in shape:
            grid[anchor[0] + dr, anchor[1] + dc] = color
        blocked.update(reservation)
        return anchor
    return None


def _expected_output(grid_shape: tuple[int, int], bg: int, plans: list[_FamilyPlan]) -> np.ndarray:
    full = np.full(grid_shape, bg, dtype=int)
    for plan in plans:
        r0, c0 = plan.missing_anchor
        for dr, dc in plan.shape:
            assert int(full[r0 + dr, c0 + dc]) == bg
            full[r0 + dr, c0 + dc] = plan.color
    return _tight_crop(full, bg)


def _semantic_validation(
    grid: np.ndarray,
    bg: int,
    plans: list[_FamilyPlan],
    distractor_keys: list[tuple[int, Shape]],
) -> bool:
    """Validate the described semantics independently of mere generator closure."""
    try:
        components = _components(grid, bg)
        groups = _group_components(components)
        family_keys = {(plan.color, plan.shape) for plan in plans}
        if len(family_keys) != len(plans) or len(plans) < 2:
            return False
        if len(set(plan.direction for plan in plans)) != len(plans):
            return False
        if len(set(plan.spacing for plan in plans)) != len(plans):
            return False
        if len(set(plan.color for plan in plans)) != len(plans):
            return False
        if len(set(plan.shape for plan in plans)) != len(plans):
            return False

        # Exactly the planned keys repeat; every distractor key occurs once.
        repeated = {key for key, members in groups.items() if len(members) >= 3}
        if repeated != family_keys:
            return False
        if any(len(groups.get(key, [])) != 1 for key in distractor_keys):
            return False
        for key, members in groups.items():
            if key in family_keys:
                plan = next(p for p in plans if (p.color, p.shape) == key)
                if len(members) != len(plan.anchors) - 1:
                    return False
            elif len(members) != 1:
                return False

        # The unique inferred midpoint, its doubled-gap structure, and every
        # output pixel are checked against construction metadata.
        for plan in plans:
            key = (plan.color, plan.shape)
            anchors = [component.anchor for component in groups[key]]
            if _missing_anchor(anchors) != plan.missing_anchor:
                return False
            step = (plan.direction[0] * plan.spacing, plan.direction[1] * plan.spacing)
            visible_indices = [i for i in range(len(plan.anchors)) if i != plan.missing_index]
            gaps = [visible_indices[i + 1] - visible_indices[i] for i in range(len(visible_indices) - 1)]
            if gaps.count(2) != 1 or any(gap not in (1, 2) for gap in gaps):
                return False
            if step == (0, 0):
                return False
            for dr, dc in plan.shape:
                r, c = plan.missing_anchor[0] + dr, plan.missing_anchor[1] + dc
                if int(grid[r, c]) != bg:
                    return False

        # Two deliberate counterexamples make both halves of the grouping key necessary.
        if not any(key[1] in {p.shape for p in plans} and key[0] not in {p.color for p in plans}
                   for key in distractor_keys):
            return False
        if not any(key[0] in {p.color for p in plans} and key[1] not in {p.shape for p in plans}
                   for key in distractor_keys):
            return False

        expected = _expected_output(grid.shape, bg, plans)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if set(map(int, np.unique(actual))) - {bg} != {p.color for p in plans}:
            return False
        if int(np.count_nonzero(actual != bg)) != sum(len(p.shape) for p in plans):
            return False
        return max(grid.shape) <= 30 and max(actual.shape) <= 30
    except (AssertionError, StopIteration):
        return False


def _generation_attempt() -> np.ndarray | None:
    family_count = random.choice((2, 2, 3))
    h = random.randint(25 if family_count == 3 else 23, 30)
    w = random.randint(25 if family_count == 3 else 23, 30)

    bg = random.choice((0, 0, random.randint(0, 9)))
    available_colors = [color for color in range(10) if color != bg]
    family_colors = random.sample(available_colors, family_count)
    family_shapes = random.sample(list(SHAPES), family_count)
    directions = random.sample(list(DIRECTIONS), family_count)
    spacings = random.sample((4, 5, 6), family_count)

    grid = np.full((h, w), bg, dtype=int)
    blocked: set[Coord] = set()
    plans: list[_FamilyPlan] = []

    order = list(range(family_count))
    random.shuffle(order)
    for index in order:
        slots = random.choice((4, 4, 5))
        missing_index = random.randint(1, slots - 2)
        plan = _place_family(
            grid,
            blocked,
            family_colors[index],
            family_shapes[index],
            directions[index],
            spacings[index],
            slots,
            missing_index,
        )
        if plan is None:
            return None
        plans.append(plan)

    # Counterexample 1: same shape as a real family, but a different color.
    new_colors = [color for color in available_colors if color not in family_colors]
    if not new_colors:
        return None
    distractors: list[tuple[int, Shape]] = [(random.choice(new_colors), family_shapes[0])]

    # Counterexample 2: same color as another family, but a different shape.
    unused_shapes = [shape for shape in SHAPES if shape not in family_shapes]
    if not unused_shapes:
        return None
    distractors.append((family_colors[-1], random.choice(unused_shapes)))

    # Occasionally add one more unique singleton for object-count diversity.
    if random.random() < 0.45:
        used_keys = set(distractors) | {(p.color, p.shape) for p in plans}
        options = [
            (color, shape)
            for color in available_colors
            for shape in SHAPES
            if (color, shape) not in used_keys
        ]
        distractors.append(random.choice(options))

    for color, shape in distractors:
        if _place_distractor(grid, blocked, color, shape) is None:
            return None

    # The background must remain uniquely dominant, and output composition must
    # depend on every family while all singleton counterexamples remain excluded.
    if _background(grid) != bg:
        return None
    if not _semantic_validation(grid, bg, plans, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse valid instance with two or three independently missing families."""
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid missing-train puzzle after {GENERATION_ATTEMPTS} attempts")
