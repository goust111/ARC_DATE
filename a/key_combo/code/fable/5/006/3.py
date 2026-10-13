from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
Velocity = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color, following the shared ARC toolbox convention."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _split_panels(grid: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    """Find the two exclusive full-height divider columns and crop the panels."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    height, width = g.shape
    bg = _background(g)

    candidates: list[tuple[int, list[int]]] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        full = [c for c in range(width) if bool(np.all(g[:, c] == color))]
        if len(full) == 2 and int(np.sum(g == color)) == 2 * height:
            candidates.append((color, full))
    assert len(candidates) == 1, "expected one color used only by two full-height dividers"

    divider, (c0, c1) = candidates[0]
    widths = (c0, c1 - c0 - 1, width - c1 - 1)
    assert widths[0] == widths[1] == widths[2] and widths[0] >= 5
    panels = [g[:, :c0].copy(), g[:, c0 + 1 : c1].copy(), g[:, c1 + 1 :].copy()]
    return bg, int(divider), panels


def _tight_object(mask: np.ndarray) -> tuple[Coord, tuple[Coord, ...]]:
    """Represent a translated object by its top-left bbox anchor and local cells."""
    coords = np.argwhere(mask)
    assert coords.size, "object must be present in every panel"
    r0, c0 = map(int, coords.min(axis=0))
    rel = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))
    return (r0, c0), rel


def _parse_scene(grid: np.ndarray) -> dict:
    bg, divider, panels = _split_panels(grid)
    panel_colors = [set(map(int, np.unique(panel))) - {bg} for panel in panels]
    assert panel_colors[0] == panel_colors[1] == panel_colors[2]
    colors = sorted(panel_colors[0])
    assert len(colors) >= 2, "expected one wall and at least one moving object"

    static = [
        color
        for color in colors
        if all(np.array_equal(panels[0] == color, panel == color) for panel in panels[1:])
    ]
    assert len(static) == 1, "the repeated panels must contain one uniquely static wall"
    wall_color = int(static[0])
    wall = panels[0] == wall_color
    assert bool(wall.any())

    objects: dict[int, dict] = {}
    for color in colors:
        if color == wall_color:
            continue
        anchors: list[Coord] = []
        signatures: list[tuple[Coord, ...]] = []
        for panel in panels:
            anchor, rel = _tight_object(panel == color)
            anchors.append(anchor)
            signatures.append(rel)
        assert signatures[0] == signatures[1] == signatures[2], (
            "each colored moving object must preserve its exact shape"
        )
        assert len(set(anchors)) > 1, "moving objects may not be static in all three frames"
        objects[int(color)] = {"offsets": signatures[0], "anchors": anchors}

    assert objects
    return {
        "bg": int(bg),
        "divider": divider,
        "panels": panels,
        "shape": panels[0].shape,
        "wall_color": wall_color,
        "wall": wall,
        "objects": objects,
    }


def _cells(anchor: Coord, offsets: tuple[Coord, ...]) -> tuple[Coord, ...]:
    ar, ac = anchor
    return tuple((ar + dr, ac + dc) for dr, dc in offsets)


def _blocked(
    anchor: Coord,
    offsets: tuple[Coord, ...],
    shape: tuple[int, int],
    wall: np.ndarray,
) -> tuple[bool, bool, bool]:
    """Return (blocked, by_boundary, by_wall) in panel-local coordinates."""
    height, width = shape
    boundary = False
    wall_hit = False
    for r, c in _cells(anchor, offsets):
        if not (0 <= r < height and 0 <= c < width):
            boundary = True
        elif bool(wall[r, c]):
            wall_hit = True
    return boundary or wall_hit, boundary, wall_hit


def _advance(
    anchor: Coord,
    offsets: tuple[Coord, ...],
    velocity: Velocity,
    shape: tuple[int, int],
    wall: np.ndarray,
) -> tuple[Coord, Velocity, bool, bool] | None:
    """Advance once, reflecting exactly the velocity components causing a collision."""
    r, c = anchor
    dr, dc = velocity
    proposed = (r + dr, c + dc)
    hit, _boundary, wall_hit = _blocked(proposed, offsets, shape, wall)
    if not hit:
        return proposed, (dr, dc), False, False

    row_bad = False
    col_bad = False
    if dr:
        row_bad = _blocked((r + dr, c), offsets, shape, wall)[0]
    if dc:
        col_bad = _blocked((r, c + dc), offsets, shape, wall)[0]

    # At a diagonal corner neither component alone may collide; together they do,
    # so both nonzero components are causal and reflect.
    if not row_bad and not col_bad:
        row_bad = dr != 0
        col_bad = dc != 0
    ndr = -dr if row_bad else dr
    ndc = -dc if col_bad else dc
    moved = (r + ndr, c + ndc)
    if _blocked(moved, offsets, shape, wall)[0]:
        return None
    return moved, (ndr, ndc), True, bool(wall_hit)


def _initial_velocities(delta: Velocity) -> list[Velocity]:
    """A reflected step preserves each component's magnitude, leaving only signs unknown."""
    dr, dc = delta
    row_values = (0,) if dr == 0 else tuple(sorted({dr, -dr}))
    col_values = (0,) if dc == 0 else tuple(sorted({dc, -dc}))
    return [(r, c) for r in row_values for c in col_values if (r, c) != (0, 0)]


def _histories(
    anchors: list[Coord],
    order: tuple[int, int, int],
    offsets: tuple[Coord, ...],
    shape: tuple[int, int],
    wall: np.ndarray,
) -> list[dict]:
    """Enumerate motion states consistent with three ordered observed positions."""
    a0, a1, a2 = (anchors[i] for i in order)
    delta = (a1[0] - a0[0], a1[1] - a0[1])
    grouped: dict[tuple[Velocity, Coord], dict] = {}

    for initial in _initial_velocities(delta):
        first = _advance(a0, offsets, initial, shape, wall)
        if first is None or first[0] != a1:
            continue
        second = _advance(a1, offsets, first[1], shape, wall)
        if second is None or second[0] != a2:
            continue
        fourth = _advance(a2, offsets, second[1], shape, wall)
        if fourth is None:
            continue

        key = (second[1], fourth[0])
        record = grouped.setdefault(
            key,
            {
                "velocity": second[1],
                "next": fourth[0],
                "bounced": False,
                "wall_bounced": False,
            },
        )
        record["bounced"] = bool(record["bounced"] or first[2] or second[2])
        record["wall_bounced"] = bool(
            record["wall_bounced"] or first[3] or second[3]
        )
    return list(grouped.values())


def _qualified_orders(scene: dict) -> list[dict]:
    """Return orders satisfying every object and the stated historical-bounce clue."""
    solutions: list[dict] = []
    for order in itertools.permutations(range(3)):
        by_color: dict[int, list[dict]] = {}
        for color, obj in scene["objects"].items():
            histories = _histories(
                obj["anchors"], order, obj["offsets"], scene["shape"], scene["wall"]
            )
            if not histories:
                break
            by_color[color] = histories
        else:
            # The bounce clue is directional: an otherwise reversible ordering
            # that merely depicts straight travel is not the described history.
            if any(h["bounced"] for histories in by_color.values() for h in histories):
                solutions.append({"order": order, "histories": by_color})
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the uniquely bounce-bearing order of three shuffled motion frames and extrapolate frame four.

    Input structure:
    - Two exclusive full-height divider columns split the grid into three equal panels.
    - Every panel contains the same fixed monochrome wall and the same set of
      small monochrome moving objects; each object keeps its color and shape.
    - The panels use their own shared local row-column coordinates, and their
      left-to-right arrangement is not the temporal order.

    Rule:
    - Test all six panel orders.  For every object, infer a constant component
      magnitude from consecutive translations; if its next translated cells
      would overlap the local panel boundary or wall, reverse exactly the
      colliding velocity component before moving.
    - Keep the sole order in which all objects obey one motion state and at
      least one object has a demonstrable bounce in the observed history.
    - Advance every object once more from that order's last panel, applying the
      same collision rule, and return one panel containing the unchanged wall.

    Key insight:
    - The panels are frames rather than adjacent space: all matching, collision,
      and extrapolation must use panel-local coordinates, while the historical
      bounce clue breaks the otherwise tempting reversed or spatial ordering.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    solutions = _qualified_orders(scene)
    assert len(solutions) == 1, "the complete motion-and-bounce history must have one time order"
    solution = solutions[0]

    predictions: dict[int, Coord] = {}
    for color, histories in solution["histories"].items():
        possible = {h["next"] for h in histories}
        assert len(possible) == 1, "the fourth-frame position must be uniquely determined"
        predictions[color] = next(iter(possible))

    out = np.full(scene["shape"], scene["bg"], dtype=int)
    out[scene["wall"]] = scene["wall_color"]
    occupied = scene["wall"].copy()
    for color, obj in scene["objects"].items():
        cells = _cells(predictions[color], obj["offsets"])
        for r, c in cells:
            assert not occupied[r, c], "objects may not overlap in the extrapolated frame"
            occupied[r, c] = True
            out[r, c] = int(color)
    return out


# ---------------------------------------------------------------------------
# Random instance generator
# ---------------------------------------------------------------------------


_PATTERNS: tuple[tuple[Coord, ...], ...] = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _path(
    start: Coord,
    offsets: tuple[Coord, ...],
    velocity: Velocity,
    shape: tuple[int, int],
    wall: np.ndarray,
    steps: int = 3,
) -> tuple[list[Coord], list[dict]] | None:
    anchors = [start]
    events: list[dict] = []
    anchor = start
    current = velocity
    for _ in range(steps):
        advanced = _advance(anchor, offsets, current, shape, wall)
        if advanced is None:
            return None
        anchor, current, bounced, wall_bounced = advanced
        anchors.append(anchor)
        events.append({"bounced": bounced, "wall_bounced": wall_bounced})
    return anchors, events


def _near(cells: tuple[Coord, ...], occupied: set[Coord], gap: int = 1) -> bool:
    for r, c in cells:
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if (rr, cc) in occupied:
                    return True
    return False


def _make_wall(shape: tuple[int, int]) -> tuple[np.ndarray, str, int, int]:
    height, width = shape
    wall = np.zeros(shape, dtype=bool)
    orientation = random.choice(("vertical", "horizontal"))
    if orientation == "vertical":
        wall_axis = random.choice(("left", "right"))
        if wall_axis == "left":
            fixed = random.randint(4, width - 2)
        else:
            fixed = random.randint(1, width - 5)
        length = random.randint(4, height - 4)
        start = random.randint(2, height - length - 2)
        wall[start : start + length, fixed] = True
    else:
        wall_axis = random.choice(("top", "bottom"))
        if wall_axis == "top":
            fixed = random.randint(4, height - 2)
        else:
            fixed = random.randint(1, height - 5)
        length = random.randint(4, width - 2)
        start = random.randint(1, width - length - 1)
        wall[fixed, start : start + length] = True
    return wall, orientation, fixed, start


def _primary_bouncer(
    shape: tuple[int, int], wall: np.ndarray, orientation: str, fixed: int, start: int
) -> tuple[tuple[Coord, ...], list[Coord], list[dict]] | None:
    """Place one object so the first observed move necessarily can bounce off the wall."""
    height, width = shape
    if orientation == "vertical":
        offsets = random.choice((((0, 0),), ((0, 0), (1, 0))))
        wall_rows = np.flatnonzero(wall[:, fixed])
        r_lo = int(wall_rows.min())
        r_hi = int(wall_rows.max()) - max(r for r, _c in offsets)
        if r_lo > r_hi:
            return None
        r = random.randint(r_lo, r_hi)
        if fixed >= 4 and (fixed > width - 5 or random.random() < 0.5):
            anchor = (r, fixed - 1)
            velocity = (0, 1)
        else:
            anchor = (r, fixed + 1)
            velocity = (0, -1)
    else:
        offsets = random.choice((((0, 0),), ((0, 0), (0, 1))))
        wall_cols = np.flatnonzero(wall[fixed, :])
        c_lo = int(wall_cols.min())
        c_hi = int(wall_cols.max()) - max(c for _r, c in offsets)
        if c_lo > c_hi:
            return None
        c = random.randint(c_lo, c_hi)
        if fixed >= 4 and (fixed > height - 5 or random.random() < 0.5):
            anchor = (fixed - 1, c)
            velocity = (1, 0)
        else:
            anchor = (fixed + 1, c)
            velocity = (-1, 0)

    result = _path(anchor, offsets, velocity, shape, wall)
    if result is None:
        return None
    anchors, events = result
    if not events[0]["wall_bounced"] or events[1]["bounced"] or events[2]["bounced"]:
        return None
    return offsets, anchors, events


def _place_extra(
    shape: tuple[int, int],
    wall: np.ndarray,
    occupied_by_frame: list[set[Coord]],
) -> tuple[tuple[Coord, ...], list[Coord], list[dict]] | None:
    height, width = shape
    for _ in range(250):
        offsets = random.choice(_PATTERNS)
        obj_h = max(r for r, _c in offsets) + 1
        obj_w = max(c for _r, c in offsets) + 1
        start = (random.randrange(height - obj_h + 1), random.randrange(width - obj_w + 1))
        velocity = random.choice(
            [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        )
        result = _path(start, offsets, velocity, shape, wall)
        if result is None:
            continue
        anchors, events = result
        # The designated wall object supplies the historical clue.  Straight
        # extras prevent accidental second bounce-bearing time orders.
        if any(event["bounced"] for event in events):
            continue
        inferred = _histories(anchors[:3], (0, 1, 2), offsets, shape, wall)
        if not inferred or any(history["bounced"] for history in inferred):
            # Reject even a counterfactual initial sign that could explain the
            # same visible translations by an unobserved first-step bounce.
            continue

        paths = [_cells(anchor, offsets) for anchor in anchors]
        if any(any(wall[r, c] for r, c in cells) for cells in paths):
            continue
        if any(_near(cells, occupied_by_frame[t], gap=1) for t, cells in enumerate(paths)):
            continue
        return offsets, anchors, events
    return None


def _render_panel(
    shape: tuple[int, int],
    bg: int,
    wall_color: int,
    wall: np.ndarray,
    specs: list[dict],
    frame: int,
) -> np.ndarray:
    panel = np.full(shape, bg, dtype=int)
    panel[wall] = wall_color
    for spec in specs:
        for r, c in _cells(spec["anchors"][frame], spec["offsets"]):
            assert int(panel[r, c]) == bg
            panel[r, c] = spec["color"]
    return panel


def _generation_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(10, 14)
    panel_width = random.choice((8, 9))
    shape = (height, panel_width)
    object_count = random.randint(2, 4)
    palette = random.sample(range(10), object_count + 3)
    bg, divider, wall_color = palette[:3]
    object_colors = palette[3:]

    wall, orientation, fixed, start = _make_wall(shape)
    primary = _primary_bouncer(shape, wall, orientation, fixed, start)
    if primary is None:
        return None
    offsets, anchors, events = primary
    specs = [
        {"color": object_colors[0], "offsets": offsets, "anchors": anchors, "events": events}
    ]

    occupied_by_frame = [set(_cells(anchors[t], offsets)) for t in range(4)]
    for color in object_colors[1:]:
        extra = _place_extra(shape, wall, occupied_by_frame)
        if extra is None:
            return None
        offsets, anchors, events = extra
        specs.append({"color": color, "offsets": offsets, "anchors": anchors, "events": events})
        for t in range(4):
            occupied_by_frame[t].update(_cells(anchors[t], offsets))

    frames = [_render_panel(shape, bg, wall_color, wall, specs, t) for t in range(4)]
    physical = random.choice([p for p in itertools.permutations(range(3)) if p != (0, 1, 2)])
    inp = np.full((height, 3 * panel_width + 2), bg, dtype=int)
    inp[:, panel_width] = divider
    inp[:, 2 * panel_width + 1] = divider
    for slot, time_index in enumerate(physical):
        c0 = slot * (panel_width + 1)
        inp[:, c0 : c0 + panel_width] = frames[time_index]

    if _background(inp) != bg:
        return None
    try:
        scene = _parse_scene(inp)
        solutions = _qualified_orders(scene)
        if len(solutions) != 1:
            return None
        intended_order = tuple(physical.index(t) for t in range(3))
        if solutions[0]["order"] != intended_order:
            return None
        if not any(
            h["wall_bounced"]
            for histories in solutions[0]["histories"].values()
            for h in histories
        ):
            return None
        solved = main(inp)
    except (AssertionError, IndexError, ValueError):
        return None
    if not np.array_equal(solved, frames[3]):
        return None
    if any(np.array_equal(solved, panel) for panel in scene["panels"]):
        return None
    return inp, frames[3]


def generate_input() -> np.ndarray:
    """Generate a varied, shuffled, uniquely solvable three-frame motion scene."""
    for _ in range(500):
        result = _generation_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a unique shuffled bounce-history instance")
