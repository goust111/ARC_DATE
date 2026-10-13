import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7
TEAL = 8
MAROON = 9

OBJECT_COLORS = (BLUE, ORANGE, TEAL, MAROON)
SPACINGS = (3, 4)
TRACK_LENGTHS = (3, 4, 5)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, *, colors: set[int], bg: int = BLACK) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg or color not in colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(sorted(coords), dtype=int)
            out.append({"color": color, "coords": arr, "area": int(arr.shape[0])})
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _coords_from_anchor(anchor: tuple[int, int], rels: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    ar, ac = anchor
    return tuple(sorted((ar + dr, ac + dc) for dr, dc in rels))


def _shift_coords(coords: tuple[tuple[int, int], ...], step: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    dr, dc = step
    return tuple(sorted((r + dr, c + dc) for r, c in coords))


def _coords_fit(shape: tuple[int, int], coords: tuple[tuple[int, int], ...]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in coords)


def _build_shape_library() -> list[tuple[tuple[int, int], ...]]:
    bases = [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, -1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ]
    seen = set()
    out = []
    for shape in bases:
        pts = np.array(shape, dtype=int)
        variants = []
        cur = pts
        for _ in range(4):
            variants.append(cur.copy())
            cur = np.stack([-cur[:, 1], cur[:, 0]], axis=1)
        for var in variants:
            for ar, ac in var:
                rels = tuple(sorted((int(r - ar), int(c - ac)) for r, c in var))
                if rels not in seen:
                    seen.add(rels)
                    out.append(rels)
    out.sort(key=lambda rels: (len(rels), rels))
    return out


SHAPE_LIBRARY = _build_shape_library()


def _track_slots(
    anchor: tuple[int, int],
    step: tuple[int, int],
    length: int,
    index: int,
) -> tuple[tuple[int, int], ...]:
    ar, ac = anchor
    dr, dc = step
    return tuple((ar + (i - index) * dr, ac + (i - index) * dc) for i in range(length))


def _iter_track_candidates(
    shape: tuple[int, int],
    green_set: set[tuple[int, int]],
    component: dict,
) -> list[dict]:
    h, w = shape
    coords = [tuple(map(int, rc)) for rc in component["coords"]]
    candidates = {}
    for anchor in coords:
        for orient in ("h", "v"):
            for spacing in SPACINGS:
                step = (0, spacing) if orient == "h" else (spacing, 0)
                for length in TRACK_LENGTHS:
                    for index in range(length):
                        slots = _track_slots(anchor, step, length, index)
                        if not all(_in_bounds(shape, r, c) for r, c in slots):
                            continue
                        visible = tuple(slot for i, slot in enumerate(slots) if i != index)
                        if not all(slot in green_set for slot in visible):
                            continue
                        key = (slots, anchor, orient, spacing, length, index)
                        candidates[key] = {
                            "anchor": anchor,
                            "orient": orient,
                            "spacing": spacing,
                            "length": length,
                            "index": index,
                            "step": step,
                            "slots": frozenset(slots),
                            "visible": frozenset(visible),
                        }
    return list(candidates.values())


def _resolve_tracks(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BLACK
    objects = _find_components(g, colors=set(OBJECT_COLORS), bg=BLACK)
    assert 3 <= len(objects) <= 4
    green_set = {tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)}
    assert green_set

    candidates_per_object = []
    for obj in objects:
        cands = _iter_track_candidates(g.shape, green_set, obj)
        assert cands, "every object must determine a track"
        candidates_per_object.append(cands)

    order = sorted(range(len(objects)), key=lambda i: len(candidates_per_object[i]))
    solutions: list[list[dict]] = []

    def backtrack(pos: int, used_visible: set[tuple[int, int]], used_slots: set[tuple[int, int]], chosen: list[dict]) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            if used_visible == green_set:
                arranged = [None] * len(objects)
                for item in chosen:
                    arranged[item["object_index"]] = item
                solutions.append(arranged)
            return

        obj_idx = order[pos]
        for cand in candidates_per_object[obj_idx]:
            if cand["visible"] & used_visible:
                continue
            if cand["slots"] & used_slots:
                continue
            chosen.append({**cand, "object_index": obj_idx})
            backtrack(pos + 1, used_visible | set(cand["visible"]), used_slots | set(cand["slots"]), chosen)
            chosen.pop()

    backtrack(0, set(), set(), [])
    assert len(solutions) == 1, "track assignment must be unique"
    return objects, solutions[0]


def _simulate(objects: list[dict], tracks: list[dict], shape: tuple[int, int]) -> tuple[list[tuple[tuple[int, int], ...]], set[int]]:
    moved_coords: list[tuple[tuple[int, int], ...]] = []
    h, w = shape
    for obj, track in zip(objects, tracks):
        coords = tuple(tuple(map(int, rc)) for rc in obj["coords"])
        step = track["step"]
        shifted = _shift_coords(coords, step)
        if all(0 <= r < h and 0 <= c < w for r, c in shifted):
            moved_coords.append(shifted)
        else:
            moved_coords.append(tuple(sorted(coords)))

    occupancy: dict[tuple[int, int], list[int]] = {}
    for idx, coords in enumerate(moved_coords):
        for cell in coords:
            occupancy.setdefault(cell, []).append(idx)

    collided = set()
    for idxs in occupancy.values():
        if len(idxs) > 1:
            collided.update(idxs)
    return moved_coords, collided


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each track object forward by one slot and repaint the ones that collide.

    Input structure:
    - A black background contains 3-4 axis-aligned tracks made from equally spaced green single-pixel slots.
    - Each track has exactly one small monochrome 4-connected object in blue, orange, teal, or maroon covering one slot.
    - Tracks may be horizontal or vertical, use spacing 3 or 4, and never share a slot.

    Rule:
    - Reconstruct every track, including the slot hidden under its object, then determine the track direction: horizontal moves right, vertical moves down.
    - Move every object simultaneously by exactly one slot spacing; if that translation would send any pixel out of bounds, keep that object in place.
    - After all moves, any objects whose new pixels overlap are recolored red; non-colliding objects keep their original colors, and all green track slots remain present underneath.

    Key insight:
    - Collision is checked only after all objects finish their one-slot move, so the solver must infer the hidden slots, simulate the simultaneous translation, and only then decide which colors turn red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BLACK

    objects, tracks = _resolve_tracks(g)
    moved_coords, collided = _simulate(objects, tracks, g.shape)

    out = np.full_like(g, BLACK)
    for track in tracks:
        for r, c in track["slots"]:
            out[r, c] = GREEN
    for idx, obj in enumerate(objects):
        color = RED if idx in collided else int(obj["color"])
        for r, c in moved_coords[idx]:
            out[r, c] = color
    return out


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        for r, c in spec["visible_slots"]:
            grid[r, c] = GREEN
    for spec in specs:
        for r, c in spec["input_cells"]:
            grid[r, c] = spec["color"]
    return grid


def _track_param_options(
    shape: tuple[int, int],
    rels: tuple[tuple[int, int], ...],
    final_anchor: tuple[int, int],
    orient: str,
    spacing: int,
    movable: bool,
) -> list[dict]:
    step = (0, spacing) if orient == "h" else (spacing, 0)
    if movable:
        input_anchor = (final_anchor[0] - step[0], final_anchor[1] - step[1])
        if not _coords_fit(shape, _coords_from_anchor(input_anchor, rels)):
            return []
    else:
        input_anchor = final_anchor
        trial = _coords_from_anchor((input_anchor[0] + step[0], input_anchor[1] + step[1]), rels)
        if _coords_fit(shape, trial):
            return []

    input_cells = _coords_from_anchor(input_anchor, rels)
    final_cells = _coords_from_anchor(final_anchor, rels)
    if not _coords_fit(shape, input_cells) or not _coords_fit(shape, final_cells):
        return []

    options = []
    for length in TRACK_LENGTHS:
        index_values = range(length - 1) if movable else [length - 1]
        for index in index_values:
            slots = _track_slots(input_anchor, step, length, index)
            if not all(_in_bounds(shape, r, c) for r, c in slots):
                continue
            visible_slots = tuple(slot for i, slot in enumerate(slots) if i != index)
            options.append(
                {
                    "orient": orient,
                    "spacing": spacing,
                    "step": step,
                    "length": length,
                    "index": index,
                    "input_anchor": input_anchor,
                    "final_anchor": final_anchor,
                    "input_cells": input_cells,
                    "final_cells": final_cells,
                    "slots": frozenset(slots),
                    "visible_slots": frozenset(visible_slots),
                    "movable": movable,
                    "rels": rels,
                }
            )
    return options


def _input_clear(spec: dict, placed_specs: list[dict]) -> bool:
    input_cells = set(spec["input_cells"])
    for other in placed_specs:
        if input_cells & set(other["input_cells"]):
            return False
    for other in placed_specs:
        if input_cells & set(other["visible_slots"]):
            return False
        if set(spec["visible_slots"]) & set(other["input_cells"]):
            return False
        if spec["slots"] & other["slots"]:
            return False
    return True


def _future_safe(spec: dict, placed_specs: list[dict]) -> bool:
    final_cells = set(spec["final_cells"])
    return all(not (final_cells & set(other["final_cells"])) for other in placed_specs)


def _tracks_isolated(specs: list[dict]) -> bool:
    for i, a in enumerate(specs):
        for b in specs[i + 1 :]:
            if a["orient"] == "h" and b["orient"] == "h":
                if a["input_anchor"][0] == b["input_anchor"][0]:
                    return False
            elif a["orient"] == "v" and b["orient"] == "v":
                if a["input_anchor"][1] == b["input_anchor"][1]:
                    return False
            elif a["orient"] == "h" and b["orient"] == "v":
                if a["input_anchor"][0] in {r for r, _ in b["slots"]}:
                    return False
                if b["input_anchor"][1] in {c for _, c in a["slots"]}:
                    return False
            else:
                if b["input_anchor"][0] in {r for r, _ in a["slots"]}:
                    return False
                if a["input_anchor"][1] in {c for _, c in b["slots"]}:
                    return False
    return True


PAIR_RELS = ((0, 0), (0, 1), (1, 0), (1, 1))
SAFE_RELS = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
)


def _build_exact_spec(
    shape: tuple[int, int],
    rels: tuple[tuple[int, int], ...],
    color: int,
    orient: str,
    spacing: int,
    length: int,
    index: int,
    input_anchor: tuple[int, int],
) -> dict | None:
    step = (0, spacing) if orient == "h" else (spacing, 0)
    slots = _track_slots(input_anchor, step, length, index)
    if not all(_in_bounds(shape, r, c) for r, c in slots):
        return None

    input_cells = _coords_from_anchor(input_anchor, rels)
    if not _coords_fit(shape, input_cells):
        return None

    movable = index < length - 1
    if movable:
        final_anchor = (input_anchor[0] + step[0], input_anchor[1] + step[1])
    else:
        final_anchor = input_anchor
        trial = _coords_from_anchor((input_anchor[0] + step[0], input_anchor[1] + step[1]), rels)
        if _coords_fit(shape, trial):
            return None
    final_cells = _coords_from_anchor(final_anchor, rels)
    if not _coords_fit(shape, final_cells):
        return None

    return {
        "color": color,
        "orient": orient,
        "spacing": spacing,
        "step": step,
        "length": length,
        "index": index,
        "input_anchor": input_anchor,
        "final_anchor": final_anchor,
        "input_cells": input_cells,
        "final_cells": final_cells,
        "slots": frozenset(slots),
        "visible_slots": frozenset(slot for i, slot in enumerate(slots) if i != index),
        "movable": movable,
        "rels": rels,
    }


def _make_pair_exact(shape: tuple[int, int], colors: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    for _ in range(800):
        spacing1 = random.choice(SPACINGS)
        spacing2 = random.choice(SPACINGS)
        length1 = random.choice((4, 5))
        length2 = random.choice((4, 5))
        index1 = random.randint(1, length1 - 2)
        index2 = random.randint(1, length2 - 2)
        input_anchor1 = (random.randint(0, h - 1), random.randint(0, w - 1))
        spec1 = _build_exact_spec(shape, PAIR_RELS, colors[0], "h", spacing1, length1, index1, input_anchor1)
        if spec1 is None:
            continue
        final_anchor1 = spec1["final_anchor"]
        input_anchor2 = (final_anchor1[0] + 1 - spacing2, final_anchor1[1] + 1)
        spec2 = _build_exact_spec(shape, PAIR_RELS, colors[1], "v", spacing2, length2, index2, input_anchor2)
        if spec2 is None:
            continue
        if len(set(spec1["final_cells"]) & set(spec2["final_cells"])) != 1:
            continue
        if not _input_clear(spec2, [spec1]):
            continue
        if not _tracks_isolated([spec1, spec2]):
            continue
        return [spec1, spec2]
    return None


def _make_safe_exact(
    shape: tuple[int, int],
    color: int,
    placed: list[dict],
    *,
    movable: bool,
) -> dict | None:
    h, w = shape
    for _ in range(800):
        rels = random.choice(SAFE_RELS)
        orient = random.choice(("h", "v"))
        spacing = random.choice(SPACINGS)
        if movable:
            length = random.choice((4, 5))
            index = random.randint(1, length - 2)
            input_anchor = (random.randint(0, h - 1), random.randint(0, w - 1))
        else:
            length = random.choice(TRACK_LENGTHS)
            index = length - 1
            if orient == "h":
                input_anchor = (random.randint(0, h - 1), random.randint(w - spacing, w - 1))
            else:
                input_anchor = (random.randint(h - spacing, h - 1), random.randint(0, w - 1))
        spec = _build_exact_spec(shape, rels, color, orient, spacing, length, index, input_anchor)
        if spec is None:
            continue
        if not _input_clear(spec, placed):
            continue
        if not _future_safe(spec, placed):
            continue
        if not _tracks_isolated(placed + [spec]):
            continue
        return spec
    return None


def _make_collision_pair(shape: tuple[int, int], colors: tuple[int, ...], orientations: list[str]) -> list[dict] | None:
    h, w = shape
    for _ in range(500):
        rels1 = random.choice(SHAPE_LIBRARY)
        rels2 = random.choice(SHAPE_LIBRARY)
        orient1, orient2 = orientations[0], orientations[1]
        spacing1 = random.choice(SPACINGS)
        spacing2 = random.choice(SPACINGS)
        final_anchor1 = (random.randint(1, h - 2), random.randint(1, w - 2))
        overlap1 = random.choice(rels1)
        overlap2 = random.choice(rels2)
        final_anchor2 = (
            final_anchor1[0] + overlap1[0] - overlap2[0],
            final_anchor1[1] + overlap1[1] - overlap2[1],
        )
        opts1 = _track_param_options(shape, rels1, final_anchor1, orient1, spacing1, movable=True)
        opts2 = _track_param_options(shape, rels2, final_anchor2, orient2, spacing2, movable=True)
        opts1 = [opt for opt in opts1 if opt["length"] >= 4]
        opts2 = [opt for opt in opts2 if opt["length"] >= 4]
        if not opts1 or not opts2:
            continue
        random.shuffle(opts1)
        random.shuffle(opts2)
        for a in opts1[:25]:
            for b in opts2[:25]:
                inter_final = set(a["final_cells"]) & set(b["final_cells"])
                if not (1 <= len(inter_final) <= 2):
                    continue
                if set(a["input_cells"]) & set(b["input_cells"]):
                    continue
                if set(a["input_cells"]) & set(b["visible_slots"]):
                    continue
                if set(b["input_cells"]) & set(a["visible_slots"]):
                    continue
                if a["slots"] & b["slots"]:
                    continue
                sa = dict(a)
                sb = dict(b)
                sa["color"] = colors[0]
                sb["color"] = colors[1]
                return [sa, sb]
    return None


def _make_safe_spec(
    shape: tuple[int, int],
    color: int,
    placed_specs: list[dict],
    orient: str,
    allow_immobile: bool,
) -> dict | None:
    h, w = shape
    modes = [False, True] if allow_immobile else [False]
    random.shuffle(modes)
    for movable in modes:
        for _ in range(500):
            rels = random.choice(SHAPE_LIBRARY)
            spacing = random.choice(SPACINGS)
            if movable:
                final_anchor = (random.randint(1, h - 2), random.randint(1, w - 2))
            else:
                if orient == "h":
                    final_anchor = (random.randint(1, h - 2), random.randint(w - spacing, w - 1))
                else:
                    final_anchor = (random.randint(h - spacing, h - 1), random.randint(1, w - 2))
            options = _track_param_options(shape, rels, final_anchor, orient, spacing, movable=movable)
            if movable:
                options = [opt for opt in options if opt["length"] >= 4]
            if not options:
                continue
            random.shuffle(options)
            for spec in options[:40]:
                if not _input_clear(spec, placed_specs):
                    continue
                if not _future_safe(spec, placed_specs):
                    continue
                out = dict(spec)
                out["color"] = color
                return out
    return None


def _object_and_collision_profile(grid: np.ndarray) -> tuple[int, int, int]:
    objects, tracks = _resolve_tracks(grid)
    moved, collided = _simulate(objects, tracks, grid.shape)
    static_count = sum(
        1
        for obj, coords in zip(objects, moved)
        if tuple(sorted(map(tuple, obj["coords"]))) == coords
    )
    return len(objects), len(collided), static_count


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 17)
    w = random.randint(13, 17)
    shape = (h, w)
    want_static = random.random() < 0.45
    object_count = 4 if want_static else 3
    colors = random.sample(list(OBJECT_COLORS), object_count)

    pair = _make_pair_exact(shape, (colors[0], colors[1]))
    if pair is None:
        return None

    placed = pair[:]
    safe = _make_safe_exact(shape, colors[2], placed, movable=True)
    if safe is None:
        return None
    placed.append(safe)
    if want_static:
        static_spec = _make_safe_exact(shape, colors[3], placed, movable=False)
        if static_spec is None:
            return None
        placed.append(static_spec)

    grid = _render_input(placed, shape)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None

    try:
        obj_count, collided_count, static_count = _object_and_collision_profile(grid)
    except Exception:
        return None
    if obj_count != object_count:
        return None
    if collided_count < 2 or collided_count >= object_count:
        return None
    if want_static and static_count == 0:
        return None
    if not want_static and static_count != 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _gen_attempt()
        if grid is None:
            continue
        return grid
    raise RuntimeError("failed to generate a valid orbit-collision sample")
