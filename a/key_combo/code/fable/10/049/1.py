from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MARKER = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
SIDES = ("up", "down", "left", "right")

MAX_AREA = 4
MIN_OBJECTS = 3
MAX_OBJECTS = 4
INPUT_FARTHER_LIMIT = 3
INPUT_CLOSER_LIMIT = 2
GEN_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([int(dr), int(dc)], dtype=int)


def _coords_in_bounds(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return (
        int(coords[:, 0].min()) >= 0
        and int(coords[:, 1].min()) >= 0
        and int(coords[:, 0].max()) < h
        and int(coords[:, 1].max()) < w
    )


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    todo = deque([start])
    seen = {start}
    h, w = mask.shape
    while todo:
        r, c = todo.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return len(seen) == int(mask.sum())


def _classify_side(coords: np.ndarray, marker: tuple[int, int]) -> str:
    mr, mc = marker
    r0, c0, r1, c1 = _bbox(coords)
    matches = []
    if r1 < mr and c0 <= mc <= c1:
        matches.append("up")
    if r0 > mr and c0 <= mc <= c1:
        matches.append("down")
    if c1 < mc and r0 <= mr <= r1:
        matches.append("left")
    if c0 > mc and r0 <= mr <= r1:
        matches.append("right")
    assert len(matches) == 1, "each object must align with the marker on exactly one side"
    return matches[0]


def _gap_to_marker(coords: np.ndarray, side: str, marker: tuple[int, int]) -> int:
    mr, mc = marker
    r0, c0, r1, c1 = _bbox(coords)
    if side == "up":
        return mr - r1 - 1
    if side == "down":
        return r0 - mr - 1
    if side == "left":
        return mc - c1 - 1
    return c0 - mc - 1


def _toward_marker_step(side: str) -> tuple[int, int]:
    if side == "up":
        return 1, 0
    if side == "down":
        return -1, 0
    if side == "left":
        return 0, 1
    return 0, -1


def _away_step(side: str) -> tuple[int, int]:
    dr, dc = _toward_marker_step(side)
    return -dr, -dc


def _ring_position(coords: np.ndarray, side: str, marker: tuple[int, int], area: int) -> np.ndarray:
    current_gap = _gap_to_marker(coords, side, marker)
    dr, dc = _toward_marker_step(side)
    shift = current_gap - int(area)
    return _translate(coords, dr * shift, dc * shift)


def _extract_objects(grid: np.ndarray) -> tuple[tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    marker_cells = np.argwhere(g == MARKER)
    assert marker_cells.shape == (1, 2), "expected exactly one red marker pixel"
    marker = tuple(map(int, marker_cells[0]))

    objects = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BG, MARKER)):
        mask = g == color
        assert _is_connected(mask), "each non-marker color must form one 4-connected object"
        coords = np.argwhere(mask).astype(int)
        objects.append({"color": color, "coords": coords, "area": int(coords.shape[0])})

    assert objects, "expected at least one movable object"
    assert len({obj["color"] for obj in objects}) == len(objects), "object colors must be distinct"
    return marker, objects


def _settle_objects(
    objects: list[dict],
    shape: tuple[int, int],
    marker: tuple[int, int],
) -> tuple[list[dict], list[dict]]:
    areas = [int(obj["area"]) for obj in objects]
    assert len(set(areas)) == len(areas), "object areas must be unique for a unique settling order"

    occupied = np.zeros(shape, dtype=bool)
    occupied[marker] = True
    settled = []
    traces = []
    for obj in sorted(objects, key=lambda item: int(item["area"])):
        color = int(obj["color"])
        start = np.asarray(obj["coords"], dtype=int)
        side = _classify_side(start, marker)
        gap_before = _gap_to_marker(start, side, marker)
        ring = _ring_position(start, side, marker, int(obj["area"]))
        final = ring.copy()
        step_r, step_c = _away_step(side)
        yield_steps = 0
        while np.any(occupied[final[:, 0], final[:, 1]]):
            final = _translate(final, step_r, step_c)
            yield_steps += 1
            assert _coords_in_bounds(final, shape), "settled object moved out of bounds"
        occupied[final[:, 0], final[:, 1]] = True
        settled.append({"color": color, "coords": final, "area": int(obj["area"])})
        traces.append(
            {
                "color": color,
                "side": side,
                "area": int(obj["area"]),
                "gap_before": gap_before,
                "yield_steps": yield_steps,
                "ring_coords": ring,
                "final_coords": final,
            }
        )
    return settled, traces


def _render(shape: tuple[int, int], marker: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[marker] = MARKER
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each aligned object onto its area-sized stopping ring around the red marker.

    Input structure:
    - The grid has a black background with exactly one red marker pixel.
    - Every other color forms one monochromatic 4-connected object, and each object lies strictly
      above, below, left, or right of the marker while still sharing the marker's column or row.
    - Object colors are all distinct, so each component is visually separable.

    Rule:
    - Process objects from smallest area to largest area.
    - Each object translates only along the row or column that aligns it with the marker, staying on
      the same side of the marker the whole time.
    - Move the object so its nearest edge is exactly its own area cells away from the marker; if that
      ring position overlaps any already settled smaller object, keep shifting it farther outward until
      the first non-overlapping position is reached. Clear the original object cells and keep the red
      marker fixed.

    Key insight:
    - The invisible docking rings are keyed by object area rather than attraction: far objects move
      inward, near objects back away, and only smaller objects get to claim their ring before larger
      ones are forced outward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    marker, objects = _extract_objects(g)
    settled, _ = _settle_objects(objects, g.shape, marker)
    return _render(g.shape, marker, settled)


def _normalize_with_anchor(cells: set[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    anchor = (-min_r, -min_c)
    return np.asarray(norm, dtype=int), (int(anchor[0]), int(anchor[1]))


def _random_polyomino(area: int, bias: str) -> tuple[np.ndarray, tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            frontier.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
        frontier.difference_update(cells)
        weighted = []
        for cand in frontier:
            r, c = cand
            if bias == "vertical":
                weight = 1 + abs(r)
            elif bias == "horizontal":
                weight = 1 + abs(c)
            else:
                weight = 1
            weighted.extend([cand] * weight)
        cells.add(random.choice(weighted))
    return _normalize_with_anchor(cells)


def _line_shape(area: int, vertical: bool) -> tuple[np.ndarray, tuple[int, int]]:
    if vertical:
        coords = {(r, 0) for r in range(area)}
    else:
        coords = {(0, c) for c in range(area)}
    return _normalize_with_anchor(coords)


def _make_shape(area: int, side: str, forced_line: bool) -> tuple[np.ndarray, tuple[int, int]]:
    if forced_line:
        return _line_shape(area, vertical=side in ("up", "down"))
    if random.random() < 0.35:
        return _line_shape(area, vertical=side in ("up", "down"))
    bias = "vertical" if side in ("up", "down") else "horizontal"
    if random.random() < 0.25:
        bias = "mixed"
    return _random_polyomino(area, bias)


def _extent_along_motion(local_coords: np.ndarray, side: str) -> int:
    if side in ("up", "down"):
        return int(local_coords[:, 0].max() - local_coords[:, 0].min())
    return int(local_coords[:, 1].max() - local_coords[:, 1].min())


def _place_local_on_ring(
    local_coords: np.ndarray,
    anchor: tuple[int, int],
    side: str,
    area: int,
    marker: tuple[int, int],
) -> np.ndarray:
    mr, mc = marker
    coords = local_coords.copy().astype(int)
    anchor_r, anchor_c = anchor
    if side in ("up", "down"):
        col_shift = mc - anchor_c
        row_shift = mr - area - 1 - int(coords[:, 0].max()) if side == "up" else mr + area + 1
        return _translate(coords, row_shift, col_shift)
    row_shift = mr - anchor_r
    col_shift = mc - area - 1 - int(coords[:, 1].max()) if side == "left" else mc + area + 1
    return _translate(coords, row_shift, col_shift)


def _choose_object_specs() -> tuple[list[dict], bool]:
    want_yield = random.random() < 0.6
    count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    if want_yield:
        pair = random.choice([(2, 3), (3, 4)])
        remaining = [area for area in range(1, MAX_AREA + 1) if area not in pair]
        areas = list(pair) + random.sample(remaining, count - 2)
        collision_side = random.choice(SIDES)
        other_sides = random.sample([side for side in SIDES if side != collision_side], count - 2)
        side_by_area = {pair[0]: collision_side, pair[1]: collision_side}
        for area, side in zip(sorted(a for a in areas if a not in pair), other_sides):
            side_by_area[area] = side
        forced_line_areas = set(pair)
    else:
        areas = random.sample(range(1, MAX_AREA + 1), count)
        sides = random.sample(SIDES, count)
        side_by_area = {area: side for area, side in zip(sorted(areas), sides)}
        forced_line_areas = set()

    colors = random.sample(PALETTE, count)
    specs = []
    for color, area in zip(colors, sorted(areas)):
        side = side_by_area[area]
        local_coords, anchor = _make_shape(area, side, area in forced_line_areas)
        specs.append(
            {
                "color": color,
                "area": area,
                "side": side,
                "local_coords": local_coords,
                "anchor": anchor,
            }
        )
    return specs, want_yield


def _choose_shape_and_marker(specs: list[dict]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    margins = {"up": 3, "down": 3, "left": 3, "right": 3}
    for spec in specs:
        extent = _extent_along_motion(spec["local_coords"], spec["side"])
        need = int(spec["area"]) + extent + max(INPUT_FARTHER_LIMIT, 3) + 1
        margins[spec["side"]] = max(margins[spec["side"]], need)

    min_h = margins["up"] + margins["down"] + 1
    min_w = margins["left"] + margins["right"] + 1
    if min_h > 30 or min_w > 30:
        return None

    h_lo, h_hi = max(18, min_h), min(26, 30)
    w_lo, w_hi = max(18, min_w), min(26, 30)
    if h_lo > h_hi or w_lo > w_hi:
        return None
    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)
    if margins["up"] > h - margins["down"] - 1 or margins["left"] > w - margins["right"] - 1:
        return None
    marker = (
        random.randint(margins["up"], h - margins["down"] - 1),
        random.randint(margins["left"], w - margins["right"] - 1),
    )
    return (h, w), marker


def _build_output(specs: list[dict], shape: tuple[int, int], marker: tuple[int, int]) -> tuple[list[dict], list[dict]] | None:
    occupied = np.zeros(shape, dtype=bool)
    occupied[marker] = True
    objects = []
    traces = []
    for spec in sorted(specs, key=lambda item: int(item["area"])):
        ring = _place_local_on_ring(spec["local_coords"], spec["anchor"], spec["side"], int(spec["area"]), marker)
        final = ring.copy()
        dr, dc = _away_step(spec["side"])
        yield_steps = 0
        while np.any(occupied[final[:, 0], final[:, 1]]):
            final = _translate(final, dr, dc)
            yield_steps += 1
            if not _coords_in_bounds(final, shape):
                return None
        if not _coords_in_bounds(final, shape):
            return None
        occupied[final[:, 0], final[:, 1]] = True
        objects.append({"color": int(spec["color"]), "coords": final, "area": int(spec["area"])})
        traces.append(
            {
                "color": int(spec["color"]),
                "side": spec["side"],
                "area": int(spec["area"]),
                "ring_coords": ring,
                "final_coords": final,
                "yield_steps": yield_steps,
            }
        )
    return objects, traces


def _input_candidates(trace: dict, shape: tuple[int, int]) -> list[tuple[np.ndarray, str]]:
    ring = np.asarray(trace["ring_coords"], dtype=int)
    final = np.asarray(trace["final_coords"], dtype=int)
    side = trace["side"]
    area = int(trace["area"])
    toward_r, toward_c = _toward_marker_step(side)
    away_r, away_c = _away_step(side)
    candidates = []

    for dist in range(1, min(INPUT_CLOSER_LIMIT, area - 1) + 1):
        coords = _translate(ring, toward_r * dist, toward_c * dist)
        if _coords_in_bounds(coords, shape) and not np.array_equal(coords, final):
            candidates.append((coords, "closer"))
    for dist in range(1, INPUT_FARTHER_LIMIT + 1):
        coords = _translate(ring, away_r * dist, away_c * dist)
        if _coords_in_bounds(coords, shape) and not np.array_equal(coords, final):
            candidates.append((coords, "farther"))

    random.shuffle(candidates)
    return candidates


def _choose_input_positions(
    traces: list[dict],
    shape: tuple[int, int],
    marker: tuple[int, int],
) -> tuple[list[dict], set[str]] | None:
    candidates = {trace["color"]: _input_candidates(trace, shape) for trace in traces}
    if any(not opts for opts in candidates.values()):
        return None

    order = sorted(traces, key=lambda trace: len(candidates[trace["color"]]))
    assigned: dict[int, tuple[np.ndarray, str]] = {}
    occupied = np.zeros(shape, dtype=bool)
    occupied[marker] = True

    def rec(index: int, used_modes: set[str]) -> tuple[list[dict], set[str]] | None:
        if index == len(order):
            if {"closer", "farther"} <= used_modes:
                objects = []
                for trace in traces:
                    coords, _mode = assigned[trace["color"]]
                    objects.append({"color": int(trace["color"]), "coords": coords, "area": int(trace["area"])})
                return objects, used_modes
            return None

        trace = order[index]
        color = int(trace["color"])
        for coords, mode in candidates[color]:
            if np.any(occupied[coords[:, 0], coords[:, 1]]):
                continue
            occupied[coords[:, 0], coords[:, 1]] = True
            assigned[color] = (coords, mode)
            result = rec(index + 1, used_modes | {mode})
            if result is not None:
                return result
            occupied[coords[:, 0], coords[:, 1]] = False
            assigned.pop(color, None)
        return None

    return rec(0, set())


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    want_yield: bool,
) -> bool:
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    marker, objects = _extract_objects(inp)
    settled, traces = _settle_objects(objects, inp.shape, marker)
    if not np.array_equal(_render(inp.shape, marker, settled), expected):
        return False

    gaps = [int(trace["gap_before"]) for trace in traces]
    areas = [int(trace["area"]) for trace in traces]
    if not any(gap < area for gap, area in zip(gaps, areas)):
        return False
    if not any(gap > area for gap, area in zip(gaps, areas)):
        return False

    has_yield = any(int(trace["yield_steps"]) > 0 for trace in traces)
    if want_yield != has_yield:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs, want_yield = _choose_object_specs()
    chosen = _choose_shape_and_marker(specs)
    if chosen is None:
        return None
    shape, marker = chosen

    built = _build_output(specs, shape, marker)
    if built is None:
        return None
    output_objects, traces = built

    input_choice = _choose_input_positions(traces, shape, marker)
    if input_choice is None:
        return None
    input_objects, _modes = input_choice

    inp = _render(shape, marker, input_objects)
    expected = _render(shape, marker, output_objects)
    if _semantic_checks(inp, expected, want_yield):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for invisible docking rings")
