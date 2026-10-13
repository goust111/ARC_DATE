from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BACKGROUND = 0
IGNITION_COLOR = 2
SIDES = ("U", "D", "L", "R")
DIR_VECTORS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
BODY_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
BODY_DIMS = [(3, 3), (3, 5), (5, 3), (5, 5)]
GRID_BOUNDS = (20, 28)
ACTIVE_COUNT_BOUNDS = (2, 5)
SILENT_COUNT_BOUNDS = (1, 3)
GAP_BOUNDS = (2, 6)
MAX_ATTEMPTS = 400
FINAL_EYE_HIT_PROB = 0.45


@dataclass(frozen=True)
class Cannon:
    center_r: int
    center_c: int
    body_h: int
    body_w: int
    eye_side: str
    body_color: int
    eye_color: int

    @property
    def half_h(self) -> int:
        return self.body_h // 2

    @property
    def half_w(self) -> int:
        return self.body_w // 2

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (
            self.center_r - self.half_h,
            self.center_c - self.half_w,
            self.center_r + self.half_h,
            self.center_c + self.half_w,
        )

    @property
    def eye_pos(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        if self.eye_side == "U":
            return r0 - 1, self.center_c
        if self.eye_side == "D":
            return r1 + 1, self.center_c
        if self.eye_side == "L":
            return self.center_r, c0 - 1
        return self.center_r, c1 + 1

    def body_cells(self) -> list[tuple[int, int]]:
        r0, c0, r1, c1 = self.bbox
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def occupied_cells(self) -> list[tuple[int, int]]:
        return [*self.body_cells(), self.eye_pos]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(g.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [rc[0] for rc in coords]
            cols = [rc[1] for rc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_filled_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return comp["area"] == (r1 - r0 + 1) * (c1 - c0 + 1)


def _segment_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    if start is None or end is None:
        return []
    r, c = start
    er, ec = end
    dr = 0 if r == er else (1 if er > r else -1)
    dc = 0 if c == ec else (1 if ec > c else -1)
    cells = []
    while True:
        cells.append((r, c))
        if (r, c) == (er, ec):
            return cells
        r += dr
        c += dc


def _paint_segment(out: np.ndarray, start: tuple[int, int] | None, end: tuple[int, int] | None, color: int) -> None:
    for r, c in _segment_cells(start, end):
        out[r, c] = int(color)


def _detect_cannons(grid: np.ndarray) -> tuple[list[Cannon], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _monochromatic_components(g, bg)

    singles: dict[tuple[int, int], dict] = {}
    bodies: list[dict] = []
    for comp in comps:
        if comp["area"] == 1:
            singles[comp["coords"][0]] = comp
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if not _is_filled_rectangle(comp):
            continue
        if (r1 - r0) % 2 != 0 or (c1 - c0) % 2 != 0:
            continue
        bodies.append(comp)

    used_eye_cells = set()
    cannons = []
    for body in sorted(bodies, key=lambda comp: comp["bbox"]):
        r0, c0, r1, c1 = body["bbox"]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        candidates = []
        for side, pos in (
            ("U", (r0 - 1, center_c)),
            ("D", (r1 + 1, center_c)),
            ("L", (center_r, c0 - 1)),
            ("R", (center_r, c1 + 1)),
        ):
            comp = singles.get(pos)
            if comp is None or comp["color"] == body["color"]:
                continue
            candidates.append((side, pos, comp["color"]))
        assert len(candidates) == 1, "each cannon body must have exactly one midpoint eye"
        eye_side, eye_pos, eye_color = candidates[0]
        assert eye_pos not in used_eye_cells
        used_eye_cells.add(eye_pos)
        cannons.append(
            Cannon(
                center_r=center_r,
                center_c=center_c,
                body_h=r1 - r0 + 1,
                body_w=c1 - c0 + 1,
                eye_side=eye_side,
                body_color=int(body["color"]),
                eye_color=int(eye_color),
            )
        )

    ignition_cells = [coord for coord in singles if coord not in used_eye_cells]
    assert len(ignition_cells) == 1, "expected exactly one ignition marker"
    return cannons, ignition_cells[0]


def _build_owner_map(shape: tuple[int, int], cannons: list[Cannon]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for idx, cannon in enumerate(cannons):
        for r, c in cannon.occupied_cells():
            owner[r, c] = idx
    return owner


def _ignition_candidate(
    cannon: Cannon,
    ignition: tuple[int, int],
    occupied: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r, c = ignition
    r0, c0, r1, c1 = cannon.bbox
    if r0 <= r <= r1:
        if c < c0 and not occupied[r, c + 1 : c0].any():
            return (0, 1), (r, c0 - 1)
        if c > c1 and not occupied[r, c1 + 1 : c].any():
            return (0, -1), (r, c1 + 1)
    if c0 <= c <= c1:
        if r < r0 and not occupied[r + 1 : r0, c].any():
            return (1, 0), (r0 - 1, c)
        if r > r1 and not occupied[r1 + 1 : r, c].any():
            return (-1, 0), (r1 + 1, c)
    return None


def _trace_until_hit(
    shape: tuple[int, int],
    owner: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int | None, tuple[int, int] | None, tuple[int, int] | None]:
    r, c = start
    dr, dc = direction
    last_free = None
    while _in_bounds(shape, r, c):
        if owner[r, c] != -1:
            return int(owner[r, c]), last_free, (r, c)
        last_free = (r, c)
        r += dr
        c += dc
    return None, last_free, None


def _simulate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    cannons, ignition = _detect_cannons(g)
    owner = _build_owner_map(g.shape, cannons)
    occupied = owner != -1

    start_candidates = []
    for idx, cannon in enumerate(cannons):
        match = _ignition_candidate(cannon, ignition, occupied)
        if match is not None:
            start_candidates.append((idx, match[0], match[1]))
    assert len(start_candidates) == 1, "ignition must uniquely select one cannon"
    start_id, initial_dir, initial_end = start_candidates[0]

    out = g.copy()
    _paint_segment(out, ignition, initial_end, IGNITION_COLOR)

    fired_order = []
    events = []
    fired = set()
    current = start_id
    while current is not None and current not in fired:
        fired.add(current)
        fired_order.append(current)
        cannon = cannons[current]
        dr, dc = DIR_VECTORS[cannon.eye_side]
        eye_r, eye_c = cannon.eye_pos
        ray_start = (eye_r + dr, eye_c + dc)
        hit_id, ray_end, hit_cell = _trace_until_hit(g.shape, owner, ray_start, (dr, dc))
        _paint_segment(out, ray_start, ray_end, cannon.body_color)
        hit_kind = None
        if hit_id is not None:
            hit_kind = "eye" if hit_cell == cannons[hit_id].eye_pos else "body"
        events.append(
            {
                "cannon_id": current,
                "ray_start": ray_start,
                "ray_end": ray_end,
                "drawn_cells": _segment_cells(ray_start, ray_end),
                "hit_id": hit_id,
                "hit_cell": hit_cell,
                "hit_kind": hit_kind,
            }
        )
        current = hit_id

    return {
        "output": out,
        "cannons": cannons,
        "ignition": ignition,
        "owner": owner,
        "start_id": start_id,
        "start_candidates": start_candidates,
        "initial_dir": initial_dir,
        "initial_end": initial_end,
        "fired_order": fired_order,
        "events": events,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite exactly one cannon chain from the red marker, then draw only the rays caused by successive hits.

    Input structure:
    - A black background contains several cannons, each made of one solid monochrome rectangular body plus one adjacent off-color eye pixel centered just outside one body side.
    - The eye side defines that cannon's firing direction: up, down, left, or right.
    - There is exactly one isolated red ignition pixel aligned with exactly one cannon body along a clear row or column.

    Rule:
    - Draw a red straight segment from the ignition pixel toward its uniquely aligned cannon until the segment reaches the cell immediately before the body.
    - That struck cannon fires once: from the cell just beyond its eye, draw a body-colored ray in the eye direction until the cell immediately before the first cannon cell hit, or to the boundary if nothing is hit.
    - Any cannon hit by that ray fires in the same way, but each cannon may fire at most once; cannons never hit remain unchanged and silent.

    Key insight:
    - The picture is a causal chain, not a simultaneous broadcast: only the cannon reached from the red marker starts the process, and every later ray depends on the exact first-hit order of previous rays.
    """
    return _simulate(np.asarray(grid, dtype=int))["output"]


def _render_grid(shape: tuple[int, int], cannons: list[Cannon], ignition: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for cannon in cannons:
        r0, c0, r1, c1 = cannon.bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = cannon.body_color
        er, ec = cannon.eye_pos
        grid[er, ec] = cannon.eye_color
    grid[ignition] = IGNITION_COLOR
    return grid


def _buffer_cells(cannon: Cannon, shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells = set()
    for r, c in cannon.occupied_cells():
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(shape, rr, cc):
                    cells.add((rr, cc))
    return cells


def _fits_cannon(
    cannon: Cannon,
    shape: tuple[int, int],
    cannons: list[Cannon],
    ignition: tuple[int, int] | None,
) -> bool:
    occupied = set(cannon.occupied_cells())
    if any(not _in_bounds(shape, r, c) for r, c in occupied):
        return False
    if ignition is not None and ignition in _buffer_cells(cannon, shape):
        return False
    new_buffer = _buffer_cells(cannon, shape)
    for other in cannons:
        if occupied & set(other.occupied_cells()):
            return False
        if new_buffer & set(other.occupied_cells()):
            return False
        if _buffer_cells(other, shape) & occupied:
            return False
    return True


def _occupied_cells(cannons: list[Cannon], ignition: tuple[int, int] | None = None) -> set[tuple[int, int]]:
    cells = set()
    if ignition is not None:
        cells.add(ignition)
    for cannon in cannons:
        cells.update(cannon.occupied_cells())
    return cells


def _line_clear(
    start: tuple[int, int],
    end_exclusive: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    sr, sc = start
    er, ec = end_exclusive
    dr = 0 if sr == er else (1 if er > sr else -1)
    dc = 0 if sc == ec else (1 if ec > sc else -1)
    r, c = sr, sc
    while (r, c) != (er, ec):
        if (r, c) in occupied:
            return False
        r += dr
        c += dc
    return True


def _choose_eye_color(body_color: int) -> int:
    return random.choice([color for color in BODY_COLORS if color != body_color])


def _make_target_from_prev(
    prev: Cannon,
    body_color: int,
    hit_kind: str,
    eye_side: str,
    dims: tuple[int, int],
    gap: int,
) -> tuple[Cannon, tuple[int, int], tuple[int, int]]:
    body_h, body_w = dims
    half_h = body_h // 2
    half_w = body_w // 2
    pr, pc = prev.center_r, prev.center_c
    eye_r, eye_c = prev.eye_pos
    direction = prev.eye_side

    if direction == "R":
        hit_col = eye_c + 1 + gap
        center_r = pr
        center_c = hit_col + half_w if hit_kind == "body" else hit_col + half_w + 1
    elif direction == "L":
        hit_col = eye_c - 1 - gap
        center_r = pr
        center_c = hit_col - half_w if hit_kind == "body" else hit_col - half_w - 1
    elif direction == "D":
        hit_row = eye_r + 1 + gap
        center_c = pc
        center_r = hit_row + half_h if hit_kind == "body" else hit_row + half_h + 1
    else:
        hit_row = eye_r - 1 - gap
        center_c = pc
        center_r = hit_row - half_h if hit_kind == "body" else hit_row - half_h - 1

    cannon = Cannon(
        center_r=center_r,
        center_c=center_c,
        body_h=body_h,
        body_w=body_w,
        eye_side=eye_side,
        body_color=body_color,
        eye_color=_choose_eye_color(body_color),
    )

    dr, dc = DIR_VECTORS[direction]
    ray_start = (eye_r + dr, eye_c + dc)
    hit_cell = cannon.eye_pos if hit_kind == "eye" else {
        "R": (center_r, cannon.bbox[1]),
        "L": (center_r, cannon.bbox[3]),
        "D": (cannon.bbox[0], center_c),
        "U": (cannon.bbox[2], center_c),
    }[direction]
    return cannon, ray_start, hit_cell


def _place_random_cannon(
    shape: tuple[int, int],
    used_colors: set[int],
    existing: list[Cannon],
    ignition: tuple[int, int],
) -> Cannon | None:
    available = [color for color in BODY_COLORS if color not in used_colors]
    random.shuffle(available)
    for body_color in available:
        for _ in range(80):
            body_h, body_w = random.choice(BODY_DIMS)
            cannon = Cannon(
                center_r=random.randint(4 + body_h // 2, shape[0] - 5 - body_h // 2),
                center_c=random.randint(4 + body_w // 2, shape[1] - 5 - body_w // 2),
                body_h=body_h,
                body_w=body_w,
                eye_side=random.choice(SIDES),
                body_color=body_color,
                eye_color=_choose_eye_color(body_color),
            )
            if _fits_cannon(cannon, shape, existing, ignition):
                return cannon
    return None


def _place_shadow_silent(
    shape: tuple[int, int],
    used_colors: set[int],
    cannons: list[Cannon],
    prev: Cannon,
    blocker: Cannon,
    ignition: tuple[int, int],
) -> Cannon | None:
    available = [color for color in BODY_COLORS if color not in used_colors]
    random.shuffle(available)
    direction = prev.eye_side
    row = prev.center_r
    col = prev.center_c
    for body_color in available:
        for _ in range(60):
            body_h, body_w = random.choice(BODY_DIMS)
            half_h = body_h // 2
            half_w = body_w // 2
            eye_side = random.choice(SIDES)
            if direction == "R":
                furthest = max(c for r, c in blocker.occupied_cells() if r == row)
                center_r, center_c = row, furthest + random.randint(3, 6) + half_w
            elif direction == "L":
                furthest = min(c for r, c in blocker.occupied_cells() if r == row)
                center_r, center_c = row, furthest - random.randint(3, 6) - half_w
            elif direction == "D":
                furthest = max(r for r, c in blocker.occupied_cells() if c == col)
                center_r, center_c = furthest + random.randint(3, 6) + half_h, col
            else:
                furthest = min(r for r, c in blocker.occupied_cells() if c == col)
                center_r, center_c = furthest - random.randint(3, 6) - half_h, col
            cannon = Cannon(center_r, center_c, body_h, body_w, eye_side, body_color, _choose_eye_color(body_color))
            if _fits_cannon(cannon, shape, cannons, ignition):
                return cannon
    return None


def _validate_generated_input(grid: np.ndarray, expect_final_eye_hit: bool) -> None:
    details = _simulate(grid)
    out = details["output"]
    cannons = details["cannons"]
    fired = details["fired_order"]
    events = details["events"]
    assert len(cannons) >= 3
    assert len(fired) >= 2
    assert len(details["start_candidates"]) == 1
    assert details["initial_end"] != details["ignition"]
    assert any(event["drawn_cells"] for event in events), "at least one cannon ray must visibly draw"

    fired_set = set(fired)
    silent_ids = [idx for idx in range(len(cannons)) if idx not in fired_set]
    assert silent_ids, "need at least one silent cannon"

    if expect_final_eye_hit:
        assert len(fired) >= 2
        assert events[-2]["hit_id"] == fired[-1]
        assert events[-2]["hit_kind"] == "eye"
        assert events[-1]["hit_id"] in fired[:-1]
    else:
        assert events[-1]["hit_id"] is None
        assert events[-1]["drawn_cells"]

    useful_silent = False
    owner = details["owner"]
    for sid in silent_ids:
        cannon = cannons[sid]
        dr, dc = DIR_VECTORS[cannon.eye_side]
        start = (cannon.eye_pos[0] + dr, cannon.eye_pos[1] + dc)
        _hit_id, end, _hit_cell = _trace_until_hit(grid.shape, owner, start, (dr, dc))
        cells = _segment_cells(start, end)
        if cells and any(out[r, c] != cannon.body_color for r, c in cells):
            useful_silent = True
            break
    assert useful_silent, "at least one silent cannon must contradict the 'all cannons fire' shortcut"


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_BOUNDS)
    width = random.randint(*GRID_BOUNDS)
    shape = (height, width)

    active_count = random.randint(*ACTIVE_COUNT_BOUNDS)
    silent_cap = min(SILENT_COUNT_BOUNDS[1], len(BODY_COLORS) - active_count)
    silent_count = random.randint(SILENT_COUNT_BOUNDS[0], silent_cap)
    final_eye_hit = random.random() < FINAL_EYE_HIT_PROB

    used_colors = set()
    active: list[Cannon] = []
    transitions: list[str] = []

    first_color = random.choice(BODY_COLORS)
    used_colors.add(first_color)
    for _ in range(120):
        body_h, body_w = random.choice(BODY_DIMS)
        first = Cannon(
            center_r=random.randint(5 + body_h // 2, height - 6 - body_h // 2),
            center_c=random.randint(5 + body_w // 2, width - 6 - body_w // 2),
            body_h=body_h,
            body_w=body_w,
            eye_side=random.choice(SIDES),
            body_color=first_color,
            eye_color=_choose_eye_color(first_color),
        )
        ignition_side_options = [side for side in SIDES if side != first.eye_side]
        ignition_side = random.choice(ignition_side_options)
        gap = random.randint(*GAP_BOUNDS)
        r0, c0, r1, c1 = first.bbox
        if ignition_side == "U":
            ignition = (r0 - 1 - gap, first.center_c)
        elif ignition_side == "D":
            ignition = (r1 + 1 + gap, first.center_c)
        elif ignition_side == "L":
            ignition = (first.center_r, c0 - 1 - gap)
        else:
            ignition = (first.center_r, c1 + 1 + gap)
        if not _in_bounds(shape, *ignition):
            continue
        if _fits_cannon(first, shape, [], ignition):
            active.append(first)
            break
    if not active:
        return None

    for idx in range(1, active_count):
        hit_kind = "eye" if final_eye_hit and idx == active_count - 1 else "body"
        prev = active[-1]
        incoming_side = OPPOSITE[prev.eye_side]
        body_color_choices = [color for color in BODY_COLORS if color not in used_colors]
        random.shuffle(body_color_choices)
        placed = None
        for body_color in body_color_choices:
            for _ in range(120):
                dims = random.choice(BODY_DIMS)
                eye_side = incoming_side if hit_kind == "eye" else random.choice([side for side in SIDES if side != incoming_side])
                gap = random.randint(*GAP_BOUNDS)
                cannon, ray_start, hit_cell = _make_target_from_prev(prev, body_color, hit_kind, eye_side, dims, gap)
                if not _fits_cannon(cannon, shape, active, ignition):
                    continue
                occupied = _occupied_cells(active, ignition)
                if not _line_clear(ray_start, hit_cell, occupied):
                    continue
                placed = cannon
                transitions.append(hit_kind)
                break
            if placed is not None:
                used_colors.add(body_color)
                active.append(placed)
                break
        if placed is None:
            return None

    cannons = list(active)

    if silent_count > 0:
        body_hit_edges = [i for i, kind in enumerate(transitions) if kind == "body"]
        if body_hit_edges:
            edge_idx = random.choice(body_hit_edges)
            shadow = _place_shadow_silent(shape, used_colors, cannons, active[edge_idx], active[edge_idx + 1], ignition)
            if shadow is not None:
                cannons.append(shadow)
                used_colors.add(shadow.body_color)

    while len(cannons) < active_count + silent_count:
        cannon = _place_random_cannon(shape, used_colors, cannons, ignition)
        if cannon is None:
            return None
        cannons.append(cannon)
        used_colors.add(cannon.body_color)

    grid = _render_grid(shape, cannons, ignition)
    try:
        details = _simulate(grid)
    except AssertionError:
        return None

    fired = details["fired_order"]
    fired_colors = [details["cannons"][idx].body_color for idx in fired]
    active_colors = [cannon.body_color for cannon in active]
    if fired_colors != active_colors:
        return None
    if final_eye_hit:
        if len(fired) < 2 or details["events"][-2]["hit_kind"] != "eye":
            return None
    else:
        if details["events"][-1]["hit_id"] is not None:
            return None

    try:
        _validate_generated_input(grid, final_eye_hit)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
