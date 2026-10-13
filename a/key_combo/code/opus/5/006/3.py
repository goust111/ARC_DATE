from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE = [BLUE, RED, GREEN, YELLOW]

ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_DIRS = ORTH_DIRS + DIAG_DIRS
DIR_SET = set(ALL_DIRS)

CORNER_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
LOCAL_TO_CORNER = {
    frozenset({(0, 0), (0, 1), (1, 0)}): "top_left",
    frozenset({(0, 0), (0, 1), (1, 1)}): "top_right",
    frozenset({(0, 0), (1, 0), (1, 1)}): "bottom_left",
    frozenset({(0, 1), (1, 0), (1, 1)}): "bottom_right",
}

MARKER_SHAPES = {
    "top_left": ((0, 0), (0, 1), (1, 0)),
    "top_right": ((0, 0), (0, 1), (1, 1)),
    "bottom_left": ((0, 0), (1, 0), (1, 1)),
    "bottom_right": ((0, 1), (1, 0), (1, 1)),
}

GEN_SIZES = [19, 21, 23]
GEN_RETRIES = 400
SCENARIO_WEIGHTS = {
    "adjacent_pair": 0.42,
    "opposite_pair": 0.28,
    "triple": 0.15,
    "quad": 0.15,
}


@dataclass(frozen=True)
class Beam:
    pos: tuple[int, int]
    direction: tuple[int, int]
    colors: tuple[int, ...]
    phase: int = 0


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _normalize_direction(dr: int, dc: int) -> tuple[int, int]:
    return (_sign(dr), _sign(dc))


def _is_opposite(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] == -b[0] and a[1] == -b[1]


def _is_orthogonal(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] * b[0] + a[1] * b[1] == 0


def _next_color(colors: tuple[int, ...], phase: int) -> tuple[int, int]:
    return int(colors[phase % len(colors)]), int(phase + 1)


def _ordered_union(a: tuple[int, ...], b: tuple[int, ...], beam_a: Beam, beam_b: Beam) -> tuple[int, ...]:
    if a == b:
        return a
    unique: list[int] = []
    for color in a:
        if color not in unique:
            unique.append(int(color))
    for color in b:
        if color not in unique:
            unique.append(int(color))
    if len(unique) == 1:
        return (unique[0],)
    if len(unique) > 2:
        raise AssertionError("beam color mixing exceeded the two-source puzzle rule")
    preds = sorted([(beam_a.pos, int(a[0])), (beam_b.pos, int(b[0]))])
    return tuple(color for _pos, color in preds)


def _draw_corner_l(out: np.ndarray, corner: str, color: int, arm: int) -> tuple[tuple[int, int], tuple[int, int]]:
    n = out.shape[0]
    if corner == "top_left":
        out[0, :arm] = color
        out[:arm, 0] = color
        return (0, arm - 1), (arm - 1, 0)
    if corner == "top_right":
        out[0, n - arm :] = color
        out[:arm, n - 1] = color
        return (0, n - arm), (arm - 1, n - 1)
    if corner == "bottom_left":
        out[n - 1, :arm] = color
        out[n - arm :, 0] = color
        return (n - 1, arm - 1), (n - arm, 0)
    out[n - 1, n - arm :] = color
    out[n - arm :, n - 1] = color
    return (n - 1, n - arm), (n - arm, n - 1)


def _launch_beams(corner: str, color: int, endpoints: tuple[tuple[int, int], tuple[int, int]]) -> list[Beam]:
    a, b = endpoints
    if corner == "top_left":
        return [Beam(a, (1, 0), (color,)), Beam(b, (0, 1), (color,))]
    if corner == "top_right":
        return [Beam(a, (1, 0), (color,)), Beam(b, (0, -1), (color,))]
    if corner == "bottom_left":
        return [Beam(a, (-1, 0), (color,)), Beam(b, (0, 1), (color,))]
    return [Beam(a, (-1, 0), (color,)), Beam(b, (0, -1), (color,))]


def _parse_markers(grid: np.ndarray) -> list[tuple[str, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert 2 <= len(comps) <= 4

    seen_corners: set[str] = set()
    markers: list[tuple[str, int]] = []
    for comp in comps:
        assert comp["area"] == 3
        r0, c0, r1, c1 = comp["bbox"]
        assert (r1 - r0, c1 - c0) == (1, 1)
        local = frozenset((r - r0, c - c0) for r, c in comp["coords"])
        assert local in LOCAL_TO_CORNER
        corner = LOCAL_TO_CORNER[local]
        assert corner not in seen_corners
        seen_corners.add(corner)
        markers.append((corner, int(comp["color"])))
    return sorted(markers, key=lambda item: CORNER_ORDER.index(item[0]))


def _simulate_rays(out: np.ndarray, beams: list[Beam], arm_mask: np.ndarray) -> np.ndarray:
    n = out.shape[0]
    active = list(beams)
    trail_mask = np.zeros_like(out, dtype=bool)
    grey_mask = np.zeros_like(out, dtype=bool)

    while active:
        arrivals: dict[tuple[int, int], list[Beam]] = {}
        next_active: list[Beam] = []

        for beam in active:
            nr = int(beam.pos[0] + beam.direction[0])
            nc = int(beam.pos[1] + beam.direction[1])
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if arm_mask[nr, nc] or grey_mask[nr, nc] or trail_mask[nr, nc]:
                continue
            arrivals.setdefault((nr, nc), []).append(beam)

        for cell, group in arrivals.items():
            if len(group) == 1:
                beam = group[0]
                color, phase = _next_color(beam.colors, beam.phase)
                out[cell] = color
                trail_mask[cell] = True
                next_active.append(Beam(cell, beam.direction, beam.colors, phase))
                continue

            dirs = [beam.direction for beam in group]
            if any(_is_opposite(dirs[i], dirs[j]) for i in range(len(dirs)) for j in range(i + 1, len(dirs))):
                out[cell] = GREY
                grey_mask[cell] = True
                continue

            if len(group) == 2 and _is_orthogonal(group[0].direction, group[1].direction):
                new_dir = _normalize_direction(
                    group[0].direction[0] + group[1].direction[0],
                    group[0].direction[1] + group[1].direction[1],
                )
                assert new_dir in DIR_SET
                colors = _ordered_union(group[0].colors, group[1].colors, group[0], group[1])
                next_active.append(Beam(cell, new_dir, colors, 0))
                continue

            continue

        active = next_active

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand interior 3-pixel L markers to outer corner Ls, then fire inward rays that collide and recolor.

    Input structure:
    - The grid is an odd square on a black background.
    - Each foreground object is a monochrome 3-pixel 4-connected L inside a 2x2 box.
    - There are 2 to 4 markers, their colors are distinct, and each orientation maps to a different outer corner.

    Rule:
    - Clear the canvas to black.
    - Map each small L orientation to the matching outer corner and draw there a same-color corner L with arm length (side-1)/2 along the border.
    - From each corner L's two non-corner arm endpoints, emit one-pixel rays toward the interior; rays advance synchronously.
    - A two-ray orthogonal collision leaves the collision cell empty and spawns one bisector ray; a two-ray opposite collision paints the meeting cell grey and stops there; if three or more rays meet and any opposite pair is present, that meeting cell is grey.
    - A bisector ray keeps the single source color if both inputs match, otherwise it alternates the two source colors pixel by pixel from its first painted cell onward.

    Key insight:
    - Marker position is irrelevant after detection: the tiny L's opening chooses an outer corner, and the expanded corner geometry determines both the ray launch points and all later collisions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    n, m = g.shape
    assert n == m and n % 2 == 1

    markers = _parse_markers(g)
    arm = n // 2
    out = np.zeros_like(g, dtype=int)
    arm_mask = np.zeros_like(g, dtype=bool)
    beams: list[Beam] = []

    for corner, color in markers:
        endpoints = _draw_corner_l(out, corner, color, arm)
        for endpoint in endpoints:
            arm_mask[endpoint] = True
        if corner == "top_left":
            arm_mask[0, :arm] = True
            arm_mask[:arm, 0] = True
        elif corner == "top_right":
            arm_mask[0, n - arm :] = True
            arm_mask[:arm, n - 1] = True
        elif corner == "bottom_left":
            arm_mask[n - 1, :arm] = True
            arm_mask[n - arm :, 0] = True
        else:
            arm_mask[n - 1, n - arm :] = True
            arm_mask[n - arm :, n - 1] = True
        beams.extend(_launch_beams(corner, color, endpoints))

    return _simulate_rays(out, beams, arm_mask)


def _scenario_corners() -> list[str]:
    scenarios = list(SCENARIO_WEIGHTS)
    weights = [SCENARIO_WEIGHTS[name] for name in scenarios]
    choice = random.choices(scenarios, weights=weights, k=1)[0]
    if choice == "adjacent_pair":
        return list(random.choice(
            [
                ("top_left", "top_right"),
                ("top_right", "bottom_right"),
                ("bottom_right", "bottom_left"),
                ("bottom_left", "top_left"),
            ]
        ))
    if choice == "opposite_pair":
        return list(random.choice([("top_left", "bottom_right"), ("top_right", "bottom_left")]))
    if choice == "triple":
        missing = random.choice(CORNER_ORDER)
        return [corner for corner in CORNER_ORDER if corner != missing]
    return list(CORNER_ORDER)


def _place_marker(grid: np.ndarray, color: int, corner: str, occupied: set[tuple[int, int]]) -> bool:
    n = grid.shape[0]
    rows = list(range(2, n - 3))
    cols = list(range(2, n - 3))
    random.shuffle(rows)
    random.shuffle(cols)

    offsets = MARKER_SHAPES[corner]
    halo = [(-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -1), (0, 2), (1, -1), (1, 2), (2, -1), (2, 0), (2, 1), (2, 2)]

    for r0 in rows:
        for c0 in cols:
            cells = {(r0 + dr, c0 + dc) for dr, dc in offsets}
            if any(cell in occupied for cell in cells):
                continue
            if any((r0 + dr, c0 + dc) in occupied for dr, dc in halo):
                continue
            if abs(r0 - n // 2) <= 1 and abs(c0 - n // 2) <= 1:
                continue
            for rr, cc in cells:
                grid[rr, cc] = color
                occupied.add((rr, cc))
            for dr, dc in halo:
                rr = r0 + dr
                cc = c0 + dc
                if 0 <= rr < n and 0 <= cc < n:
                    occupied.add((rr, cc))
            return True
    return False


def _inner_collision_origin(corner: str, n: int) -> tuple[int, int]:
    mid = n // 2
    if corner == "top_left":
        return (mid - 1, mid - 1)
    if corner == "top_right":
        return (mid - 1, mid + 1)
    if corner == "bottom_left":
        return (mid + 1, mid - 1)
    return (mid + 1, mid + 1)


def _validate_case(inp: np.ndarray, corners: list[str], colors_by_corner: dict[str, int]) -> bool:
    out = main(inp)
    n = inp.shape[0]
    center = (n // 2, n // 2)
    markers = dict(_parse_markers(inp))
    if markers != colors_by_corner:
        return False
    if len(corners) == 2:
        pair = set(corners)
        adjacent_specs = {
            frozenset({"top_left", "top_right"}): (1, 0),
            frozenset({"top_right", "bottom_right"}): (0, -1),
            frozenset({"bottom_left", "bottom_right"}): (-1, 0),
            frozenset({"top_left", "bottom_left"}): (0, 1),
        }
        pair_key = frozenset(pair)
        if pair_key in adjacent_specs:
            if int(out[center]) != BLACK:
                return False
            direction = adjacent_specs[pair_key]
            order = sorted(corners, key=lambda corner: _inner_collision_origin(corner, n))
            expected = tuple(colors_by_corner[corner] for corner in order)
            rr = center[0] + direction[0]
            cc = center[1] + direction[1]
            seen: list[int] = []
            while 0 <= rr < n and 0 <= cc < n and int(out[rr, cc]) != BLACK:
                seen.append(int(out[rr, cc]))
                rr += direction[0]
                cc += direction[1]
            if len(seen) < 2:
                return False
            for idx, color in enumerate(seen[: min(6, len(seen))]):
                if color != expected[idx % 2]:
                    return False
        else:
            if int(out[center]) != GREY:
                return False
    elif len(corners) >= 3 and int(out[center]) != GREY:
        return False
    if len(corners) >= 2 and set(corners) != {"top_left", "bottom_right"} and set(corners) != {"top_right", "bottom_left"}:
        if np.count_nonzero(out) <= sum(len(MARKER_SHAPES[c]) for c in corners):
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n = random.choice(GEN_SIZES)
    corners = _scenario_corners()
    colors = random.sample(PALETTE, len(corners))
    random.shuffle(colors)
    colors_by_corner = {corner: color for corner, color in zip(corners, colors)}

    inp = np.zeros((n, n), dtype=int)
    occupied: set[tuple[int, int]] = set()

    order = corners[:]
    random.shuffle(order)
    for corner in order:
        if not _place_marker(inp, colors_by_corner[corner], corner, occupied):
            return None

    if not _validate_case(inp, corners, colors_by_corner):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-collision ARC input")
