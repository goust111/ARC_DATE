from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
RECT_IN = 1
DIR_TRAIL = 2
RECT_OUT = 3
CROSS = 4
MARK = 9

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
SIDE_TO_TRAIL = {
    "top": "up",
    "bottom": "down",
    "left": "left",
    "right": "right",
}
CORNER_SLOTS = {
    "tl": (("top", "left"), ("left", "top")),
    "tr": (("top", "right"), ("right", "top")),
    "bl": (("bottom", "left"), ("left", "bottom")),
    "br": (("bottom", "right"), ("right", "bottom")),
}


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _line_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    dr = (r1 > r0) - (r1 < r0)
    dc = (c1 > c0) - (c1 < c0)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    assert dr == 0 or dc == 0
    return [(r0 + dr * i, c0 + dc * i) for i in range(steps + 1)]


def _iter_side_contacts(rect: BBox) -> dict[tuple[str, str], Coord]:
    r0, c0, r1, c1 = rect
    return {
        ("top", "left"): (r0, c0),
        ("top", "right"): (r0, c1),
        ("bottom", "left"): (r1, c0),
        ("bottom", "right"): (r1, c1),
        ("left", "top"): (r0, c0),
        ("left", "bottom"): (r1, c0),
        ("right", "top"): (r0, c1),
        ("right", "bottom"): (r1, c1),
    }


def _marker_from_spec(rect: BBox, spec: dict) -> dict:
    r0, c0, r1, c1 = rect
    side = spec["side"]
    align = spec["align"]
    distance = int(spec["distance"])
    trail_len = int(spec["trail_len"])
    away_dir = SIDE_TO_TRAIL[side]
    slide_dir = OPPOSITE[away_dir]
    if side == "top":
        contact = (r0, c0 if align == "left" else c1 if align == "right" else int(align))
    elif side == "bottom":
        contact = (r1, c0 if align == "left" else c1 if align == "right" else int(align))
    elif side == "left":
        contact = (r0 if align == "top" else r1 if align == "bottom" else int(align), c0)
    else:
        contact = (r0 if align == "top" else r1 if align == "bottom" else int(align), c1)

    dr, dc = DIRS[slide_dir]
    start = (contact[0] - dr * distance, contact[1] - dc * distance)
    adr, adc = DIRS[away_dir]
    trail = [(start[0] + adr * step, start[1] + adc * step) for step in range(1, trail_len + 1)]
    final = (contact[0] - dr, contact[1] - dc)
    return {
        "side": side,
        "align": align,
        "distance": distance,
        "trail_len": trail_len,
        "away_dir": away_dir,
        "slide_dir": slide_dir,
        "contact": contact,
        "start": start,
        "final_outside": final,
        "trail": trail,
        "path_to_contact": _line_cells(start, contact),
    }


def _parse_scene(grid: np.ndarray) -> tuple[BBox, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rect_box = _bbox(g == RECT_IN)
    assert rect_box is not None
    r0, c0, r1, c1 = rect_box
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == RECT_IN), "blue object must be a solid rectangle"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARK)]
    assert 3 <= len(markers) <= 5
    used_red: set[Coord] = set()
    parsed: list[dict] = []

    for mr, mc in sorted(markers):
        dirs = []
        for name, (dr, dc) in DIRS.items():
            nr, nc = mr + dr, mc + dc
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == DIR_TRAIL:
                dirs.append(name)
        assert len(dirs) == 1, "each marker must have exactly one adjacent red direction trail"
        away_dir = dirs[0]
        dr, dc = DIRS[away_dir]
        trail: list[Coord] = []
        rr, cc = mr + dr, mc + dc
        while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] == DIR_TRAIL:
            trail.append((rr, cc))
            rr += dr
            cc += dc
        assert 2 <= len(trail) <= 3, "direction trail must have length 2 or 3"
        for tr, tc in trail:
            used_red.add((tr, tc))
            side_red = 0
            for pdr, pdc in DIRS.values():
                pr, pc = tr + pdr, tc + pdc
                if 0 <= pr < g.shape[0] and 0 <= pc < g.shape[1] and g[pr, pc] == DIR_TRAIL:
                    side_red += 1
            assert side_red <= 2, "red trail must be straight"

        slide_dir = OPPOSITE[away_dir]
        if slide_dir == "down":
            assert mr < r0 and c0 <= mc <= c1
            distance = r0 - mr
            contact = (r0, mc)
            side = "top"
            align = "left" if mc == c0 else "right" if mc == c1 else mc
        elif slide_dir == "up":
            assert mr > r1 and c0 <= mc <= c1
            distance = mr - r1
            contact = (r1, mc)
            side = "bottom"
            align = "left" if mc == c0 else "right" if mc == c1 else mc
        elif slide_dir == "right":
            assert mc < c0 and r0 <= mr <= r1
            distance = c0 - mc
            contact = (mr, c0)
            side = "left"
            align = "top" if mr == r0 else "bottom" if mr == r1 else mr
        else:
            assert mc > c1 and r0 <= mr <= r1
            distance = mc - c1
            contact = (mr, c1)
            side = "right"
            align = "top" if mr == r0 else "bottom" if mr == r1 else mr

        parsed.append(
            {
                "start": (mr, mc),
                "away_dir": away_dir,
                "slide_dir": slide_dir,
                "trail": trail,
                "distance": int(distance),
                "contact": contact,
                "side": side,
                "align": align,
            }
        )

    red_pixels = {tuple(map(int, rc)) for rc in np.argwhere(g == DIR_TRAIL)}
    assert red_pixels == used_red, "all red pixels must belong to marker trails"

    distances = [item["distance"] for item in parsed]
    assert len(set(distances)) == len(distances), "contact distances must be unique"
    for item in parsed:
        item["path_to_contact"] = _line_cells(item["start"], item["contact"])
        item["trail_len"] = len(item["trail"])
    return rect_box, parsed


def _render_output(shape: tuple[int, int], rect: BBox, markers: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = RECT_OUT

    min_distance = min(item["distance"] for item in markers)
    assert sum(item["distance"] == min_distance for item in markers) == 1

    cell_to_distances: dict[Coord, list[int]] = defaultdict(list)
    for item in markers:
        for cell in item["path_to_contact"]:
            cell_to_distances[cell].append(int(item["distance"]))

    for cell, dists in cell_to_distances.items():
        r, c = cell
        if len(dists) == 1:
            out[r, c] = MARK
            continue
        assert len(dists) == len(set(dists)), "crossing priority must be unique"
        out[r, c] = CROSS
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each marker's red incoming trail, slide it toward the rectangle, and redraw the slide traces.

    Input structure:
    - Black background with one solid blue rectangle.
    - Three to five maroon single-pixel markers lie outside the rectangle, each sharing a row or a column with it.
    - Each marker has a straight 2-3 pixel red trail extending away from the rectangle, encoding the marker's incoming direction.

    Rule:
    - For each marker, infer the slide direction as the opposite of its red trail direction, then measure the marker's distance to the aligned rectangle edge.
    - The unique nearest marker is the entrant; it reaches the contacted boundary pixel, while every other marker stops one cell before the wall.
    - Recolor the blue rectangle green, erase all original red trails, and paint each marker's slide trace in maroon from its start toward the contacted edge, excluding any rectangle interior.
    - If two traces meet at one pixel, the farther marker has priority there; because all traces use the same color, mark that shared pixel yellow.

    Key insight:
    - The red input trail is only a direction code: reversing it reveals how the marker slides, and the corner contacts where multiple slides meet expose the distance-priority rule.
    """
    rect, markers = _parse_scene(np.asarray(grid, dtype=int))
    return _render_output(grid.shape, rect, markers)


def _build_input(shape: tuple[int, int], rect: BBox, marker_specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = RECT_IN
    for spec in marker_specs:
        marker = _marker_from_spec(rect, spec)
        sr, sc = marker["start"]
        grid[sr, sc] = MARK
        for rr, cc in marker["trail"]:
            grid[rr, cc] = DIR_TRAIL
    return grid


def _semantic_validate(shape: tuple[int, int], rect: BBox, specs: list[dict], require_cross: bool) -> bool:
    markers = [_marker_from_spec(rect, spec) for spec in specs]
    H, W = shape
    seen_input: dict[Coord, str] = {}
    own_red_neighbors: dict[Coord, set[Coord]] = defaultdict(set)
    for marker in markers:
        cells = [marker["start"], *marker["trail"]]
        for rr, cc in cells:
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            if (rr, cc) in seen_input:
                return False
            seen_input[(rr, cc)] = "input"
        for rr, cc in marker["trail"]:
            if rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]:
                return False
        full = [marker["start"], *marker["trail"]]
        for a, b in zip(full, full[1:]):
            own_red_neighbors[a].add(b)
            own_red_neighbors[b].add(a)

    occupied_markers = {marker["start"] for marker in markers}
    occupied_red = {cell for marker in markers for cell in marker["trail"]}
    for rr, cc in occupied_red | occupied_markers:
        for dr, dc in DIRS.values():
            nr, nc = rr + dr, cc + dc
            neighbor = (nr, nc)
            if neighbor not in occupied_red and neighbor not in occupied_markers:
                continue
            if (rr, cc) in occupied_markers:
                if neighbor not in own_red_neighbors[(rr, cc)]:
                    return False
            else:
                allowed = own_red_neighbors[(rr, cc)]
                if neighbor == (rr, cc):
                    continue
                if neighbor not in allowed:
                    return False

    distances = [marker["distance"] for marker in markers]
    if len(set(distances)) != len(distances):
        return False

    path_sets = [set(marker["path_to_contact"]) for marker in markers]
    overlaps: list[set[Coord]] = []
    all_cells: dict[Coord, int] = defaultdict(int)
    for cells in path_sets:
        for cell in cells:
            all_cells[cell] += 1
    if any(v > 2 for v in all_cells.values()):
        return False
    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            inter = path_sets[i] & path_sets[j]
            if inter:
                overlaps.append(inter)
                if len(inter) != 1:
                    return False
                cell = next(iter(inter))
                if cell not in {
                    (rect[0], rect[1]),
                    (rect[0], rect[3]),
                    (rect[2], rect[1]),
                    (rect[2], rect[3]),
                }:
                    return False
    if require_cross and not overlaps:
        return False
    if not require_cross and len(overlaps) > 2:
        return False

    grid = _build_input(shape, rect, specs)
    parsed_rect, parsed_markers = _parse_scene(grid)
    if parsed_rect != rect:
        return False
    parsed_summary = sorted(
        (item["start"], item["side"], item["align"], item["distance"], item["away_dir"], item["trail_len"])
        for item in parsed_markers
    )
    spec_summary = sorted(
        (
            _marker_from_spec(rect, spec)["start"],
            spec["side"],
            spec["align"],
            int(spec["distance"]),
            SIDE_TO_TRAIL[spec["side"]],
            int(spec["trail_len"]),
        )
        for spec in specs
    )
    if parsed_summary != spec_summary:
        return False

    expected = _render_output(shape, rect, [_marker_from_spec(rect, spec) for spec in specs])
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _random_rect(shape: tuple[int, int]) -> BBox:
    H, W = shape
    rh = random.randint(4, 5)
    rw = random.randint(4, 6)
    r0 = random.randint(4, H - rh - 5)
    c0 = random.randint(4, W - rw - 5)
    return r0, c0, r0 + rh - 1, c0 + rw - 1


def _available_distances(shape: tuple[int, int], rect: BBox, side: str, trail_cap: int = 3) -> list[int]:
    H, W = shape
    r0, c0, r1, c1 = rect
    if side == "top":
        limit = r0
    elif side == "bottom":
        limit = H - 1 - r1
    elif side == "left":
        limit = c0
    else:
        limit = W - 1 - c1
    return [d for d in range(2, limit - trail_cap + 1)]


def _free_slots(rect: BBox) -> list[tuple[str, str | int]]:
    r0, c0, r1, c1 = rect
    slots: list[tuple[str, str | int]] = []
    for c in range(c0, c1 + 1):
        slots.append(("top", "left" if c == c0 else "right" if c == c1 else c))
        slots.append(("bottom", "left" if c == c0 else "right" if c == c1 else c))
    for r in range(r0, r1 + 1):
        slots.append(("left", "top" if r == r0 else "bottom" if r == r1 else r))
        slots.append(("right", "top" if r == r0 else "bottom" if r == r1 else r))
    return slots


def _pick_specs(shape: tuple[int, int], rect: BBox) -> tuple[list[dict], bool] | None:
    target = random.randint(3, 5)
    require_cross = random.random() < 0.7

    corner_contacts = _iter_side_contacts(rect)
    chosen_slots: list[tuple[str, str | int]] = []
    if require_cross:
        corner = random.choice(list(CORNER_SLOTS))
        pair = list(CORNER_SLOTS[corner])
        chosen_slots.extend(pair)
        if target >= 4 and random.random() < 0.35:
            other_choices = [name for name in CORNER_SLOTS if name != corner]
            other = random.choice(other_choices)
            chosen_slots.extend(list(CORNER_SLOTS[other]))

    all_slots = _free_slots(rect)
    random.shuffle(all_slots)
    for slot in all_slots:
        if slot in chosen_slots:
            continue
        side, align = slot
        if require_cross and (side, align) in corner_contacts:
            pass
        chosen_slots.append(slot)
        if len(chosen_slots) >= target:
            break
    chosen_slots = chosen_slots[:target]
    if len(chosen_slots) < 3:
        return None

    side_limits = {side: _available_distances(shape, rect, side) for side, _ in chosen_slots}
    all_distances = sorted(set(d for vals in side_limits.values() for d in vals))
    if len(all_distances) < len(chosen_slots):
        return None
    random.shuffle(all_distances)

    specs: list[dict] = []
    used_distances: set[int] = set()
    for side, align in chosen_slots:
        choices = [d for d in side_limits[side] if d not in used_distances]
        if not choices:
            return None
        distance = random.choice(choices)
        used_distances.add(distance)
        specs.append(
            {
                "side": side,
                "align": align,
                "distance": distance,
                "trail_len": random.randint(2, 3),
            }
        )
    return specs, require_cross


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(16, 22), random.randint(16, 22))
    rect = _random_rect(shape)
    picked = _pick_specs(shape, rect)
    if picked is None:
        return None
    specs, require_cross = picked
    if not _semantic_validate(shape, rect, specs, require_cross):
        return None
    return _build_input(shape, rect, specs)


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
