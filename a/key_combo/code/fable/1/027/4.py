from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _border_majority(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = list(map(int, grid[0, :]))
    border.extend(map(int, grid[h - 1, :]))
    border.extend(map(int, grid[1 : h - 1, 0]))
    border.extend(map(int, grid[1 : h - 1, w - 1]))
    return int(Counter(border).most_common(1)[0][0])


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _find_wall_color(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        if any(len(comp) > 1 for comp in _components_of_mask(grid == color)):
            candidates.append(color)
    assert len(candidates) == 1, "expected exactly one non-background wall color"
    return int(candidates[0])


def _extract_chambers(grid: np.ndarray, wall: int, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    chambers: list[dict[str, object]] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == wall:
                continue

            coords: list[tuple[int, int]] = []
            markers: list[tuple[int, int, int]] = []
            touches_border = False
            queue = deque([(sr, sc)])
            seen[sr, sc] = True

            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                touches_border |= (r == 0 or c == 0 or r == h - 1 or c == w - 1)
                color = int(grid[r, c])
                if color != bg:
                    markers.append((r, c, color))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != wall:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            if touches_border:
                assert not markers, "exterior region should not contain markers"
                continue

            assert len(markers) == 1, "each enclosed chamber must have exactly one marker"
            coords.sort()
            marker_r, marker_c, marker_color = markers[0]
            chambers.append(
                {
                    "coords": tuple(coords),
                    "coord_set": frozenset(coords),
                    "marker": (int(marker_r), int(marker_c)),
                    "marker_color": int(marker_color),
                }
            )

    assert chambers, "expected at least one enclosed chamber"
    return chambers


def _rot180_coords(coords: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    return frozenset((h - 1 - r, w - 1 - c) for r, c in coords)


def _pair_chambers(chambers: list[dict[str, object]], shape: tuple[int, int]) -> list[int]:
    by_coords = {chamber["coord_set"]: idx for idx, chamber in enumerate(chambers)}
    partners: list[int] = []
    for chamber in chambers:
        rotated = _rot180_coords(chamber["coords"], shape)
        assert rotated in by_coords, "every chamber must have a 180-degree partner"
        partners.append(int(by_coords[rotated]))
    for idx, partner in enumerate(partners):
        assert partners[partner] == idx, "partnership must be symmetric"
    return partners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every enclosed chamber with the marker color from its 180-degree rotational partner.

    Input structure:
    - One repeated wall color partitions a common background into several enclosed chambers.
    - The full wall layout is invariant under a 180-degree rotation about the grid center.
    - Each enclosed chamber contains exactly one single-pixel colored marker; the exterior region has none.

    Rule:
    - Infer the wall color and extract every non-wall connected component that does not touch the border.
    - Pair chambers by rotating their full cell sets by 180 degrees; a central chamber may pair with itself.
    - Flood-fill each chamber, including its marker cell, with the marker color found in its partner chamber.

    Key insight:
    - The marker color is transferred across the wall layout's rotational symmetry, so the decisive object is the
      chamber shape under 180-degree rotation, not the marker's own position inside that chamber.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _border_majority(g)
    wall = _find_wall_color(g, bg)
    chambers = _extract_chambers(g, wall, bg)
    partners = _pair_chambers(chambers, g.shape)

    out = g.copy()
    for idx, chamber in enumerate(chambers):
        fill_color = int(chambers[partners[idx]]["marker_color"])
        for r, c in chamber["coords"]:
            out[r, c] = fill_color
    return out


def _reserve_box(room: dict[str, int]) -> tuple[int, int, int, int]:
    return room["r0"] - 2, room["r0"] + room["h"] + 2, room["c0"] - 2, room["c0"] + room["w"] + 2


def _room_coord_set(room: dict[str, int]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (r, c)
        for r in range(room["r0"], room["r0"] + room["h"])
        for c in range(room["c0"], room["c0"] + room["w"])
    )


def _rooms_fit(reserved: np.ndarray, rooms: list[dict[str, int]]) -> bool:
    temp = np.zeros_like(reserved, dtype=bool)
    for room in rooms:
        rs, re, cs, ce = _reserve_box(room)
        if temp[rs:re, cs:ce].any():
            return False
        temp[rs:re, cs:ce] = True
    return not np.any(reserved & temp)


def _reserve_rooms(reserved: np.ndarray, rooms: list[dict[str, int]]) -> None:
    for room in rooms:
        rs, re, cs, ce = _reserve_box(room)
        reserved[rs:re, cs:ce] = True


def _draw_room(grid: np.ndarray, room: dict[str, int], wall: int) -> None:
    r0, c0, h, w = room["r0"], room["c0"], room["h"], room["w"]
    grid[r0 - 1, c0 - 1 : c0 + w + 1] = wall
    grid[r0 + h, c0 - 1 : c0 + w + 1] = wall
    grid[r0 - 1 : r0 + h + 1, c0 - 1] = wall
    grid[r0 - 1 : r0 + h + 1, c0 + w] = wall


def _pick_marker_cell(room: dict[str, int]) -> tuple[int, int]:
    cells = sorted(_room_coord_set(room))
    center = ((room["r0"] + room["r0"] + room["h"] - 1) // 2, (room["c0"] + room["c0"] + room["w"] - 1) // 2)
    off_center = [cell for cell in cells if cell != center]
    if off_center and random.random() < 0.75:
        return random.choice(off_center)
    return random.choice(cells)


def _pair_candidates(h: int, w: int, room_h: int, room_w: int, reserved: np.ndarray) -> list[tuple[dict[str, int], dict[str, int]]]:
    candidates = []
    for r0 in range(2, h - room_h - 1):
        for c0 in range(2, w - room_w - 1):
            partner_r0 = h - r0 - room_h
            partner_c0 = w - c0 - room_w
            if (r0, c0) >= (partner_r0, partner_c0):
                continue
            room_a = {"r0": r0, "c0": c0, "h": room_h, "w": room_w}
            room_b = {"r0": partner_r0, "c0": partner_c0, "h": room_h, "w": room_w}
            if _rooms_fit(reserved, [room_a, room_b]):
                candidates.append((room_a, room_b))
    return candidates


def _center_candidates(h: int, w: int, reserved: np.ndarray) -> list[dict[str, int]]:
    candidates = []
    max_h = min(7, h - 4)
    max_w = min(7, w - 4)
    for room_h in range(3, max_h + 1, 2):
        for room_w in range(3, max_w + 1, 2):
            room = {
                "r0": h // 2 - room_h // 2,
                "c0": w // 2 - room_w // 2,
                "h": room_h,
                "w": room_w,
            }
            if _rooms_fit(reserved, [room]):
                candidates.append(room)
    return candidates


def _expected_output(inp: np.ndarray, rooms: list[dict[str, int]]) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        partner = rooms[room["partner"]]
        out[room["r0"] : room["r0"] + room["h"], room["c0"] : room["c0"] + room["w"]] = int(partner["marker_color"])
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray, rooms: list[dict[str, int]], bg: int, wall: int) -> bool:
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    parsed = _extract_chambers(inp, wall, bg)
    planned_sets = {_room_coord_set(room) for room in rooms}
    parsed_sets = {chamber["coord_set"] for chamber in parsed}
    if planned_sets != parsed_sets:
        return False

    h, w = inp.shape
    for room in rooms:
        coords = _room_coord_set(room)
        marker_cells = [(r, c) for r, c in coords if int(inp[r, c]) not in (bg, wall)]
        if len(marker_cells) != 1:
            return False
        if int(inp[marker_cells[0]]) != int(room["marker_color"]):
            return False
        partner_coords = _room_coord_set(rooms[room["partner"]])
        rotated = frozenset((h - 1 - r, w - 1 - c) for r, c in coords)
        if rotated != partner_coords:
            return False
        room_view = expected[room["r0"] : room["r0"] + room["h"], room["c0"] : room["c0"] + room["w"]]
        if not np.all(room_view == int(rooms[room["partner"]]["marker_color"])):
            return False

    if not np.array_equal(inp[inp == wall], expected[inp == wall]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    has_center = random.random() < 0.45
    n_pairs = random.randint(2, 3)

    base = 13 + 3 * n_pairs + (4 if has_center else 0)
    odd_sizes = [size for size in range(max(15, base), 30) if size % 2 == 1]
    if not odd_sizes:
        return None
    h = random.choice(odd_sizes)
    w = random.choice(odd_sizes)

    bg, wall = random.sample(COLORS, 2)
    reserved = np.zeros((h, w), dtype=bool)
    rooms: list[dict[str, int]] = []

    if has_center:
        candidates = _center_candidates(h, w, reserved)
        if not candidates:
            return None
        center_room = random.choice(candidates)
        center_room["partner"] = 0
        rooms.append(center_room)
        _reserve_rooms(reserved, [center_room])

    for _ in range(n_pairs):
        placed = False
        for _ in range(40):
            room_h = random.randint(2, min(6, h - 4))
            room_w = random.randint(2, min(6, w - 4))
            candidates = _pair_candidates(h, w, room_h, room_w, reserved)
            if not candidates:
                continue
            room_a, room_b = random.choice(candidates)
            ia = len(rooms)
            ib = ia + 1
            room_a["partner"] = ib
            room_b["partner"] = ia
            rooms.extend([room_a, room_b])
            _reserve_rooms(reserved, [room_a, room_b])
            placed = True
            break
        if not placed:
            return None

    if has_center:
        rooms[0]["partner"] = 0

    marker_colors = random.sample([color for color in COLORS if color not in {bg, wall}], len(rooms))
    inp = np.full((h, w), bg, dtype=int)

    for room, marker_color in zip(rooms, marker_colors):
        _draw_room(inp, room, wall)
        marker_r, marker_c = _pick_marker_cell(room)
        inp[marker_r, marker_c] = marker_color
        room["marker_color"] = int(marker_color)
        room["marker"] = (int(marker_r), int(marker_c))

    expected = _expected_output(inp, rooms)
    if not _validate_instance(inp, expected, rooms, bg, wall):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-chamber recoloring input")
