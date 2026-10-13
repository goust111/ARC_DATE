from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_ALL_COLORS = tuple(range(10))
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RING = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_GEN_RETRIES = 400
_LAYOUT_ROWS = 2
_LAYOUT_COLS = 3
_CELL_GAP = 1
_GRID_BORDER = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_multicolor_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _orientation_from_bbox(bbox: BBox) -> str:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h != w, "rooms must be non-square"
    return "H" if w > h else "V"


def _rotate_cw(offsets: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return sorted((dc, -dr, color) for dr, dc, color in offsets)


def _canonical_offsets(offsets: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return sorted((int(dr), int(dc), int(color)) for dr, dc, color in offsets)


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)

    rooms = []
    for comp in _find_multicolor_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub != bg), "each room component must fill its rectangle"
        assert int(comp["area"]) == int(sub.size), "room bbox must match the full room"

        room = {
            "bbox": comp["bbox"],
            "subgrid": sub.copy(),
            "orientation": _orientation_from_bbox(comp["bbox"]),
        }
        rooms.append(room)

    assert len(rooms) >= 2, "expected several rooms"

    all_colors = sorted(set(map(int, g.ravel())) - {bg})
    marker_candidates = []
    for color in all_colors:
        counts = [int(np.count_nonzero(room["subgrid"] == color)) for room in rooms]
        if all(count == 1 for count in counts):
            marker_candidates.append(int(color))
    assert len(marker_candidates) == 1, "expected a unique marker color"
    marker_color = marker_candidates[0]

    exemplar_candidates = []
    for room in rooms:
        sub = room["subgrid"]
        marker_positions = np.argwhere(sub == marker_color)
        assert marker_positions.shape == (1, 2), "every room must contain exactly one marker"
        mr_local, mc_local = map(int, marker_positions[0])
        r0, c0, _r1, _c1 = room["bbox"]
        room["marker"] = (r0 + mr_local, c0 + mc_local)

        non_marker = sub[sub != marker_color]
        room_color = int(Counter(map(int, non_marker.ravel())).most_common(1)[0][0])
        room["room_color"] = room_color

        decor_mask = (sub != marker_color) & (sub != room_color)
        decor_coords = np.argwhere(decor_mask)
        room["decor_count"] = int(decor_coords.shape[0])
        if decor_coords.size > 0:
            offsets = []
            for rr, cc in decor_coords:
                dr = int(rr) - mr_local
                dc = int(cc) - mc_local
                assert max(abs(dr), abs(dc)) == 1, "template must lie in the immediate 8-neighborhood"
                offsets.append((dr, dc, int(sub[rr, cc])))
            room["template_offsets"] = _canonical_offsets(offsets)
            exemplar_candidates.append(room)
        else:
            room["template_offsets"] = []

    assert len(exemplar_candidates) == 1, "expected exactly one decorated exemplar room"
    exemplar = exemplar_candidates[0]
    template_offsets = exemplar["template_offsets"]
    assert len(template_offsets) >= 3, "template must be non-empty and informative"

    return {
        "bg": bg,
        "rooms": rooms,
        "marker_color": marker_color,
        "exemplar": exemplar,
        "template_offsets": template_offsets,
    }


def _stamp_targets(
    grid: np.ndarray,
    analysis: dict,
    *,
    mode: str = "rule",
    clip_to_room: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    exemplar_orientation = analysis["exemplar"]["orientation"]
    base_offsets = analysis["template_offsets"]
    rotated_offsets = _rotate_cw(base_offsets)
    h, w = out.shape

    for room in analysis["rooms"]:
        if room is analysis["exemplar"]:
            continue
        if mode == "rule":
            offsets = base_offsets if room["orientation"] == exemplar_orientation else rotated_offsets
        elif mode == "original":
            offsets = base_offsets
        elif mode == "rotated":
            offsets = rotated_offsets
        else:
            raise ValueError(mode)

        mr, mc = room["marker"]
        r0, c0, r1, c1 = room["bbox"]
        for dr, dc, color in offsets:
            rr = mr + dr
            cc = mc + dc
            if clip_to_room:
                if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                    continue
            else:
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
            if (rr, cc) == (mr, mc):
                continue
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the decoration around the exemplar marker to every matching room marker, rotating it when the room orientation differs.

    Input structure:
    - A background contains several separated monochrome rectangular rooms, and every room is non-square.
    - Each room contains exactly one pixel of the same rare marker color.
    - Exactly one room has extra colored decoration in the marker's immediate 8-neighborhood; the other rooms are bare.

    Rule:
    - Treat the decorated room as the exemplar and record every decoration pixel's color and offset relative to the marker.
    - For each bare marker, keep the offsets unchanged if its room is oriented the same way as the exemplar (both wide or both tall).
    - If the room orientation is opposite, rotate all recorded offsets 90 degrees clockwise around the marker before stamping them.
    - Stamp only inside that room's rectangle; pixels that would leave the room are clipped away, while the room fill and marker stay unchanged elsewhere.

    Key insight:
    - The decoration is copied in marker-centered coordinates, but each target room's aspect ratio is the switch that decides whether the copied neighborhood must rotate.
    """
    analysis = _analyze(np.asarray(grid, dtype=int))
    return _stamp_targets(np.asarray(grid, dtype=int), analysis, mode="rule", clip_to_room=True)


def _other_orientation(orientation: str) -> str:
    return "V" if orientation == "H" else "H"


def _room_shape(orientation: str, cell_h: int, cell_w: int, *, large: bool) -> tuple[int, int]:
    if orientation == "H":
        h_low = 4 if large else 3
        h_high = min(5, cell_h)
        h = random.randint(h_low, h_high)
        w_low = max(h + 1, 5 if large else 4)
        w_high = min(cell_w, 7)
        w = random.randint(w_low, w_high)
        return h, w

    w_low = 4 if large else 3
    w_high = min(5, cell_w)
    w = random.randint(w_low, w_high)
    h_low = max(w + 1, 5 if large else 4)
    h_high = min(cell_h, 8)
    h = random.randint(h_low, h_high)
    return h, w


def _marker_local(h: int, w: int, mode: str) -> Coord:
    if mode == "interior":
        return random.randint(1, h - 2), random.randint(1, w - 2)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    return random.choice(corners)


def _transform_point(dr: int, dc: int, name: str) -> tuple[int, int]:
    if name == "id":
        return dr, dc
    if name == "rot90":
        return dc, -dr
    if name == "rot180":
        return -dr, -dc
    if name == "rot270":
        return -dc, dr
    if name == "flip_h":
        return -dr, dc
    if name == "flip_v":
        return dr, -dc
    if name == "diag":
        return dc, dr
    if name == "anti":
        return -dc, -dr
    raise ValueError(name)


def _template_is_asymmetric(offsets: list[tuple[int, int, int]]) -> bool:
    base = tuple(_canonical_offsets(offsets))
    for name in ("rot90", "rot180", "rot270", "flip_h", "flip_v", "diag", "anti"):
        moved = tuple(
            sorted((_transform_point(dr, dc, name)[0], _transform_point(dr, dc, name)[1], color) for dr, dc, color in offsets)
        )
        if moved == base:
            return False
    return True


def _random_template(template_colors: list[int]) -> list[tuple[int, int, int]]:
    for _ in range(200):
        count = random.randint(4, 6)
        cells = random.sample(_RING, count)
        if not any(abs(dr) + abs(dc) == 1 for dr, dc in cells):
            continue
        if not any(abs(dr) == 1 and abs(dc) == 1 for dr, dc in cells):
            continue
        used = random.randint(2, min(3, len(template_colors)))
        palette = random.sample(template_colors, used)
        color_stream = [palette[i % used] for i in range(count)]
        random.shuffle(color_stream)
        offsets = [(dr, dc, color) for (dr, dc), color in zip(cells, color_stream)]
        if len(set(color_stream)) != used:
            continue
        if _template_is_asymmetric(offsets):
            return _canonical_offsets(offsets)
    raise RuntimeError("failed to build a non-symmetric template")


def _room_global_cells(room: dict) -> list[Coord]:
    r0, c0, r1, c1 = room["bbox"]
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    room_count = random.randint(4, 5)
    cell_h = random.randint(8, 10)
    cell_w = random.randint(7, 8)
    grid_h = _GRID_BORDER * 2 + _LAYOUT_ROWS * cell_h + (_LAYOUT_ROWS - 1) * _CELL_GAP
    grid_w = _GRID_BORDER * 2 + _LAYOUT_COLS * cell_w + (_LAYOUT_COLS - 1) * _CELL_GAP

    colors = list(_ALL_COLORS)
    random.shuffle(colors)
    bg = colors[0]
    marker_color = colors[1]
    room_colors = colors[2 : 2 + room_count]
    template_colors = colors[2 + room_count : 5 + room_count]
    template_offsets = _random_template(template_colors)

    exemplar_orientation = random.choice(["H", "V"])
    roles = [
        ("exemplar", exemplar_orientation, "interior"),
        ("same_interior", exemplar_orientation, "interior"),
        ("opposite_interior", _other_orientation(exemplar_orientation), "interior"),
        ("clip_target", random.choice(["H", "V"]), "corner"),
    ]
    if room_count == 5:
        roles.append(
            (
                "extra",
                random.choice(["H", "V"]),
                random.choice(["interior", "corner"]),
            )
        )
    random.shuffle(roles)

    cell_ids = random.sample(range(_LAYOUT_ROWS * _LAYOUT_COLS), room_count)
    rooms = []
    exemplar_room: dict | None = None

    for idx, ((role, orientation, marker_mode), cell_id, room_color) in enumerate(
        zip(roles, cell_ids, room_colors)
    ):
        row_block = cell_id // _LAYOUT_COLS
        col_block = cell_id % _LAYOUT_COLS
        cell_r0 = _GRID_BORDER + row_block * (cell_h + _CELL_GAP)
        cell_c0 = _GRID_BORDER + col_block * (cell_w + _CELL_GAP)

        h, w = _room_shape(orientation, cell_h, cell_w, large=(role != "clip_target"))
        room_r0 = cell_r0 + random.randint(0, cell_h - h)
        room_c0 = cell_c0 + random.randint(0, cell_w - w)
        room_r1 = room_r0 + h - 1
        room_c1 = room_c0 + w - 1
        marker_local = _marker_local(h, w, marker_mode)
        marker = (room_r0 + marker_local[0], room_c0 + marker_local[1])

        room = {
            "role": role,
            "bbox": (room_r0, room_c0, room_r1, room_c1),
            "orientation": orientation,
            "room_color": int(room_color),
            "marker": marker,
            "marker_local": marker_local,
        }
        rooms.append(room)
        if role == "exemplar":
            exemplar_room = room

    assert exemplar_room is not None

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = room["room_color"]
        mr, mc = room["marker"]
        inp[mr, mc] = marker_color

    emr, emc = exemplar_room["marker"]
    for dr, dc, color in template_offsets:
        inp[emr + dr, emc + dc] = color

    expected = inp.copy()
    rotated_offsets = _rotate_cw(template_offsets)
    for room in rooms:
        if room is exemplar_room:
            continue
        offsets = template_offsets if room["orientation"] == exemplar_room["orientation"] else rotated_offsets
        mr, mc = room["marker"]
        r0, c0, r1, c1 = room["bbox"]
        for dr, dc, color in offsets:
            rr = mr + dr
            cc = mc + dc
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                expected[rr, cc] = color

    return inp, expected


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        analysis = _analyze(inp)
    except AssertionError:
        return False

    rooms = analysis["rooms"]
    exemplar = analysis["exemplar"]
    template_offsets = analysis["template_offsets"]
    rotated_offsets = _rotate_cw(template_offsets)

    if len(rooms) < 4:
        return False

    same_targets = [
        room for room in rooms if room is not exemplar and room["orientation"] == exemplar["orientation"]
    ]
    opposite_targets = [
        room for room in rooms if room is not exemplar and room["orientation"] != exemplar["orientation"]
    ]
    if not same_targets or not opposite_targets:
        return False

    clipped_targets = 0
    for room in rooms:
        if room is exemplar:
            continue
        offsets = template_offsets if room["orientation"] == exemplar["orientation"] else rotated_offsets
        mr, mc = room["marker"]
        r0, c0, r1, c1 = room["bbox"]
        if any(not (r0 <= mr + dr <= r1 and c0 <= mc + dc <= c1) for dr, dc, _color in offsets):
            clipped_targets += 1
    if clipped_targets == 0:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    no_rotate = _stamp_targets(inp, analysis, mode="original", clip_to_room=True)
    always_rotate = _stamp_targets(inp, analysis, mode="rotated", clip_to_room=True)
    grid_clip_only = _stamp_targets(inp, analysis, mode="rule", clip_to_room=False)
    if np.array_equal(expected, no_rotate):
        return False
    if np.array_equal(expected, always_rotate):
        return False
    if np.array_equal(expected, grid_clip_only):
        return False

    room_cells = {cell for room in rooms for cell in _room_global_cells(room)}
    outside_room_changes = {
        (r, c)
        for r, c in zip(*np.where(expected != inp))
        if (int(r), int(c)) not in room_cells
    }
    if outside_room_changes:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        inp, expected = _build_instance()
        if _validate_instance(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid room-rotation instance")
