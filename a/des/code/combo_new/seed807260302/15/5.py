from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 28)
ROOM_COUNT_OPTIONS = (2, 3, 4)
ROOM_COUNT_WEIGHTS = (2, 3, 2)
ROOM_SIZE_RANGE = (5, 8)
ROOM_PLACE_RETRIES = 120
GEN_RETRIES = 400

SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_options(room: dict, side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _axis_from_side(side: str) -> str:
    return "vertical" if side in ("top", "bottom") else "horizontal"


def _orthogonal_axis(axis: str) -> str:
    return "horizontal" if axis == "vertical" else "vertical"


def _line_capacity(room: dict, axis: str) -> int:
    return room["inner_h"] if axis == "vertical" else room["inner_w"]


def _parse_room_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict | None:
    color = int(grid[coords[0]])
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return None

    border = _perimeter_cells(r0, c0, r1, c1)
    coord_set = set(coords)
    if not coord_set <= border:
        return None
    if len(coord_set) != len(border) - 1:
        return None

    gap_cells = border - coord_set
    if len(gap_cells) != 1:
        return None
    gap = next(iter(gap_cells))
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    if gap[0] == r0:
        gap_side = "top"
    elif gap[0] == r1:
        gap_side = "bottom"
    elif gap[1] == c0:
        gap_side = "left"
    else:
        gap_side = "right"

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    marker_mask = (inner != bg) & (inner != color)
    marker_coords_local = np.argwhere(marker_mask)
    if marker_coords_local.shape[0] < 2:
        return None

    marker_colors = np.unique(inner[marker_mask])
    if marker_colors.size != 1:
        return None
    marker_color = int(marker_colors[0])

    marker_coords = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in marker_coords_local]
    rows = {r for r, _ in marker_coords}
    cols = {c for _, c in marker_coords}
    if len(rows) == 1 and len(cols) > 1:
        marker_axis = "horizontal"
    elif len(cols) == 1 and len(rows) > 1:
        marker_axis = "vertical"
    else:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "frame_color": color,
        "border": coord_set,
        "gap": gap,
        "gap_side": gap_side,
        "marker_color": marker_color,
        "marker_count": len(marker_coords),
        "marker_axis": marker_axis,
    }


def _analyze_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    rooms: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components_mask(g == color):
            room = _parse_room_component(g, comp, bg)
            if room is not None:
                rooms.append(room)
    assert rooms, "expected at least one valid room"
    frame_colors = {room["frame_color"] for room in rooms}
    assert len(frame_colors) == 1, "all rooms must share the same frame color"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit outward rays from each room gap using marker count for length and marker orientation for solid-vs-dashed style.

    Input structure:
    - The background is black.
    - One or more grey rectangular room outlines appear as 4-connected border components, each with exactly one single-cell gap on its border.
    - Each room contains one same-color marker line inside the border; marker pixels are non-black and non-grey.

    Rule:
    - For each room, count its interior marker pixels to get the ray length N.
    - The gap side determines the outward ray direction.
    - If the marker line orientation is parallel to the ray direction, draw a solid ray of length N starting one cell outside the gap.
    - If the marker line orientation is perpendicular to the ray direction, draw a dashed ray: color steps 1, 3, 5, ... and leave steps 2, 4, 6, ... black.
    - Stop early if the ray would hit another room's grey border, and keep every room border, gap, and interior marker unchanged.

    Key insight:
    - Three cues must be combined per room: marker count gives ray length, gap position gives direction, and the marker-line orientation relative to that direction chooses solid versus dashed drawing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    rooms = _analyze_rooms(g)

    frame_mask: set[tuple[int, int]] = set()
    for room in rooms:
        frame_mask |= room["border"]

    out = g.copy()
    for room in rooms:
        dr, dc = SIDE_TO_DIR[room["gap_side"]]
        ray_axis = _axis_from_side(room["gap_side"])
        is_solid = room["marker_axis"] == ray_axis
        gap_r, gap_c = room["gap"]
        for step in range(1, room["marker_count"] + 1):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if (rr, cc) in frame_mask:
                break
            if is_solid or step % 2 == 1:
                out[rr, cc] = room["marker_color"]
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _place_room_boxes(rng: random.Random, h: int, w: int, room_count: int) -> list[dict] | None:
    rooms: list[dict] = []
    dims = [
        (
            rng.randint(*ROOM_SIZE_RANGE),
            rng.randint(*ROOM_SIZE_RANGE),
        )
        for _ in range(room_count)
    ]
    dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)
    for idx, (rh, rw) in enumerate(dims):
        placed = False
        for _ in range(ROOM_PLACE_RETRIES):
            r0 = rng.randint(1, h - rh - 2)
            c0 = rng.randint(1, w - rw - 2)
            bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(_boxes_overlap(bbox, other["bbox"], pad=1) for other in rooms):
                continue
            rooms.append(
                {
                    "id": idx,
                    "bbox": bbox,
                    "outer_h": rh,
                    "outer_w": rw,
                    "inner_h": rh - 2,
                    "inner_w": rw - 2,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return rooms


def _closed_border_owner_map(rooms: list[dict]) -> dict[tuple[int, int], int]:
    owner: dict[tuple[int, int], int] = {}
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        for cell in _perimeter_cells(r0, c0, r1, c1):
            owner[cell] = room["id"]
    return owner


def _first_closed_border_hit(
    owner: dict[tuple[int, int], int],
    src_id: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
) -> tuple[int, int, tuple[int, int]] | None:
    dr, dc = direction
    rr, cc = start
    step = 0
    while True:
        rr += dr
        cc += dc
        step += 1
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        owner_id = owner.get((rr, cc))
        if owner_id is not None and owner_id != src_id:
            return owner_id, step, (rr, cc)


def _edge_steps_from_gap(gap: tuple[int, int], side: str, h: int, w: int) -> int:
    r, c = gap
    if side == "top":
        return r
    if side == "bottom":
        return h - 1 - r
    if side == "left":
        return c
    return w - 1 - c


def _path_prefix(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    sr, sc = start
    return [(sr + dr * step, sc + dc * step) for step in range(1, length + 1)]


def _marker_coords(
    room: dict,
    axis: str,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = room["bbox"]
    if axis == "horizontal":
        row = r0 + 1 + rng.randrange(room["inner_h"])
        start = c0 + 1 + rng.randrange(room["inner_w"] - count + 1)
        return [(row, start + i) for i in range(count)]
    col = c0 + 1 + rng.randrange(room["inner_w"])
    start = r0 + 1 + rng.randrange(room["inner_h"] - count + 1)
    return [(start + i, col) for i in range(count)]


def _candidate_truncated_specs(
    rooms: list[dict],
    owner: dict[tuple[int, int], int],
    h: int,
    w: int,
) -> list[dict]:
    specs: list[dict] = []
    by_id = {room["id"]: room for room in rooms}
    for src in rooms:
        sr0, sc0, sr1, sc1 = src["bbox"]
        for blocker in rooms:
            if blocker["id"] == src["id"]:
                continue
            br0, bc0, br1, bc1 = blocker["bbox"]

            if sr1 < br0:
                side = "bottom"
                direction = SIDE_TO_DIR[side]
                cols = range(max(sc0 + 1, bc0 + 1), min(sc1 - 1, bc1 - 1) + 1)
                for c in cols:
                    gap = (sr1, c)
                    hit = _first_closed_border_hit(owner, src["id"], gap, direction, h, w)
                    if hit is None or hit[0] != blocker["id"] or hit[2] != (br0, c):
                        continue
                    if hit[1] < 3:
                        continue
                    specs.extend(_expand_truncated_styles(src, side, gap, hit[1]))

            if br1 < sr0:
                side = "top"
                direction = SIDE_TO_DIR[side]
                cols = range(max(sc0 + 1, bc0 + 1), min(sc1 - 1, bc1 - 1) + 1)
                for c in cols:
                    gap = (sr0, c)
                    hit = _first_closed_border_hit(owner, src["id"], gap, direction, h, w)
                    if hit is None or hit[0] != blocker["id"] or hit[2] != (br1, c):
                        continue
                    if hit[1] < 3:
                        continue
                    specs.extend(_expand_truncated_styles(src, side, gap, hit[1]))

            if sc1 < bc0:
                side = "right"
                direction = SIDE_TO_DIR[side]
                rows = range(max(sr0 + 1, br0 + 1), min(sr1 - 1, br1 - 1) + 1)
                for r in rows:
                    gap = (r, sc1)
                    hit = _first_closed_border_hit(owner, src["id"], gap, direction, h, w)
                    if hit is None or hit[0] != blocker["id"] or hit[2] != (r, bc0):
                        continue
                    if hit[1] < 3:
                        continue
                    specs.extend(_expand_truncated_styles(src, side, gap, hit[1]))

            if bc1 < sc0:
                side = "left"
                direction = SIDE_TO_DIR[side]
                rows = range(max(sr0 + 1, br0 + 1), min(sr1 - 1, br1 - 1) + 1)
                for r in rows:
                    gap = (r, sc0)
                    hit = _first_closed_border_hit(owner, src["id"], gap, direction, h, w)
                    if hit is None or hit[0] != blocker["id"] or hit[2] != (r, bc1):
                        continue
                    if hit[1] < 3:
                        continue
                    specs.extend(_expand_truncated_styles(src, side, gap, hit[1]))
    return specs


def _expand_truncated_styles(room: dict, side: str, gap: tuple[int, int], hit_step: int) -> list[dict]:
    ray_axis = _axis_from_side(side)
    specs: list[dict] = []
    for style, marker_axis in (
        ("solid", ray_axis),
        ("dashed", _orthogonal_axis(ray_axis)),
    ):
        max_n = _line_capacity(room, marker_axis)
        if max_n < max(2, hit_step):
            continue
        specs.append(
            {
                "kind": "truncated",
                "room_id": room["id"],
                "side": side,
                "gap": gap,
                "style": style,
                "marker_axis": marker_axis,
                "n_min": max(2, hit_step),
                "n_max": max_n,
                "visible_max": hit_step - 1,
            }
        )
    return specs


def _build_untruncated_candidates(
    room: dict,
    preferred_style: str,
    owner: dict[tuple[int, int], int],
    reserved_paths: set[tuple[int, int]],
    forbidden_gaps: dict[str, set[tuple[int, int]]],
    h: int,
    w: int,
) -> list[dict]:
    candidates: list[dict] = []
    styles = [preferred_style, "solid" if preferred_style == "dashed" else "dashed"]
    seen_keys: set[tuple[str, tuple[int, int], str]] = set()

    for style in styles:
        for side in SIDES:
            direction = SIDE_TO_DIR[side]
            ray_axis = _axis_from_side(side)
            marker_axis = ray_axis if style == "solid" else _orthogonal_axis(ray_axis)
            max_line = _line_capacity(room, marker_axis)
            if max_line < 2:
                continue
            for gap in _gap_options(room, side):
                if gap in forbidden_gaps[side]:
                    continue
                edge_steps = _edge_steps_from_gap(gap, side, h, w)
                hit = _first_closed_border_hit(owner, room["id"], gap, direction, h, w)
                max_clear = edge_steps if hit is None else min(edge_steps, hit[1] - 1)
                if max_clear < 2:
                    continue
                available = 0
                prefix = _path_prefix(gap, direction, max_clear)
                for cell in prefix:
                    if cell in reserved_paths:
                        break
                    available += 1
                n_max = min(max_line, available)
                if n_max < 2:
                    continue
                key = (side, gap, style)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidates.append(
                    {
                        "kind": "full",
                        "room_id": room["id"],
                        "side": side,
                        "gap": gap,
                        "style": style,
                        "marker_axis": marker_axis,
                        "n_min": 2,
                        "n_max": n_max,
                    }
                )
    return candidates


def _choose_assignment(
    candidate: dict,
    room: dict,
    color: int,
    rng: random.Random,
) -> dict:
    count = rng.randint(candidate["n_min"], candidate["n_max"])
    markers = _marker_coords(room, candidate["marker_axis"], count, rng)
    direction = SIDE_TO_DIR[candidate["side"]]
    visible_len = min(count, candidate.get("visible_max", count))
    path = _path_prefix(candidate["gap"], direction, visible_len)
    return {
        "gap": candidate["gap"],
        "gap_side": candidate["side"],
        "marker_axis": candidate["marker_axis"],
        "marker_count": count,
        "marker_color": color,
        "marker_coords": markers,
        "ray_path": path,
        "style": candidate["style"],
    }


def _render_input(rooms: list[dict], assignments: dict[int, dict]) -> np.ndarray:
    h = max(room["bbox"][2] for room in rooms) + 2
    w = max(room["bbox"][3] for room in rooms) + 2
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        for cell in _perimeter_cells(r0, c0, r1, c1):
            grid[cell] = FRAME_COLOR
        assign = assignments[room["id"]]
        grid[assign["gap"]] = BACKGROUND
        for cell in assign["marker_coords"]:
            grid[cell] = assign["marker_color"]
    return grid


def _classify_sample(grid: np.ndarray) -> tuple[int, int]:
    rooms = _analyze_rooms(grid)
    frame_mask: set[tuple[int, int]] = set()
    for room in rooms:
        frame_mask |= room["border"]

    truncated = 0
    dashed = 0
    h, w = grid.shape
    for room in rooms:
        if room["marker_axis"] != _axis_from_side(room["gap_side"]):
            dashed += 1
        dr, dc = SIDE_TO_DIR[room["gap_side"]]
        gap_r, gap_c = room["gap"]
        for step in range(1, room["marker_count"] + 1):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if (rr, cc) in frame_mask:
                truncated += 1
                break
    return truncated, dashed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    room_count = rng.choices(ROOM_COUNT_OPTIONS, weights=ROOM_COUNT_WEIGHTS, k=1)[0]
    rooms = _place_room_boxes(rng, h, w, room_count)
    if rooms is None:
        return None

    owner = _closed_border_owner_map(rooms)
    want_truncated = rng.random() < 0.5
    colors = list(MARKER_COLORS)
    rng.shuffle(colors)
    preferred_styles = ["solid"] * ((room_count + 1) // 2) + ["dashed"] * (room_count // 2)
    rng.shuffle(preferred_styles)

    assignments: dict[int, dict] = {}
    reserved_paths: set[tuple[int, int]] = set()
    forbidden_gaps: dict[int, dict[str, set[tuple[int, int]]]] = {
        room["id"]: {side: set() for side in SIDES} for room in rooms
    }

    if want_truncated:
        specs = _candidate_truncated_specs(rooms, owner, h, w)
        if not specs:
            return None
        spec = rng.choice(specs)
        src_room = next(room for room in rooms if room["id"] == spec["room_id"])
        block_side = {
            "top": "bottom",
            "bottom": "top",
            "left": "right",
            "right": "left",
        }[spec["side"]]
        end_cell = _path_prefix(spec["gap"], SIDE_TO_DIR[spec["side"]], spec["visible_max"] + 1)[-1]
        blocker_hit = (end_cell[0], end_cell[1])
        hit_room_id = _first_closed_border_hit(owner, src_room["id"], spec["gap"], SIDE_TO_DIR[spec["side"]], h, w)
        if hit_room_id is None:
            return None
        forbidden_gaps[hit_room_id[0]][block_side].add(blocker_hit)
        forced_style = spec["style"]
        color = colors.pop()
        chosen = _choose_assignment(spec, src_room, color, rng)
        assignments[src_room["id"]] = chosen
        reserved_paths.update(chosen["ray_path"])
        if forced_style in preferred_styles:
            preferred_styles.remove(forced_style)

    unassigned = {room["id"] for room in rooms if room["id"] not in assignments}
    room_by_id = {room["id"]: room for room in rooms}
    while unassigned:
        best_room_id = None
        best_candidates = None
        for room_id in unassigned:
            room = room_by_id[room_id]
            preferred = preferred_styles[0] if preferred_styles else rng.choice(["solid", "dashed"])
            candidates = _build_untruncated_candidates(
                room,
                preferred,
                owner,
                reserved_paths,
                forbidden_gaps[room_id],
                h,
                w,
            )
            if not candidates:
                return None
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_room_id = room_id
                best_candidates = candidates
        assert best_room_id is not None and best_candidates is not None
        candidate = rng.choice(best_candidates)
        color = colors.pop()
        chosen = _choose_assignment(candidate, room_by_id[best_room_id], color, rng)
        assignments[best_room_id] = chosen
        reserved_paths.update(chosen["ray_path"])
        if candidate["style"] in preferred_styles:
            preferred_styles.remove(candidate["style"])
        unassigned.remove(best_room_id)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        for cell in _perimeter_cells(r0, c0, r1, c1):
            grid[cell] = FRAME_COLOR
        assign = assignments[room["id"]]
        grid[assign["gap"]] = BACKGROUND
        for cell in assign["marker_coords"]:
            grid[cell] = assign["marker_color"]

    if len(_analyze_rooms(grid)) != room_count:
        return None
    truncated_count, dashed_count = _classify_sample(grid)
    if want_truncated and truncated_count == 0:
        return None
    if not want_truncated and truncated_count != 0:
        return None
    if room_count >= 2 and dashed_count == 0:
        return None
    if room_count >= 2 and dashed_count == room_count:
        return None
    if np.array_equal(grid, main(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
