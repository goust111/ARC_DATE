from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


_ARROW_DATA = {
    (0, 1): {
        "shape": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
        "tail": (1, 0),
        "nose": (1, 2),
    },
    (0, -1): {
        "shape": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
        "tail": (1, 2),
        "nose": (1, 0),
    },
    (-1, 0): {
        "shape": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
        "tail": (2, 1),
        "nose": (0, 1),
    },
    (1, 0): {
        "shape": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
        "tail": (0, 1),
        "nose": (2, 1),
    },
}
_ARROW_SHAPES = {spec["shape"]: direction for direction, spec in _ARROW_DATA.items()}


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], int, int]:
    r0, c0, _r1, _c1 = _bbox(coords)
    norm = frozenset((r - r0, c - c0) for r, c in coords)
    return norm, r0, c0


def _background_color(grid: np.ndarray) -> int:
    values = np.bincount(grid.ravel().astype(int), minlength=10)
    return int(np.argmax(values))


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _is_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    expected = 2 * (h + w) - 4
    if int(comp["area"]) != expected:
        return False
    pts = set(comp["coords"])
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            present = (r, c) in pts
            if on_border != present:
                return False
    return True


def _match_arrow(comp: dict) -> dict | None:
    if int(comp["area"]) != 5:
        return None
    shape, r0, c0 = _normalize(comp["coords"])
    direction = _ARROW_SHAPES.get(shape)
    if direction is None:
        return None
    spec = _ARROW_DATA[direction]
    tail = (r0 + spec["tail"][0], c0 + spec["tail"][1])
    nose = (r0 + spec["nose"][0], c0 + spec["nose"][1])
    return {
        "direction": direction,
        "tail": tail,
        "nose": nose,
    }


def _shape_mask(comp: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored object into the nearest grey frame indicated by its same-color arrow.

    Input structure:
    - Black background.
    - One-pixel-thick grey rectangular hollow frames.
    - Colored 4-connected objects outside the frames.
    - Same-color 5-pixel arrows shaped as >, <, ^, v.

    Rule:
    - Pair each arrow with the first same-color object visible from the arrow tail in the opposite direction.
    - Follow the arrow nose to the nearest frame along the pointing direction.
    - Erase arrows and source objects.
    - Stamp each source object's tight mask just inside the reached frame wall, preserving shape and color.
      Along the perpendicular axis, keep the object's original bbox center aligned.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _components(g, bg)

    frame_ids: set[int] = set()
    arrow_ids: set[int] = set()
    frames: list[dict] = []
    arrows: list[dict] = []
    objects: list[dict] = []

    for idx, comp in enumerate(comps):
        if _is_frame(comp):
            frame_ids.add(idx)
            frames.append({"id": idx, **comp})
            continue
        arrow_info = _match_arrow(comp)
        if arrow_info is not None:
            arrow_ids.add(idx)
            arrows.append({"id": idx, **comp, **arrow_info})
        else:
            objects.append({"id": idx, **comp})

    cell_to_comp: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for cell in comp["coords"]:
            cell_to_comp[cell] = idx

    out = g.copy()
    for comp in arrows + objects:
        for r, c in comp["coords"]:
            out[r, c] = bg

    used_objects: set[int] = set()
    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = arrow["direction"]
        pair_dir = (-dr, -dc)

        rr, cc = arrow["tail"][0] + pair_dir[0], arrow["tail"][1] + pair_dir[1]
        object_id = None
        while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
            if int(g[rr, cc]) == bg:
                rr += pair_dir[0]
                cc += pair_dir[1]
                continue
            comp_id = cell_to_comp[(rr, cc)]
            if comp_id in frame_ids or comp_id in arrow_ids:
                raise AssertionError("arrow tail line of sight hit a non-object")
            if int(comps[comp_id]["color"]) != color:
                raise AssertionError("arrow/object colors must match")
            object_id = comp_id
            break
        assert object_id is not None, "each arrow must see one object behind it"
        assert object_id not in used_objects, "each object should be matched once"
        used_objects.add(object_id)
        obj = comps[object_id]

        rr, cc = arrow["nose"][0] + dr, arrow["nose"][1] + dc
        frame = None
        while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
            if int(g[rr, cc]) == bg:
                rr += dr
                cc += dc
                continue
            comp_id = cell_to_comp[(rr, cc)]
            if comp_id not in frame_ids:
                raise AssertionError("arrow nose line of sight must reach a frame first")
            frame = comps[comp_id]
            break
        assert frame is not None, "each arrow must point at a frame"

        mask, (r0, c0, r1, c1) = _shape_mask(obj)
        obj_h, obj_w = mask.shape
        fr0, fc0, fr1, fc1 = frame["bbox"]

        if (dr, dc) == (0, 1):
            top = int(round((r0 + r1) / 2.0 - (obj_h - 1) / 2.0))
            left = fc0 + 1
        elif (dr, dc) == (0, -1):
            top = int(round((r0 + r1) / 2.0 - (obj_h - 1) / 2.0))
            left = fc1 - obj_w
        elif (dr, dc) == (1, 0):
            top = fr0 + 1
            left = int(round((c0 + c1) / 2.0 - (obj_w - 1) / 2.0))
        else:
            top = fr1 - obj_h
            left = int(round((c0 + c1) / 2.0 - (obj_w - 1) / 2.0))

        assert fr0 + 1 <= top <= fr1 - obj_h, "stamp must fit vertically inside the frame"
        assert fc0 + 1 <= left <= fc1 - obj_w, "stamp must fit horizontally inside the frame"
        for rr0 in range(obj_h):
            for cc0 in range(obj_w):
                if not mask[rr0, cc0]:
                    continue
                rr1, cc1 = top + rr0, left + cc0
                assert fr0 < rr1 < fr1 and fc0 < cc1 < fc1, "stamp must stay in frame interior"
                out[rr1, cc1] = color

    return out


def _draw_frame(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _rand_shape(rng: np.random.Generator) -> list[tuple[int, int]]:
    for _ in range(200):
        target = int(rng.integers(2, 6))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    frontier.add((r + dr, c + dc))
            frontier -= cells
            choice = list(frontier)[int(rng.integers(len(frontier)))]
            cells.add(choice)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        min_r, min_c = min(rows), min(cols)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        if frozenset(norm) in _ARROW_SHAPES:
            continue
        return norm
    raise RuntimeError("failed to sample a non-arrow object shape")


def _placeable(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    return True


def _shape_rows(shape: list[tuple[int, int]]) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = {}
    for r, c in shape:
        rows.setdefault(r, []).append(c)
    return rows


def _shape_cols(shape: list[tuple[int, int]]) -> dict[int, list[int]]:
    cols: dict[int, list[int]] = {}
    for r, c in shape:
        cols.setdefault(c, []).append(r)
    return cols


def _stamp_cells(shape: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _arrow_cells(direction: tuple[int, int], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in _ARROW_DATA[direction]["shape"]]


def _candidate_placements(
    grid: np.ndarray,
    reserved: np.ndarray,
    frame: tuple[int, int, int, int],
    shape: list[tuple[int, int]],
    direction: tuple[int, int],
) -> list[dict]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    row_groups = _shape_rows(shape)
    col_groups = _shape_cols(shape)
    candidates: list[dict] = []

    if direction == (0, 1):
        stamp_left = c0 + 1
        for stamp_top in range(r0 + 1, r1 - shape_h + 1):
            stamp = _stamp_cells(shape, stamp_top, stamp_left)
            if any(reserved[r, c] or int(grid[r, c]) != BG for r, c in stamp):
                continue
            for row_off, cols in row_groups.items():
                shaft_row = stamp_top + row_off
                for gap_frame in (1, 2, 3):
                    arrow_left = c0 - gap_frame - 3
                    if arrow_left < 0:
                        continue
                    for gap_obj in (1, 2, 3):
                        obj_left = arrow_left - gap_obj - shape_w
                        if obj_left < 0:
                            continue
                        object_cells = _stamp_cells(shape, stamp_top, obj_left)
                        arrow_top = shaft_row - 1
                        arrow = _arrow_cells(direction, arrow_top, arrow_left)
                        if not _placeable(grid, object_cells) or not _placeable(grid, arrow):
                            continue
                        nearest_obj = obj_left + max(cols)
                        if any(int(grid[shaft_row, cc]) != BG for cc in range(nearest_obj + 1, arrow_left)):
                            continue
                        if any(int(grid[shaft_row, cc]) != BG for cc in range(arrow_left + 3, c0)):
                            continue
                        if int(grid[shaft_row, c0]) != FRAME_COLOR:
                            continue
                        candidates.append({"object": object_cells, "arrow": arrow, "stamp": stamp})
    elif direction == (0, -1):
        stamp_left = c1 - shape_w
        for stamp_top in range(r0 + 1, r1 - shape_h + 1):
            stamp = _stamp_cells(shape, stamp_top, stamp_left)
            if any(reserved[r, c] or int(grid[r, c]) != BG for r, c in stamp):
                continue
            for row_off, cols in row_groups.items():
                shaft_row = stamp_top + row_off
                for gap_frame in (1, 2, 3):
                    arrow_left = c1 + gap_frame + 1
                    if arrow_left + 2 >= w:
                        continue
                    for gap_obj in (1, 2, 3):
                        obj_left = arrow_left + 3 + gap_obj
                        if obj_left + shape_w > w:
                            continue
                        object_cells = _stamp_cells(shape, stamp_top, obj_left)
                        arrow_top = shaft_row - 1
                        arrow = _arrow_cells(direction, arrow_top, arrow_left)
                        if not _placeable(grid, object_cells) or not _placeable(grid, arrow):
                            continue
                        nearest_obj = obj_left + min(cols)
                        if any(int(grid[shaft_row, cc]) != BG for cc in range(arrow_left + 3, nearest_obj)):
                            continue
                        if any(int(grid[shaft_row, cc]) != BG for cc in range(c1 + 1, arrow_left)):
                            continue
                        if int(grid[shaft_row, c1]) != FRAME_COLOR:
                            continue
                        candidates.append({"object": object_cells, "arrow": arrow, "stamp": stamp})
    elif direction == (1, 0):
        stamp_top = r0 + 1
        for stamp_left in range(c0 + 1, c1 - shape_w + 1):
            stamp = _stamp_cells(shape, stamp_top, stamp_left)
            if any(reserved[r, c] or int(grid[r, c]) != BG for r, c in stamp):
                continue
            for col_off, rows in col_groups.items():
                shaft_col = stamp_left + col_off
                for gap_frame in (1, 2, 3):
                    arrow_top = r0 - gap_frame - 3
                    if arrow_top < 0:
                        continue
                    for gap_obj in (1, 2, 3):
                        obj_top = arrow_top - gap_obj - shape_h
                        if obj_top < 0:
                            continue
                        object_cells = _stamp_cells(shape, obj_top, stamp_left)
                        arrow_left = shaft_col - 1
                        arrow = _arrow_cells(direction, arrow_top, arrow_left)
                        if not _placeable(grid, object_cells) or not _placeable(grid, arrow):
                            continue
                        nearest_obj = obj_top + max(rows)
                        if any(int(grid[rr, shaft_col]) != BG for rr in range(nearest_obj + 1, arrow_top)):
                            continue
                        if any(int(grid[rr, shaft_col]) != BG for rr in range(arrow_top + 3, r0)):
                            continue
                        if int(grid[r0, shaft_col]) != FRAME_COLOR:
                            continue
                        candidates.append({"object": object_cells, "arrow": arrow, "stamp": stamp})
    else:
        stamp_top = r1 - shape_h
        for stamp_left in range(c0 + 1, c1 - shape_w + 1):
            stamp = _stamp_cells(shape, stamp_top, stamp_left)
            if any(reserved[r, c] or int(grid[r, c]) != BG for r, c in stamp):
                continue
            for col_off, rows in col_groups.items():
                shaft_col = stamp_left + col_off
                for gap_frame in (1, 2, 3):
                    arrow_top = r1 + gap_frame + 1
                    if arrow_top + 2 >= h:
                        continue
                    for gap_obj in (1, 2, 3):
                        obj_top = arrow_top + 3 + gap_obj
                        if obj_top + shape_h > h:
                            continue
                        object_cells = _stamp_cells(shape, obj_top, stamp_left)
                        arrow_left = shaft_col - 1
                        arrow = _arrow_cells(direction, arrow_top, arrow_left)
                        if not _placeable(grid, object_cells) or not _placeable(grid, arrow):
                            continue
                        nearest_obj = obj_top + min(rows)
                        if any(int(grid[rr, shaft_col]) != BG for rr in range(arrow_top + 3, nearest_obj)):
                            continue
                        if any(int(grid[rr, shaft_col]) != BG for rr in range(r1 + 1, arrow_top)):
                            continue
                        if int(grid[r1, shaft_col]) != FRAME_COLOR:
                            continue
                        candidates.append({"object": object_cells, "arrow": arrow, "stamp": stamp})

    return candidates


def _sample_frames(rng: np.random.Generator, h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    n_frames = int(rng.integers(2, 4))
    frames: list[tuple[int, int, int, int]] = []
    for _ in range(n_frames):
        placed = False
        for _ in range(200):
            fh = int(rng.integers(6, 11))
            fw = int(rng.integers(6, 11))
            if fh >= h - 6 or fw >= w - 6:
                continue
            r0 = int(rng.integers(3, h - fh - 2))
            c0 = int(rng.integers(3, w - fw - 2))
            frame = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            ok = True
            for or0, oc0, or1, oc1 in frames:
                if not (frame[2] + 3 < or0 or or1 + 3 < frame[0] or frame[3] + 3 < oc0 or oc1 + 3 < frame[1]):
                    ok = False
                    break
            if ok:
                frames.append(frame)
                placed = True
                break
        if not placed:
            return None
    return frames


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(22, 31))
    w = int(rng.integers(22, 31))
    frames = _sample_frames(rng, h, w)
    if frames is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)

    reserved = np.zeros((h, w), dtype=bool)
    n_pairs = int(rng.integers(2, 5))
    colors = list(rng.choice(PAIR_COLORS, size=n_pairs, replace=False))
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for color in colors:
        placed = False
        for _ in range(100):
            shape = _rand_shape(rng)
            direction = directions[int(rng.integers(len(directions)))]
            frame_order = list(range(len(frames)))
            rng.shuffle(frame_order)
            candidates: list[dict] = []
            for idx in frame_order:
                candidates.extend(_candidate_placements(grid, reserved, frames[idx], shape, direction))
            if not candidates:
                continue
            choice = candidates[int(rng.integers(len(candidates)))]
            for r, c in choice["object"]:
                grid[r, c] = color
            for r, c in choice["arrow"]:
                grid[r, c] = color
            for r, c in choice["stamp"]:
                reserved[r, c] = True
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
