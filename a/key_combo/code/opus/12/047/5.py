from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BAR = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bar_orientation(coords: np.ndarray) -> tuple[str, int, int, int]:
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    if len(rows) == 1:
        row = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1))
        length = len(cols)
        assert length % 2 == 1
        return "H", row, cols[length // 2], length
    assert len(cols) == 1
    col = cols[0]
    assert rows == list(range(rows[0], rows[-1] + 1))
    length = len(rows)
    assert length % 2 == 1
    return "V", rows[length // 2], col, length


def _segment_cells(center: tuple[int, int], length: int, vertical: bool) -> list[tuple[int, int]]:
    r, c = center
    half = length // 2
    if vertical:
        return [(r + dr, c) for dr in range(-half, half + 1)]
    return [(r, c + dc) for dc in range(-half, half + 1)]


def _rotated_bar_cells(coords: np.ndarray) -> list[tuple[int, int]]:
    orient, cr, cc, length = _bar_orientation(coords)
    return _segment_cells((cr, cc), length, vertical=(orient == "H"))


def _ray_meta(
    bbox: tuple[int, int, int, int], marker: tuple[int, int]
) -> dict[str, object]:
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    horizontal = r0 <= mr <= r1 and (mc < c0 or mc > c1)
    vertical = c0 <= mc <= c1 and (mr < r0 or mr > r1)
    assert horizontal ^ vertical

    if horizontal:
        dc = -1 if mc < c0 else 1
        side_c = c0 - 1 if dc < 0 else c1 + 1
        line = [(mr, c) for c in range(side_c, mc + dc, dc) if c != mc]
        return {
            "axis": "H",
            "step": (0, dc),
            "start": (mr, side_c),
            "side": "left" if dc < 0 else "right",
            "line": line,
            "rect_cap": [(mr - 1, side_c), (mr, side_c), (mr + 1, side_c)],
            "marker_cap": [(mr - 1, mc), (mr + 1, mc)],
            "target": (mr, mc),
            "beyond": (mr, mc + dc),
        }

    dr = -1 if mr < r0 else 1
    side_r = r0 - 1 if dr < 0 else r1 + 1
    line = [(r, mc) for r in range(side_r, mr + dr, dr) if r != mr]
    return {
        "axis": "V",
        "step": (dr, 0),
        "start": (side_r, mc),
        "side": "top" if dr < 0 else "bottom",
        "line": line,
        "rect_cap": [(side_r, mc - 1), (side_r, mc), (side_r, mc + 1)],
        "marker_cap": [(mr, mc - 1), (mr, mc + 1)],
        "target": (mr, mc),
        "beyond": (mr + dr, mc),
    }


def _find_structures(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    groups: list[dict[str, object]] = []
    markers: set[tuple[int, int]] = set()
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (bg, BAR)):
        comps = _components(grid == color)
        rects = [comp for comp in comps if len(comp) > 1]
        singles = [tuple(map(int, comp[0])) for comp in comps if len(comp) == 1]
        assert len(rects) == 1 and singles
        box = _bbox(rects[0])
        r0, c0, r1, c1 = box
        assert len(rects[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)
        marker_list = sorted(singles)
        for pos in marker_list:
            markers.add(pos)
        groups.append({"color": color, "bbox": box, "markers": marker_list})
    assert groups
    return groups, markers


def _scene_after_bars(grid: np.ndarray, bg: int, rotate_bars: bool) -> np.ndarray:
    scene = np.array(grid, dtype=int, copy=True)
    if not rotate_bars:
        return scene

    parts = _components(scene == BAR)
    scene[scene == BAR] = bg
    for comp in parts:
        for r, c in _rotated_bar_cells(comp):
            assert 0 <= r < scene.shape[0] and 0 <= c < scene.shape[1]
            assert int(scene[r, c]) == bg
            scene[r, c] = BAR
    return scene


def _ray_status(
    scene: np.ndarray,
    bg: int,
    bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
    marker_positions: set[tuple[int, int]],
) -> dict[str, object]:
    meta = _ray_meta(bbox, marker)
    r, c = meta["start"]
    dr, dc = meta["step"]
    target = meta["target"]
    trail: list[tuple[int, int]] = []
    h, w = scene.shape

    while 0 <= r < h and 0 <= c < w:
        if (r, c) == target:
            return {"success": True, "trail": trail, "meta": meta}
        if int(scene[r, c]) != bg and (r, c) not in marker_positions:
            return {"success": False, "trail": trail, "meta": meta, "obstacle": (r, c)}
        trail.append((r, c))
        r += dr
        c += dc
    raise AssertionError("marker should be reached or blocked before leaving the grid")


def _success_cells(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], marker: tuple[int, int]
) -> list[tuple[int, int]]:
    meta = _ray_meta(bbox, marker)
    cells = set(meta["line"])
    cells.update(meta["rect_cap"])
    cells.update(meta["marker_cap"])
    h, w = shape
    for r, c in cells:
        assert 0 <= r < h and 0 <= c < w
    return sorted(cells)


def _transform(grid: np.ndarray, rotate_bars: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    groups, marker_positions = _find_structures(g, bg)
    scene = _scene_after_bars(g, bg, rotate_bars=rotate_bars)

    out = scene.copy()
    for group in groups:
        color = int(group["color"])
        box = group["bbox"]
        for marker in group["markers"]:
            status = _ray_status(scene, bg, box, marker, marker_positions)
            paint = (
                _success_cells(scene.shape, box, marker)
                if status["success"]
                else list(status["trail"])
            )
            for r, c in paint:
                if int(out[r, c]) not in (bg, color):
                    assert int(out[r, c]) == color
                out[r, c] = color

    for r, c in marker_positions:
        out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate grey bars into blockers, then extend rectangle rays until marks or rotated obstacles.

    Input structure:
    - The background is black, with two or more non-grey colors each forming one solid filled rectangle plus several same-color singleton markers.
    - Every marker lies outside its rectangle but aligned with the rectangle on one shared row or one shared column.
    - Grey foreground objects are odd-length one-pixel-wide horizontal or vertical bars separate from the rectangles and markers.

    Rule:
    - Rotate every grey bar 90 degrees around its own center, erase its original cells, and keep only the rotated bar.
    - For each rectangle and each same-color aligned marker, trace a one-pixel ray from the adjacent rectangle side toward that marker.
    - If the traced route hits any rotated grey pixel or any other non-background non-marker pixel first, stop one cell before the obstacle and draw no caps.
    - If the route reaches the marker, draw the full a2d730bd-style protrusion: the straight ray, a three-pixel cap at the rectangle side, and the crossed cap around the marker while clearing the marker cell itself.
    - Clear all original marker pixels in the final output.

    Key insight:
    - The grey bars matter only after rotation: their rotated footprints, not their input footprints, decide which otherwise aligned rectangle-to-marker rays survive to draw endpoint caps.
    """
    return _transform(grid, rotate_bars=True)


def _place_rectangle(zone: str, h: int, w: int, height: int, width: int) -> tuple[int, int]:
    if zone == "tl":
        r0 = random.randint(4, height // 2 - h - 1)
        c0 = random.randint(4, width // 2 - w - 1)
        return r0, c0
    r0 = random.randint(height // 2 + 1, height - h - 5)
    c0 = random.randint(width // 2 + 1, width - w - 5)
    return r0, c0


def _sample_marker_specs(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
    forced_block_axis: str,
) -> list[dict[str, object]] | None:
    r0, c0, r1, c1 = bbox
    horiz_side = random.choice(["left", "right"])
    vert_side = random.choice(["top", "bottom"])
    count = random.choice([2, 3])
    sides = [horiz_side, vert_side]
    if count == 3:
        remaining = [side for side in SIDES if side not in sides]
        sides.append(random.choice(remaining))

    blocked_side = horiz_side if forced_block_axis == "H" else vert_side
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    specs: list[dict[str, object]] = []

    for side in sides:
        blocked = side == blocked_side
        if side in ("left", "right"):
            row_choices = [r for r in range(r0 + 1, r1) if r not in used_rows]
            if not row_choices:
                return None
            row = random.choice(row_choices)
            used_rows.add(row)
            room = c0 if side == "left" else width - 1 - c1
            dmin = 6 if blocked else 4
            dmax = min(7, room - 1)
            if dmin > dmax:
                return None
            dist = random.randint(dmin, dmax)
            col = c0 - dist if side == "left" else c1 + dist
            specs.append({"side": side, "pos": (row, col), "blocked": blocked})
        else:
            col_choices = [c for c in range(c0 + 1, c1) if c not in used_cols]
            if not col_choices:
                return None
            col = random.choice(col_choices)
            used_cols.add(col)
            room = r0 if side == "top" else height - 1 - r1
            dmin = 6 if blocked else 4
            dmax = min(7, room - 1)
            if dmin > dmax:
                return None
            dist = random.randint(dmin, dmax)
            row = r0 - dist if side == "top" else r1 + dist
            specs.append({"side": side, "pos": (row, col), "blocked": blocked})

    assert any(spec["blocked"] for spec in specs)
    return specs


def _rect_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _raw_ray_cells(bbox: tuple[int, int, int, int], marker: tuple[int, int]) -> dict[str, object]:
    meta = _ray_meta(bbox, marker)
    line = list(meta["line"])
    assert line
    return {
        "line": line,
        "interior": line[:-1],
        "caps": list(meta["rect_cap"]) + list(meta["marker_cap"]),
        "beyond": meta["beyond"],
        "axis": meta["axis"],
    }


def _coords_in_bounds(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _touch4(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        out.add((r, c))
        out.add((r - 1, c))
        out.add((r + 1, c))
        out.add((r, c - 1))
        out.add((r, c + 1))
    return out


def _render_input(
    shape: tuple[int, int],
    rects: list[dict[str, object]],
    bars: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        color = int(rect["color"])
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        for marker in rect["markers"]:
            mr, mc = marker["pos"]
            grid[mr, mc] = color
    for cells in bars:
        for r, c in cells:
            grid[r, c] = BAR
    return grid


def _place_block_bar(
    shape: tuple[int, int],
    ray: dict[str, object],
    occupied_static: set[tuple[int, int]],
    used_input: set[tuple[int, int]],
    used_rotated: set[tuple[int, int]],
    forbidden_input: set[tuple[int, int]],
    forbidden_rotated: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    interior = list(ray["interior"])
    random.shuffle(interior)
    lengths = [5, 3, 7]
    random.shuffle(lengths)

    for center in interior:
        if center in reserved_paths:
            continue
        for length in lengths:
            rotated_vertical = ray["axis"] == "H"
            rotated = _segment_cells(center, length, vertical=rotated_vertical)
            original = _segment_cells(center, length, vertical=not rotated_vertical)
            if not _coords_in_bounds(rotated, shape) or not _coords_in_bounds(original, shape):
                continue
            if set(rotated) & occupied_static or set(original) & occupied_static:
                continue
            if set(rotated) & used_rotated or set(original) & used_input:
                continue
            if set(rotated) & forbidden_rotated or set(original) & forbidden_input:
                continue
            return original, rotated
    return None


def _place_extra_bar(
    shape: tuple[int, int],
    forbidden_rotated: set[tuple[int, int]],
    occupied_static: set[tuple[int, int]],
    used_input: set[tuple[int, int]],
    used_rotated: set[tuple[int, int]],
    forbidden_input: set[tuple[int, int]],
    forbidden_rotated_bars: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape
    for _ in range(200):
        length = random.choice([3, 5, 7])
        vertical = random.choice([False, True])
        half = length // 2
        r = random.randint(half, h - 1 - half)
        c = random.randint(half, w - 1 - half)
        original = _segment_cells((r, c), length, vertical=vertical)
        rotated = _segment_cells((r, c), length, vertical=not vertical)
        if set(original) & occupied_static or set(rotated) & occupied_static:
            continue
        if set(original) & used_input or set(rotated) & used_rotated:
            continue
        if set(original) & forbidden_input or set(rotated) & forbidden_rotated_bars:
            continue
        if set(rotated) & forbidden_rotated:
            continue
        return original, rotated
    return None


def _validate_candidate(grid: np.ndarray, rects: list[dict[str, object]]) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    try:
        alt = _transform(grid, rotate_bars=False)
        if np.array_equal(out, alt):
            return False
    except AssertionError:
        pass

    bg = _get_bg(grid)
    groups, marker_positions = _find_structures(grid, bg)
    by_color = {int(group["color"]): group for group in groups}
    scene = _scene_after_bars(grid, bg, rotate_bars=True)

    success_counts: dict[int, int] = {}
    blocked_total = 0
    blocked_h = False
    blocked_v = False

    for rect in rects:
        color = int(rect["color"])
        group = by_color[color]
        for marker in rect["markers"]:
            pos = marker["pos"]
            expected_blocked = bool(marker["blocked"])
            status = _ray_status(scene, bg, group["bbox"], pos, marker_positions)
            if status["success"] == expected_blocked:
                return False
            if status["success"]:
                success_counts[color] = success_counts.get(color, 0) + 1
            else:
                blocked_total += 1
                if marker["side"] in ("left", "right"):
                    blocked_h = True
                else:
                    blocked_v = True
            if int(out[pos]) != bg:
                return False

    if blocked_total < 2 or not blocked_h or not blocked_v:
        return False
    if any(success_counts.get(int(rect["color"]), 0) < 1 for rect in rects):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(22, 28)
    width = random.randint(22, 28)
    colors = random.sample(list(RECT_COLORS), 2)

    rects: list[dict[str, object]] = []
    for idx, zone in enumerate(("tl", "br")):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        if min(h, w) < 3:
            return None
        r0, c0 = _place_rectangle(zone, h, w, height, width)
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        forced_axis = "H" if idx == 0 else "V"
        markers = _sample_marker_specs(bbox, height, width, forced_axis)
        if markers is None:
            return None
        rects.append({"color": colors[idx], "bbox": bbox, "markers": markers})

    occupied_static: set[tuple[int, int]] = set()
    for rect in rects:
        occupied_static |= _rect_cells(rect["bbox"])
        for marker in rect["markers"]:
            if marker["pos"] in occupied_static:
                return None
            occupied_static.add(marker["pos"])

    ray_sets: dict[tuple[int, tuple[int, int]], dict[str, object]] = {}
    cell_count: Counter[tuple[int, int]] = Counter()
    cap_cells: set[tuple[int, int]] = set()
    for rect in rects:
        color = int(rect["color"])
        for marker in rect["markers"]:
            ray = _raw_ray_cells(rect["bbox"], marker["pos"])
            if any(cell in occupied_static for cell in ray["interior"]):
                return None
            if not marker["blocked"]:
                footprint = set(ray["caps"]) | {ray["beyond"]}
                if footprint & occupied_static:
                    return None
                cap_cells |= footprint
            key = (color, marker["pos"])
            ray_sets[key] = ray
            for cell in ray["interior"]:
                cell_count[cell] += 1

    bars_input: list[list[tuple[int, int]]] = []
    used_input: set[tuple[int, int]] = set()
    used_rotated: set[tuple[int, int]] = set()
    forbidden_input_bars: set[tuple[int, int]] = set()
    forbidden_rotated_bars: set[tuple[int, int]] = set()

    for rect in rects:
        color = int(rect["color"])
        for marker in rect["markers"]:
            if not marker["blocked"]:
                continue
            ray = ray_sets[(color, marker["pos"])]
            unique_path = {cell for cell in ray["interior"] if cell_count[cell] == 1}
            bar = _place_block_bar(
                (height, width),
                ray,
                occupied_static,
                used_input,
                used_rotated,
                forbidden_input_bars,
                forbidden_rotated_bars,
                reserved_paths=set(ray["interior"]) - unique_path,
            )
            if bar is None:
                return None
            original, rotated = bar
            bars_input.append(original)
            used_input.update(original)
            used_rotated.update(rotated)
            forbidden_input_bars.update(_touch4(original))
            forbidden_rotated_bars.update(_touch4(rotated))

    extra_count = random.choice([0, 1])
    forbidden_rotated = set().union(*(set(ray["interior"]) for ray in ray_sets.values())) | cap_cells
    for _ in range(extra_count):
        bar = _place_extra_bar(
            (height, width),
            forbidden_rotated,
            occupied_static,
            used_input,
            used_rotated,
            forbidden_input_bars,
            forbidden_rotated_bars,
        )
        if bar is None:
            return None
        original, rotated = bar
        bars_input.append(original)
        used_input.update(original)
        used_rotated.update(rotated)
        forbidden_input_bars.update(_touch4(original))
        forbidden_rotated_bars.update(_touch4(rotated))

    if len(bars_input) not in (2, 3):
        return None

    grid = _render_input((height, width), rects, bars_input)
    if not _validate_candidate(grid, rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotated-bar ray-cutting input")
