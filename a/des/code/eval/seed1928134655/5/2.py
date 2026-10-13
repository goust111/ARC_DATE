from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
BORDER_COLOR = 5
GEN_RETRIES = 400
SIDES = ("up", "right", "down", "left")
SIDE_TO_TURNS = {"up": 0, "right": 1, "down": 2, "left": 3}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rotate_offsets(offsets: frozenset[tuple[int, int]], turns: int) -> frozenset[tuple[int, int]]:
    out = set(offsets)
    for _ in range(turns % 4):
        out = {(c, -r) for r, c in out}
    return frozenset(out)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(cells),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(cells),
                }
            )
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_side(missing: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> str | None:
    if not missing:
        return None
    rows = {r for r, _ in missing}
    cols = {c for _, c in missing}
    if len(rows) == 1:
        row = next(iter(rows))
        ordered = sorted(cols)
        if any(b != a + 1 for a, b in zip(ordered, ordered[1:])):
            return None
        if row == r0:
            return "up"
        if row == r1:
            return "down"
    if len(cols) == 1:
        col = next(iter(cols))
        ordered = sorted(rows)
        if any(b != a + 1 for a, b in zip(ordered, ordered[1:])):
            return None
        if col == c0:
            return "left"
        if col == c1:
            return "right"
    return None


def _classify_canvas(grid: np.ndarray, comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    perimeter = _perimeter_cells(r0, c0, r1, c1)
    if not comp["coord_set"].issubset(perimeter):
        return None
    missing = perimeter - comp["coord_set"]
    if len(missing) not in (2, 3):
        return None
    side = _gap_side(missing, r0, c0, r1, c1)
    if side is None:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or np.any(inner == comp["color"]):
        return None
    marker_cells = np.argwhere(inner != BG)
    if len(marker_cells) != 1:
        return None
    mr, mc = map(int, marker_cells[0])
    marker_color = int(inner[mr, mc])
    if marker_color == comp["color"]:
        return None
    return {
        "bbox": comp["bbox"],
        "border_color": comp["color"],
        "gap_side": side,
        "gap_size": len(missing),
        "missing": missing,
        "anchor": (r0 + 1 + mr, c0 + 1 + mc),
        "marker_color": marker_color,
        "component": comp,
    }


def _inside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _classify_stamp(comp: dict, canvases: list[dict]) -> dict | None:
    if comp["area"] < 2 or comp["area"] > 5:
        return None
    if any(_inside_bbox(tuple(coord), canvas["bbox"]) for coord in comp["coords"] for canvas in canvases):
        return None
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h % 2 == 0 or w % 2 == 0:
        return None
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    if center not in comp["coord_set"]:
        return None
    offsets = frozenset((r - center[0], c - center[1]) for r, c in comp["coord_set"])
    return {"color": comp["color"], "center": center, "offsets": offsets, "bbox": comp["bbox"]}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify canvases with a single border gap, pair them with same-color outside
    stamps, rotate the stamp toward the gap direction, and stamp it onto the
    canvas anchor.
    """
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g)

    canvases: list[dict] = []
    used_ids: set[int] = set()
    for idx, comp in enumerate(comps):
        canvas = _classify_canvas(g, comp)
        if canvas is not None:
            canvases.append(canvas)
            used_ids.add(idx)
    assert canvases, "at least one canvas is required"

    marker_colors = [canvas["marker_color"] for canvas in canvases]
    assert len(set(marker_colors)) == len(marker_colors), "canvas marker colors must be unique"

    stamps: list[dict] = []
    border_colors = {canvas["border_color"] for canvas in canvases}
    for idx, comp in enumerate(comps):
        if idx in used_ids or comp["color"] in border_colors:
            continue
        stamp = _classify_stamp(comp, canvases)
        if stamp is not None:
            stamps.append(stamp)

    stamp_by_color: dict[int, dict] = {}
    for stamp in stamps:
        assert stamp["color"] not in stamp_by_color, "stamp colors must be unique"
        stamp_by_color[stamp["color"]] = stamp

    out = g.copy()
    for canvas in canvases:
        assert canvas["marker_color"] in stamp_by_color, "every canvas must have a matching stamp"
        stamp = stamp_by_color[canvas["marker_color"]]
        rotated = _rotate_offsets(stamp["offsets"], SIDE_TO_TURNS[canvas["gap_side"]])
        r0, c0, r1, c1 = canvas["bbox"]
        ar, ac = canvas["anchor"]
        for dr, dc in rotated:
            rr = ar + dr
            cc = ac + dc
            assert r0 < rr < r1 and c0 < cc < c1, "stamp must stay inside the canvas interior"
            out[rr, cc] = stamp["color"]
    return out


def _shape_library() -> list[frozenset[tuple[int, int]]]:
    pts = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)]
    shapes = []
    for size in (4, 5):
        for cells in itertools.combinations(pts, size):
            shape = frozenset(cells)
            if (0, 0) not in shape or (-1, 0) not in shape:
                continue
            rows = [r for r, _ in shape]
            cols = [c for _, c in shape]
            if min(rows) != -1 or max(rows) != 1 or min(cols) != -1 or max(cols) != 1:
                continue
            rots = {_rotate_offsets(shape, k) for k in range(4)}
            if len(rots) != 4:
                continue
            shapes.append(shape)
    assert shapes
    return shapes


SHAPES = _shape_library()


def _bbox_free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_cells(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _draw_canvas(grid: np.ndarray, spec: dict) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    color = spec["border_color"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    for r, c in spec["missing"]:
        grid[r, c] = BG
    ar, ac = spec["anchor"]
    grid[ar, ac] = spec["marker_color"]


def _canvas_cells(spec: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    cells = sorted(_perimeter_cells(r0, c0, r1, c1) - spec["missing"])
    cells.append(spec["anchor"])
    return cells


def _gap_positions(side: str, bbox: tuple[int, int, int, int], gap_size: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side in {"up", "down"}:
        start = random.randint(c0 + 1, c1 - gap_size)
        row = r0 if side == "up" else r1
        return {(row, c) for c in range(start, start + gap_size)}
    start = random.randint(r0 + 1, r1 - gap_size)
    col = c0 if side == "left" else c1
    return {(r, col) for r in range(start, start + gap_size)}


def _make_canvas_spec(h: int, w: int, color: int, marker_color: int) -> dict:
    inner_h = random.choice((5, 7))
    inner_w = random.choice((5, 7))
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r0 = random.randint(2, h - outer_h - 2)
    c0 = random.randint(2, w - outer_w - 2)
    bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    anchor = (r0 + 1 + inner_h // 2, c0 + 1 + inner_w // 2)
    side = random.choice(SIDES)
    gap_size = random.choice((2, 3))
    missing = _gap_positions(side, bbox, gap_size)
    return {
        "bbox": bbox,
        "border_color": color,
        "gap_side": side,
        "gap_size": gap_size,
        "missing": missing,
        "anchor": anchor,
        "marker_color": marker_color,
    }


def _stamp_cells(shape: frozenset[tuple[int, int]], center: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr, dc in shape]


def _stamp_bbox(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return (r - 1, c - 1, r + 1, c + 1)


def _candidate_stamp_centers(canvas: dict, grid_shape: tuple[int, int], side: str) -> list[tuple[int, int]]:
    h, w = grid_shape
    r0, c0, r1, c1 = canvas["bbox"]
    centers = []
    distances = [1, 2, 3]
    random.shuffle(distances)
    if side in {"up", "down"}:
        cols = list(range(max(1, c0 + 1), min(w - 2, c1 - 1) + 1))
        random.shuffle(cols)
        for dist in distances:
            row = r0 - dist - 2 if side == "up" else r1 + dist + 2
            for col in cols:
                centers.append((row, col))
    else:
        rows = list(range(max(1, r0 + 1), min(h - 2, r1 - 1) + 1))
        random.shuffle(rows)
        for dist in distances:
            col = c0 - dist - 2 if side == "left" else c1 + dist + 2
            for row in rows:
                centers.append((row, col))
    return centers


def _place_canvases(h: int, w: int, colors: list[int]) -> list[dict] | None:
    occupied = np.zeros((h, w), dtype=bool)
    canvases: list[dict] = []
    for marker_color in colors:
        placed = False
        for _ in range(80):
            spec = _make_canvas_spec(h, w, BORDER_COLOR, marker_color)
            if not _bbox_free(occupied, spec["bbox"], pad=2):
                continue
            cells = _canvas_cells(spec)
            _mark_cells(occupied, cells)
            canvases.append(spec)
            placed = True
            break
        if not placed:
            return None
    return canvases


def _place_stamps(grid: np.ndarray, canvases: list[dict], colors: list[int], shapes: list[frozenset[tuple[int, int]]]) -> list[dict] | None:
    occupied = grid != BG
    stamps = []
    for color, canvas, shape in zip(colors, canvases, shapes):
        sides = list(SIDES)
        random.shuffle(sides)
        placed = False
        for side in sides:
            for center in _candidate_stamp_centers(canvas, grid.shape, side):
                bbox = _stamp_bbox(center)
                if not _bbox_free(occupied, bbox, pad=1):
                    continue
                cells = _stamp_cells(shape, center)
                if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in cells):
                    continue
                for r, c in cells:
                    grid[r, c] = color
                _mark_cells(occupied, cells)
                stamps.append({"color": color, "center": center, "offsets": shape, "bbox": bbox})
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return stamps


def _expected_output(inp: np.ndarray, canvases: list[dict], stamps: list[dict]) -> np.ndarray:
    by_color = {stamp["color"]: stamp for stamp in stamps}
    out = inp.copy()
    for canvas in canvases:
        shape = by_color[canvas["marker_color"]]["offsets"]
        rotated = _rotate_offsets(shape, SIDE_TO_TURNS[canvas["gap_side"]])
        ar, ac = canvas["anchor"]
        for dr, dc in rotated:
            out[ar + dr, ac + dc] = canvas["marker_color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    n_canvases = random.randint(1, 3)
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    random.shuffle(palette)
    colors = palette[:n_canvases]

    canvases = _place_canvases(h, w, colors)
    if canvases is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for canvas in canvases:
        _draw_canvas(grid, canvas)

    shapes = [random.choice(SHAPES) for _ in range(n_canvases)]
    stamps = _place_stamps(grid, canvases, colors, shapes)
    if stamps is None:
        return None

    expected = _expected_output(grid, canvases, stamps)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
