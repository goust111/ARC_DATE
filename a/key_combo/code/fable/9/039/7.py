from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


RED = 2
PALETTE = tuple(range(10))
BBox = tuple[int, int, int, int]
Coord = tuple[int, int]

GRID_RETRIES = 400


@dataclass(frozen=True)
class SceneObject:
    kind: str  # "frame" or "rect"
    color: int
    bbox: BBox
    z: int
    missing: frozenset[Coord] = field(default_factory=frozenset)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != int(bg))


def _bbox_h(bbox: BBox) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: BBox) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _on_bbox_border(r: int, c: int, bbox: BBox) -> bool:
    r0, c0, r1, c1 = bbox
    return r in (r0, r1) or c in (c0, c1)


def _bbox_border_cells(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _diagonal_corner_cells(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    return [(r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)]


def _actual_cells(obj: SceneObject) -> set[Coord]:
    r0, c0, r1, c1 = obj.bbox
    if obj.kind == "rect":
        return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    cells = set(_bbox_border_cells(obj.bbox))
    cells.difference_update(obj.missing)
    return cells


def _render_scene(shape: tuple[int, int], bg: int, objects: list[SceneObject]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for obj in sorted(objects, key=lambda item: item.z):
        for r, c in _actual_cells(obj):
            grid[r, c] = int(obj.color)
    return grid


def _is_frame_candidate(grid: np.ndarray, color: int) -> tuple[bool, BBox | None]:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return False, None
    bbox = _bbox_from_coords(coords)
    if _bbox_h(bbox) != _bbox_w(bbox) or _bbox_h(bbox) < 3:
        return False, None
    if not all(_on_bbox_border(int(r), int(c), bbox) for r, c in coords):
        return False, None
    return True, bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark every square frame whose only missing-looking border parts are truly hidden by visible top objects.

    Input structure:
    - A dominant background contains several monochrome square frames and several
      monochrome filled rectangles; only the topmost object's color is visible at each cell.
    - Each color belongs to a single object whose visible pixels still reveal its full bbox.
    - Square-frame colors show only bbox-border pixels, while rectangle colors may fill
      interior cells of a non-square bbox.

    Rule:
    - Keep the whole input unchanged.
    - Recover each square-frame candidate from all visible pixels of one color, using its
      full square bbox border as the intended frame outline.
    - If any cell on that outline still shows background, the frame has a true visible gap,
      including gaps seen through the cavity of a higher hollow frame; otherwise the frame
      counts as complete because every non-self border cell is visibly covered by a top object.
    - For every complete frame, paint one red cell on each diagonal outside corner position,
      but only when that corner-adjacent cell is in bounds and currently background.

    Key insight:
    - Forgiveness depends on what is visibly covering the border cell, not on another object's
      bbox: background visible through a hollow occluder's cavity is still decisive evidence of a real gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in _colors_present(g, bg):
        is_frame, bbox = _is_frame_candidate(g, int(color))
        if not is_frame or bbox is None:
            continue
        if any(int(g[r, c]) == int(bg) for r, c in _bbox_border_cells(bbox)):
            continue
        for r, c in _diagonal_corner_cells(bbox):
            if _in_bounds(g.shape, r, c) and int(g[r, c]) == int(bg):
                out[r, c] = RED
    return out


def _frame(r0: int, c0: int, side: int, color: int, z: int, missing: set[Coord] | None = None) -> SceneObject:
    return SceneObject("frame", color, (r0, c0, r0 + side - 1, c0 + side - 1), z, frozenset(missing or set()))


def _rect(r0: int, c0: int, h: int, w: int, color: int, z: int) -> SceneObject:
    return SceneObject("rect", color, (r0, c0, r0 + h - 1, c0 + w - 1), z)


def _rand_zone_frame(zone: tuple[int, int, int, int], side_lo: int, side_hi: int) -> tuple[int, int, int]:
    zr0, zr1, zc0, zc1 = zone
    max_side = min(side_hi, zr1 - zr0 - 1, zc1 - zc0 - 1)
    min_side = min(side_lo, max_side)
    if max_side < 4:
        raise ValueError("zone too small for a frame")
    side = random.randint(max(4, min_side), max_side)
    r0 = random.randint(zr0 + 1, zr1 - side)
    c0 = random.randint(zc0 + 1, zc1 - side)
    return r0, c0, side


def _gap_cells_for_side(bbox: BBox, side: str, length: int = 1) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    side_len = _bbox_w(bbox) if side in ("top", "bottom") else _bbox_h(bbox)
    assert side_len >= length + 2
    offset = random.randint(1, side_len - length - 1)
    if side == "top":
        return [(r0, c0 + offset + k) for k in range(length)]
    if side == "bottom":
        return [(r1, c0 + offset + k) for k in range(length)]
    if side == "left":
        return [(r0 + offset + k, c0) for k in range(length)]
    if side == "right":
        return [(r0 + offset + k, c1) for k in range(length)]
    raise ValueError(side)


def _occluding_rect_for_frame(frame_bbox: BBox, color: int, z: int) -> SceneObject:
    r0, c0, r1, c1 = frame_bbox
    side = random.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        h = random.randint(2, 3)
        w = random.randint(3, max(3, min(4, _bbox_w(frame_bbox) - 1)))
        w = min(w, _bbox_w(frame_bbox) - 1)
        start_c = random.randint(c0 + 1, c1 - w + 1)
        start_r = r0 - (h - 1) if side == "top" else r1
    else:
        h = random.randint(3, max(3, min(4, _bbox_h(frame_bbox) - 1)))
        w = random.randint(2, 3)
        h = min(h, _bbox_h(frame_bbox) - 1)
        start_r = random.randint(r0 + 1, r1 - h + 1)
        start_c = c0 - (w - 1) if side == "left" else c1
    if h == w:
        if side in ("top", "bottom"):
            w += 1
        else:
            h += 1
    return _rect(start_r, start_c, h, w, color, z)


def _bottom_right_rect(zone: tuple[int, int, int, int], color: int, z: int) -> SceneObject:
    zr0, zr1, zc0, zc1 = zone
    h = random.randint(2, min(4, zr1 - zr0 - 1))
    w = random.randint(3, min(5, zc1 - zc0 - 1))
    if h == w:
        w += 1 if w + 1 <= zc1 - zc0 - 1 else -1
    r0 = random.randint(zr0 + 1, zr1 - h)
    c0 = random.randint(zc0 + 1, zc1 - w)
    return _rect(r0, c0, h, w, color, z)


def _candidate_bbox_from_render(grid: np.ndarray, color: int) -> BBox:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        raise AssertionError(f"color {color} disappeared")
    return _bbox_from_coords(coords)


def _frame_complete_in_scene(rendered: np.ndarray, frame_obj: SceneObject, bg: int) -> bool:
    return all(int(rendered[r, c]) != int(bg) for r, c in _bbox_border_cells(frame_obj.bbox))


def _expected_from_scene(rendered: np.ndarray, objects: list[SceneObject], bg: int) -> np.ndarray:
    out = rendered.copy()
    for obj in objects:
        if obj.kind != "frame":
            continue
        if not _frame_complete_in_scene(rendered, obj, bg):
            continue
        for r, c in _diagonal_corner_cells(obj.bbox):
            if _in_bounds(rendered.shape, r, c) and int(rendered[r, c]) == int(bg):
                out[r, c] = RED
    return out


def _inside_bbox(cell: Coord, bbox: BBox) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _naive_bbox_thinks_complete(rendered: np.ndarray, frame_obj: SceneObject, objects: list[SceneObject], bg: int) -> bool:
    higher = [obj for obj in objects if obj.z > frame_obj.z]
    for cell in _bbox_border_cells(frame_obj.bbox):
        visible = int(rendered[cell])
        if visible == frame_obj.color:
            continue
        if visible != bg:
            continue
        if not any(_inside_bbox(cell, obj.bbox) for obj in higher):
            return False
    return True


def _validate_scene(rendered: np.ndarray, objects: list[SceneObject], bg: int) -> None:
    assert int(_get_bg(rendered)) == int(bg)
    expected = _expected_from_scene(rendered, objects, bg)
    actual = main(rendered)
    assert np.array_equal(actual, expected)

    good_frames: list[SceneObject] = []
    bad_frames: list[SceneObject] = []
    forgiven_witness = False
    exposed_gap_witness = False
    cavity_witness = False
    naive_failure_witness = False

    for obj in objects:
        visible_bbox = _candidate_bbox_from_render(rendered, obj.color)
        assert visible_bbox == obj.bbox
        coords = np.argwhere(rendered == int(obj.color))
        if obj.kind == "frame":
            assert all(_on_bbox_border(int(r), int(c), obj.bbox) for r, c in coords)
        else:
            assert _bbox_h(obj.bbox) != _bbox_w(obj.bbox)
            inner = rendered[obj.bbox[0] + 1 : obj.bbox[2], obj.bbox[1] + 1 : obj.bbox[3]]
            if inner.size:
                assert np.any(inner == obj.color)

    for obj in objects:
        if obj.kind != "frame":
            continue
        is_complete = _frame_complete_in_scene(rendered, obj, bg)
        if is_complete:
            good_frames.append(obj)
            if any(int(rendered[r, c]) not in (int(bg), obj.color) for r, c in _bbox_border_cells(obj.bbox)):
                forgiven_witness = True
        else:
            bad_frames.append(obj)
            bg_cells = [(r, c) for r, c in _bbox_border_cells(obj.bbox) if int(rendered[r, c]) == int(bg)]
            assert bg_cells
            if any(not any(_inside_bbox(cell, other.bbox) for other in objects if other.z > obj.z) for cell in bg_cells):
                exposed_gap_witness = True
            for cell in bg_cells:
                for other in objects:
                    if other.kind != "frame" or other.z <= obj.z:
                        continue
                    if _inside_bbox(cell, other.bbox) and cell not in _actual_cells(other):
                        cavity_witness = True
            if _naive_bbox_thinks_complete(rendered, obj, objects, bg):
                naive_failure_witness = True

    assert good_frames and bad_frames
    assert forgiven_witness
    assert exposed_gap_witness
    assert cavity_witness
    assert naive_failure_witness


def _build_scene() -> np.ndarray | None:
    h = random.randint(22, 28)
    w = random.randint(22, 28)
    bg = random.choice([0, 7])
    colors = random.sample([c for c in PALETTE if c not in {bg, RED}], 6)
    frame_a_color, rect_a_color, frame_b_color, frame_c_color, frame_d_color, rect_e_color = colors

    top_split = h // 2
    left_split = w // 2
    zones = {
        "A": (1, top_split - 1, 1, left_split - 1),
        "B": (1, top_split - 1, left_split + 1, w - 2),
        "C": (top_split + 1, h - 2, 1, left_split - 1),
        "E": (top_split + 1, h - 2, left_split + 1, w - 2),
    }

    try:
        a_r0, a_c0, a_side = _rand_zone_frame(zones["A"], 5, 7)
        b_r0, b_c0, b_side = _rand_zone_frame(zones["B"], 5, 7)
        c_r0, c_c0, c_side = _rand_zone_frame(zones["C"], 6, 8)
    except ValueError:
        return None

    frame_a = _frame(a_r0, a_c0, a_side, frame_a_color, z=0)
    rect_a = _occluding_rect_for_frame(frame_a.bbox, rect_a_color, z=3)

    gap_b_len = random.choice([1, 2])
    frame_b = _frame(
        b_r0,
        b_c0,
        b_side,
        frame_b_color,
        z=0,
        missing=set(_gap_cells_for_side((b_r0, b_c0, b_r0 + b_side - 1, b_c0 + b_side - 1), random.choice(["top", "bottom", "left", "right"]), gap_b_len)),
    )

    c_bbox = (c_r0, c_c0, c_r0 + c_side - 1, c_c0 + c_side - 1)
    gap_c = random.choice(_gap_cells_for_side(c_bbox, "top", 1))
    frame_c = _frame(c_r0, c_c0, c_side, frame_c_color, z=0, missing={gap_c})
    gap_r, gap_c_col = gap_c
    d_side = 4
    d_r0 = gap_r - 1
    d_c0 = gap_c_col - 1
    frame_d = _frame(d_r0, d_c0, d_side, frame_d_color, z=4)

    rect_e = _bottom_right_rect(zones["E"], rect_e_color, z=1)

    objects = [frame_a, frame_b, frame_c, rect_e, rect_a, frame_d]

    if any(not all(_in_bounds((h, w), r, c) for r, c in _actual_cells(obj)) for obj in objects):
        return None

    rendered = _render_scene((h, w), bg, objects)
    if np.count_nonzero(rendered == bg) <= rendered.size // 2:
        return None

    try:
        _validate_scene(rendered, objects, bg)
    except Exception:
        return None
    return rendered


def generate_input() -> np.ndarray:
    for _ in range(GRID_RETRIES):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GRID_RETRIES} attempts")
