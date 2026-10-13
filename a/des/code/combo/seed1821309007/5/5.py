from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _frame_color_and_rooms(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidates: list[tuple[int, list[dict]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        rooms: list[dict] = []
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            expected = np.zeros_like(patch, dtype=bool)
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, 0] = True
            expected[:, -1] = True
            if not np.array_equal(patch, expected):
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            marker_cells = np.argwhere((inner != bg) & (inner != color))
            if marker_cells.shape[0] != 1:
                continue
            mr, mc = marker_cells[0]
            rooms.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "inner_shape": (r1 - r0 - 1, c1 - c0 - 1),
                    "marker": (int(r0 + 1 + mr), int(c0 + 1 + mc)),
                    "marker_color": int(inner[mr, mc]),
                }
            )
        if rooms:
            candidates.append((color, rooms))
    assert len(candidates) == 1, "expected a unique wall color with rectangular rooms"
    wall_color, rooms = candidates[0]
    return int(wall_color), rooms


def _direction_from_marker(room: dict) -> str:
    r0, c0, r1, c1 = room["bbox"]
    mr, mc = room["marker"]
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    dr = mr - cr
    dc = mc - cc
    if abs(dr) > abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _shape_from_component(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shape[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return shape


def _stamp_shape(out: np.ndarray, room: dict, shape: np.ndarray, wall_color: int) -> None:
    r0, c0, r1, c1 = room["bbox"]
    mr, mc = room["marker"]
    inner_top = r0 + 1
    inner_left = c0 + 1
    inner_bottom = r1 - 1
    inner_right = c1 - 1
    side = _direction_from_marker(room)
    sh, sw = shape.shape
    for sr in range(sh):
        for sc in range(sw):
            if not shape[sr, sc]:
                continue
            if side == "top":
                rr = inner_top + sr
                cc = mc + sc
            elif side == "bottom":
                rr = inner_bottom - sr
                cc = mc + sc
            elif side == "left":
                rr = mr + sr
                cc = inner_left + sc
            else:
                rr = mr + sr
                cc = inner_right - sc
            out[rr, cc] = wall_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    In closed rectangular rooms bounded by grey walls, use each room's marker color
    to select the matching outside monochrome blob, erase that blob, and stamp its
    shape as a grey inward protrusion on the marker-indicated inner wall.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    wall_color, rooms = _frame_color_and_rooms(g, bg)

    room_interiors = np.zeros(g.shape, dtype=bool)
    room_markers = {tuple(room["marker"]) for room in rooms}
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        room_interiors[r0 + 1 : r1, c0 + 1 : c1] = True

    external_by_color: dict[int, np.ndarray] = {}
    for color in np.unique(g):
        color = int(color)
        if color in {bg, wall_color}:
            continue
        for coords in _components(g == color):
            cells = {tuple(map(int, xy)) for xy in coords}
            if len(cells) == 1 and next(iter(cells)) in room_markers:
                continue
            assert not any(room_interiors[r, c] for r, c in cells), "non-marker colored cells cannot be inside rooms"
            assert color not in external_by_color, "external blob colors must be unique"
            external_by_color[color] = coords

    out = g.copy()
    for room in rooms:
        color = int(room["marker_color"])
        if color not in external_by_color:
            continue
        coords = external_by_color[color]
        out[coords[:, 0], coords[:, 1]] = bg
        _stamp_shape(out, room, _shape_from_component(coords), wall_color)
    return out


SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _variants(shape: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for rot in range(4):
        candidate = np.rot90(shape, rot)
        for arr in (candidate, np.fliplr(candidate)):
            if not any(np.array_equal(arr, old) for old in variants):
                variants.append(arr.copy())
    return variants


def _padded_box_ok(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    return not occ[rr0:rr1, cc0:cc1].any()


def _place_room(
    occ: np.ndarray, room_h: int, room_w: int
) -> tuple[int, int] | None:
    choices = [
        (r, c)
        for r in range(0, occ.shape[0] - room_h + 1)
        for c in range(0, occ.shape[1] - room_w + 1)
        if _padded_box_ok(occ, r, c, room_h, room_w, pad=2)
    ]
    if not choices:
        return None
    return random.choice(choices)


def _marker_candidates(inner_h: int, inner_w: int, shape: np.ndarray, side: str) -> list[tuple[int, int]]:
    sh, sw = shape.shape
    candidates: list[tuple[int, int]] = []
    cr = (inner_h - 1) / 2.0
    cc = (inner_w - 1) / 2.0
    for mr in range(inner_h):
        for mc in range(inner_w):
            if side in {"top", "bottom"} and mc + sw > inner_w:
                continue
            if side in {"left", "right"} and mr + sh > inner_h:
                continue
            dr = mr - cr
            dc = mc - cc
            inferred = "bottom" if abs(dr) > abs(dc) and dr > 0 else None
            if abs(dr) > abs(dc) and dr < 0:
                inferred = "top"
            if abs(dr) <= abs(dc) and dc < 0:
                inferred = "left"
            if abs(dr) <= abs(dc) and dc > 0:
                inferred = "right"
            if inferred == side:
                candidates.append((mr, mc))
    return candidates


def _place_blob(
    grid: np.ndarray, occ: np.ndarray, color: int, shape: np.ndarray
) -> np.ndarray | None:
    sh, sw = shape.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(grid.shape[0] - sh + 1):
        for c0 in range(grid.shape[1] - sw + 1):
            if not _padded_box_ok(occ, r0, c0, sh, sw, pad=1):
                continue
            ok = True
            for sr in range(sh):
                for sc in range(sw):
                    if shape[sr, sc] and grid[r0 + sr, c0 + sc] != BG:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    coords: list[tuple[int, int]] = []
    for sr in range(sh):
        for sc in range(sw):
            if shape[sr, sc]:
                grid[r0 + sr, c0 + sc] = color
                coords.append((r0 + sr, c0 + sc))
    occ[r0 : r0 + sh, c0 : c0 + sw] |= shape
    return np.array(coords, dtype=int)


def _manual_expected(grid: np.ndarray, room_specs: list[dict], blob_specs: dict[int, np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for room in room_specs:
        color = int(room["marker_color"])
        if color not in blob_specs:
            continue
        coords = blob_specs[color]
        out[coords[:, 0], coords[:, 1]] = BG
        _stamp_shape(out, room, _shape_from_component(coords), WALL)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    usable_colors = [c for c in range(1, 10) if c != WALL]
    random.shuffle(usable_colors)
    room_count = random.randint(1, 3)
    distractor_count = random.randint(0, 2)
    if len(usable_colors) < room_count + distractor_count:
        return None

    sides = random.sample(["top", "bottom", "left", "right"], k=room_count) if room_count <= 4 else [
        random.choice(["top", "bottom", "left", "right"]) for _ in range(room_count)
    ]
    room_specs: list[dict] = []
    matched_blob_shapes: dict[int, np.ndarray] = {}
    blob_specs: dict[int, np.ndarray] = {}

    for i in range(room_count):
        marker_color = usable_colors[i]
        base_shape = random.choice(SHAPES)
        shape = random.choice(_variants(base_shape))
        sh, sw = shape.shape
        inner_h = random.randint(max(4, sh + 1), max(4, sh + 4))
        inner_w = random.randint(max(4, sw + 1), max(4, sw + 4))
        room_h = inner_h + 2
        room_w = inner_w + 2
        spot = _place_room(occ, room_h, room_w)
        if spot is None:
            return None
        r0, c0 = spot
        marker_choices = _marker_candidates(inner_h, inner_w, shape, sides[i])
        if not marker_choices:
            return None
        mr_in, mc_in = random.choice(marker_choices)
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        marker = (r0 + 1 + mr_in, c0 + 1 + mc_in)
        grid[marker] = marker_color
        occ[r0 : r1 + 1, c0 : c1 + 1] = True
        room_specs.append(
            {
                "bbox": (r0, c0, r1, c1),
                "marker": marker,
                "marker_color": marker_color,
            }
        )
        matched_blob_shapes[marker_color] = shape

    for color, shape in matched_blob_shapes.items():
        coords = _place_blob(grid, occ, color, shape)
        if coords is None:
            return None
        blob_specs[color] = coords

    offset = room_count
    for j in range(distractor_count):
        color = usable_colors[offset + j]
        shape = random.choice(_variants(random.choice(SHAPES)))
        if _place_blob(grid, occ, color, shape) is None:
            return None

    expected = _manual_expected(grid, room_specs, blob_specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid sample")
