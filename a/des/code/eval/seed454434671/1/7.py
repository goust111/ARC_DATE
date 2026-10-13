from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray, *, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    elif connectivity == 8:
        dirs = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
    else:
        raise ValueError("connectivity must be 4 or 8")

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1))
    )


def _cropped_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _ring_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    out = []
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            out.append((r, c))
    return out


def _pack_patterns(
    patterns: list[np.ndarray], height: int, width: int, color: int, bg: int
) -> np.ndarray:
    canvas = np.full((height, width), bg, dtype=int)
    cur_r = 0
    cur_c = 0
    row_h = 0
    for mask in patterns:
        ph, pw = map(int, mask.shape)
        if cur_c > 0 and cur_c + pw > width:
            cur_r += row_h
            cur_c = 0
            row_h = 0
        assert pw <= width, "pattern wider than room"
        assert cur_r + ph <= height, "patterns do not fit in room"
        window = canvas[cur_r : cur_r + ph, cur_c : cur_c + pw]
        window[mask] = color
        cur_c += pw
        row_h = max(row_h, ph)
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count wall markers for each closed room, consume the shelf patterns in order,
    recolor them by room marker color, and pack them into room interiors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _get_bg(g)

    border_parts = [
        coords
        for coords in _components(g != bg, connectivity=4)
        if _touches_edge(coords, h, w)
    ]
    assert border_parts, "expected shelf patterns on the grid edge"

    union = np.concatenate(border_parts, axis=0)
    sr0, sc0, sr1, sc1 = _bbox(union)
    edges = []
    if sr0 == 0:
        edges.append("top")
    if sr1 == h - 1:
        edges.append("bottom")
    if sc0 == 0:
        edges.append("left")
    if sc1 == w - 1:
        edges.append("right")
    assert len(edges) == 1, "shelf must occupy exactly one outer edge"
    shelf_edge = edges[0]
    horizontal = shelf_edge in ("top", "bottom")

    shelf_patterns = []
    shelf_bboxes = {id(coords): _bbox(coords) for coords in border_parts}
    for coords in sorted(
        border_parts,
        key=lambda pts: (
            shelf_bboxes[id(pts)][1],
            shelf_bboxes[id(pts)][0],
        )
        if horizontal
        else (
            shelf_bboxes[id(pts)][0],
            shelf_bboxes[id(pts)][1],
        ),
    ):
        values = np.unique(g[coords[:, 0], coords[:, 1]])
        assert values.size == 1, "each shelf item must be monochrome"
        shelf_patterns.append(_cropped_mask(coords))

    rooms = []
    for coords in _components(g == bg, connectivity=4):
        if _touches_edge(coords, h, w):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue

        ring = _ring_coords(r0, c0, r1, c1)
        ring_vals = [int(g[r, c]) for r, c in ring if int(g[r, c]) != bg]
        counts = Counter(ring_vals)
        if len(counts) < 2:
            continue
        wall_color, _wall_count = max(counts.items(), key=lambda item: item[1])
        marker_colors = [color for color in counts if color != wall_color]
        if len(marker_colors) != 1:
            continue
        marker_color = int(marker_colors[0])
        marker_positions = [(r, c) for r, c in ring if int(g[r, c]) == marker_color]
        if not marker_positions:
            continue
        first_pos = min(r * w + c for r, c in marker_positions)
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "marker_color": marker_color,
                "marker_count": int(counts[marker_color]),
                "first_pos": int(first_pos),
            }
        )

    assert rooms, "expected enclosed rectangular rooms"
    rooms.sort(key=lambda room: room["first_pos"])

    expected_pattern_total = sum(int(room["marker_count"]) for room in rooms)
    assert expected_pattern_total == len(shelf_patterns), "shelf count must match markers"

    out = g.copy()
    for coords in border_parts:
        out[coords[:, 0], coords[:, 1]] = bg

    cursor = 0
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        count = int(room["marker_count"])
        chosen = shelf_patterns[cursor : cursor + count]
        cursor += count
        packed = _pack_patterns(
            chosen,
            r1 - r0 + 1,
            c1 - c0 + 1,
            int(room["marker_color"]),
            bg,
        )
        out[r0 : r1 + 1, c0 : c1 + 1] = packed

    return out


# --------------- generator ---------------

BG = 0
GRID_MIN = 20
GRID_MAX = 30
ROOM_COUNT_RANGE = (2, 4)
TOTAL_PATTERN_MAX = 6
PATTERNS_PER_ROOM_MAX = 3
INNER_MIN = 3
INNER_MAX = 7
SHELF_THICKNESS = 3
MAX_GENERATION_ATTEMPTS = 400

PATTERN_LIBRARY = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
)


def _sample_counts(n_rooms: int) -> list[int]:
    low = max(3, n_rooms)
    high = min(TOTAL_PATTERN_MAX, PATTERNS_PER_ROOM_MAX * n_rooms)
    total = random.randint(low, high)
    counts = []
    remaining = total
    for i in range(n_rooms):
        rooms_left = n_rooms - i - 1
        lo = max(1, remaining - rooms_left * PATTERNS_PER_ROOM_MAX)
        hi = min(PATTERNS_PER_ROOM_MAX, remaining - rooms_left)
        counts.append(random.randint(lo, hi))
        remaining -= counts[-1]
    random.shuffle(counts)
    return counts


def _used_bbox_dims(packed: np.ndarray, bg: int) -> tuple[int, int]:
    coords = np.argwhere(packed != bg)
    if coords.size == 0:
        return 0, 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _sample_room_patterns(count: int) -> dict[str, object] | None:
    patterns = [random.choice(PATTERN_LIBRARY).copy() for _ in range(count)]
    max_ph = max(int(mask.shape[0]) for mask in patterns)
    max_pw = max(int(mask.shape[1]) for mask in patterns)
    total_pw = sum(int(mask.shape[1]) for mask in patterns)

    for _ in range(150):
        width_choices = list(range(max(INNER_MIN, max_pw), INNER_MAX + 1))
        if count > 1 and random.random() < 0.5:
            cutoff = max(1, len(width_choices) // 2)
            width_choices = width_choices[: cutoff + 1]
        inner_w = random.choice(width_choices)
        inner_h = random.randint(max(INNER_MIN, max_ph), INNER_MAX)
        try:
            packed = _pack_patterns(patterns, inner_h, inner_w, 1, BG)
        except AssertionError:
            continue
        used_h, used_w = _used_bbox_dims(packed, BG)
        if inner_h > used_h + 2 or inner_w > used_w + 2:
            continue
        if count > 1 and inner_w >= total_pw and random.random() < 0.6:
            continue
        return {
            "patterns": patterns,
            "inner_h": inner_h,
            "inner_w": inner_w,
        }
    return None


def _rect_far_enough(
    r0: int,
    c0: int,
    h: int,
    w: int,
    placed: list[tuple[int, int, int, int]],
    *,
    pad: int = 1,
) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for pr0, pc0, ph, pw in placed:
        pr1 = pr0 + ph - 1
        pc1 = pc0 + pw - 1
        if not (r1 + pad < pr0 or pr1 + pad < r0 or c1 + pad < pc0 or pc1 + pad < c0):
            return False
    return True


def _place_rooms(
    rooms: list[dict[str, object]], height: int, width: int, edge: str
) -> list[tuple[int, int]] | None:
    top_pad = SHELF_THICKNESS + 1 if edge == "top" else 1
    bottom_pad = SHELF_THICKNESS + 1 if edge == "bottom" else 1
    left_pad = SHELF_THICKNESS + 1 if edge == "left" else 1
    right_pad = SHELF_THICKNESS + 1 if edge == "right" else 1

    order = sorted(
        range(len(rooms)),
        key=lambda i: int(rooms[i]["outer_h"]) * int(rooms[i]["outer_w"]),
        reverse=True,
    )
    for _ in range(80):
        placed_boxes: list[tuple[int, int, int, int]] = []
        locations: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in order:
            outer_h = int(rooms[idx]["outer_h"])
            outer_w = int(rooms[idx]["outer_w"])
            min_r = top_pad
            max_r = height - bottom_pad - outer_h
            min_c = left_pad
            max_c = width - right_pad - outer_w
            if min_r > max_r or min_c > max_c:
                ok = False
                break
            candidates = [
                (r, c)
                for r in range(min_r, max_r + 1)
                for c in range(min_c, max_c + 1)
            ]
            random.shuffle(candidates)
            placed = None
            for r0, c0 in candidates:
                if _rect_far_enough(r0, c0, outer_h, outer_w, placed_boxes, pad=1):
                    placed = (r0, c0)
                    break
            if placed is None:
                ok = False
                break
            locations[idx] = placed
            placed_boxes.append((placed[0], placed[1], outer_h, outer_w))
        if ok:
            return [locations[i] for i in range(len(rooms))]
    return None


def _marker_cells_for_room(r0: int, c0: int, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    cells: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _choose_marker_positions(
    r0: int, c0: int, outer_h: int, outer_w: int, count: int
) -> list[tuple[int, int]] | None:
    cells = _marker_cells_for_room(r0, c0, outer_h, outer_w)
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    if len(cells) >= count:
        return random.sample(cells, count)
    return None


def _draw_room(
    grid: np.ndarray,
    room: dict[str, object],
    top: int,
    left: int,
    wall_color: int,
    marker_color: int,
) -> list[tuple[int, int]] | None:
    inner_h = int(room["inner_h"])
    inner_w = int(room["inner_w"])
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top : top + outer_h, left : left + outer_w] = wall_color
    grid[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w] = BG

    markers = _choose_marker_positions(top, left, outer_h, outer_w, int(room["count"]))
    if markers is None:
        return None
    for r, c in markers:
        grid[r, c] = marker_color
    room["top"] = top
    room["left"] = left
    room["outer_h"] = outer_h
    room["outer_w"] = outer_w
    room["marker_positions"] = markers
    return markers


def _place_shelf(
    grid: np.ndarray,
    patterns: list[np.ndarray],
    shelf_color: int,
    edge: str,
) -> list[tuple[int, int, np.ndarray]] | None:
    h, w = map(int, grid.shape)
    horizontal = edge in ("top", "bottom")
    major = w if horizontal else h
    extents = [int(mask.shape[1] if horizontal else mask.shape[0]) for mask in patterns]
    gaps = [random.randint(1, 2) for _ in range(max(0, len(patterns) - 1))]
    required = 2 + sum(extents) + sum(gaps)
    if required > major:
        return None

    lead_extra = random.randint(0, major - required)
    cursor = 1 + lead_extra
    placed: list[tuple[int, int, np.ndarray]] = []
    for i, mask in enumerate(patterns):
        ph, pw = map(int, mask.shape)
        if horizontal:
            row = 0 if edge == "top" else h - ph
            col = cursor
        else:
            row = cursor
            col = 0 if edge == "left" else w - pw
        window = grid[row : row + ph, col : col + pw]
        if np.any(window[mask] != BG):
            return None
        window[mask] = shelf_color
        placed.append((row, col, mask))
        cursor += extents[i]
        if i < len(gaps):
            cursor += gaps[i]
    return placed


def _build_expected(inp: np.ndarray, rooms: list[dict[str, object]], order: list[int]) -> np.ndarray:
    out = inp.copy()
    h, w = map(int, inp.shape)
    for coords in _components(inp != BG, connectivity=4):
        if _touches_edge(coords, h, w):
            out[coords[:, 0], coords[:, 1]] = BG

    for idx in order:
        room = rooms[idx]
        packed = _pack_patterns(
            room["patterns"],
            int(room["inner_h"]),
            int(room["inner_w"]),
            int(room["marker_color"]),
            BG,
        )
        r0 = int(room["top"]) + 1
        c0 = int(room["left"]) + 1
        out[r0 : r0 + int(room["inner_h"]), c0 : c0 + int(room["inner_w"])] = packed
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rooms = random.randint(*ROOM_COUNT_RANGE)
    counts = _sample_counts(n_rooms)

    colors = list(range(1, 10))
    wall_color, shelf_color = random.sample(colors, 2)
    marker_colors = random.sample([c for c in colors if c not in (wall_color, shelf_color)], n_rooms)

    rooms: list[dict[str, object]] = []
    for count, marker_color in zip(counts, marker_colors):
        spec = _sample_room_patterns(count)
        if spec is None:
            return None
        spec["count"] = count
        spec["marker_color"] = marker_color
        spec["outer_h"] = int(spec["inner_h"]) + 2
        spec["outer_w"] = int(spec["inner_w"]) + 2
        rooms.append(spec)

    edge = random.choice(["top", "bottom", "left", "right"])
    horizontal = edge in ("top", "bottom")
    shelf_sequence = [mask for room in rooms for mask in room["patterns"]]
    extents = [int(mask.shape[1] if horizontal else mask.shape[0]) for mask in shelf_sequence]
    required_major = 2 + sum(extents) + (len(extents) - 1)

    min_h = GRID_MIN
    min_w = GRID_MIN
    if horizontal:
        min_w = max(min_w, required_major)
        min_h = max(min_h, SHELF_THICKNESS + max(int(room["outer_h"]) for room in rooms) + 4)
    else:
        min_h = max(min_h, required_major)
        min_w = max(min_w, SHELF_THICKNESS + max(int(room["outer_w"]) for room in rooms) + 4)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = random.randint(min_h, GRID_MAX)
    width = random.randint(min_w, GRID_MAX)
    locations = _place_rooms(rooms, height, width, edge)
    if locations is None:
        return None

    inp = np.full((height, width), BG, dtype=int)
    for room, (top, left) in zip(rooms, locations):
        markers = _draw_room(inp, room, top, left, wall_color, int(room["marker_color"]))
        if markers is None:
            return None

    room_order = sorted(
        range(n_rooms),
        key=lambda idx: min(
            int(r) * width + int(c)
            for r, c in rooms[idx]["marker_positions"]
        ),
    )
    ordered_shelf_patterns = [mask for idx in room_order for mask in rooms[idx]["patterns"]]
    if _place_shelf(inp, ordered_shelf_patterns, shelf_color, edge) is None:
        return None

    expected = _build_expected(inp, rooms, room_order)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
