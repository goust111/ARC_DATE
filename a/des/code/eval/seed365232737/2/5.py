import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
WALL_COLORS = (1, 2, 3, 4, 6, 7, 8)

PATTERN_LIBRARY = (
    np.array([[MAROON, BLACK, BLACK], [BLACK, MAROON, BLACK], [BLACK, BLACK, MAROON]], dtype=int),
    np.array([[BLACK, BLACK, MAROON], [BLACK, MAROON, BLACK], [MAROON, BLACK, BLACK]], dtype=int),
    np.array([[MAROON, BLACK, MAROON], [BLACK, MAROON, BLACK], [MAROON, BLACK, MAROON]], dtype=int),
    np.array([[BLACK, MAROON, BLACK], [MAROON, MAROON, MAROON], [BLACK, MAROON, BLACK]], dtype=int),
    np.array([[MAROON, MAROON, MAROON], [MAROON, BLACK, MAROON], [MAROON, MAROON, MAROON]], dtype=int),
    np.array([[MAROON, BLACK, MAROON], [MAROON, BLACK, MAROON], [MAROON, BLACK, MAROON]], dtype=int),
    np.array([[MAROON, MAROON, MAROON], [BLACK, BLACK, BLACK], [MAROON, MAROON, MAROON]], dtype=int),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    checks = [
        ("tl", (0, 0)),
        ("tr", (0, w - 1)),
        ("bl", (h - 1, 0)),
        ("br", (h - 1, w - 1)),
    ]
    for corner, (r, c) in checks:
        if int(grid[r, c]) != GREY:
            continue
        if corner == "tl":
            c1 = 0
            while c1 + 1 < w and int(grid[0, c1 + 1]) == GREY:
                c1 += 1
            r1 = 0
            while r1 + 1 < h and int(grid[r1 + 1, 0]) == GREY:
                r1 += 1
            if c1 >= 2 and r1 >= 2 and np.all(grid[0, : c1 + 1] == GREY) and np.all(grid[: r1 + 1, 0] == GREY):
                if np.all(grid[r1, : c1 + 1] == GREY) and np.all(grid[: r1 + 1, c1] == GREY):
                    return (0, 0, r1, c1)
        if corner == "tr":
            c0 = w - 1
            while c0 - 1 >= 0 and int(grid[0, c0 - 1]) == GREY:
                c0 -= 1
            r1 = 0
            while r1 + 1 < h and int(grid[r1 + 1, w - 1]) == GREY:
                r1 += 1
            if c0 <= w - 3 and r1 >= 2 and np.all(grid[0, c0:] == GREY) and np.all(grid[: r1 + 1, w - 1] == GREY):
                if np.all(grid[r1, c0:] == GREY) and np.all(grid[: r1 + 1, c0] == GREY):
                    return (0, c0, r1, w - 1)
        if corner == "bl":
            c1 = 0
            while c1 + 1 < w and int(grid[h - 1, c1 + 1]) == GREY:
                c1 += 1
            r0 = h - 1
            while r0 - 1 >= 0 and int(grid[r0 - 1, 0]) == GREY:
                r0 -= 1
            if c1 >= 2 and r0 <= h - 3 and np.all(grid[h - 1, : c1 + 1] == GREY) and np.all(grid[r0:, 0] == GREY):
                if np.all(grid[r0, : c1 + 1] == GREY) and np.all(grid[r0:, c1] == GREY):
                    return (r0, 0, h - 1, c1)
        if corner == "br":
            c0 = w - 1
            while c0 - 1 >= 0 and int(grid[h - 1, c0 - 1]) == GREY:
                c0 -= 1
            r0 = h - 1
            while r0 - 1 >= 0 and int(grid[r0 - 1, w - 1]) == GREY:
                r0 -= 1
            if c0 <= w - 3 and r0 <= h - 3 and np.all(grid[h - 1, c0:] == GREY) and np.all(grid[r0:, w - 1] == GREY):
                if np.all(grid[r0, c0:] == GREY) and np.all(grid[r0:, c0] == GREY):
                    return (r0, c0, h - 1, w - 1)
    raise AssertionError("legend frame not found")


def _legend_mapping(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    mapping: dict[int, np.ndarray] = {}
    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            val = int(grid[rr, cc])
            if val in (BLACK, GREY, MAROON):
                continue
            top = rr - 1
            left = cc + 1
            if top < r0 + 1 or left + 2 > c1 - 1:
                continue
            patch = grid[top : top + 3, left : left + 3]
            if patch.shape != (3, 3):
                continue
            if not set(np.unique(patch).tolist()).issubset({BLACK, MAROON}):
                continue
            mapping[val] = patch.copy()
    if not mapping:
        raise AssertionError("legend entries not found")
    return mapping


def _room_info(grid: np.ndarray, legend_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    passable = ((grid == BLACK) | (grid == GREY)) & (~legend_mask)
    rooms: list[dict] = []
    for pts in _components(passable):
        rows = [r for r, _c in pts]
        cols = [c for _r, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        coord_set = set(pts)
        if len(coord_set) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        has_marker = any(int(grid[r, c]) == GREY for r, c in pts)
        candidates: list[tuple[str, int, int]] = []
        if r0 > 0:
            line = grid[r0 - 1, c0 : c1 + 1]
            if line.size > 0 and int(line[0]) not in (BLACK, GREY, MAROON) and np.all(line == line[0]):
                candidates.append(("top", line.size, int(line[0])))
        if c1 < w - 1:
            line = grid[r0 : r1 + 1, c1 + 1]
            if line.size > 0 and int(line[0]) not in (BLACK, GREY, MAROON) and np.all(line == line[0]):
                candidates.append(("right", line.size, int(line[0])))
        if r1 < h - 1:
            line = grid[r1 + 1, c0 : c1 + 1]
            if line.size > 0 and int(line[0]) not in (BLACK, GREY, MAROON) and np.all(line == line[0]):
                candidates.append(("bottom", line.size, int(line[0])))
        if c0 > 0:
            line = grid[r0 : r1 + 1, c0 - 1]
            if line.size > 0 and int(line[0]) not in (BLACK, GREY, MAROON) and np.all(line == line[0]):
                candidates.append(("left", line.size, int(line[0])))

        control_color = None
        best_len = -1
        for _side, side_len, color in candidates:
            if side_len > best_len:
                best_len = side_len
                control_color = color

        tie_count = 0
        if best_len >= 0:
            tie_count = sum(side_len == best_len for _side, side_len, _color in candidates)

        rooms.append(
            {
                "pts": pts,
                "bbox": (r0, c0, r1, c1),
                "has_marker": has_marker,
                "control_color": control_color,
                "tie": tie_count >= 2,
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
        )
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_bbox = _find_legend_bbox(g)
    legend_mask = np.zeros_like(g, dtype=bool)
    r0, c0, r1, c1 = legend_bbox
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    mapping = _legend_mapping(g, legend_bbox)

    out = g.copy()
    for room in _room_info(g, legend_mask):
        if not room["has_marker"] or room["control_color"] is None:
            continue
        color = int(room["control_color"])
        if color not in mapping:
            continue
        pattern = mapping[color]
        rr0, cc0, _rr1, _cc1 = room["bbox"]
        for r, c in room["pts"]:
            out[r, c] = int(pattern[(r - rr0) % 3, (c - cc0) % 3])
    return out


def _make_legend(entries: list[tuple[int, np.ndarray]]) -> np.ndarray:
    inner_h = 3 * len(entries) + max(0, len(entries) - 1)
    inner_w = 4
    patch = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    row = 1
    for idx, (color, pattern) in enumerate(entries):
        patch[row + 1, 1] = int(color)
        patch[row : row + 3, 2:5] = pattern
        row += 4 if idx < len(entries) - 1 else 3
    return patch


def _draw_wall(grid: np.ndarray, rect: tuple[int, int, int, int], orientation: str, pos: int, color: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = rect
    if orientation == "v":
        grid[r0 : r1 + 1, pos] = int(color)
        left = (r0, c0, r1, pos - 1)
        right = (r0, pos + 1, r1, c1)
        return left, right
    grid[pos, c0 : c1 + 1] = int(color)
    top = (r0, c0, pos - 1, c1)
    bottom = (pos + 1, c0, r1, c1)
    return top, bottom


def _rect_valid(rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r1 and c0 <= c1


def _rect_dims(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _split_candidates(rect: tuple[int, int, int, int], min_side: int) -> list[str]:
    h, w = _rect_dims(rect)
    choices: list[str] = []
    if w >= 2 * min_side + 1:
        choices.append("v")
    if h >= 2 * min_side + 1:
        choices.append("h")
    return choices


def _choose_split_pos(rect: tuple[int, int, int, int], orientation: str, min_side: int) -> int:
    r0, c0, r1, c1 = rect
    if orientation == "v":
        options = list(range(c0 + min_side, c1 - min_side + 1))
    else:
        options = list(range(r0 + min_side, r1 - min_side + 1))
    if len(options) == 1:
        return options[0]
    if random.random() < 0.45:
        mid = (options[0] + options[-1]) / 2.0
        return min(options, key=lambda x: (abs(x - mid), random.random()))
    return random.choice(options)


def _wall_line_for_room(room: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = room
    if side == "top" and r0 > 0:
        return (r0 - 1, c0, r0 - 1, c1)
    if side == "right":
        return (r0, c1 + 1, r1, c1 + 1)
    if side == "bottom":
        return (r1 + 1, c0, r1 + 1, c1)
    if side == "left" and c0 > 0:
        return (r0, c0 - 1, r1, c0 - 1)
    return None


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _corner_setup(h: int, w: int, legend_h: int, legend_w: int, corner: str) -> tuple[tuple[int, int, int, int], int, int, list[tuple[int, int, int, int]]]:
    if corner == "tl":
        legend_rect = (0, 0, legend_h - 1, legend_w - 1)
        hr, vc = legend_h, legend_w
        rects = [
            (0, vc + 1, hr - 1, w - 1),
            (hr + 1, 0, h - 1, vc - 1),
            (hr + 1, vc + 1, h - 1, w - 1),
        ]
    elif corner == "tr":
        legend_rect = (0, w - legend_w, legend_h - 1, w - 1)
        hr, vc = legend_h, w - legend_w - 1
        rects = [
            (0, 0, hr - 1, vc - 1),
            (hr + 1, 0, h - 1, vc - 1),
            (hr + 1, vc + 1, h - 1, w - 1),
        ]
    elif corner == "bl":
        legend_rect = (h - legend_h, 0, h - 1, legend_w - 1)
        hr, vc = h - legend_h - 1, legend_w
        rects = [
            (0, 0, hr - 1, vc - 1),
            (0, vc + 1, hr - 1, w - 1),
            (hr + 1, vc + 1, h - 1, w - 1),
        ]
    else:
        legend_rect = (h - legend_h, w - legend_w, h - 1, w - 1)
        hr, vc = h - legend_h - 1, w - legend_w - 1
        rects = [
            (0, 0, hr - 1, vc - 1),
            (0, vc + 1, hr - 1, w - 1),
            (hr + 1, 0, h - 1, vc - 1),
        ]
    rects = [rect for rect in rects if _rect_valid(rect)]
    return legend_rect, hr, vc, rects


def _make_candidate() -> np.ndarray:
    n_entries = random.randint(2, 4)
    colors = random.sample(WALL_COLORS, n_entries)
    patterns = random.sample(PATTERN_LIBRARY, n_entries)
    legend_entries = [(colors[i], patterns[i].copy()) for i in range(n_entries)]
    legend = _make_legend(legend_entries)
    legend_h, legend_w = legend.shape

    min_h = max(legend_h + 8, 18)
    min_w = max(legend_w + 10, 18)
    h = random.randint(min_h, max(min_h, 30))
    w = random.randint(min_w, max(min_w, 30))

    corner = random.choice(("tl", "tr", "bl", "br"))
    grid = np.full((h, w), BLACK, dtype=int)

    legend_rect, hr, vc, rooms = _corner_setup(h, w, legend_h, legend_w, corner)
    isolate_color = random.choice(colors)
    grid[hr, :] = isolate_color
    grid[:, vc] = isolate_color

    additional_splits = random.randint(max(3, n_entries - 1), max(5, n_entries + 2))
    color_order = [c for c in colors if c != isolate_color]
    random.shuffle(color_order)
    while len(color_order) < additional_splits:
        color_order.append(random.choice(colors))

    min_side = 2
    split_idx = 0
    attempts = 0
    while split_idx < additional_splits and attempts < 200:
        attempts += 1
        candidates = [i for i, rect in enumerate(rooms) if _split_candidates(rect, min_side)]
        if not candidates:
            break
        idx = random.choice(candidates)
        rect = rooms.pop(idx)
        choices = _split_candidates(rect, min_side)
        h0, w0 = _rect_dims(rect)
        if len(choices) == 2:
            if h0 > w0 and random.random() < 0.65:
                orient = "h"
            elif w0 > h0 and random.random() < 0.65:
                orient = "v"
            else:
                orient = random.choice(choices)
        else:
            orient = choices[0]
        pos = _choose_split_pos(rect, orient, min_side)
        color = color_order[split_idx]
        a, b = _draw_wall(grid, rect, orient, pos, color)
        rooms.extend([a, b])
        split_idx += 1

    r0, c0, r1, c1 = legend_rect
    _paste(grid, legend, r0, c0)

    usable_rooms = [rect for rect in rooms if _rect_valid(rect)]
    if len(usable_rooms) < 4:
        raise AssertionError("too few rooms")

    mark_count = random.randint(1, len(usable_rooms) - 1)
    marked_indices = set(random.sample(range(len(usable_rooms)), mark_count))
    for idx, rect in enumerate(usable_rooms):
        rr0, cc0, rr1, cc1 = rect
        area = (rr1 - rr0 + 1) * (cc1 - cc0 + 1)
        if idx not in marked_indices:
            continue
        n_marks = random.randint(1, min(4, area))
        cells = [(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]
        for r, c in random.sample(cells, n_marks):
            grid[r, c] = GREY

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        require_tie = random.random() < 0.5
        grid = _make_candidate()

        legend_bbox = _find_legend_bbox(grid)
        legend_mask = np.zeros_like(grid, dtype=bool)
        r0, c0, r1, c1 = legend_bbox
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        mapping = _legend_mapping(grid, legend_bbox)
        rooms = _room_info(grid, legend_mask)

        marked = [room for room in rooms if room["has_marker"]]
        unmarked = [room for room in rooms if not room["has_marker"]]
        if len(mapping) < 2 or not marked or not unmarked:
            continue

        tie_exists = any(room["tie"] for room in rooms)
        if require_tie and not tie_exists:
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue

        untouched_unmarked = False
        for room in unmarked:
            if all(int(out[r, c]) == BLACK for r, c in room["pts"]):
                untouched_unmarked = True
                break
        if not untouched_unmarked:
            continue

        used_patterns = 0
        for room in marked:
            ctrl = room["control_color"]
            if ctrl is None or ctrl not in mapping:
                continue
            if any(int(out[r, c]) == MAROON for r, c in room["pts"]):
                used_patterns += 1
        if used_patterns == 0:
            continue

        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
