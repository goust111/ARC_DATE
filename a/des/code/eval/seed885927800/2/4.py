from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BG = 0
SEP = 8
DOOR_COLORS = (1, 2, 3, 4)
PATTERN_COLORS = (5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    return int(colors[np.argmax(counts)])


def _infer_separator_color(g: np.ndarray, bg: int) -> int:
    h, w = map(int, g.shape)
    best_color = None
    best_score = None
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        row_hits = sum(int(np.count_nonzero(g[r] == color) >= max(2, (w + 1) // 2)) for r in range(h))
        col_hits = sum(int(np.count_nonzero(g[:, c] == color) >= max(2, (h + 1) // 2)) for c in range(w))
        score = (row_hits > 0 and col_hits > 0, row_hits + col_hits, int(np.count_nonzero(g == color)))
        if best_score is None or score > best_score:
            best_color = color
            best_score = score
    assert best_color is not None, "failed to infer separator color"
    return int(best_color)


def _infer_separator_lines(g: np.ndarray, sep: int) -> tuple[list[int], list[int]]:
    h, w = map(int, g.shape)
    row_threshold = max(2, (w + 1) // 2)
    col_threshold = max(2, (h + 1) // 2)
    sep_rows = [int(r) for r in range(h) if int(np.count_nonzero(g[r] == sep)) >= row_threshold]
    sep_cols = [int(c) for c in range(w) if int(np.count_nonzero(g[:, c] == sep)) >= col_threshold]
    assert sep_rows and sep_cols, "expected both horizontal and vertical separator lines"
    return sep_rows, sep_cols


def _build_rooms(
    h: int,
    w: int,
    sep_rows: list[int],
    sep_cols: list[int],
) -> list[dict]:
    row_bounds = [-1, *sorted(sep_rows), h]
    col_bounds = [-1, *sorted(sep_cols), w]
    rooms: list[dict] = []
    room_id = 0
    for i in range(len(row_bounds) - 1):
        r0 = row_bounds[i] + 1
        r1 = row_bounds[i + 1] - 1
        if r0 > r1:
            continue
        for j in range(len(col_bounds) - 1):
            c0 = col_bounds[j] + 1
            c1 = col_bounds[j + 1] - 1
            if c0 > c1:
                continue
            rooms.append(
                {
                    "id": room_id,
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                    "top_wall": row_bounds[i] if 0 <= row_bounds[i] < h else None,
                    "bottom_wall": row_bounds[i + 1] if 0 <= row_bounds[i + 1] < h else None,
                    "left_wall": col_bounds[j] if 0 <= col_bounds[j] < w else None,
                    "right_wall": col_bounds[j + 1] if 0 <= col_bounds[j + 1] < w else None,
                    "row_index": i,
                    "col_index": j,
                }
            )
            room_id += 1
    assert rooms, "expected at least one room"
    return rooms


def _door_candidates(rooms: list[dict], r: int, c: int) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    for room in rooms:
        if room["top_wall"] is not None and r == room["top_wall"] and room["c0"] <= c <= room["c1"]:
            candidates.append((int(room["id"]), "horizontal"))
        if room["bottom_wall"] is not None and r == room["bottom_wall"] and room["c0"] <= c <= room["c1"]:
            candidates.append((int(room["id"]), "horizontal"))
        if room["left_wall"] is not None and c == room["left_wall"] and room["r0"] <= r <= room["r1"]:
            candidates.append((int(room["id"]), "vertical"))
        if room["right_wall"] is not None and c == room["right_wall"] and room["r0"] <= r <= room["r1"]:
            candidates.append((int(room["id"]), "vertical"))
    return candidates


def _collect_doors(g: np.ndarray, sep: int, rooms: list[dict], sep_rows: list[int], sep_cols: list[int]) -> list[dict]:
    line_mask = np.zeros(g.shape, dtype=bool)
    line_mask[sep_rows, :] = True
    line_mask[:, sep_cols] = True
    door_coords = np.argwhere(line_mask & (g != sep))
    doors: list[dict] = []
    for r, c in door_coords:
        candidates = _door_candidates(rooms, int(r), int(c))
        assert candidates, "door must belong to at least one room wall"
        room_ids = sorted({room_id for room_id, _ in candidates})
        orientations = {orientation for _room_id, orientation in candidates}
        assert len(orientations) == 1, "door orientation should be well-defined"
        doors.append(
            {
                "color": int(g[int(r), int(c)]),
                "r": int(r),
                "c": int(c),
                "orientation": orientations.pop(),
                "candidates": room_ids,
            }
        )
    return doors


def _assign_doors(doors: list[dict]) -> dict[int, int]:
    order = sorted(range(len(doors)), key=lambda idx: len(doors[idx]["candidates"]))
    used_rooms: set[int] = set()
    assignment: dict[int, int] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        door_idx = order[pos]
        for room_id in doors[door_idx]["candidates"]:
            if room_id in used_rooms:
                continue
            used_rooms.add(int(room_id))
            assignment[int(door_idx)] = int(room_id)
            if dfs(pos + 1):
                return True
            used_rooms.remove(int(room_id))
            del assignment[int(door_idx)]
        return False

    assert dfs(0), "failed to assign doors to rooms"
    return assignment


def _extract_patch(g: np.ndarray, room: dict, bg: int) -> np.ndarray:
    interior = g[room["r0"] : room["r1"] + 1, room["c0"] : room["c1"] + 1]
    mask = interior != bg
    if not bool(mask.any()):
        return np.zeros((0, 0), dtype=int)
    rows, cols = np.where(mask)
    return interior[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _mirror_patch(patch: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "horizontal":
        return np.flipud(patch)
    if orientation == "vertical":
        return np.fliplr(patch)
    raise AssertionError("unknown door orientation")


def _paste_centered(out: np.ndarray, room: dict, patch: np.ndarray, bg: int) -> None:
    if patch.size == 0:
        return
    ph, pw = map(int, patch.shape)
    assert ph <= room["height"] and pw <= room["width"], "pattern must fit inside target room"
    top = int(room["r0"] + (room["height"] - ph) // 2)
    left = int(room["c0"] + (room["width"] - pw) // 2)
    target = out[top : top + ph, left : left + pw]
    mask = patch != bg
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored wall doors, swap the room patterns, and mirror each pattern
    according to the orientation of the door it exits through.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = map(int, g.shape)

    bg = _most_common_color(g)
    sep = _infer_separator_color(g, bg)
    sep_rows, sep_cols = _infer_separator_lines(g, sep)
    rooms = _build_rooms(h, w, sep_rows, sep_cols)
    doors = _collect_doors(g, sep, rooms, sep_rows, sep_cols)
    assignment = _assign_doors(doors)

    room_doors: dict[int, dict] = {}
    by_color: defaultdict[int, list[int]] = defaultdict(list)
    for door_idx, room_id in assignment.items():
        door = doors[int(door_idx)]
        room_doors[int(room_id)] = door
        by_color[int(door["color"])].append(int(room_id))

    room_by_id = {int(room["id"]): room for room in rooms}
    out = g.copy()

    for room_ids in by_color.values():
        assert len(room_ids) == 2, "each door color should pair exactly two rooms"
        for room_id in room_ids:
            room = room_by_id[int(room_id)]
            out[room["r0"] : room["r1"] + 1, room["c0"] : room["c1"] + 1] = bg

    for room_ids in by_color.values():
        room_a_id, room_b_id = map(int, room_ids)
        room_a = room_by_id[room_a_id]
        room_b = room_by_id[room_b_id]
        patch_a = _extract_patch(g, room_a, bg)
        patch_b = _extract_patch(g, room_b, bg)
        moved_a = _mirror_patch(patch_a, room_doors[room_a_id]["orientation"])
        moved_b = _mirror_patch(patch_b, room_doors[room_b_id]["orientation"])
        _paste_centered(out, room_b, moved_a, bg)
        _paste_centered(out, room_a, moved_b, bg)

    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _build_random_mask(h: int, w: int) -> np.ndarray | None:
    min_cells = max(4, (h + w) // 2 + 1)
    max_cells = min(h * w - 1, min_cells + 3)
    if min_cells > max_cells:
        return None
    for _ in range(200):
        target = random.randint(min_cells, max_cells)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], h, w))
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nxt in _neighbors4(cell[0], cell[1], h, w):
                if nxt not in cells:
                    frontier.add(nxt)
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != h - 1 or cols.min() != 0 or cols.max() != w - 1:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _pattern_size_options(max_h: int, max_w: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for h in range(2, min(4, max_h) + 1):
        for w in range(2, min(4, max_w) + 1):
            if h * w >= 6:
                options.append((h, w))
    return options


def _choose_pattern_patch(room: dict, partner: dict | None) -> np.ndarray | None:
    max_h = room["height"] - 2
    max_w = room["width"] - 2
    if partner is not None:
        max_h = min(max_h, partner["height"] - 2)
        max_w = min(max_w, partner["width"] - 2)
    options = _pattern_size_options(int(max_h), int(max_w))
    if not options:
        return None
    for _ in range(100):
        ph, pw = random.choice(options)
        mask = _build_random_mask(int(ph), int(pw))
        if mask is None:
            continue
        patch = np.full((ph, pw), BG, dtype=int)
        patch[mask] = random.choice(PATTERN_COLORS)
        return patch
    return None


def _room_boundaries(sizes: list[int]) -> list[int]:
    bounds = [0]
    pos = 0
    for size in sizes:
        pos += int(size) + 1
        bounds.append(int(pos))
    return bounds


def _build_room_specs(row_sizes: list[int], col_sizes: list[int]) -> list[dict]:
    row_bounds = _room_boundaries(row_sizes)
    col_bounds = _room_boundaries(col_sizes)
    rooms: list[dict] = []
    room_id = 0
    for ri, height in enumerate(row_sizes):
        for ci, width in enumerate(col_sizes):
            rooms.append(
                {
                    "id": room_id,
                    "row_index": int(ri),
                    "col_index": int(ci),
                    "r0": int(row_bounds[ri] + 1),
                    "r1": int(row_bounds[ri + 1] - 1),
                    "c0": int(col_bounds[ci] + 1),
                    "c1": int(col_bounds[ci + 1] - 1),
                    "height": int(height),
                    "width": int(width),
                    "top_wall": int(row_bounds[ri]),
                    "bottom_wall": int(row_bounds[ri + 1]),
                    "left_wall": int(col_bounds[ci]),
                    "right_wall": int(col_bounds[ci + 1]),
                }
            )
            room_id += 1
    return rooms


def _outer_sides(room: dict, n_rows: int, n_cols: int) -> list[str]:
    sides: list[str] = []
    if room["row_index"] == 0:
        sides.append("top")
    if room["row_index"] == n_rows - 1:
        sides.append("bottom")
    if room["col_index"] == 0:
        sides.append("left")
    if room["col_index"] == n_cols - 1:
        sides.append("right")
    return sides


def _choose_side(sides: list[str]) -> str:
    horizontal = [side for side in sides if side in ("top", "bottom")]
    vertical = [side for side in sides if side in ("left", "right")]
    if horizontal and vertical:
        sides = horizontal if random.random() < 0.5 else vertical
    return random.choice(sides)


def _door_position(room: dict, side: str) -> tuple[int, int]:
    if side == "top":
        return int(room["top_wall"]), random.randint(int(room["c0"]), int(room["c1"]))
    if side == "bottom":
        return int(room["bottom_wall"]), random.randint(int(room["c0"]), int(room["c1"]))
    if side == "left":
        return random.randint(int(room["r0"]), int(room["r1"])), int(room["left_wall"])
    if side == "right":
        return random.randint(int(room["r0"]), int(room["r1"])), int(room["right_wall"])
    raise AssertionError("unknown side")


def _paste_patch(grid: np.ndarray, room: dict, patch: np.ndarray) -> tuple[int, int]:
    ph, pw = map(int, patch.shape)
    top = random.randint(1, int(room["height"] - ph - 1))
    left = random.randint(1, int(room["width"] - pw - 1))
    r0 = int(room["r0"] + top)
    c0 = int(room["c0"] + left)
    target = grid[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != BG
    target[mask] = patch[mask]
    return r0, c0


def _render_case(
    rooms: list[dict],
    row_sizes: list[int],
    col_sizes: list[int],
    pair_by_color: dict[int, tuple[int, int]],
    room_doors: dict[int, dict],
    room_patches: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    grid_h = sum(row_sizes) + len(row_sizes) + 1
    grid_w = sum(col_sizes) + len(col_sizes) + 1
    inp = np.full((grid_h, grid_w), BG, dtype=int)

    row_bounds = _room_boundaries(row_sizes)
    col_bounds = _room_boundaries(col_sizes)
    inp[row_bounds, :] = SEP
    inp[:, col_bounds] = SEP

    for room in rooms:
        _paste_patch(inp, room, room_patches[int(room["id"])])

    for room_id, door in room_doors.items():
        r, c = _door_position(next(room for room in rooms if room["id"] == room_id), door["side"])
        inp[int(r), int(c)] = int(door["color"])

    out = inp.copy()
    room_by_id = {int(room["id"]): room for room in rooms}
    for room_a_id, room_b_id in pair_by_color.values():
        for room_id in (room_a_id, room_b_id):
            room = room_by_id[int(room_id)]
            out[room["r0"] : room["r1"] + 1, room["c0"] : room["c1"] + 1] = BG

    for room_a_id, room_b_id in pair_by_color.values():
        room_a = room_by_id[int(room_a_id)]
        room_b = room_by_id[int(room_b_id)]
        moved_a = _mirror_patch(room_patches[int(room_a_id)], room_doors[int(room_a_id)]["orientation"])
        moved_b = _mirror_patch(room_patches[int(room_b_id)], room_doors[int(room_b_id)]["orientation"])
        _paste_centered(out, room_b, moved_a, BG)
        _paste_centered(out, room_a, moved_b, BG)

    return inp, out


def _generate_case() -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    row_sizes = [random.randint(5, 7) for _ in range(n_rows)]
    col_sizes = [random.randint(5, 7) for _ in range(n_cols)]
    rooms = _build_room_specs(row_sizes, col_sizes)

    border_rooms = [room for room in rooms if _outer_sides(room, n_rows, n_cols)]
    max_pairs = min(3, len(DOOR_COLORS), len(border_rooms) // 2)
    if max_pairs < 1:
        return None
    n_pairs = random.randint(1, max_pairs)

    chosen_rooms = random.sample([int(room["id"]) for room in border_rooms], 2 * n_pairs)
    random.shuffle(chosen_rooms)
    partners: dict[int, int] = {}
    pair_by_color: dict[int, tuple[int, int]] = {}
    room_doors: dict[int, dict] = {}

    for color in random.sample(DOOR_COLORS, n_pairs):
        room_a_id = int(chosen_rooms.pop())
        room_b_id = int(chosen_rooms.pop())
        partners[room_a_id] = room_b_id
        partners[room_b_id] = room_a_id
        pair_by_color[int(color)] = (room_a_id, room_b_id)

    room_by_id = {int(room["id"]): room for room in rooms}
    for color, (room_a_id, room_b_id) in pair_by_color.items():
        for room_id in (room_a_id, room_b_id):
            room = room_by_id[int(room_id)]
            side = _choose_side(_outer_sides(room, n_rows, n_cols))
            room_doors[int(room_id)] = {
                "color": int(color),
                "side": side,
                "orientation": "horizontal" if side in ("top", "bottom") else "vertical",
            }

    room_patches: dict[int, np.ndarray] = {}
    for room in rooms:
        partner = room_by_id[partners[int(room["id"])]] if int(room["id"]) in partners else None
        patch = _choose_pattern_patch(room, partner)
        if patch is None:
            return None
        room_patches[int(room["id"])] = patch

    inp, expected = _render_case(rooms, row_sizes, col_sizes, pair_by_color, room_doors, room_patches)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
