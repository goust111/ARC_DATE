import random
import numpy as np


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _most_common(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _separator_color(grid: np.ndarray) -> int:
    candidates: list[tuple[int, int]] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        full_rows = int(np.all(grid == color, axis=1).sum())
        full_cols = int(np.all(grid == color, axis=0).sum())
        if full_rows > 0 and full_cols > 0:
            candidates.append((full_rows + full_cols, color))
    assert candidates, "expected at least one separator color"
    candidates.sort()
    return int(candidates[-1][1])


def _intervals(separator_indices: np.ndarray, limit: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    prev = -1
    for idx in [*(int(v) for v in separator_indices.tolist()), int(limit)]:
        if prev + 1 <= idx - 1:
            intervals.append((int(prev + 1), int(idx - 1)))
        prev = int(idx)
    assert intervals, "expected at least one room interval"
    return intervals


def _analyze_room(
    grid: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    bg: int,
    sep: int,
) -> dict:
    r0, c0, r1, c1 = map(int, room_bbox)
    region = grid[r0 : r1 + 1, c0 : c1 + 1]

    object_pixels = (region != bg) & (region != sep)
    colors = np.unique(region[object_pixels])
    assert len(colors) == 1, "each room must contain exactly one object color"
    color = int(colors[0])

    color_mask = region == color
    br0, bc0, br1, bc1 = _tight_bbox(color_mask)
    assert br0 > 0 and bc0 > 0, "object must not touch the room boundary"
    assert br1 < region.shape[0] - 1 and bc1 < region.shape[1] - 1, "object must not touch the room boundary"

    tight_mask = color_mask[br0 : br1 + 1, bc0 : bc1 + 1]
    global_bbox = (int(r0 + br0), int(c0 + bc0), int(r0 + br1), int(c0 + bc1))

    if tight_mask.shape[0] >= 3 and tight_mask.shape[1] >= 3:
        if np.array_equal(tight_mask, _border_mask(tight_mask.shape)):
            return {
                "kind": "lock",
                "color": color,
                "bbox": global_bbox,
                "cavity_top": (int(r0 + br0 + 1), int(c0 + bc0 + 1)),
                "cavity_shape": (int(tight_mask.shape[0] - 2), int(tight_mask.shape[1] - 2)),
            }

    coords = np.argwhere(color_mask)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return {
        "kind": "key",
        "color": color,
        "bbox": global_bbox,
        "coords": coords,
        "mask": tight_mask,
    }


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_color(g)
    bg = _most_common(g[g != sep])

    sep_rows = np.flatnonzero(np.all(g == sep, axis=1))
    sep_cols = np.flatnonzero(np.all(g == sep, axis=0))
    assert len(sep_rows) > 0 and len(sep_cols) > 0, "expected both horizontal and vertical separators"

    row_intervals = _intervals(sep_rows, g.shape[0])
    col_intervals = _intervals(sep_cols, g.shape[1])

    rooms = []
    for rr in row_intervals:
        for cc in col_intervals:
            rooms.append(_analyze_room(g, (rr[0], cc[0], rr[1], cc[1]), bg, sep))

    by_color: dict[int, list[dict]] = {}
    for room in rooms:
        by_color.setdefault(int(room["color"]), []).append(room)

    out = g.copy()
    for room in rooms:
        if room["kind"] == "key":
            coords = room["coords"]
            out[coords[:, 0], coords[:, 1]] = bg

    for color, items in by_color.items():
        assert len(items) == 2, "each color must appear in exactly two rooms"
        key_rooms = [item for item in items if item["kind"] == "key"]
        lock_rooms = [item for item in items if item["kind"] == "lock"]
        assert len(key_rooms) == 1 and len(lock_rooms) == 1, "each color must map one key to one lock"

        key = key_rooms[0]
        lock = lock_rooms[0]
        placed = False
        for turns in range(4):
            rotated = np.rot90(key["mask"], int(turns))
            if rotated.shape != lock["cavity_shape"]:
                continue
            top, left = map(int, lock["cavity_top"])
            out[top : top + rotated.shape[0], left : left + rotated.shape[1]][rotated] = int(color)
            placed = True
            break
        assert placed, "every key must match its lock cavity after rotation"

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Match each colored key room to its same-colored lock room and insert the rotated key into the lock cavity.

    Input structure:
    - The background is black, and one-cell-wide grey full rows and full columns partition the canvas into rectangular rooms.
    - Each room contains exactly one monochrome non-grey object that does not touch the room boundary.
    - A key room contains a solid 4-connected shape, while a lock room contains a one-cell-wide rectangular frame with a black rectangular cavity.
    - Each non-grey color appears in exactly two rooms: once as a key and once as a lock.

    Rule:
    - Detect the grey partition lines to recover every room.
    - Classify each room by its colored object: rectangular border means lock, otherwise key.
    - Pair each key with the lock of the same color, rotate the key by 0/90/180/270 degrees until its tight bounding box matches the lock cavity, erase the key from its source room, and draw the rotated key aligned to the cavity's top-left corner.
    - Keep the grey partition lines and every lock frame unchanged.

    Key insight:
    - The transformation requires two separate global inferences: same-color objects across different rooms define the key-lock pairs, and only after pairing can the key's correct rotation be chosen from the cavity size.
    """
    return _solve(grid)


# --------------- generator ---------------

_GEN_BG = 0
_GEN_SEP = 5
_GEN_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_LAYOUTS = {
    2: [(2, 2)],
    3: [(2, 3), (3, 2)],
    4: [(2, 4), (4, 2)],
}
_GEN_MIN_ROOM = 6
_GEN_MAX_ROOM = 12
_GEN_MAX_ATTEMPTS = 300


def _sample_axis_sizes(count: int) -> list[int] | None:
    total_limit = 30 - (count - 1)
    min_total = count * _GEN_MIN_ROOM
    max_total = min(total_limit, count * _GEN_MAX_ROOM)
    if min_total > max_total:
        return None

    total = random.randint(min_total, max_total)
    sizes = [_GEN_MIN_ROOM for _ in range(count)]
    for _ in range(total - min_total):
        options = [idx for idx, size in enumerate(sizes) if size < _GEN_MAX_ROOM]
        if not options:
            return None
        sizes[random.choice(options)] += 1

    if count > 1 and len(set(sizes)) == 1:
        for _ in range(20):
            lo, hi = random.sample(range(count), 2)
            if sizes[lo] > _GEN_MIN_ROOM and sizes[hi] < _GEN_MAX_ROOM:
                sizes[lo] -= 1
                sizes[hi] += 1
                break

    return sizes


def _build_rooms(layout: tuple[int, int]) -> tuple[np.ndarray, list[dict]] | None:
    row_count, col_count = map(int, layout)
    row_sizes = _sample_axis_sizes(row_count)
    col_sizes = _sample_axis_sizes(col_count)
    if row_sizes is None or col_sizes is None:
        return None

    height = sum(row_sizes) + row_count - 1
    width = sum(col_sizes) + col_count - 1
    grid = np.full((height, width), _GEN_BG, dtype=int)

    row_bounds: list[tuple[int, int]] = []
    sep_rows: list[int] = []
    cursor = 0
    for idx, size in enumerate(row_sizes):
        row_bounds.append((int(cursor), int(cursor + size - 1)))
        cursor += size
        if idx < row_count - 1:
            sep_rows.append(int(cursor))
            cursor += 1

    col_bounds: list[tuple[int, int]] = []
    sep_cols: list[int] = []
    cursor = 0
    for idx, size in enumerate(col_sizes):
        col_bounds.append((int(cursor), int(cursor + size - 1)))
        cursor += size
        if idx < col_count - 1:
            sep_cols.append(int(cursor))
            cursor += 1

    if sep_rows:
        grid[sep_rows, :] = _GEN_SEP
    if sep_cols:
        grid[:, sep_cols] = _GEN_SEP

    rooms = []
    for rr in row_bounds:
        for cc in col_bounds:
            rooms.append(
                {
                    "r0": int(rr[0]),
                    "c0": int(cc[0]),
                    "h": int(rr[1] - rr[0] + 1),
                    "w": int(cc[1] - cc[0] + 1),
                }
            )

    return grid, rooms


def _prefer_larger(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    larger = [dims for dims in candidates if dims[0] * dims[1] >= 3]
    return larger if larger else candidates


def _choose_pair_spec(key_room: dict, lock_room: dict) -> dict:
    key_max_h = int(key_room["h"] - 2)
    key_max_w = int(key_room["w"] - 2)
    lock_max_h = int(lock_room["h"] - 4)
    lock_max_w = int(lock_room["w"] - 4)

    same: list[tuple[int, int]] = []
    turned: list[tuple[int, int]] = []
    for cavity_h in range(1, lock_max_h + 1):
        for cavity_w in range(1, lock_max_w + 1):
            if cavity_h * cavity_w < 2:
                continue
            if cavity_h <= key_max_h and cavity_w <= key_max_w:
                same.append((int(cavity_h), int(cavity_w)))
            if cavity_h != cavity_w and cavity_w <= key_max_h and cavity_h <= key_max_w:
                turned.append((int(cavity_h), int(cavity_w)))

    assert same or turned, "every room pair must admit at least one valid placement"
    if same and turned:
        branch = random.choice(["same", "turn"])
    else:
        branch = "same" if same else "turn"

    cavity_h, cavity_w = random.choice(_prefer_larger(same if branch == "same" else turned))
    key_shape = (int(cavity_h), int(cavity_w)) if branch == "same" else (int(cavity_w), int(cavity_h))
    return {
        "cavity_shape": (int(cavity_h), int(cavity_w)),
        "key_shape": key_shape,
    }


def _place_key(grid: np.ndarray, room: dict, color: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    top = int(room["r0"] + random.randint(1, int(room["h"] - h - 1)))
    left = int(room["c0"] + random.randint(1, int(room["w"] - w - 1)))
    grid[top : top + h, left : left + w] = int(color)
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _place_lock(grid: np.ndarray, room: dict, color: int, cavity_shape: tuple[int, int]) -> tuple[int, int]:
    cavity_h, cavity_w = map(int, cavity_shape)
    frame_h = int(cavity_h + 2)
    frame_w = int(cavity_w + 2)
    top = int(room["r0"] + random.randint(1, int(room["h"] - frame_h - 1)))
    left = int(room["c0"] + random.randint(1, int(room["w"] - frame_w - 1)))

    grid[top, left : left + frame_w] = int(color)
    grid[top + frame_h - 1, left : left + frame_w] = int(color)
    grid[top : top + frame_h, left] = int(color)
    grid[top : top + frame_h, left + frame_w - 1] = int(color)
    return int(top + 1), int(left + 1)


def _expected_output(input_grid: np.ndarray, pair_specs: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    for spec in pair_specs:
        kr0, kc0, kr1, kc1 = map(int, spec["key_bbox"])
        out[kr0 : kr1 + 1, kc0 : kc1 + 1] = _GEN_BG

        lr, lc = map(int, spec["lock_top"])
        cavity_h, cavity_w = map(int, spec["cavity_shape"])
        out[lr : lr + cavity_h, lc : lc + cavity_w] = int(spec["color"])
    return out


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    layout = random.choice(_GEN_LAYOUTS[pair_count])
    built = _build_rooms(layout)
    if built is None:
        return None

    grid, rooms = built
    assert len(rooms) == 2 * pair_count
    colors = random.sample(_GEN_COLOR_POOL, pair_count)
    room_ids = list(range(len(rooms)))
    random.shuffle(room_ids)

    pair_specs: list[dict] = []
    for color, room_a_id, room_b_id in zip(colors, room_ids[::2], room_ids[1::2]):
        room_a = rooms[int(room_a_id)]
        room_b = rooms[int(room_b_id)]
        if random.random() < 0.5:
            key_room, lock_room = room_a, room_b
        else:
            key_room, lock_room = room_b, room_a

        pair_spec = _choose_pair_spec(key_room, lock_room)
        key_bbox = _place_key(grid, key_room, int(color), pair_spec["key_shape"])
        lock_top = _place_lock(grid, lock_room, int(color), pair_spec["cavity_shape"])
        pair_specs.append(
            {
                "color": int(color),
                "key_bbox": key_bbox,
                "lock_top": lock_top,
                "cavity_shape": pair_spec["cavity_shape"],
            }
        )

    expected = _expected_output(grid, pair_specs)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-key-lock puzzle")
