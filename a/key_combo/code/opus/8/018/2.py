from __future__ import annotations

import itertools
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, find_components, label_mask


BG = 0
WALL = 5
SEGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))


def _room_pattern(shape: tuple[int, int], segments: list[tuple[int, int]], open_room: bool) -> np.ndarray:
    h, w = shape
    patch = np.zeros((h, w), dtype=int)
    ordered = sorted(segments, key=lambda item: item[0], reverse=not open_room)
    if open_room:
        for r, (length, color) in enumerate(ordered):
            patch[r, :length] = color
    else:
        for i, (length, color) in enumerate(ordered):
            r = h - 1 - i
            patch[r, w - length : w] = color
    return patch


def _segment_signature(segments: list[dict]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(seg["length"]), int(seg["color"])) for seg in segments))


def _analyze_rooms(grid: np.ndarray, *, require_segments: bool = True) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

    black_labels, room_count = label_mask(g == BG, connectivity=4)
    assert room_count >= 1

    rooms: list[dict] = []
    for label, coords in enumerate(components_mask(g == BG, connectivity=4), start=1):
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        rooms.append(
            {
                "label": label,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "open": _touches_edge(coords, g.shape),
                "segments": [],
            }
        )

    segment_colors = sorted(int(color) for color in np.unique(g) if color not in {BG, WALL})
    components = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=segment_colors,
    )

    for component in components:
        color = int(component["color"])
        assert color != WALL
        r0, c0, r1, c1 = map(int, component["bbox"])
        area = int(component["area"])
        assert r0 == r1, "segments must be one row high"
        assert area == c1 - c0 + 1, "segments must be contiguous horizontal bars"

        labels = set()
        for rr, cc in component["coords"]:
            r = int(rr)
            c = int(cc)
            if r > 0 and black_labels[r - 1, c] > 0:
                labels.add(int(black_labels[r - 1, c]))
            if r + 1 < g.shape[0] and black_labels[r + 1, c] > 0:
                labels.add(int(black_labels[r + 1, c]))
            if c > 0 and black_labels[r, c - 1] > 0:
                labels.add(int(black_labels[r, c - 1]))
            if c + 1 < g.shape[1] and black_labels[r, c + 1] > 0:
                labels.add(int(black_labels[r, c + 1]))
        assert len(labels) == 1, "each segment must belong to exactly one room"

        room = rooms[next(iter(labels)) - 1]
        room["segments"].append(
            {
                "color": color,
                "length": area,
                "coords": component["coords"],
                "bbox": (r0, c0, r1, c1),
            }
        )

    open_rooms = 0
    closed_rooms = 0
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        segments = room["segments"]
        if require_segments:
            assert segments, "each room must contain segments"
            assert 2 <= len(segments) <= 4
            assert h >= len(segments) + 1, "each room needs an empty row"
            lengths = [int(seg["length"]) for seg in segments]
            assert len(set(lengths)) == len(lengths), "segment lengths must be unique within a room"
            assert all(1 <= length <= w - 2 for length in lengths)
        open_rooms += int(room["open"])
        closed_rooms += int(not room["open"])
    if require_segments:
        assert open_rooms >= 2
        assert closed_rooms >= 2
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """Stack room segments in opposite directions depending on whether the room reaches the grid edge.

    Input structure
    The grid has black background space and grey orthogonal walls that partition
    the canvas into rectangular rooms. Each room contains several one-cell-high
    colored horizontal segments whose lengths are unique within that room.

    Rule
    Detect the 4-connected black regions between the walls. If a room's black
    region touches the outer border, treat it as open: clear its original
    segments, sort them by increasing length, then place them from the room's
    top row downward with left alignment. Otherwise treat the room as enclosed:
    clear its original segments, sort them by decreasing length, then place them
    from the room's bottom row upward with right alignment. Keep every grey wall
    unchanged and leave all unused room cells black.

    Key insight
    The same collection of segments is rearranged differently solely from the
    room's topology: edge-reaching rooms stack short-to-long from the top-left,
    while enclosed rooms stack long-to-short from the bottom-right.
    """
    g = np.asarray(grid, dtype=int)
    rooms = _analyze_rooms(g)

    out = g.copy()
    out[(out != BG) & (out != WALL)] = BG

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        segments = room["segments"]
        ordered = sorted(segments, key=lambda seg: int(seg["length"]), reverse=not room["open"])
        if room["open"]:
            for i, seg in enumerate(ordered):
                length = int(seg["length"])
                out[r0 + i, c0 : c0 + length] = int(seg["color"])
        else:
            for i, seg in enumerate(ordered):
                length = int(seg["length"])
                out[r1 - i, c1 - length + 1 : c1 + 1] = int(seg["color"])

    return out


def _choose_nonadjacent_rows(height: int, count: int) -> list[int] | None:
    choices = []
    for rows in itertools.combinations(range(height), count):
        if all(rows[i + 1] - rows[i] >= 2 for i in range(count - 1)):
            choices.append(rows)
    if not choices:
        return None
    return list(random.choice(choices))


def _render_input_patch(shape: tuple[int, int], placements: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    patch = np.zeros((h, w), dtype=int)
    for row, col, length, color in placements:
        patch[row, col : col + length] = color
    return patch


def _sample_room_segments(
    shape: tuple[int, int],
    open_room: bool,
    *,
    shared_spec: list[tuple[int, int]] | None = None,
    forbid_target: bool = False,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = shape
    max_segments = min(4, (h + 1) // 2, w - 2)
    assert max_segments >= 2

    if shared_spec is None:
        count = random.randint(2, max_segments)
        lengths = random.sample(range(1, w - 1), count)
        colors = random.sample(list(SEGMENT_COLORS), count)
        spec = list(zip(lengths, colors))
    else:
        spec = list(shared_spec)
        count = len(spec)
        assert count <= max_segments
        assert all(1 <= length <= w - 2 for length, _color in spec)

    target = _room_pattern(shape, spec, open_room)
    rows = _choose_nonadjacent_rows(h, count)
    assert rows is not None

    for _ in range(120):
        placements = []
        perm = list(range(count))
        random.shuffle(perm)
        for row, idx in zip(rows, perm):
            length, color = spec[idx]
            col = random.randint(0, w - length)
            placements.append((row, col, length, color))
        patch = _render_input_patch(shape, placements)
        if forbid_target and np.array_equal(patch, target):
            continue
        return spec, patch

    raise RuntimeError("failed to sample a non-degenerate room layout")


def _build_layout() -> tuple[np.ndarray, list[dict]]:
    top_h = random.randint(4, 7)
    mid_h = random.randint(4, 7)
    bottom_h = random.randint(4, 7)
    mid_widths = [random.randint(4, 7) for _ in range(4)]

    h = top_h + mid_h + bottom_h + 2
    w = sum(mid_widths) + 3
    assert max(h, w) <= 30

    r_top_wall = top_h
    r_bottom_wall = top_h + mid_h + 1
    c1 = mid_widths[0]
    c2 = mid_widths[0] + mid_widths[1] + 1
    c3 = mid_widths[0] + mid_widths[1] + mid_widths[2] + 2

    grid = np.zeros((h, w), dtype=int)
    grid[r_top_wall, :] = WALL
    grid[r_bottom_wall, :] = WALL
    for col in (c1, c2, c3):
        grid[r_top_wall : r_bottom_wall + 1, col] = WALL

    top_split = random.choice([None, c1, c2, c3])
    bottom_split = random.choice([None, c1, c2, c3])
    if top_split is not None:
        grid[: r_top_wall + 1, top_split] = WALL
    if bottom_split is not None:
        grid[r_bottom_wall:, bottom_split] = WALL

    rooms = _analyze_rooms(grid, require_segments=False)
    assert 6 <= len(rooms) <= 8
    assert sum(int(room["open"]) for room in rooms) >= 2
    assert sum(int(not room["open"]) for room in rooms) >= 2
    return grid, rooms


def _place_segments_on_grid(base: np.ndarray, rooms: list[dict]) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    grid = base.copy()
    room_specs: dict[int, list[tuple[int, int]]] = {}

    middle_open = [
        room for room in rooms
        if room["open"] and room["bbox"][0] > 0 and room["bbox"][2] < grid.shape[0] - 1
    ]
    closed_rooms = [room for room in rooms if not room["open"]]
    assert len(middle_open) >= 2
    assert len(closed_rooms) >= 2

    pair_open = random.choice(middle_open)
    pair_closed = random.choice(closed_rooms)
    pair_h = min(pair_open["bbox"][2] - pair_open["bbox"][0] + 1, pair_closed["bbox"][2] - pair_closed["bbox"][0] + 1)
    pair_w = min(pair_open["bbox"][3] - pair_open["bbox"][1] + 1, pair_closed["bbox"][3] - pair_closed["bbox"][1] + 1)
    pair_cap = min(4, (pair_h + 1) // 2, pair_w - 2)
    assert pair_cap >= 2
    pair_count = random.randint(2, pair_cap)
    pair_lengths = random.sample(range(1, pair_w - 1), pair_count)
    pair_colors = random.sample(list(SEGMENT_COLORS), pair_count)
    shared_spec = list(zip(pair_lengths, pair_colors))

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        room_shape = (r1 - r0 + 1, c1 - c0 + 1)
        paired = room["label"] in {pair_open["label"], pair_closed["label"]}
        spec, patch = _sample_room_segments(
            room_shape,
            room["open"],
            shared_spec=shared_spec if paired else None,
            forbid_target=paired or random.random() < 0.75,
        )
        room_specs[room["label"]] = spec
        mask = patch != BG
        grid[r0 : r1 + 1, c0 : c1 + 1][mask] = patch[mask]

    return grid, room_specs


def _semantic_validation(inp: np.ndarray) -> bool:
    try:
        rooms = _analyze_rooms(inp)
        out = main(inp)
        out_rooms = _analyze_rooms(out)
    except Exception:
        return False

    if np.array_equal(inp, out):
        return False
    if len(rooms) != len(out_rooms):
        return False

    open_sigs = {}
    closed_sigs = {}
    open_changed = False
    closed_changed = False

    for room, out_room in zip(sorted(rooms, key=lambda item: item["label"]), sorted(out_rooms, key=lambda item: item["label"])):
        if room["bbox"] != out_room["bbox"] or room["open"] != out_room["open"]:
            return False

        r0, c0, r1, c1 = room["bbox"]
        room_shape = (r1 - r0 + 1, c1 - c0 + 1)
        expected_patch = _room_pattern(room_shape, _segment_signature(room["segments"]), room["open"])
        actual_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.array_equal(expected_patch, actual_patch):
            return False

        in_patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        if room["open"]:
            open_changed |= not np.array_equal(in_patch, expected_patch)
            open_sigs.setdefault(_segment_signature(room["segments"]), 0)
            open_sigs[_segment_signature(room["segments"])] += 1
        else:
            closed_changed |= not np.array_equal(in_patch, expected_patch)
            closed_sigs.setdefault(_segment_signature(room["segments"]), 0)
            closed_sigs[_segment_signature(room["segments"])] += 1

    if not open_changed or not closed_changed:
        return False

    shared = set(open_sigs) & set(closed_sigs)
    if not shared:
        return False

    for sig in shared:
        lengths_only = [length for length, _color in sig]
        if len(lengths_only) < 2 or len(set(lengths_only)) != len(lengths_only):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        base, rooms = _build_layout()
        inp, _room_specs = _place_segments_on_grid(base, rooms)
    except Exception:
        return None

    if not _semantic_validation(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid topological-room-stacking puzzle")
