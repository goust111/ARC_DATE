from __future__ import annotations

from collections import deque
from itertools import permutations
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
ROOM_COUNT_RANGE = (3, 4)
GRID_SIDE_RANGE = (18, 28)
ROOM_MIN_SIDE = 5
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[np.ndarray] = []
    next_label = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
            next_label += 1
    return labels, comps


def _is_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return False
    return len(_label_components(mask)[1]) == 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _build_stamp_library() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if area < 5 or area > 7:
            continue
        box = _bbox(mask)
        if box != (0, 0, 2, 2):
            continue
        if not _is_connected(mask):
            continue
        masks.append(mask)
    return masks


STAMP_LIBRARY = _build_stamp_library()


def _analyze_scene(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    black_labels, black_regions = _label_components(g == BLACK)
    assert len(black_regions) >= 2

    infos: list[dict[str, object]] = []
    seen_labels: set[int] = set()
    colors = sorted(int(color) for color in np.unique(g) if color not in (BLACK, GREY))
    assert colors

    for color in colors:
        color_mask = g == color
        comps = _label_components(color_mask)[1]
        assert len(comps) == 1
        box = _bbox(color_mask)
        assert box is not None
        r0, c0, r1, c1 = box
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3)

        stamp_mask = color_mask[r0 : r1 + 1, c0 : c1 + 1]
        adjacent_labels = set()
        for rr, cc in np.argwhere(color_mask):
            for nr, nc in _neighbors4(int(rr), int(cc)):
                if _in_bounds(g.shape, nr, nc) and black_labels[nr, nc] > 0:
                    adjacent_labels.add(int(black_labels[nr, nc]))
        assert len(adjacent_labels) == 1
        (label,) = tuple(adjacent_labels)
        assert label not in seen_labels
        seen_labels.add(label)

        room_coords = black_regions[label - 1]
        infos.append(
            {
                "label": label,
                "color": color,
                "mask": stamp_mask.copy(),
                "stamp_area": int(stamp_mask.sum()),
                "black_area": int(room_coords.shape[0]),
                "bbox": (r0, c0, r1, c1),
            }
        )

    assert len(infos) == len(black_regions)
    black_areas = [int(info["black_area"]) for info in infos]
    assert len(set(black_areas)) == len(black_areas)
    return infos


def _overlay(infos: list[dict[str, object]], order: list[int]) -> np.ndarray:
    out = np.zeros((3, 3), dtype=int)
    for idx in order:
        info = infos[idx]
        out[info["mask"]] = int(info["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the room stamps onto a 3x3 canvas from largest black room to smallest black room.

    Input structure:
    - Grey one-cell walls partition the black background into several 4-connected black rooms.
    - Each room contains exactly one monochromatic stamp whose occupied cells fit in a 3x3 bounding box.
    - Room black areas are all distinct, and stamp colors are all distinct.

    Rule:
    - Measure the visible black area of every room.
    - Sort the rooms from larger black area to smaller black area.
    - Start from a black 3x3 canvas and overlay each room's 3x3 stamp mask in that order using the stamp's color.
    - Later stamps overwrite earlier colors on overlapping cells, so each cell keeps the color of the stamp in the smallest room that covers it.

    Key insight:
    - The layering key is not any property of the stamp itself; it is the size of the room that contains that stamp.
    """
    infos = _analyze_scene(np.asarray(grid, dtype=int))
    order = sorted(range(len(infos)), key=lambda idx: -int(infos[idx]["black_area"]))
    return _overlay(infos, order)


def _split_candidates(room: tuple[int, int, int, int]) -> list[tuple[str, int]]:
    r0, r1, c0, c1 = room
    candidates: list[tuple[str, int]] = []
    if r1 - r0 + 1 >= 2 * ROOM_MIN_SIDE + 1:
        for wall_r in range(r0 + ROOM_MIN_SIDE, r1 - ROOM_MIN_SIDE + 1):
            candidates.append(("h", wall_r))
    if c1 - c0 + 1 >= 2 * ROOM_MIN_SIDE + 1:
        for wall_c in range(c0 + ROOM_MIN_SIDE, c1 - ROOM_MIN_SIDE + 1):
            candidates.append(("v", wall_c))
    return candidates


def _partition_grid(rng: random.Random, n_rooms: int, height: int, width: int):
    rooms = [(0, height - 1, 0, width - 1)]
    walls: list[tuple[str, int, int, int]] = []
    while len(rooms) < n_rooms:
        options = []
        for idx, room in enumerate(rooms):
            for orient, wall in _split_candidates(room):
                options.append((idx, orient, wall))
        if not options:
            return None
        idx, orient, wall = rng.choice(options)
        r0, r1, c0, c1 = rooms.pop(idx)
        if orient == "h":
            walls.append(("h", wall, c0, c1))
            rooms.append((r0, wall - 1, c0, c1))
            rooms.append((wall + 1, r1, c0, c1))
        else:
            walls.append(("v", wall, r0, r1))
            rooms.append((r0, r1, c0, wall - 1))
            rooms.append((r0, r1, wall + 1, c1))
    grid = np.full((height, width), BLACK, dtype=int)
    for orient, line, a0, a1 in walls:
        if orient == "h":
            grid[line, a0 : a1 + 1] = GREY
        else:
            grid[a0 : a1 + 1, line] = GREY
    return rooms, grid


def _sample_rooms(rng: random.Random):
    n_rooms = rng.randint(ROOM_COUNT_RANGE[0], ROOM_COUNT_RANGE[1])
    for _ in range(80):
        height = rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1])
        width = rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1])
        made = _partition_grid(rng, n_rooms, height, width)
        if made is None:
            continue
        rooms, grid = made
        room_areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, r1, c0, c1 in rooms]
        if len(set(room_areas)) != len(room_areas):
            continue
        sorted_areas = sorted(room_areas, reverse=True)
        if min(a - b for a, b in zip(sorted_areas, sorted_areas[1:])) > 2:
            continue
        return rooms, grid
    return None


def _place_stamp_in_room(
    rng: random.Random,
    room: tuple[int, int, int, int],
    mask: np.ndarray,
    color: int,
    grid: np.ndarray,
) -> dict[str, object]:
    r0, r1, c0, c1 = room
    top = rng.randint(r0 + 1, r1 - 3)
    left = rng.randint(c0 + 1, c1 - 3)
    stamp = np.zeros((3, 3), dtype=int)
    stamp[mask] = color
    grid[top : top + 3, left : left + 3][mask] = color
    return {
        "room": room,
        "room_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
        "room_pos": (r0, c0),
        "stamp_area": int(mask.sum()),
        "mask": mask.copy(),
        "color": color,
        "bbox": (top, left, top + 2, left + 2),
        "black_area": int((r1 - r0 + 1) * (c1 - c0 + 1) - mask.sum()),
    }


def _alternative_orders(records: list[dict[str, object]], correct_order: list[int]) -> dict[str, list[int]]:
    return {
        "room_bbox_area": sorted(
            range(len(records)),
            key=lambda idx: (-int(records[idx]["room_area"]), records[idx]["room_pos"]),
        ),
        "stamp_pixels": sorted(
            range(len(records)),
            key=lambda idx: (-int(records[idx]["stamp_area"]), records[idx]["room_pos"]),
        ),
        "reading_order": sorted(range(len(records)), key=lambda idx: records[idx]["room_pos"]),
        "color_id": sorted(range(len(records)), key=lambda idx: int(records[idx]["color"])),
        "reverse": list(reversed(correct_order)),
    }


def _records_semantically_valid(records: list[dict[str, object]]) -> bool:
    correct_order = sorted(range(len(records)), key=lambda idx: -int(records[idx]["black_area"]))
    correct = _overlay(records, correct_order)

    coverage = np.zeros((3, 3), dtype=int)
    for record in records:
        coverage += np.asarray(record["mask"], dtype=int)
    if int(coverage.max()) < 2:
        return False

    for idx in range(len(records)):
        reduced_order = [j for j in correct_order if j != idx]
        if np.array_equal(_overlay(records, reduced_order), correct):
            return False

    alt_orders = _alternative_orders(records, correct_order)
    room_area_order = alt_orders["room_bbox_area"]
    if room_area_order == correct_order:
        return False
    if np.array_equal(_overlay(records, room_area_order), correct):
        return False

    if np.array_equal(_overlay(records, alt_orders["reverse"]), correct):
        return False

    shortcut_breaks = 0
    for name in ("stamp_pixels", "reading_order", "color_id"):
        order = alt_orders[name]
        if order != correct_order and not np.array_equal(_overlay(records, order), correct):
            shortcut_breaks += 1
    if shortcut_breaks < 2:
        return False

    black_areas = [int(record["black_area"]) for record in records]
    if len(set(black_areas)) != len(black_areas):
        return False

    stamp_areas = [int(record["stamp_area"]) for record in records]
    if len(set(stamp_areas)) < 2:
        return False

    if len(set(int(color) for color in correct.ravel() if color != BLACK)) != len(records):
        return False
    return True


def _mask_sequence_valid(masks: list[np.ndarray]) -> bool:
    records = []
    for idx, mask in enumerate(masks):
        records.append(
            {
                "room_area": 100 - idx * 10,
                "room_pos": (idx, 0),
                "stamp_area": int(mask.sum()),
                "mask": mask.copy(),
                "color": idx + 1,
                "black_area": 100 - idx * 10,
            }
        )
    correct = _overlay(records, list(range(len(records))))
    coverage = np.zeros((3, 3), dtype=int)
    for mask in masks:
        coverage += np.asarray(mask, dtype=int)
    if int(coverage.max()) < 2:
        return False
    if len(set(int(color) for color in correct.ravel() if color != BLACK)) != len(masks):
        return False
    for idx in range(len(masks)):
        reduced = [j for j in range(len(masks)) if j != idx]
        if np.array_equal(_overlay(records, reduced), correct):
            return False
    if np.array_equal(_overlay(records, list(reversed(range(len(masks))))), correct):
        return False
    return True


def _build_ordered_masks(rng: random.Random, n_rooms: int) -> list[np.ndarray] | None:
    cells = [(r, c) for r in range(3) for c in range(3)]
    for _ in range(60):
        anchors = rng.sample(cells, n_rooms)
        blocked: set[tuple[int, int]] = set()
        masks: list[np.ndarray] = [np.zeros((3, 3), dtype=bool) for _ in range(n_rooms)]
        used_keys: set[bytes] = set()
        ok = True
        for idx in range(n_rooms - 1, -1, -1):
            anchor = anchors[idx]
            candidates = []
            for mask in STAMP_LIBRARY:
                if not bool(mask[anchor]):
                    continue
                if any(bool(mask[cell]) for cell in blocked):
                    continue
                key = mask.tobytes()
                if key in used_keys:
                    continue
                candidates.append(mask)
            if not candidates:
                ok = False
                break
            chosen = rng.choice(candidates)
            masks[idx] = chosen
            used_keys.add(chosen.tobytes())
            blocked.add(anchor)
        if ok and len({int(mask.sum()) for mask in masks}) >= 2 and _mask_sequence_valid(masks):
            return masks
    return None


def _precompute_mask_sequences() -> dict[int, list[list[np.ndarray]]]:
    rng = random.Random(0)
    sequences: dict[int, list[list[np.ndarray]]] = {}
    for n_rooms in range(ROOM_COUNT_RANGE[0], ROOM_COUNT_RANGE[1] + 1):
        seen: set[tuple[bytes, ...]] = set()
        seqs: list[list[np.ndarray]] = []
        for _ in range(600):
            masks = _build_ordered_masks(rng, n_rooms)
            if masks is None:
                continue
            key = tuple(mask.tobytes() for mask in masks)
            if key in seen:
                continue
            seen.add(key)
            seqs.append([mask.copy() for mask in masks])
            if len(seqs) >= 10:
                break
        if not seqs:
            raise RuntimeError(f"failed to precompute mask sequences for {n_rooms} rooms")
        sequences[n_rooms] = seqs
    return sequences


VALID_MASK_SEQUENCES = _precompute_mask_sequences()


def _validate_generated(grid: np.ndarray, records: list[dict[str, object]]) -> bool:
    if not _records_semantically_valid(records):
        return False
    try:
        analyzed = _analyze_scene(grid)
    except Exception:
        return False

    if len(analyzed) != len(records):
        return False

    by_color = {int(record["color"]): record for record in records}
    for info in analyzed:
        color = int(info["color"])
        if color not in by_color:
            return False
        record = by_color[color]
        if int(info["black_area"]) != int(record["black_area"]):
            return False
        if not np.array_equal(info["mask"], record["mask"]):
            return False

    correct_order = sorted(range(len(records)), key=lambda idx: -int(records[idx]["black_area"]))
    correct = _overlay(records, correct_order)
    return np.array_equal(main(grid), correct)


def _find_valid_records(
    rng: random.Random, rooms: list[tuple[int, int, int, int]]
) -> list[dict[str, object]] | None:
    n_rooms = len(rooms)
    room_meta = []
    for room in rooms:
        r0, r1, c0, c1 = room
        room_meta.append(
            {
                "room": room,
                "room_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "room_pos": (r0, c0),
            }
        )

    raw_area_order = sorted(
        range(n_rooms),
        key=lambda idx: (-int(room_meta[idx]["room_area"]), room_meta[idx]["room_pos"]),
    )
    perms = list(permutations(range(n_rooms)))
    mask_sequences = VALID_MASK_SEQUENCES[n_rooms][:]
    rng.shuffle(mask_sequences)
    for masks in mask_sequences[:15]:
        colors = rng.sample(PALETTE, n_rooms)
        rng.shuffle(perms)
        for perm in perms:
            if list(perm) == raw_area_order:
                continue
            black_areas = [
                int(room_meta[perm[layer]]["room_area"] - masks[layer].sum())
                for layer in range(n_rooms)
            ]
            if any(a <= b for a, b in zip(black_areas, black_areas[1:])):
                continue
            records = []
            for room_idx, room_info in enumerate(room_meta):
                layer = perm.index(room_idx)
                mask = masks[layer]
                records.append(
                    {
                        "room": room_info["room"],
                        "room_area": room_info["room_area"],
                        "room_pos": room_info["room_pos"],
                        "stamp_area": int(mask.sum()),
                        "mask": mask.copy(),
                        "color": int(colors[layer]),
                        "black_area": int(room_info["room_area"] - mask.sum()),
                    }
                )
            if _records_semantically_valid(records):
                return records
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    made = _sample_rooms(rng)
    if made is None:
        return None
    rooms, grid = made
    base_records = _find_valid_records(rng, rooms)
    if base_records is None:
        return None

    records = []
    for base in base_records:
        placed = _place_stamp_in_room(
            rng,
            base["room"],
            np.asarray(base["mask"], dtype=bool),
            int(base["color"]),
            grid,
        )
        placed["room_area"] = int(base["room_area"])
        placed["room_pos"] = base["room_pos"]
        placed["black_area"] = int(base["black_area"])
        records.append(placed)

    return grid if _validate_generated(grid, records) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-overlay scene")
