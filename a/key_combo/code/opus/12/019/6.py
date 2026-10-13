from __future__ import annotations

import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import bbox_coords, find_components


BG = 0
TRACK = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7)
MAX_GEN_ATTEMPTS = 400

SHAPES = [
    {"coords": np.array([(0, 0), (0, 1)], dtype=int), "h": 1, "w": 2},
    {"coords": np.array([(0, 0), (1, 0)], dtype=int), "h": 2, "w": 1},
    {"coords": np.array([(0, 0), (1, 0), (1, 1)], dtype=int), "h": 2, "w": 2},
    {"coords": np.array([(0, 1), (1, 0), (1, 1)], dtype=int), "h": 2, "w": 2},
    {"coords": np.array([(0, 0), (1, 0), (2, 0)], dtype=int), "h": 3, "w": 1},
    {"coords": np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int), "h": 2, "w": 2},
    {"coords": np.array([(0, 0), (1, 0), (2, 0), (2, 1)], dtype=int), "h": 3, "w": 2},
    {"coords": np.array([(0, 1), (1, 1), (2, 0), (2, 1)], dtype=int), "h": 3, "w": 2},
    {"coords": np.array([(0, 0), (1, 0), (2, 0), (3, 0)], dtype=int), "h": 4, "w": 1},
]


def _track_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in np.flatnonzero(np.all(grid == TRACK, axis=1))]


def _component_records(grid: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, TRACK)]
    comps = find_components(
        grid,
        bg=BG,
        colors=colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    records = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        records.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return records


def _drop_record(record: dict, track_rows: list[int], height: int) -> dict:
    coords = np.asarray(record["coords"], dtype=int)
    bottom = int(coords[:, 0].max())
    lower_tracks = [int(r) for r in track_rows if int(r) > int(bottom)]
    if lower_tracks:
        target_track = min(lower_tracks)
        dr = int(target_track - 1 - bottom)
    else:
        target_track = None
        dr = int(height - 1 - bottom)
    fallen = coords.copy()
    fallen[:, 0] += int(dr)
    r0, c0, r1, c1 = bbox_coords(fallen)
    return {
        "color": int(record["color"]),
        "coords": fallen,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "track": target_track,
    }


def _slide_order_key(record: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = record["bbox"]
    return (-int(c1), -int(c0), int(r0), int(r1))


def _paint_object(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each colored object onto the nearest lower grey rail, then slide rail-sharing objects rightward without overlap.

    Input structure:
    - The background is black.
    - Several full-width grey rows mark horizontal rails with a fixed vertical period.
    - Above and between those rails sit small monochromatic 4-connected colored objects.

    Rule:
    - Treat every full-width grey row as a rail and every non-grey, non-black component as one object.
    - Each object falls straight down until its bottom row is exactly one cell above the nearest lower rail; if no lower rail exists, it falls to the grid bottom instead.
    - For each rail, consider only the objects now resting directly above it, sort them by original post-fall right edge from right to left, keep the first object fixed, and slide each later object right until the next step would hit the wall or an already settled object on that rail.
    - Preserve object shapes and colors, keep the grey rails unchanged, and clear the original floating positions.

    Key insight:
    The grey stripe period determines which objects merge onto the same landing rail, and only those merged objects participate in the rightward collision-packing step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    track_rows = _track_rows(g)
    records = [_drop_record(rec, track_rows, h) for rec in _component_records(g)]

    out = np.full_like(g, BG)
    if track_rows:
        out[np.asarray(track_rows, dtype=int), :] = TRACK
    occupied = np.zeros((h, w), dtype=bool)
    if track_rows:
        occupied[np.asarray(track_rows, dtype=int), :] = True

    grouped: dict[int, list[dict]] = {}
    floor_records: list[dict] = []
    for record in records:
        if record["track"] is None:
            floor_records.append(record)
        else:
            grouped.setdefault(int(record["track"]), []).append(record)

    for record in floor_records:
        _paint_object(out, record["coords"], record["color"])
        occupied[record["coords"][:, 0], record["coords"][:, 1]] = True

    for track in track_rows:
        group = sorted(grouped.get(int(track), []), key=_slide_order_key)
        for idx, record in enumerate(group):
            coords = np.asarray(record["coords"], dtype=int)
            shift = 0
            if idx > 0:
                while True:
                    moved = coords.copy()
                    moved[:, 1] += int(shift + 1)
                    if int(moved[:, 1].max()) >= int(w):
                        break
                    if np.any(occupied[moved[:, 0], moved[:, 1]]):
                        break
                    shift += 1
            dest = coords.copy()
            dest[:, 1] += int(shift)
            _paint_object(out, dest, record["color"])
            occupied[dest[:, 0], dest[:, 1]] = True

    return out


def _shape_coords(shape: dict, top: int, left: int) -> np.ndarray:
    coords = np.asarray(shape["coords"], dtype=int).copy()
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _bbox_from_top_left(shape: dict, top: int, left: int) -> tuple[int, int, int, int]:
    return (
        int(top),
        int(left),
        int(top + shape["h"] - 1),
        int(left + shape["w"] - 1),
    )


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (int(a1) < int(b0) or int(b1) < int(a0))


def _grid_from_layout(height: int, width: int, track_rows: list[int], objects: list[dict], key: str) -> np.ndarray:
    grid = np.full((int(height), int(width)), BG, dtype=int)
    if track_rows:
        grid[np.asarray(track_rows, dtype=int), :] = TRACK
    for obj in objects:
        coords = np.asarray(obj[key], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return grid


def _valid_shadow_free(objects: list[dict]) -> bool:
    bboxes = [obj["input_bbox"] for obj in objects]
    for i in range(len(bboxes)):
        r0a, c0a, r1a, c1a = bboxes[i]
        for j in range(i + 1, len(bboxes)):
            r0b, c0b, r1b, c1b = bboxes[j]
            if _intervals_overlap(c0a, c1a, c0b, c1b) and not _intervals_overlap(r0a, r1a, r0b, r1b):
                return False
    return True


def _sample_geometry() -> tuple[int, int, list[int], int] | None:
    period = random.choice((3, 4))
    head = random.randint(4, 6)
    n_tracks = random.randint(3, 5 if period == 3 else 4)
    tail = random.randint(1, period + 1)
    track_rows = [int(head + i * period) for i in range(n_tracks)]
    height = int(track_rows[-1] + tail + 1)
    if height > 26:
        return None
    width = random.randint(14, 19)
    return height, width, track_rows, tail


def _band_start(track_rows: list[int], idx: int) -> int:
    return 0 if idx == 0 else int(track_rows[idx - 1] + 1)


def _pick_shape(max_height: int, *, require_drop: bool = False) -> dict | None:
    candidates = [shape for shape in SHAPES if int(shape["h"]) <= int(max_height)]
    if require_drop:
        candidates = [shape for shape in candidates if int(shape["h"]) < int(max_height)]
    if not candidates:
        return None
    return random.choice(candidates)


def _assign_counts(track_rows: list[int], tail: int) -> tuple[dict[int, int], int]:
    n_tracks = len(track_rows)
    focus_idx = 0 if random.random() < 0.5 else random.randrange(n_tracks)
    focus_size = random.choice((2, 3))
    counts = {idx: 0 for idx in range(n_tracks)}
    counts[focus_idx] = focus_size
    if focus_idx != 0:
        counts[0] = 1

    bottom_count = 1 if random.random() < 0.45 and int(tail) >= 1 else 0
    minimum = sum(counts.values()) + bottom_count
    total = random.randint(max(4, minimum), 6)
    remaining = int(total - minimum)
    while remaining > 0:
        candidates = [idx for idx, count in counts.items() if int(count) < 3]
        idx = random.choice(candidates)
        counts[idx] += 1
        remaining -= 1
    return counts, bottom_count


def _build_track_group(
    height: int,
    width: int,
    track_rows: list[int],
    idx: int,
    count: int,
    colors: list[int],
    must_drop: bool,
) -> list[dict] | None:
    track = int(track_rows[idx])
    band_start = _band_start(track_rows, idx)
    band_height = int(track - band_start)
    if count <= 0:
        return []

    shapes: list[dict] = []
    for k in range(count):
        shape = _pick_shape(band_height, require_drop=bool(must_drop and k == 0))
        if shape is None:
            return None
        shapes.append(shape)

    total_width = sum(int(shape["w"]) for shape in shapes)
    if total_width > int(width):
        return None
    right_gap = random.randint(0, min(4 if count == 1 else 3, width - total_width))

    final_lefts_rt: list[int] = []
    cursor = int(width - 1 - right_gap)
    for shape in shapes:
        left = int(cursor - shape["w"] + 1)
        if left < 0:
            return None
        final_lefts_rt.append(left)
        cursor = int(left - 1)

    input_lefts_rt = [int(final_lefts_rt[0])]
    for order in range(1, count):
        shape = shapes[order]
        max_left = min(int(final_lefts_rt[order]), int(input_lefts_rt[order - 1] - shape["w"]))
        if max_left < 0:
            return None
        low = max(0, int(max_left - 4))
        high = int(max_left)
        if high <= low:
            left = high
        else:
            prefer_high = high
            if int(final_lefts_rt[order]) > low and random.random() < 0.85:
                prefer_high = min(prefer_high, int(final_lefts_rt[order] - 1))
            if prefer_high < low:
                left = random.randint(low, high)
            else:
                left = random.randint(low, prefer_high)
        input_lefts_rt.append(int(left))

    objects = []
    for order, shape in enumerate(shapes):
        final_top = int(track - shape["h"])
        max_drop = int(final_top - band_start)
        if max_drop < 0:
            return None
        if max_drop == 0:
            drop = 0
        elif must_drop and order == 0:
            drop = random.randint(1, max_drop)
        else:
            drop = random.randint(0, max_drop)
            if max_drop > 0 and random.random() < 0.75:
                drop = max(drop, 1)
        input_top = int(final_top - drop)
        color = int(colors[order])
        final_left = int(final_lefts_rt[order])
        input_left = int(input_lefts_rt[order])
        final_coords = _shape_coords(shape, final_top, final_left)
        input_coords = _shape_coords(shape, input_top, input_left)
        objects.append(
            {
                "color": color,
                "track": track,
                "shape": shape,
                "final_coords": final_coords,
                "input_coords": input_coords,
                "final_bbox": _bbox_from_top_left(shape, final_top, final_left),
                "input_bbox": _bbox_from_top_left(shape, input_top, input_left),
            }
        )
    return objects


def _build_bottom_group(height: int, width: int, track_rows: list[int], color: int) -> list[dict] | None:
    band_start = int(track_rows[-1] + 1)
    band_height = int(height - band_start)
    shape = _pick_shape(band_height, require_drop=(band_height >= 2))
    if shape is None:
        return None
    left = random.randint(0, int(width - shape["w"]))
    final_top = int(height - shape["h"])
    max_drop = int(final_top - band_start)
    if max_drop < 0:
        return None
    drop = 0 if max_drop == 0 else random.randint(1, max_drop)
    input_top = int(final_top - drop)
    final_coords = _shape_coords(shape, final_top, left)
    input_coords = _shape_coords(shape, input_top, left)
    return [
        {
            "color": int(color),
            "track": None,
            "shape": shape,
            "final_coords": final_coords,
            "input_coords": input_coords,
            "final_bbox": _bbox_from_top_left(shape, final_top, left),
            "input_bbox": _bbox_from_top_left(shape, input_top, left),
        }
    ]


def _non_overlapping(objects: list[dict], key: str, height: int, width: int) -> bool:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    for obj in objects:
        coords = np.asarray(obj[key], dtype=int)
        if coords.size == 0:
            return False
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= int(height)):
            return False
        if np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= int(width)):
            return False
        if np.any(mask[coords[:, 0], coords[:, 1]]):
            return False
        mask[coords[:, 0], coords[:, 1]] = True
    return True


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, counts: dict[int, int], bottom_count: int) -> bool:
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    track_rows = _track_rows(inp)
    if len(track_rows) < 3:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not any(int(count) >= 2 for count in counts.values()):
        return False
    if bottom_count and not np.any(expected[-1] != BG):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    geometry = _sample_geometry()
    if geometry is None:
        return None
    height, width, track_rows, tail = geometry
    counts, bottom_count = _assign_counts(track_rows, tail)
    total_objects = int(sum(counts.values()) + bottom_count)
    colors = random.sample(list(OBJECT_COLORS), total_objects)
    color_idx = 0

    objects: list[dict] = []
    for idx in range(len(track_rows)):
        count = int(counts[idx])
        if count == 0:
            continue
        group_colors = colors[color_idx : color_idx + count]
        color_idx += count
        must_drop = idx == 0
        group = _build_track_group(height, width, track_rows, idx, count, group_colors, must_drop)
        if group is None:
            return None
        objects.extend(group)

    if bottom_count:
        group = _build_bottom_group(height, width, track_rows, colors[color_idx])
        if group is None:
            return None
        objects.extend(group)

    if len(objects) != total_objects:
        return None
    if not _non_overlapping(objects, "input_coords", height, width):
        return None
    if not _non_overlapping(objects, "final_coords", height, width):
        return None
    if not _valid_shadow_free(objects):
        return None

    input_grid = _grid_from_layout(height, width, track_rows, objects, "input_coords")
    expected = _grid_from_layout(height, width, track_rows, objects, "final_coords")

    moved_down = any(int(obj["input_bbox"][0]) < int(obj["final_bbox"][0]) for obj in objects)
    moved_right = any(
        int(obj["track"] is not None) and int(obj["input_bbox"][1]) < int(obj["final_bbox"][1])
        for obj in objects
    )
    if not moved_down or not moved_right:
        return None
    if not _semantic_checks(input_grid, expected, counts, bottom_count):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
