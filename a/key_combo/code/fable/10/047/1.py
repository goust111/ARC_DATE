from __future__ import annotations

import random

import numpy as np


DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
CARDINAL_DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIAGONAL_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
GEN_ATTEMPTS = 400


def _in_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _can_place_bbox(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    placed_bboxes: list[tuple[int, int, int, int]],
    protected_bg: np.ndarray,
) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
        return False
    if np.any(protected_bg[r0 : r1 + 1, c0 : c1 + 1]):
        return False
    return all(not _bbox_overlap(bbox, other, margin=1) for other in placed_bboxes)


def _draw_square(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, hole: tuple[int, int] | None = None) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    if hole is not None:
        grid[hole] = 0


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        side = int(r1 - r0 + 1)
        assert side == int(c1 - c0 + 1), "every object must be a square"
        assert side >= 3 and side % 2 == 1, "every square must have odd side length"
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == color
        assert np.all(mask | (sub == 0)), "square bbox may only contain its color or a black hole"
        missing = np.argwhere(~mask)
        assert len(missing) in (0, 1), "each square is solid or has exactly one 1x1 hole"

        center_local = side // 2
        hole = None
        direction = None
        if len(missing) == 1:
            hr, hc = map(int, missing[0])
            assert 0 < hr < side - 1 and 0 < hc < side - 1, "hole must stay inside the square"
            dr = int(np.sign(hr - center_local))
            dc = int(np.sign(hc - center_local))
            assert (dr, dc) != (0, 0), "hole may not sit at the geometric center"
            hole = (int(r0 + hr), int(c0 + hc))
            direction = (dr, dc)

        expected_area = side * side - (1 if hole is not None else 0)
        assert int(coords.shape[0]) == expected_area
        objects.append(
            {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "side": side,
                "center": (int(r0 + center_local), int(c0 + center_local)),
                "hole": hole,
                "direction": direction,
            }
        )
    assert len(objects) >= 2
    assert any(obj["hole"] is not None for obj in objects), "expected at least one hole-bearing square"
    return objects


def _ray_hit(grid: np.ndarray, obj: dict) -> tuple[int, int, int]:
    hole = obj["hole"]
    direction = obj["direction"]
    assert hole is not None and direction is not None
    h, w = grid.shape
    dr, dc = direction
    r, c = hole
    while True:
        r += dr
        c += dc
        assert 0 <= r < h and 0 <= c < w, "every emitter must eventually hit another square"
        if _in_bbox(r, c, obj["bbox"]):
            continue
        if grid[r, c] != 0:
            return int(r), int(c), int(grid[r, c])


def main(grid: np.ndarray) -> np.ndarray:
    """Use each off-center hole as a compass, recoloring the first struck square pixel and back-filling the hole with the struck color.

    Input structure:
    - A black background contains several isolated monochrome odd-sized squares, all in different colors.
    - Some squares have exactly one internal 1x1 black hole, and the hole is offset from the square's geometric center in one of eight compass directions.
    - Other squares are solid and act as targets or distractions.

    Rule:
    - For each square with a hole, take the hole's direction relative to the square center as the ray direction.
    - Start at the hole, advance along that direction through the source square and beyond it, and find the first non-black pixel belonging to another square.
    - Recolor that struck pixel to the source square's color, and fill the source hole with the struck square's original color.
    - The ray leaves no trail, solid squares do not emit, and every other pixel stays unchanged.

    Key insight:
    - The hole is not damage to repair; its off-center placement is a compass that selects both where to shoot and which color returns in the handshake.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    emitters = [obj for obj in objects if obj["hole"] is not None]

    handshakes = []
    seen_hits: set[tuple[int, int]] = set()
    for obj in emitters:
        hit_r, hit_c, target_color = _ray_hit(g, obj)
        assert target_color != obj["color"], "ray must hit a different square"
        hit = (hit_r, hit_c)
        assert hit not in seen_hits, "hit cells must be uniquely determined"
        seen_hits.add(hit)
        handshakes.append((obj["color"], obj["hole"], hit, target_color))

    out = g.copy()
    for source_color, hole, hit, target_color in handshakes:
        out[hit] = source_color
        out[hole] = target_color
    return out


def _first_outside_step(hole: tuple[int, int], bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    r, c = hole
    step = 1
    while _in_bbox(r + step * dr, c + step * dc, bbox):
        step += 1
    return step


def _target_bbox_from_hit(
    hit: tuple[int, int], side: int, direction: tuple[int, int], border_offset: int
) -> tuple[int, int, int, int]:
    hr, hc = hit
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return hr - border_offset, hc, hr - border_offset + side - 1, hc + side - 1
    if (dr, dc) == (0, -1):
        return hr - border_offset, hc - side + 1, hr - border_offset + side - 1, hc
    if (dr, dc) == (1, 0):
        return hr, hc - border_offset, hr + side - 1, hc - border_offset + side - 1
    if (dr, dc) == (-1, 0):
        return hr - side + 1, hc - border_offset, hr, hc - border_offset + side - 1
    if (dr, dc) == (1, 1):
        return hr, hc, hr + side - 1, hc + side - 1
    if (dr, dc) == (-1, -1):
        return hr - side + 1, hc - side + 1, hr, hc
    if (dr, dc) == (-1, 1):
        return hr - side + 1, hc, hr, hc + side - 1
    if (dr, dc) == (1, -1):
        return hr, hc - side + 1, hr + side - 1, hc
    raise AssertionError("invalid direction")


def _cell_on_ray(origin: tuple[int, int], direction: tuple[int, int], cell: tuple[int, int]) -> bool:
    rr = cell[0] - origin[0]
    cc = cell[1] - origin[1]
    dr, dc = direction
    if dr == 0:
        return rr == 0 and cc * dc > 0
    if dc == 0:
        return cc == 0 and rr * dr > 0
    return rr * dc == cc * dr and rr * dr > 0 and cc * dc > 0


def _bbox_on_ray(origin: tuple[int, int], direction: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    return any(_cell_on_ray(origin, direction, cell) for cell in _bbox_cells(bbox))


def _bbox_min_distances(origin: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    d_cheb = min(max(abs(r - origin[0]), abs(c - origin[1])) for r, c in _bbox_cells(bbox))
    d_manhattan = min(abs(r - origin[0]) + abs(c - origin[1]) for r, c in _bbox_cells(bbox))
    return int(d_cheb), int(d_manhattan)


def _sample_emitter_directions(count: int) -> list[tuple[int, int]]:
    assert count in (2, 3)
    choices: list[tuple[int, int]] = [random.choice(CARDINAL_DIRS), random.choice(DIAGONAL_DIRS)]
    used = set(choices)
    while len(choices) < count:
        direction = random.choice(DIRS8)
        if direction not in used:
            choices.append(direction)
            used.add(direction)
    random.shuffle(choices)
    return choices


def _choose_decoy_bbox(
    shape: tuple[int, int],
    hole: tuple[int, int],
    target_step: int,
    source_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    placed_bboxes: list[tuple[int, int, int, int]],
    protected_bg: np.ndarray,
) -> tuple[int, int, int, int] | None:
    candidates = [d for d in DIRS8 if d != direction]
    random.shuffle(candidates)
    side_choices = [3, 5] if target_step >= 5 else [3]
    for dec_dir in candidates:
        distances = list(range(2, max(3, min(6, target_step))))
        random.shuffle(distances)
        for side in side_choices:
            radius = side // 2
            for dist in distances:
                cr = hole[0] + dec_dir[0] * dist
                cc = hole[1] + dec_dir[1] * dist
                bbox = (cr - radius, cc - radius, cr + radius, cc + radius)
                merged = placed_bboxes + [source_bbox, target_bbox]
                if not _can_place_bbox(shape, bbox, merged, protected_bg):
                    continue
                if _bbox_on_ray(hole, direction, bbox):
                    continue
                cheb, manhattan = _bbox_min_distances(hole, bbox)
                hit_manhattan = abs(direction[0]) * target_step + abs(direction[1]) * target_step
                if direction[0] != 0 and direction[1] != 0:
                    hit_manhattan = 2 * target_step
                if cheb >= target_step or manhattan >= hit_manhattan:
                    continue
                return bbox
    return None


def _place_emitter_triplet(
    shape: tuple[int, int],
    placed_bboxes: list[tuple[int, int, int, int]],
    protected_bg: np.ndarray,
    direction: tuple[int, int],
) -> dict | None:
    h, w = shape
    for _ in range(250):
        source_side = random.choice([3, 5, 7])
        target_side = random.choice([3, 5, 7])
        source_radius = source_side // 2
        hole_dist = random.randint(1, max(1, source_radius - 1))
        gap = random.randint(1, 6)
        source_r0 = random.randint(0, h - source_side)
        source_c0 = random.randint(0, w - source_side)
        source_bbox = (source_r0, source_c0, source_r0 + source_side - 1, source_c0 + source_side - 1)
        if not _can_place_bbox(shape, source_bbox, placed_bboxes, protected_bg):
            continue

        center = (source_r0 + source_radius, source_c0 + source_radius)
        hole = (center[0] + direction[0] * hole_dist, center[1] + direction[1] * hole_dist)
        if not (source_r0 < hole[0] < source_bbox[2] and source_c0 < hole[1] < source_bbox[3]):
            continue

        outside_step = _first_outside_step(hole, source_bbox, direction)
        hit_step = outside_step + gap
        hit = (hole[0] + direction[0] * hit_step, hole[1] + direction[1] * hit_step)
        if not (0 <= hit[0] < h and 0 <= hit[1] < w):
            continue

        border_offset = random.randint(0, target_side - 1) if direction in CARDINAL_DIRS else 0
        target_bbox = _target_bbox_from_hit(hit, target_side, direction, border_offset)
        if not _can_place_bbox(shape, target_bbox, placed_bboxes + [source_bbox], protected_bg):
            continue
        if _bbox_overlap(source_bbox, target_bbox, margin=1):
            continue

        path_ok = True
        for step in range(1, hit_step):
            r = hole[0] + direction[0] * step
            c = hole[1] + direction[1] * step
            if not (0 <= r < h and 0 <= c < w):
                path_ok = False
                break
            if _in_bbox(r, c, source_bbox):
                continue
            if _in_bbox(r, c, target_bbox):
                path_ok = False
                break
            if any(_in_bbox(r, c, bbox) for bbox in placed_bboxes):
                path_ok = False
                break
        if not path_ok:
            continue

        decoy_bbox = _choose_decoy_bbox(
            shape, hole, hit_step, source_bbox, target_bbox, direction, placed_bboxes, protected_bg
        )
        if decoy_bbox is None:
            continue

        corridor = []
        for step in range(1, hit_step):
            r = hole[0] + direction[0] * step
            c = hole[1] + direction[1] * step
            if not _in_bbox(r, c, source_bbox):
                corridor.append((r, c))

        return {
            "source_bbox": source_bbox,
            "target_bbox": target_bbox,
            "decoy_bbox": decoy_bbox,
            "hole": hole,
            "direction": direction,
            "hit": hit,
            "corridor": corridor,
        }
    return None


def _place_extra_solid(
    shape: tuple[int, int],
    placed_bboxes: list[tuple[int, int, int, int]],
    protected_bg: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    for _ in range(120):
        side = random.choice([3, 5, 7])
        r0 = random.randint(0, h - side)
        c0 = random.randint(0, w - side)
        bbox = (r0, c0, r0 + side - 1, c0 + side - 1)
        if _can_place_bbox(shape, bbox, placed_bboxes, protected_bg):
            return bbox
    return None


def _render_scene(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        _draw_square(grid, obj["bbox"], obj["color"], obj.get("hole"))
    return grid


def _semantic_check(
    inp: np.ndarray,
    out: np.ndarray,
    objects: list[dict],
    emitter_records: list[dict],
    directions: list[tuple[int, int]],
) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    if np.array_equal(inp, out):
        return False

    parsed = _extract_objects(inp)
    parsed_emitters = [obj for obj in parsed if obj["hole"] is not None]
    if len(parsed_emitters) != len(emitter_records):
        return False
    if not any(direction in CARDINAL_DIRS for direction in directions):
        return False
    if not any(direction in DIAGONAL_DIRS for direction in directions):
        return False

    diff_count = int(np.count_nonzero(inp != out))
    if diff_count != 2 * len(emitter_records):
        return False

    hit_cells = set()
    for record in emitter_records:
        source = objects[record["source_id"]]
        target = objects[record["target_id"]]
        decoy = objects[record["decoy_id"]]
        hit = record["hit"]
        if hit in hit_cells:
            return False
        hit_cells.add(hit)
        if int(out[source["hole"]]) != target["color"]:
            return False
        if int(out[hit]) != source["color"]:
            return False
        if int(inp[hit]) != target["color"]:
            return False

        cheb_target, man_target = _bbox_min_distances(source["hole"], target["bbox"])
        cheb_decoy, man_decoy = _bbox_min_distances(source["hole"], decoy["bbox"])
        if not (cheb_decoy < cheb_target and man_decoy < man_target):
            return False
        if _bbox_on_ray(source["hole"], source["direction"], decoy["bbox"]):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 29)
    w = random.randint(18, 29)
    shape = (h, w)

    emitter_count = random.choice([2, 3])
    directions = _sample_emitter_directions(emitter_count)
    extra_count = 0 if emitter_count == 3 else random.choice([0, 1])

    placed_bboxes: list[tuple[int, int, int, int]] = []
    protected_bg = np.zeros(shape, dtype=bool)
    objects: list[dict] = []
    emitter_records: list[dict] = []

    for direction in directions:
        placed = _place_emitter_triplet(shape, placed_bboxes, protected_bg, direction)
        if placed is None:
            return None

        source_id = len(objects)
        objects.append({"bbox": placed["source_bbox"], "hole": placed["hole"], "direction": direction})
        target_id = len(objects)
        objects.append({"bbox": placed["target_bbox"]})
        decoy_id = len(objects)
        objects.append({"bbox": placed["decoy_bbox"]})
        emitter_records.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "decoy_id": decoy_id,
                "hit": placed["hit"],
            }
        )

        placed_bboxes.extend([placed["source_bbox"], placed["target_bbox"], placed["decoy_bbox"]])
        for cell in placed["corridor"]:
            protected_bg[cell] = True

    for _ in range(extra_count):
        bbox = _place_extra_solid(shape, placed_bboxes, protected_bg)
        if bbox is None:
            return None
        objects.append({"bbox": bbox})
        placed_bboxes.append(bbox)

    if len(objects) > 9:
        return None

    palette = random.sample(list(range(1, 10)), len(objects))
    for obj, color in zip(objects, palette):
        obj["color"] = int(color)

    inp = _render_scene(shape, objects)
    out = inp.copy()
    for record in emitter_records:
        source = objects[record["source_id"]]
        target = objects[record["target_id"]]
        out[source["hole"]] = target["color"]
        out[record["hit"]] = source["color"]

    if not _semantic_check(inp, out, objects, emitter_records, directions):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-compass handshake puzzle")
