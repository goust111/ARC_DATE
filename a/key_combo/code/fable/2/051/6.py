from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BG = 0
MARKER = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

GENERATION_RETRIES = 500
ANGLE_EPS = 0.16
MARKER_SIDE_EPS = 0.14
_GEN_CALLS = 0
WEIGHTED_TEMPLATE_OFFSETS = np.array(
    [
        (7.0, 0.0),
        (4.33, 2.5),
        (4.0, 6.93),
        (0.0, 8.0),
        (-4.33, -2.5),
    ],
    dtype=float,
)
WEIGHTED_TEMPLATE_AREAS = (4, 7, 8, 5, 3)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield int(r + dr), int(c + dc)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _clockwise_angle(point: np.ndarray, center: np.ndarray) -> float:
    dr = float(point[0] - center[0])
    dc = float(point[1] - center[1])
    angle = math.atan2(dc, -dr)
    if angle < 0:
        angle += 2.0 * math.pi
    return angle


def _signed_cw_delta(base: float, target: float) -> float:
    delta = (float(target) - float(base)) % (2.0 * math.pi)
    if delta > math.pi:
        delta -= 2.0 * math.pi
    return delta


def _global_centroid(objects: list[dict]) -> np.ndarray:
    coords = np.concatenate([obj["coords"] for obj in objects], axis=0)
    return coords.astype(float).mean(axis=0)


def _collect_objects(grid: np.ndarray, bg: int) -> list[dict]:
    colors = sorted(set(map(int, np.unique(grid))) - {int(bg), MARKER})
    objects: list[dict] = []
    for color in colors:
        comps = components_mask(grid == int(color))
        assert len(comps) == 1, "each non-marker color must form exactly one 4-connected object"
        coords = comps[0]
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "coord_set": {tuple(map(int, rc)) for rc in coords},
                "area": int(coords.shape[0]),
                "centroid": coords.astype(float).mean(axis=0),
            }
        )
    assert len(objects) >= 3, "expected several colored objects"
    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas), "object areas must be unique"
    return objects


def _sort_by_angle(objects: list[dict], center: np.ndarray) -> list[dict]:
    ranked = []
    for obj in objects:
        angle = _clockwise_angle(obj["centroid"], center)
        radius = float(np.linalg.norm(obj["centroid"] - center))
        ranked.append((angle, radius, obj["color"], obj))
    ordered = [obj for _angle, _radius, _color, obj in sorted(ranked, key=lambda item: item[:3])]
    angles = [float(_clockwise_angle(obj["centroid"], center)) for obj in ordered]
    gaps = [
        (angles[(i + 1) % len(angles)] - angles[i]) % (2.0 * math.pi)
        for i in range(len(angles))
    ]
    assert min(gaps) > 1.0e-6, "object angular order must be unambiguous"
    return ordered


def _try_sort_by_angle(
    objects: list[dict],
    center: np.ndarray,
    *,
    min_gap: float = ANGLE_EPS,
) -> list[dict] | None:
    ranked = []
    for obj in objects:
        angle = _clockwise_angle(obj["centroid"], center)
        radius = float(np.linalg.norm(obj["centroid"] - center))
        ranked.append((angle, radius, obj["color"], obj))
    ordered = [obj for _angle, _radius, _color, obj in sorted(ranked, key=lambda item: item[:3])]
    angles = [float(_clockwise_angle(obj["centroid"], center)) for obj in ordered]
    gaps = [
        (angles[(i + 1) % len(angles)] - angles[i]) % (2.0 * math.pi)
        for i in range(len(angles))
    ]
    if min(gaps) <= float(min_gap):
        return None
    return ordered


def _marker_info(grid: np.ndarray, objects: list[dict], center: np.ndarray) -> tuple[int, bool]:
    marker_cells = np.argwhere(grid == MARKER)
    assert marker_cells.shape == (1, 2), "expected exactly one gray marker pixel"
    mr, mc = map(int, marker_cells[0])

    hits = []
    for idx, obj in enumerate(objects):
        if any((rr, cc) in obj["coord_set"] for rr, cc in neighbors4(mr, mc)):
            hits.append(idx)
    assert len(hits) == 1, "gray marker must touch exactly one object orthogonally"

    obj = objects[hits[0]]
    obj_angle = _clockwise_angle(obj["centroid"], center)
    marker_angle = _clockwise_angle(np.array([mr, mc], dtype=float), center)
    delta = _signed_cw_delta(obj_angle, marker_angle)
    assert abs(delta) > 1.0e-6, "gray marker must sit on a definite angular side of the start object"
    return hits[0], bool(delta > 0.0)


def _ordered_cycle(objects: list[dict], center: np.ndarray, start_idx: int, clockwise: bool) -> list[dict]:
    ordered = _sort_by_angle(objects, center)
    start_color = int(objects[start_idx]["color"])
    pos = next(i for i, obj in enumerate(ordered) if int(obj["color"]) == start_color)
    if clockwise:
        return [ordered[(pos + k) % len(ordered)] for k in range(len(ordered))]
    return [ordered[(pos - k) % len(ordered)] for k in range(len(ordered))]


def _try_ordered_cycle(
    objects: list[dict],
    center: np.ndarray,
    start_idx: int,
    clockwise: bool,
    *,
    min_gap: float = ANGLE_EPS,
) -> list[dict] | None:
    ordered = _try_sort_by_angle(objects, center, min_gap=min_gap)
    if ordered is None:
        return None
    start_color = int(objects[start_idx]["color"])
    pos = next(i for i, obj in enumerate(ordered) if int(obj["color"]) == start_color)
    if clockwise:
        return [ordered[(pos + k) % len(ordered)] for k in range(len(ordered))]
    return [ordered[(pos - k) % len(ordered)] for k in range(len(ordered))]


def _barcode(rows: list[dict]) -> np.ndarray:
    widths = [int(obj["area"]) for obj in rows]
    out = np.zeros((len(rows), max(widths)), dtype=int)
    for r, obj in enumerate(rows):
        out[r, : int(obj["area"])] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn scattered monochrome objects into a barcode ordered by angular position around their invisible shared centroid.

    Input structure:
    - A dominant black background contains several separated monochrome 4-connected objects in distinct non-gray colors.
    - The objects have pairwise different pixel counts.
    - Exactly one gray pixel lies on a background cell and touches exactly one object orthogonally.

    Rule:
    - Ignore the gray pixel when computing geometry.
    - Compute the centroid of all colored object pixels together, then compute each object's own centroid and sort the
      objects by clockwise angle around that invisible shared centroid.
    - The gray-adjacent object is the first row. Compare the gray pixel's angle to that start object's angle around the
      shared centroid: if the gray pixel lies on the clockwise side, continue clockwise through the ring; otherwise
      continue counterclockwise.
    - Output one row per object, with that row filled from the left by the object's color for exactly its pixel count;
      remaining cells stay black.

    Key insight:
    - Neither reading order nor size order matters: the hidden global centroid defines the ring, while the single gray
      pixel simultaneously chooses the start object and which way to walk around that ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a nonempty 2D grid"
    bg = get_bg(g)
    objects = _collect_objects(g, bg)
    center = _global_centroid(objects)
    start_idx, clockwise = _marker_info(g, objects, center)
    return _barcode(_ordered_cycle(objects, center, start_idx, clockwise))


def _normalize(coords: np.ndarray) -> np.ndarray:
    out = np.array(coords, dtype=int)
    out[:, 0] -= out[:, 0].min()
    out[:, 1] -= out[:, 1].min()
    return out


def _random_polyomino(area: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        h = rng.randint(2, min(5, area))
        w = rng.randint(2, min(5, area))
        if h * w < area:
            continue
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue
        coords = _normalize(np.array(sorted(cells), dtype=int))
        if len(set(map(int, coords[:, 0]))) < 2 or len(set(map(int, coords[:, 1]))) < 2:
            continue
        return coords
    return None


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray, pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _coords_fit(coords: np.ndarray, shape: tuple[int, int], blocked: np.ndarray) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    if rows.min() < 1 or cols.min() < 1 or rows.max() >= h - 1 or cols.max() >= w - 1:
        return False
    return not bool(np.any(blocked[rows, cols]))


def _sample_angles(count: int, rng: random.Random) -> list[float] | None:
    for _ in range(120):
        angles = sorted(rng.random() * 2.0 * math.pi for _ in range(count))
        gaps = [(angles[(i + 1) % count] - angles[i]) % (2.0 * math.pi) for i in range(count)]
        if min(gaps) < 0.34:
            continue
        if max(gaps) > 2.8:
            continue
        if not any(gap < 0.78 for gap in gaps):
            continue
        return angles
    return None


def _place_objects(
    shape: tuple[int, int],
    colors: list[int],
    areas: list[int],
    rng: random.Random,
) -> list[dict] | None:
    h, w = shape
    center = np.array([h / 2.0 + rng.uniform(-1.0, 1.0), w / 2.0 + rng.uniform(-1.0, 1.0)], dtype=float)
    max_radius = int(min(center[0] - 3, h - center[0] - 4, center[1] - 3, w - center[1] - 4))
    if max_radius < 6:
        return None

    angles = _sample_angles(len(colors), rng)
    if angles is None:
        return None

    specs = []
    for color, area, angle in zip(colors, areas, angles):
        poly = _random_polyomino(int(area), rng)
        if poly is None:
            return None
        specs.append({"color": int(color), "area": int(area), "angle": float(angle), "poly": poly})

    blocked = np.zeros(shape, dtype=bool)
    placed: list[dict | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["area"], reverse=True)
    jitter = [(0, 0)]
    for span in (1, 2, 3):
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                if (dr, dc) != (0, 0):
                    jitter.append((dr, dc))
    rng.shuffle(jitter[1:])

    for idx in order:
        spec = specs[idx]
        local = spec["poly"]
        local_center = local.astype(float).mean(axis=0)
        radii = list(range(max(5, max_radius - 3), max_radius + 1))
        rng.shuffle(radii)
        placed_here = None
        for radius in radii:
            target = np.array(
                [
                    center[0] - radius * math.cos(spec["angle"]),
                    center[1] + radius * math.sin(spec["angle"]),
                ],
                dtype=float,
            )
            base = np.rint(target - local_center).astype(int)
            for dr, dc in jitter:
                coords = local + base + np.array([dr, dc], dtype=int)
                if not _coords_fit(coords, shape, blocked):
                    continue
                placed_here = coords
                break
            if placed_here is not None:
                break
        if placed_here is None:
            return None

        _mark_blocked(blocked, placed_here, pad=1)
        placed[idx] = {
            "color": int(spec["color"]),
            "area": int(spec["area"]),
            "coords": placed_here.astype(int),
            "coord_set": {tuple(map(int, rc)) for rc in placed_here},
            "centroid": placed_here.astype(float).mean(axis=0),
        }

    return [obj for obj in placed if obj is not None]


def _transform_offsets(offsets: np.ndarray, mode: int) -> np.ndarray:
    pts = np.array(offsets, dtype=float)
    if mode >= 4:
        pts[:, 1] *= -1.0
        mode -= 4
    for _ in range(mode):
        pts = np.column_stack([pts[:, 1], -pts[:, 0]])
    return pts


def _place_weighted_template_objects(shape: tuple[int, int], colors: list[int], rng: random.Random) -> list[dict] | None:
    offsets = _transform_offsets(WEIGHTED_TEMPLATE_OFFSETS, rng.randrange(8))
    anchor = np.array([shape[0] / 2.0 + rng.uniform(-0.6, 0.6), shape[1] / 2.0 + rng.uniform(-0.6, 0.6)], dtype=float)
    specs = []
    for color, area, offset in zip(colors, WEIGHTED_TEMPLATE_AREAS, offsets):
        poly = _random_polyomino(int(area), rng)
        if poly is None:
            return None
        specs.append({"color": int(color), "area": int(area), "offset": np.array(offset, dtype=float), "poly": poly})

    blocked = np.zeros(shape, dtype=bool)
    placed: list[dict] = []
    for spec in sorted(specs, key=lambda item: item["area"], reverse=True):
        local = spec["poly"]
        local_center = local.astype(float).mean(axis=0)
        target = anchor + spec["offset"]
        base = np.rint(target - local_center).astype(int)
        options = [(0, 0)]
        for span in (1, 2):
            for dr in range(-span, span + 1):
                for dc in range(-span, span + 1):
                    if (dr, dc) != (0, 0):
                        options.append((dr, dc))
        rng.shuffle(options[1:])

        coords_here = None
        for dr, dc in options:
            coords = local + base + np.array([dr, dc], dtype=int)
            if _coords_fit(coords, shape, blocked):
                coords_here = coords
                break
        if coords_here is None:
            return None

        _mark_blocked(blocked, coords_here, pad=1)
        placed.append(
            {
                "color": int(spec["color"]),
                "area": int(spec["area"]),
                "coords": coords_here.astype(int),
                "coord_set": {tuple(map(int, rc)) for rc in coords_here},
                "centroid": coords_here.astype(float).mean(axis=0),
            }
        )

    objects = sorted(placed, key=lambda obj: int(obj["color"]))
    center = _global_centroid(objects)
    obj_center = np.mean([obj["centroid"] for obj in objects], axis=0)
    true_cycle = _try_sort_by_angle(objects, center, min_gap=ANGLE_EPS)
    wrong_cycle = _try_sort_by_angle(objects, obj_center, min_gap=ANGLE_EPS)
    if true_cycle is None or wrong_cycle is None:
        return None
    order_true = [int(obj["color"]) for obj in true_cycle]
    order_wrong = [int(obj["color"]) for obj in wrong_cycle]
    if order_true == order_wrong:
        return None
    return objects


def _candidate_marker_cells(shape: tuple[int, int], objects: list[dict], center: np.ndarray) -> list[dict]:
    h, w = shape
    occupied = set()
    for obj in objects:
        occupied |= obj["coord_set"]

    candidates: list[dict] = []
    for idx, obj in enumerate(objects):
        seen = set()
        obj_angle = _clockwise_angle(obj["centroid"], center)
        obj_radius = float(np.linalg.norm(obj["centroid"] - center))
        tangent = np.array([math.sin(obj_angle), math.cos(obj_angle)], dtype=float)
        radial = obj["centroid"] - center
        for r, c in obj["coord_set"]:
            for rr, cc in neighbors4(r, c):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in occupied or (rr, cc) in seen:
                    continue
                seen.add((rr, cc))
                touching = 0
                for other in objects:
                    if any((nr, nc) in other["coord_set"] for nr, nc in neighbors4(rr, cc)):
                        touching += 1
                if touching != 1:
                    continue
                marker_pt = np.array([rr, cc], dtype=float)
                marker_angle = _clockwise_angle(marker_pt, center)
                delta = _signed_cw_delta(obj_angle, marker_angle)
                if abs(delta) < MARKER_SIDE_EPS:
                    continue
                marker_radius = float(np.linalg.norm(marker_pt - center))
                side_vec = marker_pt - obj["centroid"]
                if abs(marker_radius - obj_radius) > 1.35:
                    continue
                if abs(float(np.dot(side_vec, tangent))) < abs(float(np.dot(side_vec, radial))):
                    continue
                candidates.append(
                    {
                        "obj_idx": idx,
                        "cell": (int(rr), int(cc)),
                        "clockwise": bool(delta > 0.0),
                        "strength": abs(float(delta)),
                    }
                )
    return candidates


def _barcode_from_sequence(seq: list[dict]) -> np.ndarray:
    return _barcode(seq)


def _rotated_sequence(base: list[dict], start_color: int, forward: bool) -> list[dict]:
    seq = list(base)
    if not forward:
        seq = list(reversed(seq))
    pos = next(i for i, obj in enumerate(seq) if int(obj["color"]) == int(start_color))
    return seq[pos:] + seq[:pos]


def _semantic_checks(
    grid: np.ndarray,
    objects: list[dict],
    start_idx: int,
    clockwise: bool,
    require_wrong_center_diff: bool,
) -> bool:
    center = _global_centroid(objects)
    sorted_true = _try_sort_by_angle(objects, center, min_gap=ANGLE_EPS)
    if sorted_true is None:
        return False
    true_rows = _try_ordered_cycle(objects, center, start_idx, clockwise, min_gap=ANGLE_EPS)
    if true_rows is None:
        return False
    expected = _barcode_from_sequence(true_rows)

    rounded = np.rint(center).astype(int)
    if not (0 <= rounded[0] < grid.shape[0] and 0 <= rounded[1] < grid.shape[1]):
        return False
    if grid[int(rounded[0]), int(rounded[1])] != BG:
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    start_color = int(objects[start_idx]["color"])
    obj_center = np.mean([obj["centroid"] for obj in objects], axis=0)
    if require_wrong_center_diff:
        wrong_center_rows = _try_ordered_cycle(objects, obj_center, start_idx, clockwise, min_gap=ANGLE_EPS)
        if wrong_center_rows is None:
            return False
        if np.array_equal(_barcode_from_sequence(wrong_center_rows), expected):
            return False

    area_base = sorted(objects, key=lambda obj: (int(obj["area"]), int(obj["color"])))
    if np.array_equal(_barcode_from_sequence(_rotated_sequence(area_base, start_color, True)), expected):
        return False
    if np.array_equal(_barcode_from_sequence(_rotated_sequence(area_base, start_color, False)), expected):
        return False

    reading_base = sorted(objects, key=lambda obj: (float(obj["centroid"][0]), float(obj["centroid"][1])))
    if np.array_equal(_barcode_from_sequence(_rotated_sequence(reading_base, start_color, True)), expected):
        return False
    if np.array_equal(_barcode_from_sequence(_rotated_sequence(reading_base, start_color, False)), expected):
        return False

    opposite_rows = _try_ordered_cycle(objects, center, start_idx, not clockwise, min_gap=ANGLE_EPS)
    if opposite_rows is None:
        return False
    if np.array_equal(_barcode_from_sequence(opposite_rows), expected):
        return False

    pos = next(i for i, obj in enumerate(sorted_true) if int(obj["color"]) == start_color)
    shifted_start_color = int(sorted_true[(pos + 1) % len(sorted_true)]["color"])
    shifted_idx = next(i for i, obj in enumerate(objects) if int(obj["color"]) == shifted_start_color)
    shifted_rows = _try_ordered_cycle(objects, center, shifted_idx, clockwise, min_gap=ANGLE_EPS)
    if shifted_rows is None:
        return False
    if np.array_equal(_barcode_from_sequence(shifted_rows), expected):
        return False

    return True


def _gen_attempt(required_clockwise: bool, require_wrong_center_diff: bool, rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(19, 27), rng.randint(19, 27))
    obj_count = len(WEIGHTED_TEMPLATE_AREAS) if require_wrong_center_diff else rng.randint(4, 6)
    colors = rng.sample(list(OBJECT_COLORS), obj_count)
    if require_wrong_center_diff:
        objects = _place_weighted_template_objects(shape, colors, rng)
    else:
        areas = rng.sample(list(range(3, 10)), obj_count)
        objects = _place_objects(shape, colors, areas, rng)
    if objects is None:
        return None

    center = _global_centroid(objects)
    marker_candidates = [c for c in _candidate_marker_cells(shape, objects, center) if c["clockwise"] == required_clockwise]
    if not marker_candidates:
        return None

    marker_candidates.sort(key=lambda item: item["strength"], reverse=True)
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        coords = obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(obj["color"])

    for marker in marker_candidates:
        trial_grid = grid.copy()
        mr, mc = marker["cell"]
        trial_grid[mr, mc] = MARKER
        start_idx = int(marker["obj_idx"])
        if _semantic_checks(trial_grid, objects, start_idx, required_clockwise, require_wrong_center_diff):
            return trial_grid
    return None


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    _GEN_CALLS += 1
    required_clockwise = bool(_GEN_CALLS % 2)
    require_wrong_center_diff = (_GEN_CALLS % 4 == 1)
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)

    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(required_clockwise, require_wrong_center_diff, rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate angular-barcode input")
