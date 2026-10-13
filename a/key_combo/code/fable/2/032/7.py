from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            box = bbox_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def mask_bytes(mask: np.ndarray) -> bytes:
    return np.ascontiguousarray(mask.astype(np.uint8)).tobytes()


def canonical_mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = [
        mask,
        np.fliplr(mask),
        np.flipud(mask),
        np.flipud(np.fliplr(mask)),
    ]
    return mask.shape, min(mask_bytes(v) for v in variants)


def bbox_center2(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return int(r0 + r1), int(c0 + c1)


def bbox_from_center2(center_r2: int, center_c2: int, shape: tuple[int, int]) -> BBox | None:
    h, w = map(int, shape)
    r_num = int(center_r2 - (h - 1))
    c_num = int(center_c2 - (w - 1))
    if r_num % 2 != 0 or c_num % 2 != 0:
        return None
    r0 = r_num // 2
    c0 = c_num // 2
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def quadrant_of(center_r2: int, center_c2: int, axis_r2: int, axis_c2: int) -> tuple[int, int] | None:
    if int(center_r2) == int(axis_r2) or int(center_c2) == int(axis_c2):
        return None
    return (
        0 if int(center_r2) < int(axis_r2) else 1,
        0 if int(center_c2) < int(axis_c2) else 1,
    )


def transform_mask(mask: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = mask
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return out


def orbit_centers(center_r2: int, center_c2: int, axis_r2: int, axis_c2: int) -> tuple[tuple[int, int], ...]:
    rows = sorted({int(center_r2), int(2 * axis_r2 - center_r2)})
    cols = sorted({int(center_c2), int(2 * axis_c2 - center_c2)})
    return tuple(sorted((int(r2), int(c2)) for r2 in rows for c2 in cols))


def _component_records(grid: np.ndarray) -> tuple[int, list[dict], dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    records: list[dict] = []
    lookup: dict[tuple[int, int, int, int, bytes], int] = {}
    for comp in find_components(g, bg):
        box = comp["bbox"]
        crop = crop_bbox(g, box)
        mask = crop != int(bg)
        h, w = mask.shape
        center_r2, center_c2 = bbox_center2(box)
        rec = {
            "color": int(comp["color"]),
            "bbox": box,
            "mask": mask,
            "shape": (int(h), int(w)),
            "center_r2": int(center_r2),
            "center_c2": int(center_c2),
        }
        idx = len(records)
        records.append(rec)
        key = (int(box[0]), int(box[1]), int(h), int(w), mask_bytes(mask))
        lookup[key] = idx
    return bg, records, lookup


def _candidate_axes(records: list[dict], grid_shape: tuple[int, int]) -> tuple[list[int], list[int]]:
    h, w = map(int, grid_shape)
    row_cands: set[int] = set()
    col_cands: set[int] = set()
    for i, a in enumerate(records):
        ah, aw = a["shape"]
        am = a["mask"]
        for j in range(i + 1, len(records)):
            b = records[j]
            if b["shape"] != (ah, aw):
                continue
            if int(a["center_r2"]) == int(b["center_r2"]) and np.array_equal(np.fliplr(am), b["mask"]):
                col_cands.add((int(a["center_c2"]) + int(b["center_c2"])) // 2)
            if int(a["center_c2"]) == int(b["center_c2"]) and np.array_equal(np.flipud(am), b["mask"]):
                row_cands.add((int(a["center_r2"]) + int(b["center_r2"])) // 2)
    if not row_cands:
        row_cands.update(range(0, 2 * h - 1))
    if not col_cands:
        col_cands.update(range(0, 2 * w - 1))
    return sorted(row_cands), sorted(col_cands)


def _evaluate_axes(
    grid: np.ndarray,
    bg: int,
    records: list[dict],
    lookup: dict[tuple[int, int, int, int, bytes], int],
    axis_r2: int,
    axis_c2: int,
) -> dict | None:
    h, w = grid.shape

    quadrants: dict[int, tuple[int, int] | None] = {}
    qcolor: dict[tuple[int, int], int] = {}
    for idx, rec in enumerate(records):
        q = quadrant_of(rec["center_r2"], rec["center_c2"], axis_r2, axis_c2)
        if q is None:
            return None
        quadrants[idx] = q
        if q in qcolor and int(qcolor[q]) != int(rec["color"]):
            return None
        qcolor[q] = int(rec["color"])
    if len(qcolor) != 4:
        return None

    seen_orbits: set[tuple[tuple[int, int], ...]] = set()
    covered: set[int] = set()
    missing_specs: list[dict] = []
    pair_score = 0

    for idx, rec in enumerate(records):
        centers = orbit_centers(rec["center_r2"], rec["center_c2"], axis_r2, axis_c2)
        if len(centers) != 4:
            return None
        if centers in seen_orbits:
            continue
        seen_orbits.add(centers)

        base_q = quadrants[idx]
        assert base_q is not None
        hm, wm = rec["shape"]
        base_mask = rec["mask"]

        present: list[int] = []
        missing: list[dict] = []
        for target_r2, target_c2 in centers:
            q = quadrant_of(target_r2, target_c2, axis_r2, axis_c2)
            if q is None:
                return None
            box = bbox_from_center2(target_r2, target_c2, rec["shape"])
            if box is None:
                return None
            r0, c0, r1, c1 = map(int, box)
            if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                return None
            mask = transform_mask(base_mask, q[0] != base_q[0], q[1] != base_q[1])
            key = (int(r0), int(c0), int(hm), int(wm), mask_bytes(mask))
            found = lookup.get(key)
            if found is None:
                missing.append({"bbox": box, "mask": mask, "quadrant": q})
                continue
            if quadrants[found] != q:
                return None
            if int(records[found]["color"]) != int(qcolor[q]):
                return None
            present.append(found)

        if len(missing) > 1 or len(set(present)) != 4 - len(missing):
            return None
        if len(missing) == 1:
            missing_specs.extend(missing)
        pair_score += len(present) - 1
        covered.update(present)

    if len(covered) != len(records):
        return None
    if len(missing_specs) != 1:
        return None

    spec = missing_specs[0]
    missing_q = spec["quadrant"]
    if missing_q not in qcolor:
        return None
    return {
        "axis_r2": int(axis_r2),
        "axis_c2": int(axis_c2),
        "quadrant_colors": qcolor,
        "missing": spec,
        "score": int(pair_score),
    }


def _solution_outputs(grid: np.ndarray) -> list[tuple[dict, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg, records, lookup = _component_records(g)
    if not records:
        return []
    row_cands, col_cands = _candidate_axes(records, g.shape)
    solutions: list[tuple[dict, np.ndarray]] = []
    for axis_r2 in row_cands:
        for axis_c2 in col_cands:
            sol = _evaluate_axes(g, bg, records, lookup, int(axis_r2), int(axis_c2))
            if sol is None:
                continue
            out = g.copy()
            box = sol["missing"]["bbox"]
            mask = sol["missing"]["mask"]
            color = int(sol["quadrant_colors"][sol["missing"]["quadrant"]])
            r0, c0, r1, c1 = map(int, box)
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            if np.any((patch != int(bg)) & mask):
                continue
            patch[mask] = int(color)
            solutions.append((sol, out))
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the only missing member of an implicit two-axis mirror orbit and color it by quadrant.

    Input structure:
    - The background is the dominant color, and every foreground component is a small monochrome sprite.
    - All sprites belong to one global horizontal reflection axis and one global vertical reflection axis; the axes are not drawn and need not pass through the grid center.
    - Within each four-member orbit, the quadrant members are geometric mirrors of one another, while the color of each member depends only on its quadrant, shared across all orbits.
    - Exactly one orbit is missing one quadrant member.

    Rule:
    - Infer the hidden horizontal and vertical axes from the repeated mirror relationships between existing sprites.
    - Use the surviving members of the incomplete orbit to determine the missing quadrant, its mirrored bbox position, and the reflected sprite shape.
    - Infer the quadrant-color mapping from the complete orbits, then paint the missing mirrored sprite with that quadrant's color.
    - Leave every existing cell unchanged.

    Key insight:
    - Position, shape, and color come from three separate constraints: the hidden axes locate the orbit, mirror transforms recover the missing geometry, and the complete orbits reveal the quadrant color law.
    """
    solutions = _solution_outputs(np.asarray(grid, dtype=int))
    assert solutions, "expected at least one valid axis/orbit completion"

    best_score = max(sol["score"] for sol, _out in solutions)
    best = [(sol, out) for sol, out in solutions if int(sol["score"]) == int(best_score)]
    sigs = {
        (tuple(out.shape), out.tobytes())
        for _sol, out in best
    }
    assert len(sigs) == 1, "solution should be uniquely determined"
    return best[0][1]


GRID_MIN = 18
GRID_MAX = 28
ORBIT_COUNT_CHOICES = (3, 4, 5)
SHAPE_H_RANGE = (2, 4)
SHAPE_W_RANGE = (2, 4)
AREA_MARGIN = 1
MAX_SHAPE_TRIES = 160
MAX_INSTANCE_TRIES = 320
PADDING = 1


def _random_connected_shape(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    if len(cells) != area:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _shape_spans_bbox(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    r0, c0, r1, c1 = bbox_coords(pts)
    h, w = mask.shape
    return (int(r0), int(c0), int(r1), int(c1)) == (0, 0, h - 1, w - 1)


def _choose_unique_shape(used: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_TRIES):
        h = random.randint(*SHAPE_H_RANGE)
        w = random.randint(*SHAPE_W_RANGE)
        area_lo = max(3, h + w - 1)
        area_hi = max(area_lo, min(h * w - AREA_MARGIN, h * w - 1))
        area = random.randint(area_lo, area_hi)
        mask = _random_connected_shape(h, w, area)
        if mask is None or not _shape_spans_bbox(mask):
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        key = canonical_mask_key(mask)
        if key in used:
            continue
        used.add(key)
        return mask
    return None


def _mirror_bbox_row(r0: int, h: int, axis_r2: int) -> int:
    return int(axis_r2 - (r0 + h - 1))


def _mirror_bbox_col(c0: int, w: int, axis_c2: int) -> int:
    return int(axis_c2 - (c0 + w - 1))


def _bbox_ok(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - PADDING)
    cc0 = max(0, c0 - PADDING)
    rr1 = min(hh, r0 + h + PADDING)
    cc1 = min(ww, c0 + w + PADDING)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _mark_bbox(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - PADDING)
    cc0 = max(0, c0 - PADDING)
    rr1 = min(reserved.shape[0], r0 + h + PADDING)
    cc1 = min(reserved.shape[1], c0 + w + PADDING)
    reserved[rr0:rr1, cc0:cc1] = True


def _paint_sprite(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _quadrant_order() -> list[tuple[int, int]]:
    return [(0, 0), (0, 1), (1, 0), (1, 1)]


def _place_orbits(
    h: int,
    w: int,
    axis_r2: int,
    axis_c2: int,
    masks: list[np.ndarray],
) -> list[dict] | None:
    reserved = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []
    row_choices = list(range(h))
    col_choices = list(range(w))
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    for mask in masks:
        mh, mw = mask.shape
        found = None
        for r0 in row_choices:
            if 2 * (r0 + mh - 1) >= axis_r2:
                continue
            rb = _mirror_bbox_row(r0, mh, axis_r2)
            if rb <= r0 + mh:
                continue
            if rb < 0 or rb + mh > h:
                continue
            for c0 in col_choices:
                if 2 * (c0 + mw - 1) >= axis_c2:
                    continue
                cr = _mirror_bbox_col(c0, mw, axis_c2)
                if cr <= c0 + mw:
                    continue
                if cr < 0 or cr + mw > w:
                    continue
                boxes = [
                    (int(r0), int(c0)),
                    (int(r0), int(cr)),
                    (int(rb), int(c0)),
                    (int(rb), int(cr)),
                ]
                if any(not _bbox_ok(reserved, rr, cc, mh, mw) for rr, cc in boxes):
                    continue
                found = boxes
                break
            if found is not None:
                break
        if found is None:
            return None
        for rr, cc in found:
            _mark_bbox(reserved, rr, cc, mh, mw)
        placements.append({"mask": mask, "boxes": found})
    return placements


def _expected_from_instance(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(meta["shape"], int(meta["bg"]), dtype=int)
    out = np.full(meta["shape"], int(meta["bg"]), dtype=int)
    missing_orbit = int(meta["missing_orbit"])
    missing_q = meta["missing_quadrant"]

    for orbit_idx, orbit in enumerate(meta["orbits"]):
        base = orbit["mask"]
        oriented = {
            (0, 0): base,
            (0, 1): np.fliplr(base),
            (1, 0): np.flipud(base),
            (1, 1): np.flipud(np.fliplr(base)),
        }
        boxes = {
            (0, 0): orbit["boxes"][0],
            (0, 1): orbit["boxes"][1],
            (1, 0): orbit["boxes"][2],
            (1, 1): orbit["boxes"][3],
        }
        for q in _quadrant_order():
            rr, cc = boxes[q]
            color = int(meta["quadrant_colors"][q])
            _paint_sprite(out, oriented[q], rr, cc, color)
            if orbit_idx == missing_orbit and q == missing_q:
                continue
            _paint_sprite(inp, oriented[q], rr, cc, color)
    return inp, out


def _semantic_ok(inp: np.ndarray, out: np.ndarray, axis_r2: int, axis_c2: int) -> bool:
    sols = _solution_outputs(inp)
    if len(sols) != 1:
        sigs = {(tuple(candidate.shape), candidate.tobytes()) for _sol, candidate in sols}
        if len(sigs) != 1:
            return False
    sol, candidate = sols[0]
    if not np.array_equal(candidate, out):
        return False
    if int(sol["axis_r2"]) != int(axis_r2) or int(sol["axis_c2"]) != int(axis_c2):
        return False

    bg = get_bg(out)
    comps = find_components(out, bg)
    qcolor: dict[tuple[int, int], int] = {}
    for comp in comps:
        q = quadrant_of(*bbox_center2(comp["bbox"]), axis_r2, axis_c2)
        if q is None:
            return False
        if q in qcolor and int(qcolor[q]) != int(comp["color"]):
            return False
        qcolor[q] = int(comp["color"])
    return len(qcolor) == 4


def _random_instance() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)

    bg = random.randrange(10)
    palette = [c for c in range(10) if c != bg]
    qcols = random.sample(palette, 4)
    quadrant_colors = {
        (0, 0): int(qcols[0]),
        (0, 1): int(qcols[1]),
        (1, 0): int(qcols[2]),
        (1, 1): int(qcols[3]),
    }

    axis_r_choices = [v for v in range(5, 2 * h - 6, 2) if v != h - 1]
    axis_c_choices = [v for v in range(5, 2 * w - 6, 2) if v != w - 1]
    if not axis_r_choices or not axis_c_choices:
        return None
    axis_r2 = random.choice(axis_r_choices)
    axis_c2 = random.choice(axis_c_choices)

    orbit_count = random.choice(ORBIT_COUNT_CHOICES)
    used: set[tuple[tuple[int, int], bytes]] = set()
    masks: list[np.ndarray] = []
    for _ in range(orbit_count):
        mask = _choose_unique_shape(used)
        if mask is None:
            return None
        masks.append(mask)

    random.shuffle(masks)
    orbits = _place_orbits(h, w, axis_r2, axis_c2, masks)
    if orbits is None:
        return None

    missing_orbit = random.randrange(orbit_count)
    missing_quadrant = random.choice(_quadrant_order())
    meta = {
        "shape": (int(h), int(w)),
        "bg": int(bg),
        "quadrant_colors": quadrant_colors,
        "orbits": orbits,
        "missing_orbit": int(missing_orbit),
        "missing_quadrant": missing_quadrant,
    }
    inp, out = _expected_from_instance(meta)
    if not _semantic_ok(inp, out, axis_r2, axis_c2):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_INSTANCE_TRIES):
        instance = _random_instance()
        if instance is not None:
            return instance[0]
    raise RuntimeError("generate_input failed to build a valid instance")
