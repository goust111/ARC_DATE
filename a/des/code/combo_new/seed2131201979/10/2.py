from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 300
GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
INPUT_PAD = 1
OUTPUT_PAD = 1
PAIR_RETRIES = 60
PLACE_RETRIES = 80
OBJECT_RETRIES = 60
OVERLAP_PROB = 0.5

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)
DIRECTION_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray, conn8: bool = False) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS8 if conn8 else NEIGHBORS4
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _fit_inside(coords: np.ndarray, h: int, w: int) -> bool:
    if coords.size == 0:
        return False
    return (
        int(coords[:, 0].min()) >= 0
        and int(coords[:, 1].min()) >= 0
        and int(coords[:, 0].max()) < h
        and int(coords[:, 1].max()) < w
    )


def _shift_coords(coords: np.ndarray, top: int, left: int) -> np.ndarray:
    return coords + np.array([top, left], dtype=int)


def _conflicts(occ: np.ndarray, coords: np.ndarray, pad: int) -> bool:
    h, w = occ.shape
    for r, c in coords:
        r0 = max(0, int(r) - pad)
        r1 = min(h, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(w, int(c) + pad + 1)
        if bool(occ[r0:r1, c0:c1].any()):
            return True
    return False


def _mark(occ: np.ndarray, coords: np.ndarray, pad: int) -> None:
    h, w = occ.shape
    for r, c in coords:
        r0 = max(0, int(r) - pad)
        r1 = min(h, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(w, int(c) + pad + 1)
        occ[r0:r1, c0:c1] = True


def _set_from_coords(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _pair_conflicts(a: np.ndarray, b: np.ndarray, pad: int) -> bool:
    bset = _set_from_coords(b)
    for r, c in a:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                if (int(r) + dr, int(c) + dc) in bset:
                    return True
    return False


def _coords_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for r, c in cells:
        mask[r - min_r, c - min_c] = True
    return mask


def _hole_cells(direction: str, tip_r: int, tip_c: int) -> set[tuple[int, int]]:
    if direction == "up":
        return {(tip_r, tip_c), (tip_r + 1, tip_c - 1), (tip_r + 1, tip_c + 1)}
    if direction == "down":
        return {(tip_r, tip_c), (tip_r - 1, tip_c - 1), (tip_r - 1, tip_c + 1)}
    if direction == "left":
        return {(tip_r, tip_c), (tip_r - 1, tip_c + 1), (tip_r + 1, tip_c + 1)}
    if direction == "right":
        return {(tip_r, tip_c), (tip_r - 1, tip_c - 1), (tip_r + 1, tip_c - 1)}
    raise ValueError(direction)


def _count_internal_arrow_holes(mask: np.ndarray) -> int:
    holes = 0
    bg_mask = ~mask
    for comp in _components(bg_mask, conn8=True):
        if (
            int(comp[:, 0].min()) == 0
            or int(comp[:, 1].min()) == 0
            or int(comp[:, 0].max()) == mask.shape[0] - 1
            or int(comp[:, 1].max()) == mask.shape[1] - 1
        ):
            continue
        if int(comp.shape[0]) == 3:
            holes += 1
    return holes


def _make_object_mask(rng: random.Random, direction: str) -> np.ndarray | None:
    for _ in range(OBJECT_RETRIES):
        base_h = rng.randint(5, 8)
        base_w = rng.randint(5, 8)
        cells = {(r, c) for r in range(base_h) for c in range(base_w)}

        for _ in range(rng.choice([0, 1, 1, 2])):
            side = rng.choice(["top", "bottom", "left", "right"])
            depth = rng.randint(1, 2)
            if side in ("top", "bottom"):
                span = rng.randint(1, base_w)
                start = rng.randint(0, base_w - span)
                if side == "top":
                    rows = range(-depth, 0)
                else:
                    rows = range(base_h, base_h + depth)
                for rr in rows:
                    for cc in range(start, start + span):
                        cells.add((rr, cc))
            else:
                span = rng.randint(1, base_h)
                start = rng.randint(0, base_h - span)
                if side == "left":
                    cols = range(-depth, 0)
                else:
                    cols = range(base_w, base_w + depth)
                for rr in range(start, start + span):
                    for cc in cols:
                        cells.add((rr, cc))

        if direction == "up":
            tip_r = rng.randint(1, base_h - 3)
            tip_c = rng.randint(2, base_w - 3)
        elif direction == "down":
            tip_r = rng.randint(2, base_h - 2)
            tip_c = rng.randint(2, base_w - 3)
        elif direction == "left":
            tip_r = rng.randint(2, base_h - 3)
            tip_c = rng.randint(1, base_w - 3)
        else:
            tip_r = rng.randint(2, base_h - 3)
            tip_c = rng.randint(2, base_w - 2)

        hole = _hole_cells(direction, tip_r, tip_c)
        if not hole.issubset(cells):
            continue
        cells -= hole
        mask = _coords_to_mask(cells)
        if len(_components(mask, conn8=False)) != 1:
            continue
        if _count_internal_arrow_holes(mask) != 1:
            continue
        return mask
    return None


def _find_arrow_hole(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    holes: list[np.ndarray] = []
    for comp in _components(~mask, conn8=True):
        if (
            int(comp[:, 0].min()) == 0
            or int(comp[:, 1].min()) == 0
            or int(comp[:, 0].max()) == mask.shape[0] - 1
            or int(comp[:, 1].max()) == mask.shape[1] - 1
        ):
            continue
        if int(comp.shape[0]) == 3:
            holes.append(comp + np.array([r0, c0], dtype=int))

    assert len(holes) == 1, "each object must contain exactly one internal 3-pixel arrow hole"
    return holes[0]


def _decode_arrow_direction(hole: np.ndarray) -> tuple[int, int]:
    row_counts = Counter(map(int, hole[:, 0]))
    col_counts = Counter(map(int, hole[:, 1]))

    pair_rows = [r for r, n in row_counts.items() if n == 2]
    pair_cols = [c for c, n in col_counts.items() if n == 2]

    if pair_rows:
        pair_row = int(pair_rows[0])
        single_row = int([r for r, n in row_counts.items() if n == 1][0])
        pair_cols_sorted = sorted(int(c) for r, c in hole if int(r) == pair_row)
        single_col = int([c for r, c in hole if int(r) == single_row][0])
        assert pair_cols_sorted[1] - pair_cols_sorted[0] == 2
        assert single_col == pair_cols_sorted[0] + 1
        return (-1, 0) if pair_row > single_row else (1, 0)

    assert pair_cols, "arrow hole must have a doubled row or doubled column"
    pair_col = int(pair_cols[0])
    single_col = int([c for c, n in col_counts.items() if n == 1][0])
    pair_rows_sorted = sorted(int(r) for r, c in hole if int(c) == pair_col)
    single_row = int([r for r, c in hole if int(c) == single_col][0])
    assert pair_rows_sorted[1] - pair_rows_sorted[0] == 2
    assert single_row == pair_rows_sorted[0] + 1
    return (0, -1) if pair_col > single_col else (0, 1)


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components(g == color, conn8=False):
            hole = _find_arrow_hole(comp)
            direction = _decode_arrow_direction(hole)
            objects.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "area": int(comp.shape[0]),
                    "direction": direction,
                }
            )
    return objects


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _extract_objects(g)
    counts = Counter(int(obj["color"]) for obj in objects)
    out = np.zeros((h, w), dtype=int)

    ordered = sorted(
        objects,
        key=lambda obj: (
            -int(obj["area"]),
            int(obj["color"]),
            int(np.min(obj["coords"][:, 0])),
            int(np.min(obj["coords"][:, 1])),
        ),
    )
    for obj in ordered:
        dr, dc = obj["direction"]
        dist = counts[int(obj["color"])]
        shifted = obj["coords"] + np.array([dr * dist, dc * dist], dtype=int)
        assert _fit_inside(shifted, h, w), "translated object must stay inside the grid"
        out[shifted[:, 0], shifted[:, 1]] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each monochrome object in the direction encoded by its internal arrow-shaped hole.

    Input structure:
    - The background is black.
    - The foreground consists of several monochrome 4-connected objects.
    - Every object contains exactly one internal 3-pixel black hole shaped like an arrowhead: one tip pixel and two diagonal base pixels on the opposite side.
    - The same color may appear on multiple separate objects.

    Rule:
    - For each color, count how many objects of that color appear in the grid.
    - Decode each object's arrow direction from its 3-pixel internal hole.
    - Translate that whole object by N cells in the decoded direction, where N is the object count of its color.
    - If translated objects overlap, larger objects form lower layers and smaller objects overwrite them; the arrow holes remain empty.

    Key insight:
    - The hole shape gives a local direction code, while the translation distance comes from a global same-color count, so both cues must be combined.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _count_templates(rng: random.Random) -> list[int]:
    return rng.choice([
        [2, 2],
        [3, 1],
        [2, 1, 1],
        [3, 2],
        [2, 2, 1],
        [3, 1, 1],
    ]).copy()


def _build_specs(rng: random.Random) -> list[dict[str, object]] | None:
    counts = _count_templates(rng)
    colors = rng.sample(PALETTE, len(counts))

    directions = ["up", "down", "left", "right"]
    if sum(counts) == 5:
        directions.append(rng.choice(["up", "down", "left", "right"]))
    rng.shuffle(directions)

    specs: list[dict[str, object]] = []
    used_areas: set[int] = set()
    k = 0
    for color, count in zip(colors, counts):
        for _ in range(count):
            direction = directions[k]
            k += 1
            mask = None
            area = -1
            for _ in range(16):
                cand = _make_object_mask(rng, direction)
                if cand is None:
                    continue
                cand_area = int(cand.sum())
                if cand_area in used_areas:
                    continue
                mask = cand
                area = cand_area
                break
            if mask is None:
                return None
            used_areas.add(area)
            coords = np.argwhere(mask)
            specs.append(
                {
                    "color": int(color),
                    "count": int(count),
                    "direction_name": direction,
                    "direction": DIRECTION_VECS[direction],
                    "mask": mask,
                    "coords": coords,
                    "shape": mask.shape,
                    "area": area,
                    "bbox_area": int(mask.shape[0] * mask.shape[1]),
                }
            )
    rng.shuffle(specs)
    return specs


def _place_regular_objects(
    specs: list[dict[str, object]],
    order: list[int],
    h: int,
    w: int,
    rng: random.Random,
    occ_in: np.ndarray,
    occ_out: np.ndarray,
    placements: dict[int, dict[str, np.ndarray]],
) -> bool:
    for idx in order:
        spec = specs[idx]
        coords = spec["coords"]
        oh, ow = spec["shape"]
        dr, dc = spec["direction"]
        dist = int(spec["count"])
        for _ in range(PLACE_RETRIES):
            out_top = rng.randint(0, h - oh)
            out_left = rng.randint(0, w - ow)
            out_coords = _shift_coords(coords, out_top, out_left)
            in_coords = _shift_coords(coords, out_top - dr * dist, out_left - dc * dist)
            if not _fit_inside(in_coords, h, w):
                continue
            if _conflicts(occ_out, out_coords, OUTPUT_PAD):
                continue
            if _conflicts(occ_in, in_coords, INPUT_PAD):
                continue
            placements[idx] = {"input": in_coords, "output": out_coords}
            _mark(occ_out, out_coords, OUTPUT_PAD)
            _mark(occ_in, in_coords, INPUT_PAD)
            break
        else:
            return False
    return True


def _find_overlap_pair(
    specs: list[dict[str, object]],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[int, int, dict[int, dict[str, np.ndarray]]] | None:
    candidates = [
        (i, j)
        for i in range(len(specs))
        for j in range(i + 1, len(specs))
        if int(specs[i]["color"]) != int(specs[j]["color"])
        and int(specs[i]["area"]) != int(specs[j]["area"])
        and tuple(specs[i]["direction"]) != tuple(specs[j]["direction"])
    ]
    if not candidates:
        return None

    rng.shuffle(candidates)
    for ia, ib in candidates:
        coords_a = specs[ia]["coords"]
        coords_b = specs[ib]["coords"]
        ah, aw = specs[ia]["shape"]
        bh, bw = specs[ib]["shape"]
        dra, dca = specs[ia]["direction"]
        drb, dcb = specs[ib]["direction"]
        dist_a = int(specs[ia]["count"])
        dist_b = int(specs[ib]["count"])
        aset = _set_from_coords(coords_a)
        offsets: list[tuple[int, int]] = []
        for dtop in range(-3, 4):
            for dleft in range(-3, 4):
                if abs(dtop) + abs(dleft) not in (1, 2, 3):
                    continue
                shifted_b = _shift_coords(coords_b, dtop, dleft)
                overlap = len(aset & _set_from_coords(shifted_b))
                if overlap == 0:
                    continue
                if overlap >= min(int(specs[ia]["area"]), int(specs[ib]["area"])) - 2:
                    continue
                offsets.append((dtop, dleft))

        rng.shuffle(offsets)
        for dtop, dleft in offsets[:PAIR_RETRIES]:
            delta_in_r = dra * dist_a + dtop - drb * dist_b
            delta_in_c = dca * dist_a + dleft - dcb * dist_b
            if _pair_conflicts(coords_a, _shift_coords(coords_b, delta_in_r, delta_in_c), INPUT_PAD):
                continue

            row_lo = max(0, -dra * dist_a, -delta_in_r, -delta_in_r - drb * dist_b)
            row_hi = min(h - ah, h - ah - dra * dist_a, h - bh - delta_in_r, h - bh - delta_in_r - drb * dist_b)
            col_lo = max(0, -dca * dist_a, -delta_in_c, -delta_in_c - dcb * dist_b)
            col_hi = min(w - aw, w - aw - dca * dist_a, w - bw - delta_in_c, w - bw - delta_in_c - dcb * dist_b)
            if row_lo > row_hi or col_lo > col_hi:
                continue

            in_top_a = rng.randint(row_lo, row_hi)
            in_left_a = rng.randint(col_lo, col_hi)
            in_top_b = in_top_a + delta_in_r
            in_left_b = in_left_a + delta_in_c
            out_top_a = in_top_a + dra * dist_a
            out_left_a = in_left_a + dca * dist_a
            out_top_b = in_top_b + drb * dist_b
            out_left_b = in_left_b + dcb * dist_b

            in_a = _shift_coords(coords_a, in_top_a, in_left_a)
            in_b = _shift_coords(coords_b, in_top_b, in_left_b)
            out_a = _shift_coords(coords_a, out_top_a, out_left_a)
            out_b = _shift_coords(coords_b, out_top_b, out_left_b)

            return (
                ia,
                ib,
                {
                    ia: {"input": in_a, "output": out_a},
                    ib: {"input": in_b, "output": out_b},
                },
            )
    return None


def _render_input(specs: list[dict[str, object]], placements: dict[int, dict[str, np.ndarray]], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for idx, spec in enumerate(specs):
        coords = placements[idx]["input"]
        grid[coords[:, 0], coords[:, 1]] = int(spec["color"])
    return grid


def _render_expected_output(
    specs: list[dict[str, object]],
    placements: dict[int, dict[str, np.ndarray]],
    h: int,
    w: int,
) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    order = sorted(
        range(len(specs)),
        key=lambda i: (
            -int(specs[i]["area"]),
            int(specs[i]["color"]),
            int(placements[i]["input"][:, 0].min()),
            int(placements[i]["input"][:, 1].min()),
        ),
    )
    for idx in order:
        coords = placements[idx]["output"]
        out[coords[:, 0], coords[:, 1]] = int(specs[idx]["color"])
    return out


def _has_output_overlap(placements: dict[int, dict[str, np.ndarray]]) -> bool:
    keys = sorted(placements)
    sets = [_set_from_coords(placements[k]["output"]) for k in keys]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _build_specs(rng)
    if specs is None:
        return None

    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    overlap_branch = rng.random() < OVERLAP_PROB

    placements: dict[int, dict[str, np.ndarray]] = {}
    occ_in = np.zeros((h, w), dtype=bool)
    occ_out = np.zeros((h, w), dtype=bool)

    place_order = sorted(range(len(specs)), key=lambda i: int(specs[i]["bbox_area"]), reverse=True)

    if overlap_branch:
        pair = _find_overlap_pair(specs, h, w, rng)
        if pair is None:
            return None
        ia, ib, pair_places = pair
        placements.update(pair_places)
        _mark(occ_in, placements[ia]["input"], INPUT_PAD)
        _mark(occ_in, placements[ib]["input"], INPUT_PAD)
        _mark(occ_out, placements[ia]["output"], OUTPUT_PAD)
        _mark(occ_out, placements[ib]["output"], OUTPUT_PAD)
        remaining = [idx for idx in place_order if idx not in (ia, ib)]
        if not _place_regular_objects(specs, remaining, h, w, rng, occ_in, occ_out, placements):
            return None
    else:
        if not _place_regular_objects(specs, place_order, h, w, rng, occ_in, occ_out, placements):
            return None

    grid = _render_input(specs, placements, h, w)
    if _has_output_overlap(placements) != overlap_branch:
        return None
    if np.array_equal(grid, _render_expected_output(specs, placements, h, w)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
