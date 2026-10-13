from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if bool(cell) else "0" for cell in row) for row in mask.astype(bool))


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected a non-empty mask")
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(_mask_key(_tight_mask(rotated)))
        variants.append(_mask_key(_tight_mask(np.fliplr(rotated))))
    return min(variants)


def _all_variants(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], tuple[str, ...]]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            tight = _tight_mask(candidate)
            key = (tuple(map(int, tight.shape)), _mask_key(tight))
            if key in seen:
                continue
            seen.add(key)
            out.append(tight.copy())
    return out


def _find_components(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    mask = (grid != int(bg)) & (grid != RED)
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    coord_to_idx: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pixels.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(pixels, dtype=int)
            r0, c0, r1, c1 = _bbox_coords(coords)
            tight = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            tight[coords[:, 0] - r0, coords[:, 1] - c0] = True
            idx = len(comps)
            comp = {
                "id": idx,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": tight,
                "sig": _shape_signature(tight),
                "raw_key": _mask_key(tight),
                "area": int(tight.sum()),
            }
            for rr, cc in pixels:
                coord_to_idx[(int(rr), int(cc))] = idx
            comps.append(comp)

    return comps, coord_to_idx


def _start_and_direction(grid: np.ndarray) -> tuple[int, int, int, int]:
    starts = np.argwhere(grid == RED)
    if starts.shape != (1, 2):
        raise AssertionError("expected exactly one red border marker")
    r, c = map(int, starts[0])
    h, w = grid.shape
    if r == 0:
        return r, c, 1, 0
    if r == h - 1:
        return r, c, -1, 0
    if c == 0:
        return r, c, 0, 1
    if c == w - 1:
        return r, c, 0, -1
    raise AssertionError("red marker must lie on the border")


def _front_point(bbox: tuple[int, int, int, int], dr: int, dc: int, lane: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if dr == 1:
        return r0 - 1, lane
    if dr == -1:
        return r1 + 1, lane
    if dc == 1:
        return lane, c0 - 1
    return lane, c1 + 1


def _back_point(bbox: tuple[int, int, int, int], dr: int, dc: int, lane: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if dr == 1:
        return r1 + 1, lane
    if dr == -1:
        return r0 - 1, lane
    if dc == 1:
        return lane, c1 + 1
    return lane, c0 - 1


def _side_points(
    front: tuple[int, int],
    back: tuple[int, int],
    bbox: tuple[int, int, int, int],
    normal: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    nr, nc = normal
    if nr == -1:
        rr = r0 - 1
        return (rr, front[1]), (rr, back[1])
    if nr == 1:
        rr = r1 + 1
        return (rr, front[1]), (rr, back[1])
    if nc == -1:
        cc = c0 - 1
        return (front[0], cc), (back[0], cc)
    cc = c1 + 1
    return (front[0], cc), (back[0], cc)


def _paint_cell(out: np.ndarray, base: np.ndarray, r: int, c: int, bg: int) -> None:
    if not _in_bounds(base.shape, r, c):
        raise AssertionError("path left the grid unexpectedly")
    if int(base[r, c]) not in (int(bg), RED):
        raise AssertionError("path attempted to overwrite an obstacle")
    out[r, c] = RED


def _paint_line(
    out: np.ndarray,
    base: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    bg: int,
) -> None:
    ar, ac = map(int, a)
    br, bc = map(int, b)
    if ar == br:
        step = 1 if bc >= ac else -1
        for cc in range(ac, bc + step, step):
            _paint_cell(out, base, ar, cc, bg)
        return
    if ac == bc:
        step = 1 if br >= ar else -1
        for rr in range(ar, br + step, step):
            _paint_cell(out, base, rr, ac, bg)
        return
    raise AssertionError("trace segments must be axis-aligned")


def _simulate(grid: np.ndarray, repeat_by_id: dict[int, bool]) -> tuple[np.ndarray, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps, coord_to_idx = _find_components(g, bg)
    comp_by_id = {comp["id"]: comp for comp in comps}
    r, c, dr, dc = _start_and_direction(g)
    lane = c if dr else r
    out = g.copy()
    hits: list[dict] = []
    cur = (r, c)

    while True:
        rr, cc = cur[0] + dr, cur[1] + dc
        hit_idx = None
        while _in_bounds(g.shape, rr, cc):
            if int(g[rr, cc]) not in (bg, RED):
                hit_idx = coord_to_idx[(rr, cc)]
                break
            rr += dr
            cc += dc

        if hit_idx is None:
            last = (rr - dr, cc - dc)
            _paint_line(out, g, cur, last, bg)
            break

        comp = comp_by_id[int(hit_idx)]
        front = _front_point(comp["bbox"], dr, dc, lane)
        expected_hit = (front[0] + dr, front[1] + dc)
        if expected_hit != (rr, cc):
            raise AssertionError("each hit obstacle must meet the lane on its front face")
        back = _back_point(comp["bbox"], dr, dc, lane)
        normal = (-dc, dr) if repeat_by_id[int(hit_idx)] else (dc, -dr)
        side_front, side_back = _side_points(front, back, comp["bbox"], normal)
        for point in (front, back, side_front, side_back):
            if not _in_bounds(g.shape, point[0], point[1]):
                raise AssertionError("detour requires a one-cell clearance around the bbox")
            if int(g[point]) not in (bg, RED):
                raise AssertionError("detour corridor is obstructed")

        _paint_line(out, g, cur, front, bg)
        _paint_line(out, g, front, side_front, bg)
        _paint_line(out, g, side_front, side_back, bg)
        _paint_line(out, g, side_back, back, bg)

        hits.append(
            {
                "id": int(hit_idx),
                "repeat": bool(repeat_by_id[int(hit_idx)]),
                "sig": comp["sig"],
                "raw_key": comp["raw_key"],
                "area": int(comp["area"]),
                "bbox": comp["bbox"],
            }
        )
        cur = back

    return out, hits, comps


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the inward red trace and choose each bbox detour side from global shape frequency.

    Input structure:
    - The background is black and there is exactly one red marker on the border.
    - The interior contains several small solid obstacle components whose colors are irrelevant.
    - Obstacles may repeat elsewhere under rotation/flip equivalence, or may be globally unique.

    Rule:
    - Start from the red border marker and move straight inward along the perpendicular direction, painting background cells red.
    - When the next step would enter an obstacle, classify that obstacle by its tight binary shape up to rotation and reflection over the whole grid.
    - If that normalized shape appears at least twice, detour around the obstacle's bounding box on the travel-left side; otherwise detour on the travel-right side.
    - After passing the bbox, return to the original straight lane and continue until leaving the grid; obstacles remain unchanged.

    Key insight:
    - The path decision is not local to the obstacle's color, size, or position: it depends on a global census of dihedrally equivalent shapes elsewhere in the scene.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps, _coord_to_idx = _find_components(g, bg)
    sig_freq = Counter(comp["sig"] for comp in comps)
    repeat_by_id = {int(comp["id"]): int(sig_freq[comp["sig"]]) >= 2 for comp in comps}
    out, _hits, _comps = _simulate(g, repeat_by_id)
    return out


def _bool_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_BASE_SHAPES = {
    "T4": _bool_mask(["111", "010"]),
    "S4": _bool_mask(["011", "110"]),
    "J4": _bool_mask(["111", "100"]),
    "L3": _bool_mask(["10", "11"]),
    "P5": _bool_mask(["111", "110"]),
    "U5": _bool_mask(["101", "111"]),
    "PLUS5": _bool_mask(["010", "111", "010"]),
    "BAR3": _bool_mask(["111"]),
}


def _build_shape_library() -> list[dict]:
    library = []
    for name, mask in _BASE_SHAPES.items():
        variants = _all_variants(mask)
        library.append(
            {
                "name": name,
                "mask": mask,
                "sig": _shape_signature(mask),
                "area": int(mask.sum()),
                "variants": variants,
                "variant_keys": [_mask_key(v) for v in variants],
            }
        )
    return library


SHAPE_LIBRARY = _build_shape_library()
SIG_TO_INFO = {entry["sig"]: entry for entry in SHAPE_LIBRARY}
AREA_TO_INFOS: dict[int, list[dict]] = {}
for entry in SHAPE_LIBRARY:
    AREA_TO_INFOS.setdefault(int(entry["area"]), []).append(entry)


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 0) -> None:
    h, w = blocked.shape
    rr0 = max(0, int(r0) - int(margin))
    cc0 = max(0, int(c0) - int(margin))
    rr1 = min(h - 1, int(r1) + int(margin))
    cc1 = min(w - 1, int(c1) + int(margin))
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _choose_hit_shape_set(n_hits: int) -> tuple[list[tuple[str, dict]], dict]:
    area_choices = [area for area, infos in AREA_TO_INFOS.items() if len(infos) >= 2]
    shared_area = int(random.choice(area_choices))
    repeat_pool = [info for info in AREA_TO_INFOS[shared_area] if len(info["variants"]) >= 2]
    if not repeat_pool:
        raise RuntimeError("need an asymmetric repeated shape")
    repeat_info = random.choice(repeat_pool)
    unique_same_area_pool = [info for info in AREA_TO_INFOS[shared_area] if info["sig"] != repeat_info["sig"]]
    unique_same_area = random.choice(unique_same_area_pool)

    used_sigs = {repeat_info["sig"], unique_same_area["sig"]}
    hit_specs: list[tuple[str, dict]] = [("repeat", repeat_info), ("unique", unique_same_area)]
    extra_count = int(n_hits) - 2
    for _ in range(extra_count):
        pool = [info for info in SHAPE_LIBRARY if info["sig"] not in used_sigs]
        same_area_pool = [info for info in pool if int(info["area"]) == shared_area]
        chosen = random.choice(same_area_pool or pool)
        hit_specs.append(("unique", chosen))
        used_sigs.add(chosen["sig"])

    random.shuffle(hit_specs)
    return hit_specs, repeat_info


def _choose_front_aligned_col(mask: np.ndarray, lane_c: int, width: int) -> int | None:
    cols = list(np.where(mask[0])[0])
    random.shuffle(cols)
    w = int(mask.shape[1])
    for front_c in cols:
        c0 = int(lane_c - int(front_c))
        if c0 < 2:
            continue
        if c0 + w > width - 2:
            continue
        return c0
    return None


def _place_off_path(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    lane_c: int,
) -> bool:
    h, w = map(int, mask.shape)
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            c1 = c0 + w - 1
            if not (c1 < lane_c - 2 or c0 > lane_c + 2):
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(H - 1, r0 + h)
            cc1 = min(W - 1, c0 + w)
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    _stamp_mask(grid, mask, r0, c0, color)
    _mark_rect(blocked, r0, c0, r0 + h - 1, c0 + w - 1, margin=1)
    return True


def _apply_random_global_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _repeat_lookup_from_key(comps: list[dict], key_name: str) -> dict[int, bool]:
    freq = Counter(comp[key_name] for comp in comps)
    return {int(comp["id"]): int(freq[comp[key_name]]) >= 2 for comp in comps}


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    correct = main(g)
    bg = _get_bg(g)
    comps, _coord_to_idx = _find_components(g, bg)
    sig_freq = Counter(comp["sig"] for comp in comps)
    repeat_lookup = {int(comp["id"]): int(sig_freq[comp["sig"]]) >= 2 for comp in comps}
    traced, hits, traced_comps = _simulate(g, repeat_lookup)
    if not np.array_equal(correct, traced):
        return False
    if len(hits) < 3:
        return False
    repeated_hits = [hit for hit in hits if hit["repeat"]]
    unique_hits = [hit for hit in hits if not hit["repeat"]]
    if not repeated_hits or not unique_hits:
        return False
    if correct.sum() == g.sum():
        return False
    if not (set(hit["area"] for hit in repeated_hits) & set(hit["area"] for hit in unique_hits)):
        return False

    if not any(sig_freq[hit["sig"]] >= 2 and sum(comp["raw_key"] == hit["raw_key"] for comp in traced_comps) == 1 for hit in repeated_hits):
        return False

    raw_lookup = _repeat_lookup_from_key(traced_comps, "raw_key")
    area_lookup = _repeat_lookup_from_key(traced_comps, "area")
    raw_out, _raw_hits, _ = _simulate(g, raw_lookup)
    area_out, _area_hits, _ = _simulate(g, area_lookup)

    hit_sig_freq = Counter(hit["sig"] for hit in hits)
    local_lookup = {int(comp["id"]): False for comp in traced_comps}
    for hit in hits:
        local_lookup[int(hit["id"])] = int(hit_sig_freq[hit["sig"]]) >= 2
    local_out, _local_hits, _ = _simulate(g, local_lookup)

    if np.array_equal(correct, raw_out):
        return False
    if np.array_equal(correct, area_out):
        return False
    if np.array_equal(correct, local_out):
        return False
    return True


def _generate_base_downward() -> np.ndarray | None:
    H = random.randint(18, 28)
    W = random.randint(18, 28)
    lane_c = random.randint(5, W - 6)
    n_hits = random.choice([3, 4])
    hit_specs, repeat_info = _choose_hit_shape_set(n_hits)

    grid = np.full((H, W), BG, dtype=int)
    grid[0, lane_c] = RED
    blocked = np.zeros((H, W), dtype=bool)
    blocked[0, lane_c] = True
    colors = PALETTE.copy()
    random.shuffle(colors)
    color_idx = 0

    partner_variant: np.ndarray | None = None
    cursor = random.randint(3, 5)
    used_unique_sigs: set[tuple[str, ...]] = set()

    for kind, info in hit_specs:
        if kind == "repeat":
            hit_mask = random.choice(info["variants"])
            diff_variants = [variant for variant in info["variants"] if _mask_key(variant) != _mask_key(hit_mask)]
            if not diff_variants:
                return None
            partner_variant = random.choice(diff_variants)
        else:
            hit_mask = random.choice(info["variants"])
            used_unique_sigs.add(info["sig"])

        h, w = map(int, hit_mask.shape)
        c0 = _choose_front_aligned_col(hit_mask, lane_c, W)
        if c0 is None:
            return None
        r0 = cursor
        if r0 < 2 or r0 + h > H - 2:
            return None
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if blocked[max(0, r0 - 2) : min(H, r1 + 3), max(0, c0 - 2) : min(W, c1 + 3)].any():
            return None

        color = colors[color_idx]
        color_idx += 1
        _stamp_mask(grid, hit_mask, r0, c0, color)
        _mark_rect(blocked, r0 - 1, c0 - 1, r1 + 1, c1 + 1, margin=1)
        cursor = r1 + random.randint(4, 6)

    if partner_variant is None:
        return None

    partner_color = colors[color_idx]
    color_idx += 1
    if not _place_off_path(grid, blocked, partner_variant, partner_color, lane_c):
        return None

    distractor_count = random.randint(1, 3)
    distractor_pool = [info for info in SHAPE_LIBRARY if info["sig"] not in used_unique_sigs and info["sig"] != repeat_info["sig"]]
    random.shuffle(distractor_pool)
    for info in distractor_pool[:distractor_count]:
        if color_idx >= len(colors):
            return None
        variant = random.choice(info["variants"])
        if not _place_off_path(grid, blocked, variant, colors[color_idx], lane_c):
            return None
        color_idx += 1

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        base = _generate_base_downward()
        if base is None:
            continue
        grid = _apply_random_global_transform(base)
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
