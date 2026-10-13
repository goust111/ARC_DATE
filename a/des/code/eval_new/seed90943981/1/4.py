from __future__ import annotations

import numpy as np


BG = 0
SEED_COLOR = 7
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
DIRS = ("up", "down", "left", "right")
PATTERN_TEMPLATES = (
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=int),
)
MAX_ATTEMPTS = 400


def _most_common_value(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _interval_overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_objects(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = _most_common_value(g)

    objects: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            expected = _frame_mask(h, w)
            actual = g[r0 : r1 + 1, c0 : c1 + 1] == color
            if len(comp) != int(expected.sum()) or not bool(np.array_equal(actual, expected)):
                continue

            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_mask = interior != bg
            if not bool(seed_mask.any()):
                continue

            seed_coords = list(map(tuple, np.argwhere(seed_mask)))
            sr0, sc0, sr1, sc1 = _bbox(seed_coords)
            touches = []
            if sr0 == 0:
                touches.append("up")
            if sr1 == interior.shape[0] - 1:
                touches.append("down")
            if sc0 == 0:
                touches.append("left")
            if sc1 == interior.shape[1] - 1:
                touches.append("right")
            if len(touches) != 1:
                continue

            pattern = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
            objects.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "frame_color": color,
                    "interior_h": int(interior.shape[0]),
                    "interior_w": int(interior.shape[1]),
                    "seed_box": (sr0, sc0, sr1, sc1),
                    "pattern": pattern,
                    "direction": touches[0],
                }
            )

    objects.sort(key=lambda obj: obj["bbox"])
    return objects


def _partner_indices(objects: list[dict]) -> list[int]:
    partners: list[int] = []
    for i, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["bbox"]
        exact: list[int] = []
        loose: list[int] = []
        for j, other in enumerate(objects):
            if i == j:
                continue
            rr0, cc0, rr1, cc1 = other["bbox"]
            row_overlap = _interval_overlap_len(r0, r1, rr0, rr1) > 0
            col_overlap = _interval_overlap_len(c0, c1, cc0, cc1) > 0
            if row_overlap or col_overlap:
                loose.append(j)
            if row_overlap ^ col_overlap:
                exact.append(j)
        if len(exact) == 1:
            partners.append(exact[0])
        else:
            assert len(loose) == 1, "expected exactly one paired object"
            partners.append(loose[0])
    return partners


def _paint_growth(out: np.ndarray, obj: dict, length: int, bg: int) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    sr0, sc0, _sr1, _sc1 = obj["seed_box"]
    pattern = obj["pattern"]
    ph, pw = pattern.shape
    direction = obj["direction"]

    if direction == "up":
        base_row = r0 - 1
        base_col = c0 + 1 + sc0
        for step in range(length):
            rr = base_row - step
            src_r = step % ph
            for cc in range(pw):
                value = int(pattern[src_r, cc])
                if value != bg:
                    out[rr, base_col + cc] = value
    elif direction == "down":
        base_row = r1 + 1
        base_col = c0 + 1 + sc0
        for step in range(length):
            rr = base_row + step
            src_r = ph - 1 - (step % ph)
            for cc in range(pw):
                value = int(pattern[src_r, cc])
                if value != bg:
                    out[rr, base_col + cc] = value
    elif direction == "left":
        base_row = r0 + 1 + sr0
        base_col = c0 - 1
        for step in range(length):
            cc = base_col - step
            src_c = step % pw
            for rr in range(ph):
                value = int(pattern[rr, src_c])
                if value != bg:
                    out[base_row + rr, cc] = value
    else:
        base_row = r0 + 1 + sr0
        base_col = c1 + 1
        for step in range(length):
            cc = base_col + step
            src_c = pw - 1 - (step % pw)
            for rr in range(ph):
                value = int(pattern[rr, src_c])
                if value != bg:
                    out[base_row + rr, cc] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each framed seed outward using the paired frame's short interior side as the strip length.

    Input structure:
    - Background is black.
    - Several hollow rectangular frames with distinct border colors are scattered on the grid.
    - Inside each frame, a small orange seed pattern sits flush against exactly one inner wall, with black space elsewhere.
    - Every frame has exactly one partner whose row span or column span overlaps its own.

    Rule:
    - Detect every hollow frame and crop the non-black seed pattern inside it.
    - The inner wall touched by the seed determines the growth direction: top, bottom, left, or right.
    - The unique overlapping partner frame provides the growth length, equal to that partner's interior short side.
    - Starting just outside the chosen wall, repeat the seed pattern outward for that many cells, copying seed rows or columns from the touched side inward in a periodic cycle.
    - Keep the original frames and interior seeds unchanged; only paint the new exterior strip.

    Key insight:
    The transformation combines three independent cues: which wall the seed touches gives direction, overlap with another frame selects the partner, and the partner's interior short side sets how far the seed pattern must be tiled outside.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_value(g)
    objects = _extract_objects(g, bg=bg)
    if not objects:
        return g.copy()

    partners = _partner_indices(objects)
    out = g.copy()
    for i, obj in enumerate(objects):
        partner = objects[partners[i]]
        length = min(int(partner["interior_h"]), int(partner["interior_w"]))
        _paint_growth(out, obj, length, bg)
    return out


def _random_pattern(rng: np.random.Generator) -> np.ndarray:
    pattern = PATTERN_TEMPLATES[int(rng.integers(len(PATTERN_TEMPLATES)))].copy()
    if bool(rng.integers(2)):
        pattern = pattern[::-1, :]
    if bool(rng.integers(2)):
        pattern = pattern[:, ::-1]
    pattern = np.rot90(pattern, int(rng.integers(4)))
    return pattern


def _sample_object_spec(rng: np.random.Generator, zone_h: int, zone_w: int) -> dict | None:
    direction = DIRS[int(rng.integers(len(DIRS)))]
    for _ in range(40):
        pattern = _random_pattern(rng)
        ph, pw = pattern.shape
        ih = int(rng.integers(4, min(7, zone_h - 1) + 1))
        iw = int(rng.integers(4, min(7, zone_w - 1) + 1))

        if direction in ("up", "down"):
            if ih < ph + 1 or iw < pw + 2:
                continue
            sr = 0 if direction == "up" else ih - ph
            sc = int(rng.integers(1, iw - pw))
        else:
            if iw < pw + 1 or ih < ph + 2:
                continue
            sr = int(rng.integers(1, ih - ph))
            sc = 0 if direction == "left" else iw - pw

        return {
            "direction": direction,
            "pattern": pattern,
            "interior_h": ih,
            "interior_w": iw,
            "outer_h": ih + 2,
            "outer_w": iw + 2,
            "seed_r": sr,
            "seed_c": sc,
        }
    return None


def _border_cells(spec: dict, top: int, left: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    h = int(spec["outer_h"])
    w = int(spec["outer_w"])
    for c in range(left, left + w):
        out.add((top, c))
        out.add((top + h - 1, c))
    for r in range(top + 1, top + h - 1):
        out.add((r, left))
        out.add((r, left + w - 1))
    return out


def _seed_cells(spec: dict, top: int, left: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    pattern = spec["pattern"]
    seed_r = int(spec["seed_r"])
    seed_c = int(spec["seed_c"])
    for rr in range(pattern.shape[0]):
        for cc in range(pattern.shape[1]):
            if int(pattern[rr, cc]) != 0:
                out.add((top + 1 + seed_r + rr, left + 1 + seed_c + cc))
    return out


def _growth_cells(spec: dict, top: int, left: int, length: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    pattern = spec["pattern"]
    ph, pw = pattern.shape
    seed_r = int(spec["seed_r"])
    seed_c = int(spec["seed_c"])
    direction = spec["direction"]
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])

    if direction == "up":
        row0 = top - 1
        col0 = left + 1 + seed_c
        for step in range(length):
            src_r = step % ph
            rr = row0 - step
            for cc in range(pw):
                if int(pattern[src_r, cc]) != 0:
                    out.add((rr, col0 + cc))
    elif direction == "down":
        row0 = top + outer_h
        col0 = left + 1 + seed_c
        for step in range(length):
            src_r = ph - 1 - (step % ph)
            rr = row0 + step
            for cc in range(pw):
                if int(pattern[src_r, cc]) != 0:
                    out.add((rr, col0 + cc))
    elif direction == "left":
        row0 = top + 1 + seed_r
        col0 = left - 1
        for step in range(length):
            src_c = step % pw
            cc = col0 - step
            for rr in range(ph):
                if int(pattern[rr, src_c]) != 0:
                    out.add((row0 + rr, cc))
    else:
        row0 = top + 1 + seed_r
        col0 = left + outer_w
        for step in range(length):
            src_c = pw - 1 - (step % pw)
            cc = col0 + step
            for rr in range(ph):
                if int(pattern[rr, src_c]) != 0:
                    out.add((row0 + rr, cc))
    return out


def _position_bounds(spec: dict, zone_h: int, zone_w: int, growth_length: int) -> tuple[int, int, int, int] | None:
    min_r = growth_length if spec["direction"] == "up" else 0
    max_r = zone_h - int(spec["outer_h"]) - (growth_length if spec["direction"] == "down" else 0)
    min_c = growth_length if spec["direction"] == "left" else 0
    max_c = zone_w - int(spec["outer_w"]) - (growth_length if spec["direction"] == "right" else 0)
    if min_r > max_r or min_c > max_c:
        return None
    return min_r, max_r, min_c, max_c


def _enumerate_pair_positions(
    spec_a: dict,
    spec_b: dict,
    zone_h: int,
    zone_w: int,
    orientation: str,
) -> list[tuple[int, int, int, int]]:
    len_a = min(int(spec_b["interior_h"]), int(spec_b["interior_w"]))
    len_b = min(int(spec_a["interior_h"]), int(spec_a["interior_w"]))
    bounds_a = _position_bounds(spec_a, zone_h, zone_w, len_a)
    bounds_b = _position_bounds(spec_b, zone_h, zone_w, len_b)
    if bounds_a is None or bounds_b is None:
        return []

    a_r0, a_r1, a_c0, a_c1 = bounds_a
    b_r0, b_r1, b_c0, b_c1 = bounds_b
    candidates: list[tuple[int, int, int, int]] = []

    for ra in range(a_r0, a_r1 + 1):
        for ca in range(a_c0, a_c1 + 1):
            a_bbox = (ra, ca, ra + int(spec_a["outer_h"]) - 1, ca + int(spec_a["outer_w"]) - 1)
            a_out = _border_cells(spec_a, ra, ca) | _seed_cells(spec_a, ra, ca) | _growth_cells(spec_a, ra, ca, len_a)
            for rb in range(b_r0, b_r1 + 1):
                for cb in range(b_c0, b_c1 + 1):
                    b_bbox = (rb, cb, rb + int(spec_b["outer_h"]) - 1, cb + int(spec_b["outer_w"]) - 1)
                    row_overlap = _interval_overlap_len(a_bbox[0], a_bbox[2], b_bbox[0], b_bbox[2])
                    col_overlap = _interval_overlap_len(a_bbox[1], a_bbox[3], b_bbox[1], b_bbox[3])
                    if orientation == "horizontal":
                        if row_overlap < 2 or col_overlap > 0:
                            continue
                        if not (a_bbox[3] + 1 < b_bbox[1] or b_bbox[3] + 1 < a_bbox[1]):
                            continue
                    else:
                        if col_overlap < 2 or row_overlap > 0:
                            continue
                        if not (a_bbox[2] + 1 < b_bbox[0] or b_bbox[2] + 1 < a_bbox[0]):
                            continue

                    b_out = _border_cells(spec_b, rb, cb) | _seed_cells(spec_b, rb, cb) | _growth_cells(spec_b, rb, cb, len_b)
                    if a_out & b_out:
                        continue
                    candidates.append((ra, ca, rb, cb))
    return candidates


def _paint_object(grid: np.ndarray, spec: dict, top: int, left: int, frame_color: int) -> None:
    h = int(spec["outer_h"])
    w = int(spec["outer_w"])
    grid[top, left : left + w] = frame_color
    grid[top + h - 1, left : left + w] = frame_color
    grid[top : top + h, left] = frame_color
    grid[top : top + h, left + w - 1] = frame_color

    pattern = spec["pattern"]
    sr = int(spec["seed_r"])
    sc = int(spec["seed_c"])
    for rr in range(pattern.shape[0]):
        for cc in range(pattern.shape[1]):
            if int(pattern[rr, cc]) != 0:
                grid[top + 1 + sr + rr, left + 1 + sc + cc] = SEED_COLOR


def _generate_pair_in_zone(
    rng: np.random.Generator,
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    colors: tuple[int, int],
    orientation: str,
) -> bool:
    zr0, zc0, zr1, zc1 = zone
    zone_h = zr1 - zr0
    zone_w = zc1 - zc0
    if zone_h < 10 or zone_w < 10:
        return False

    for _ in range(120):
        spec_a = _sample_object_spec(rng, zone_h, zone_w)
        spec_b = _sample_object_spec(rng, zone_h, zone_w)
        if spec_a is None or spec_b is None:
            continue
        candidates = _enumerate_pair_positions(spec_a, spec_b, zone_h, zone_w, orientation)
        if not candidates:
            continue
        ra, ca, rb, cb = candidates[int(rng.integers(len(candidates)))]
        _paint_object(grid, spec_a, zr0 + ra, zc0 + ca, colors[0])
        _paint_object(grid, spec_b, zr0 + rb, zc0 + cb, colors[1])
        return True
    return False


def _zones_for_pair_count(rng: np.random.Generator, pair_count: int) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    if pair_count == 1:
        h = int(rng.integers(18, 31))
        w = int(rng.integers(18, 31))
        return h, w, [(0, 0, h, w)]

    h = int(rng.integers(28, 31))
    w = int(rng.integers(28, 31))
    split_r = int(rng.integers(h // 2 - 1, h // 2 + 2))
    split_c = int(rng.integers(w // 2 - 1, w // 2 + 2))
    gap = 2
    zones = [
        (0, 0, split_r, split_c),
        (split_r + gap, split_c + gap, h, w),
    ]
    return h, w, zones


def _validate_generated(inp: np.ndarray) -> bool:
    try:
        objects = _extract_objects(inp, bg=BG)
        if len(objects) not in (2, 4):
            return False
        partners = _partner_indices(objects)
        if sorted(partners.count(i) for i in range(len(objects))) != [1] * len(objects):
            return False
        out = main(inp)
    except Exception:
        return False
    if inp.shape != out.shape or not bool(np.any(inp != out)):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = 1 if bool(rng.integers(2)) else 2
    h, w, zones = _zones_for_pair_count(rng, pair_count)
    grid = np.full((h, w), BG, dtype=int)

    n_objects = pair_count * 2
    frame_colors = rng.choice(np.array(FRAME_COLORS, dtype=int), size=n_objects, replace=False)
    frame_colors = [int(x) for x in frame_colors]

    color_idx = 0
    for zone in zones:
        orientation = "horizontal" if bool(rng.integers(2)) else "vertical"
        if not _generate_pair_in_zone(
            rng,
            grid,
            zone,
            (frame_colors[color_idx], frame_colors[color_idx + 1]),
            orientation,
        ):
            return None
        color_idx += 2

    if _validate_generated(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
