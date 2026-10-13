from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPS = tuple((flip_lr, rot_k) for flip_lr in (False, True) for rot_k in range(4))


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _transform_mask(mask: np.ndarray, flip_lr: bool, rot_k: int) -> np.ndarray:
    out = np.fliplr(mask) if flip_lr else mask
    out = np.rot90(out, int(rot_k))
    return _crop_mask(out)


def _mask_variants(mask: np.ndarray) -> list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    variants: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = []
    for flip_lr, rot_k in OPS:
        variant = _transform_mask(mask, flip_lr, rot_k)
        coords = tuple(map(tuple, np.argwhere(variant)))
        norm = _normalize_coords(coords)
        if norm in seen:
            continue
        seen.add(norm)
        variants.append((norm, (int(variant.shape[0]), int(variant.shape[1]))))
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(coords for coords, _shape in _mask_variants(mask))


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_solid_rect(mask: np.ndarray) -> bool:
    return int(mask.sum()) == int(mask.shape[0] * mask.shape[1])


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, c0 = min(rows), min(cols)
            r1, c1 = max(rows), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "mask": mask,
                    "area": int(mask.sum()),
                    "shape_key": _canonical_shape_key(mask),
                }
            )
    return components


def _ordered_rectangle_dims(area: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(1, area + 1):
        if area % h != 0:
            continue
        w = area // h
        dims.append((int(h), int(w)))
    dims.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    return dims


def _pair_fit(mask_a: np.ndarray, mask_b: np.ndarray) -> dict | None:
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    total = area_a + area_b
    b_variants = {coords for coords, _shape in _mask_variants(mask_b)}
    results: set[tuple[int, int]] = set()

    for rect_h, rect_w in _ordered_rectangle_dims(total):
        for coords_a, (ha, wa) in _mask_variants(mask_a):
            if ha > rect_h or wa > rect_w:
                continue
            for r0 in range(rect_h - ha + 1):
                for c0 in range(rect_w - wa + 1):
                    occ = np.zeros((rect_h, rect_w), dtype=bool)
                    for rr, cc in coords_a:
                        occ[r0 + rr, c0 + cc] = True
                    comp = [(rr, cc) for rr in range(rect_h) for cc in range(rect_w) if not bool(occ[rr, cc])]
                    comp_norm = _normalize_coords(comp)
                    if comp_norm in b_variants:
                        results.add(tuple(sorted((int(rect_h), int(rect_w)))))

    if len(results) != 1:
        return None
    rect_h, rect_w = next(iter(results))
    return {"height": rect_h, "width": rect_w}


def _unique_pairing(components: list[dict]) -> list[dict]:
    n = len(components)
    assert n >= 4 and n % 2 == 0, "expected an even number of fragments and at least two pairs"
    compat: dict[tuple[int, int], dict] = {}
    for i in range(n):
        for j in range(i + 1, n):
            fit = _pair_fit(components[i]["mask"], components[j]["mask"])
            if fit is not None:
                compat[(i, j)] = fit

    solutions: list[list[tuple[int, int]]] = []

    def search(remaining: list[int], acc: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(list(acc))
            return
        first = remaining[0]
        for idx in range(1, len(remaining)):
            second = remaining[idx]
            key = (first, second)
            if key not in compat:
                continue
            nxt = [remaining[k] for k in range(1, len(remaining)) if k != idx]
            acc.append(key)
            search(nxt, acc)
            acc.pop()

    search(list(range(n)), [])
    assert len(solutions) == 1, "fragment pairing must be unique"

    pairs: list[dict] = []
    for i, j in solutions[0]:
        fit = compat[(i, j)]
        comp_a = components[i]
        comp_b = components[j]
        color_even, color_odd = sorted((int(comp_a["color"]), int(comp_b["color"])))
        pairs.append(
            {
                "height": int(fit["height"]),
                "width": int(fit["width"]),
                "area": int(fit["height"] * fit["width"]),
                "color_even": color_even,
                "color_odd": color_odd,
                "shape_key": tuple(sorted((comp_a["shape_key"], comp_b["shape_key"]))),
            }
        )
    return pairs


def _checkerboard(height: int, width: int, color_even: int, color_odd: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            out[r, c] = int(color_even if (r + c) % 2 == 0 else color_odd)
    return out


def _render_output(pairs: list[dict]) -> np.ndarray:
    ordered = sorted(
        pairs,
        key=lambda item: (-item["area"], -item["height"], -item["width"], item["shape_key"]),
    )
    out_h = sum(int(item["height"]) for item in ordered) + max(0, len(ordered) - 1)
    out_w = max(int(item["width"]) for item in ordered)
    out = np.zeros((out_h, out_w), dtype=int)
    cursor = 0
    for idx, item in enumerate(ordered):
        patch = _checkerboard(item["height"], item["width"], item["color_even"], item["color_odd"])
        out[cursor : cursor + item["height"], : item["width"]] = patch
        cursor += int(item["height"])
        if idx + 1 < len(ordered):
            cursor += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary colored fragments into full rectangles, then repaint each rectangle as a checkerboard.

    The solver reconstructs every unique rectangle pair under rotations/flips,
    normalizes each rectangle so height <= width, then fills it as a checkerboard.
    Even cells use the lower-numbered color from the matched pair, matching the prompt.
    """
    components = _extract_components(np.asarray(grid, dtype=int))
    pairs = _unique_pairing(components)
    return _render_output(pairs)


def _random_partition(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(160):
        transpose = bool(random.getrandbits(1))
        base_h, base_w = (rect_w, rect_h) if transpose else (rect_h, rect_w)
        if base_w < 3:
            return None

        if base_w >= 5:
            first = random.randint(1, base_w - 3)
        else:
            first = random.randint(0, base_w - 2)
        cuts = [first]
        for _row in range(1, base_h):
            prev = cuts[-1]
            lo = max(0, prev - 1)
            hi = min(base_w - 2, prev + 1)
            cuts.append(random.randint(lo, hi))

        if len(set(cuts)) == 1:
            continue

        mask_a = np.zeros((base_h, base_w), dtype=bool)
        for r, cut in enumerate(cuts):
            mask_a[r, : cut + 1] = True
        mask_b = ~mask_a
        if transpose:
            mask_a = mask_a.T
            mask_b = mask_b.T

        crop_a = _crop_mask(mask_a)
        crop_b = _crop_mask(mask_b)
        if min(crop_a.shape) < 2 or min(crop_b.shape) < 2:
            continue
        if _is_solid_rect(crop_a) and _is_solid_rect(crop_b):
            continue
        return mask_a, mask_b
    return None


def _piece_from_full_mask(mask: np.ndarray, color: int) -> dict:
    flip_lr, rot_k = random.choice(OPS)
    transformed = _transform_mask(mask, flip_lr, rot_k)
    return {
        "mask": transformed,
        "color": int(color),
        "shape_key": _canonical_shape_key(transformed),
    }


def _place_pieces(pieces: list[dict], grid_h: int, grid_w: int) -> np.ndarray | None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    order = list(range(len(pieces)))
    random.shuffle(order)

    for idx in order:
        piece = pieces[idx]
        mask = piece["mask"]
        ph, pw = mask.shape
        candidates = [(r, c) for r in range(grid_h - ph + 1) for c in range(grid_w - pw + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            rr0 = max(0, r0 - 1)
            rr1 = min(grid_h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(grid_w, c0 + pw + 1)
            if bool(occupied[rr0:rr1, cc0:cc1].any()):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = int(piece["color"])
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
            placed = True
            break
        if not placed:
            return None
    return grid


def _pair_specs() -> list[dict] | None:
    pair_count = random.randint(2, 3)
    dims_by_area: dict[int, list[tuple[int, int]]] = {}
    for rect_h in range(3, 7):
        for rect_w in range(3, 7):
            area = rect_h * rect_w
            if 9 <= area <= 24:
                dims_by_area.setdefault(area, []).append((rect_h, rect_w))
    if len(dims_by_area) < pair_count:
        return None

    chosen_areas = random.sample(sorted(dims_by_area), pair_count)
    colors = random.sample(list(range(1, 10)), 2 * pair_count)
    specs: list[dict] = []

    for pair_idx, area in enumerate(chosen_areas):
        rect_h, rect_w = random.choice(dims_by_area[area])
        partition = _random_partition(rect_h, rect_w)
        if partition is None:
            return None
        mask_a, mask_b = partition
        low_color, high_color = sorted(colors[2 * pair_idx : 2 * pair_idx + 2])
        top_left_in_a = bool(mask_a[0, 0])
        color_a = low_color if top_left_in_a else high_color
        color_b = high_color if top_left_in_a else low_color
        piece_a = _piece_from_full_mask(mask_a, color_a)
        piece_b = _piece_from_full_mask(mask_b, color_b)
        out_h, out_w = sorted((int(rect_h), int(rect_w)))
        specs.append(
            {
                "height": out_h,
                "width": out_w,
                "area": int(area),
                "color_even": int(low_color),
                "color_odd": int(high_color),
                "shape_key": tuple(sorted((piece_a["shape_key"], piece_b["shape_key"]))),
                "pieces": [piece_a, piece_b],
            }
        )
    return specs


def _generate_attempt() -> np.ndarray | None:
    specs = _pair_specs()
    if specs is None:
        return None
    pieces = [piece for spec in specs for piece in spec["pieces"]]
    max_piece_h = max(int(piece["mask"].shape[0]) for piece in pieces)
    max_piece_w = max(int(piece["mask"].shape[1]) for piece in pieces)
    min_h = max(18, max_piece_h + 2)
    min_w = max(18, max_piece_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(30):
        grid_h = random.randint(min_h, 30)
        grid_w = random.randint(min_w, 30)
        grid = _place_pieces(pieces, grid_h, grid_w)
        if grid is None:
            continue
        components = _extract_components(grid)
        if len(components) != len(pieces):
            continue
        expected = _render_output(
            [
                {
                    "height": spec["height"],
                    "width": spec["width"],
                    "area": spec["area"],
                    "color_even": spec["color_even"],
                    "color_odd": spec["color_odd"],
                    "shape_key": spec["shape_key"],
                }
                for spec in specs
            ]
        )
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
