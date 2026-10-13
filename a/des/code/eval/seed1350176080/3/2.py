from __future__ import annotations

from collections import Counter, deque

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_RNG = np.random.default_rng()

# --------------- solver helpers ---------------


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    rs = coords[:, 0]
    cs = coords[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rs - r0, cs - c0] = True
    return mask


def _true_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            arr = np.asarray(cand, dtype=bool)
            key = (tuple(map(int, arr.shape)), arr.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(arr.copy())
    return variants


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(np.sqrt(area)) + 1):
        if area % h == 0:
            w = area // h
            if h <= w:
                pairs.append((int(h), int(w)))
    return pairs


def _fills_rectangle(mask_a: np.ndarray, mask_b_keys: set[tuple[tuple[int, int], bytes]], h: int, w: int) -> bool:
    ha, wa = map(int, mask_a.shape)
    if ha > h or wa > w:
        return False

    full = np.ones((h, w), dtype=bool)
    for r0 in range(h - ha + 1):
        for c0 in range(w - wa + 1):
            placed = np.zeros((h, w), dtype=bool)
            placed[r0 : r0 + ha, c0 : c0 + wa] = mask_a
            comp = full & (~placed)
            bbox = _true_bbox(comp)
            if bbox is None:
                continue
            br0, bc0, br1, bc1 = bbox
            cropped = comp[br0 : br1 + 1, bc0 : bc1 + 1]
            key = (tuple(map(int, cropped.shape)), cropped.tobytes())
            if key in mask_b_keys:
                return True
    return False


def _rectangle_solutions(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int]]:
    area = int(mask_a.sum() + mask_b.sum())
    b_keys = {
        (tuple(map(int, variant.shape)), variant.tobytes())
        for variant in _all_transforms(mask_b)
    }

    solutions: set[tuple[int, int]] = set()
    for h, w in _factor_pairs(area):
        for variant_a in _all_transforms(mask_a):
            if _fills_rectangle(variant_a, b_keys, h, w):
                solutions.add((int(h), int(w)))
                break
    return sorted(solutions)


def _reconstruct_rectangle(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    solutions = _rectangle_solutions(mask_a, mask_b)
    assert solutions, "fragment pair must reconstruct into at least one rectangle"
    return solutions[0]


def _extract_fragment_pairs(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        comps = _label_components(grid == int(color))
        assert len(comps) == 2, "each non-background color must form exactly two fragments"
        masks = [_coords_to_mask(coords) for coords in comps]
        rect = _reconstruct_rectangle(masks[0], masks[1])
        pairs.append(
            {
                "color": int(color),
                "masks": masks,
                "dims": rect,
                "area": int(rect[0] * rect[1]),
            }
        )
    assert pairs, "input must contain at least one fragment pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-color fragments into rectangles, then stack the rectangles concentrically by area."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    pairs = _extract_fragment_pairs(g, bg)

    pairs.sort(key=lambda item: (int(item["area"]), *map(int, item["dims"])), reverse=True)
    out_h, out_w = map(int, pairs[0]["dims"])
    out = np.full((out_h, out_w), int(pairs[0]["color"]), dtype=int)

    for item in pairs[1:]:
        h, w = map(int, item["dims"])
        r0 = (out_h - h) // 2
        c0 = (out_w - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = int(item["color"])

    return out


# --------------- generator helpers ---------------

PALETTE = tuple(range(10))
PAIR_COUNT_RANGE = (2, 4)  # inclusive
SMALLEST_H_RANGE = (3, 5)
SMALLEST_W_DELTA_RANGE = (0, 4)
GROWTH_STEP_RANGE = (1, 2)
MAX_RECT_H = 11
MAX_RECT_W = 15
MIN_PART_AREA = 3
CANVAS_SIDE_MAX = 30
CANVAS_SLACK = 5
PLACEMENT_PADDING = 1
GENERATION_RETRIES = 400


def _sample_rectangles() -> list[tuple[int, int]] | None:
    for _ in range(120):
        n = int(_RNG.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
        max_small_h = min(SMALLEST_H_RANGE[1], MAX_RECT_H - 2 * (n - 1))
        if max_small_h < SMALLEST_H_RANGE[0]:
            continue

        h = int(_RNG.integers(SMALLEST_H_RANGE[0], max_small_h + 1))
        min_small_w = h + SMALLEST_W_DELTA_RANGE[0]
        max_small_w = min(h + SMALLEST_W_DELTA_RANGE[1], MAX_RECT_W - 2 * (n - 1))
        if max_small_w < min_small_w:
            continue
        w = int(_RNG.integers(min_small_w, max_small_w + 1))

        sizes = [(h, w)]
        ok = True
        while len(sizes) < n:
            prev_h, prev_w = sizes[-1]
            remain = n - len(sizes) - 1
            h_choices = [
                prev_h + 2 * dh
                for dh in range(GROWTH_STEP_RANGE[0], GROWTH_STEP_RANGE[1] + 1)
                if prev_h + 2 * dh <= MAX_RECT_H - 2 * remain
            ]
            w_choices = [
                prev_w + 2 * dw
                for dw in range(GROWTH_STEP_RANGE[0], GROWTH_STEP_RANGE[1] + 1)
                if prev_w + 2 * dw <= MAX_RECT_W - 2 * remain
            ]
            next_sizes = [(nh, nw) for nh in h_choices for nw in w_choices if nh <= nw]
            if not next_sizes:
                ok = False
                break
            pick = next_sizes[int(_RNG.integers(0, len(next_sizes)))]
            sizes.append((int(pick[0]), int(pick[1])))

        if not ok:
            continue

        areas = [h0 * w0 for h0, w0 in sizes]
        if len(set(areas)) == len(areas):
            return list(reversed(sizes))
    return None


def _walk_cuts(length: int, upper: int) -> np.ndarray:
    cuts = np.empty(length, dtype=int)
    cuts[0] = int(_RNG.integers(1, upper + 1))
    for i in range(1, length):
        step_choices = [-1, 0, 1]
        if cuts[i - 1] == 1:
            step_choices = [0, 1]
        elif cuts[i - 1] == upper:
            step_choices = [-1, 0]
        step = int(step_choices[int(_RNG.integers(0, len(step_choices)))])
        cuts[i] = int(np.clip(cuts[i - 1] + step, 1, upper))
    return cuts


def _make_fragment_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        axis = "vertical" if bool(_RNG.integers(0, 2)) else "horizontal"
        if axis == "vertical":
            cuts = _walk_cuts(h, w - 1)
            if len(set(map(int, cuts.tolist()))) == 1:
                continue
            mask_a = np.zeros((h, w), dtype=bool)
            for r, cut in enumerate(cuts):
                mask_a[int(r), : int(cut)] = True
        else:
            cuts = _walk_cuts(w, h - 1)
            if len(set(map(int, cuts.tolist()))) == 1:
                continue
            mask_a = np.zeros((h, w), dtype=bool)
            for c, cut in enumerate(cuts):
                mask_a[: int(cut), int(c)] = True

        mask_b = ~mask_a
        if int(mask_a.sum()) < MIN_PART_AREA or int(mask_b.sum()) < MIN_PART_AREA:
            continue

        bbox_a = _true_bbox(mask_a)
        bbox_b = _true_bbox(mask_b)
        assert bbox_a is not None and bbox_b is not None
        cropped_a = mask_a[bbox_a[0] : bbox_a[2] + 1, bbox_a[1] : bbox_a[3] + 1]
        cropped_b = mask_b[bbox_b[0] : bbox_b[2] + 1, bbox_b[1] : bbox_b[3] + 1]
        sols = _rectangle_solutions(cropped_a, cropped_b)
        if sols == [(int(h), int(w))]:
            return cropped_a, cropped_b
    raise RuntimeError("failed to build an unambiguous fragment pair")


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, int(_RNG.integers(0, 4)))
    if bool(_RNG.integers(0, 2)):
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool).copy()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    expanded[1:, :] |= mask[:-1, :]
    expanded[:-1, :] |= mask[1:, :]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[1:, 1:] |= mask[:-1, :-1]
    expanded[1:, :-1] |= mask[:-1, 1:]
    expanded[:-1, 1:] |= mask[1:, :-1]
    expanded[:-1, :-1] |= mask[1:, 1:]
    return expanded


def _render_expected(rectangles: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: (int(item["area"]), *map(int, item["dims"])), reverse=True)
    out_h, out_w = map(int, ordered[0]["dims"])
    out = np.full((out_h, out_w), int(ordered[0]["color"]), dtype=int)
    for item in ordered[1:]:
        h, w = map(int, item["dims"])
        r0 = (out_h - h) // 2
        c0 = (out_w - w) // 2
        out[r0 : r0 + h, c0 : c0 + w] = int(item["color"])
    return out


def _sample_canvas(pieces: list[dict[str, object]]) -> tuple[int, int] | None:
    total_area = sum(int(piece["mask"].sum()) for piece in pieces)
    max_h = max(int(piece["mask"].shape[0]) for piece in pieces)
    max_w = max(int(piece["mask"].shape[1]) for piece in pieces)

    target_area = int(np.ceil(total_area * float(_RNG.uniform(2.8, 3.5))))
    h_low = max(max_h + 4, int(np.ceil(np.sqrt(target_area))) - 2)
    if h_low > CANVAS_SIDE_MAX:
        return None
    h_high = min(CANVAS_SIDE_MAX, h_low + CANVAS_SLACK)
    H = int(_RNG.integers(h_low, h_high + 1))

    w_low = max(max_w + 4, int(np.ceil(target_area / H)))
    if w_low > CANVAS_SIDE_MAX:
        return None
    w_high = min(CANVAS_SIDE_MAX, max(w_low + CANVAS_SLACK, h_low + 1))
    W = int(_RNG.integers(w_low, w_high + 1))

    if bool(_RNG.integers(0, 2)):
        H, W = W, H
    if H > CANVAS_SIDE_MAX or W > CANVAS_SIDE_MAX:
        return None
    if H * W <= 2 * total_area:
        return None
    return int(H), int(W)


def _place_pieces(pieces: list[dict[str, object]], bg: int) -> np.ndarray | None:
    canvas = _sample_canvas(pieces)
    if canvas is None:
        return None
    H, W = canvas
    grid = np.full((H, W), int(bg), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    order = list(range(len(pieces)))
    order.sort(
        key=lambda i: (
            int(pieces[i]["mask"].shape[0]) * int(pieces[i]["mask"].shape[1]),
            int(pieces[i]["mask"].sum()),
        ),
        reverse=True,
    )

    for idx in order:
        piece = pieces[idx]
        mask = np.asarray(piece["mask"], dtype=bool)
        color = int(piece["color"])
        h, w = map(int, mask.shape)
        candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        perm = _RNG.permutation(len(candidates))
        dilated = _dilate8(occupied) if PLACEMENT_PADDING else occupied

        placed = False
        for cand_idx in perm:
            r0, c0 = candidates[int(cand_idx)]
            if bool((dilated[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                continue
            view = grid[r0 : r0 + h, c0 : c0 + w]
            view[mask] = color
            occupied[r0 : r0 + h, c0 : c0 + w] |= mask
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    rectangles = _sample_rectangles()
    if rectangles is None:
        return None
    colors = list(PALETTE)
    bg = int(_RNG.choice(colors))
    colors.remove(bg)
    pair_colors = [int(c) for c in _RNG.choice(colors, size=len(rectangles), replace=False)]

    rect_specs: list[dict[str, object]] = []
    pieces: list[dict[str, object]] = []
    for color, (h, w) in zip(pair_colors, rectangles):
        mask_a, mask_b = _make_fragment_pair(int(h), int(w))
        rect_specs.append({"color": int(color), "dims": (int(h), int(w)), "area": int(h * w)})
        for raw_mask in (mask_a, mask_b):
            pieces.append({"color": int(color), "mask": _random_transform(raw_mask)})

    expected = _render_expected(rect_specs)
    grid = _place_pieces(pieces, bg)
    if grid is None:
        return None

    try:
        return grid if np.array_equal(main(grid), expected) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
