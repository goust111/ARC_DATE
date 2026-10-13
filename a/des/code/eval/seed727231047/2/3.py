import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_NON_BG_COLORS = tuple(range(1, 10))
_NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_CANDIDATE_DIMS = (
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 3),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 4),
    (4, 5),
    (5, 5),
)
_MAX_GENERATION_ATTEMPTS = 300


def _component_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    return [(int(r), int(c)) for r, c in pts]


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start_r, start_c = map(int, pts[0])
    seen = {(start_r, start_c)}
    stack = [(start_r, start_c)]
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS8:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_valid_pair_mask(mask: np.ndarray) -> bool:
    other = ~mask
    return bool(
        mask.any()
        and other.any()
        and _touches_all_sides(mask)
        and _touches_all_sides(other)
        and _is_connected(mask)
        and _is_connected(other)
    )


def _random_pair_mask(h: int, w: int) -> np.ndarray:
    mask = (np.indices((h, w)).sum(axis=0) + int(_RNG.integers(0, 2))) % 2 == 0
    n_steps = int(_RNG.integers(h * w, 3 * h * w + 1))
    for _ in range(n_steps):
        r = int(_RNG.integers(0, h))
        c = int(_RNG.integers(0, w))
        cand = mask.copy()
        cand[r, c] = ~cand[r, c]
        if _is_valid_pair_mask(cand):
            mask = cand
    assert _is_valid_pair_mask(mask)
    return mask


def _extract_components(grid: np.ndarray, bg: int = _BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or bool(seen[r0, c0]):
                continue

            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in _NEIGHBORS8:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(seen[nr, nc]) or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            min_r = min(rs)
            max_r = max(rs)
            min_c = min(cs)
            max_c = max(cs)
            mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
            for r, c in coords:
                mask[r - min_r, c - min_c] = True

            comps.append(
                {
                    "color": color,
                    "bbox": (int(min_r), int(min_c), int(max_r), int(max_c)),
                    "anchor": (int(min_r), int(min_c)),
                    "mask": mask,
                    "shape": tuple(map(int, mask.shape)),
                }
            )

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary monochrome fragments, merge each pair into a bicolor rectangle,
    then stack the rectangles from smallest area to largest.
    """
    g = np.asarray(grid, dtype=int)
    comps = _extract_components(g, bg=_BG)
    assert comps and len(comps) % 2 == 0

    shape_groups: dict[tuple[int, int], list[int]] = {}
    for idx, comp in enumerate(comps):
        shape_groups.setdefault(comp["shape"], []).append(int(idx))

    pairs: list[dict] = []
    for shape, idxs in shape_groups.items():
        mask_to_indices: dict[bytes, list[int]] = {}
        for idx in idxs:
            key = comps[idx]["mask"].tobytes()
            mask_to_indices.setdefault(key, []).append(int(idx))

        used: set[int] = set()
        ordered = sorted(idxs, key=lambda i: comps[i]["anchor"])
        for idx in ordered:
            if idx in used:
                continue
            comp = comps[idx]
            complement_key = (~comp["mask"]).tobytes()
            candidates = [
                j
                for j in mask_to_indices.get(complement_key, [])
                if j not in used and j != idx
            ]
            assert candidates, "every fragment must have an exact complementary partner"
            mate = min(candidates, key=lambda j: comps[j]["anchor"])
            used.add(int(idx))
            used.add(int(mate))
            assert int(comps[idx]["color"]) != int(comps[mate]["color"])

            block = np.full(shape, _BG, dtype=int)
            block[comps[idx]["mask"]] = int(comps[idx]["color"])
            block[comps[mate]["mask"]] = int(comps[mate]["color"])
            assert bool((block != _BG).all())
            pairs.append(
                {
                    "block": block,
                    "shape": shape,
                    "area": int(shape[0] * shape[1]),
                    "anchor": min(comps[idx]["anchor"], comps[mate]["anchor"]),
                }
            )

        assert len(used) == len(idxs)

    assert len(pairs) * 2 == len(comps)
    pairs.sort(key=lambda item: (item["area"], item["shape"][0], item["shape"][1], item["anchor"]))

    out_h = sum(int(item["shape"][0]) for item in pairs) + len(pairs) - 1
    out_w = max(int(item["shape"][1]) for item in pairs)
    out = np.full((out_h, out_w), _BG, dtype=int)

    r = 0
    for item in pairs:
        block = item["block"]
        h, w = map(int, block.shape)
        out[r : r + h, :w] = block
        r += h + 1

    return out


def _bbox_fits(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    return not bool(occupied[top : top + h, left : left + w].any())


def _mark_bbox_with_margin(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    h0, w0 = occupied.shape
    r0 = max(0, int(top - 1))
    c0 = max(0, int(left - 1))
    r1 = min(h0, int(top + h + 1))
    c1 = min(w0, int(left + w + 1))
    occupied[r0:r1, c0:c1] = True


def _place_rectangles(rects: list[dict], H: int, W: int) -> bool:
    occupied = np.zeros((H, W), dtype=bool)
    for rect in sorted(rects, key=lambda item: item["shape"][0] * item["shape"][1], reverse=True):
        h, w = map(int, rect["shape"])
        slots = [
            (r, c)
            for r in range(H - h + 1)
            for c in range(W - w + 1)
            if _bbox_fits(occupied, r, c, h, w)
        ]
        if not slots:
            return False
        top, left = slots[int(_RNG.integers(0, len(slots)))]
        rect["top"] = int(top)
        rect["left"] = int(left)
        _mark_bbox_with_margin(occupied, top, left, h, w)
    return True


def _build_input_grid(rects: list[dict], H: int, W: int) -> np.ndarray:
    grid = np.full((H, W), _BG, dtype=int)
    for rect in rects:
        top = int(rect["top"])
        left = int(rect["left"])
        mask = rect["mask"]
        color = int(rect["color"])
        sub = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
        sub[mask] = color
    return grid


def _generate_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.integers(2, 5))
    dims_idx = _RNG.choice(len(_CANDIDATE_DIMS), size=pair_count, replace=False)
    dims = [_CANDIDATE_DIMS[int(i)] for i in dims_idx]
    colors = _RNG.choice(_NON_BG_COLORS, size=2 * pair_count, replace=False)

    rects: list[dict] = []
    for pair_idx, (h, w) in enumerate(dims):
        mask = _random_pair_mask(int(h), int(w))
        color_a = int(colors[2 * pair_idx])
        color_b = int(colors[2 * pair_idx + 1])
        if bool(_RNG.integers(0, 2)):
            mask_a = mask
        else:
            mask_a = ~mask
        mask_b = ~mask_a
        rects.append({"shape": (int(h), int(w)), "mask": mask_a, "color": color_a})
        rects.append({"shape": (int(h), int(w)), "mask": mask_b, "color": color_b})

    total_bbox_area = sum(int(h * w) for h, w in dims)
    max_h = max(int(h) for h, _ in dims)
    max_w = max(int(w) for _, w in dims)
    min_side = max(14, max_h + 4, max_w + 4, int(np.ceil(np.sqrt(total_bbox_area * 3))))
    H = int(_RNG.integers(min_side, 31))
    W = int(_RNG.integers(min_side, 31))
    if not _place_rectangles(rects, H, W):
        return None

    grid = _build_input_grid(rects, H, W)
    comps = _extract_components(grid, bg=_BG)
    if len(comps) != len(rects):
        return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
