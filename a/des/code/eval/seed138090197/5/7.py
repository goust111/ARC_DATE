import random
from collections import Counter

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                cell = (rr, cc)
                if cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(sorted(coords), dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return comps


def _side_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    left[:, : (w + 1) // 2] = True
    right[:, w // 2 :] = True
    return left, right


def _covers_side(mask: np.ndarray, full: np.ndarray, side_mask: np.ndarray) -> bool:
    return bool(np.all(~full[side_mask] | mask[side_mask]))


def _missing_on_side(mask: np.ndarray, full: np.ndarray, side_mask: np.ndarray) -> int:
    return int(np.count_nonzero(full[side_mask] & ~mask[side_mask]))


def _completion(main_mask: np.ndarray, patch_mask: np.ndarray) -> np.ndarray | None:
    if main_mask.shape != patch_mask.shape:
        return None

    full = main_mask | np.fliplr(patch_mask)
    if not full.any() or not np.array_equal(full, np.fliplr(full)):
        return None
    if not _is_connected(full):
        return None
    if np.any(main_mask & ~full) or np.any(patch_mask & ~full):
        return None
    if not np.array_equal(full, main_mask | patch_mask):
        return None
    if np.array_equal(main_mask, full) or np.array_equal(patch_mask, full):
        return None

    left_mask, right_mask = _side_masks(full.shape)
    left_role = (
        _covers_side(main_mask, full, left_mask)
        and _covers_side(patch_mask, full, right_mask)
        and _missing_on_side(main_mask, full, right_mask) > 0
        and _missing_on_side(patch_mask, full, left_mask) > 0
    )
    right_role = (
        _covers_side(main_mask, full, right_mask)
        and _covers_side(patch_mask, full, left_mask)
        and _missing_on_side(main_mask, full, left_mask) > 0
        and _missing_on_side(patch_mask, full, right_mask) > 0
    )
    if not (left_role or right_role):
        return None
    return full


def _analyze_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray | None:
    full_ab = _completion(mask_a, mask_b)
    if full_ab is None:
        return None
    full_ba = _completion(mask_b, mask_a)
    if full_ba is None or not np.array_equal(full_ab, full_ba):
        return None
    return full_ab


def _build_pair_info(comps: list[dict]) -> dict[tuple[int, int], np.ndarray]:
    info: dict[tuple[int, int], np.ndarray] = {}
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            full = _analyze_pair(comps[i]["mask"], comps[j]["mask"])
            if full is not None:
                info[(i, j)] = full
    return info


def _candidate_partners(index: int, remaining: tuple[int, ...], pair_info: dict[tuple[int, int], np.ndarray]) -> list[int]:
    partners = []
    for other in remaining:
        if other == index:
            continue
        key = (index, other) if index < other else (other, index)
        if key in pair_info:
            partners.append(other)
    return partners


def _find_matching(
    remaining: tuple[int, ...], pair_info: dict[tuple[int, int], np.ndarray]
) -> list[tuple[int, int]] | None:
    if not remaining:
        return []

    best = None
    best_partners = None
    for idx in remaining:
        partners = _candidate_partners(idx, remaining, pair_info)
        if not partners:
            return None
        if best is None or len(partners) < len(best_partners):
            best = idx
            best_partners = partners

    assert best is not None and best_partners is not None
    rest = tuple(x for x in remaining if x != best)
    for other in best_partners:
        reduced = tuple(x for x in rest if x != other)
        result = _find_matching(reduced, pair_info)
        if result is not None:
            pair = (best, other) if best < other else (other, best)
            return [pair] + result
    return None


def _count_matchings(
    remaining: tuple[int, ...], pair_info: dict[tuple[int, int], np.ndarray], limit: int = 2
) -> int:
    if not remaining:
        return 1

    best = None
    best_partners = None
    for idx in remaining:
        partners = _candidate_partners(idx, remaining, pair_info)
        if not partners:
            return 0
        if best is None or len(partners) < len(best_partners):
            best = idx
            best_partners = partners

    assert best is not None and best_partners is not None
    total = 0
    rest = tuple(x for x in remaining if x != best)
    for other in best_partners:
        reduced = tuple(x for x in rest if x != other)
        total += _count_matchings(reduced, pair_info, limit=limit)
        if total >= limit:
            return total
    return total


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background(g)
    comps = _find_components(g, bg)
    assert len(comps) % 2 == 0 and comps

    pair_info = _build_pair_info(comps)
    matching = _find_matching(tuple(range(len(comps))), pair_info)
    assert matching is not None and len(matching) * 2 == len(comps)

    out = g.copy()
    for i, j in matching:
        comp_i = comps[i]
        comp_j = comps[j]
        full = pair_info[(i, j)]
        assert comp_i["area"] != comp_j["area"]
        main_comp = comp_i if comp_i["area"] > comp_j["area"] else comp_j
        patch_comp = comp_j if main_comp is comp_i else comp_i

        out[main_comp["coords"][:, 0], main_comp["coords"][:, 1]] = bg
        out[patch_comp["coords"][:, 0], patch_comp["coords"][:, 1]] = bg

        r0, c0, r1, c1 = main_comp["bbox"]
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[full] = int(main_comp["color"])

    return out, pair_info, comps


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)[0]


GEN_BG = 0
GEN_GRID_MIN = 16
GEN_GRID_MAX = 30
GEN_PAIRS = (1, 2, 3, 4)
GEN_HALF_H = (4, 9)
GEN_HALF_W = (2, 5)
GEN_PLACEMENT_PAD = 1
GEN_RETRIES = 400


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _random_half_mask() -> np.ndarray | None:
    h = random.randint(*GEN_HALF_H)
    w = random.randint(*GEN_HALF_W)
    max_area = h * w
    min_area = max(5, h // 2 + 2)
    if min_area > max_area:
        return None
    target = random.randint(min_area, max_area)

    axis_len = random.randint(2, h)
    axis_top = random.randint(0, h - axis_len)
    mask = np.zeros((h, w), dtype=bool)
    mask[axis_top : axis_top + axis_len, w - 1] = True

    while int(mask.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in DIRS4:
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.add((rr, cc))
        if not frontier:
            break
        rr, cc = random.choice(list(frontier))
        mask[rr, cc] = True

    mask = _crop_mask(mask)
    if int(mask.sum()) < 5 or np.count_nonzero(mask[:, -1]) < 2 or not _is_connected(mask):
        return None
    return mask


def _subset_mask(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _build_pair_spec() -> dict | None:
    left = _random_half_mask()
    if left is None:
        return None

    coords = [tuple(map(int, pt)) for pt in np.argwhere(left)]
    axis_cells = [cell for cell in coords if cell[1] == left.shape[1] - 1]
    if len(coords) < 5 or len(axis_cells) < 2:
        return None

    left_larger = random.choice([True, False])
    if left_larger:
        lo, hi = len(coords) // 2 + 1, len(coords) - 2
    else:
        lo, hi = 2, (len(coords) - 1) // 2
    if lo > hi:
        return None
    subset_size = random.randint(lo, hi)

    anchor_a, anchor_b = random.sample(axis_cells, 2)
    remaining = [cell for cell in coords if cell not in {anchor_a, anchor_b}]
    subset_a = {anchor_a}
    need = subset_size - 1
    if need > len(remaining):
        return None
    if need:
        subset_a.update(random.sample(remaining, need))
    subset_b = set(coords) - subset_a
    if not subset_b or not any(c == left.shape[1] - 1 for _r, c in subset_b):
        return None

    sa_mask = _subset_mask(left.shape, subset_a)
    sb_mask = _subset_mask(left.shape, subset_b)
    right = np.fliplr(left)

    h, w = left.shape
    full = np.zeros((h, 2 * w), dtype=bool)
    left_full = np.zeros_like(full)
    right_full = np.zeros_like(full)
    left_full[:, :w] = left
    left_full[:, w:] = np.fliplr(sa_mask)
    right_full[:, :w] = sb_mask
    right_full[:, w:] = right
    full[:, :w] = left
    full[:, w:] = right

    if not (_is_connected(full) and _is_connected(left_full) and _is_connected(right_full)):
        return None
    if int(left_full.sum()) == int(right_full.sum()):
        return None

    return {
        "full": full,
        "left_fragment": left_full,
        "right_fragment": right_full,
        "left_area": int(left_full.sum()),
        "right_area": int(right_full.sum()),
        "shape_sig": _mask_signature(full),
    }


def _place_rectangles(shapes: list[tuple[int, int]], h: int, w: int, pad: int) -> list[tuple[int, int]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)

    for idx in order:
        sh, sw = shapes[idx]
        candidates = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        random.shuffle(candidates)
        picked = None
        for r, c in candidates:
            if blocked[r : r + sh, c : c + sw].any():
                continue
            picked = (r, c)
            r0 = max(0, r - pad)
            c0 = max(0, c - pad)
            r1 = min(h, r + sh + pad)
            c1 = min(w, c + sw + pad)
            blocked[r0:r1, c0:c1] = True
            break
        if picked is None:
            return None
        positions[idx] = picked

    return [pos for pos in positions if pos is not None]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    view = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    view[mask] = int(color)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(GEN_PAIRS)
    colors = random.sample(range(1, 10), 2 * pair_count)

    specs = []
    seen_shapes = set()
    for _ in range(pair_count):
        spec = None
        for _inner in range(80):
            spec = _build_pair_spec()
            if spec is None or spec["shape_sig"] in seen_shapes:
                spec = None
                continue
            seen_shapes.add(spec["shape_sig"])
            break
        if spec is None:
            return None
        specs.append(spec)

    max_h = max(spec["full"].shape[0] for spec in specs)
    max_w = max(spec["full"].shape[1] for spec in specs)
    dim_floor = max(GEN_GRID_MIN, max_h + 2, max_w + 2, 12 + 3 * (pair_count - 1))
    if dim_floor > GEN_GRID_MAX:
        return None
    h = random.randint(dim_floor, GEN_GRID_MAX)
    w = random.randint(dim_floor, GEN_GRID_MAX)

    objects = []
    for pair_idx, spec in enumerate(specs):
        left_color = colors[2 * pair_idx]
        right_color = colors[2 * pair_idx + 1]
        objects.append(
            {
                "pair": pair_idx,
                "kind": "left",
                "mask": spec["left_fragment"],
                "color": left_color,
                "is_main": spec["left_area"] > spec["right_area"],
            }
        )
        objects.append(
            {
                "pair": pair_idx,
                "kind": "right",
                "mask": spec["right_fragment"],
                "color": right_color,
                "is_main": spec["right_area"] > spec["left_area"],
            }
        )

    shapes = [obj["mask"].shape for obj in objects]
    positions = _place_rectangles(shapes, h, w, pad=GEN_PLACEMENT_PAD)
    if positions is None:
        return None

    inp = np.full((h, w), GEN_BG, dtype=int)
    out = np.full((h, w), GEN_BG, dtype=int)
    for obj, (top, left) in zip(objects, positions):
        obj["top"] = top
        obj["left"] = left
        _paint_mask(inp, obj["mask"], top, left, obj["color"])

    for pair_idx, spec in enumerate(specs):
        mains = [obj for obj in objects if obj["pair"] == pair_idx and obj["is_main"]]
        if len(mains) != 1:
            return None
        main_obj = mains[0]
        _paint_mask(out, spec["full"], main_obj["top"], main_obj["left"], main_obj["color"])

    try:
        solved, pair_info, comps = _solve(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if len(comps) != 2 * pair_count:
        return None
    if _count_matchings(tuple(range(len(comps))), pair_info, limit=2) != 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
