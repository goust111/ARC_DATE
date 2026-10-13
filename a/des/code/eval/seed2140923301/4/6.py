from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
ORANGE = 7
MAROON = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 300
COLOR_POOL = [1, 2, 3, 4, 5, 6, 8]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _scale_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((k, k), dtype=np.uint8)).astype(bool)


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict[str, object]] = []
    for comp in _components(g != BLACK):
        colors = {int(g[r, c]) for r, c in comp}
        if MAROON not in colors:
            continue
        marker_cells = [(r, c) for r, c in comp if int(g[r, c]) == MAROON]
        if len(marker_cells) != 1:
            continue
        base_colors = colors - {MAROON}
        if len(base_colors) != 1:
            continue
        base = int(next(iter(base_colors)))
        mask, bbox = _coords_to_mask(comp)
        r0, c0, _r1, _c1 = bbox
        mr, mc = marker_cells[0]
        objects.append(
            {
                "coords": comp,
                "coord_set": set(comp),
                "mask": mask,
                "bbox": bbox,
                "color": base,
                "marker": (mr, mc),
                "marker_rel": (mr - r0, mc - c0),
                "area": len(comp),
            }
        )
    return objects


def _scale_relation(
    obj_a: dict[str, object], obj_b: dict[str, object]
) -> tuple[int, int, int] | None:
    mask_a = obj_a["mask"]
    mask_b = obj_b["mask"]
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape

    def _check(smaller_idx: int, bigger_idx: int, small_mask: np.ndarray, big_mask: np.ndarray):
        sh, sw = small_mask.shape
        bh, bw = big_mask.shape
        if bh % sh != 0 or bw % sw != 0:
            return None
        k = bh // sh
        if k <= 1 or bw // sw != k:
            return None
        if np.array_equal(_scale_mask(small_mask, k), big_mask):
            return smaller_idx, bigger_idx, k
        return None

    rel = _check(0, 1, mask_a, mask_b)
    if rel is not None:
        return rel
    rel = _check(1, 0, mask_b, mask_a)
    if rel is not None:
        return rel
    return None


def _pair_group(group: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object], int]]:
    n = len(group)
    candidates: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for i, j in combinations(range(n), 2):
        rel = _scale_relation(group[i], group[j])
        if rel is None:
            continue
        small_side, big_side, k = rel
        if small_side == 0:
            candidates[i].append((j, k))
        else:
            candidates[j].append((i, k))

    order = sorted(range(n), key=lambda idx: int(group[idx]["area"]))
    best: list[tuple[int, int, int]] = []

    def dfs(pos: int, used: set[int], current: list[tuple[int, int, int]]) -> None:
        nonlocal best
        if pos == len(order):
            if len(current) > len(best):
                best = current.copy()
            return
        idx = order[pos]
        if idx in used:
            dfs(pos + 1, used, current)
            return

        dfs(pos + 1, used | {idx}, current)
        for other, k in candidates.get(idx, []):
            if other in used:
                continue
            current.append((idx, other, k))
            dfs(pos + 1, used | {idx, other}, current)
            current.pop()

    dfs(0, set(), [])
    return [(group[i], group[j], k) for i, j, k in best]


def _find_pairs(objects: list[dict[str, object]]) -> list[tuple[dict[str, object], dict[str, object], int]]:
    by_color: dict[int, list[dict[str, object]]] = {}
    for obj in objects:
        by_color.setdefault(int(obj["color"]), []).append(obj)
    pairs: list[tuple[dict[str, object], dict[str, object], int]] = []
    for group in by_color.values():
        pairs.extend(_pair_group(group))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find same-color object pairs where one is an integer-scaled copy of the other.
    Enlarge the smaller object and overlay it inside the larger one, anchoring the
    overlay by the maroon marker positions. The smaller original object disappears,
    and covered pixels in the larger object become orange.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _extract_objects(g)
    pairs = _find_pairs(objects)

    for small, big, k in pairs:
        for r, c in small["coords"]:
            out[r, c] = BLACK

        scaled_small = _scale_mask(small["mask"], k)
        smr, smc = small["marker_rel"]
        bmr, bmc = big["marker"]

        # The enlarged marker is anchored by the top-left cell of its k x k block.
        top = int(bmr) - int(smr) * int(k)
        left = int(bmc) - int(smc) * int(k)
        big_cells = big["coord_set"]

        for rr, cc in np.argwhere(scaled_small):
            gr = top + int(rr)
            gc = left + int(cc)
            if (gr, gc) in big_cells:
                out[gr, gc] = ORANGE

    return out


def _random_connected_mask(h: int, w: int, target: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = [start]
    while frontier and len(cells) < target:
        cr, cc = random.choice(frontier)
        nbrs = []
        for dr, dc in DIR4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove((cr, cc))
            continue
        nxt = random.choice(nbrs)
        cells.add(nxt)
        frontier.append(nxt)

    if len(cells) != target:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    coords = list(cells)
    r0, c0, r1, c1 = _bbox(coords)
    if r0 != 0 or c0 != 0 or r1 != h - 1 or c1 != w - 1:
        return None
    if h > 1 and w > 1 and mask.all():
        return None
    return mask


def _choose_shape(num_pairs: int) -> tuple[np.ndarray, int]:
    max_dim = 4 if num_pairs == 3 else 5
    for _ in range(80):
        k = random.choice([2, 3])
        h = random.randint(2, max_dim)
        w = random.randint(2, max_dim)
        if num_pairs == 3 and h * k > 10 and w * k > 10:
            continue
        area_min = max(3, (h * w + 1) // 2)
        area_max = min(h * w, area_min + random.randint(1, max(1, h * w - area_min)))
        target = random.randint(area_min, area_max)
        mask = _random_connected_mask(h, w, target)
        if mask is None:
            continue
        return mask, k
    raise RuntimeError("failed to choose shape")


def _mask_overlap_count(mask: np.ndarray, shift_r: int, shift_c: int) -> int:
    h, w = mask.shape
    total = 0
    for r, c in np.argwhere(mask):
        nr = int(r) + int(shift_r)
        nc = int(c) + int(shift_c)
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            total += 1
    return total


def _choose_marker_positions(mask: np.ndarray, k: int) -> tuple[tuple[int, int], tuple[int, int]]:
    small_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    small_marker = random.choice(small_cells)
    scaled = _scale_mask(mask, k)
    big_cells = [tuple(map(int, rc)) for rc in np.argwhere(scaled)]
    anchor = (small_marker[0] * k, small_marker[1] * k)
    area = len(big_cells)

    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for br, bc in big_cells:
        shift_r = br - anchor[0]
        shift_c = bc - anchor[1]
        overlap = _mask_overlap_count(scaled, shift_r, shift_c)
        if max(2, area // 5) <= overlap < area:
            preferred.append((br, bc))
        else:
            fallback.append((br, bc))

    if preferred and random.random() < 0.85:
        big_marker = random.choice(preferred)
    else:
        big_marker = random.choice(fallback if fallback else big_cells)
    return small_marker, big_marker


def _make_patch(mask: np.ndarray, color: int, marker: tuple[int, int]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    patch[int(marker[0]), int(marker[1])] = MAROON
    return patch


def _can_place(blocked: np.ndarray, top: int, left: int, h: int, w: int, margin: int = 1) -> bool:
    H, W = blocked.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(H, top + h + margin)
    c1 = min(W, left + w + margin)
    return not bool(blocked[r0:r1, c0:c1].any())


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> bool:
    H, W = grid.shape
    h, w = patch.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if not _can_place(blocked, top, left, h, w):
            continue
        sub = grid[top : top + h, left : left + w]
        if bool((sub != BLACK).any()):
            continue
        mask = patch != BLACK
        sub[mask] = patch[mask]
        blocked[max(0, top - 1) : min(H, top + h + 1), max(0, left - 1) : min(W, left + w + 1)] = True
        return True
    return False


def _instance_ok(grid: np.ndarray, expected_pairs: int) -> bool:
    try:
        objects = _extract_objects(grid)
        if len(objects) != expected_pairs * 2:
            return False
        pairs = _find_pairs(objects)
        if len(pairs) != expected_pairs:
            return False
        paired_ids = {id(obj) for pair in pairs for obj in pair[:2]}
        if len(paired_ids) != expected_pairs * 2:
            return False
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    orange_count = int((out == ORANGE).sum())
    if orange_count < expected_pairs:
        return False
    if (grid == ORANGE).any():
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    num_pairs = random.choice([1, 2, 3])
    min_side = 17 + num_pairs * 3
    h = random.randint(min_side, min_side + 6)
    w = random.randint(min_side, min_side + 6)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = random.sample(COLOR_POOL, num_pairs)
    patches: list[np.ndarray] = []
    for color in colors:
        small_mask, k = _choose_shape(num_pairs)
        big_mask = _scale_mask(small_mask, k)
        small_marker, big_marker = _choose_marker_positions(small_mask, k)
        small_patch = _make_patch(small_mask, int(color), small_marker)
        big_patch = _make_patch(big_mask, int(color), big_marker)
        if random.random() < 0.5:
            patches.extend([big_patch, small_patch])
        else:
            patches.extend([small_patch, big_patch])

    patches.sort(key=lambda arr: int(arr.shape[0] * arr.shape[1]), reverse=True)
    for patch in patches:
        if not _place_patch(grid, blocked, patch):
            return None

    if not _instance_ok(grid, num_pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
