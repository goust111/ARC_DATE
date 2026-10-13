from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BG = 4  # yellow
COLORS = [0, 1, 2, 3, 5, 6, 7, 8, 9]
RNG = np.random.default_rng()

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
NUM_PAIRS_CHOICES = [1, 2, 3, 4]
SOURCE_PAD = 1
OUTPUT_PAD = 1
PAIR_RETRIES = 300
GEN_RETRIES = 400

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox_from_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not mask[r, c] and not outside[r, c]:
            outside[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            add(r + dr, c + dc)

    return mask | (~mask & ~outside)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sr, sc = map(int, pts[0])
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    seen[sr, sc] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for dr, dc in N4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return count == int(mask.sum())


def _paste(out: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    out[r0 : r0 + h, c0 : c0 + w][mask] = int(color)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    filled = _fill_holes(mask)
    return filled.shape, filled.tobytes()


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox_from_points(pts)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in pts:
                mask[rr - r0, cc - c0] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "filled": _fill_holes(mask),
                    "area": int(mask.sum()),
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each pair of same-outline objects by sliding them to their midpoint.

    Pairing uses the normalized outer silhouette: enclosed yellow holes are ignored
    for shape matching but still count when comparing visible areas. The smaller-area
    object is drawn on top; ties go to the left object for horizontal pairs or the
    upper object for vertical pairs.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)

    objects = _find_objects(g)
    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for obj in objects:
        groups[_shape_key(obj["mask"])].append(obj)

    for members in groups.values():
        assert len(members) == 2, "each outline shape must appear exactly twice"
        a, b = members
        assert a["color"] != b["color"], "paired objects must have different colors"
        assert a["filled"].shape == b["filled"].shape

        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        same_row_center = int(ar0 + ar1) == int(br0 + br1)
        same_col_center = int(ac0 + ac1) == int(bc0 + bc1)
        assert same_row_center ^ same_col_center, "pair must align horizontally or vertically"

        dest_r0 = (int(ar0) + int(br0)) // 2
        dest_c0 = (int(ac0) + int(bc0)) // 2
        assert (int(ar0) + int(br0)) % 2 == 0
        assert (int(ac0) + int(bc0)) % 2 == 0

        if same_row_center:
            left, right = (a, b) if int(ac0) < int(bc0) else (b, a)
            if int(left["area"]) == int(right["area"]):
                top, bottom = left, right
            elif int(left["area"]) < int(right["area"]):
                top, bottom = left, right
            else:
                top, bottom = right, left
        else:
            upper, lower = (a, b) if int(ar0) < int(br0) else (b, a)
            if int(upper["area"]) == int(lower["area"]):
                top, bottom = upper, lower
            elif int(upper["area"]) < int(lower["area"]):
                top, bottom = upper, lower
            else:
                top, bottom = lower, upper

        _paste(out, bottom["mask"], dest_r0, dest_c0, bottom["color"])
        _paste(out, top["mask"], dest_r0, dest_c0, top["color"])

    return out


def _random_silhouette() -> np.ndarray | None:
    for _ in range(120):
        h = int(RNG.integers(4, 8))
        w = int(RNG.integers(4, 8))
        mask = np.zeros((h, w), dtype=bool)

        core_h = int(RNG.integers(3, h + 1))
        core_w = int(RNG.integers(3, w + 1))
        core_r = int(RNG.integers(0, h - core_h + 1))
        core_c = int(RNG.integers(0, w - core_w + 1))
        mask[core_r : core_r + core_h, core_c : core_c + core_w] = True

        sides = list(RNG.permutation(4))
        for side in sides[: int(RNG.integers(0, 4))]:
            if side == 0 and core_r > 0:
                ext_h = int(RNG.integers(1, min(2, core_r) + 1))
                ext_w = int(RNG.integers(1, core_w + 1))
                start = int(RNG.integers(core_c, core_c + core_w - ext_w + 1))
                mask[core_r - ext_h : core_r, start : start + ext_w] = True
            elif side == 1 and core_r + core_h < h:
                ext_h = int(RNG.integers(1, min(2, h - core_r - core_h) + 1))
                ext_w = int(RNG.integers(1, core_w + 1))
                start = int(RNG.integers(core_c, core_c + core_w - ext_w + 1))
                mask[core_r + core_h : core_r + core_h + ext_h, start : start + ext_w] = True
            elif side == 2 and core_c > 0:
                ext_w = int(RNG.integers(1, min(2, core_c) + 1))
                ext_h = int(RNG.integers(1, core_h + 1))
                start = int(RNG.integers(core_r, core_r + core_h - ext_h + 1))
                mask[start : start + ext_h, core_c - ext_w : core_c] = True
            elif side == 3 and core_c + core_w < w:
                ext_w = int(RNG.integers(1, min(2, w - core_c - core_w) + 1))
                ext_h = int(RNG.integers(1, core_h + 1))
                start = int(RNG.integers(core_r, core_r + core_h - ext_h + 1))
                mask[start : start + ext_h, core_c + core_w : core_c + core_w + ext_w] = True

        mask = _crop_mask(mask)
        if not _is_connected(mask):
            continue
        if int(mask.sum()) < 8:
            continue
        if int(mask.shape[0]) < 3 or int(mask.shape[1]) < 3:
            continue
        return mask
    return None


def _interior_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cand = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if all(mask[r + dr, c + dc] for dr, dc in N4):
                cand.append((r, c))
    return cand


def _hole_variant(mask: np.ndarray, holes: int, avoid: frozenset[tuple[int, int]] | None = None) -> tuple[np.ndarray, frozenset[tuple[int, int]]] | None:
    if holes == 0:
        return mask.copy(), frozenset()

    candidates = _interior_candidates(mask)
    if len(candidates) < holes:
        return None

    avoid = avoid or frozenset()
    candidate_list = [pt for pt in candidates if pt not in avoid]
    attempts = 60

    for _ in range(attempts):
        pool = list(candidate_list if len(candidate_list) >= holes else candidates)
        RNG.shuffle(pool)
        chosen: list[tuple[int, int]] = []
        for pt in pool:
            if all(abs(pt[0] - q[0]) + abs(pt[1] - q[1]) > 1 for q in chosen):
                chosen.append(pt)
            if len(chosen) == holes:
                break
        if len(chosen) != holes:
            continue
        variant = mask.copy()
        for r, c in chosen:
            variant[r, c] = False
        if _is_connected(variant) and np.array_equal(_fill_holes(variant), mask):
            return variant, frozenset(chosen)
    return None


def _build_pair_masks(silhouette: np.ndarray, branch: str) -> tuple[np.ndarray, np.ndarray] | None:
    max_holes = min(3, len(_interior_candidates(silhouette)))

    if branch == "unequal":
        if max_holes == 0:
            return None
        holes_a = int(RNG.integers(0, max_holes + 1))
        holes_b_choices = [k for k in range(max_holes + 1) if k != holes_a]
        if not holes_b_choices:
            return None
        holes_b = int(RNG.choice(holes_b_choices))
    else:
        if max_holes == 0 or RNG.random() < 0.25:
            holes_a = holes_b = 0
        else:
            holes_a = holes_b = int(RNG.integers(1, max_holes + 1))

    first = _hole_variant(silhouette, holes_a)
    if first is None:
        return None
    mask_a, removed_a = first

    second = _hole_variant(silhouette, holes_b, avoid=removed_a if holes_a == holes_b and holes_a > 0 else None)
    if second is None:
        second = _hole_variant(silhouette, holes_b)
    if second is None:
        return None
    mask_b, _removed_b = second

    if branch == "unequal" and int(mask_a.sum()) == int(mask_b.sum()):
        return None

    if RNG.random() < 0.5:
        mask_a, mask_b = mask_b, mask_a

    return mask_a, mask_b


def _rect_free(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    bh, bw = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(bh, r0 + h + pad)
    cc1 = min(bw, c0 + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    bh, bw = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(bh, r0 + h + pad)
    cc1 = min(bw, c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _pair_spec(used_keys: set[tuple[tuple[int, int], bytes]]) -> dict | None:
    for _ in range(120):
        branch = str(RNG.choice(["equal", "unequal"]))
        silhouette = _random_silhouette()
        if silhouette is None:
            continue
        key = (silhouette.shape, silhouette.tobytes())
        if key in used_keys:
            continue
        masks = _build_pair_masks(silhouette, branch)
        if masks is None:
            continue
        left_mask, right_mask = masks
        color_a, color_b = map(int, RNG.choice(COLORS, size=2, replace=False))
        return {
            "shape_key": key,
            "mask_a": left_mask,
            "mask_b": right_mask,
            "color_a": color_a,
            "color_b": color_b,
            "orientation": str(RNG.choice(["horizontal", "vertical"])),
            "gap": int(RNG.integers(1, 6)),
        }
    return None


def _place_pair(grid: np.ndarray, output_occ: np.ndarray, source_occ: np.ndarray, spec: dict) -> bool:
    h, w = spec["mask_a"].shape
    gh, gw = grid.shape

    if spec["orientation"] == "horizontal":
        delta = max(1, (w + spec["gap"] + 1) // 2)
        if gw < w + 2 * delta or gh < h:
            return False
        for _ in range(PAIR_RETRIES):
            dest_r = int(RNG.integers(0, gh - h + 1))
            dest_c = int(RNG.integers(delta, gw - w - delta + 1))
            left_c = dest_c - delta
            right_c = dest_c + delta
            if not _rect_free(output_occ, dest_r, dest_c, h, w, OUTPUT_PAD):
                continue
            if not _rect_free(source_occ, dest_r, left_c, h, w, SOURCE_PAD):
                continue
            if not _rect_free(source_occ, dest_r, right_c, h, w, SOURCE_PAD):
                continue

            _paste(grid, spec["mask_a"], dest_r, left_c, spec["color_a"])
            _paste(grid, spec["mask_b"], dest_r, right_c, spec["color_b"])
            _mark_rect(output_occ, dest_r, dest_c, h, w, OUTPUT_PAD)
            _mark_rect(source_occ, dest_r, left_c, h, w, SOURCE_PAD)
            _mark_rect(source_occ, dest_r, right_c, h, w, SOURCE_PAD)
            return True
    else:
        delta = max(1, (h + spec["gap"] + 1) // 2)
        if gh < h + 2 * delta or gw < w:
            return False
        for _ in range(PAIR_RETRIES):
            dest_r = int(RNG.integers(delta, gh - h - delta + 1))
            dest_c = int(RNG.integers(0, gw - w + 1))
            top_r = dest_r - delta
            bot_r = dest_r + delta
            if not _rect_free(output_occ, dest_r, dest_c, h, w, OUTPUT_PAD):
                continue
            if not _rect_free(source_occ, top_r, dest_c, h, w, SOURCE_PAD):
                continue
            if not _rect_free(source_occ, bot_r, dest_c, h, w, SOURCE_PAD):
                continue

            _paste(grid, spec["mask_a"], top_r, dest_c, spec["color_a"])
            _paste(grid, spec["mask_b"], bot_r, dest_c, spec["color_b"])
            _mark_rect(output_occ, dest_r, dest_c, h, w, OUTPUT_PAD)
            _mark_rect(source_occ, top_r, dest_c, h, w, SOURCE_PAD)
            _mark_rect(source_occ, bot_r, dest_c, h, w, SOURCE_PAD)
            return True

    return False


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.choice(NUM_PAIRS_CHOICES))
    min_h = GRID_H_RANGE[0] + max(0, pair_count - 2) * 2
    min_w = GRID_W_RANGE[0] + max(0, pair_count - 2) * 2
    gh = int(RNG.integers(min_h, GRID_H_RANGE[1] + 1))
    gw = int(RNG.integers(min_w, GRID_W_RANGE[1] + 1))

    grid = np.full((gh, gw), BG, dtype=int)
    output_occ = np.zeros((gh, gw), dtype=bool)
    source_occ = np.zeros((gh, gw), dtype=bool)
    used_keys: set[tuple[tuple[int, int], bytes]] = set()

    for _ in range(pair_count):
        spec = _pair_spec(used_keys)
        if spec is None:
            return None
        used_keys.add(spec["shape_key"])
        if not _place_pair(grid, output_occ, source_occ, spec):
            return None

    if len(_find_objects(grid)) != 2 * pair_count:
        return None
    if not np.array_equal(main(grid), main(np.array(grid, copy=True))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
