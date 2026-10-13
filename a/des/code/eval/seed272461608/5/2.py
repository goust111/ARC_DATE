from __future__ import annotations

from collections import deque
import numpy as np


BG = 8
COLOR_POOL = tuple(color for color in range(10) if color != BG)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_RNG = np.random.default_rng()


def _component_mask(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> np.ndarray:
    """Extract one 4-connected boolean component from `mask`."""
    h, w = mask.shape
    comp = np.zeros_like(mask, dtype=bool)
    q = deque([start])
    seen[start] = True
    comp[start] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                comp[nr, nc] = True
                q.append((nr, nc))
    return comp


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    comp = _component_mask(mask, tuple(map(int, pts[0])), seen)
    return int(comp.sum()) == int(mask.sum())


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_mask(mask)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _dihedral_variants(arr: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flipped in (False, True):
        base = np.fliplr(arr) if flipped else arr
        for k in range(4):
            cur = np.rot90(base, k)
            key = (tuple(map(int, cur.shape)), cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(cur.copy())
    return out


def _canonical_bool_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    best: tuple[int, int, bytes] | None = None
    for variant in _dihedral_variants(np.asarray(mask, dtype=bool)):
        key = (int(variant.shape[0]), int(variant.shape[1]), variant.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _canonical_label_arr(labels: np.ndarray) -> tuple[np.ndarray, tuple[int, int, bytes]]:
    best_arr: np.ndarray | None = None
    best_key: tuple[int, int, bytes] | None = None
    for variant in _dihedral_variants(np.asarray(labels, dtype=np.int8)):
        key = (int(variant.shape[0]), int(variant.shape[1]), variant.tobytes())
        if best_key is None or key < best_key:
            best_key = key
            best_arr = variant
    assert best_arr is not None and best_key is not None
    return best_arr, best_key


def _extract_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            mask = g == color
            comp_mask = _component_mask(mask, (r, c), seen)
            r0, c0, r1, c1 = _bbox_of_mask(comp_mask)
            tight = np.asarray(comp_mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)
            comps.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": tight,
                    "area": int(tight.sum()),
                }
            )
    return comps


def _find_complement_labels(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, tuple[int, int, bytes]] | None:
    a = _tight_mask(np.asarray(mask_a, dtype=bool))
    b0 = _tight_mask(np.asarray(mask_b, dtype=bool))
    ha, wa = a.shape

    def try_layout(h: int, w: int, ar: int, ac: int, b: np.ndarray, br: int, bc: int) -> tuple[np.ndarray, tuple[int, int, bytes]] | None:
        hb, wb = b.shape
        if ar < 0 or ac < 0 or br < 0 or bc < 0:
            return None
        if ar + ha > h or ac + wa > w or br + hb > h or bc + wb > w:
            return None
        full_a = np.zeros((h, w), dtype=bool)
        full_b = np.zeros((h, w), dtype=bool)
        full_a[ar : ar + ha, ac : ac + wa] = a
        full_b[br : br + hb, bc : bc + wb] = b
        if np.any(full_a & full_b) or not np.all(full_a | full_b):
            return None
        labels = np.zeros((h, w), dtype=np.int8)
        labels[full_a] = 1
        labels[full_b] = 2
        return _canonical_label_arr(labels)

    for b in _dihedral_variants(b0):
        hb, wb = b.shape

        # Same tight frame: the stricter version from the prompt.
        if (ha, wa) == (hb, wb) and np.array_equal(a, ~b):
            labels = np.where(a, 1, 2).astype(np.int8)
            return _canonical_label_arr(labels)

        # Left/right or right/left cuts. These cover the connected fragments
        # produced by the generator and remain cheap enough for pair search.
        h = max(ha, hb)
        for w in range(max(wa, wb), wa + wb + 1):
            for ar in range(h - ha + 1):
                for br in range(h - hb + 1):
                    for ac, bc in ((0, w - wb), (w - wa, 0)):
                        found = try_layout(h, w, ar, ac, b, br, bc)
                        if found is not None:
                            return found

        # Top/bottom or bottom/top cuts, after possible rotations.
        w = max(wa, wb)
        for h in range(max(ha, hb), ha + hb + 1):
            for ac in range(w - wa + 1):
                for bc in range(w - wb + 1):
                    for ar, br in ((0, h - hb), (h - ha, 0)):
                        found = try_layout(h, w, ar, ac, b, br, bc)
                        if found is not None:
                            return found
    return None


def _are_complements(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return _find_complement_labels(mask_a, mask_b) is not None


def _pair_components(comps: list[dict]) -> list[tuple[int, int]]:
    n = len(comps)
    assert n > 0 and n % 2 == 0
    mates = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _are_complements(comps[i]["mask"], comps[j]["mask"]):
                mates[i].append(j)
                mates[j].append(i)
    for options in mates:
        assert options, "every fragment must have a complementary partner"

    full = (1 << n) - 1
    memo: dict[int, list[tuple[int, int]] | None] = {}

    def search(used: int) -> list[tuple[int, int]] | None:
        if used == full:
            return []
        if used in memo:
            return memo[used]

        i = 0
        while (used >> i) & 1:
            i += 1

        result = None
        for j in sorted(mates[i]):
            if (used >> j) & 1:
                continue
            tail = search(used | (1 << i) | (1 << j))
            if tail is None:
                continue
            result = [(i, j)] + tail
            break
        memo[used] = result
        return result

    pairs = search(0)
    assert pairs is not None, "complementary fragments must admit a perfect matching"
    return pairs


def _component_role_key(mask: np.ndarray) -> tuple[tuple[int, int, bytes], int, int, bytes]:
    tight = np.asarray(mask, dtype=bool)
    return _canonical_bool_key(tight), int(tight.shape[0]), int(tight.shape[1]), tight.tobytes()


def _assemble_pair(comp_a: dict, comp_b: dict) -> tuple[np.ndarray, tuple[int, int, bytes]]:
    mask_a = np.asarray(comp_a["mask"], dtype=bool)
    mask_b = np.asarray(comp_b["mask"], dtype=bool)
    found = _find_complement_labels(mask_a, mask_b)
    assert found is not None
    canon, label_key = found
    out = np.empty(canon.shape, dtype=int)
    out[canon == 1] = int(comp_a["color"])
    out[canon == 2] = int(comp_b["color"])
    return out, label_key


def _compose_output(rectangles: list[tuple[np.ndarray, int, tuple[int, int, bytes]]]) -> np.ndarray:
    assert rectangles
    ordered = sorted(
        rectangles,
        key=lambda item: (int(item[1]), int(item[0].shape[0]), int(item[0].shape[1]), item[2]),
    )
    height = max(int(rect.shape[0]) for rect, _, _ in ordered)
    width = sum(int(rect.shape[1]) for rect, _, _ in ordered) + len(ordered) - 1
    out = np.full((height, width), BG, dtype=int)
    c = 0
    for rect, _, _ in ordered:
        h, w = rect.shape
        out[:h, c : c + w] = rect
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments, assemble each pair into a solid rectangle,
    then sort the rectangles by area and place them left-to-right.
    """
    g = np.asarray(grid, dtype=int)
    comps = _extract_components(g, BG)
    pairs = _pair_components(comps)

    rectangles = []
    for i, j in pairs:
        rect, label_key = _assemble_pair(comps[i], comps[j])
        rectangles.append((rect, int(rect.shape[0] * rect.shape[1]), label_key))
    return _compose_output(rectangles)


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if int(_RNG.integers(0, 2)):
        out = np.fliplr(out)
    return np.rot90(out, int(_RNG.integers(0, 4)))


def _split_mask(h: int, w: int) -> np.ndarray:
    cuts = _RNG.integers(1, w, size=h)
    if len(set(map(int, cuts))) == 1:
        cuts[int(_RNG.integers(0, h))] = int(_RNG.integers(1, w))

    mask = np.zeros((h, w), dtype=bool)
    for r, cut in enumerate(cuts):
        mask[r, : int(cut)] = True
    return mask


def _generate_shape_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    base = _split_mask(h, w)
    other = ~base
    if not _is_connected(base) or not _is_connected(other):
        return None
    return _tight_mask(_random_dihedral(base)), _tight_mask(_random_dihedral(other))


def _paint_component(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _reserve_ok(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    return not bool(reserved[top : top + h, left : left + w].any())


def _reserve_bbox(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    h0, w0 = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h0, top + h + 1)
    c1 = min(w0, left + w + 1)
    reserved[r0:r1, c0:c1] = True


def _scatter_components(components: list[dict]) -> np.ndarray | None:
    max_h = max(int(comp["patch"].shape[0]) for comp in components)
    max_w = max(int(comp["patch"].shape[1]) for comp in components)
    bbox_area = sum(int(comp["patch"].shape[0] * comp["patch"].shape[1]) for comp in components)

    for _ in range(120):
        extra_h = int(_RNG.integers(8, 15))
        extra_w = int(_RNG.integers(8, 18))
        h = max_h + extra_h + int(_RNG.integers(0, 7))
        w = max(max_w + extra_w, int(np.ceil(np.sqrt(bbox_area * 3.2)))) + int(_RNG.integers(6, 15))

        grid = np.full((h, w), BG, dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        order = sorted(range(len(components)), key=lambda idx: components[idx]["patch"].size, reverse=True)
        placed = True

        for idx in order:
            patch = components[idx]["patch"]
            ph, pw = patch.shape
            spots = []
            for _ in range(90):
                top = int(_RNG.integers(0, h - ph + 1))
                left = int(_RNG.integers(0, w - pw + 1))
                if _reserve_ok(reserved, top, left, ph, pw):
                    spots.append((top, left))
            if not spots:
                placed = False
                break
            top, left = spots[int(_RNG.integers(0, len(spots)))]
            mask = patch != BG
            grid[top : top + ph, left : left + pw][mask] = patch[mask]
            _reserve_bbox(reserved, top, left, ph, pw)

        if placed:
            return grid
    return None


def _build_expected(pair_components: list[tuple[dict, dict]]) -> np.ndarray:
    rectangles = []
    for comp_a, comp_b in pair_components:
        rect, label_key = _assemble_pair(comp_a, comp_b)
        rectangles.append((rect, int(rect.shape[0] * rect.shape[1]), label_key))
    return _compose_output(rectangles)


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.integers(2, 5))

    dims_pool = [
        (5, 6),
        (5, 7),
        (5, 8),
        (6, 6),
        (6, 7),
        (6, 8),
        (7, 7),
        (7, 8),
    ]
    _RNG.shuffle(dims_pool)

    pair_dims = []
    used_areas: set[int] = set()
    for h, w in dims_pool:
        area = h * w
        if area in used_areas:
            continue
        pair_dims.append((h, w))
        used_areas.add(area)
        if len(pair_dims) == pair_count:
            break
    if len(pair_dims) != pair_count:
        return None

    colors = _RNG.choice(np.array(COLOR_POOL, dtype=int), size=pair_count * 2, replace=False)
    components: list[dict] = []
    pair_components: list[tuple[dict, dict]] = []

    for idx, (h, w) in enumerate(pair_dims):
        pair = _generate_shape_pair(h, w)
        if pair is None:
            return None
        mask_a, mask_b = pair
        color_a = int(colors[2 * idx])
        color_b = int(colors[2 * idx + 1])
        comp_a = {"color": color_a, "mask": mask_a}
        comp_b = {"color": color_b, "mask": mask_b}
        comp_a["patch"] = _paint_component(mask_a, color_a)
        comp_b["patch"] = _paint_component(mask_b, color_b)
        components.extend([comp_a, comp_b])
        pair_components.append((comp_a, comp_b))

    inp = _scatter_components(components)
    if inp is None:
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if not isinstance(out, np.ndarray) or out.size == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
