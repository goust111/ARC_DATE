import random
from collections import deque

import numpy as np


BG = 0
SCALES = (2, 3)
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500
PAIR_COUNT_CHOICES = (1, 2, 3)
GRID_MIN_SIDE = 18
GRID_MAX_SIDE = 32


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == len(pts)


def _has_internal_background(mask: np.ndarray) -> bool:
    bg = ~mask
    seen = np.zeros(mask.shape, dtype=bool)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if seen[r, c] or not bg[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = r in (0, mask.shape[0] - 1) or c in (0, mask.shape[1] - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, mask.shape[0] - 1) or nc in (0, mask.shape[1] - 1):
                            touches_border = True
                        q.append((nr, nc))
            if not touches_border:
                return True
    return False


def _internal_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, c in zip(*np.where(mask), strict=False):
        rr = int(r)
        cc = int(c)
        good = True
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]):
                good = False
                break
        if good:
            cells.append((rr, cc))
    return cells


def _crop(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _transform(mask: np.ndarray, code: int) -> np.ndarray:
    if code < 4:
        return np.rot90(mask, code)
    return np.fliplr(np.rot90(mask, code - 4))


def _unique_orientations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for code in range(8):
        cur = _transform(mask, code)
        key = (cur.shape, cur.astype(np.uint8).tobytes())
        if key not in seen:
            seen.add(key)
            out.append(cur)
    return out


def _canonical_signature(mask: np.ndarray) -> tuple:
    best = None
    for cur in _unique_orientations(mask):
        pts = tuple(map(tuple, np.argwhere(cur)))
        cand = (cur.shape, pts)
        if best is None or cand < best:
            best = cand
    assert best is not None
    return best


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        coords.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                    "area": len(coords),
                }
            )
    return comps


def _missing_is_internal(big_mask: np.ndarray, missing: np.ndarray) -> bool:
    if not missing.any():
        return False
    bg = ~big_mask
    seen = np.zeros(big_mask.shape, dtype=bool)
    for r in range(big_mask.shape[0]):
        for c in range(big_mask.shape[1]):
            if seen[r, c] or not bg[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            touches_border = r in (0, big_mask.shape[0] - 1) or c in (0, big_mask.shape[1] - 1)
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < big_mask.shape[0] and 0 <= nc < big_mask.shape[1] and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, big_mask.shape[0] - 1) or nc in (0, big_mask.shape[1] - 1):
                            touches_border = True
                        q.append((nr, nc))
            if any(missing[rr, cc] for rr, cc in comp) and touches_border:
                return False
    return True


def _match_small_big(small_mask: np.ndarray, big_mask: np.ndarray) -> dict | None:
    small_area = int(small_mask.sum())
    big_area = int(big_mask.sum())
    if big_area <= small_area:
        return None

    for oriented in _unique_orientations(small_mask):
        for scale in SCALES:
            if oriented.shape[0] * scale != big_mask.shape[0] or oriented.shape[1] * scale != big_mask.shape[1]:
                continue
            full = _scale_mask(oriented, scale)
            if np.any(big_mask & ~full):
                continue
            missing = full & ~big_mask
            if not missing.any():
                continue

            good = True
            hole_logic = np.zeros_like(oriented, dtype=bool)
            for r, c in zip(*np.where(oriented), strict=False):
                rr = int(r)
                cc = int(c)
                block = big_mask[rr * scale : (rr + 1) * scale, cc * scale : (cc + 1) * scale]
                filled = int(block.sum())
                if filled == 0:
                    hole_logic[rr, cc] = True
                elif filled != scale * scale:
                    good = False
                    break
            if not good or not hole_logic.any():
                continue
            if not _missing_is_internal(big_mask, missing):
                continue
            return {
                "scale": scale,
                "full": full,
                "missing": missing,
                "hole_logic": hole_logic,
            }
    return None


def _pair_components(comps: list[dict]) -> list[tuple[int, int, dict]]:
    n = len(comps)
    assert n > 0 and n % 2 == 0

    edges: dict[int, list[int]] = {i: [] for i in range(n)}
    info: dict[tuple[int, int], dict] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            match = _match_small_big(comps[i]["mask"], comps[j]["mask"])
            if match is not None:
                edges[i].append(j)
                info[(i, j)] = match

    used = [False] * n
    answer: list[tuple[int, int, dict]] = []

    def dfs() -> bool:
        if all(used):
            return True
        choices = [i for i in range(n) if not used[i] and any(not used[j] for j in edges[i])]
        if not choices:
            return False
        i = min(choices, key=lambda x: sum(not used[j] for j in edges[x]))
        for j in edges[i]:
            if used[j]:
                continue
            used[i] = True
            used[j] = True
            answer.append((i, j, info[(i, j)]))
            if dfs():
                return True
            answer.pop()
            used[i] = False
            used[j] = False
        return False

    assert dfs(), "expected a unique valid pairing of components"
    return answer


def main(grid: np.ndarray) -> np.ndarray:
    """Pair scaled shape-matched objects, erase each small one, and fill only the large object's holes with the small object's color."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _find_components(g)
    pairs = _pair_components(comps)

    out = g.copy()
    for small_idx, big_idx, match in pairs:
        small = comps[small_idx]
        big = comps[big_idx]

        for r, c in small["coords"]:
            out[r, c] = BG

        r0, c0, r1, c1 = big["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[match["missing"]] = int(small["color"])

    return out


def _random_shape(max_side: int) -> np.ndarray | None:
    h = random.randint(3, max_side)
    w = random.randint(3, max_side)
    mask = np.ones((h, w), dtype=bool)
    target = random.randint(max(5, (h * w + 1) // 2), h * w)

    attempts = 0
    while int(mask.sum()) > target and attempts < 200:
        attempts += 1
        boundary = []
        for r, c in zip(*np.where(mask), strict=False):
            rr = int(r)
            cc = int(c)
            if any(not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]) for nr, nc in _neighbors4(rr, cc)):
                boundary.append((rr, cc))
        random.shuffle(boundary)
        removed = False
        for rr, cc in boundary:
            cand = mask.copy()
            cand[rr, cc] = False
            if not _is_connected(cand):
                continue
            cand = _crop(cand)
            if _has_internal_background(cand):
                continue
            if not _internal_cells(cand):
                continue
            mask = cand
            h, w = mask.shape
            removed = True
            break
        if not removed:
            break

    mask = _crop(mask)
    if int(mask.sum()) < 5 or not _is_connected(mask) or _has_internal_background(mask):
        return None
    if not _internal_cells(mask):
        return None
    return mask


def _choose_holes(mask: np.ndarray, scale: int) -> np.ndarray | None:
    inside = _internal_cells(mask)
    if not inside:
        return None
    max_remove = min(3, len(inside))
    for _ in range(200):
        hole_count = random.randint(1, max_remove)
        chosen = random.sample(inside, hole_count)
        holes = np.zeros_like(mask, dtype=bool)
        for r, c in chosen:
            holes[r, c] = True
        visible = mask & ~holes
        if not _is_connected(visible):
            continue
        if scale * scale * int(visible.sum()) <= int(mask.sum()):
            continue
        return holes
    return None


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_objects(rows: int, cols: int, objects: list[dict]) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    rects: list[tuple[int, int, int, int] | None] = [None] * len(objects)
    order = sorted(
        range(len(objects)),
        key=lambda idx: objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1],
        reverse=True,
    )

    for idx in order:
        h, w = objects[idx]["mask"].shape
        candidates = [(r, c) for r in range(1, rows - h) for c in range(1, cols - w)]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            rect = (r, c, r + h - 1, c + w - 1)
            ok = True
            for other in rects:
                if other is not None and not _rects_separated(rect, other, gap=1):
                    ok = False
                    break
            if ok:
                placements[idx] = (r, c)
                rects[idx] = rect
                placed = True
                break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _build_expected(grid: np.ndarray, objects: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    by_pair: dict[int, dict[str, int]] = {}
    for idx, obj in enumerate(objects):
        entry = by_pair.setdefault(obj["pair_id"], {})
        entry[obj["kind"]] = idx

    for pair in by_pair.values():
        small = objects[pair["small"]]
        big = objects[pair["big"]]
        sr, sc = placements[pair["small"]]
        br, bc = placements[pair["big"]]

        small_patch = out[sr : sr + small["mask"].shape[0], sc : sc + small["mask"].shape[1]]
        small_patch[small["mask"]] = BG

        hole_mask = _scale_mask(big["hole_mask"], big["scale"])
        big_patch = out[br : br + big["mask"].shape[0], bc : bc + big["mask"].shape[1]]
        big_patch[hole_mask] = int(small["color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    object_count = pair_count * 2
    if object_count > len(COLORS):
        return None

    colors = random.sample(COLORS, object_count)
    used_shapes: set[tuple] = set()
    objects: list[dict] = []

    for pair_id in range(pair_count):
        scale = random.choice(SCALES)
        max_side = 4 if pair_count == 3 else 5

        base = None
        for _ in range(200):
            cand = _random_shape(max_side)
            if cand is None:
                continue
            sig = _canonical_signature(cand)
            if sig in used_shapes:
                continue
            holes = _choose_holes(cand, scale)
            if holes is None:
                continue
            base = cand
            used_shapes.add(sig)
            break
        if base is None:
            return None

        holes = _choose_holes(base, scale)
        if holes is None:
            return None

        small_code = random.randrange(8)
        big_code = random.randrange(8)
        small_mask = _transform(base, small_code)
        visible_logic = base & ~holes
        big_logic = _transform(visible_logic, big_code)
        hole_mask = _transform(holes, big_code)
        big_mask = _scale_mask(big_logic, scale)

        small_color = colors[2 * pair_id]
        big_color = colors[2 * pair_id + 1]

        objects.append(
            {
                "pair_id": pair_id,
                "kind": "small",
                "color": small_color,
                "mask": small_mask,
            }
        )
        objects.append(
            {
                "pair_id": pair_id,
                "kind": "big",
                "color": big_color,
                "mask": big_mask,
                "hole_mask": hole_mask,
                "scale": scale,
            }
        )

    max_h = max(obj["mask"].shape[0] for obj in objects)
    max_w = max(obj["mask"].shape[1] for obj in objects)
    padded_area = sum((obj["mask"].shape[0] + 2) * (obj["mask"].shape[1] + 2) for obj in objects)
    base_side = max(int(np.ceil(np.sqrt(padded_area))) + random.randint(1, 5), max(max_h, max_w) + 3, GRID_MIN_SIDE)
    rows = random.randint(base_side, min(GRID_MAX_SIDE, base_side + 5))
    cols = random.randint(base_side, min(GRID_MAX_SIDE, base_side + 5))

    placements = _place_objects(rows, cols, objects)
    if placements is None:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    for obj, (r, c) in zip(objects, placements, strict=False):
        _paint_mask(grid, obj["mask"], r, c, int(obj["color"]))

    expected = _build_expected(grid, objects, placements)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
