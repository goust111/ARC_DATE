import random
from collections import Counter

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
EDGE_BAND = 2
FENCE_MARGIN = 2
KEY_MARGIN = 1
FENCE_KEY_MARGIN = 4

RNG = random.Random()


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _rot_cw(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, -int(k) % 4)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    assert rows.size > 0
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp = np.zeros_like(mask, dtype=bool)
            for rr, cc in cells:
                comp[rr, cc] = True
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask)) > 0 and len(_component_masks(mask)) == 1


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_mask(mask)
    variants = []
    for k in range(4):
        rot = _crop_mask(_rot_cw(cropped, k))
        variants.append((tuple(int(x) for x in rot.shape), rot.tobytes()))
    return min(variants)


def _is_rect_frame(mask: np.ndarray) -> bool:
    mask = _crop_mask(mask)
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, 0].all() or not mask[:, -1].all():
        return False
    return bool(np.any(~mask[1:-1, 1:-1]))


def _make_patch_from_hole(hole: np.ndarray, color: int, bg: int = BG) -> np.ndarray:
    ih, iw = hole.shape
    patch = np.full((ih + 2, iw + 2), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1][~hole] = color
    return patch


def _looks_like_fence(patch: np.ndarray, color: int, bg: int) -> bool:
    h, w = patch.shape
    if h < 4 or w < 4:
        return False
    if not np.all(patch[0, :] == color) or not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color) or not np.all(patch[:, -1] == color):
        return False
    if not np.any(patch[1:-1, 1:-1] == bg):
        return False
    color_mask = patch == color
    seen = np.zeros((h, w), dtype=bool)
    stack = [(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)] + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)]
    while stack:
        r, c = stack.pop()
        if not color_mask[r, c] or seen[r, c]:
            continue
        seen[r, c] = True
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and color_mask[nr, nc] and not seen[nr, nc]:
                stack.append((nr, nc))
    return not np.any(seen[1:-1, 1:-1])


def _extract_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    fences = []
    keys = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        mask = grid == color
        coords = np.argwhere(mask)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), bg, dtype=int)
        patch[mask[r0 : r1 + 1, c0 : c1 + 1]] = color
        if _looks_like_fence(patch, color, bg):
            hole_full = (patch == bg)
            hole_full[0, :] = False
            hole_full[-1, :] = False
            hole_full[:, 0] = False
            hole_full[:, -1] = False
            assert np.any(hole_full)
            hole = _crop_mask(hole_full)
            assert _is_connected(hole_full)
            fences.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "color": color,
                    "patch": patch,
                    "hole": hole,
                    "area": int(patch.shape[0] * patch.shape[1]),
                }
            )
        else:
            comps = _component_masks(mask)
            assert len(comps) == 1
            keys.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "color": color,
                    "mask": _crop_mask(mask[r0 : r1 + 1, c0 : c1 + 1]),
                }
            )
    fences.sort(key=lambda obj: obj["bbox"])
    keys.sort(key=lambda obj: obj["bbox"])
    assert fences and len(fences) == len(keys)
    return fences, keys


def _match_keys(keys: list[dict], fences: list[dict]) -> dict[int, int]:
    options: list[list[tuple[int, int]]] = []
    for key in keys:
        cur = []
        for fi, fence in enumerate(fences):
            for rot in range(4):
                if np.array_equal(_crop_mask(_rot_cw(key["mask"], rot)), fence["hole"]):
                    cur.append((fi, rot))
        assert cur
        options.append(cur)

    order = sorted(range(len(keys)), key=lambda i: len(options[i]))
    assignments: list[tuple[int, int] | None] = [None] * len(keys)
    used_fences: set[int] = set()
    solutions: list[list[tuple[int, int]]] = []

    def dfs(idx: int) -> None:
        if len(solutions) > 1:
            return
        if idx == len(order):
            solutions.append([item for item in assignments if item is not None])
            return
        key_idx = order[idx]
        for fence_idx, rot in options[key_idx]:
            if fence_idx in used_fences:
                continue
            assignments[key_idx] = (fence_idx, rot)
            used_fences.add(fence_idx)
            dfs(idx + 1)
            used_fences.remove(fence_idx)
            assignments[key_idx] = None

    dfs(0)
    assert len(solutions) == 1

    rot_by_fence: dict[int, int] = {}
    for fence_idx, rot in solutions[0]:
        assert fence_idx not in rot_by_fence
        rot_by_fence[fence_idx] = rot
    assert len(rot_by_fence) == len(fences)
    return rot_by_fence


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each hollow fence according to the matching keyhole sprite, then center-nest the fences by size.

    Input structure:
    - The background is a dominant color. Several differently colored one-cell-thick rectangular fences appear in the grid.
    - Each fence has same-color interior pattern pixels that leave one connected background "keyhole" shape inside the frame.
    - Separate monochrome key sprites sit away from the fences, and each key matches exactly one fence keyhole up to rotation.

    Rule:
    - Extract every fence, compute its interior background keyhole shape, and extract every key sprite as a binary mask.
    - Match each key to the unique fence whose keyhole equals the key after a clockwise rotation of 0, 90, 180, or 270 degrees.
    - Rotate the entire corresponding fence by that same clockwise amount.
    - Sort the rotated fences by outer-frame area from largest to smallest and place them centered on one canvas, producing the nested output and removing the keys.

    Key insight:
    - The key sprite encodes both identity and instruction: the rotation needed to align the key to a fence keyhole is exactly the rotation applied to that fence before nesting.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fences, keys = _extract_objects(g, bg)
    rotations = _match_keys(keys, fences)

    rotated = []
    for fi, fence in enumerate(fences):
        patch = _rot_cw(fence["patch"], rotations[fi])
        rotated.append({"patch": patch, "area": fence["area"]})
    rotated.sort(key=lambda obj: (-obj["area"], -obj["patch"].shape[0], -obj["patch"].shape[1]))

    out_h, out_w = rotated[0]["patch"].shape
    out = np.full((out_h, out_w), bg, dtype=int)
    for obj in rotated:
        patch = obj["patch"]
        h, w = patch.shape
        r0 = (out_h - h) // 2
        c0 = (out_w - w) // 2
        window = out[r0 : r0 + h, c0 : c0 + w]
        mask = patch != bg
        assert np.all(window[mask] == bg)
        window[mask] = patch[mask]
    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _edge_positions(limit: int, size: int, edge: int) -> tuple[int, int]:
    if edge == 0:
        return 0, 0
    if edge == 1:
        return limit - size, limit - size
    return 0, limit - size


def _random_int(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(int(lo), int(hi))


def _make_hole(
    outer_h: int,
    outer_w: int,
    child_dims: tuple[int, int] | None,
    used_shapes: set[tuple[tuple[int, int], bytes]],
    rng: random.Random,
) -> np.ndarray | None:
    ih, iw = outer_h - 2, outer_w - 2
    protect = np.zeros((ih, iw), dtype=bool)
    if child_dims is not None:
        child_h, child_w = child_dims
        if child_h > ih or child_w > iw:
            return None
        pr0 = (ih - child_h) // 2
        pc0 = (iw - child_w) // 2
        protect[pr0 : pr0 + child_h, pc0 : pc0 + child_w] = True

    safe = np.ones((ih, iw), dtype=bool)
    safe[0, :] = False
    safe[-1, :] = False
    safe[:, 0] = False
    safe[:, -1] = False
    safe &= ~protect
    candidates = [tuple(int(x) for x in rc) for rc in np.argwhere(safe)]
    if not candidates:
        return None

    edge_like = [
        cell
        for cell in candidates
        if min(cell[0], cell[1], ih - 1 - cell[0], iw - 1 - cell[1]) == 1
    ]
    starts = edge_like or candidates

    for _ in range(120):
        target = _random_int(rng, 1, min(5, len(candidates)))
        blocker = {rng.choice(starts)}
        while len(blocker) < target:
            frontier = []
            for r, c in blocker:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ih and 0 <= nc < iw and safe[nr, nc] and (nr, nc) not in blocker:
                        frontier.append((nr, nc))
            if not frontier:
                break
            frontier = sorted(set(frontier), key=lambda cell: (cell[0], cell[1]))
            blocker.add(rng.choice(frontier))

        hole = np.ones((ih, iw), dtype=bool)
        for r, c in blocker:
            hole[r, c] = False
        if child_dims is not None and not np.all(hole[protect]):
            continue
        if not _is_connected(hole):
            continue
        cropped = _crop_mask(hole)
        if any(np.array_equal(cropped, _crop_mask(_rot_cw(cropped, k))) for k in (1, 2, 3)):
            continue
        canon = _canonical_mask(cropped)
        if canon in used_shapes:
            continue
        return hole
    return None


def _sample_output_dims(n_pairs: int, rng: random.Random) -> list[tuple[int, int]] | None:
    max_outer = 15 if n_pairs == 3 else 18
    for _ in range(120):
        small_h = _random_int(rng, 5, 7)
        small_w = _random_int(rng, 5, 7)
        if small_h == 5 and small_w == 5:
            small_w = 6
        dims = [(small_h, small_w)]
        ok = True
        for _ in range(1, n_pairs):
            prev_h, prev_w = dims[-1]
            dh, dw = rng.choice([(4, 6), (6, 4), (5, 6), (6, 5)])
            new_h = prev_h + dh
            new_w = prev_w + dw
            if new_h > max_outer or new_w > max_outer:
                ok = False
                break
            dims.append((new_h, new_w))
        if ok:
            dims.reverse()
            return dims
    return None


def _build_specs(rng: random.Random) -> list[dict] | None:
    n_pairs = rng.choice([2, 2, 3])
    dims = _sample_output_dims(n_pairs, rng)
    if dims is None:
        return None

    colors = list(rng.sample(COLORS, 2 * n_pairs))
    fence_colors = colors[:n_pairs]
    key_colors = colors[n_pairs:]
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    specs = []

    for idx, (outer_h, outer_w) in enumerate(dims):
        child_dims = dims[idx + 1] if idx + 1 < len(dims) else None
        hole_out = _make_hole(outer_h, outer_w, child_dims, used_shapes, rng)
        if hole_out is None:
            return None
        canon = _canonical_mask(hole_out)
        used_shapes.add(canon)

        rot = _random_int(rng, 0, 3)
        hole_in = np.rot90(hole_out, rot)
        key_mask = _crop_mask(np.rot90(hole_in, rot))

        out_patch = _make_patch_from_hole(hole_out, fence_colors[idx], BG)
        in_patch = _make_patch_from_hole(hole_in, fence_colors[idx], BG)
        key_patch = np.full(key_mask.shape, BG, dtype=int)
        key_patch[key_mask] = key_colors[idx]

        specs.append(
            {
                "area": int(outer_h * outer_w),
                "out_patch": out_patch,
                "in_patch": in_patch,
                "key_patch": key_patch,
                "rot": rot,
            }
        )

    return specs


def _compose_expected(specs: list[dict]) -> np.ndarray | None:
    specs = sorted(specs, key=lambda obj: -obj["area"])
    out_h, out_w = specs[0]["out_patch"].shape
    out = np.full((out_h, out_w), BG, dtype=int)
    for spec in specs:
        patch = spec["out_patch"]
        h, w = patch.shape
        r0 = (out_h - h) // 2
        c0 = (out_w - w) // 2
        window = out[r0 : r0 + h, c0 : c0 + w]
        mask = patch != BG
        if not np.all(window[mask] == BG):
            return None
        window[mask] = patch[mask]
    return out


def _choose_canvas(specs: list[dict], rng: random.Random) -> tuple[int, int] | None:
    objs = [spec["in_patch"] for spec in specs] + [spec["key_patch"] for spec in specs]
    total_bbox_area = sum(int(obj.shape[0] * obj.shape[1]) for obj in objs)
    max_h = max(int(obj.shape[0]) for obj in objs)
    max_w = max(int(obj.shape[1]) for obj in objs)
    side_min = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(total_bbox_area * 1.35))))
    side_min = max(side_min, 22)
    if side_min > 30:
        return None
    h = _random_int(rng, side_min, 30)
    w = _random_int(rng, side_min, 30)
    return h, w


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    window = grid[top : top + h, left : left + w]
    window[mask] = patch[mask]


def _place_objects(specs: list[dict], rng: random.Random) -> np.ndarray | None:
    canvas = _choose_canvas(specs, rng)
    if canvas is None:
        return None
    h, w = canvas
    grid = np.full((h, w), BG, dtype=int)

    placed_fences: list[tuple[int, int, int, int]] = []
    placed_keys: list[tuple[int, int, int, int]] = []

    fence_specs = sorted(specs, key=lambda obj: -obj["area"])
    for spec in fence_specs:
        patch = spec["in_patch"]
        ph, pw = patch.shape
        if ph + 4 > h or pw + 4 > w:
            return None
        placed = False
        for _ in range(300):
            top = _random_int(rng, 2, h - ph - 2)
            left = _random_int(rng, 2, w - pw - 2)
            box = (top, left, top + ph - 1, left + pw - 1)
            if any(_boxes_too_close(box, other, FENCE_MARGIN) for other in placed_fences):
                continue
            placed_fences.append(box)
            _paint_patch(grid, patch, top, left)
            placed = True
            break
        if not placed:
            return None

    key_specs = sorted(specs, key=lambda obj: -(obj["key_patch"].shape[0] * obj["key_patch"].shape[1]))
    for spec in key_specs:
        patch = spec["key_patch"]
        ph, pw = patch.shape
        if ph > h or pw > w:
            return None
        placed = False
        for _ in range(400):
            edge = _random_int(rng, 0, 3)
            if edge == 0:
                top = _random_int(rng, 0, min(EDGE_BAND, h - ph))
                left = _random_int(rng, 0, w - pw)
            elif edge == 1:
                top = _random_int(rng, h - ph - min(EDGE_BAND, h - ph), h - ph)
                left = _random_int(rng, 0, w - pw)
            elif edge == 2:
                top = _random_int(rng, 0, h - ph)
                left = _random_int(rng, 0, min(EDGE_BAND, w - pw))
            else:
                top = _random_int(rng, 0, h - ph)
                left = _random_int(rng, w - pw - min(EDGE_BAND, w - pw), w - pw)
            box = (top, left, top + ph - 1, left + pw - 1)
            if any(_boxes_too_close(box, other, KEY_MARGIN) for other in placed_keys):
                continue
            if any(_boxes_too_close(box, other, FENCE_KEY_MARGIN) for other in placed_fences):
                continue
            placed_keys.append(box)
            _paint_patch(grid, patch, top, left)
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _build_specs(rng)
    if specs is None:
        return None
    expected = _compose_expected(specs)
    if expected is None or max(expected.shape) > 30:
        return None

    grid = _place_objects(specs, rng)
    if grid is None or max(grid.shape) > 30:
        return None
    if np.count_nonzero(grid == BG) <= grid.size // 2:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
