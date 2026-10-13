from __future__ import annotations

import random

import numpy as np


GREY = 5
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=bool)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _hole_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    holes: list[list[tuple[int, int]]] = []
    for comp in _find_components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes.append(comp)
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_find_components(mask)) == 1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _key_objects(grid: np.ndarray, bg: int) -> list[dict]:
    keys = []
    key_mask = (grid != bg) & (grid != GREY)
    for coords in _find_components(key_mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each key must be monochrome"
        color = colors.pop()
        keys.append(
            {
                "color": color,
                "coords": coords,
                "mask": _coords_to_mask(coords),
            }
        )
    return keys


def _lockholes(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    holes = []
    black_mask = grid == bg
    for coords in _find_components(black_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        neighbor_colors = set()
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != bg:
                    neighbor_colors.add(int(grid[nr, nc]))
        if neighbor_colors == {GREY}:
            holes.append({"coords": coords, "mask": _coords_to_mask(coords)})
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the uniquely equal black lockhole shape inside a grey block.
    Erase the original key and fill the matched lockhole with the key color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    keys = _key_objects(g, bg)
    locks = _lockholes(g, bg)
    assert keys, "expected at least one key"
    assert len(keys) == len(locks), "keys and lockholes must pair one-to-one"

    by_shape: dict[tuple[tuple[int, int], bytes], dict] = {}
    for lock in locks:
        sig = _shape_signature(lock["mask"])
        assert sig not in by_shape, "lockhole shapes must be unique"
        by_shape[sig] = lock

    out = g.copy()
    for key in keys:
        sig = _shape_signature(key["mask"])
        assert sig in by_shape, "every key must match one lockhole"
        for r, c in key["coords"]:
            out[r, c] = bg
        for r, c in by_shape[sig]["coords"]:
            out[r, c] = int(key["color"])

    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    mask = np.zeros((h, w), dtype=bool)
    mask[start] = True
    frontier = set()
    for dr, dc in DIR4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < area:
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        if mask[nr, nc]:
            continue
        mask[nr, nc] = True
        for dr, dc in DIR4:
            rr, cc = nr + dr, nc + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                frontier.add((rr, cc))
    return _crop_mask(mask)


def _make_solid_mask() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        max_area = h * w
        min_area = max(5, (h * w + 1) // 2)
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(h, w, area)
        if mask is None:
            continue
        mh, mw = map(int, mask.shape)
        if not (3 <= mh <= 5 and 3 <= mw <= 5):
            continue
        if not _is_connected(mask):
            continue
        if _hole_components(mask):
            continue
        if int(mask.sum()) == mh * mw and random.random() < 0.7:
            continue
        return _random_transform(mask)
    return None


def _hole_seeds(h: int, w: int) -> list[list[tuple[int, int]]]:
    seeds: list[list[tuple[int, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            seeds.append([(r, c)])
            if c + 1 < w - 1:
                seeds.append([(r, c), (r, c + 1)])
            if r + 1 < h - 1:
                seeds.append([(r, c), (r + 1, c)])
    return seeds


def _valid_holey_mask(mask: np.ndarray) -> bool:
    mh, mw = map(int, mask.shape)
    if not (3 <= mh <= 5 and 3 <= mw <= 5):
        return False
    if not _is_connected(mask):
        return False
    holes = _hole_components(mask)
    if len(holes) != 1:
        return False
    hole_area = len(holes[0])
    if not (1 <= hole_area <= 3):
        return False
    return int(mask.sum()) >= max(6, (mh * mw) // 2)


def _make_holey_mask() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        seeds = _hole_seeds(h, w)
        if not seeds:
            continue
        mask = np.ones((h, w), dtype=bool)
        for r, c in random.choice(seeds):
            mask[r, c] = False
        for _ in range(random.randint(1, 6)):
            cells = np.argwhere(mask)
            if len(cells) <= 6:
                break
            rr, cc = map(int, random.choice(cells.tolist()))
            trial = mask.copy()
            trial[rr, cc] = False
            trial = _crop_mask(trial)
            if _valid_holey_mask(trial):
                mask = trial
        mask = _crop_mask(mask)
        if not _valid_holey_mask(mask):
            continue
        return _random_transform(mask)
    return None


def _mask_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _lock_patch(mask: np.ndarray, bg: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    extra_h = random.randint(2, min(4, 8 - h))
    extra_w = random.randint(2, min(4, 8 - w))
    top = random.randint(1, extra_h - 1)
    bottom = extra_h - top
    left = random.randint(1, extra_w - 1)
    right = extra_w - left
    patch = np.full((h + top + bottom, w + left + right), GREY, dtype=int)
    window = patch[top : top + h, left : left + w]
    window[mask] = bg
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _place_patches(canvas: np.ndarray, patches: list[dict]) -> bool:
    h, w = canvas.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    for item in sorted(patches, key=lambda d: d["patch"].shape[0] * d["patch"].shape[1], reverse=True):
        ph, pw = map(int, item["patch"].shape)
        for _ in range(250):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_overlap(box, other, pad=1) for other in placed_boxes):
                continue
            window = canvas[r0 : r0 + ph, c0 : c0 + pw]
            patch = item["patch"]
            mask = patch != 0
            if np.any(window[mask] != 0):
                continue
            window[mask] = patch[mask]
            placed_boxes.append(box)
            item["pos"] = (r0, c0)
            break
        else:
            return False
    return True


def _random_shape(want_hole: bool) -> np.ndarray | None:
    return _make_holey_mask() if want_hole else _make_solid_mask()


def generate_input() -> np.ndarray:
    for _ in range(400):
        pair_count = random.randint(2, 4)
        holey_count = 0 if random.random() < 0.2 else random.randint(1, pair_count)
        holey_flags = [True] * holey_count + [False] * (pair_count - holey_count)
        random.shuffle(holey_flags)
        colors = random.sample(KEY_COLORS, pair_count)

        masks: list[np.ndarray] = []
        used_shapes: set[tuple[tuple[int, int], bytes]] = set()
        ok = True
        for want_hole in holey_flags:
            mask = None
            for _ in range(80):
                candidate = _random_shape(want_hole)
                if candidate is None:
                    continue
                sig = _shape_signature(candidate)
                if sig in used_shapes:
                    continue
                mask = candidate
                used_shapes.add(sig)
                break
            if mask is None:
                ok = False
                break
            masks.append(mask)
        if not ok:
            continue

        patches: list[dict] = []
        total_area = 0
        for mask, color in zip(masks, colors, strict=False):
            key_patch = _mask_patch(mask, color, 0)
            lock_patch = _lock_patch(mask, 0)
            total_area += int(np.prod(key_patch.shape) + np.prod(lock_patch.shape))
            patches.append({"kind": "key", "patch": key_patch, "shape": _shape_signature(mask)})
            patches.append({"kind": "lock", "patch": lock_patch, "shape": _shape_signature(mask)})

        min_side = max(18, int(np.ceil(np.sqrt(total_area))) + pair_count + 2)
        max_side = max(min_side, min(30, min_side + 6))
        if min_side > 30:
            continue
        grid_h = random.randint(min_side, max_side)
        grid_w = random.randint(min_side, max_side)
        grid = np.zeros((grid_h, grid_w), dtype=int)
        random.shuffle(patches)
        if not _place_patches(grid, patches):
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue

        key_count = len(_key_objects(grid, 0))
        lock_count = len(_lockholes(grid, 0))
        if key_count != pair_count or lock_count != pair_count:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
