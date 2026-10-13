from __future__ import annotations

import random

import numpy as np


BG = 8
FG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 300

GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
OBJECT_COUNT_RANGE = (3, 6)

DENSE_HEIGHT_RANGE = (3, 7)
DENSE_WIDTH_RANGE = (3, 7)
SPARSE_HEIGHT_RANGE = (4, 8)
SPARSE_WIDTH_RANGE = (4, 8)

DENSE_KEEP_FRAC_RANGE = (0.78, 1.00)
SPARSE_FILL_FRAC_RANGE = (0.30, 0.60)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = map(int, mask.shape)
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            nxt = (int(nr), int(nc))
            if bool(mask[nr, nc]) and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randint(0, 3)
    return np.rot90(out, k)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = np.zeros((h + 2, w + 2), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out[dr : dr + h, dc : dc + w] |= mask
    return out


def _shrink_mask(mask: np.ndarray) -> np.ndarray:
    """Compress the mask into a bbox that is one cell smaller per axis."""
    h, w = map(int, mask.shape)
    new_h = max(1, h - 1)
    new_w = max(1, w - 1)
    out = np.zeros((new_h, new_w), dtype=bool)
    rr, cc = np.argwhere(mask).T

    if h == 1 or new_h == 1:
        nr = np.zeros_like(rr)
    else:
        nr = np.rint(rr * (new_h - 1) / (h - 1)).astype(int)

    if w == 1 or new_w == 1:
        nc = np.zeros_like(cc)
    else:
        nc = np.rint(cc * (new_w - 1) / (w - 1)).astype(int)

    out[nr, nc] = True
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or bool(seen[r, c]):
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(seen[nr, nc]) or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            pts = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox(pts)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[pts[:, 0] - r0, pts[:, 1] - c0] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": local,
                    "area": int(local.sum()),
                }
            )

    objects.sort(key=lambda item: item["bbox"])
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grade each foreground object by its bbox occupancy:
    dense objects shrink, sparse objects dilate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full_like(g, BG)
    occupied = np.zeros_like(g, dtype=bool)

    for obj in _extract_objects(g):
        r0, c0, r1, c1 = obj["bbox"]
        mask = obj["mask"]
        h, w = map(int, mask.shape)
        box_area = int(h * w)
        dense = 10 * int(obj["area"]) > 7 * box_area

        if dense:
            new_mask = _shrink_mask(mask)
            top = int(r0 + (h - new_mask.shape[0]) // 2)
            left = int(c0 + (w - new_mask.shape[1]) // 2)
        else:
            new_mask = _dilate8(mask)
            top = int(r0 - 1)
            left = int(c0 - 1)

        coords = np.argwhere(new_mask)
        rr = coords[:, 0] + top
        cc = coords[:, 1] + left
        assert int(rr.min()) >= 0 and int(cc.min()) >= 0
        assert int(rr.max()) < g.shape[0] and int(cc.max()) < g.shape[1]
        assert not bool(occupied[rr, cc].any()), "transformed objects must stay disjoint"
        out[rr, cc] = int(obj["color"])
        occupied[rr, cc] = True

    return out


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    if random.random() < 0.5:
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
    else:
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
    mask[tr, tc] = True


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c), h, w):
            if not bool(mask[nr, nc]):
                cells.add((int(nr), int(nc)))
    return list(cells)


def _grow_connected(mask: np.ndarray, target: int) -> np.ndarray:
    out = mask.copy()
    while int(out.sum()) < target:
        fringe = _frontier(out)
        if not fringe:
            break
        rr, cc = random.choice(fringe)
        out[rr, cc] = True
    return out


def _make_sparse_mask(height: int, width: int) -> np.ndarray | None:
    area = int(height * width)
    low = max(int(np.ceil(SPARSE_FILL_FRAC_RANGE[0] * area)), 1)
    high = int(np.floor(SPARSE_FILL_FRAC_RANGE[1] * area))
    if low > high:
        return None

    family = random.choice(["cross", "corner_l", "stair"])
    mask = np.zeros((height, width), dtype=bool)

    if family == "cross":
        rr = random.randint(0, height - 1)
        cc = random.randint(0, width - 1)
        mask[rr, :] = True
        mask[:, cc] = True
    elif family == "corner_l":
        mask[0, :] = True
        mask[:, 0] = True
    else:
        r = c = 0
        mask[r, c] = True
        steps = ["D"] * (height - 1) + ["R"] * (width - 1)
        random.shuffle(steps)
        for step in steps:
            if step == "D":
                r += 1
            else:
                c += 1
            mask[r, c] = True

    base_area = int(mask.sum())
    if base_area > high:
        return None

    target = random.randint(max(low, base_area), high)
    mask = _grow_connected(mask, target)
    mask = _random_transform(mask)

    if not _connected4(mask):
        return None
    if not _touches_all_sides(mask):
        return None
    if not (10 * int(mask.sum()) <= 7 * area):
        return None
    return mask


def _make_dense_mask(height: int, width: int) -> np.ndarray | None:
    area = int(height * width)
    keep_low = int(np.ceil(DENSE_KEEP_FRAC_RANGE[0] * area))
    keep_high = int(np.floor(DENSE_KEEP_FRAC_RANGE[1] * area))
    if keep_low > keep_high:
        return None

    mask = np.ones((height, width), dtype=bool)
    target = random.randint(keep_low, keep_high)
    remove_goal = area - target

    for _ in range(remove_goal):
        candidates = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(candidates)
        removed = False
        for r, c in candidates:
            mask[r, c] = False
            if _touches_all_sides(mask) and _connected4(mask):
                removed = True
                break
            mask[r, c] = True
        if not removed:
            break

    mask = _random_transform(mask)
    if not _connected4(mask):
        return None
    if not _touches_all_sides(mask):
        return None
    if not (10 * int(mask.sum()) > 7 * area):
        return None
    return mask


def _build_spec(branch: str, color: int) -> dict | None:
    if branch == "dense":
        height = random.randint(*DENSE_HEIGHT_RANGE)
        width = random.randint(*DENSE_WIDTH_RANGE)
        mask = _make_dense_mask(height, width)
    else:
        height = random.randint(*SPARSE_HEIGHT_RANGE)
        width = random.randint(*SPARSE_WIDTH_RANGE)
        mask = _make_sparse_mask(height, width)

    if mask is None:
        return None

    h, w = map(int, mask.shape)
    if branch == "dense":
        out_mask = _shrink_mask(mask)
        out_top = (h - out_mask.shape[0]) // 2
        out_left = (w - out_mask.shape[1]) // 2
    else:
        out_mask = _dilate8(mask)
        out_top = -1
        out_left = -1

    return {
        "branch": branch,
        "color": int(color),
        "mask": mask,
        "height": h,
        "width": w,
        "out_mask": out_mask,
        "out_top": int(out_top),
        "out_left": int(out_left),
    }


def _slice_is_clear(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return 0 <= r0 < r1 <= blocked.shape[0] and 0 <= c0 < c1 <= blocked.shape[1] and not bool(blocked[r0:r1, c0:c1].any())


def _place_specs(specs: list[dict], height: int, width: int) -> np.ndarray | None:
    order = sorted(
        specs,
        key=lambda item: max(
            int(item["height"] * item["width"]),
            int(item["out_mask"].shape[0] * item["out_mask"].shape[1]),
        ),
        reverse=True,
    )

    grid = np.full((height, width), BG, dtype=int)
    input_blocked = np.zeros((height, width), dtype=bool)
    output_blocked = np.zeros((height, width), dtype=bool)

    for spec in order:
        h = int(spec["height"])
        w = int(spec["width"])
        out_h, out_w = map(int, spec["out_mask"].shape)
        out_dr = int(spec["out_top"])
        out_dc = int(spec["out_left"])
        candidates: list[tuple[int, int]] = []

        for top in range(height - h + 1):
            for left in range(width - w + 1):
                if not _slice_is_clear(input_blocked, top - 1, left - 1, top + h + 1, left + w + 1):
                    continue

                out_r0 = top + out_dr
                out_c0 = left + out_dc
                if not _slice_is_clear(output_blocked, out_r0 - 1, out_c0 - 1, out_r0 + out_h + 1, out_c0 + out_w + 1):
                    continue

                candidates.append((int(top), int(left)))

        if not candidates:
            return None

        top, left = random.choice(candidates)
        coords = np.argwhere(spec["mask"])
        rr = coords[:, 0] + top
        cc = coords[:, 1] + left
        grid[rr, cc] = int(spec["color"])

        input_blocked[top - 1 : top + h + 1, left - 1 : left + w + 1] = True
        out_r0 = top + out_dr
        out_c0 = left + out_dc
        output_blocked[out_r0 - 1 : out_r0 + out_h + 1, out_c0 - 1 : out_c0 + out_w + 1] = True

    return grid


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    dense_count = count // 2
    sparse_count = count // 2
    if dense_count + sparse_count < count:
        if random.random() < 0.5:
            dense_count += 1
        else:
            sparse_count += 1

    branches = ["dense"] * dense_count + ["sparse"] * sparse_count
    random.shuffle(branches)
    colors = random.sample(FG_COLORS, count)

    specs: list[dict] = []
    for branch, color in zip(branches, colors):
        spec = _build_spec(branch, color)
        if spec is None:
            return None
        specs.append(spec)

    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = _place_specs(specs, height, width)
    if grid is None:
        return None

    objects = _extract_objects(grid)
    if len(objects) != count:
        return None

    dense_seen = 0
    sparse_seen = 0
    for obj in objects:
        h = int(obj["mask"].shape[0])
        w = int(obj["mask"].shape[1])
        if 10 * int(obj["area"]) > 7 * h * w:
            dense_seen += 1
        else:
            sparse_seen += 1

    if dense_seen == 0 or sparse_seen == 0:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
