import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for r0 in range(height):
        for c0 in range(width):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _hole_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    bg = ~mask
    seen = np.zeros((height, width), dtype=bool)
    holes = 0

    for r0 in range(height):
        for c0 in range(width):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == height - 1 or c == width - 1:
                    touches_border = True
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict[str, object]] = []

    for r0 in range(height):
        for c0 in range(width):
            color = int(g[r0, c0])
            if color == 0 or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(pts, dtype=int)
            rr0, cc0, rr1, cc1 = _bbox(coords)
            mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
            mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
            patch = np.zeros(mask.shape, dtype=int)
            patch[mask] = color
            objects.append(
                {
                    "color": color,
                    "top": rr0,
                    "left": cc0,
                    "mask": mask,
                    "patch": patch,
                    "holes": _hole_count(mask),
                }
            )
    return objects


def _render_groups(height: int, objects: list[dict[str, object]]) -> np.ndarray:
    kept = [obj for obj in objects if int(obj["holes"]) > 0]
    if not kept:
        return np.zeros((height, 1), dtype=int)

    kept.sort(key=lambda obj: (int(obj["holes"]), int(obj["top"]), int(obj["left"])))
    total_width = 0
    for idx, obj in enumerate(kept):
        hole_count = int(obj["holes"])
        patch = np.asarray(obj["patch"], dtype=int)
        total_width += hole_count * patch.shape[1] + (hole_count - 1)
        if idx + 1 < len(kept):
            total_width += 2

    out = np.zeros((height, total_width), dtype=int)
    cursor = 0
    for idx, obj in enumerate(kept):
        top = int(obj["top"])
        hole_count = int(obj["holes"])
        patch = np.asarray(obj["patch"], dtype=int)
        patch_h, patch_w = patch.shape
        for rep in range(hole_count):
            left = cursor + rep * (patch_w + 1)
            view = out[top : top + patch_h, left : left + patch_w]
            mask = patch != 0
            view[mask] = patch[mask]
        cursor += hole_count * patch_w + (hole_count - 1)
        if idx + 1 < len(kept):
            cursor += 2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each object as many times as its number of internal holes and lay the groups out from left to right.

    Input structure:
    - The background is black.
    - The grid contains 3-5 separated monochrome 4-connected objects, each in a different color.
    - Every object may contain 0-3 enclosed black holes inside its own bounding box.

    Rule:
    - Count the enclosed black holes of each object by looking for background components inside the object's tight bounding box that do not touch that box border.
    - Discard every object with 0 holes.
    - For each remaining object with H holes, place H unchanged copies of that object in one horizontal group, with 1 black column between adjacent copies.
    - Order the groups from fewer holes to more holes; if two objects have the same hole count, keep their input reading order.
    - Keep each copied object's original top row, and separate consecutive groups by 2 black columns.

    Key insight:
    - Hole count controls three things at once: whether an object survives, how many times it is copied, and where its copy-group appears in the left-to-right order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _render_groups(int(g.shape[0]), _extract_objects(g))


def _pick_hole_sites(base_h: int, base_w: int, hole_count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []

    interior = [(r, c) for r in range(1, base_h - 1) for c in range(1, base_w - 1)]
    rng.shuffle(interior)

    def backtrack(start: int, chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == hole_count:
            return list(chosen)
        for idx in range(start, len(interior)):
            r, c = interior[idx]
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                res = backtrack(idx + 1, chosen)
                if res is not None:
                    return res
                chosen.pop()
        return None

    return backtrack(0, [])


def _sample_base_dims(hole_count: int, rng: random.Random) -> tuple[int, int]:
    options = []
    for h in range(2 if hole_count == 0 else 3, 7):
        for w in range(2, 6):
            if hole_count == 0:
                options.append((h, w))
                continue
            if w < 3:
                continue
            if ((h - 2) * (w - 2) + 1) // 2 >= hole_count:
                options.append((h, w))

    if hole_count >= 2:
        narrow = [dims for dims in options if dims[1] <= 4]
        if narrow:
            options = narrow
    return rng.choice(options)


def _add_protrusion(mask: np.ndarray, r0: int, c0: int, base_h: int, base_w: int, rng: random.Random) -> None:
    side = rng.choice(("top", "bottom", "left", "right"))
    depth = 1 if rng.random() < 0.8 else 2

    if side in ("top", "bottom"):
        span = rng.randint(1, min(2, base_w))
        start = c0 + rng.randint(0, base_w - span)
        if side == "top":
            rr = slice(r0 - depth, r0)
        else:
            rr = slice(r0 + base_h, r0 + base_h + depth)
        cc = slice(start, start + span)
    else:
        span = rng.randint(1, min(2, base_h))
        start = r0 + rng.randint(0, base_h - span)
        rr = slice(start, start + span)
        if side == "left":
            cc = slice(c0 - depth, c0)
        else:
            cc = slice(c0 + base_w, c0 + base_w + depth)

    mask[rr, cc] = True


def _optional_notches(mask: np.ndarray, hole_count: int, rng: random.Random) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(rng.randint(0, 2)):
        coords = np.argwhere(out)
        boundary = []
        height, width = out.shape
        for r, c in coords:
            if any(not (0 <= nr < height and 0 <= nc < width) or not out[nr, nc] for nr, nc in _neighbors4(int(r), int(c))):
                boundary.append((int(r), int(c)))
        rng.shuffle(boundary)
        for r, c in boundary:
            test = out.copy()
            test[r, c] = False
            if test.any() and _is_connected(test) and _hole_count(test) == hole_count:
                out = test
                break
    return out


def _make_object_mask(hole_count: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        base_h, base_w = _sample_base_dims(hole_count, rng)
        holes = _pick_hole_sites(base_h, base_w, hole_count, rng)
        if holes is None:
            continue

        canvas = np.zeros((base_h + 4, base_w + 4), dtype=bool)
        r0 = c0 = 2
        canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True
        for hr, hc in holes:
            canvas[r0 + hr, c0 + hc] = False

        for _ in range(rng.randint(0, 4)):
            _add_protrusion(canvas, r0, c0, base_h, base_w, rng)

        mask = _crop_mask(canvas)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.3:
            mask = mask.T

        mask = _optional_notches(mask, hole_count, rng)
        if not _is_connected(mask):
            continue
        if _hole_count(mask) != hole_count:
            continue
        if mask.shape[0] > 8 or mask.shape[1] > 6:
            continue
        return mask.astype(bool)
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(height, height - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(width, width - dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _place_objects(
    height: int,
    width: int,
    objects: list[dict[str, object]],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, object]] = []

    for obj in sorted(objects, key=lambda item: int(np.sum(np.asarray(item["mask"], dtype=bool))), reverse=True):
        mask = np.asarray(obj["mask"], dtype=bool)
        patch_h, patch_w = mask.shape
        blocked = _dilate8(occupied)
        candidates = []
        for top in range(1, height - patch_h):
            for left in range(1, width - patch_w):
                if np.any(blocked[top : top + patch_h, left : left + patch_w] & mask):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        occupied[top : top + patch_h, left : left + patch_w] |= mask
        placed.append(
            {
                "color": int(obj["color"]),
                "holes": int(obj["holes"]),
                "mask": mask,
                "top": int(top),
                "left": int(left),
            }
        )
    return placed


def _render_input(height: int, width: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for obj in placed:
        mask = np.asarray(obj["mask"], dtype=bool)
        top = int(obj["top"])
        left = int(obj["left"])
        color = int(obj["color"])
        view = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
        view[mask] = color
    return grid


def _planned_output(height: int, placed: list[dict[str, object]]) -> np.ndarray:
    objects = []
    for obj in placed:
        mask = np.asarray(obj["mask"], dtype=bool)
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(obj["color"])
        objects.append(
            {
                "patch": patch,
                "holes": int(obj["holes"]),
                "top": int(obj["top"]),
                "left": int(obj["left"]),
            }
        )
    objects.sort(key=lambda obj: (int(obj["holes"]), int(obj["top"]), int(obj["left"])))
    return _render_groups(height, objects)


def _hole_plan(rng: random.Random) -> list[int]:
    total_objects = rng.randint(3, 5)
    positive_count = rng.randint(1, min(3, total_objects))
    positive = rng.sample([1, 2, 3], positive_count)
    holes = positive + [0] * (total_objects - positive_count)
    rng.shuffle(holes)
    return holes


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    holes = _hole_plan(rng)
    colors = rng.sample(list(range(1, 10)), len(holes))

    specs = []
    for hole_count, color in zip(holes, colors):
        mask = _make_object_mask(hole_count, rng)
        if mask is None:
            return None
        specs.append({"holes": hole_count, "color": color, "mask": mask})

    positive_specs = [spec for spec in specs if int(spec["holes"]) > 0]
    output_width = 0
    for idx, spec in enumerate(sorted(positive_specs, key=lambda item: int(item["holes"]))):
        patch_w = int(np.asarray(spec["mask"], dtype=bool).shape[1])
        output_width += int(spec["holes"]) * patch_w + (int(spec["holes"]) - 1)
        if idx + 1 < len(positive_specs):
            output_width += 2
    if output_width > 30:
        return None

    max_h = max(int(np.asarray(spec["mask"], dtype=bool).shape[0]) for spec in specs)
    max_w = max(int(np.asarray(spec["mask"], dtype=bool).shape[1]) for spec in specs)
    min_height = max(10, max_h + 2)
    min_width = max(12, max_w + 2)
    height = rng.randint(min_height, min(22, max(min_height, min_height + 6)))
    width = rng.randint(min_width, min(24, max(min_width, min_width + 8)))

    placed = _place_objects(height, width, specs, rng)
    if placed is None:
        return None

    grid = _render_input(height, width, placed)
    extracted = _extract_objects(grid)
    if len(extracted) != len(specs):
        return None
    if sorted(int(obj["holes"]) for obj in extracted) != sorted(holes):
        return None

    expected = _planned_output(height, placed)
    if max(expected.shape) > 30:
        return None
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
