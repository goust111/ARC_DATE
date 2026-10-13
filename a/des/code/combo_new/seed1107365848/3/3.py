from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


YELLOW = 4
SIDES = ("top", "bottom", "left", "right")
GEN_BG = 0
GEN_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_RETRIES = 400
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
NUM_RINGS_CHOICES = [2, 3, 4]
OUTER_SIZE_SMALL = (5, 8)
OUTER_SIZE_LARGE = (6, 10)
FRAGMENT_DIM_CAP = 5
GAP_SEPARATION_RANGE = (1, 4)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == bg or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            rs = arr[:, 0]
            cs = arr[:, 1]
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _ring_info(component: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in component["coords"]:
        mask[int(r) - r0, int(c) - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    missing = np.argwhere(border & ~mask)
    if missing.shape[0] < 2:
        return None

    if np.count_nonzero(mask) != np.count_nonzero(border) - missing.shape[0]:
        return None

    rs = missing[:, 0]
    cs = missing[:, 1]

    if np.all(rs == 0):
        cols = sorted(map(int, cs))
        if cols[0] >= 1 and cols[-1] <= w - 2 and cols == list(range(cols[0], cols[-1] + 1)):
            return {"side": "top", "gap_start": int(c0 + cols[0]), "gap_end": int(c0 + cols[-1])}
    if np.all(rs == h - 1):
        cols = sorted(map(int, cs))
        if cols[0] >= 1 and cols[-1] <= w - 2 and cols == list(range(cols[0], cols[-1] + 1)):
            return {"side": "bottom", "gap_start": int(c0 + cols[0]), "gap_end": int(c0 + cols[-1])}
    if np.all(cs == 0):
        rows = sorted(map(int, rs))
        if rows[0] >= 1 and rows[-1] <= h - 2 and rows == list(range(rows[0], rows[-1] + 1)):
            return {"side": "left", "gap_start": int(r0 + rows[0]), "gap_end": int(r0 + rows[-1])}
    if np.all(cs == w - 1):
        rows = sorted(map(int, rs))
        if rows[0] >= 1 and rows[-1] <= h - 2 and rows == list(range(rows[0], rows[-1] + 1)):
            return {"side": "right", "gap_start": int(r0 + rows[0]), "gap_end": int(r0 + rows[-1])}
    return None


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _flip_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.flipud(mask)
    return np.fliplr(mask)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Absorb same-color fragments through each broken ring and mirror them into yellow interior inserts.

    Input structure:
    - Black background with several different-colored one-pixel rectangular rings.
    - Each ring is missing one continuous segment on exactly one side, leaving a gap of width at least two.
    - Outside the gap direction of each ring sit disconnected same-color fragment pixels; the ring interiors are otherwise empty.

    Rule:
    - Identify every ring and determine whether its gap is on the top, bottom, left, or right side.
    - Collect all same-color pixels that lie outside that ring on the gap side, take their tight bounding box, and erase them from the exterior.
    - Slide that fragment bbox straight into the ring interior along the gap direction while preserving the perpendicular coordinate, so it touches the opposite inner wall.
    - Mirror the inserted fragment group across the corresponding axis (vertical flip for top/bottom gaps, horizontal flip for left/right gaps) and repaint it yellow.

    Key insight:
    - The missing wall segment encodes two things at once: where the matching fragments come from, and which mirror axis must be applied after the fragments are pulled inside.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    comps = _find_components(g, bg=bg)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    rings = []
    for color, color_comps in by_color.items():
        candidates = []
        for comp in color_comps:
            info = _ring_info(comp)
            if info is not None:
                candidates.append((comp, info))
        if candidates:
            comp, info = max(candidates, key=lambda item: int(item[0]["area"]))
            rings.append({"color": int(color), "component": comp, **info})

    for ring in rings:
        color = int(ring["color"])
        side = ring["side"]
        r0, c0, r1, c1 = map(int, ring["component"]["bbox"])
        ring_cells = {tuple(map(int, xy)) for xy in ring["component"]["coords"]}
        color_cells = [tuple(map(int, xy)) for xy in np.argwhere(g == color)]

        if side == "top":
            frag_coords = [(r, c) for r, c in color_cells if (r, c) not in ring_cells and r < r0]
        elif side == "bottom":
            frag_coords = [(r, c) for r, c in color_cells if (r, c) not in ring_cells and r > r1]
        elif side == "left":
            frag_coords = [(r, c) for r, c in color_cells if (r, c) not in ring_cells and c < c0]
        else:
            frag_coords = [(r, c) for r, c in color_cells if (r, c) not in ring_cells and c > c1]

        if not frag_coords:
            continue

        frag_mask, frag_bbox = _mask_from_coords(frag_coords)
        fr0, fc0, _fr1, _fc1 = map(int, frag_bbox)
        fh, fw = map(int, frag_mask.shape)

        for r, c in frag_coords:
            out[r, c] = int(bg)

        transformed = _flip_mask(frag_mask, side)
        if side == "top":
            dest_r0 = int(r1 - fh)
            dest_c0 = int(fc0)
        elif side == "bottom":
            dest_r0 = int(r0 + 1)
            dest_c0 = int(fc0)
        elif side == "left":
            dest_r0 = int(fr0)
            dest_c0 = int(c1 - fw)
        else:
            dest_r0 = int(fr0)
            dest_c0 = int(c0 + 1)

        _paint_mask(out, transformed, dest_r0, dest_c0, YELLOW)

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _random_fragment_mask(h: int, w: int) -> np.ndarray:
    assert h >= 1 and w >= 1 and h * w >= 2
    target_min = max(2, h, w)
    target_max = h * w
    if target_max > target_min and h * w > 2 and random.random() < 0.7:
        target_max -= 1
    target = random.randint(target_min, target_max)

    for _ in range(80):
        cells: set[tuple[int, int]] = set()
        rows = list(range(h))
        cols = list(range(w))
        random.shuffle(rows)
        random.shuffle(cols)

        if h >= w:
            assignment = cols[:] + [random.choice(cols) for _ in range(h - w)]
            random.shuffle(assignment)
            for r, c in zip(rows, assignment):
                cells.add((r, c))
        else:
            assignment = rows[:] + [random.choice(rows) for _ in range(w - h)]
            random.shuffle(assignment)
            for c, r in zip(cols, assignment):
                cells.add((r, c))

        remaining = [(r, c) for r in range(h) for c in range(w) if (r, c) not in cells]
        need = target - len(cells)
        if need > 0:
            cells.update(random.sample(remaining, need))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _ring_info(
            {
                "bbox": (0, 0, h - 1, w - 1),
                "coords": np.argwhere(mask),
            }
        ) is None:
            return mask

    raise RuntimeError("failed to generate fragment mask")


def _ring_patch(h: int, w: int, side: str, gap_start: int, gap_len: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        patch[0, gap_start : gap_start + gap_len] = 0
    elif side == "bottom":
        patch[-1, gap_start : gap_start + gap_len] = 0
    elif side == "left":
        patch[gap_start : gap_start + gap_len, 0] = 0
    else:
        patch[gap_start : gap_start + gap_len, -1] = 0
    return patch


def _make_object(color: int, side: str, *, small: bool) -> dict:
    size_lo, size_hi = OUTER_SIZE_SMALL if small else OUTER_SIZE_LARGE

    for _ in range(120):
        h = random.randint(size_lo, size_hi)
        w = random.randint(size_lo, size_hi)
        inner_h = h - 2
        inner_w = w - 2

        if side in ("top", "bottom"):
            gap_len = random.randint(2, inner_w)
            gap_start = random.randint(1, w - gap_len - 1)
            dims = [
                (fh, fw)
                for fh in range(1, min(inner_h, FRAGMENT_DIM_CAP) + 1)
                for fw in range(1, min(gap_len, FRAGMENT_DIM_CAP) + 1)
                if fh * fw >= 2
            ]
            fh, fw = random.choice(dims)
            frag_c = random.randint(gap_start, gap_start + gap_len - fw)
            gap_sep = random.randint(*GAP_SEPARATION_RANGE)
            frag_mask = _random_fragment_mask(fh, fw)

            if side == "top":
                patch = np.zeros((fh + gap_sep + h, w), dtype=int)
                ring_r, ring_c = fh + gap_sep, 0
                frag_r, frag_c0 = 0, frag_c
            else:
                patch = np.zeros((h + gap_sep + fh, w), dtype=int)
                ring_r, ring_c = 0, 0
                frag_r, frag_c0 = h + gap_sep, frag_c
        else:
            gap_len = random.randint(2, inner_h)
            gap_start = random.randint(1, h - gap_len - 1)
            dims = [
                (fh, fw)
                for fh in range(1, min(gap_len, FRAGMENT_DIM_CAP) + 1)
                for fw in range(1, min(inner_w, FRAGMENT_DIM_CAP) + 1)
                if fh * fw >= 2
            ]
            fh, fw = random.choice(dims)
            frag_r = random.randint(gap_start, gap_start + gap_len - fh)
            gap_sep = random.randint(*GAP_SEPARATION_RANGE)
            frag_mask = _random_fragment_mask(fh, fw)

            if side == "left":
                patch = np.zeros((h, fw + gap_sep + w), dtype=int)
                ring_r, ring_c = 0, fw + gap_sep
                frag_r0, frag_c = frag_r, 0
            else:
                patch = np.zeros((h, w + gap_sep + fw), dtype=int)
                ring_r, ring_c = 0, 0
                frag_r0, frag_c = frag_r, w + gap_sep

        ring = _ring_patch(h, w, side, gap_start, gap_len, color)
        patch[ring_r : ring_r + h, ring_c : ring_c + w] = ring

        if side in ("top", "bottom"):
            _paint_mask(patch, frag_mask, frag_r, frag_c0, color)
            fragment_origin = (frag_r, frag_c0)
        else:
            _paint_mask(patch, frag_mask, frag_r0, frag_c, color)
            fragment_origin = (frag_r0, frag_c)

        return {
            "color": int(color),
            "side": side,
            "patch": patch,
            "mask": patch != 0,
            "ring_origin": (ring_r, ring_c),
            "ring_shape": (h, w),
            "fragment_origin": fragment_origin,
            "fragment_mask": frag_mask,
        }
    raise RuntimeError("failed to build object")


def _place_objects(objects: list[dict], height: int, width: int) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    order = sorted(range(len(objects)), key=lambda i: int(objects[i]["patch"].size), reverse=True)
    for idx in order:
        obj = objects[idx]
        patch = obj["patch"]
        mask = obj["mask"]
        ph, pw = map(int, patch.shape)
        if ph > height or pw > width:
            return None

        valid = []
        blocked = _dilate8(occupied)
        for r0 in range(height - ph + 1):
            for c0 in range(width - pw + 1):
                if not np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & mask):
                    valid.append((r0, c0))
        if not valid:
            return None

        base_r, base_c = random.choice(valid)
        grid[base_r : base_r + ph, base_c : base_c + pw][mask] = patch[mask]
        occupied[base_r : base_r + ph, base_c : base_c + pw] |= mask

        ring_r, ring_c = obj["ring_origin"]
        ring_h, ring_w = obj["ring_shape"]
        frag_r, frag_c = obj["fragment_origin"]
        frag_h, frag_w = obj["fragment_mask"].shape

        placed.append(
            {
                "color": obj["color"],
                "side": obj["side"],
                "ring_bbox": (
                    int(base_r + ring_r),
                    int(base_c + ring_c),
                    int(base_r + ring_r + ring_h - 1),
                    int(base_c + ring_c + ring_w - 1),
                ),
                "fragment_bbox": (
                    int(base_r + frag_r),
                    int(base_c + frag_c),
                    int(base_r + frag_r + frag_h - 1),
                    int(base_c + frag_c + frag_w - 1),
                ),
                "fragment_mask": obj["fragment_mask"],
            }
        )
    return grid, placed


def _expected_output(inp: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for obj in placed:
        fr0, fc0, _fr1, _fc1 = obj["fragment_bbox"]
        ring_r0, ring_c0, ring_r1, ring_c1 = obj["ring_bbox"]
        mask = obj["fragment_mask"]
        fh, fw = mask.shape
        side = obj["side"]

        patch = out[fr0 : fr0 + fh, fc0 : fc0 + fw]
        patch[mask] = GEN_BG

        transformed = _flip_mask(mask, side)
        if side == "top":
            dest_r0 = int(ring_r1 - fh)
            dest_c0 = int(fc0)
        elif side == "bottom":
            dest_r0 = int(ring_r0 + 1)
            dest_c0 = int(fc0)
        elif side == "left":
            dest_r0 = int(fr0)
            dest_c0 = int(ring_c1 - fw)
        else:
            dest_r0 = int(fr0)
            dest_c0 = int(ring_c0 + 1)
        _paint_mask(out, transformed, dest_r0, dest_c0, YELLOW)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    num_rings = random.choice(NUM_RINGS_CHOICES)
    colors = random.sample(GEN_COLORS, num_rings)
    sides = random.sample(list(SIDES), num_rings)

    objects = [_make_object(color, side, small=num_rings >= 4) for color, side in zip(colors, sides)]
    placed = _place_objects(objects, height, width)
    if placed is None:
        return None

    inp, metadata = placed
    expected = _expected_output(inp, metadata)
    if expected.shape != inp.shape or max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
