from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
MARKER = 2
OBJECT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 27
GRID_MAX = 30
GEN_ATTEMPTS = 500
MASK_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            nxt = (int(nr), int(nc))
            if mask[nr, nc] and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _spans_full_bbox(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    return (
        rows.size > 0
        and int(rows.min()) == 0
        and int(rows.max()) == mask.shape[0] - 1
        and int(cols.min()) == 0
        and int(cols.max()) == mask.shape[1] - 1
    )


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                holes += 1
    return holes


def _mirror_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.fliplr(mask).copy()
    return np.flipud(mask).copy()


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _paste_mask(
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    top: int,
    left: int,
    preserve_color: int | None = None,
) -> None:
    h, w = out.shape
    mh, mw = mask.shape
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        if 0 <= r < h and 0 <= c < w:
            if preserve_color is not None and int(out[r, c]) == int(preserve_color):
                continue
            out[r, c] = int(color)


def _stamp_bool(out: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = out.shape
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = True


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr0 = max(0, int(r) - 1)
        rr1 = min(h, int(r) + 2)
        cc0 = max(0, int(c) - 1)
        cc1 = min(w, int(c) + 2)
        out[rr0:rr1, cc0:cc1] = True
    return out


def _find_objects(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, MARKER) or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            bbox = _bbox_from_coords(coords)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": _mask_from_coords(coords, bbox),
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each object toward its aligned red edge marker by a hole-count distance, then draw a mirrored copy at the landing spot.

    Input structure:
    - The background is black.
    - The outer border contains zero or more isolated red marker pixels, and every other border cell is background.
    - Inside the border are several 4-connected single-color objects in colors other than black and red; each fits inside a bbox of at most 5x5 and contains one to four 1x1 black holes.
    - Each object aligns with exactly one red border marker: a top/bottom marker shares a column with the object's bbox, while a left/right marker shares a row with the object's bbox.

    Rule:
    - For each object, find its unique aligned red marker to determine the projection direction: top -> up, bottom -> down, left -> left, right -> right.
    - Count the object's enclosed 1x1 black holes. Let that count be N.
    - Translate the whole object by N times its bbox span in the projection direction, using bbox height for up/down motion and bbox width for left/right motion.
    - At the landing position, draw a mirrored copy of the object in the same color: use a left-right mirror for up/down projections and an up-down mirror for left/right projections. Keep the original objects and red markers.

    Key insight:
    The marker-object alignment picks the direction, the number of enclosed black holes picks the travel distance, and the landing copy is a reflection rather than a plain translation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = g.shape

    for obj in _find_objects(g):
        r0, c0, r1, c1 = obj["bbox"]
        candidates: list[str] = []
        for c in range(c0, c1 + 1):
            if int(g[0, c]) == MARKER:
                candidates.append("top")
            if int(g[height - 1, c]) == MARKER:
                candidates.append("bottom")
        for r in range(r0, r1 + 1):
            if int(g[r, 0]) == MARKER:
                candidates.append("left")
            if int(g[r, width - 1]) == MARKER:
                candidates.append("right")
        assert len(candidates) == 1, "each object must align with exactly one border marker"

        side = candidates[0]
        mask = obj["mask"]
        holes = _count_holes(mask)
        assert 1 <= holes <= 4, "objects must contain one to four unit holes"

        span = mask.shape[0] if side in ("top", "bottom") else mask.shape[1]
        dist = holes * span
        dr = -dist if side == "top" else dist if side == "bottom" else 0
        dc = -dist if side == "left" else dist if side == "right" else 0
        mirrored = _mirror_mask(mask, side)
        _paste_mask(out, mirrored, int(obj["color"]), r0 + dr, c0 + dc, preserve_color=MARKER)

    return out


def _build_hole_layouts() -> dict[int, list[np.ndarray]]:
    layouts = {n: [] for n in range(1, 5)}
    for h in range(3, 6):
        for w in range(3, 6):
            interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            for n in range(1, min(4, len(interior)) + 1):
                for holes in combinations(interior, n):
                    if any(abs(r1 - r2) + abs(c1 - c2) == 1 for (r1, c1), (r2, c2) in combinations(holes, 2)):
                        continue
                    mask = np.ones((h, w), dtype=bool)
                    for r, c in holes:
                        mask[r, c] = False
                    if _connected(mask) and _count_holes(mask) == n:
                        layouts[n].append(mask)
    return layouts


HOLE_LAYOUTS = _build_hole_layouts()


def _random_symmetry(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.randrange(2):
        out = np.fliplr(out)
    if random.randrange(2):
        out = np.flipud(out)
    return out.copy()


def _random_object_mask(n_holes: int) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        mask = random.choice(HOLE_LAYOUTS[n_holes]).copy()
        hole_cells = [
            (r, c)
            for r in range(1, mask.shape[0] - 1)
            for c in range(1, mask.shape[1] - 1)
            if not mask[r, c]
            and mask[r - 1, c]
            and mask[r + 1, c]
            and mask[r, c - 1]
            and mask[r, c + 1]
        ]
        protected = set()
        for r, c in hole_cells:
            protected.update(_neighbors4(r, c, mask.shape[0], mask.shape[1]))

        boundary = [
            (r, c)
            for r in range(mask.shape[0])
            for c in range(mask.shape[1])
            if mask[r, c]
            and (r in (0, mask.shape[0] - 1) or c in (0, mask.shape[1] - 1))
            and (r, c) not in protected
        ]
        random.shuffle(boundary)
        target_removals = random.randint(0, min(4, len(boundary)))
        removed = 0
        for r, c in boundary:
            if removed >= target_removals:
                break
            trial = mask.copy()
            trial[r, c] = False
            if _connected(trial) and _spans_full_bbox(trial) and _count_holes(trial) == n_holes:
                mask = trial
                removed += 1

        mask = _random_symmetry(mask)
        if _connected(mask) and _spans_full_bbox(mask) and _count_holes(mask) == n_holes:
            return mask
    raise RuntimeError(f"failed to build mask with {n_holes} holes")


def _landing_origin(obj: dict[str, object]) -> tuple[int, int]:
    mask = obj["mask"]
    holes = int(obj["holes"])
    side = str(obj["side"])
    span = mask.shape[0] if side in ("top", "bottom") else mask.shape[1]
    dist = holes * span
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    if side == "top":
        return r0 - dist, c0
    if side == "bottom":
        return r0 + dist, c0
    if side == "left":
        return r0, c0 - dist
    return r0, c0 + dist


def _group_occupancy(shape: tuple[int, int], obj: dict[str, object]) -> np.ndarray:
    occ = np.zeros(shape, dtype=bool)
    mask = obj["mask"]
    _stamp_bool(occ, mask, int(obj["r0"]), int(obj["c0"]))
    lr, lc = _landing_origin(obj)
    _stamp_bool(occ, _mirror_mask(mask, str(obj["side"])), lr, lc)
    return occ


def _placement_ranges(height: int, width: int, side: str, holes: int, mask: np.ndarray):
    h, w = mask.shape
    if side == "top":
        return 1 + holes * h, height - h - 1, 1, width - w - 1
    if side == "bottom":
        return 1, height - (holes + 1) * h - 1, 1, width - w - 1
    if side == "left":
        return 1, height - h - 1, 1 + holes * w, width - w - 1
    return 1, height - h - 1, 1, width - (holes + 1) * w - 1


def _exclusive_columns(objects: list[dict[str, object]], idx: int) -> list[int]:
    obj = objects[idx]
    c0 = int(obj["c0"])
    w = obj["mask"].shape[1]
    choices = []
    for c in range(c0, c0 + w):
        if all(
            not (int(other["c0"]) <= c < int(other["c0"]) + other["mask"].shape[1])
            for j, other in enumerate(objects)
            if j != idx
        ):
            choices.append(c)
    return choices


def _exclusive_rows(objects: list[dict[str, object]], idx: int) -> list[int]:
    obj = objects[idx]
    r0 = int(obj["r0"])
    h = obj["mask"].shape[0]
    choices = []
    for r in range(r0, r0 + h):
        if all(
            not (int(other["r0"]) <= r < int(other["r0"]) + other["mask"].shape[0])
            for j, other in enumerate(objects)
            if j != idx
        ):
            choices.append(r)
    return choices


def _assign_markers(height: int, width: int, objects: list[dict[str, object]]) -> bool:
    for idx, obj in enumerate(objects):
        side = str(obj["side"])
        if side in ("top", "bottom"):
            choices = _exclusive_columns(objects, idx)
            if not choices:
                return False
            c = random.choice(choices)
            obj["marker"] = (0 if side == "top" else height - 1, c)
        else:
            choices = _exclusive_rows(objects, idx)
            if not choices:
                return False
            r = random.choice(choices)
            obj["marker"] = (r, 0 if side == "left" else width - 1)
    return True


def _aligned_with_marker(obj: dict[str, object], marker: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = marker
    h, w = shape
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    r1 = r0 + obj["mask"].shape[0] - 1
    c1 = c0 + obj["mask"].shape[1] - 1
    if r == 0:
        return c0 <= c <= c1
    if r == h - 1:
        return c0 <= c <= c1
    if c == 0:
        return r0 <= r <= r1
    if c == w - 1:
        return r0 <= r <= r1
    return False


def _render_input(height: int, width: int, objects: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for obj in objects:
        _paste_mask(grid, obj["mask"], int(obj["color"]), int(obj["r0"]), int(obj["c0"]))
    for obj in objects:
        mr, mc = obj["marker"]
        grid[mr, mc] = MARKER
    return grid


def _render_expected(inp: np.ndarray, objects: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        lr, lc = _landing_origin(obj)
        mirrored = _mirror_mask(obj["mask"], str(obj["side"]))
        _paste_mask(out, mirrored, int(obj["color"]), lr, lc, preserve_color=MARKER)
    return out


def _try_place_object(
    height: int,
    width: int,
    blocked: np.ndarray,
    side: str,
    color: int,
    holes: int,
) -> dict[str, object] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        mask = _random_object_mask(holes)
        r_lo, r_hi, c_lo, c_hi = _placement_ranges(height, width, side, holes, mask)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            obj = {
                "side": side,
                "color": color,
                "holes": holes,
                "mask": mask,
                "r0": r0,
                "c0": c0,
            }
            occ = _group_occupancy((height, width), obj)
            if np.any(occ & blocked):
                continue
            obj["occ"] = occ
            return obj
    return None


def _validate_instance(inp: np.ndarray, objects: list[dict[str, object]], expected: np.ndarray) -> bool:
    h, w = inp.shape
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if int(inp[r, c]) != BG and int(inp[r, c]) != MARKER:
                    return False

    markers = [tuple(map(int, obj["marker"])) for obj in objects]
    for idx, obj in enumerate(objects):
        aligned = [m for m in markers if _aligned_with_marker(obj, m, inp.shape)]
        if len(aligned) != 1 or aligned[0] != markers[idx]:
            return False
    for marker in markers:
        matches = [obj for obj in objects if _aligned_with_marker(obj, marker, inp.shape)]
        if len(matches) != 1:
            return False

    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    n_objects = random.randint(1, 4)
    sides = random.sample(SIDES, n_objects)
    colors = random.sample(OBJECT_COLORS, n_objects)

    blocked = np.zeros((height, width), dtype=bool)
    objects: list[dict[str, object]] = []
    for side, color in zip(sides, colors):
        holes = random.randint(1, 4)
        obj = _try_place_object(height, width, blocked, side, color, holes)
        if obj is None:
            return None
        blocked |= _dilate8(obj["occ"])
        objects.append(obj)

    if not _assign_markers(height, width, objects):
        return None

    inp = _render_input(height, width, objects)
    expected = _render_expected(inp, objects)
    if not _validate_instance(inp, objects, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
