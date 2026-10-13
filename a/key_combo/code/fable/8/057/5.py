from __future__ import annotations

from collections import deque
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        raise ValueError("empty mask")
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    return total > 0 and len(_components(mask)) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _hole_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    bg = ~mask
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if bg[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                add(nr, nc)

    holes = bg & ~exterior
    return _components(holes)


def _hole_count(mask: np.ndarray) -> int:
    return len(_hole_components(mask))


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    objects: list[dict] = []
    for color in colors:
        full_mask = g == int(color)
        comps = _components(full_mask)
        assert len(comps) == 1, "each foreground color must form exactly one component"
        mask = _crop_mask(full_mask)
        objects.append({"color": int(color), "mask": mask})
    return bg, objects


def _compose_output(
    stamp_mask: np.ndarray,
    stamp_color: int,
    layout_mask: np.ndarray,
    layout_color: int,
) -> np.ndarray:
    sh, sw = stamp_mask.shape
    lh, lw = layout_mask.shape
    out = np.zeros((lh * sh, lw * sw), dtype=int)
    inv_stamp = ~stamp_mask
    for r in range(lh):
        for c in range(lw):
            block = out[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw]
            if layout_mask[r, c]:
                block[stamp_mask] = int(stamp_color)
            else:
                block[inv_stamp] = int(layout_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the hole-bearing mask as a stamp and use its complement on the other mask's background blocks.

    Input structure:
    - A black background contains exactly two separated monochrome binary masks in different colors.
    - Each colored mask is interpreted inside its own tight bounding box.
    - Exactly one mask contains at least one fully enclosed hole and the other contains none.

    Rule:
    - The mask with a hole is the stamp S in color s; the hole-free mask is the layout L in color l.
    - Output size is `(rows(L) * rows(S), cols(L) * cols(S))`, using the cropped bbox sizes of both masks.
    - For each foreground cell of L, place one copy of S: stamp foreground stays color s and stamp background stays black.
    - For each background cell inside L's bbox, place the complement of S: stamp foreground becomes black and stamp background becomes color l.

    Key insight:
    - Role assignment is determined by topology, not size: the holed object must be used as the stamp, and every layout-background block is filled by the stamp's visible complement.
    """
    _bg, objects = _extract_objects(grid)
    hole_flags = [_hole_count(obj["mask"]) > 0 for obj in objects]
    assert sum(hole_flags) == 1, "exactly one object must contain a closed hole"

    stamp = objects[0] if hole_flags[0] else objects[1]
    layout = objects[1] if hole_flags[0] else objects[0]
    return _compose_output(stamp["mask"], stamp["color"], layout["mask"], layout["color"])


MIN_DIM = 3
MAX_DIM = 6
MAX_OUTPUT = 30
MIN_CANVAS = 12
MAX_CANVAS = 24
MIN_GAP = 2
GEN_ATTEMPTS = 400


def _random_connected_subset(domain: list[tuple[int, int]], area: int) -> set[tuple[int, int]] | None:
    domain_set = set(domain)
    if not domain_set or area < 1 or area > len(domain_set):
        return None

    start = random.choice(domain)
    chosen = {start}
    frontier = {cell for cell in _neighbors4(*start) if cell in domain_set}
    while len(chosen) < area:
        frontier -= chosen
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        chosen.add(cell)
        for nbr in _neighbors4(*cell):
            if nbr in domain_set and nbr not in chosen:
                frontier.add(nbr)
    return chosen


def _carve_border_notch(mask: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape
    candidate = mask.copy()
    side = random.choice(["top", "bottom", "left", "right"])

    if side in {"top", "bottom"}:
        if w < 3:
            return None
        depth = random.randint(1, max(1, min(2, h - 2)))
        run = random.randint(1, w - 1)
        start = random.randint(0, w - run)
        rows = range(0, depth) if side == "top" else range(h - depth, h)
        for r in rows:
            for c in range(start, start + run):
                candidate[r, c] = False
    else:
        if h < 3:
            return None
        depth = random.randint(1, max(1, min(2, w - 2)))
        run = random.randint(1, h - 1)
        start = random.randint(0, h - run)
        cols = range(0, depth) if side == "left" else range(w - depth, w)
        for r in range(start, start + run):
            for c in cols:
                candidate[r, c] = False

    if not candidate.any():
        return None
    if not _touches_all_sides(candidate):
        return None
    if not _is_connected(candidate):
        return None
    if _hole_count(candidate) != 1:
        return None
    return candidate


def _make_stamp_mask(h: int, w: int, large_hole: bool) -> np.ndarray | None:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if not interior:
        return None

    for _ in range(200):
        mask = np.ones((h, w), dtype=bool)
        max_area = len(interior)
        if large_hole:
            lo = max(2, max_area // 4)
            hi = max(lo, min(max_area, (2 * max_area) // 3))
        else:
            lo = 1
            hi = min(max_area, 2)
        hole_area = random.randint(lo, hi)
        hole = _random_connected_subset(interior, hole_area)
        if hole is None:
            continue
        for r, c in hole:
            mask[r, c] = False

        for _ in range(random.randint(0, 2)):
            notched = _carve_border_notch(mask)
            if notched is not None:
                mask = notched

        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        if _hole_count(mask) != 1:
            continue
        return mask
    return None


def _path_cells(a: tuple[int, int], b: tuple[int, int], order: str) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    cells = [(r, c)]
    if order == "vh":
        step = 1 if rr >= r else -1
        for nr in range(r + step, rr + step, step):
            cells.append((nr, c))
        step = 1 if cc >= c else -1
        for nc in range(c + step, cc + step, step):
            cells.append((rr, nc))
    else:
        step = 1 if cc >= c else -1
        for nc in range(c + step, cc + step, step):
            cells.append((r, nc))
        step = 1 if rr >= r else -1
        for nr in range(r + step, rr + step, step):
            cells.append((nr, cc))
    return cells


def _safe_growth_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                trial = mask.copy()
                trial[r, c] = True
                if _hole_count(trial) == 0:
                    candidates.append((r, c))
    return candidates


def _make_layout_mask(h: int, w: int, denser: bool) -> np.ndarray | None:
    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)
        hub_r = random.randrange(h)
        hub_c = random.randrange(w)
        hub = (hub_r, hub_c)
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        for anchor in anchors:
            for r, c in _path_cells(anchor, hub, random.choice(["vh", "hv"])):
                mask[r, c] = True

        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        if _hole_count(mask) != 0:
            continue

        limit = min(h * w - 1, int((0.72 if denser else 0.58) * h * w))
        base_area = int(mask.sum())
        target_lo = max(base_area + 1, (h * w) // 3)
        target_hi = max(base_area, limit)
        target = random.randint(target_lo, target_hi) if target_hi >= target_lo else base_area
        while int(mask.sum()) < target:
            candidates = _safe_growth_candidates(mask)
            if not candidates:
                break
            r, c = random.choice(candidates)
            mask[r, c] = True

        if int(mask.sum()) == h * w:
            continue
        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        if _hole_count(mask) != 0:
            continue
        return mask
    return None


def _colored_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_two_objects(mask_a: np.ndarray, color_a: int, mask_b: np.ndarray, color_b: int) -> np.ndarray | None:
    h1, w1 = mask_a.shape
    h2, w2 = mask_b.shape
    for _ in range(150):
        canvas_h = random.randint(max(MIN_CANVAS, max(h1, h2) + 4), min(MAX_CANVAS, h1 + h2 + 8))
        canvas_w = random.randint(max(MIN_CANVAS, max(w1, w2) + 4), min(MAX_CANVAS, w1 + w2 + 8))
        max_r1, max_c1 = canvas_h - h1, canvas_w - w1
        max_r2, max_c2 = canvas_h - h2, canvas_w - w2
        r1, c1 = random.randint(0, max_r1), random.randint(0, max_c1)
        r2, c2 = random.randint(0, max_r2), random.randint(0, max_c2)
        separated = (
            r1 + h1 + MIN_GAP <= r2
            or r2 + h2 + MIN_GAP <= r1
            or c1 + w1 + MIN_GAP <= c2
            or c2 + w2 + MIN_GAP <= c1
        )
        if not separated:
            continue

        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        patch_a = _colored_patch(mask_a, color_a)
        patch_b = _colored_patch(mask_b, color_b)
        grid[r1 : r1 + h1, c1 : c1 + w1] = patch_a
        grid[r2 : r2 + h2, c2 : c2 + w2] = patch_b
        return grid
    return None


def _sample_dims(stamp_smaller: bool) -> tuple[int, int, int, int] | None:
    for _ in range(200):
        hs = random.randint(MIN_DIM, MAX_DIM)
        ws = random.randint(MIN_DIM, MAX_DIM)
        hl = random.randint(MIN_DIM, MAX_DIM)
        wl = random.randint(MIN_DIM, MAX_DIM)
        if hs * hl > MAX_OUTPUT or ws * wl > MAX_OUTPUT:
            continue
        stamp_area = hs * ws
        layout_area = hl * wl
        if stamp_smaller and stamp_area >= layout_area:
            continue
        if not stamp_smaller and stamp_area <= layout_area:
            continue
        return hs, ws, hl, wl
    return None


def _validate_case(grid: np.ndarray) -> bool:
    try:
        _bg, objects = _extract_objects(grid)
    except Exception:
        return False

    hole_counts = [_hole_count(obj["mask"]) for obj in objects]
    if sorted(hole_counts) != [0, 1]:
        return False

    stamp = objects[0] if hole_counts[0] == 1 else objects[1]
    layout = objects[1] if hole_counts[0] == 1 else objects[0]
    if stamp["mask"].all() or layout["mask"].all():
        return False
    if stamp["color"] == layout["color"]:
        return False

    out = _compose_output(stamp["mask"], stamp["color"], layout["mask"], layout["color"])
    if out.ndim != 2 or max(out.shape) > 30:
        return False

    colors = set(map(int, np.unique(out)))
    if colors != {0, int(stamp["color"]), int(layout["color"])}:
        return False

    swapped = _compose_output(layout["mask"], layout["color"], stamp["mask"], stamp["color"])
    if swapped.shape == out.shape and np.array_equal(swapped, out):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        dims = _sample_dims(stamp_smaller=bool(random.getrandbits(1)))
        if dims is None:
            continue
        hs, ws, hl, wl = dims

        stamp = _make_stamp_mask(hs, ws, large_hole=bool(random.getrandbits(1)))
        if stamp is None:
            continue
        layout = _make_layout_mask(hl, wl, denser=bool(random.getrandbits(1)))
        if layout is None:
            continue

        colors = random.sample(list(range(1, 10)), 2)
        grid = _place_two_objects(stamp, colors[0], layout, colors[1])
        if grid is None:
            continue
        if _validate_case(grid):
            return grid

    raise RuntimeError("failed to generate a valid complement-stamp ARC input")
