from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
STAMP_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_NAMES = ("tl", "tr", "bl", "br")

# Generator tuning.
PAIR_COUNT_CHOICES = (1, 2, 3)
GRID_SIZE_BY_PAIRS = {
    1: (18, 24, 18, 24),
    2: (22, 30, 24, 32),
    3: (28, 36, 28, 38),
}
CANVAS_PAD = 2
STAMP_PAD = 1
PLACEMENT_ATTEMPTS = 250
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_mask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _missing_corner_name(
    missing: tuple[int, int], bbox: tuple[int, int, int, int]
) -> str | None:
    r0, c0, r1, c1 = bbox
    if missing == (r0, c0):
        return "tl"
    if missing == (r0, c1):
        return "tr"
    if missing == (r1, c0):
        return "bl"
    if missing == (r1, c1):
        return "br"
    return None


def _find_stamps(g: np.ndarray) -> dict[int, np.ndarray]:
    stamps: dict[int, np.ndarray] = {}
    for color in [int(v) for v in np.unique(g) if int(v) not in (BACKGROUND, FRAME_COLOR)]:
        comps = _components(g == int(color))
        large = [comp for comp in comps if len(comp) > 1]
        if not large:
            continue
        assert len(large) == 1, "each stamp color must have exactly one non-singleton component"
        stamps[int(color)] = _crop_mask(large[0])
    return stamps


def _find_canvases(g: np.ndarray) -> list[dict[str, object]]:
    canvases: list[dict[str, object]] = []
    for comp in _components(g == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _border_cells(r0, c0, r1, c1)
        comp_set = set(comp)
        if not comp_set.issubset(border):
            continue
        missing = list(border - comp_set)
        if len(missing) != 1:
            continue
        corner = _missing_corner_name(missing[0], (r0, c0, r1, c1))
        if corner is None or len(comp_set) != len(border) - 1:
            continue

        mr, mc = missing[0]
        color = int(g[mr, mc])
        if color in (BACKGROUND, FRAME_COLOR):
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        anchors = np.argwhere(inner == int(color))
        if len(anchors) != 1:
            continue
        if np.count_nonzero((inner != BACKGROUND) & (inner != int(color))) != 0:
            continue

        ar, ac = map(int, anchors[0])
        canvases.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "color": int(color),
                "corner": corner,
                "anchor": (r0 + 1 + ar, c0 + 1 + ac),
            }
        )

    canvases.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    return canvases


def _aligned_origin(
    anchor: tuple[int, int], stamp_shape: tuple[int, int], corner: str
) -> tuple[int, int]:
    ar, ac = anchor
    sh, sw = stamp_shape
    if corner == "tl":
        return ar, ac
    if corner == "tr":
        return ar, ac - sw + 1
    if corner == "bl":
        return ar - sh + 1, ac
    if corner == "br":
        return ar - sh + 1, ac - sw + 1
    raise AssertionError("unknown corner")


def _axis_starts(start: int, step: int, size: int, lo: int, hi: int) -> list[int]:
    starts: list[int] = []
    pos = start
    if step > 0:
        while pos <= hi:
            if pos + size - 1 >= lo:
                starts.append(pos)
            pos += step
    else:
        while pos + size - 1 >= lo:
            if pos <= hi:
                starts.append(pos)
            pos += step
    return starts


def _tile_stamp(
    out: np.ndarray,
    inner_bbox: tuple[int, int, int, int],
    anchor: tuple[int, int],
    stamp_mask: np.ndarray,
    color: int,
    corner: str,
) -> None:
    r0, c0, r1, c1 = inner_bbox
    sh, sw = stamp_mask.shape
    base_r, base_c = _aligned_origin(anchor, (sh, sw), corner)
    row_step = sh if corner in ("tl", "tr") else -sh
    col_step = sw if corner in ("tl", "bl") else -sw
    row_starts = _axis_starts(base_r, row_step, sh, r0, r1)
    col_starts = _axis_starts(base_c, col_step, sw, c0, c1)
    coords = np.argwhere(stamp_mask)

    for rs in row_starts:
        for cs in col_starts:
            for dr, dc in coords:
                rr = int(rs + dr)
                cc = int(cs + dc)
                if r0 <= rr <= r1 and c0 <= cc <= c1 and int(out[rr, cc]) == BACKGROUND:
                    out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey canvas to the same-colored stamp, then tile that stamp inside
    the canvas from the interior anchor point. The colored corner marker selects
    which stamp bbox corner aligns to the anchor and which two directions the
    tiling expands toward.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    stamps = _find_stamps(g)
    canvases = _find_canvases(g)
    assert canvases, "expected at least one valid canvas"

    for canvas in canvases:
        color = int(canvas["color"])
        assert color in stamps, "every canvas marker color must match a stamp"
        _tile_stamp(
            out,
            canvas["inner_bbox"],  # type: ignore[arg-type]
            canvas["anchor"],  # type: ignore[arg-type]
            stamps[color],
            color,
            str(canvas["corner"]),
        )

    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.tobytes()


def _random_stamp_mask(rng: random.Random) -> np.ndarray:
    for _ in range(120):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        mask = np.ones((h, w), dtype=bool)
        corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        removable = [(r, c) for r in range(h) for c in range(w) if (r, c) not in corners]
        rng.shuffle(removable)

        for r, c in removable:
            if rng.random() >= 0.4:
                continue
            mask[r, c] = False
            if not _is_connected(mask):
                mask[r, c] = True

        if int(mask.sum()) < max(4, (h * w + 1) // 2):
            continue
        if not _is_connected(mask):
            continue
        return mask

    return np.ones((3, 3), dtype=bool)


def _sample_anchor_rel(
    rng: random.Random, inner_h: int, inner_w: int, stamp_h: int, stamp_w: int, corner: str
) -> tuple[int, int]:
    if corner == "tl":
        r_lo, r_hi = 0, inner_h - stamp_h - 1
        c_lo, c_hi = 0, inner_w - stamp_w - 1
    elif corner == "tr":
        r_lo, r_hi = 0, inner_h - stamp_h - 1
        c_lo, c_hi = stamp_w, inner_w - 1
    elif corner == "bl":
        r_lo, r_hi = stamp_h, inner_h - 1
        c_lo, c_hi = 0, inner_w - stamp_w - 1
    else:
        r_lo, r_hi = stamp_h, inner_h - 1
        c_lo, c_hi = stamp_w, inner_w - 1

    assert r_lo <= r_hi and c_lo <= c_hi
    return rng.randint(r_lo, r_hi), rng.randint(c_lo, c_hi)


def _expanded_clear(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _place_bboxes(
    rng: random.Random,
    occupied: np.ndarray,
    sizes: list[tuple[int, int]],
    pad: int,
) -> list[tuple[int, int]] | None:
    hh, ww = occupied.shape
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)

    for idx in order:
        h, w = sizes[idx]
        if hh - h - 1 < 1 or ww - w - 1 < 1:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(1, hh - h - 1)
            c0 = rng.randint(1, ww - w - 1)
            if not _expanded_clear(occupied, r0, c0, h, w, pad):
                continue
            placements[idx] = (r0, c0)
            _mark_bbox(occupied, r0, c0, h, w)
            placed = True
            break
        if not placed:
            return None

    return [pos for pos in placements if pos is not None]  # type: ignore[return-value]


def _draw_stamp(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    for dr, dc in np.argwhere(mask):
        grid[top + int(dr), left + int(dc)] = int(color)


def _draw_canvas(
    grid: np.ndarray,
    color: int,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    corner: str,
    anchor_rel: tuple[int, int],
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r0, c0 = top, left
    r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1

    for c in range(c0, c1 + 1):
        grid[r0, c] = FRAME_COLOR
        grid[r1, c] = FRAME_COLOR
    for r in range(r0, r1 + 1):
        grid[r, c0] = FRAME_COLOR
        grid[r, c1] = FRAME_COLOR

    if corner == "tl":
        marker = (r0, c0)
    elif corner == "tr":
        marker = (r0, c1)
    elif corner == "bl":
        marker = (r1, c0)
    else:
        marker = (r1, c1)
    grid[marker] = int(color)

    ar, ac = anchor_rel
    grid[r0 + 1 + ar, c0 + 1 + ac] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    h_min, h_max, w_min, w_max = GRID_SIZE_BY_PAIRS[pair_count]
    grid_h = rng.randint(h_min, h_max)
    grid_w = rng.randint(w_min, w_max)

    colors = rng.sample(STAMP_COLORS, pair_count)
    corners = rng.sample(list(CORNER_NAMES), pair_count)

    specs: list[dict[str, object]] = []
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for color, corner in zip(colors, corners, strict=False):
        mask = _random_stamp_mask(rng)
        if rng.random() < 0.7:
            for _ in range(40):
                key = _mask_key(mask)
                if key not in used_shapes:
                    break
                mask = _random_stamp_mask(rng)
            used_shapes.add(_mask_key(mask))

        stamp_h, stamp_w = mask.shape
        inner_h_low = max(2 * stamp_h, 5)
        inner_w_low = max(2 * stamp_w, 5)
        inner_h = rng.randint(inner_h_low, inner_h_low + 5)
        inner_w = rng.randint(inner_w_low, inner_w_low + 6)
        anchor_rel = _sample_anchor_rel(rng, inner_h, inner_w, stamp_h, stamp_w, corner)

        specs.append(
            {
                "color": int(color),
                "corner": corner,
                "stamp_mask": mask,
                "stamp_size": (stamp_h, stamp_w),
                "canvas_size": (inner_h + 2, inner_w + 2),
                "inner_size": (inner_h, inner_w),
                "anchor_rel": anchor_rel,
            }
        )

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    canvas_sizes = [spec["canvas_size"] for spec in specs]  # type: ignore[list-item]
    canvas_positions = _place_bboxes(rng, occupied, canvas_sizes, CANVAS_PAD)
    if canvas_positions is None:
        return None

    stamp_sizes = [spec["stamp_size"] for spec in specs]  # type: ignore[list-item]
    stamp_positions = _place_bboxes(rng, occupied, stamp_sizes, STAMP_PAD)
    if stamp_positions is None:
        return None

    for spec, (r0, c0) in zip(specs, canvas_positions, strict=False):
        inner_h, inner_w = spec["inner_size"]  # type: ignore[misc]
        _draw_canvas(
            grid,
            int(spec["color"]),
            r0,
            c0,
            int(inner_h),
            int(inner_w),
            str(spec["corner"]),
            spec["anchor_rel"],  # type: ignore[arg-type]
        )

    for spec, (r0, c0) in zip(specs, stamp_positions, strict=False):
        _draw_stamp(
            grid,
            spec["stamp_mask"],  # type: ignore[arg-type]
            int(spec["color"]),
            r0,
            c0,
        )

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != grid)) < pair_count * 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
