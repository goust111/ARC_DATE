from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SCALE_CHOICES = (2, 3)
PAIR_COUNT_CHOICES = (1, 2, 3)
GRID_AREA_FACTOR_RANGE = (2.0, 2.8)
GRID_ASPECT_RANGE = (0.75, 1.35)
GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 46
PLACEMENT_PADDING = 1
GENERATION_ATTEMPTS = 400


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
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(coords)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(
    coords: list[tuple[int, int]],
    box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if box is None:
        box = _bbox(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, box


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _touches_crop_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _find_hole_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    bg_comps = _components(crop == BG)
    interior = [comp for comp in bg_comps if not _touches_crop_border(comp, crop.shape)]
    assert len(interior) == 1, "expected exactly one interior hole"
    hole_box_rel = _bbox(interior[0])
    hr0, hc0, hr1, hc1 = hole_box_rel
    hole_mask = crop[hr0 : hr1 + 1, hc0 : hc1 + 1] == BG
    assert bool(np.all(hole_mask)), "hole must be a filled rectangle"
    return r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each small monochrome object into the rectangular hole inside its larger same-color partner.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        comps = _components(g == color)
        assert len(comps) == 2, "each non-background color must form exactly two objects"

        records = []
        for comp in comps:
            mask, box = _mask_from_coords(comp)
            records.append(
                {
                    "coords": comp,
                    "mask": mask,
                    "bbox": box,
                    "area": int(mask.sum()),
                }
            )

        records.sort(key=lambda item: item["area"])
        small, large = records

        hole_box = _find_hole_bbox(g, large["bbox"])
        hole_h = hole_box[2] - hole_box[0] + 1
        hole_w = hole_box[3] - hole_box[1] + 1
        assert small["mask"].shape == (hole_h, hole_w), "hole size must match small bbox size"

        for r, c in small["coords"]:
            out[r, c] = BG

        target = out[hole_box[0] : hole_box[2] + 1, hole_box[1] : hole_box[3] + 1]
        target[small["mask"]] = color
        out[hole_box[0] : hole_box[2] + 1, hole_box[1] : hole_box[3] + 1] = target

    return out


def _edge_coverage(mask: np.ndarray) -> bool:
    return (
        bool(mask[0].any())
        and bool(mask[-1].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            if (
                r == 0
                or c == 0
                or r == h - 1
                or c == w - 1
                or not bool(mask[r - 1, c])
                or not bool(mask[r + 1, c])
                or not bool(mask[r, c - 1])
                or not bool(mask[r, c + 1])
            ):
                cells.append((r, c))
    return cells


def _make_small_mask(scale: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if scale == 2:
        h = random.randint(6, 8)
        w = random.randint(6, 8)
    else:
        h = random.randint(4, 6)
        w = random.randint(4, 6)

    support_h = (h + 2 + scale - 1) // scale
    support_w = (w + 2 + scale - 1) // scale
    assert h >= support_h + 2 and w >= support_w + 2

    support_r = random.randint(1, h - support_h - 1)
    support_c = random.randint(1, w - support_w - 1)

    mask = np.ones((h, w), dtype=bool)
    protected = np.zeros((h, w), dtype=bool)
    protected[support_r : support_r + support_h, support_c : support_c + support_w] = True

    min_area = max(int(np.ceil(0.58 * h * w)), int(protected.sum()) + 4)
    max_remove = random.randint(0, max(1, int(0.35 * h * w)))

    removed = 0
    stalled = 0
    while removed < max_remove and stalled < h * w * 3:
        candidates = _boundary_cells(mask)
        random.shuffle(candidates)
        changed = False
        for r, c in candidates:
            if bool(protected[r, c]):
                continue
            trial = mask.copy()
            trial[r, c] = False
            if int(trial.sum()) < min_area:
                continue
            if not _edge_coverage(trial):
                continue
            if not _is_connected(trial):
                continue
            mask = trial
            removed += 1
            changed = True
            break
        if changed:
            stalled = 0
        else:
            stalled += 1

    return mask, (support_r, support_c, support_h, support_w)


def _make_pair(color: int) -> dict:
    scale = random.choice(SCALE_CHOICES)
    small_mask, support = _make_small_mask(scale)
    h, w = map(int, small_mask.shape)
    support_r, support_c, support_h, support_w = support

    large_mask = np.repeat(np.repeat(small_mask, scale, axis=0), scale, axis=1)
    block_r0 = support_r * scale
    block_c0 = support_c * scale
    block_h = support_h * scale
    block_w = support_w * scale

    hole_r = block_r0 + random.randint(1, block_h - h - 1)
    hole_c = block_c0 + random.randint(1, block_w - w - 1)
    large_mask[hole_r : hole_r + h, hole_c : hole_c + w] = False

    small_patch = np.where(small_mask, color, BG)
    large_patch = np.where(large_mask, color, BG)
    filled_large_patch = large_patch.copy()
    sub = filled_large_patch[hole_r : hole_r + h, hole_c : hole_c + w]
    sub[small_mask] = color
    filled_large_patch[hole_r : hole_r + h, hole_c : hole_c + w] = sub

    return {
        "color": int(color),
        "small_patch": small_patch,
        "small_mask": small_mask,
        "large_patch": large_patch,
        "filled_large_patch": filled_large_patch,
        "hole_bbox": (hole_r, hole_c, hole_r + h - 1, hole_c + w - 1),
    }


def _estimate_grid_shape(records: list[dict]) -> tuple[int, int]:
    total_box_area = 0
    max_h = 0
    max_w = 0
    for rec in records:
        for key in ("large_patch", "small_patch"):
            patch = rec[key]
            ph, pw = map(int, patch.shape)
            total_box_area += ph * pw
            max_h = max(max_h, ph)
            max_w = max(max_w, pw)

    area = int(np.ceil(total_box_area * random.uniform(*GRID_AREA_FACTOR_RANGE)))
    aspect = random.uniform(*GRID_ASPECT_RANGE)
    h = max(GRID_SIDE_MIN, max_h + 4, int(np.ceil(np.sqrt(area / aspect))))
    w = max(GRID_SIDE_MIN, max_w + 4, int(np.ceil(area / h)))
    h = min(h, GRID_SIDE_MAX)
    w = min(w, GRID_SIDE_MAX)
    if h * w < area or h < max_h + 2 or w < max_w + 2:
        h = min(GRID_SIDE_MAX, max(h, max_h + 6))
        w = min(GRID_SIDE_MAX, max(w, max_w + 6))
    return h, w


def _all_positions(
    reserved: np.ndarray,
    patch_shape: tuple[int, int],
    padding: int,
) -> list[tuple[int, int]]:
    h, w = reserved.shape
    ph, pw = patch_shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + ph + padding)
            cc1 = min(w, c0 + pw + padding)
            if not bool(reserved[rr0:rr1, cc0:cc1].any()):
                out.append((r0, c0))
    random.shuffle(out)
    return out


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    target[mask] = patch[mask]
    grid[r0 : r0 + h, c0 : c0 + w] = target


def _reserve(reserved: np.ndarray, patch_shape: tuple[int, int], r0: int, c0: int, padding: int) -> None:
    h, w = reserved.shape
    ph, pw = patch_shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + ph + padding)
    cc1 = min(w, c0 + pw + padding)
    reserved[rr0:rr1, cc0:cc1] = True


def _build_scene(records: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    grid_h, grid_w = _estimate_grid_shape(records)
    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    placements: dict[tuple[int, str], tuple[int, int]] = {}

    items: list[tuple[int, str, np.ndarray]] = []
    for idx, rec in enumerate(records):
        items.append((idx, "large", rec["large_patch"]))
        items.append((idx, "small", rec["small_patch"]))
    items.sort(key=lambda item: int(item[2].shape[0] * item[2].shape[1]), reverse=True)

    for idx, kind, patch in items:
        positions = _all_positions(reserved, patch.shape, PLACEMENT_PADDING)
        if not positions:
            return None
        r0, c0 = positions[0]
        placements[(idx, kind)] = (r0, c0)
        _paste(inp, patch, r0, c0)
        _reserve(reserved, patch.shape, r0, c0, PLACEMENT_PADDING)

    expected[:] = inp
    for idx, rec in enumerate(records):
        sr, sc = placements[(idx, "small")]
        lr, lc = placements[(idx, "large")]
        small_patch = rec["small_patch"]
        filled_large_patch = rec["filled_large_patch"]
        small_mask = rec["small_mask"]
        hole_r0, hole_c0, hole_r1, hole_c1 = rec["hole_bbox"]

        small_slice = expected[sr : sr + small_patch.shape[0], sc : sc + small_patch.shape[1]]
        small_slice[small_mask] = BG
        expected[sr : sr + small_patch.shape[0], sc : sc + small_patch.shape[1]] = small_slice

        hole = expected[lr + hole_r0 : lr + hole_r1 + 1, lc + hole_c0 : lc + hole_c1 + 1]
        source = filled_large_patch[hole_r0 : hole_r1 + 1, hole_c0 : hole_c1 + 1]
        source_mask = source != BG
        hole[source_mask] = source[source_mask]
        expected[lr + hole_r0 : lr + hole_r1 + 1, lc + hole_c0 : lc + hole_c1 + 1] = hole

    return inp, expected


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, pair_count)
    records = [_make_pair(color) for color in colors]
    scene = _build_scene(records)
    if scene is None:
        return None
    inp, expected = scene
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
