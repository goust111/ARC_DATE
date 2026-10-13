import random
from collections import deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator bounds.
PALETTE = tuple(range(10))
GRID_MIN = 16
GRID_MAX = 28
SHAPE_COUNT_RANGE = (1, 3)
SHAPE_H_RANGE = (5, 10)
SHAPE_W_RANGE = (6, 12)
PLACEMENT_BORDER = 1
PLACEMENT_GAP = 1
GEN_RETRIES = 400


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _most_common_color(grid: np.ndarray) -> int:
    counts = _color_counts(grid)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1
    return winners[0]


def _occluder_color(grid: np.ndarray, bg: int) -> int:
    counts = _color_counts(grid)
    non_bg = {color: count for color, count in counts.items() if color != bg}
    assert len(non_bg) >= 2
    best = min(non_bg.values())
    winners = [color for color, count in non_bg.items() if count == best]
    assert len(winners) == 1
    return winners[0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Restore vertically hidden content from each component's top-bottom symmetry."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    occ = _occluder_color(g, bg)
    out = g.copy()

    for coords in _components(g != bg):
        occ_cells = [tuple(map(int, rc)) for rc in coords if int(g[tuple(rc)]) == occ]
        if not occ_cells:
            continue
        r0, c0 = np.min(coords, axis=0)
        r1, c1 = np.max(coords, axis=0)
        assert int(c0) >= 0 and int(c1) < g.shape[1]

        unresolved = set(occ_cells)
        progress = True
        while unresolved and progress:
            progress = False
            for r, c in list(unresolved):
                mr = int(r0 + r1 - r)
                assert 0 <= mr < g.shape[0]
                mirror_value = int(out[mr, c])
                if mirror_value == occ:
                    continue
                out[r, c] = mirror_value
                unresolved.remove((r, c))
                progress = True
        assert not unresolved

    return out


def _shape_mask(height: int, width: int) -> np.ndarray | None:
    half = height // 2
    spec_count = half + (height % 2)
    min_width = max(4, width // 2 + 1)

    for _ in range(120):
        specs: list[tuple[int, int]] = []
        for _idx in range(spec_count):
            span_width = random.randint(min_width, width)
            left = random.randint(0, width - span_width)
            specs.append((left, left + span_width - 1))

        if not any(left == 0 for left, _ in specs):
            continue
        if not any(right == width - 1 for _, right in specs):
            continue
        if not any(not (left == 0 and right == width - 1) for left, right in specs[:half]):
            continue

        mask = np.zeros((height, width), dtype=bool)
        for idx, (left, right) in enumerate(specs[:half]):
            mask[idx, left : right + 1] = True
            mask[height - 1 - idx, left : right + 1] = True
        if height % 2:
            left, right = specs[-1]
            mask[half, left : right + 1] = True

        return mask
    return None


def _side_rows(height: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, height // 2
    return (height + 1) // 2, height


def _choose_occluder(mask: np.ndarray, side: str, require_bg_cover: bool) -> tuple[int, int, int, int] | None:
    height, width = mask.shape
    row_start, row_stop = _side_rows(height, side)
    if row_stop <= row_start:
        return None

    candidates: list[tuple[int, int, int, int]] = []
    for rect_h in range(1, min(3, row_stop - row_start) + 1):
        for rect_w in range(1, min(4, width) + 1):
            area = rect_h * rect_w
            if area < 2 or area > 6:
                continue
            for r0 in range(row_start, row_stop - rect_h + 1):
                for c0 in range(width - rect_w + 1):
                    patch = mask[r0 : r0 + rect_h, c0 : c0 + rect_w]
                    covered = int(np.count_nonzero(patch))
                    if covered == 0:
                        continue
                    if require_bg_cover:
                        if covered == area:
                            continue
                    else:
                        if covered != area:
                            continue
                    candidates.append((r0, c0, rect_h, rect_w))

    if not candidates:
        return None
    return random.choice(candidates)


def _bbox_candidates(blocked: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    out: list[tuple[int, int]] = []
    for r0 in range(PLACEMENT_BORDER, h - PLACEMENT_BORDER - height + 1):
        for c0 in range(PLACEMENT_BORDER, w - PLACEMENT_BORDER - width + 1):
            rr0 = r0 - PLACEMENT_GAP
            cc0 = c0 - PLACEMENT_GAP
            rr1 = r0 + height + PLACEMENT_GAP
            cc1 = c0 + width + PLACEMENT_GAP
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            out.append((r0, c0))
    return out


def _place_shapes(shapes: list[dict], grid_h: int, grid_w: int) -> bool:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    order = sorted(range(len(shapes)), key=lambda idx: int(shapes[idx]["mask"].size), reverse=True)
    for idx in order:
        mask = shapes[idx]["mask"]
        height, width = mask.shape
        candidates = _bbox_candidates(blocked, height, width)
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        shapes[idx]["origin"] = (r0, c0)
        rr0 = r0 - PLACEMENT_GAP
        cc0 = c0 - PLACEMENT_GAP
        rr1 = r0 + height + PLACEMENT_GAP
        cc1 = c0 + width + PLACEMENT_GAP
        blocked[rr0:rr1, cc0:cc1] = True
    return True


def _paint_shape(grid: np.ndarray, mask: np.ndarray, origin: tuple[int, int], color: int) -> None:
    r0, c0 = origin
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = color


def _paint_occluder(grid: np.ndarray, origin: tuple[int, int], rect: tuple[int, int, int, int], color: int) -> int:
    r0, c0 = origin
    rr, cc, rect_h, rect_w = rect
    grid[r0 + rr : r0 + rr + rect_h, c0 + cc : c0 + cc + rect_w] = color
    return rect_h * rect_w


def _gen_attempt() -> np.ndarray | None:
    shape_count = random.randint(*SHAPE_COUNT_RANGE)
    colors = random.sample(PALETTE, shape_count + 2)
    bg = colors[0]
    occ = colors[1]
    shape_colors = colors[2:]

    shapes: list[dict] = []
    bbox_area = 0
    for idx in range(shape_count):
        height = random.randint(*SHAPE_H_RANGE)
        width = random.randint(*SHAPE_W_RANGE)
        mask = _shape_mask(height, width)
        if mask is None:
            return None
        shapes.append(
            {
                "mask": mask,
                "color": shape_colors[idx],
                "side": random.choice(("top", "bottom")),
            }
        )
        bbox_area += height * width

    min_side = max(
        GRID_MIN,
        max(int(shape["mask"].shape[0]) for shape in shapes) + 2 * PLACEMENT_BORDER,
        max(int(shape["mask"].shape[1]) for shape in shapes) + 2 * PLACEMENT_BORDER,
        int(np.ceil(np.sqrt(bbox_area * 2.5))),
    )
    if min_side > GRID_MAX:
        return None
    grid_h = random.randint(min_side, GRID_MAX)
    grid_w = random.randint(min_side, GRID_MAX)
    if not _place_shapes(shapes, grid_h, grid_w):
        return None

    clean = np.full((grid_h, grid_w), bg, dtype=int)
    for shape in shapes:
        _paint_shape(clean, shape["mask"], shape["origin"], int(shape["color"]))

    want_bg_cover = random.random() < 0.5
    bg_cover_indices = set(random.sample(range(shape_count), random.randint(1, shape_count))) if want_bg_cover else set()

    inp = clean.copy()
    total_occ = 0
    for idx, shape in enumerate(shapes):
        require_bg_cover = idx in bg_cover_indices
        rect = _choose_occluder(shape["mask"], str(shape["side"]), require_bg_cover)
        if rect is None:
            return None
        overlap = int(np.count_nonzero(shape["mask"][rect[0] : rect[0] + rect[2], rect[1] : rect[1] + rect[3]]))
        if overlap * 2 >= int(np.count_nonzero(shape["mask"])):
            return None
        total_occ += _paint_occluder(inp, shape["origin"], rect, occ)

    counts = _color_counts(inp)
    if _most_common_color(inp) != bg:
        return None
    if _occluder_color(inp, bg) != occ:
        return None
    if any(counts.get(int(shape["color"]), 0) <= total_occ for shape in shapes):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
