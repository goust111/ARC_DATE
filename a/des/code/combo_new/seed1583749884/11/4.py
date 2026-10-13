from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
INPUT_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
MIN_OBJECTS = 2
MAX_OBJECTS = 4
MIN_SHORT = 2
MAX_SHORT = 6
MIN_LONG_EXTRA = 2
MAX_LONG_EXTRA = 6
MAX_LONG = 11
MIN_GRID_SIDE = 16
MAX_GRID_SIDE = 30
MIN_FILL_RATIO = 0.45
MAX_FILL_RATIO = 0.80
OBJECT_ATTEMPTS = 120
SCENE_ATTEMPTS = 400


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _mask_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    height, width = mask.shape
    start = tuple(map(int, points[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, height, width):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_bbox_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _component_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _is_boundary_cell(mask: np.ndarray, r: int, c: int) -> bool:
    if not mask[r, c]:
        return False
    height, width = mask.shape
    for nr, nc in _neighbors4(r, c, height, width):
        if not mask[nr, nc]:
            return True
    return r in (0, height - 1) or c in (0, width - 1)


def _fold_category_flags(mask: np.ndarray, orient: str) -> tuple[bool, bool, bool]:
    height, width = mask.shape
    has_keep_only = False
    has_fold_only = False
    has_overlap = False

    if orient == "wide":
        keep_rows = (height + 1) // 2
        for rr in range(keep_rows):
            src_rr = height - 1 - rr
            fold_available = src_rr >= keep_rows
            for cc in range(width):
                keep = bool(mask[rr, cc])
                fold = bool(mask[src_rr, cc]) if fold_available else False
                has_keep_only |= keep and not fold
                has_fold_only |= fold and not keep
                has_overlap |= keep and fold
    else:
        keep_cols = (width + 1) // 2
        for cc in range(keep_cols):
            src_cc = width - 1 - cc
            fold_available = src_cc >= keep_cols
            for rr in range(height):
                keep = bool(mask[rr, cc])
                fold = bool(mask[rr, src_cc]) if fold_available else False
                has_keep_only |= keep and not fold
                has_fold_only |= fold and not keep
                has_overlap |= keep and fold

    return has_keep_only, has_fold_only, has_overlap


def _build_object_mask(orient: str, short_parity: int | None) -> np.ndarray | None:
    short_choices = [s for s in range(MIN_SHORT, MAX_SHORT + 1) if short_parity is None or s % 2 == short_parity]
    assert short_choices

    for _ in range(OBJECT_ATTEMPTS):
        short_side = random.choice(short_choices)
        long_min = short_side + MIN_LONG_EXTRA
        long_max = min(MAX_LONG, short_side + MAX_LONG_EXTRA)
        if long_min > long_max:
            continue
        long_side = random.randint(long_min, long_max)
        height, width = (short_side, long_side) if orient == "wide" else (long_side, short_side)

        mask = np.ones((height, width), dtype=bool)
        protected_true: set[tuple[int, int]] = set()
        forced_false: set[tuple[int, int]] = set()

        if orient == "wide":
            keep_rows = (height + 1) // 2
            overlap_row = keep_rows - 1 if height % 2 == 0 else keep_rows - 2
            overlap_row = max(0, overlap_row)
            overlap_col = random.randrange(1, width - 1) if width > 2 else 0

            protected_true.add((0, 0))
            protected_true.add((height - 1, width - 1))
            protected_true.add((overlap_row, overlap_col))
            protected_true.add((height - 1 - overlap_row, overlap_col))
            forced_false.add((height - 1, 0))
            forced_false.add((0, width - 1))

            if height % 2 == 1:
                center_row = keep_rows - 1
                protected_true.add((center_row, overlap_col))
        else:
            keep_cols = (width + 1) // 2
            overlap_col = keep_cols - 1 if width % 2 == 0 else keep_cols - 2
            overlap_col = max(0, overlap_col)
            overlap_row = random.randrange(1, height - 1) if height > 2 else 0

            protected_true.add((0, 0))
            protected_true.add((height - 1, width - 1))
            protected_true.add((overlap_row, overlap_col))
            protected_true.add((overlap_row, width - 1 - overlap_col))
            forced_false.add((0, width - 1))
            forced_false.add((height - 1, 0))

            if width % 2 == 1:
                center_col = keep_cols - 1
                protected_true.add((overlap_row, center_col))

        if protected_true & forced_false:
            continue

        for r, c in forced_false:
            mask[r, c] = False

        area = height * width
        min_target = max(len(protected_true) + 2, int(np.ceil(area * MIN_FILL_RATIO)))
        max_target = max(min_target, int(np.floor(area * MAX_FILL_RATIO)))
        target_area = random.randint(min_target, max_target)

        while int(mask.sum()) > target_area:
            candidates = [
                (r, c)
                for r, c in _component_cells(mask)
                if (r, c) not in protected_true and _is_boundary_cell(mask, r, c)
            ]
            random.shuffle(candidates)

            removed = False
            for r, c in candidates:
                trial = mask.copy()
                trial[r, c] = False
                if not _touches_all_bbox_edges(trial):
                    continue
                if not _mask_connected(trial):
                    continue
                mask = trial
                removed = True
                break

            if not removed:
                break

        removed_count = area - int(mask.sum())
        if removed_count < 3:
            continue
        if not _touches_all_bbox_edges(mask):
            continue
        if not _mask_connected(mask):
            continue
        if not all(mask[r, c] for r, c in protected_true):
            continue
        if any(mask[r, c] for r, c in forced_false):
            continue
        if _fold_category_flags(mask, orient) != (True, True, True):
            continue
        return mask

    return None


def _extract_components(grid: np.ndarray) -> list[tuple[int, np.ndarray, tuple[int, int, int, int]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue

            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append((color, arr, (r0, c0, r1, c1)))

    return components


def _paint_folded_component(
    out: np.ndarray,
    color: int,
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    mask = np.zeros((height, width), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    assert height != width, "each object must have a unique long axis"
    if width > height:
        keep_rows = (height + 1) // 2
        for rr in range(keep_rows):
            src_rr = height - 1 - rr
            fold_available = src_rr >= keep_rows
            for cc in range(width):
                keep = bool(mask[rr, cc])
                fold = bool(mask[src_rr, cc]) if fold_available else False
                if not (keep or fold):
                    continue
                out[r0 + rr, c0 + cc] = YELLOW if keep and fold else color if keep else GREY
    else:
        keep_cols = (width + 1) // 2
        for cc in range(keep_cols):
            src_cc = width - 1 - cc
            fold_available = src_cc >= keep_cols
            for rr in range(height):
                keep = bool(mask[rr, cc])
                fold = bool(mask[rr, src_cc]) if fold_available else False
                if not (keep or fold):
                    continue
                out[r0 + rr, c0 + cc] = YELLOW if keep and fold else color if keep else GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each monochrome object onto the upper/left side of its long-axis midline and recolor the compressed result.

    Input structure:
    - The background is Black.
    - The foreground contains several 4-connected monochrome objects, each separated by background.
    - Every object has a non-square bounding box, so its long axis is unambiguous.

    Rule:
    - For each object, use the longer bbox dimension to choose the fold axis: wide objects fold across the horizontal midline, and tall objects fold across the vertical midline.
    - Keep the upper half for wide objects and the left half for tall objects fixed in place, then reflect the lower or right half across that midline onto the kept side.
    - Where kept and reflected pixels both land, paint Yellow; where only the kept pixel exists, keep the object's original color; where only the reflected pixel exists, paint Grey.
    - Clear the folded-away lower or right half back to Black.

    Key insight:
    - The bbox orientation identifies the crease direction, and the folded output uses three colors to expose which parts are symmetric across that long-axis fold and which parts come from only one side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    out = np.full_like(g, BLACK)
    components = _extract_components(g)
    assert components, "expected at least one foreground object"

    for color, coords, bbox in components:
        _paint_folded_component(out, int(color), coords, bbox)

    return out


def _choose_short_side(parity: int | None) -> int:
    choices = [s for s in range(MIN_SHORT, MAX_SHORT + 1) if parity is None or s % 2 == parity]
    return random.choice(choices)


def _expanded_bbox(r0: int, c0: int, height: int, width: int) -> tuple[int, int, int, int]:
    return r0 - 1, c0 - 1, r0 + height, c0 + width


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_objects(
    grid_shape: tuple[int, int],
    masks: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    height, width = grid_shape
    order = sorted(range(len(masks)), key=lambda idx: masks[idx].shape[0] * masks[idx].shape[1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    occupied: list[tuple[int, int, int, int]] = []

    for idx in order:
        obj_h, obj_w = masks[idx].shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(height - obj_h + 1):
            for c0 in range(width - obj_w + 1):
                bbox = _expanded_bbox(r0, c0, obj_h, obj_w)
                if any(_bboxes_overlap(bbox, other) for other in occupied):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        placements[idx] = pos
        occupied.append(_expanded_bbox(pos[0], pos[1], obj_h, obj_w))

    return [tuple(pos) for pos in placements if pos is not None]


def _build_scene() -> np.ndarray | None:
    n_objects = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    first_parity = random.randint(0, 1)
    specs: list[tuple[str, int | None]] = [("wide", first_parity), ("tall", 1 - first_parity)]
    for _ in range(n_objects - 2):
        specs.append((random.choice(("wide", "tall")), random.randint(0, 1)))
    random.shuffle(specs)

    masks: list[np.ndarray] = []
    for orient, parity in specs:
        mask = _build_object_mask(orient, parity)
        if mask is None:
            return None
        masks.append(mask)

    min_height = max(MIN_GRID_SIDE, max(mask.shape[0] for mask in masks) + 4)
    min_width = max(MIN_GRID_SIDE, max(mask.shape[1] for mask in masks) + 4)
    if min_height > MAX_GRID_SIDE or min_width > MAX_GRID_SIDE:
        return None

    grid_height = random.randint(min_height, MAX_GRID_SIDE)
    grid_width = random.randint(min_width, MAX_GRID_SIDE)
    positions = _place_objects((grid_height, grid_width), masks)
    if positions is None:
        return None

    colors = random.sample(INPUT_COLORS, len(masks))
    grid = np.full((grid_height, grid_width), BLACK, dtype=int)
    for mask, color, (r0, c0) in zip(masks, colors, positions, strict=True):
        obj_h, obj_w = mask.shape
        sub = grid[r0 : r0 + obj_h, c0 : c0 + obj_w]
        assert not np.any(sub[mask] != BLACK), "objects must not overlap"
        sub[mask] = int(color)

    out = main(grid)
    if not np.any(out == YELLOW):
        return None
    if not np.any(out == GREY):
        return None
    if not set(map(int, np.unique(out))) - {BLACK, YELLOW, GREY}:
        return None

    components = _extract_components(grid)
    if len(components) != len(masks):
        return None

    has_wide = False
    has_tall = False
    has_even_short = False
    has_odd_short = False
    for _color, _coords, (r0, c0, r1, c1) in components:
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        if obj_w > obj_h:
            has_wide = True
            has_even_short |= obj_h % 2 == 0
            has_odd_short |= obj_h % 2 == 1
        else:
            has_tall = True
            has_even_short |= obj_w % 2 == 0
            has_odd_short |= obj_w % 2 == 1

    if not (has_wide and has_tall and has_even_short and has_odd_short):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding scene")
