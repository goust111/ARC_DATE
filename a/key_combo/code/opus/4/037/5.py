import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
CYAN = 8

GEN_RETRY_LIMIT = 500
LOCAL_RETRY_LIMIT = 120
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
OBJECT_COUNT_OPTIONS = (2, 3)
ISOLATED_COUNT_RANGE = (1, 4)
EMBEDDED_COUNT_RANGE = (1, 3)
MIN_VISIBLE_COPY_CELLS = 4
ORIGINAL_PADDING = 2


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    r = int(r)
    c = int(c)
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None, clip: bool = False) -> np.ndarray:
    h, w = dst.shape
    r0 = int(r0)
    c0 = int(c0)
    ph, pw = patch.shape
    if not clip:
        if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
            raise ValueError("paste out of bounds")
        rr0, cc0, rr1, cc1 = r0, c0, r0 + ph, c0 + pw
        pr0 = pc0 = 0
    else:
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(h, r0 + ph)
        cc1 = min(w, c0 + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - r0
        pc0 = cc0 - c0
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    view = patch[pr0:pr1, pc0:pc1]
    if transparent is None:
        dst[rr0:rr1, cc0:cc1] = view
    else:
        mask = view != int(transparent)
        dst_view = dst[rr0:rr1, cc0:cc1]
        dst_view[mask] = view[mask]
    return dst


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(_components_from_mask(mask)) == 1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[0:0, 0:0].copy()
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _count_green_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    return sum(1 for nr, nc in neighbors4(r, c) if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREEN)


def _classify_component_reds(grid: np.ndarray, coords: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    embedded = []
    other = []
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if int(grid[rr, cc]) != RED:
            continue
        if _count_green_neighbors(grid, rr, cc) >= 2:
            embedded.append((rr, cc))
        else:
            other.append((rr, cc))
    return embedded, other


def _interior_mask_for_component(bbox: tuple[int, int, int, int], coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    wall = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        wall[int(r) - r0, int(c) - c0] = True

    h, w = wall.shape
    exterior = np.zeros_like(wall, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not wall[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return ~wall & ~exterior


def _copy_layout(grid_shape: tuple[int, int], bbox: tuple[int, int, int, int], patch_shape: tuple[int, int], isolated_count: int) -> tuple[str, int, int, int]:
    h, w = map(int, grid_shape)
    r0, c0, r1, c1 = map(int, bbox)
    ph, pw = map(int, patch_shape)
    if isolated_count % 2 == 1:
        direction = "horizontal"
        color = BLUE
        if c0 + c1 <= w - 1:
            top = r0
            left = c1 + 2
        else:
            top = r0
            left = c0 - pw - 1
    else:
        direction = "vertical"
        color = CYAN
        if r0 + r1 <= h - 1:
            top = r1 + 2
            left = c0
        else:
            top = r0 - ph - 1
            left = c0
    return direction, color, int(top), int(left)


def _visible_cells(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for r, c in np.argwhere(mask):
        rr = int(r) + int(top)
        cc = int(c) + int(left)
        if 0 <= rr < h and 0 <= cc < w:
            cells.add((rr, cc))
    return cells


def _analyze_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    components = _components_from_mask(g != BG)
    objects = []
    for coords in components:
        green_coords = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == GREEN]
        if not green_coords:
            continue
        embedded, _attached_or_internal_reds = _classify_component_reds(g, coords)
        wall_coords = np.asarray(green_coords + embedded, dtype=int)
        bbox = bbox_coords(wall_coords)
        interior = _interior_mask_for_component(bbox, wall_coords)
        r0, c0, r1, c1 = bbox
        isolated = []
        for rr, cc in np.argwhere(interior):
            ar = int(rr) + r0
            ac = int(cc) + c0
            if int(g[ar, ac]) == RED:
                isolated.append((ar, ac))
        repaired_patch = crop_bbox(g, bbox)
        for r, c in embedded:
            repaired_patch[int(r) - r0, int(c) - c0] = GREEN
        objects.append(
            {
                "bbox": bbox,
                "coords": wall_coords,
                "embedded": sorted(embedded),
                "isolated": sorted(isolated),
                "interior_mask": interior,
                "repaired_patch": repaired_patch,
            }
        )
    assert objects, "expected at least one contour object"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair embedded red defects on each green outline, then mirror the repaired object to the opposite side using isolated-marker parity.

    Input structure:
    - A black background contains 2-3 one-pixel-thick green outline objects, each enclosing a hole.
    - Some red pixels replace contour pixels; these embedded markers have at least two green 4-neighbors.
    - Additional red pixels lie inside the enclosed holes as isolated markers, with at most one green 4-neighbor.

    Rule:
    - For each outline object, classify red pixels on the contour as embedded and repaint them green.
    - Count the red isolated markers inside that object's enclosed interior.
    - If the count is odd, mirror the repaired object horizontally and place the copy one column away on the bbox side opposite the grid center, coloring the whole copy blue.
    - If the count is even, mirror the repaired object vertically and place the copy one row away on the bbox side opposite the grid center, coloring the whole copy cyan; clip any off-grid part.

    Key insight:
    - The red pixels split into two roles: embedded ones only repair the original contour, while the parity of the interior isolated ones chooses the mirror axis and therefore the copy direction and color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    for obj in _analyze_grid(g):
        r0, c0, r1, c1 = obj["bbox"]
        for r, c in obj["embedded"]:
            out[int(r), int(c)] = GREEN

        repaired_patch = obj["repaired_patch"]
        isolated_count = len(obj["isolated"])
        direction, copy_color, top, left = _copy_layout(g.shape, obj["bbox"], repaired_patch.shape, isolated_count)
        mirrored = np.fliplr(repaired_patch) if direction == "horizontal" else np.flipud(repaired_patch)
        stamp = np.where(mirrored != BG, copy_color, BG).astype(int)
        paste(out, stamp, top, left, transparent=BG, clip=True)

    return out


def _ring_neighbor_count(mask: np.ndarray, r: int, c: int) -> int:
    return sum(1 for nr, nc in neighbors4(r, c) if in_bounds(mask.shape, nr, nc) and mask[nr, nc])


def _sample_non_adjacent(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    candidates = candidates[:]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_hole_mask() -> np.ndarray | None:
    hole = np.ones((random.randint(3, 5), random.randint(3, 5)), dtype=bool)
    removal_budget = random.randint(0, 3)
    for _ in range(removal_budget):
        boundary = [(r, c) for r, c in np.argwhere(hole) if r in (0, hole.shape[0] - 1) or c in (0, hole.shape[1] - 1)]
        random.shuffle(boundary)
        changed = False
        for r, c in boundary:
            candidate = hole.copy()
            candidate[int(r), int(c)] = False
            cropped = _crop_mask(candidate)
            if cropped.size == 0 or min(cropped.shape) < 3:
                continue
            if not _connected(cropped):
                continue
            hole = cropped
            changed = True
            break
        if not changed:
            break
    return hole


def _build_object_spec(required_parity: int) -> dict | None:
    for _ in range(LOCAL_RETRY_LIMIT):
        hole = _build_hole_mask()
        if hole is None:
            continue
        hole_big = np.zeros((hole.shape[0] + 2, hole.shape[1] + 2), dtype=bool)
        hole_big[1:-1, 1:-1] = hole
        ring = _dilate8(hole_big) & ~hole_big
        if len(_components_from_mask(ring)) != 1:
            continue

        embedded_candidates = [
            (int(r), int(c))
            for r, c in np.argwhere(ring)
            if _ring_neighbor_count(ring, int(r), int(c)) >= 2
        ]
        isolated_candidates = [
            (int(r), int(c))
            for r, c in np.argwhere(hole_big)
            if _ring_neighbor_count(ring, int(r), int(c)) <= 1
        ]
        if len(embedded_candidates) < EMBEDDED_COUNT_RANGE[0] or len(isolated_candidates) < max(ISOLATED_COUNT_RANGE):
            continue

        possible_isolated = [n for n in range(ISOLATED_COUNT_RANGE[0], ISOLATED_COUNT_RANGE[1] + 1) if n % 2 == required_parity]
        random.shuffle(possible_isolated)
        isolated = None
        isolated_count = None
        for count in possible_isolated:
            sample = _sample_non_adjacent(isolated_candidates, count)
            if sample is not None:
                isolated = sample
                isolated_count = count
                break
        if isolated is None:
            continue

        embedded_options = list(range(EMBEDDED_COUNT_RANGE[0], min(EMBEDDED_COUNT_RANGE[1], len(embedded_candidates)) + 1))
        random.shuffle(embedded_options)
        embedded = None
        for count in embedded_options:
            sample = _sample_non_adjacent(embedded_candidates, count)
            if sample is not None:
                embedded = sample
                break
        if embedded is None:
            continue

        patch = np.full(ring.shape, BG, dtype=int)
        patch[ring] = GREEN
        for r, c in embedded:
            patch[int(r), int(c)] = RED
        for r, c in isolated:
            patch[int(r), int(c)] = RED

        repaired = np.full(ring.shape, BG, dtype=int)
        repaired[ring] = GREEN
        for r, c in isolated:
            repaired[int(r), int(c)] = RED

        return {
            "patch": patch,
            "repaired_patch": repaired,
            "ring_mask": ring,
            "embedded": embedded,
            "isolated": isolated,
            "isolated_count": int(isolated_count),
            "embedded_count": len(embedded),
        }
    return None


def _pad_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    if radius <= 0:
        return set(cells)
    h, w = shape
    out = set()
    for r, c in cells:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(patch != BG):
        rr = int(r) + int(top)
        cc = int(c) + int(left)
        grid[rr, cc] = int(patch[int(r), int(c)])
        cells.add((rr, cc))
    return cells


def _place_objects(specs: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.full(shape, BG, dtype=int)
    occupied_output: set[tuple[int, int]] = set()
    occupied_original: set[tuple[int, int]] = set()
    occupied_bboxes: set[tuple[int, int]] = set()
    padded_original: set[tuple[int, int]] = set()

    order = specs[:]
    random.shuffle(order)
    for spec in order:
        patch = spec["patch"]
        ph, pw = patch.shape
        candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            original_cells = {(top + int(r), left + int(c)) for r, c in np.argwhere(patch != BG)}
            bbox_cells = {(rr, cc) for rr in range(top, top + ph) for cc in range(left, left + pw)}
            if original_cells & padded_original:
                continue
            if bbox_cells & occupied_output:
                continue
            bbox = (top, left, top + ph - 1, left + pw - 1)
            direction, copy_color, copy_top, copy_left = _copy_layout(shape, bbox, spec["repaired_patch"].shape, spec["isolated_count"])
            mirrored_mask = np.fliplr(spec["repaired_patch"] != BG) if direction == "horizontal" else np.flipud(spec["repaired_patch"] != BG)
            copy_cells = _visible_cells(mirrored_mask, copy_top, copy_left, shape)
            if len(copy_cells) < MIN_VISIBLE_COPY_CELLS:
                continue
            if original_cells & occupied_output:
                continue
            if copy_cells & occupied_output:
                continue
            if copy_cells & occupied_bboxes:
                continue
            spec["bbox"] = bbox
            spec["embedded"] = sorted((top + int(r), left + int(c)) for r, c in spec["embedded"])
            spec["isolated"] = sorted((top + int(r), left + int(c)) for r, c in spec["isolated"])
            spec["copy_direction"] = direction
            spec["copy_color"] = copy_color
            spec["copy_top_left"] = (copy_top, copy_left)
            spec["copy_visible_cells"] = copy_cells
            _stamp_patch(grid, patch, top, left)
            occupied_original |= original_cells
            occupied_bboxes |= bbox_cells
            padded_original |= _pad_cells(original_cells, shape, ORIGINAL_PADDING)
            occupied_output |= original_cells | copy_cells
            placed = True
            break
        if not placed:
            return None
    return grid


def _semantic_signature(obj: dict) -> tuple:
    return (
        obj["bbox"],
        tuple(obj["embedded"]),
        tuple(obj["isolated"]),
        len(obj["isolated"]),
        obj["repaired_patch"].shape,
    )


def _validate_generated(grid: np.ndarray, specs: list[dict]) -> bool:
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2 and max(grid.shape) <= 30):
        return False
    if set(map(int, np.unique(grid))) - {BG, RED, GREEN}:
        return False

    analyzed = _analyze_grid(grid)
    if len(analyzed) != len(specs):
        return False

    analyzed_map = {obj["bbox"]: obj for obj in analyzed}
    if set(analyzed_map) != {tuple(spec["bbox"]) for spec in specs}:
        return False

    isolated_parities = set()
    embedded_parity_flip = False
    for spec in specs:
        obj = analyzed_map[tuple(spec["bbox"])]
        if tuple(obj["embedded"]) != tuple(spec["embedded"]):
            return False
        if tuple(obj["isolated"]) != tuple(spec["isolated"]):
            return False
        if len(obj["isolated"]) != spec["isolated_count"]:
            return False
        if len(obj["embedded"]) != spec["embedded_count"]:
            return False
        if any(_count_green_neighbors(grid, r, c) < 2 for r, c in obj["embedded"]):
            return False
        if any(_count_green_neighbors(grid, r, c) > 1 for r, c in obj["isolated"]):
            return False
        isolated_parities.add(len(obj["isolated"]) % 2)
        embedded_parity_flip |= spec["embedded_count"] % 2 == 1

    if isolated_parities != {0, 1}:
        return False
    if not embedded_parity_flip:
        return False

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.ndim != 2 or out.shape != grid.shape or max(out.shape) > 30:
        return False
    if BLUE not in out or CYAN not in out:
        return False

    for spec in specs:
        bbox = tuple(spec["bbox"])
        obj = analyzed_map[bbox]
        r0, c0, r1, c1 = bbox
        repaired = obj["repaired_patch"]
        if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], repaired):
            return False
        direction, copy_color, top, left = _copy_layout(grid.shape, bbox, repaired.shape, len(obj["isolated"]))
        if direction != spec["copy_direction"] or copy_color != spec["copy_color"]:
            return False
        mirrored = np.fliplr(repaired) if direction == "horizontal" else np.flipud(repaired)
        expected = np.where(mirrored != BG, copy_color, BG)
        visible = _visible_cells(expected != BG, top, left, grid.shape)
        if visible != spec["copy_visible_cells"]:
            return False
        for rr, cc in visible:
            if int(out[rr, cc]) != copy_color:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        object_count = random.choice(OBJECT_COUNT_OPTIONS)
        parities = [0, 1]
        if object_count == 3:
            parities.append(random.randint(0, 1))
        random.shuffle(parities)

        specs = []
        for parity in parities:
            spec = _build_object_spec(parity)
            if spec is None:
                specs = []
                break
            specs.append(spec)
        if not specs:
            continue
        if not any(spec["embedded_count"] % 2 == 1 for spec in specs):
            continue

        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        grid = _place_objects(specs, (h, w))
        if grid is None:
            continue
        if _validate_generated(grid, specs):
            return grid
    raise RuntimeError("failed to generate valid input")
