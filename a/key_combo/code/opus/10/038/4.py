from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
EDGE = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
EDGE_SIDES = ("top", "bottom", "left", "right")
OPPOSITE_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
_GEN_COUNTER = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _cells_to_patch(cells: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(cells)
    _r0, _c0, r1, c1 = _bbox(norm)
    patch = np.zeros((r1 + 1, c1 + 1), dtype=int)
    for r, c in norm:
        patch[r, c] = BLUE
    return patch


def _is_connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _cell_degree(cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))


def _candidate_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [cell for cell in cells if _cell_degree(cells, cell) >= 3]


def _patch_defect_candidates(patch: np.ndarray) -> list[tuple[int, int]]:
    cells = {(int(r), int(c)) for r, c in np.argwhere(patch == BLUE)}
    return sorted(_candidate_cells(cells))


def _patch_signature(patch: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, patch.shape)), np.ascontiguousarray(patch).tobytes()


def _detect_edge_side(grid: np.ndarray) -> str:
    candidates = []
    if np.all(grid[0, :] == EDGE):
        candidates.append("top")
    if np.all(grid[-1, :] == EDGE):
        candidates.append("bottom")
    if np.all(grid[:, 0] == EDGE):
        candidates.append("left")
    if np.all(grid[:, -1] == EDGE):
        candidates.append("right")
    if len(candidates) != 1:
        raise ValueError("expected exactly one full gray edge line")
    return candidates[0]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    mask = (grid == BLUE) | (grid == RED)
    components = _connected_components(mask)
    if not components:
        raise ValueError("expected at least one blue/red object")

    objects: list[dict] = []
    for idx, cells in enumerate(components):
        r0, c0, r1, c1 = _bbox(cells)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        defect_count = 0
        for r, c in cells:
            patch[r - r0, c - c0] = BLUE
            if int(grid[r, c]) == RED:
                defect_count += 1
        objects.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
                "top": r0,
                "left": c0,
                "height": patch.shape[0],
                "width": patch.shape[1],
                "area": int(np.count_nonzero(patch)),
                "defects": defect_count,
                "patch": patch,
            }
        )
    return objects


def _render_sorted_patches(patches: list[np.ndarray], side: str) -> np.ndarray:
    if not patches:
        raise ValueError("expected at least one patch")

    if side in {"top", "bottom"}:
        out_h = max(int(p.shape[0]) for p in patches)
        out_w = sum(int(p.shape[1]) for p in patches) + max(0, len(patches) - 1)
        out = np.zeros((out_h, out_w), dtype=int)
        cursor = 0
        for patch in patches:
            h, w = map(int, patch.shape)
            row = 0 if side == "top" else out_h - h
            out[row : row + h, cursor : cursor + w][patch == BLUE] = BLUE
            cursor += w + 1
        return out

    out_h = sum(int(p.shape[0]) for p in patches) + max(0, len(patches) - 1)
    out_w = max(int(p.shape[1]) for p in patches)
    out = np.zeros((out_h, out_w), dtype=int)
    cursor = 0
    for patch in patches:
        h, w = map(int, patch.shape)
        col = 0 if side == "left" else out_w - w
        out[cursor : cursor + h, col : col + w][patch == BLUE] = BLUE
        cursor += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the red defects inside each object, crop the repaired shapes, then arrange them by pre-repair defect count.

    Input structure:
    - A black background contains 3 or 4 separated blue objects whose defect pixels are recolored red.
    - Each object is one 4-connected blue/red component, and each red pixel is embedded inside its object.
    - Exactly one full grid border is a gray line, which indicates both the output axis and the alignment side.

    Rule:
    - Treat every 4-connected blue/red component as one object and count its red pixels before changing anything.
    - Recolor each red pixel to blue, crop every repaired object to its tight bounding box, and sort the cropped objects by increasing red-count.
    - If the gray line is on the top or bottom edge, place the sorted crops left-to-right with one black column between them and align them to the top or bottom respectively.
    - If the gray line is on the left or right edge, place the sorted crops top-to-bottom with one black row between them and align them to the left or right respectively.
    - The output is only this packed arrangement; the gray edge line is removed.

    Key insight:
    - The sorting key disappears after repair, so the red defect counts must be captured before recoloring, and the border line separately encodes both packing axis and alignment side.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    side = _detect_edge_side(g)
    objects = _extract_objects(g)
    ordered = sorted(objects, key=lambda obj: (obj["defects"], obj["top"], obj["left"]))
    return _render_sorted_patches([obj["patch"] for obj in ordered], side)


def _make_repaired_shape(defect_count: int) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    for _ in range(120):
        target_area = random.randint(max(6, defect_count + 4), 12)
        rect_choices = [
            (h, w)
            for h in range(2, 5)
            for w in range(2, 5)
            if h * w >= target_area and h * w <= min(16, target_area + 4)
        ]
        random.shuffle(rect_choices)
        for h, w in rect_choices:
            cells = {(r, c) for r in range(h) for c in range(w)}
            while len(cells) > target_area:
                removable = []
                for cell in cells:
                    trial = set(cells)
                    trial.remove(cell)
                    if not _is_connected_cells(trial):
                        continue
                    tr0, tc0, tr1, tc1 = _bbox(trial)
                    if tr1 - tr0 + 1 < 2 or tc1 - tc0 + 1 < 2:
                        continue
                    if len(_candidate_cells(trial)) < defect_count:
                        continue
                    removable.append(cell)
                if not removable:
                    break
                weights = [4 if _cell_degree(cells, cell) <= 2 else 1 for cell in removable]
                chosen = random.choices(removable, weights=weights, k=1)[0]
                cells.remove(chosen)

            if len(cells) != target_area:
                continue

            patch = _cells_to_patch(cells)
            patch = np.rot90(patch, random.randint(0, 3))
            if random.choice((True, False)):
                patch = np.fliplr(patch)
            if random.choice((True, False)):
                patch = np.flipud(patch)
            patch = np.ascontiguousarray(patch)

            candidates = _patch_defect_candidates(patch)
            if len(candidates) < defect_count:
                continue
            defect_sites = set(random.sample(candidates, defect_count))
            return patch, defect_sites
    return None


def _paint_object(grid: np.ndarray, patch: np.ndarray, defect_sites: set[tuple[int, int]], top: int, left: int) -> None:
    for r, c in np.argwhere(patch == BLUE):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = RED if (int(r), int(c)) in defect_sites else BLUE


def _place_objects(
    specs: list[dict],
    side: str,
    height: int,
    width: int,
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    if side == "top":
        grid[0, :] = EDGE
    elif side == "bottom":
        grid[-1, :] = EDGE
    elif side == "left":
        grid[:, 0] = EDGE
    else:
        grid[:, -1] = EDGE

    top_margin = 2 if side == "top" else 1
    bottom_margin = 2 if side == "bottom" else 1
    left_margin = 2 if side == "left" else 1
    right_margin = 2 if side == "right" else 1

    order = list(specs)
    random.shuffle(order)
    placed: list[dict] = []

    for spec in order:
        patch = spec["patch"]
        h, w = map(int, patch.shape)
        r_lo = top_margin
        r_hi = height - bottom_margin - h
        c_lo = left_margin
        c_hi = width - right_margin - w
        if r_lo > r_hi or c_lo > c_hi:
            return None

        success = False
        for _ in range(400):
            top = random.randint(r_lo, r_hi)
            left = random.randint(c_lo, c_hi)
            br0 = max(0, top - 2)
            bc0 = max(0, left - 2)
            br1 = min(height, top + h + 2)
            bc1 = min(width, left + w + 2)
            if blocked[br0:br1, bc0:bc1].any():
                continue
            _paint_object(grid, patch, spec["defect_sites"], top, left)
            blocked[br0:br1, bc0:bc1] = True
            placed_spec = dict(spec)
            placed_spec["top"] = top
            placed_spec["left"] = left
            placed_spec["bbox"] = (top, left, top + h - 1, left + w - 1)
            placed.append(placed_spec)
            success = True
            break
        if not success:
            return None

    return grid, placed


def _intended_output(specs: list[dict], side: str) -> np.ndarray:
    ordered = sorted(specs, key=lambda spec: spec["defects"])
    return _render_sorted_patches([spec["patch"] for spec in ordered], side)


def _validate_red_embeddings(grid: np.ndarray) -> bool:
    for r, c in np.argwhere(grid == RED):
        support = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in {BLUE, RED}:
                support += 1
        if support < 3:
            return False
    return True


def _object_order(specs: list[dict], side: str) -> list[int]:
    if side in {"top", "bottom"}:
        return [spec["id"] for spec in sorted(specs, key=lambda spec: (spec["left"], spec["top"]))]
    return [spec["id"] for spec in sorted(specs, key=lambda spec: (spec["top"], spec["left"]))]


def _alt_area_order(specs: list[dict]) -> list[int]:
    return [spec["id"] for spec in sorted(specs, key=lambda spec: (spec["area"], spec["height"], spec["width"], spec["id"]))]


def _gen_attempt(preferred_side: str, preferred_count: int) -> np.ndarray | None:
    side = preferred_side
    object_count = preferred_count
    defect_counts = [1, 2, 3, 4]
    if object_count == 3:
        defect_counts = random.sample(defect_counts, 3)
    random.shuffle(defect_counts)

    specs: list[dict] = []
    used_signatures: set[tuple[tuple[int, int], bytes]] = set()
    for obj_id, defects in enumerate(defect_counts):
        shape = _make_repaired_shape(defects)
        if shape is None:
            return None
        patch, defect_sites = shape
        sig = _patch_signature(patch)
        if sig in used_signatures:
            return None
        used_signatures.add(sig)
        specs.append(
            {
                "id": obj_id,
                "defects": defects,
                "patch": patch,
                "defect_sites": defect_sites,
                "height": int(patch.shape[0]),
                "width": int(patch.shape[1]),
                "area": int(np.count_nonzero(patch)),
            }
        )

    heights = [spec["height"] for spec in specs]
    widths = [spec["width"] for spec in specs]
    if side in {"top", "bottom"} and len(set(heights)) < 2:
        return None
    if side in {"left", "right"} and len(set(widths)) < 2:
        return None

    defect_order = [spec["id"] for spec in sorted(specs, key=lambda spec: spec["defects"])]
    if defect_order == _alt_area_order(specs):
        return None

    grid_h = random.randint(18, 24)
    grid_w = random.randint(15, 22)
    placed = _place_objects(specs, side, grid_h, grid_w)
    if placed is None:
        return None
    inp, placed_specs = placed

    if not _validate_red_embeddings(inp):
        return None

    spatial_order = _object_order(placed_specs, side)
    if defect_order == spatial_order:
        return None

    expected = _intended_output(placed_specs, side)
    opposite = _intended_output(placed_specs, OPPOSITE_SIDE[side])
    if np.array_equal(expected, opposite):
        return None
    if max(expected.shape) > 30:
        return None

    extracted = _extract_objects(inp)
    if sorted(obj["defects"] for obj in extracted) != sorted(spec["defects"] for spec in placed_specs):
        return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    preferred_side = EDGE_SIDES[_GEN_COUNTER % len(EDGE_SIDES)]
    preferred_count = 3 if ((_GEN_COUNTER // len(EDGE_SIDES)) % 2 == 0) else 4
    _GEN_COUNTER += 1

    for _ in range(500):
        grid = _gen_attempt(preferred_side, preferred_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid defect-sort input")
