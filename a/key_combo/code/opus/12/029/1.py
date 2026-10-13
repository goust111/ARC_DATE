from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
RECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_ROWS = (18, 24)
GRID_COLS = (18, 24)
RECT_COUNT_RANGE = (3, 5)
RECT_SIDE_RANGE = (4, 7)
COUNTS_POOL = [0, 1, 2, 3, 4]
GEN_RETRIES = 400
PLACE_RETRIES = 300


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_solid_rectangle(comp: dict, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return int(comp["area"]) >= 9 and np.all(patch == color)


def _border_mask(box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _touches_rect_border(small: dict, rect: dict) -> bool:
    r0, c0, r1, c1 = rect["bbox"]
    border = _border_mask(rect["bbox"])
    border_coords = set()
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if border[rr - r0, cc - c0]:
                border_coords.add((rr, cc))
    for rr, cc in small["coords"]:
        for nr, nc in _neighbors4(int(rr), int(cc)):
            if (nr, nc) in border_coords:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve each solid rectangle only when an odd number of adjacent tiny gray bodies vote for it.

    Input structure:
    - The grid has black background, several non-gray solid monochrome rectangles, and scattered gray micro-objects.
    - Each rectangle is a filled axis-aligned block with a one-cell border and a nonempty interior.
    - Each gray object is a 4-connected component of area one or two that touches exactly one rectangle from outside.

    Rule:
    - Find all non-black 4-connected monochrome components.
    - Treat filled components of area at least nine as the rectangles, and gray components of area at most two as the voting micro-objects.
    - For each rectangle, count how many gray micro-objects are 4-neighbor adjacent to its outer border.
    - If that count is odd, replace interior cells whose row-offset plus column-offset from the interior top-left is even with black, preserving the border.
    - If that count is even, leave the rectangle solid.
    - Finally erase all gray micro-objects to black.

    Key insight:
    - The gray objects disappear in the output, so their only visible role is the parity vote that decides whether each rectangle stays solid or becomes a checkerboard carving.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rects = []
    smalls = []
    for comp in _find_components(g):
        if _is_solid_rectangle(comp, g):
            rects.append(comp)
        elif int(comp["color"]) == GREY and int(comp["area"]) <= 2:
            smalls.append(comp)

    for rect in rects:
        touches = sum(1 for small in smalls if _touches_rect_border(small, rect))
        if touches % 2 == 1:
            r0, c0, r1, c1 = rect["bbox"]
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            rows, cols = np.indices(interior.shape)
            interior[(rows + cols) % 2 == 0] = BLACK

    for small in smalls:
        coords = small["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 3) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_rectangles(h: int, w: int, count: int) -> list[dict] | None:
    for _ in range(PLACE_RETRIES):
        sizes = []
        for _idx in range(count):
            rh = random.randint(*RECT_SIDE_RANGE)
            rw = random.randint(*RECT_SIDE_RANGE)
            sizes.append((rh, rw))
        sizes.sort(key=lambda item: item[0] * item[1], reverse=True)

        rects = []
        total_area = 0
        colors = random.sample(RECT_COLORS, count)
        ok = True
        for color, (rh, rw) in zip(colors, sizes):
            positions = []
            for r0 in range(2, h - rh - 1):
                for c0 in range(2, w - rw - 1):
                    box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                    if any(_boxes_too_close(box, other["bbox"]) for other in rects):
                        continue
                    positions.append(box)
            if not positions:
                ok = False
                break
            box = random.choice(positions)
            rects.append({"bbox": box, "color": int(color)})
            total_area += rh * rw
        if ok and total_area <= int(0.5 * h * w):
            random.shuffle(rects)
            return rects
    return None


def _choose_counts(rect_count: int) -> list[int] | None:
    for _ in range(PLACE_RETRIES):
        counts = [random.choice(COUNTS_POOL) for _ in range(rect_count)]
        if sum(counts) == 0:
            continue
        parities = {count % 2 for count in counts}
        if parities != {0, 1}:
            continue
        if max(counts) < 2:
            continue
        return counts
    return None


def _cells_are_free(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False
    return True


def _grey_adjacent_to_existing(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and int(grid[nr, nc]) == GREY:
                return True
    return False


def _component_rect_contacts(cells: list[tuple[int, int]], rects: list[dict]) -> set[int]:
    contacts = set()
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if r0 <= nr <= r1 and c0 <= nc <= c1:
                    contacts.add(idx)
                    break
            if idx in contacts:
                break
    return contacts


def _side_candidates(rect: dict) -> list[tuple[str, int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    candidates = []
    for c in range(c0, c1 + 1):
        candidates.append(("top", r0 - 1, c))
        candidates.append(("bottom", r1 + 1, c))
    for r in range(r0, r1 + 1):
        candidates.append(("left", r, c0 - 1))
        candidates.append(("right", r, c1 + 1))
    random.shuffle(candidates)
    return candidates


def _extend_outward(side: str, r: int, c: int) -> tuple[int, int]:
    if side == "top":
        return r - 1, c
    if side == "bottom":
        return r + 1, c
    if side == "left":
        return r, c - 1
    return r, c + 1


def _place_small_components(grid: np.ndarray, rects: list[dict], rect_idx: int, count: int) -> list[list[tuple[int, int]]] | None:
    placed = []
    if count == 0:
        return placed

    rect = rects[rect_idx]
    preferred_two = count >= 2 and random.random() < 0.7
    two_used = False
    for side, r, c in _side_candidates(rect):
        if len(placed) == count:
            break
        single = [(r, c)]
        if not _cells_are_free(grid, single) or _grey_adjacent_to_existing(grid, single):
            continue

        use_two = preferred_two and not two_used and random.random() < 0.6
        cells = single
        if use_two:
            rr, cc = _extend_outward(side, r, c)
            domino = [(r, c), (rr, cc)]
            if _cells_are_free(grid, domino) and not _grey_adjacent_to_existing(grid, domino):
                cells = domino
                two_used = True

        if _component_rect_contacts(cells, rects) != {rect_idx}:
            continue

        for rr, cc in cells:
            grid[rr, cc] = GREY
        placed.append(cells)

    if len(placed) != count:
        return None
    return placed


def _expected_from_metadata(grid: np.ndarray, rects: list[dict], counts: list[int], smalls: list[list[tuple[int, int]]]) -> np.ndarray:
    out = grid.copy()
    for rect, count in zip(rects, counts):
        if count % 2 == 1:
            r0, c0, r1, c1 = rect["bbox"]
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            rows, cols = np.indices(interior.shape)
            interior[(rows + cols) % 2 == 0] = BLACK
    for cells in smalls:
        for r, c in cells:
            out[r, c] = BLACK
    return out


def _validate_generated_case(
    grid: np.ndarray,
    rects: list[dict],
    counts: list[int],
    smalls: list[list[tuple[int, int]]],
    expected: np.ndarray,
) -> bool:
    comps = _find_components(grid)
    rect_comps = [comp for comp in comps if _is_solid_rectangle(comp, grid)]
    small_comps = [comp for comp in comps if int(comp["color"]) == GREY and int(comp["area"]) <= 2]
    if len(rect_comps) != len(rects) or len(small_comps) != len(smalls):
        return False

    expected_counts = {int(rect["color"]): int(count) for rect, count in zip(rects, counts)}
    detected_counts = {}
    for rect in rect_comps:
        color = int(rect["color"])
        detected_counts[color] = sum(1 for small in small_comps if _touches_rect_border(small, rect))
    if detected_counts != expected_counts:
        return False

    for small in small_comps:
        contacts = sum(1 for rect in rect_comps if _touches_rect_border(small, rect))
        if contacts != 1:
            return False

    if not ({count % 2 for count in counts} == {0, 1}):
        return False
    if np.count_nonzero(grid == BLACK) <= grid.size // 2:
        return False
    return np.array_equal(main(grid), expected)


def _generate_case() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    rect_count = random.randint(*RECT_COUNT_RANGE)
    rects = _sample_rectangles(h, w, rect_count)
    counts = _choose_counts(rect_count)
    if rects is None or counts is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])

    smalls = []
    order = list(range(rect_count))
    random.shuffle(order)
    for idx in order:
        placed = _place_small_components(grid, rects, idx, counts[idx])
        if placed is None:
            return None
        smalls.extend(placed)

    expected = _expected_from_metadata(grid, rects, counts, smalls)
    if not _validate_generated_case(grid, rects, counts, smalls, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
