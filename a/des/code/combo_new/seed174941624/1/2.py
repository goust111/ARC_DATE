from __future__ import annotations

import random
from collections import Counter

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- shared helpers ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_fragments(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    fragments: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords_abs: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords_abs.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords_abs]
            cols = [cc for _, cc in coords_abs]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords_abs:
                mask[rr - r0, cc - c0] = True

            coords_rel = tuple(sorted((rr - r0, cc - c0) for rr, cc in coords_abs))
            fragments.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "coords": coords_rel,
                    "h": int(mask.shape[0]),
                    "w": int(mask.shape[1]),
                    "area": len(coords_rel),
                }
            )

    fragments.sort(key=lambda frag: (frag["bbox"][1], frag["bbox"][0], frag["color"]))
    return fragments


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _canonical_patch_key(patch: np.ndarray) -> bytes:
    relabeled = np.zeros(patch.shape, dtype=np.uint8)
    mapping: dict[int, int] = {}
    next_label = 1
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            value = int(patch[r, c])
            if value not in mapping:
                mapping[value] = next_label
                next_label += 1
            relabeled[r, c] = mapping[value]
    return relabeled.tobytes()


def _rectangle_order_key(patch: np.ndarray) -> tuple[int, int, int, bytes]:
    h, w = map(int, patch.shape)
    return (h * w, h, w, _canonical_patch_key(patch))


def _merge_options(a: dict, b: dict) -> list[np.ndarray]:
    area_total = int(a["area"]) + int(b["area"])
    options: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for rect_h in range(1, area_total + 1):
        if area_total % rect_h != 0:
            continue
        rect_w = area_total // rect_h
        if rect_h < max(int(a["h"]), int(b["h"])) or rect_w < max(int(a["w"]), int(b["w"])):
            continue

        for r0 in range(rect_h - int(a["h"]) + 1):
            for c0 in range(rect_w - int(a["w"]) + 1):
                placed = np.zeros((rect_h, rect_w), dtype=bool)
                for rr, cc in a["coords"]:
                    placed[r0 + rr, c0 + cc] = True

                complement = ~placed
                sub = _trim_mask(complement)
                if sub.shape != b["mask"].shape or not np.array_equal(sub, b["mask"]):
                    continue

                patch = np.full((rect_h, rect_w), int(b["color"]), dtype=int)
                patch[placed] = int(a["color"])
                options[(tuple(patch.shape), patch.tobytes())] = patch

    merged = list(options.values())
    merged.sort(key=_rectangle_order_key)
    return merged


def _enumerate_pairings(fragments: list[dict], limit: int | None = None) -> list[list[np.ndarray]]:
    n = len(fragments)
    pair_options: dict[tuple[int, int], list[np.ndarray]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            options = _merge_options(fragments[i], fragments[j])
            if options:
                pair_options[(i, j)] = options

    solutions: list[list[np.ndarray]] = []

    def search(remaining: tuple[int, ...], chosen: list[np.ndarray]) -> bool:
        if not remaining:
            solutions.append([patch.copy() for patch in chosen])
            return limit is not None and len(solutions) >= limit

        first = remaining[0]
        for pos in range(1, len(remaining)):
            second = remaining[pos]
            options = pair_options.get((first, second))
            if not options:
                continue

            nxt = remaining[1:pos] + remaining[pos + 1 :]
            for patch in options:
                chosen.append(patch)
                if search(nxt, chosen):
                    return True
                chosen.pop()
        return False

    search(tuple(range(n)), [])
    return solutions


def _stack_rectangles(rectangles: list[np.ndarray], bg: int) -> np.ndarray:
    rectangles = sorted(rectangles, key=_rectangle_order_key)
    out_h = sum(int(rect.shape[0]) for rect in rectangles) + max(0, len(rectangles) - 1)
    out_w = max(int(rect.shape[1]) for rect in rectangles)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    row = 0
    for rect in rectangles:
        h, w = map(int, rect.shape)
        out[row : row + h, :w] = rect
        row += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered fragments by geometric complementarity, rebuild each full rectangle, then stack the rectangles by area.

    Input structure:
    - A black or otherwise dominant background surrounds an even number of monochrome 4-connected fragments.
    - Each foreground fragment has its own color and appears as an irregular single component.
    - The fragments come in complementary pairs: after translation only, each pair fills one solid axis-aligned rectangle.

    Rule:
    - Extract every fragment and test every pair geometrically.
    - Two fragments belong together exactly when some translation places them inside one rectangle with no overlap and no gap.
    - Rebuild each paired rectangle while preserving the source colors of the two fragments.
    - Sort the rebuilt rectangles by area from small to large, then stack them top-to-bottom with one background row between them and left-align every rectangle.

    Key insight:
    - Colors do not determine the pairing; the solver must first find which two shapes are exact complements of one rectangle, and only then sort the resulting rectangles by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    fragments = _extract_fragments(g, bg)
    assert 4 <= len(fragments) <= 8 and len(fragments) % 2 == 0

    solutions = _enumerate_pairings(fragments, limit=1)
    assert solutions, "no complete complementary pairing found"
    return _stack_rectangles(solutions[0], bg)


# --------------- generator ---------------

_RNG = random.Random()

ARC_MAX = 30
BG = 0
PAIR_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (3, 7)
GRID_SIDE_RANGE = (15, 30)
MAX_ATTEMPTS = 400


def _random_frontier(length: int, upper_exclusive: int) -> list[int] | None:
    if length < 2 or upper_exclusive < 2:
        return None
    values = [_RNG.randint(1, upper_exclusive - 1) for _ in range(length)]
    if len(set(values)) == 1:
        idx = _RNG.randrange(length)
        alternatives = [value for value in range(1, upper_exclusive) if value != values[idx]]
        if not alternatives:
            return None
        values[idx] = _RNG.choice(alternatives)
    return values


def _split_rectangle(rect_h: int, rect_w: int, orientation: str) -> tuple[np.ndarray, np.ndarray] | None:
    if orientation == "vertical":
        frontier = _random_frontier(rect_h, rect_w)
        if frontier is None:
            return None
        first = np.zeros((rect_h, rect_w), dtype=bool)
        second = np.zeros((rect_h, rect_w), dtype=bool)
        for r, cut in enumerate(frontier):
            first[r, :cut] = True
            second[r, cut:] = True
        return first, second

    frontier = _random_frontier(rect_w, rect_h)
    if frontier is None:
        return None
    first = np.zeros((rect_h, rect_w), dtype=bool)
    second = np.zeros((rect_h, rect_w), dtype=bool)
    for c, cut in enumerate(frontier):
        first[:cut, c] = True
        second[cut:, c] = True
    return first, second


def _piece_from_mask(mask: np.ndarray, color: int) -> dict:
    trimmed = _trim_mask(mask)
    coords = tuple(sorted(map(tuple, np.argwhere(trimmed))))
    patch = np.zeros(trimmed.shape, dtype=int)
    patch[trimmed] = int(color)
    return {
        "color": int(color),
        "mask": trimmed,
        "coords": coords,
        "h": int(trimmed.shape[0]),
        "w": int(trimmed.shape[1]),
        "area": len(coords),
        "patch": patch,
    }


def _dilate(mask: np.ndarray) -> np.ndarray:
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


def _can_place_piece(blocked: np.ndarray, piece: dict, r0: int, c0: int) -> bool:
    h, w = int(piece["h"]), int(piece["w"])
    return not bool((blocked[r0 : r0 + h, c0 : c0 + w] & piece["mask"]).any())


def _scatter_pieces(pieces: list[dict], grid_h: int, grid_w: int) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(range(len(pieces)), key=lambda i: (-pieces[i]["area"], -pieces[i]["h"] * pieces[i]["w"]))
    for idx in order:
        piece = pieces[idx]
        h, w = int(piece["h"]), int(piece["w"])
        if h > grid_h or w > grid_w:
            return None

        blocked = _dilate(occupied)
        placed = False
        tries = 0
        while tries < 250:
            tries += 1
            r0 = _RNG.randint(0, grid_h - h)
            c0 = _RNG.randint(0, grid_w - w)
            if not _can_place_piece(blocked, piece, r0, c0):
                continue

            region = grid[r0 : r0 + h, c0 : c0 + w]
            region[piece["mask"]] = int(piece["color"])
            occupied[r0 : r0 + h, c0 : c0 + w] |= piece["mask"]
            placed = True
            break

        if not placed:
            return None

    return grid


def _build_pair(color_a: int, color_b: int) -> tuple[dict, dict, np.ndarray] | None:
    rect_h = _RNG.randint(*RECT_SIDE_RANGE)
    rect_w = _RNG.randint(*RECT_SIDE_RANGE)
    orientation = _RNG.choice(["vertical", "horizontal"])
    masks = _split_rectangle(rect_h, rect_w, orientation)
    if masks is None:
        return None
    first_mask, second_mask = masks

    patch = np.full((rect_h, rect_w), int(color_b), dtype=int)
    patch[first_mask] = int(color_a)
    return _piece_from_mask(first_mask, color_a), _piece_from_mask(second_mask, color_b), patch


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = _RNG.randint(*PAIR_COUNT_RANGE)
    color_pool = _RNG.sample(list(range(1, 10)), 2 * pair_count)

    rectangles: list[np.ndarray] = []
    pieces: list[dict] = []
    areas: set[int] = set()
    total_piece_area = 0

    for i in range(pair_count):
        pair = _build_pair(color_pool[2 * i], color_pool[2 * i + 1])
        if pair is None:
            return None
        first_piece, second_piece, rectangle = pair
        area = int(rectangle.size)
        if area in areas:
            return None
        areas.add(area)
        rectangles.append(rectangle)
        pieces.extend([first_piece, second_piece])
        total_piece_area += int(first_piece["area"]) + int(second_piece["area"])

    expected = _stack_rectangles(rectangles, BG)
    if max(expected.shape) > ARC_MAX:
        return None

    min_side = max(
        12,
        max(max(piece["h"], piece["w"]) for piece in pieces) + 2,
        int(np.ceil(np.sqrt(total_piece_area * 2.2))),
    )
    if min_side > ARC_MAX:
        return None

    grid_h = _RNG.randint(min_side, ARC_MAX)
    grid_w = _RNG.randint(min_side, ARC_MAX)
    if grid_h < GRID_SIDE_RANGE[0]:
        grid_h = GRID_SIDE_RANGE[0]
    if grid_w < GRID_SIDE_RANGE[0]:
        grid_w = GRID_SIDE_RANGE[0]
    grid_h = min(grid_h, ARC_MAX)
    grid_w = min(grid_w, ARC_MAX)

    grid = _scatter_pieces(pieces, grid_h, grid_w)
    if grid is None:
        return None
    if np.count_nonzero(grid) * 2 >= grid.size:
        return None

    found = _extract_fragments(grid, BG)
    if len(found) != 2 * pair_count:
        return None

    solutions = _enumerate_pairings(found, limit=2)
    if len(solutions) != 1:
        return None
    if not np.array_equal(_stack_rectangles(solutions[0], BG), expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _generate_case()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate a valid sample after {MAX_ATTEMPTS} attempts")
