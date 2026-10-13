from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
PARTITION_ATTEMPTS = 120
SCATTER_ATTEMPTS = 120
GLOBAL_LAYOUT_ATTEMPTS = 120
RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (3, 7)

_RNG = np.random.default_rng()
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return rel, r1 - r0 + 1, c1 - c0 + 1


def _is_solid_rectangle(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != BG)


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rel, ph, pw = _normalize(coords)
            pieces.append(
                {
                    "coords": tuple(coords),
                    "rel": rel,
                    "h": ph,
                    "w": pw,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )

    return pieces


def _placement_mask(rel: tuple[tuple[int, int], ...], top: int, left: int, out_w: int) -> int:
    mask = 0
    for r, c in rel:
        mask |= 1 << ((top + r) * out_w + left + c)
    return mask


def _can_tile_rectangle(pieces: list[dict], out_h: int, out_w: int) -> bool:
    placements: list[list[int]] = []
    for piece in pieces:
        opts: list[int] = []
        for top in range(out_h - piece["h"] + 1):
            for left in range(out_w - piece["w"] + 1):
                opts.append(_placement_mask(piece["rel"], top, left, out_w))
        if not opts:
            return False
        placements.append(opts)

    order = sorted(
        range(len(pieces)),
        key=lambda idx: (len(placements[idx]), -pieces[idx]["area"], -pieces[idx]["h"] * pieces[idx]["w"]),
    )
    dead: set[tuple[int, int]] = set()

    def dfs(depth: int, used: int) -> bool:
        key = (depth, used)
        if key in dead:
            return False
        if depth == len(order):
            return True

        piece_idx = order[depth]
        for mask in placements[piece_idx]:
            if mask & used:
                continue
            if dfs(depth + 1, used | mask):
                return True

        dead.add(key)
        return False

    return dfs(0, 0)


def _projection_hint(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    rows = {int(r) for r, _ in coords}
    cols = {int(c) for _, c in coords}
    return len(rows), len(cols)


def _solve_rectangle_dims(grid: np.ndarray, color: int, pieces: list[dict]) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    area = int(coords.shape[0])
    row_hint, col_hint = _projection_hint(grid, color)

    if area == row_hint * col_hint:
        return row_hint, col_hint

    max_h = max(piece["h"] for piece in pieces)
    max_w = max(piece["w"] for piece in pieces)
    valid: list[tuple[int, int]] = []

    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        for out_h, out_w in ((h, w), (w, h)):
            if out_h < max_h or out_w < max_w:
                continue
            if (out_h, out_w) in valid:
                continue
            if _can_tile_rectangle(pieces, out_h, out_w):
                valid.append((out_h, out_w))

    if not valid:
        raise AssertionError("pieces must tile a rectangle")
    if len(valid) == 1:
        return valid[0]

    valid.sort(
        key=lambda dims: (
            abs(dims[0] - row_hint) + abs(dims[1] - col_hint),
            abs(dims[0] - dims[1]),
            dims[0],
            dims[1],
        )
    )
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's fragments into a solid rectangle, then stack the rectangles
    from fewest fragments to most fragments.
    """
    g = np.asarray(grid, dtype=int)
    rectangles: list[dict] = []

    for color in _colors_present(g):
        pieces = _components_for_color(g, color)
        dims = _solve_rectangle_dims(g, color, pieces)
        top, left, _bottom, _right = _bbox([coord for piece in pieces for coord in piece["coords"]])
        rectangles.append(
            {
                "color": color,
                "count": len(pieces),
                "area": sum(piece["area"] for piece in pieces),
                "h": dims[0],
                "w": dims[1],
                "anchor": (top, left),
            }
        )

    rectangles.sort(key=lambda item: (item["count"], item["area"], item["anchor"][0], item["anchor"][1]))

    out_h = sum(item["h"] for item in rectangles)
    out_w = max(item["w"] for item in rectangles)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for item in rectangles:
        out[row : row + item["h"], : item["w"]] = item["color"]
        row += item["h"]

    return out


def _random_partition(rect_h: int, rect_w: int, piece_count: int) -> list[list[tuple[int, int]]] | None:
    cells = [(r, c) for r in range(rect_h) for c in range(rect_w)]

    for _ in range(PARTITION_ATTEMPTS):
        seed_indices = _RNG.choice(len(cells), size=piece_count, replace=False)
        owner = -np.ones((rect_h, rect_w), dtype=int)
        frontiers = [set() for _ in range(piece_count)]
        sizes = [1] * piece_count

        for idx, cell_index in enumerate(seed_indices):
            r, c = cells[int(cell_index)]
            owner[r, c] = idx

        def add_frontier(piece_idx: int, cell: tuple[int, int]) -> None:
            r, c = cell
            for dr, dc in _DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rect_h and 0 <= nc < rect_w and owner[nr, nc] == -1:
                    frontiers[piece_idx].add((nr, nc))

        for idx, cell_index in enumerate(seed_indices):
            add_frontier(idx, cells[int(cell_index)])

        remaining = rect_h * rect_w - piece_count
        while remaining > 0:
            active = [idx for idx in range(piece_count) if frontiers[idx]]
            if not active:
                break

            min_size = min(sizes[idx] for idx in active)
            growable = [idx for idx in active if sizes[idx] <= min_size + 1]
            piece_idx = int(growable[int(_RNG.integers(len(growable)))])

            frontier_list = list(frontiers[piece_idx])
            choice = frontier_list[int(_RNG.integers(len(frontier_list)))]
            frontiers[piece_idx].discard(choice)
            rr, cc = choice
            if owner[rr, cc] != -1:
                continue

            owner[rr, cc] = piece_idx
            sizes[piece_idx] += 1
            remaining -= 1
            add_frontier(piece_idx, choice)

        if remaining != 0:
            continue

        pieces: list[list[tuple[int, int]]] = []
        for idx in range(piece_count):
            coords = [(r, c) for r in range(rect_h) for c in range(rect_w) if owner[r, c] == idx]
            pieces.append(coords)

        if min(len(piece) for piece in pieces) < 2:
            continue
        if not any(not _is_solid_rectangle(piece) for piece in pieces):
            continue
        return pieces

    return None


def _can_place_piece(
    cells: tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
) -> bool:
    if any(cell in occupied for cell in cells):
        return False
    for r, c in cells:
        for dr, dc in _DIR4:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _scatter_pieces(pieces: list[dict], rect_h: int, rect_w: int) -> np.ndarray | None:
    total_area = sum(piece["area"] for piece in pieces)
    max_piece_h = max(piece["h"] for piece in pieces)
    max_piece_w = max(piece["w"] for piece in pieces)

    for _ in range(SCATTER_ATTEMPTS):
        canvas_h = rect_h + max_piece_h + int(_RNG.integers(2, 7))
        canvas_w = rect_w + max_piece_w + int(_RNG.integers(2, 7))

        order = list(range(len(pieces)))
        _RNG.shuffle(order)
        order.sort(key=lambda idx: -pieces[idx]["area"])

        occupied: set[tuple[int, int]] = set()
        placed: list[tuple[tuple[int, int], ...]] = []

        success = True
        for piece_idx in order:
            piece = pieces[piece_idx]
            candidates: list[tuple[tuple[int, int], ...]] = []
            for top in range(canvas_h - piece["h"] + 1):
                for left in range(canvas_w - piece["w"] + 1):
                    shifted = tuple((top + r, left + c) for r, c in piece["rel"])
                    if _can_place_piece(shifted, occupied):
                        candidates.append(shifted)
            if not candidates:
                success = False
                break
            choice = candidates[int(_RNG.integers(len(candidates)))]
            placed.append(choice)
            occupied.update(choice)

        if not success:
            continue

        merged = [cell for shape in placed for cell in shape]
        r0, c0, r1, c1 = _bbox(merged)
        if (r1 - r0 + 1) * (c1 - c0 + 1) == total_area:
            continue

        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in merged:
            mask[r - r0, c - c0] = True
        return mask

    return None


def _sample_fragment_counts(color_count: int) -> list[int]:
    if color_count == 2:
        if float(_RNG.random()) < 0.5:
            count = int(_RNG.integers(2, 5))
            return [count, count]
        counts = [2, 3, 4]
        _RNG.shuffle(counts)
        return counts[:2]

    if color_count == 3:
        if float(_RNG.random()) < 0.5:
            repeated = int(_RNG.integers(2, 5))
            other = int(_RNG.choice([count for count in (2, 3, 4) if count != repeated]))
            counts = [repeated, repeated, other]
        else:
            counts = [2, 3, 4]
        _RNG.shuffle(counts)
        return counts

    counts = [2, 3, 4, int(_RNG.integers(2, 5))]
    _RNG.shuffle(counts)
    return counts


def _sample_rectangle_dims(piece_count: int, forbidden: set[tuple[int, int]]) -> tuple[int, int] | None:
    candidates = []
    min_area = max(8, 2 * piece_count + 2)
    for rect_h in range(RECT_H_RANGE[0], RECT_H_RANGE[1] + 1):
        for rect_w in range(RECT_W_RANGE[0], RECT_W_RANGE[1] + 1):
            area = rect_h * rect_w
            if area < min_area or area > 18:
                continue
            if (piece_count, area) in forbidden:
                continue
            candidates.append((rect_h, rect_w))

    if not candidates:
        return None
    _RNG.shuffle(candidates)
    return candidates[0]


def _build_expected_output(specs: list[dict]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: (item["count"], item["area"]))
    out_h = sum(item["rect_h"] for item in ordered)
    out_w = max(item["rect_w"] for item in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for item in ordered:
        out[row : row + item["rect_h"], : item["rect_w"]] = item["color"]
        row += item["rect_h"]

    return out


def _place_color_masks(specs: list[dict]) -> np.ndarray | None:
    masks = [spec["mask"] for spec in specs]
    max_mask_h = max(mask.shape[0] for mask in masks)
    max_mask_w = max(mask.shape[1] for mask in masks)
    bbox_area = sum(mask.shape[0] * mask.shape[1] for mask in masks)

    for _ in range(GLOBAL_LAYOUT_ATTEMPTS):
        base = max(int(np.ceil(np.sqrt(bbox_area))) + 4, max(max_mask_h, max_mask_w) + 2, 16)
        grid_h = int(_RNG.integers(base, 31))
        grid_w = int(_RNG.integers(base, 31))

        order = list(range(len(specs)))
        order.sort(key=lambda idx: -(masks[idx].shape[0] * masks[idx].shape[1]))
        placements: list[tuple[int, int, int, int, int]] = []

        success = True
        for spec_idx in order:
            mask = masks[spec_idx]
            mask_h, mask_w = mask.shape
            candidates: list[tuple[int, int]] = []
            for top in range(grid_h - mask_h + 1):
                for left in range(grid_w - mask_w + 1):
                    ok = True
                    for placed_top, placed_left, placed_h, placed_w, _ in placements:
                        rows_disjoint = top + mask_h + 1 <= placed_top or placed_top + placed_h + 1 <= top
                        cols_disjoint = left + mask_w + 1 <= placed_left or placed_left + placed_w + 1 <= left
                        if not (rows_disjoint or cols_disjoint):
                            ok = False
                            break
                    if ok:
                        candidates.append((top, left))
            if not candidates:
                success = False
                break

            top, left = candidates[int(_RNG.integers(len(candidates)))]
            placements.append((top, left, mask_h, mask_w, spec_idx))

        if not success:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        for top, left, mask_h, mask_w, spec_idx in placements:
            sub = grid[top : top + mask_h, left : left + mask_w]
            sub[masks[spec_idx]] = specs[spec_idx]["color"]

        if int((grid != BG).sum()) * 2 >= grid_h * grid_w:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    color_count = int(_RNG.integers(2, 5))
    counts = _sample_fragment_counts(color_count)
    colors = [int(color) for color in _RNG.choice(PALETTE, size=color_count, replace=False)]
    specs: list[dict] = []
    forbidden_keys: set[tuple[int, int]] = set()

    for color, count in zip(colors, counts):
        built = None
        for _ in range(80):
            dims = _sample_rectangle_dims(count, forbidden_keys)
            if dims is None:
                return None
            rect_h, rect_w = dims
            raw_pieces = _random_partition(rect_h, rect_w, count)
            if raw_pieces is None:
                continue

            pieces = []
            for coords in raw_pieces:
                rel, piece_h, piece_w = _normalize(coords)
                pieces.append(
                    {
                        "rel": rel,
                        "h": piece_h,
                        "w": piece_w,
                        "area": len(coords),
                    }
                )

            mask = _scatter_pieces(pieces, rect_h, rect_w)
            if mask is None:
                continue

            built = {
                "color": color,
                "count": count,
                "area": rect_h * rect_w,
                "rect_h": rect_h,
                "rect_w": rect_w,
                "mask": mask,
            }
            forbidden_keys.add((count, rect_h * rect_w))
            break

        if built is None:
            return None
        specs.append(built)

    grid = _place_color_masks(specs)
    if grid is None:
        return None

    expected = _build_expected_output(specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
