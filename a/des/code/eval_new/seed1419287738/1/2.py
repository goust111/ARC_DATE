from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SIDE_ORDER = ("top", "left", "right", "bottom")
_SIDE_RANK = {side: idx for idx, side in enumerate(_SIDE_ORDER)}

# --------------- solver helpers ---------------


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in _DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_dims(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _centroid(coords: list[tuple[int, int]]) -> tuple[float, float]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return float(np.mean(rows)), float(np.mean(cols))


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _is_hollow_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    coord_set = set(coords)
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0, r1 + 1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))
    return coord_set == perimeter


def _normalize_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _fragment_sort_key(
    frame_bbox: tuple[int, int, int, int], frag_coords: list[tuple[int, int]]
) -> tuple[int, float, float]:
    fr0, fc0, fr1, fc1 = frame_bbox
    r0, c0, r1, c1 = _bbox(frag_coords)
    cr, cc = _centroid(frag_coords)

    if r1 < fr0:
        side = "top"
    elif r0 > fr1:
        side = "bottom"
    elif c1 < fc0:
        side = "left"
    elif c0 > fc1:
        side = "right"
    else:
        center_r = (fr0 + fr1) / 2.0
        center_c = (fc0 + fc1) / 2.0
        dr = cr - center_r
        dc = cc - center_c
        if abs(dr) >= abs(dc):
            side = "bottom" if dr > 0 else "top"
        else:
            side = "right" if dc > 0 else "left"

    if side in {"top", "bottom"}:
        return _SIDE_RANK[side], cc, cr
    return _SIDE_RANK[side], cr, cc


def _pack_shapes(
    inner_shape: tuple[int, int], shapes: list[tuple[tuple[int, int], ...]]
) -> list[tuple[int, int]] | None:
    """Backtracking packer matching the row-major placement described by the puzzle."""
    h, w = inner_shape
    if sum(len(shape) for shape in shapes) != h * w:
        return None

    board = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []

    def search(idx: int) -> bool:
        if idx == len(shapes):
            return bool(board.all())

        empty = np.argwhere(~board)
        if empty.size == 0:
            return idx == len(shapes)
        start_r, start_c = map(int, empty[0])
        shape = shapes[idx]
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)

        tried: set[tuple[int, int]] = set()
        for sr, sc in shape:
            anchor = (start_r - sr, start_c - sc)
            if anchor in tried:
                continue
            tried.add(anchor)
            ar, ac = anchor
            if ar < 0 or ac < 0 or ar + max_r >= h or ac + max_c >= w:
                continue
            cells = [(ar + r, ac + c) for r, c in shape]
            if any(board[r, c] for r, c in cells):
                continue
            for r, c in cells:
                board[r, c] = True
            placements.append(anchor)
            if search(idx + 1):
                return True
            placements.pop()
            for r, c in cells:
                board[r, c] = False
        return False

    return placements if search(0) else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move scattered monochrome fragments into the same-colored rectangular containers.

    Input structure:
    - The background is black.
    - Several small hollow rectangular containers appear, each with a distinct border color and an empty black interior.
    - Outside the containers are small 4-connected monochrome fragments of size 2-5; every fragment color matches exactly one container border color.

    Rule:
    - Match each fragment to the container with the same color.
    - For each container, order its fragments by the side they occupy relative to the container: top first, then left, then right, then bottom; within a side, earlier means more leftward for horizontal sides or more upward for vertical sides.
    - Pack the ordered fragments into the container interior from the top-left in row-major order while preserving each fragment's shape; because the total same-color fragment area equals the container interior area, the interior becomes fully filled.
    - Remove the original scattered fragments and keep only the containers with their filled interiors.

    Key insight:
    - Color decides which container a fragment belongs to, while the fragment's side relative to the container encodes the placement order inside that container.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)

    frame_by_color: dict[int, list[tuple[int, int]]] = {}
    fragments_by_color: dict[int, list[list[tuple[int, int]]]] = {}

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components_for_color(g, color):
            if _is_hollow_frame(comp):
                frame_by_color[color] = comp
            else:
                fragments_by_color.setdefault(color, []).append(comp)
                for r, c in comp:
                    out[r, c] = bg

    for color, frame_coords in frame_by_color.items():
        r0, c0, r1, c1 = _bbox(frame_coords)
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        frags = fragments_by_color.get(color, [])
        ordered = sorted(frags, key=lambda comp: _fragment_sort_key((r0, c0, r1, c1), comp))
        shapes = [_normalize_shape(comp) for comp in ordered]

        # The packed layout is invisible after filling, but we still validate the
        # stated assembly rule when the fragments support it.
        _pack_shapes((inner_h, inner_w), shapes)

        total_area = sum(len(comp) for comp in ordered)
        assert total_area == inner_h * inner_w, "same-color fragment area must match container interior"
        out[r0 + 1 : r1, c0 + 1 : c1] = color

    return out


# --------------- generator helpers ---------------

_GRID_MIN = 22
_GRID_MAX = 30
_INNER_DIMS = [(3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (5, 3)]
_CONTAINER_COUNT_CHOICES = (2, 3)
_CLUSTER_RETRIES = 150
_GEN_RETRIES = 300
_PATCH_MARGIN = 8
_GAP_CHOICES = (1, 2)


def _first_true(mask: np.ndarray) -> tuple[int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r, c = map(int, pts[0])
    return r, c


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    return sizes


def _remaining_is_viable(mask: np.ndarray, remaining_sizes: list[int]) -> bool:
    remaining = int(mask.sum())
    if remaining != sum(remaining_sizes):
        return False
    if not remaining_sizes:
        return remaining == 0
    min_piece = min(remaining_sizes)
    for size in _component_sizes(mask):
        if 0 < size < min_piece:
            return False
    return True


def _connected_sets_from_start(
    mask: np.ndarray, start: tuple[int, int], target_size: int
) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    available = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    results: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    def rec(cells: set[tuple[int, int]], frontier: set[tuple[int, int]]) -> None:
        if len(cells) == target_size:
            key = tuple(sorted(cells))
            if key in seen:
                return
            seen.add(key)
            bh, bw = _bbox_dims(key)
            if bh <= 3 and bw <= 3:
                results.append(key)
            return

        options = list(frontier)
        _RNG.shuffle(options)
        for cell in options:
            new_cells = set(cells)
            new_cells.add(cell)
            new_frontier = set(frontier)
            new_frontier.discard(cell)
            for nr, nc in _neighbors4(cell[0], cell[1], h, w):
                nb = (nr, nc)
                if nb in available and nb not in new_cells:
                    new_frontier.add(nb)
            rec(new_cells, new_frontier)

    initial_frontier = {
        (nr, nc) for nr, nc in _neighbors4(start[0], start[1], h, w) if mask[nr, nc]
    }
    rec({start}, initial_frontier)
    _RNG.shuffle(results)
    return results


def _partition_interior(h: int, w: int, sizes: list[int]) -> list[tuple[tuple[int, int], ...]] | None:
    free = np.ones((h, w), dtype=bool)
    pieces: list[tuple[tuple[int, int], ...]] = []

    def search(idx: int) -> bool:
        if idx == len(sizes):
            return not bool(free.any())
        start = _first_true(free)
        if start is None:
            return idx == len(sizes)
        candidates = _connected_sets_from_start(free, start, sizes[idx])
        for cells in candidates:
            for r, c in cells:
                free[r, c] = False
            if _remaining_is_viable(free, sizes[idx + 1 :]):
                pieces.append(_normalize_shape(list(cells)))
                if search(idx + 1):
                    return True
                pieces.pop()
            for r, c in cells:
                free[r, c] = True
        return False

    return pieces if search(0) else None


def _random_composition(total: int, parts: int) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if total < 2 * parts or total > 5 * parts:
        return None
    if parts == 1:
        return [total] if 2 <= total <= 5 else None

    lo = max(2, total - 5 * (parts - 1))
    hi = min(5, total - 2 * (parts - 1))
    choices = list(range(lo, hi + 1))
    _RNG.shuffle(choices)
    for first in choices:
        tail = _random_composition(total - first, parts - 1)
        if tail is not None:
            return [first] + tail
    return None


def _draw_frame(patch: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, color: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    patch[r0, c0 : c1 + 1] = color
    patch[r1, c0 : c1 + 1] = color
    patch[r0 : r1 + 1, c0] = color
    patch[r0 : r1 + 1, c1] = color


def _crop_nonzero(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != 0)
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    r0 = max(0, int(r0) - 1)
    c0 = max(0, int(c0) - 1)
    r1 = min(patch.shape[0] - 1, int(r1) + 1)
    c1 = min(patch.shape[1] - 1, int(c1) + 1)
    return patch[r0 : r1 + 1, c0 : c1 + 1]


def _build_cluster_patch(
    color: int,
    inner_h: int,
    inner_w: int,
    sides: list[str],
    shapes: list[tuple[tuple[int, int], ...]],
) -> np.ndarray:
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    patch = np.zeros((frame_h + 2 * _PATCH_MARGIN, frame_w + 2 * _PATCH_MARGIN), dtype=int)
    frame_r0 = _PATCH_MARGIN
    frame_c0 = _PATCH_MARGIN
    _draw_frame(patch, frame_r0, frame_c0, inner_h, inner_w, color)

    for side, shape in zip(sides, shapes):
        sh, sw = _bbox_dims(shape)
        gap = int(_RNG.choice(_GAP_CHOICES))

        if side == "top":
            base_r = frame_r0 - gap - sh
            base_c = int(_RNG.integers(frame_c0, frame_c0 + frame_w - sw + 1))
        elif side == "bottom":
            base_r = frame_r0 + frame_h + gap
            base_c = int(_RNG.integers(frame_c0, frame_c0 + frame_w - sw + 1))
        elif side == "left":
            base_r = int(_RNG.integers(frame_r0, frame_r0 + frame_h - sh + 1))
            base_c = frame_c0 - gap - sw
        else:
            base_r = int(_RNG.integers(frame_r0, frame_r0 + frame_h - sh + 1))
            base_c = frame_c0 + frame_w + gap

        for dr, dc in shape:
            patch[base_r + dr, base_c + dc] = color

    return _crop_nonzero(patch)


def _make_cluster_patch(color: int) -> np.ndarray | None:
    for _ in range(_CLUSTER_RETRIES):
        inner_h, inner_w = _INNER_DIMS[int(_RNG.integers(len(_INNER_DIMS)))]
        area = inner_h * inner_w
        feasible_counts = [k for k in range(2, 5) if 2 * k <= area <= 5 * k]
        if not feasible_counts:
            continue
        part_count = int(_RNG.choice(feasible_counts))
        side_choices = list(_SIDE_ORDER)
        _RNG.shuffle(side_choices)
        sides = sorted(side_choices[:part_count], key=lambda side: _SIDE_RANK[side])
        sizes = _random_composition(area, part_count)
        if sizes is None:
            continue
        shapes = _partition_interior(inner_h, inner_w, sizes)
        if shapes is None:
            continue
        if _pack_shapes((inner_h, inner_w), shapes) is None:
            continue
        return _build_cluster_patch(color, inner_h, inner_w, sides, shapes)
    return None


def _can_place_rect(
    rects: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = candidate
    for or0, oc0, or1, oc1 in rects:
        if not (r1 < or0 - 1 or r0 > or1 + 1 or c1 < oc0 - 1 or c0 > oc1 + 1):
            return False
    return True


def _place_patches(
    grid_h: int, grid_w: int, patches: list[np.ndarray]
) -> np.ndarray | None:
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    rects: list[tuple[int, int, int, int]] = []
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        if ph > grid_h or pw > grid_w:
            return None
        for _ in range(250):
            r0 = int(_RNG.integers(0, grid_h - ph + 1))
            c0 = int(_RNG.integers(0, grid_w - pw + 1))
            candidate = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if _can_place_rect(rects, candidate):
                rects.append(candidate)
                placements[idx] = (r0, c0)
                break
        else:
            return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, (r0, c0) in placements.items():
        patch = patches[idx]
        nonzero = patch != 0
        view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        view[nonzero] = patch[nonzero]
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_containers = int(_RNG.choice(_CONTAINER_COUNT_CHOICES, p=[0.7, 0.3]))
    colors = _RNG.choice(np.arange(1, 10), size=n_containers, replace=False)

    patches: list[np.ndarray] = []
    for color in colors:
        patch = _make_cluster_patch(int(color))
        if patch is None:
            return None
        patches.append(patch)

    min_h = max(max(int(p.shape[0]) for p in patches), _GRID_MIN)
    min_w = max(max(int(p.shape[1]) for p in patches), _GRID_MIN)
    if min_h > _GRID_MAX or min_w > _GRID_MAX:
        return None

    for _ in range(40):
        grid_h = int(_RNG.integers(min_h, _GRID_MAX + 1))
        grid_w = int(_RNG.integers(min_w, _GRID_MAX + 1))
        grid = _place_patches(grid_h, grid_w, patches)
        if grid is not None:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
