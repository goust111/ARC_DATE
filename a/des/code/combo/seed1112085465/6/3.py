from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
MARKER = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]

GEN_ATTEMPTS = 500
GRID_BASE_RANGE = (18, 24)
GRID_EXTRA_PER_OBJECT = 3
OBJECT_COUNT_CHOICES = (2, 3, 4)
EDGE_BIAS_PROB = 0.35
MIN_VISIBLE_FRACTION = 0.45


def neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def discrete_center_top_left(shape: tuple[int, int], center: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return center[0] - (h - 1) // 2, center[1] - (w - 1) // 2


def scaled_visible_mask(
    local_mask: np.ndarray,
    scale: int,
    center: tuple[int, int],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    scaled = np.kron(local_mask.astype(bool), np.ones((scale, scale), dtype=bool))
    top, left = discrete_center_top_left(scaled.shape, center)
    h, w = grid_shape
    visible = np.zeros((h, w), dtype=bool)
    for rr, cc in np.argwhere(scaled):
        gr = top + int(rr)
        gc = left + int(cc)
        if 0 <= gr < h and 0 <= gc < w:
            visible[gr, gc] = True
    return visible


def exterior_reachable(obj_mask: np.ndarray) -> np.ndarray:
    h, w = obj_mask.shape
    passable = ~obj_mask
    seen = np.zeros_like(obj_mask, dtype=bool)
    queue = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    while queue:
        cr, cc = queue.popleft()
        for nr, nc in neighbors4(cr, cc, h, w):
            if passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def extract_entities(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    coord_to_object: dict[tuple[int, int], int] = {}
    objects: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BG, MARKER):
            continue
        for comp in components(g == color):
            obj_mask = np.zeros((h, w), dtype=bool)
            obj_mask[comp[:, 0], comp[:, 1]] = True
            for r, c in comp:
                coord_to_object[(int(r), int(c))] = len(objects)
            objects.append(
                {
                    "color": color,
                    "mask": obj_mask,
                    "local_mask": crop_mask(obj_mask),
                    "markers": [],
                    "exterior": exterior_reachable(obj_mask),
                }
            )

    assert objects, "expected at least one non-marker object"

    red_positions = np.argwhere(g == MARKER)
    assert red_positions.size > 0, "expected red markers"
    for rr, cc in red_positions:
        r = int(rr)
        c = int(cc)
        adjacent = set()
        for nr, nc in neighbors4(r, c, h, w):
            idx = coord_to_object.get((nr, nc))
            if idx is None:
                continue
            if bool(objects[idx]["exterior"][r, c]):
                adjacent.add(idx)
        assert len(adjacent) == 1, "each red marker must belong to exactly one object"
        idx = adjacent.pop()
        objects[idx]["markers"].append((r, c))

    for obj in objects:
        markers = obj["markers"]
        assert 1 <= len(markers) <= 4, "each object needs 1-4 red markers"

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each monochrome object by the number of exterior red markers touching it,
    erase the original object and markers, and place the scaled copy at the marker
    centroid using discrete bbox-center alignment.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    entities = extract_entities(g)

    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    for obj in entities:
        local_mask = np.asarray(obj["local_mask"], dtype=bool)
        markers = list(obj["markers"])
        scale = len(markers)
        center = (
            sum(r for r, _ in markers) // scale,
            sum(c for _, c in markers) // scale,
        )
        visible = scaled_visible_mask(local_mask, scale, center, (h, w))
        assert np.any(visible), "scaled object must remain at least partially visible"
        assert not np.any(occupied & visible), "scaled objects must not overlap"
        occupied |= visible
        out[visible] = int(obj["color"])
    return out


def random_shape(rng: np.random.Generator, scale: int) -> np.ndarray:
    if scale == 4:
        height = int(rng.integers(2, 4))
        width = int(rng.integers(2, 4))
        target_area = int(rng.integers(3, min(height * width, 5) + 1))
    elif scale == 3:
        height = int(rng.integers(2, 5))
        width = int(rng.integers(2, 5))
        target_area = int(rng.integers(3, min(height * width, 6) + 1))
    else:
        height = int(rng.integers(2, 5))
        width = int(rng.integers(2, 5))
        target_area = int(rng.integers(3, min(height * width, 8) + 1))

    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        seed = (int(rng.integers(height)), int(rng.integers(width)))
        mask[seed] = True
        frontier = {
            (nr, nc)
            for nr, nc in neighbors4(seed[0], seed[1], height, width)
            if not mask[nr, nc]
        }
        while mask.sum() < target_area and frontier:
            idx = int(rng.integers(len(frontier)))
            cell = list(frontier)[idx]
            frontier.remove(cell)
            mask[cell] = True
            for nr, nc in neighbors4(cell[0], cell[1], height, width):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))

        local = crop_mask(mask)
        if rng.random() < 0.5:
            local = np.flipud(local)
        if rng.random() < 0.5:
            local = np.fliplr(local)
        local = np.rot90(local, int(rng.integers(4)))

        if local.shape[0] < 2 or local.shape[1] < 2:
            continue
        if int(local.sum()) < 3:
            continue
        candidate_count = len(
            local_boundary_candidates(
                local,
                (1, 1),
                (local.shape[0] + 2, local.shape[1] + 2),
            )
        )
        if candidate_count < 4:
            continue
        return local.astype(bool)
    raise RuntimeError("failed to sample a connected shape")


def local_boundary_candidates(local_mask: np.ndarray, top_left: tuple[int, int], grid_shape: tuple[int, int]):
    top, left = top_left
    h, w = grid_shape
    cells = {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(local_mask)
    }
    candidates = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c, h, w):
            if (nr, nc) not in cells:
                candidates.add((nr, nc))
    return sorted(candidates)


def sample_origin(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    local_shape: tuple[int, int],
) -> tuple[int, int]:
    h, w = grid_shape
    lh, lw = local_shape
    max_r = h - lh
    max_c = w - lw

    def sample_coord(limit: int) -> int:
        if limit <= 0:
            return 0
        if rng.random() < EDGE_BIAS_PROB:
            edge_span = min(3, limit + 1)
            left_choices = list(range(edge_span))
            right_start = max(0, limit - edge_span + 1)
            choices = sorted(set(left_choices + list(range(right_start, limit + 1))))
            return int(choices[int(rng.integers(len(choices)))])
        return int(rng.integers(limit + 1))

    return sample_coord(max_r), sample_coord(max_c)


def dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def try_place_object(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked_input: np.ndarray,
    output_occ: np.ndarray,
    scale: int,
    color: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    h, w = grid.shape
    local = random_shape(rng, scale)

    for _ in range(120):
        top, left = sample_origin(rng, (h, w), local.shape)
        object_cells = np.zeros((h, w), dtype=bool)
        coords = np.argwhere(local)
        object_cells[top + coords[:, 0], left + coords[:, 1]] = True
        if np.any(blocked_input & object_cells):
            continue

        candidates = [
            pos
            for pos in local_boundary_candidates(local, (top, left), (h, w))
            if not blocked_input[pos]
        ]
        if len(candidates) < scale:
            continue

        for _ in range(80):
            choice_idx = rng.choice(len(candidates), size=scale, replace=False)
            markers = [candidates[int(idx)] for idx in np.atleast_1d(choice_idx)]
            marker_mask = np.zeros((h, w), dtype=bool)
            for r, c in markers:
                marker_mask[r, c] = True

            combined = object_cells | marker_mask
            if np.any(blocked_input & combined):
                continue

            center = (
                sum(r for r, _ in markers) // scale,
                sum(c for _, c in markers) // scale,
            )
            visible = scaled_visible_mask(local, scale, center, (h, w))
            visible_count = int(visible.sum())
            total_scaled = int(local.sum()) * scale * scale
            if visible_count == 0:
                continue
            if visible_count < int(np.ceil(total_scaled * MIN_VISIBLE_FRACTION)):
                continue
            if np.any(output_occ & visible):
                continue

            grid[object_cells] = color
            grid[marker_mask] = MARKER
            blocked_input |= dilate(combined)
            output_occ |= visible
            return grid, blocked_input, output_occ
    return None


def gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    object_count = int(rng.choice(OBJECT_COUNT_CHOICES))
    base_low = GRID_BASE_RANGE[0] + GRID_EXTRA_PER_OBJECT * (object_count - 2)
    base_high = GRID_BASE_RANGE[1] + GRID_EXTRA_PER_OBJECT * (object_count - 2)
    height = int(rng.integers(base_low, base_high + 1))
    width = int(rng.integers(base_low, base_high + 1))

    grid = np.full((height, width), BG, dtype=int)
    blocked_input = np.zeros((height, width), dtype=bool)
    output_occ = np.zeros((height, width), dtype=bool)

    scales = [1, 2, 3, 4]
    rng.shuffle(scales)
    scales = scales[:object_count]

    colors = PALETTE.copy()
    rng.shuffle(colors)
    colors = colors[:object_count]

    for scale, color in zip(scales, colors):
        placed = try_place_object(rng, grid, blocked_input, output_occ, scale, color)
        if placed is None:
            return None
        grid, blocked_input, output_occ = placed

    try:
        entities = extract_entities(grid)
        assert len(entities) == object_count
        marker_counts = sorted(len(obj["markers"]) for obj in entities)
        assert marker_counts == sorted(scales)
        out = main(grid)
        assert np.any(out != BG)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
