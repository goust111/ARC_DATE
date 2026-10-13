import random
from collections import deque

import numpy as np


YELLOW = 4
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400
MAX_YELLOW_COUNT = 4


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 8:
        return [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = _neighbors(connectivity)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _enclosed_open_cells(local_mask: np.ndarray) -> np.ndarray:
    h, w = local_mask.shape
    open_mask = ~local_mask
    seen = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in _neighbors(4):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    interior = open_mask & ~seen
    return np.argwhere(interior)


def _arrow_direction(local_coords: np.ndarray) -> str | None:
    if len(local_coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(local_coords)
    norm = {(int(r - r0), int(c - c0)) for r, c in local_coords}

    if (r1 - r0 + 1, c1 - c0 + 1) == (3, 2):
        if norm == {(0, 0), (2, 0), (1, 1)}:
            return "right"
        if norm == {(0, 1), (2, 1), (1, 0)}:
            return "left"
    if (r1 - r0 + 1, c1 - c0 + 1) == (2, 3):
        if norm == {(1, 0), (1, 2), (0, 1)}:
            return "up"
        if norm == {(0, 0), (0, 2), (1, 1)}:
            return "down"
    return None


def _extract_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rings: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {bg, YELLOW}):
        for comp in _components_mask(g == color, connectivity=8):
            r0, c0, r1, c1 = _bbox(comp)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[comp[:, 0] - r0, comp[:, 1] - c0] = True
            interior_local = _enclosed_open_cells(local)
            if interior_local.size == 0:
                continue

            interior_mask = np.zeros_like(local, dtype=bool)
            interior_mask[interior_local[:, 0], interior_local[:, 1]] = True

            arrow_hits: list[tuple[str, np.ndarray]] = []
            local_patch = g[r0 : r1 + 1, c0 : c1 + 1]
            candidate_colors = sorted(
                int(v) for v in np.unique(local_patch)
                if int(v) not in {bg, color, YELLOW}
            )
            for arrow_color in candidate_colors:
                arrow_mask = interior_mask & (local_patch == arrow_color)
                for arrow_comp in _components_mask(arrow_mask, connectivity=8):
                    direction = _arrow_direction(arrow_comp)
                    if direction is not None:
                        arrow_hits.append((direction, arrow_comp))

            if len(arrow_hits) != 1:
                continue

            direction, arrow_local = arrow_hits[0]
            clear_coords = np.argwhere(interior_mask)
            yellow_count = int(np.sum(local_patch[interior_mask] == YELLOW))
            rings.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "ring_rel": comp - np.array([r0, c0], dtype=int),
                    "clear_rel": clear_coords,
                    "arrow_rel": arrow_local,
                    "direction": direction,
                    "count": yellow_count,
                }
            )

    rings.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp each closed ring outline repeatedly in its internal arrow direction, using the yellow count as the repeat count.

    Input structure:
    - bg is the most common color.
    - The grid contains one or more single-color closed ring outlines; each ring encloses an interior background region.
    - Inside each ring there is one 3-pixel arrow marker of another color and some yellow pixels elsewhere in the interior.

    Rule:
    - For every ring, recover the outline shape as the ring component relative to its bounding box.
    - Read the arrow direction, count the enclosed yellow pixels, and translate the outline by whole bounding-box steps in that direction.
    - Stamp that translated outline once per yellow pixel, stopping early if the next translated bbox would leave the grid.
    - Keep the original ring in place, but erase the interior arrow and yellow pixels back to background.

    Key insight:
    - The solution combines three independent signals inside each closed outline: the ring gives the stamp shape, the 3-pixel arrow gives the axis direction, and the yellow count gives how many whole-bbox translations to apply.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = g.copy()
    objects = _extract_rings(g)

    for obj in objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        clear_rel = obj["clear_rel"]
        if clear_rel.size:
            out[r0 + clear_rel[:, 0], c0 + clear_rel[:, 1]] = bg

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        ring_rel = obj["ring_rel"]
        color = int(obj["color"])
        direction = obj["direction"]
        count = int(obj["count"])
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        dr, dc = DIR_TO_DELTA[direction]
        step_r = dr * box_h
        step_c = dc * box_w
        for k in range(1, count + 1):
            nr0 = r0 + step_r * k
            nc0 = c0 + step_c * k
            nr1 = nr0 + box_h - 1
            nc1 = nc0 + box_w - 1
            if nr0 < 0 or nc0 < 0 or nr1 >= h or nc1 >= w:
                break
            out[nr0 + ring_rel[:, 0], nc0 + ring_rel[:, 1]] = color

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    return len(_components_mask(mask, connectivity=4)) == 1


def _dilate8_padded(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1 : h + 1, 1 : w + 1] = mask
    out = np.zeros_like(padded)
    for r, c in np.argwhere(padded):
        out[r - 1 : r + 2, c - 1 : c + 2] = True
    return out


def _random_inner_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        mask = np.ones((h, w), dtype=bool)

        boundary_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if r in {0, h - 1} or c in {0, w - 1}
        ]
        rng.shuffle(boundary_cells)
        remove_budget = rng.randint(0, min(4, len(boundary_cells)))
        removed = 0
        for r, c in boundary_cells:
            if removed >= remove_budget:
                break
            trial = mask.copy()
            trial[r, c] = False
            if not np.any(trial[r]):
                continue
            if not np.any(trial[:, c]):
                continue
            if not _mask_connected(trial):
                continue
            mask = trial
            removed += 1

        if np.sum(mask) < 5:
            continue
        if not _mask_connected(mask):
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        return mask
    return None


def _find_arrow_placements(inner_mask: np.ndarray) -> dict[str, list[tuple[list[tuple[int, int]], tuple[int, int]]]]:
    h, w = inner_mask.shape
    placements = {name: [] for name in DIR_TO_DELTA}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not inner_mask[r, c]:
                continue
            if inner_mask[r - 1, c] and inner_mask[r + 1, c] and inner_mask[r, c + 1]:
                placements["right"].append(([(r - 1, c), (r + 1, c), (r, c + 1)], (r, c)))
            if inner_mask[r - 1, c] and inner_mask[r + 1, c] and inner_mask[r, c - 1]:
                placements["left"].append(([(r - 1, c), (r + 1, c), (r, c - 1)], (r, c)))
            if inner_mask[r, c - 1] and inner_mask[r, c + 1] and inner_mask[r - 1, c]:
                placements["up"].append(([(r, c - 1), (r, c + 1), (r - 1, c)], (r, c)))
            if inner_mask[r, c - 1] and inner_mask[r, c + 1] and inner_mask[r + 1, c]:
                placements["down"].append(([(r, c - 1), (r, c + 1), (r + 1, c)], (r, c)))
    return placements


def _build_object_spec(
    rng: random.Random,
    ring_color: int,
    arrow_color: int,
    direction: str,
    truncated: bool,
) -> dict | None:
    for _ in range(120):
        inner = _random_inner_mask(rng)
        if inner is None:
            continue
        placements = _find_arrow_placements(inner)
        if not placements[direction]:
            continue

        arrow_cells, center = rng.choice(placements[direction])
        padded_inner = np.zeros((inner.shape[0] + 2, inner.shape[1] + 2), dtype=bool)
        padded_inner[1:-1, 1:-1] = inner
        outer = _dilate8_padded(inner)
        ring = outer & ~padded_inner
        if len(_components_mask(ring, connectivity=8)) != 1:
            continue

        arrow_rel = np.array([(r + 1, c + 1) for r, c in arrow_cells], dtype=int)
        center_rel = (center[0] + 1, center[1] + 1)
        inner_rel = [tuple(map(int, p)) for p in np.argwhere(padded_inner)]
        arrow_set = {tuple(map(int, p)) for p in arrow_rel.tolist()}
        yellow_candidates = [
            cell for cell in inner_rel
            if cell not in arrow_set and cell != center_rel
        ]
        max_yellow = min(MAX_YELLOW_COUNT, len(yellow_candidates))
        min_yellow = 2 if truncated else 1
        if max_yellow < min_yellow:
            continue
        yellow_count = rng.randint(min_yellow, max_yellow)
        yellow_cells = rng.sample(yellow_candidates, yellow_count)

        return {
            "ring_color": ring_color,
            "arrow_color": arrow_color,
            "direction": direction,
            "truncated": truncated,
            "yellow_count": yellow_count,
            "ring_rel": np.argwhere(ring).astype(int),
            "arrow_rel": arrow_rel,
            "yellow_rel": np.array(yellow_cells, dtype=int),
            "box_h": int(ring.shape[0]),
            "box_w": int(ring.shape[1]),
        }
    return None


def _max_copies_fit(h: int, w: int, r0: int, c0: int, box_h: int, box_w: int, direction: str) -> int:
    if direction == "right":
        return int((w - (c0 + box_w)) // box_w)
    if direction == "left":
        return int(c0 // box_w)
    if direction == "down":
        return int((h - (r0 + box_h)) // box_h)
    return int(r0 // box_h)


def _strip_rect(r0: int, c0: int, box_h: int, box_w: int, direction: str, copies: int) -> tuple[int, int, int, int]:
    if direction == "right":
        return r0, c0, r0 + box_h - 1, c0 + box_w * (copies + 1) - 1
    if direction == "left":
        return r0, c0 - box_w * copies, r0 + box_h - 1, c0 + box_w - 1
    if direction == "down":
        return r0, c0, r0 + box_h * (copies + 1) - 1, c0 + box_w - 1
    return r0 - box_h * copies, c0, r0 + box_h - 1, c0 + box_w - 1


def _rect_conflicts(occupied: np.ndarray, rect: tuple[int, int, int, int], margin: int = 1) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return bool(np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _mark_rect(occupied: np.ndarray, rect: tuple[int, int, int, int], margin: int = 1) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_object(rng: random.Random, occupied: np.ndarray, spec: dict) -> dict | None:
    h, w = occupied.shape
    box_h = int(spec["box_h"])
    box_w = int(spec["box_w"])
    n = int(spec["yellow_count"])
    direction = spec["direction"]
    truncated = bool(spec["truncated"])

    for _ in range(300):
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(0, w - box_w)
        max_fit = _max_copies_fit(h, w, r0, c0, box_h, box_w, direction)
        if truncated:
            if not (1 <= max_fit < n):
                continue
        else:
            if max_fit < n:
                continue
        copies = min(n, max_fit)
        strip = _strip_rect(r0, c0, box_h, box_w, direction, copies)
        if _rect_conflicts(occupied, strip, margin=1):
            continue
        _mark_rect(occupied, strip, margin=1)
        placed = dict(spec)
        placed["top"] = r0
        placed["left"] = c0
        placed["copies"] = copies
        return placed
    return None


def _draw_object(grid: np.ndarray, spec: dict) -> None:
    r0 = int(spec["top"])
    c0 = int(spec["left"])
    ring_color = int(spec["ring_color"])
    arrow_color = int(spec["arrow_color"])
    ring_rel = spec["ring_rel"]
    arrow_rel = spec["arrow_rel"]
    yellow_rel = spec["yellow_rel"]

    grid[r0 + ring_rel[:, 0], c0 + ring_rel[:, 1]] = ring_color
    grid[r0 + arrow_rel[:, 0], c0 + arrow_rel[:, 1]] = arrow_color
    grid[r0 + yellow_rel[:, 0], c0 + yellow_rel[:, 1]] = YELLOW


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choices([1, 2, 3], weights=[2, 4, 3])[0]
    min_side = 12 + 4 * n_objects
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)

    palette = [c for c in range(10) if c != YELLOW]
    bg = rng.choice(palette)
    free = [c for c in range(10) if c not in {bg, YELLOW}]
    rng.shuffle(free)
    if len(free) < 2 * n_objects:
        return None

    directions = list(DIR_TO_DELTA)
    rng.shuffle(directions)
    occupied = np.zeros((h, w), dtype=bool)
    placed_specs: list[dict] = []

    for i in range(n_objects):
        direction = directions[i % 4] if n_objects >= 2 else rng.choice(directions)
        if n_objects >= 2 and rng.random() < 0.35:
            direction = rng.choice(directions)
        truncated = bool(rng.randint(0, 1))
        spec = _build_object_spec(
            rng,
            ring_color=int(free[i]),
            arrow_color=int(free[n_objects + i]),
            direction=direction,
            truncated=truncated,
        )
        if spec is None:
            return None
        placed = _place_object(rng, occupied, spec)
        if placed is None:
            return None
        placed_specs.append(placed)

    grid = np.full((h, w), bg, dtype=int)
    for spec in placed_specs:
        _draw_object(grid, spec)

    detected = _extract_rings(grid)
    if len(detected) != n_objects:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamping-ring input")
