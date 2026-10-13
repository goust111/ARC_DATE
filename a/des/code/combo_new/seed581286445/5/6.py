from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np

BG_COLOR = 0
PANEL_COLOR = 5
KEY_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATE_MAX_RETRIES = 400


def _border_background(grid: np.ndarray) -> int:
    if grid.shape[0] == 1 or grid.shape[1] == 1:
        vals, cnts = np.unique(grid, return_counts=True)
        return int(vals[int(cnts.argmax())])
    border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(rot)
    return out


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    keys = []
    for k in range(4):
        rot = np.rot90(mask, k)
        keys.append((int(rot.shape[0]), int(rot.shape[1]), rot.tobytes()))
    return min(keys)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[top + rows, left + cols] = int(color)


def _find_panel(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best = None
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.isin(sub, [bg, color]).all():
            continue
        if not np.any(sub == bg):
            continue
        if not (
            np.all(sub[0] == color)
            and np.all(sub[-1] == color)
            and np.all(sub[:, 0] == color)
            and np.all(sub[:, -1] == color)
        ):
            continue
        score = (int(sub.shape[0] * sub.shape[1]), int(np.count_nonzero(sub == color)))
        if best is None or score > best[0]:
            best = (score, color, (int(r0), int(c0), int(r1), int(c1)))
    assert best is not None, "expected a rectangular panel with internal holes"
    return best[1], best[2]


def _extract_holes(
    grid: np.ndarray, panel_bbox: tuple[int, int, int, int], bg: int
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = panel_bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict[str, object]] = []
    for coords in _components(sub == bg):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == sub.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == sub.shape[1] - 1)
        ):
            continue
        mask, local_bbox = _coords_to_mask(coords)
        abs_coords = coords + np.array([r0, c0], dtype=int)
        holes.append(
            {
                "mask": mask,
                "coords": abs_coords,
                "bbox": (
                    int(r0 + local_bbox[0]),
                    int(c0 + local_bbox[1]),
                    int(r0 + local_bbox[2]),
                    int(c0 + local_bbox[3]),
                ),
                "area": int(mask.sum()),
            }
        )
    holes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["area"]))  # type: ignore[index]
    assert holes, "panel must contain at least one hole"
    return holes


def _extract_keys(grid: np.ndarray, bg: int, panel_color: int) -> list[dict[str, object]]:
    keys: list[dict[str, object]] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color in {bg, panel_color}:
            continue
        for coords in _components(grid == color):
            mask, bbox = _coords_to_mask(coords)
            keys.append(
                {
                    "color": color,
                    "mask": mask,
                    "coords": coords,
                    "bbox": bbox,
                    "area": int(mask.sum()),
                }
            )
    keys.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))  # type: ignore[index]
    assert keys, "expected at least one key outside the panel"
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the identically shaped lockhole in the grey panel, then transfer the key color into that hole.

    Input structure:
    - The background is black, and the grid border is background.
    - One large monochrome rectangular region acts as the grey lock panel; inside its bbox, only panel color and background appear.
    - Background components fully enclosed by that rectangle are the lockholes.
    - Outside the panel, each remaining monochrome 4-connected component is a key.

    Rule:
    - For every key, compare its shape against each lockhole under 0°, 90°, 180°, and 270° rotations.
    - Find the unique lockhole whose shape matches exactly up to rotation.
    - Fill that lockhole with the key's color and erase the original key cells back to background.
    - Leave the rest of the panel and the rest of the background unchanged.

    Key insight:
    - The keys do not move geometrically into the panel; only their shape identity, modulo right-angle rotation, determines which enclosed background cavity gets recolored.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_background(g)
    panel_color, panel_bbox = _find_panel(g, bg)
    holes = _extract_holes(g, panel_bbox, bg)
    keys = _extract_keys(g, bg, panel_color)
    assert len(keys) == len(holes), "keys and holes must pair one-to-one"

    out = g.copy()
    unmatched = set(range(len(holes)))
    for key in keys:
        matches = []
        key_variants = _rotation_variants(key["mask"])  # type: ignore[arg-type]
        for hole_idx in sorted(unmatched):
            hole = holes[hole_idx]
            if key["area"] != hole["area"]:
                continue
            hole_mask = hole["mask"]
            if any(
                rot.shape == hole_mask.shape and np.array_equal(rot, hole_mask)
                for rot in key_variants
            ):
                matches.append(hole_idx)
        assert len(matches) == 1, "every key must match exactly one hole"

        hole = holes[matches[0]]
        unmatched.remove(matches[0])
        key_coords = key["coords"]
        hole_coords = hole["coords"]
        out[key_coords[:, 0], key_coords[:, 1]] = bg
        out[hole_coords[:, 0], hole_coords[:, 1]] = int(key["color"])

    assert not unmatched, "every hole must be filled"
    return out


def _random_polyomino(rng: random.Random) -> np.ndarray | None:
    area = rng.randint(3, 6)
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        seen = set()
        for r, c in cells:
            for dr, dc in DIR4:
                cand = (r + dr, c + dc)
                if cand in cells or cand in seen:
                    continue
                seen.add(cand)
                frontier.append(cand)
        weights = []
        for r, c in frontier:
            adj = sum((r + dr, c + dc) in cells for dr, dc in DIR4)
            weights.append(adj * adj)
        cells.add(rng.choices(frontier, weights=weights, k=1)[0])

    mask, _ = _coords_to_mask(np.array(sorted(cells), dtype=int))
    if max(mask.shape) > 4:
        return None
    if len(_rotation_variants(mask)) != 4:
        return None
    return mask


def _choose_base_shapes(count: int, rng: random.Random) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for _ in range(500):
        if len(shapes) == count:
            return shapes
        mask = _random_polyomino(rng)
        if mask is None:
            continue
        key = _shape_key(mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
    return None


def _place_boxes(
    canvas_shape: tuple[int, int],
    box_shapes: list[tuple[int, int]],
    rng: random.Random,
    occupied: np.ndarray | None = None,
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    occ = np.zeros((h, w), dtype=bool) if occupied is None else occupied.copy()
    positions: list[tuple[int, int] | None] = [None] * len(box_shapes)
    order = sorted(range(len(box_shapes)), key=lambda i: box_shapes[i][0] * box_shapes[i][1], reverse=True)
    for idx in order:
        bh, bw = box_shapes[idx]
        candidates = []
        for r0 in range(1, h - bh):
            for c0 in range(1, w - bw):
                if occ[r0 - 1 : r0 + bh + 1, c0 - 1 : c0 + bw + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        pos = rng.choice(candidates)
        positions[idx] = pos
        r0, c0 = pos
        occ[r0 - 1 : r0 + bh + 1, c0 - 1 : c0 + bw + 1] = True
    return [pos for pos in positions if pos is not None]


def _choose_panel_size(hole_masks: list[np.ndarray], rng: random.Random) -> tuple[int, int] | None:
    need = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in hole_masks)
    min_side = int(np.ceil(np.sqrt(need))) + 2
    min_h = max(8, max(mask.shape[0] for mask in hole_masks) + 4, min_side)
    min_w = max(8, max(mask.shape[1] for mask in hole_masks) + 4, min_side)
    max_h = min(20, min_h + 6)
    max_w = min(20, min_w + 6)
    if min_h > max_h or min_w > max_w:
        return None
    return rng.randint(min_h, max_h), rng.randint(min_w, max_w)


def _choose_grid_size(
    panel_shape: tuple[int, int], key_masks: list[np.ndarray], rng: random.Random
) -> tuple[int, int] | None:
    panel_h, panel_w = panel_shape
    min_h = max(14, panel_h + 4, max(mask.shape[0] for mask in key_masks) + 4)
    min_w = max(14, panel_w + 4, max(mask.shape[1] for mask in key_masks) + 4)
    max_h = min(30, panel_h + 8 + 2 * len(key_masks))
    max_w = min(30, panel_w + 8 + 2 * len(key_masks))
    if min_h > max_h or min_w > max_w:
        return None
    return rng.randint(min_h, max_h), rng.randint(min_w, max_w)


def _build_expected_output(
    inp: np.ndarray,
    panel_pos: tuple[int, int],
    hole_masks: list[np.ndarray],
    hole_positions: list[tuple[int, int]],
    key_masks: list[np.ndarray],
    key_positions: list[tuple[int, int]],
    key_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    panel_r, panel_c = panel_pos
    for mask, (r0, c0) in zip(key_masks, key_positions):
        rows, cols = np.nonzero(mask)
        out[r0 + rows, c0 + cols] = BG_COLOR
    for mask, (r0, c0), color in zip(hole_masks, hole_positions, key_colors):
        _paint_mask(out, mask, panel_r + r0, panel_c + c0, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_keys = rng.choice([1, 2, 3, 4])
    base_shapes = _choose_base_shapes(n_keys, rng)
    if base_shapes is None:
        return None

    hole_masks = [np.rot90(mask, rng.randrange(4)) for mask in base_shapes]
    key_masks = [np.rot90(mask, rng.randrange(4)) for mask in base_shapes]
    panel_shape = _choose_panel_size(hole_masks, rng)
    if panel_shape is None:
        return None
    panel_h, panel_w = panel_shape

    hole_positions = _place_boxes(panel_shape, [tuple(map(int, mask.shape)) for mask in hole_masks], rng)
    if hole_positions is None:
        return None

    grid_shape = _choose_grid_size(panel_shape, key_masks, rng)
    if grid_shape is None:
        return None
    h, w = grid_shape

    panel_candidates = [
        (r0, c0)
        for r0 in range(1, h - panel_h)
        for c0 in range(1, w - panel_w)
    ]
    rng.shuffle(panel_candidates)

    scene = None
    for panel_pos in panel_candidates:
        occupied = np.zeros((h, w), dtype=bool)
        pr0, pc0 = panel_pos
        occupied[pr0 - 1 : pr0 + panel_h + 1, pc0 - 1 : pc0 + panel_w + 1] = True
        key_positions = _place_boxes(
            grid_shape,
            [tuple(map(int, mask.shape)) for mask in key_masks],
            rng,
            occupied=occupied,
        )
        if key_positions is not None:
            scene = (panel_pos, key_positions)
            break
    if scene is None:
        return None

    panel_pos, key_positions = scene
    key_colors = rng.sample(KEY_COLOR_POOL, n_keys)

    inp = np.full((h, w), BG_COLOR, dtype=int)
    pr0, pc0 = panel_pos
    inp[pr0 : pr0 + panel_h, pc0 : pc0 + panel_w] = PANEL_COLOR
    for mask, (r0, c0) in zip(hole_masks, hole_positions):
        _paint_mask(inp, mask, pr0 + r0, pc0 + c0, BG_COLOR)
    for mask, (r0, c0), color in zip(key_masks, key_positions, key_colors):
        _paint_mask(inp, mask, r0, c0, color)

    expected = _build_expected_output(
        inp,
        panel_pos,
        hole_masks,
        hole_positions,
        key_masks,
        key_positions,
        key_colors,
    )

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    if not np.all(border == BG_COLOR):
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
