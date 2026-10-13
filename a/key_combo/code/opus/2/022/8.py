from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PANEL_SIZE = 5
INNER_SIZE = 3
SPRITE_COLORS = [1, 2, 3, 6, 7]
GRID_H_RANGE = (16, 20)
GRID_W_RANGE = (16, 20)
OBJECT_COUNT_RANGE = (3, 5)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _find_monochrome_components(grid: np.ndarray, *, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            coords = []
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            box = (min(rows), min(cols), max(rows), max(cols))
            sprite = np.zeros((box[2] - box[0] + 1, box[3] - box[1] + 1), dtype=int)
            for rr, cc in coords:
                sprite[rr - box[0], cc - box[1]] = color
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "sprite": sprite,
                    "area": len(coords),
                }
            )
    return out


def _has_target_symmetry(sprite: np.ndarray) -> bool:
    mask = sprite != BG
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _find_panel(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    border_mask = np.ones((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    border_mask[1:-1, 1:-1] = False
    matches = []
    for r0 in range(h - PANEL_SIZE + 1):
        for c0 in range(w - PANEL_SIZE + 1):
            patch = g[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE]
            if not np.all(patch[border_mask] == GREY):
                continue
            inner = patch[1:-1, 1:-1]
            if np.any((inner != BG) & (inner != MAROON)):
                continue
            marker_coords = np.argwhere(inner == MAROON)
            if marker_coords.shape != (1, 2):
                continue
            pr, pc = map(int, marker_coords[0])
            matches.append((r0, c0, pr, pc))
    assert len(matches) == 1, f"expected one unique panel, got {len(matches)}"
    return matches[0]


def _anchor_start(size: int, sprite_size: int, index: int) -> int:
    if index == 0:
        return 0
    if index == 1:
        return (size - sprite_size) // 2
    if index == 2:
        return size - sprite_size
    raise ValueError(f"invalid anchor index: {index}")


def _render_output(shape: tuple[int, int], sprite: np.ndarray, pr: int, pc: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    h, w = sprite.shape
    top = _anchor_start(shape[0], h, pr)
    left = _anchor_start(shape[1], w, pc)
    mask = sprite != BG
    out[top : top + h, left : left + w][mask] = sprite[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the unique symmetric sprite and place it by the panel's position code.

    Input structure:
    - The grid has black background, several small monochrome sprites, and one fixed-corner 5x5 grey framed panel.
    - The panel has a grey 1-cell border, a black 3x3 interior, and exactly one maroon marker pixel inside that interior.
    - The sprites are separate 4-connected monochrome components outside the panel area.

    Rule:
    - Detect the unique 5x5 position panel and read the maroon marker location (pr, pc) inside its 3x3 interior.
    - Remove the panel, then find every remaining monochrome 4-connected sprite and crop each one to its minimal bounding box.
    - Select the unique sprite whose cropped shape is symmetric under vertical reflection, horizontal reflection, or 180-degree rotation.
    - Clear the whole grid to black and place that cropped sprite back using (pr, pc) as top/center/bottom and left/center/right alignment codes.

    Key insight:
    - The target identity and the target placement come from two independent clues: symmetry chooses which sprite to keep, while the panel marker chooses where its bbox must be anchored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panel_r0, panel_c0, pr, pc = _find_panel(g)
    work = g.copy()
    work[panel_r0 : panel_r0 + PANEL_SIZE, panel_c0 : panel_c0 + PANEL_SIZE] = BG

    components = _find_monochrome_components(work, bg=BG)
    assert components, "expected at least one sprite"

    symmetric = [comp for comp in components if _has_target_symmetry(comp["sprite"])]
    assert len(symmetric) == 1, f"expected one symmetric sprite, got {len(symmetric)}"
    return _render_output(g.shape, symmetric[0]["sprite"], pr, pc)


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return mask.shape[0], mask.shape[1], tuple(int(v) for v in mask.ravel())


def _build_shape_library() -> tuple[dict[str, list[dict]], list[dict]]:
    targets = {"vertical": [], "horizontal": [], "rot180": []}
    asym = []
    seen = set()
    for h in range(2, 5):
        for w in range(2, 5):
            n = h * w
            for area in range(4, min(8, n) + 1):
                for combo in itertools.combinations(range(n), area):
                    mask = np.zeros((h, w), dtype=np.uint8)
                    mask.flat[list(combo)] = 1
                    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                        continue
                    if not _is_connected(mask):
                        continue
                    sig = _mask_signature(mask)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    vertical = np.array_equal(mask, np.fliplr(mask))
                    horizontal = np.array_equal(mask, np.flipud(mask))
                    rot180 = np.array_equal(mask, np.rot90(mask, 2))
                    entry = {
                        "mask": mask.astype(int),
                        "shape": mask.shape,
                        "area": int(mask.sum()),
                        "sym": (vertical, horizontal, rot180),
                    }
                    if vertical and not horizontal and not rot180:
                        targets["vertical"].append(entry)
                    elif horizontal and not vertical and not rot180:
                        targets["horizontal"].append(entry)
                    elif rot180 and not vertical and not horizontal:
                        targets["rot180"].append(entry)
                    elif not (vertical or horizontal or rot180):
                        asym.append(entry)
    assert all(targets.values()), "shape library missing a symmetric class"
    assert asym, "shape library missing asymmetric distractors"
    return targets, asym


TARGET_POOLS, ASYM_SHAPES = _build_shape_library()


def _shape_to_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask.astype(bool)] = int(color)
    return sprite


def _reserve_rect(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, *, padding: int = 1) -> None:
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(blocked.shape[0], r0 + h + padding)
    cc1 = min(blocked.shape[1], c0 + w + padding)
    blocked[rr0:rr1, cc0:cc1] = True


def _reserve_sprite(blocked: np.ndarray, sprite: np.ndarray, top: int, left: int, *, padding: int = 1) -> None:
    h, w = sprite.shape
    for r, c in np.argwhere(sprite != BG):
        rr = top + int(r)
        cc = left + int(c)
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
                    blocked[nr, nc] = True
    _reserve_rect(blocked, top, left, h, w, padding=0)


def _can_place(blocked: np.ndarray, sprite: np.ndarray, top: int, left: int) -> bool:
    h, w = sprite.shape
    if top < 0 or left < 0 or top + h > blocked.shape[0] or left + w > blocked.shape[1]:
        return False
    mask = sprite != BG
    return not np.any(blocked[top : top + h, left : left + w] & mask)


def _paste(dst: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    mask = sprite != BG
    dst[top : top + h, left : left + w][mask] = sprite[mask]


def _draw_panel(grid: np.ndarray, pr: int, pc: int) -> tuple[int, int]:
    h, w = grid.shape
    r0 = h - PANEL_SIZE
    c0 = w - PANEL_SIZE
    grid[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE] = GREY
    grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = BG
    grid[r0 + 1 + pr, c0 + 1 + pc] = MAROON
    return r0, c0


def _adjust_size_for_center(size: int, sprite_size: int, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    if (size - sprite_size) % 2 == 1:
        return size
    if size < hi:
        return size + 1
    if size > lo:
        return size - 1
    return size


def _pick_canvas_shape(sprite_shape: tuple[int, int], pr: int, pc: int) -> tuple[int, int]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    if pr == 1 and random.random() < 0.6:
        h = _adjust_size_for_center(h, sprite_shape[0], GRID_H_RANGE)
    if pc == 1 and random.random() < 0.6:
        w = _adjust_size_for_center(w, sprite_shape[1], GRID_W_RANGE)
    return h, w


def _choose_distractor_shapes(target: dict, count: int) -> list[dict] | None:
    target_sig = _mask_signature(target["mask"])
    area = target["area"]
    shape = target["shape"]
    mode = random.choice(["area", "bbox", "either"])
    if mode == "area":
        primary = [s for s in ASYM_SHAPES if s["area"] == area]
    elif mode == "bbox":
        primary = [s for s in ASYM_SHAPES if s["shape"] == shape]
    else:
        primary = [s for s in ASYM_SHAPES if s["area"] == area or s["shape"] == shape]
    if not primary:
        return None

    chosen = [random.choice(primary)]
    used = {target_sig, _mask_signature(chosen[0]["mask"])}
    while len(chosen) < count:
        pool = [s for s in ASYM_SHAPES if _mask_signature(s["mask"]) not in used]
        if not pool:
            return None
        cand = random.choice(pool)
        used.add(_mask_signature(cand["mask"]))
        chosen.append(cand)
    return chosen


def _place_objects(
    shape: tuple[int, int],
    sprites: list[np.ndarray],
    *,
    target_index: int,
    forbidden_top_left: tuple[int, int],
) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    blocked = np.zeros(shape, dtype=bool)
    _reserve_rect(blocked, shape[0] - PANEL_SIZE, shape[1] - PANEL_SIZE, PANEL_SIZE, PANEL_SIZE, padding=1)

    order = list(range(len(sprites)))
    random.shuffle(order)
    for idx in order:
        sprite = sprites[idx]
        h, w = sprite.shape
        candidates = [(r, c) for r in range(shape[0] - h + 1) for c in range(shape[1] - w + 1)]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            if idx == target_index and (top, left) == forbidden_top_left:
                continue
            if not _can_place(blocked, sprite, top, left):
                continue
            _paste(grid, sprite, top, left)
            _reserve_sprite(blocked, sprite, top, left, padding=1)
            placed = True
            break
        if not placed:
            return None
    return grid


def _all_marker_outputs_distinct(sprite: np.ndarray, shape: tuple[int, int]) -> bool:
    sigs = set()
    for pr in range(3):
        for pc in range(3):
            out = _render_output(shape, sprite, pr, pc)
            sigs.add((out.shape, out.tobytes()))
    return len(sigs) == 9


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    panel_r0, panel_c0, pr, pc = _find_panel(grid)
    if (panel_r0, panel_c0) != (grid.shape[0] - PANEL_SIZE, grid.shape[1] - PANEL_SIZE):
        return False

    work = grid.copy()
    work[panel_r0 : panel_r0 + PANEL_SIZE, panel_c0 : panel_c0 + PANEL_SIZE] = BG
    components = _find_monochrome_components(work, bg=BG)
    if not (OBJECT_COUNT_RANGE[0] <= len(components) <= OBJECT_COUNT_RANGE[1]):
        return False
    if len({comp["color"] for comp in components}) != len(components):
        return False

    symmetric = [comp for comp in components if _has_target_symmetry(comp["sprite"])]
    if len(symmetric) != 1:
        return False
    target = symmetric[0]["sprite"]
    distractors = [comp["sprite"] for comp in components if comp is not symmetric[0]]

    if not _all_marker_outputs_distinct(target, grid.shape):
        return False
    if not any(d.shape == target.shape or int(np.count_nonzero(d)) == int(np.count_nonzero(target)) for d in distractors):
        return False
    if any(np.array_equal(_render_output(grid.shape, d, pr, pc), expected) for d in distractors):
        return False
    return True


def _generate_case() -> np.ndarray | None:
    sym_kind = random.choice(["vertical", "horizontal", "rot180"])
    target_shape = random.choice(TARGET_POOLS[sym_kind])
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    distractor_shapes = _choose_distractor_shapes(target_shape, object_count - 1)
    if distractor_shapes is None:
        return None

    pr = random.randint(0, 2)
    pc = random.randint(0, 2)
    grid_shape = _pick_canvas_shape(target_shape["shape"], pr, pc)

    colors = random.sample(SPRITE_COLORS, object_count)
    target_index = random.randrange(object_count)
    sprites = []
    for i in range(object_count):
        shape = target_shape if i == target_index else distractor_shapes.pop()
        sprites.append(_shape_to_sprite(shape["mask"], colors[i]))

    expected = _render_output(grid_shape, sprites[target_index], pr, pc)
    target_bbox = _bbox(sprites[target_index] != BG)
    assert target_bbox is not None
    target_crop = _crop_bbox(sprites[target_index], target_bbox)
    forbidden_top_left = (
        _anchor_start(grid_shape[0], target_crop.shape[0], pr),
        _anchor_start(grid_shape[1], target_crop.shape[1], pc),
    )

    grid = _place_objects(grid_shape, sprites, target_index=target_index, forbidden_top_left=forbidden_top_left)
    if grid is None:
        return None
    _draw_panel(grid, pr, pc)

    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
