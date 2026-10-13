from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 60
MASK_ATTEMPTS = 80
SHADOW_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or visited[r, c]:
                continue
            coords: list[tuple[int, int]] = []
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return components


def _pair_candidates(entity: dict, shadow: dict) -> list[dict]:
    entity_set = entity["coord_set"]
    diffs: set[tuple[int, int]] = set()
    for sr, sc in shadow["coords"]:
        for er, ec in entity["coords"]:
            dr = sr - er
            dc = sc - ec
            if (dr > 0 and dc == 0) or (dr == 0 and dc > 0):
                diffs.add((dr, dc))

    candidates = []
    for dr, dc in diffs:
        shifted_back = {(r - dr, c - dc) for r, c in shadow["coords"]}
        if not shifted_back < entity_set:
            continue
        missing = sorted(entity_set - shifted_back)
        if not missing:
            continue
        candidates.append(
            {
                "dr": dr,
                "dc": dc,
                "missing": missing,
            }
        )
    return candidates


def _match_components(components: list[dict]) -> list[dict]:
    n = len(components)
    assert n % 2 == 0 and n > 0

    pairs: list[dict] = []
    by_component = [[] for _ in range(n)]
    for entity_idx, entity in enumerate(components):
        for shadow_idx, shadow in enumerate(components):
            if entity_idx == shadow_idx or shadow["area"] >= entity["area"]:
                continue
            for info in _pair_candidates(entity, shadow):
                pair = {
                    "entity": entity_idx,
                    "shadow": shadow_idx,
                    "dr": info["dr"],
                    "dc": info["dc"],
                    "missing": info["missing"],
                }
                pair_id = len(pairs)
                pairs.append(pair)
                by_component[entity_idx].append(pair_id)
                by_component[shadow_idx].append(pair_id)

    solutions: list[list[int]] = []
    full_mask = (1 << n) - 1

    def dfs(used_mask: int, chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if used_mask == full_mask:
            solutions.append(chosen.copy())
            return
        first_unused = next(i for i in range(n) if not (used_mask & (1 << i)))
        for pair_id in by_component[first_unused]:
            pair = pairs[pair_id]
            e_idx = pair["entity"]
            s_idx = pair["shadow"]
            if used_mask & (1 << e_idx) or used_mask & (1 << s_idx):
                continue
            chosen.append(pair_id)
            dfs(used_mask | (1 << e_idx) | (1 << s_idx), chosen)
            chosen.pop()

    dfs(0, [])
    assert len(solutions) == 1, "entity-shadow pairing must be unique"
    return [pairs[pair_id] for pair_id in solutions[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """Highlight each entity with the cells missing from its paired translated shadow.

    Input structure:
    - The background is black.
    - The foreground consists of several pairs of monochrome objects: a 4-connected entity and a shadow in another color.
    - Each shadow sits directly below or directly right of its entity and is an incomplete copy of the entity shape.

    Rule:
    - Pair every entity with the shadow whose occupied cells match a translated strict subset of the entity.
    - For each pair, compare the entity cells to the translated shadow cells and find which entity positions are absent from the shadow.
    - Recolor exactly those missing positions on the entity with the shadow color, and leave the shadow unchanged.

    Key insight:
    - The shadow acts as a reverse mask: the places where the shadow is missing are precisely the entity cells that must be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    matched_pairs = _match_components(components)

    out = g.copy()
    for pair in matched_pairs:
        shadow_color = int(components[pair["shadow"]]["color"])
        for r, c in pair["missing"]:
            out[r, c] = shadow_color
    return out


def _random_shape_mask() -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        box_h = random.randint(3, 6)
        box_w = random.randint(3, 6)
        max_area = box_h * box_w
        target_area = random.randint(4, max_area - (1 if max_area > 4 else 0))

        mask = np.zeros((box_h, box_w), dtype=bool)
        start = (random.randrange(box_h), random.randrange(box_w))
        mask[start] = True

        while int(mask.sum()) < target_area:
            frontier = set()
            for r, c in zip(*np.where(mask), strict=False):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < box_h and 0 <= nc < box_w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) != target_area:
            continue
        cropped = _crop_mask(mask)
        if int(cropped.sum()) < 4:
            continue
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        return cropped
    return None


def _shadow_from_entity(entity_mask: np.ndarray) -> np.ndarray | None:
    area = int(entity_mask.sum())
    min_shadow_area = max(3, area // 2)
    max_remove = min(4, area - min_shadow_area)
    if max_remove < 1:
        return None

    for _ in range(SHADOW_ATTEMPTS):
        shadow = entity_mask.copy()
        remove_goal = random.randint(1, max_remove)
        removed = 0
        while removed < remove_goal:
            removable = []
            for r, c in zip(*np.where(shadow), strict=False):
                if int(shadow.sum()) <= min_shadow_area:
                    break
                trial = shadow.copy()
                trial[int(r), int(c)] = False
                if _mask_connected(trial):
                    removable.append((int(r), int(c)))
            if not removable:
                break
            rr, cc = random.choice(removable)
            shadow[rr, cc] = False
            removed += 1

        if removed >= 1 and _mask_connected(shadow) and int(shadow.sum()) < area:
            return shadow
    return None


def _build_pair_patches(
    entity_mask: np.ndarray,
    shadow_mask: np.ndarray,
    entity_color: int,
    shadow_color: int,
    direction: str,
    gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = entity_mask.shape
    if direction == "down":
        rows, cols = 2 * h + gap, w
        entity_top, entity_left = 0, 0
        shadow_top, shadow_left = h + gap, 0
    else:
        rows, cols = h, 2 * w + gap
        entity_top, entity_left = 0, 0
        shadow_top, shadow_left = 0, w + gap

    inp = np.full((rows, cols), BG, dtype=int)
    out = np.full((rows, cols), BG, dtype=int)

    entity_patch = np.full((h, w), BG, dtype=int)
    entity_patch[entity_mask] = entity_color
    highlight_mask = entity_mask & ~shadow_mask
    highlighted_entity = entity_patch.copy()
    highlighted_entity[highlight_mask] = shadow_color

    shadow_patch = np.full((h, w), BG, dtype=int)
    shadow_patch[shadow_mask] = shadow_color

    inp[entity_top : entity_top + h, entity_left : entity_left + w] = entity_patch
    out[entity_top : entity_top + h, entity_left : entity_left + w] = highlighted_entity
    inp[shadow_top : shadow_top + h, shadow_left : shadow_left + w] = shadow_patch
    out[shadow_top : shadow_top + h, shadow_left : shadow_left + w] = shadow_patch
    return inp, out


def _choose_canvas(patches: list[np.ndarray]) -> tuple[int, int] | None:
    heights = [patch.shape[0] for patch in patches]
    widths = [patch.shape[1] for patch in patches]
    footprint = sum((h + 2) * (w + 2) for h, w in zip(heights, widths, strict=False))
    base = int(np.ceil(np.sqrt(footprint)))

    min_rows = max(12, max(heights) + 2, base - 2)
    min_cols = max(12, max(widths) + 2, base - 2)
    if min_rows > 30 or min_cols > 30:
        return None

    rows = random.randint(min_rows, 30)
    cols = random.randint(min_cols, 30)
    return rows, cols


def _reserve(occupied: np.ndarray, top: int, left: int, height: int, width: int, padding: int = 1) -> None:
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(occupied.shape[0], top + height + padding)
    c1 = min(occupied.shape[1], left + width + padding)
    occupied[r0:r1, c0:c1] = True


def _find_position(occupied: np.ndarray, height: int, width: int, padding: int = 1) -> tuple[int, int] | None:
    valid = []
    h, w = occupied.shape
    for top in range(h - height + 1):
        for left in range(w - width + 1):
            r0 = max(0, top - padding)
            c0 = max(0, left - padding)
            r1 = min(h, top + height + padding)
            c1 = min(w, left + width + padding)
            if not occupied[r0:r1, c0:c1].any():
                valid.append((top, left))
    if not valid:
        return None
    return random.choice(valid)


def _render_scene(
    rows: int,
    cols: int,
    inputs: list[np.ndarray],
    outputs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    order = sorted(range(len(inputs)), key=lambda idx: inputs[idx].shape[0] * inputs[idx].shape[1], reverse=True)
    occupied = np.zeros((rows, cols), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(inputs)

    for idx in order:
        patch = inputs[idx]
        pos = _find_position(occupied, patch.shape[0], patch.shape[1], padding=1)
        if pos is None:
            return None
        positions[idx] = pos
        _reserve(occupied, pos[0], pos[1], patch.shape[0], patch.shape[1], padding=1)

    inp = np.full((rows, cols), BG, dtype=int)
    out = np.full((rows, cols), BG, dtype=int)
    for idx, pos in enumerate(positions):
        assert pos is not None
        top, left = pos
        patch_in = inputs[idx]
        patch_out = outputs[idx]
        h, w = patch_in.shape
        inp_slice = inp[top : top + h, left : left + w]
        out_slice = out[top : top + h, left : left + w]
        inp_slice[patch_in != BG] = patch_in[patch_in != BG]
        out_slice[patch_out != BG] = patch_out[patch_out != BG]
        inp[top : top + h, left : left + w] = inp_slice
        out[top : top + h, left : left + w] = out_slice
    return inp, out


def _oracle_ok(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    direction = random.choice(("down", "right"))
    gap = random.randint(1, 3)
    colors = random.sample(range(1, 10), 2 * pair_count)

    input_patches: list[np.ndarray] = []
    output_patches: list[np.ndarray] = []
    for pair_idx in range(pair_count):
        entity_mask = _random_shape_mask()
        if entity_mask is None:
            return None
        shadow_mask = _shadow_from_entity(entity_mask)
        if shadow_mask is None:
            return None
        inp_patch, out_patch = _build_pair_patches(
            entity_mask,
            shadow_mask,
            colors[2 * pair_idx],
            colors[2 * pair_idx + 1],
            direction,
            gap,
        )
        input_patches.append(inp_patch)
        output_patches.append(out_patch)

    for _ in range(PLACEMENT_ATTEMPTS):
        canvas = _choose_canvas(input_patches)
        if canvas is None:
            return None
        rendered = _render_scene(canvas[0], canvas[1], input_patches, output_patches)
        if rendered is None:
            continue
        inp, expected = rendered
        if _oracle_ok(inp, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
