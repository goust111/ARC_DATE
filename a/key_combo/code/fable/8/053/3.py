from __future__ import annotations

import random
from collections import Counter, deque
from itertools import combinations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, mask.shape)), tuple(map(int, mask.astype(np.uint8).ravel()))


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((factor, factor), dtype=np.uint8)).astype(bool)


def _find_mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            mask = _crop_mask_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "mask": mask,
                    "sig": _mask_signature(mask),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _enumerate_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    fg_colors = sorted(set(map(int, g.ravel())) - {bg})
    comps = _find_mono_components(g, bg)
    comp_counts = Counter(int(comp["color"]) for comp in comps)
    residue_color_candidates = [color for color in fg_colors if comp_counts.get(int(color), 0) >= 2]
    if not residue_color_candidates:
        residue_color_candidates = fg_colors
    candidates: list[dict] = []

    for residue_color in residue_color_candidates:
        residue_coords = np.argwhere(g == residue_color)
        if residue_coords.size == 0:
            continue
        residue_rows = residue_coords[:, 0]
        residue_cols = residue_coords[:, 1]
        other_fg_mask = (g != bg) & (g != residue_color)
        sprite_comps = [comp for comp in comps if int(comp["color"]) != residue_color]
        if not sprite_comps:
            continue

        for sprite_index, comp in enumerate(sprite_comps):
            sprite_mask = comp["mask"]
            sprite_sig = comp["sig"]
            for scale in (2, 3):
                scaled = _scale_mask(sprite_mask, scale)
                sh, sw = map(int, scaled.shape)
                if sh > h or sw > w:
                    continue

                scaled_coords = np.argwhere(scaled)
                top_lo = max(0, int(residue_rows.max()) - sh + 1)
                top_hi = min(h - sh, int(residue_rows.min()))
                left_lo = max(0, int(residue_cols.max()) - sw + 1)
                left_hi = min(w - sw, int(residue_cols.min()))
                if top_lo > top_hi or left_lo > left_hi:
                    continue

                for top in range(top_lo, top_hi + 1):
                    local_rows = residue_rows - top
                    if np.any(local_rows < 0) or np.any(local_rows >= sh):
                        continue
                    for left in range(left_lo, left_hi + 1):
                        local_cols = residue_cols - left
                        if np.any(local_cols < 0) or np.any(local_cols >= sw):
                            continue
                        if not np.all(scaled[local_rows, local_cols]):
                            continue

                        rr = scaled_coords[:, 0] + top
                        cc = scaled_coords[:, 1] + left
                        if np.any(other_fg_mask[rr, cc]):
                            continue

                        block_hits = {
                            (int((r - top) // scale), int((c - left) // scale))
                            for r, c in residue_coords
                        }
                        candidates.append(
                            {
                                "residue_color": residue_color,
                                "sprite_sig": sprite_sig,
                                "sprite_mask": sprite_mask,
                                "sprite_index": sprite_index,
                                "scale": scale,
                                "top": int(top),
                                "left": int(left),
                                "scaled_coords": scaled_coords,
                                "scaled_shape": (sh, sw),
                                "residue_coords": residue_coords.copy(),
                                "block_hits": block_hits,
                            }
                        )
    return candidates


def _candidate_key(candidate: dict) -> tuple:
    return (
        int(candidate["residue_color"]),
        candidate["sprite_sig"],
        int(candidate["scale"]),
        int(candidate["top"]),
        int(candidate["left"]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the unique scaled sprite whose scattered residue pixels already mark part of its footprint.

    Input structure:
    - A dominant background contains several small monochromatic sprites with pairwise different shapes.
    - One extra foreground color appears only as scattered residue pixels, separate from the sprites.
    - Exactly one sprite can be enlarged by factor 2 or 3 and translated so every residue pixel lies inside the enlarged mask while the full enlarged sprite avoids all other foreground.

    Rule:
    - Treat each foreground color as a possible residue color.
    - For every remaining monochromatic sprite, test scale factors 2 and 3 and every in-bounds translation.
    - Keep only combinations whose enlarged mask covers all residue pixels of that color and does not overlap any other foreground; the puzzle guarantees exactly one valid combination.
    - Paint the complete enlarged mask in the residue color, leaving background and the original small sprites unchanged.

    Key insight:
    - The residue pixels simultaneously identify the source sprite, the enlargement factor, and the placement, so matching must happen on the enlarged mask rather than at the sprite's original scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    candidates = _enumerate_candidates(g)
    assert len(candidates) == 1, f"expected exactly one valid completion, found {len(candidates)}"
    winner = candidates[0]

    out = g.copy()
    rr = winner["scaled_coords"][:, 0] + int(winner["top"])
    cc = winner["scaled_coords"][:, 1] + int(winner["left"])
    out[rr, cc] = int(winner["residue_color"])
    return out


COLOR_IDS = tuple(range(10))
GRID_RANGE = (18, 28)
SPRITE_COUNT_RANGE = (3, 5)
SPRITE_BBOX_RANGE = (2, 4)
SPRITE_AREA_RANGE = (4, 7)
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 500


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(*SPRITE_BBOX_RANGE)
        w = random.randint(*SPRITE_BBOX_RANGE)
        max_area = min(h * w, SPRITE_AREA_RANGE[1])
        min_area = min(max_area, SPRITE_AREA_RANGE[0])
        area = random.randint(min_area, max_area)

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for nbr in _neighbors8(*start):
            if 0 <= nbr[0] < h and 0 <= nbr[1] < w:
                frontier.add(nbr)

        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nbr in _neighbors8(*cell):
                if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr not in cells:
                    frontier.add(nbr)

        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        mask = _crop_mask_from_coords(coords)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if len(set(coords[:, 0])) < 2 or len(set(coords[:, 1])) < 2:
            continue
        return mask
    raise RuntimeError("failed to build a connected sprite")


def _choose_sprite_set() -> tuple[list[np.ndarray], int]:
    for _ in range(200):
        count = random.randint(*SPRITE_COUNT_RANGE)
        masks: list[np.ndarray] = []
        sigs: set[tuple] = set()
        while len(masks) < count:
            mask = _random_connected_mask()
            sig = _mask_signature(mask)
            if sig in sigs:
                continue
            sigs.add(sig)
            masks.append(mask)

        candidates = []
        for i, mask_i in enumerate(masks):
            area_i = int(mask_i.sum())
            shape_i = tuple(map(int, mask_i.shape))
            if any(
                i != j
                and (int(masks[j].sum()) == area_i or tuple(map(int, masks[j].shape)) == shape_i)
                for j in range(len(masks))
            ):
                candidates.append(i)
        if candidates:
            return masks, random.choice(candidates)
    raise RuntimeError("failed to choose non-degenerate sprite set")


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors8(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _place_mask(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    *,
    prefer_edge: bool = False,
) -> tuple[int, int] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    coords = np.argwhere(mask)

    edge_positions = []
    inner_positions = []
    for top in range(h - mh + 1):
        for left in range(w - mw + 1):
            rr = coords[:, 0] + top
            cc = coords[:, 1] + left
            if np.any(blocked[rr, cc]):
                continue
            touches_edge = top == 0 or left == 0 or top + mh == h or left + mw == w
            if touches_edge:
                edge_positions.append((top, left, rr, cc))
            else:
                inner_positions.append((top, left, rr, cc))

    choices = edge_positions if prefer_edge and edge_positions else inner_positions + edge_positions
    if not choices:
        return None
    top, left, rr, cc = random.choice(choices)
    grid[rr, cc] = int(color)
    halo = np.zeros_like(blocked)
    halo[rr, cc] = True
    blocked[:, :] |= _dilate8(halo)
    return int(top), int(left)


def _sample_block_subset(scale: int) -> set[tuple[int, int]]:
    cells = [(r, c) for r in range(scale) for c in range(scale)]
    if scale == 2:
        size = random.choices([1, 2, 3, 4], weights=[4, 4, 3, 1])[0]
    else:
        size = random.choices([1, 2, 3, 4, 5, 6, 9], weights=[3, 4, 5, 4, 3, 2, 1])[0]
    return set(random.sample(cells, size))


def _sample_residue_coords(
    sprite_mask: np.ndarray,
    scale: int,
    style: str,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(sprite_mask)]
    full_scaled = set(map(tuple, np.argwhere(_scale_mask(sprite_mask, scale))))

    if style == "near_full":
        residue = set(full_scaled)
        max_remove = min(len(residue) - 1, max(2, len(residue) // 4))
        remove_count = random.randint(1, max_remove)
        for cell in random.sample(tuple(residue), remove_count):
            residue.remove(cell)
        return residue, set(cells)

    if style == "structured":
        block_count = random.randint(max(2, min(3, len(cells))), len(cells))
    elif style == "sparse":
        block_count = random.randint(2, min(len(cells), 3))
    else:
        raise ValueError(f"unknown residue style: {style}")

    chosen_blocks = set(random.sample(cells, block_count))
    residue: set[tuple[int, int]] = set()
    for br, bc in chosen_blocks:
        for dr, dc in _sample_block_subset(scale):
            residue.add((br * scale + dr, bc * scale + dc))

    if residue == full_scaled:
        residue.remove(random.choice(tuple(residue)))
    return residue, chosen_blocks


def _has_essential_residue(grid: np.ndarray, expected_key: tuple) -> bool:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    for r, c in np.argwhere(g == expected_key[0]):
        test = g.copy()
        test[int(r), int(c)] = bg
        cands = _enumerate_candidates(test)
        if len(cands) != 1 or _candidate_key(cands[0]) != expected_key:
            return True
    return False


def _semantic_ok(grid: np.ndarray, expected: dict) -> bool:
    bg = _get_bg(grid)
    residue_components = [
        comp for comp in _find_mono_components(grid, bg) if int(comp["color"]) == int(expected["residue_color"])
    ]
    if len(residue_components) < 2:
        return False

    candidates = _enumerate_candidates(grid)
    if len(candidates) != 1:
        return False
    winner = candidates[0]
    if _candidate_key(winner) != _candidate_key(expected):
        return False

    residue_coords = winner["residue_coords"]
    if residue_coords.shape[0] < 3:
        return False
    hit_rows = {br for br, _bc in winner["block_hits"]}
    hit_cols = {bc for _br, bc in winner["block_hits"]}
    if len(winner["block_hits"]) < 2 or len(hit_rows) < 2 or len(hit_cols) < 2:
        return False
    density = residue_coords.shape[0] / float(expected["full_area"])
    if density < 0.75 and not _has_essential_residue(grid, _candidate_key(expected)):
        return False
    return True


def _build_attempt() -> np.ndarray | None:
    masks, target_index = _choose_sprite_set()
    scale = random.choice((2, 3))
    target_mask = masks[target_index]
    scaled_target = _scale_mask(target_mask, scale)

    min_h = max(18, scaled_target.shape[0] + 6)
    min_w = max(18, scaled_target.shape[1] + 6)
    if min_h > GRID_RANGE[1] or min_w > GRID_RANGE[1]:
        return None
    h = random.randint(min_h, GRID_RANGE[1])
    w = random.randint(min_w, GRID_RANGE[1])

    color_count = len(masks) + 2
    bg, residue_color, *sprite_colors = random.sample(COLOR_IDS, color_count)
    grid_out = np.full((h, w), int(bg), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    prefer_target_edge = random.random() < 0.3
    target_pos = _place_mask(grid_out, blocked, scaled_target, residue_color, prefer_edge=prefer_target_edge)
    if target_pos is None:
        return None
    target_top, target_left = target_pos

    sprite_order = list(range(len(masks)))
    random.shuffle(sprite_order)
    placements: dict[int, tuple[int, int]] = {}
    for idx in sprite_order:
        pos = _place_mask(grid_out, blocked, masks[idx], sprite_colors[idx], prefer_edge=False)
        if pos is None:
            return None
        placements[idx] = pos

    scaled_coords = np.argwhere(scaled_target)
    rr = scaled_coords[:, 0] + target_top
    cc = scaled_coords[:, 1] + target_left

    styles = random.choices(["near_full", "structured", "sparse"], weights=[5, 3, 2], k=160)
    for style in styles:
        input_grid = grid_out.copy()
        input_grid[rr, cc] = int(bg)

        residue_local, chosen_blocks = _sample_residue_coords(target_mask, scale, style)
        residue_abs = {(target_top + r, target_left + c) for r, c in residue_local}
        for r, c in residue_abs:
            input_grid[int(r), int(c)] = int(residue_color)

        expected = {
            "residue_color": int(residue_color),
            "sprite_sig": _mask_signature(target_mask),
            "sprite_mask": target_mask,
            "scale": int(scale),
            "top": int(target_top),
            "left": int(target_left),
            "scaled_coords": np.argwhere(scaled_target),
            "residue_coords": np.array(sorted(residue_abs), dtype=int),
            "block_hits": chosen_blocks,
            "full_area": int(scaled_target.sum()),
        }

        if not _semantic_ok(input_grid, expected):
            continue
        if not np.array_equal(main(input_grid), grid_out):
            continue
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residue-completion puzzle")
