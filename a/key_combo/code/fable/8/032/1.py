from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
GREY = 5
NON_GREY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) != bg:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            rel = coords - np.array([[r0, c0]], dtype=int)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rel[:, 0], rel[:, 1]] = True
            colors = [int(grid[rr, cc]) for rr, cc in cells]
            objects.append(
                {
                    "coords": coords,
                    "rel_coords": [tuple(map(int, xy)) for xy in rel.tolist()],
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "shape_sig": tuple(sorted(tuple(map(int, xy)) for xy in rel.tolist())),
                    "colors": Counter(colors),
                    "area": int(len(cells)),
                }
            )
    return objects


def _is_sprite(obj: dict) -> bool:
    return len(obj["colors"]) == 1 and GREY not in obj["colors"]


def _is_pure_shadow(obj: dict) -> bool:
    return set(obj["colors"]) == {GREY}


def _is_stained_shadow(obj: dict) -> bool:
    if GREY not in obj["colors"] or len(obj["colors"]) != 2:
        return False
    non_grey = [color for color in obj["colors"] if color != GREY]
    return len(non_grey) == 1 and obj["colors"][non_grey[0]] == 1 and obj["colors"][GREY] == obj["area"] - 1


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    objects = _extract_objects(g, bg)
    assert objects, "expected foreground objects"

    sprites = [obj for obj in objects if _is_sprite(obj)]
    pure_shadows = [obj for obj in objects if _is_pure_shadow(obj)]
    stained_shadows = [obj for obj in objects if _is_stained_shadow(obj)]
    shadows = pure_shadows + stained_shadows

    assert len(stained_shadows) == 1, "expected exactly one stained orphan shadow"
    assert len(shadows) == len(sprites) + 1, "expected one more shadow than sprite"
    assert len(sprites) >= 2, "need complete pairs to infer the shared offset"

    sprite_colors = {next(iter(obj["colors"])) for obj in sprites}
    stain_color = next(color for color in stained_shadows[0]["colors"] if color != GREY)
    assert stain_color not in sprite_colors, "stain color must be unused by existing sprites"

    shadow_lookup: dict[tuple[tuple[tuple[int, int], ...], int, int], int] = {}
    for idx, shadow in enumerate(shadows):
        r0, c0, _r1, _c1 = shadow["bbox"]
        key = (shadow["shape_sig"], r0, c0)
        assert key not in shadow_lookup, "duplicate shadow placement would be ambiguous"
        shadow_lookup[key] = idx

    candidate_offsets = set()
    for sprite in sprites:
        sr0, sc0, _sr1, _sc1 = sprite["bbox"]
        for shadow in shadows:
            if sprite["shape_sig"] != shadow["shape_sig"]:
                continue
            hr0, hc0, _hr1, _hc1 = shadow["bbox"]
            candidate_offsets.add((hr0 - sr0, hc0 - sc0))

    valid_offsets: list[tuple[tuple[int, int], int]] = []
    for dr, dc in candidate_offsets:
        matched_shadow_indices = []
        for sprite in sprites:
            sr0, sc0, _sr1, _sc1 = sprite["bbox"]
            key = (sprite["shape_sig"], sr0 + dr, sc0 + dc)
            shadow_idx = shadow_lookup.get(key)
            if shadow_idx is None:
                matched_shadow_indices = []
                break
            matched_shadow_indices.append(shadow_idx)
        if len(matched_shadow_indices) != len(sprites):
            continue
        if len(set(matched_shadow_indices)) != len(matched_shadow_indices):
            continue
        if any(not _is_pure_shadow(shadows[idx]) for idx in matched_shadow_indices):
            continue
        unmatched = set(range(len(shadows))) - set(matched_shadow_indices)
        if len(unmatched) != 1:
            continue
        orphan_idx = next(iter(unmatched))
        if not _is_stained_shadow(shadows[orphan_idx]):
            continue
        valid_offsets.append(((int(dr), int(dc)), orphan_idx))

    assert len(valid_offsets) == 1, "expected a unique global sprite-to-shadow offset"
    (dr, dc), orphan_idx = valid_offsets[0]
    orphan = shadows[orphan_idx]

    target_top = orphan["bbox"][0] - dr
    target_left = orphan["bbox"][1] - dc
    out = g.copy()

    stain_coords = [(r, c) for r, c in map(tuple, orphan["coords"].tolist()) if int(g[r, c]) != GREY]
    assert len(stain_coords) == 1, "expected exactly one stain cell"
    stain_r, stain_c = stain_coords[0]
    out[stain_r, stain_c] = GREY

    for rr, cc in orphan["rel_coords"]:
        r = target_top + rr
        c = target_left + cc
        assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "reconstructed sprite must stay in bounds"
        out[r, c] = stain_color

    return out, {
        "bg": bg,
        "offset": (dr, dc),
        "orphan": orphan,
        "sprites": sprites,
        "shadows": shadows,
        "stain_color": stain_color,
        "valid_offsets": [offset for offset, _ in valid_offsets],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing colored sprite from a gray orphan shadow and its residue stain.

    Input structure:
    - The background is black, and every foreground object is a disconnected multi-cell sprite or shadow.
    - Each colored sprite has a same-shape gray shadow, and all sprite-to-shadow displacements are identical.
    - Exactly one gray shadow has no matching sprite; one cell of that orphan shadow is recolored to a unique stain color.

    Rule:
    - Infer the single global sprite-to-shadow offset from the complete sprite-shadow pairs.
    - Use the orphan shadow's shape and translate it backward by that offset to locate the missing sprite.
    - Paint the full missing sprite with the stain color, and restore the orphan shadow's stained cell back to gray.
    - Leave every other object unchanged.

    Key insight:
    - Three separate clues encode one absent object: complete pairs reveal the displacement, the orphan shadow reveals the shape and position, and the lone stain reveals the missing sprite color.
    """
    return _solve(grid)[0]


def _random_connected_shape(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        target_area = int(rng.integers(3, min(8, h * w) + 1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != target_area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([[r0, c0]], dtype=int)
        mask[rel[:, 0], rel[:, 1]] = True
        if mask.shape[0] == 1 and mask.shape[1] == 1:
            continue
        return mask
    raise RuntimeError("failed to sample a connected sprite shape")


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(tuple(map(int, xy)) for xy in np.argwhere(mask).tolist()))


def _sample_shapes(rng: np.random.Generator, shadow_count: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    duplicate_budget = int(rng.integers(0, min(2, shadow_count - 1) + 1))
    for idx in range(shadow_count):
        reuse_existing = idx > 0 and duplicate_budget > 0 and rng.random() < 0.4
        if reuse_existing:
            shapes.append(shapes[int(rng.integers(len(shapes)))].copy())
            duplicate_budget -= 1
            continue
        for _ in range(200):
            candidate = _random_connected_shape(rng)
            if idx == 0 or rng.random() < 0.7:
                shapes.append(candidate)
                break
        else:
            raise RuntimeError("failed to prepare shapes")
    return shapes


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    rel = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in rel]


def _cells_clear(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if blocked[r0:r1, c0:c1].any():
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        blocked[r0:r1, c0:c1] = True


def _sample_offset(rng: np.random.Generator) -> tuple[int, int]:
    magnitudes = np.array([2, 3, 4, 5, 6])
    if rng.random() < 0.5:
        dr = int(rng.choice(np.append(-magnitudes, magnitudes)))
        dc = int(rng.choice(np.append(-magnitudes, magnitudes)))
        return dr, dc
    if rng.random() < 0.5:
        return 0, int(rng.choice(np.append(-magnitudes, magnitudes)))
    return int(rng.choice(np.append(-magnitudes, magnitudes))), 0


def _place_objects(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    shapes: list[np.ndarray],
    offset: tuple[int, int],
    orphan_index: int,
) -> list[dict] | None:
    h, w = grid_shape
    dr, dc = offset
    blocked = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(shapes)), key=lambda idx: int(shapes[idx].sum()), reverse=True)
    placements: list[dict | None] = [None] * len(shapes)

    for idx in order:
        mask = shapes[idx]
        sh, sw = mask.shape
        candidates = []
        for shadow_top in range(h - sh + 1):
            for shadow_left in range(w - sw + 1):
                sprite_top = shadow_top - dr
                sprite_left = shadow_left - dc
                if sprite_top < 0 or sprite_left < 0 or sprite_top + sh > h or sprite_left + sw > w:
                    continue
                shadow_cells = _cells_from_mask(mask, shadow_top, shadow_left)
                sprite_cells = _cells_from_mask(mask, sprite_top, sprite_left)
                temp = blocked.copy()
                if not _cells_clear(temp, shadow_cells):
                    continue
                _mark_blocked(temp, shadow_cells)
                if not _cells_clear(temp, sprite_cells):
                    continue
                candidates.append((shadow_top, shadow_left, sprite_top, sprite_left))
        if not candidates:
            return None
        shadow_top, shadow_left, sprite_top, sprite_left = candidates[int(rng.integers(len(candidates)))]
        shadow_cells = _cells_from_mask(mask, shadow_top, shadow_left)
        sprite_cells = _cells_from_mask(mask, sprite_top, sprite_left)
        _mark_blocked(blocked, shadow_cells)
        _mark_blocked(blocked, sprite_cells)
        placements[idx] = {
            "shape": mask,
            "shadow_top": int(shadow_top),
            "shadow_left": int(shadow_left),
            "sprite_top": int(sprite_top),
            "sprite_left": int(sprite_left),
            "is_orphan": idx == orphan_index,
        }

    return [item for item in placements if item is not None]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[top + rows, left + cols] = int(color)


def _build_instance(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    grid_h = int(rng.integers(18, 29))
    grid_w = int(rng.integers(18, 29))
    shadow_count = int(rng.integers(3, 6))
    orphan_index = int(rng.integers(shadow_count))
    offset = _sample_offset(rng)
    shapes = _sample_shapes(rng, shadow_count)
    placements = _place_objects(rng, (grid_h, grid_w), shapes, offset, orphan_index)
    if placements is None:
        return None

    needed_colors = shadow_count
    palette = rng.choice(NON_GREY_COLORS, size=needed_colors, replace=False).tolist()
    stain_color = int(palette[-1])
    sprite_colors = [int(color) for color in palette[:-1]]
    rng.shuffle(sprite_colors)

    inp = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    expected = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    complete_color_iter = iter(sprite_colors)
    metadata: dict = {"offset": offset, "orphan_shape_area": None}

    for idx, placement in enumerate(placements):
        mask = placement["shape"]
        shadow_top = placement["shadow_top"]
        shadow_left = placement["shadow_left"]
        sprite_top = placement["sprite_top"]
        sprite_left = placement["sprite_left"]
        _paint_mask(inp, mask, shadow_top, shadow_left, GREY)
        _paint_mask(expected, mask, shadow_top, shadow_left, GREY)

        if placement["is_orphan"]:
            rel_cells = np.argwhere(mask)
            stain_idx = int(rng.integers(len(rel_cells)))
            rr, cc = map(int, rel_cells[stain_idx])
            inp[shadow_top + rr, shadow_left + cc] = stain_color
            _paint_mask(expected, mask, sprite_top, sprite_left, stain_color)
            metadata["orphan_shape_area"] = int(mask.sum())
            metadata["stain_shadow_cell"] = (shadow_top + rr, shadow_left + cc)
        else:
            color = int(next(complete_color_iter))
            _paint_mask(inp, mask, sprite_top, sprite_left, color)
            _paint_mask(expected, mask, sprite_top, sprite_left, color)

    return inp, expected, metadata


def _validate_instance(inp: np.ndarray, expected: np.ndarray, metadata: dict) -> bool:
    try:
        actual, solved_meta = _solve(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(solved_meta["sprites"]) < 2:
        return False
    if solved_meta["offset"] != metadata["offset"]:
        return False
    if len(solved_meta["valid_offsets"]) != 1:
        return False
    if np.array_equal(inp, expected):
        return False
    diff_count = int(np.count_nonzero(inp != expected))
    if diff_count != int(metadata["orphan_shape_area"]) + 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        built = _build_instance(rng)
        if built is None:
            continue
        inp, expected, metadata = built
        if _validate_instance(inp, expected, metadata):
            return inp
    raise RuntimeError("failed to generate a valid input for residual-shadow reconstruction")
