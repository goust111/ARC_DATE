from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 5
MAIN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CORNERS = ("TL", "TR", "BR", "BL")

GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 250


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _rotate_to_marker_tl(patch: np.ndarray, corner: str) -> np.ndarray:
    if corner == "TL":
        return patch.copy()
    if corner == "TR":
        return np.rot90(patch, 1).copy()
    if corner == "BR":
        return np.rot90(patch, 2).copy()
    if corner == "BL":
        return np.rot90(patch, 3).copy()
    raise ValueError(corner)


def _corner_name(local_rc: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = local_rc
    h, w = shape
    if (r, c) == (0, 0):
        return "TL"
    if (r, c) == (0, w - 1):
        return "TR"
    if (r, c) == (h - 1, w - 1):
        return "BR"
    if (r, c) == (h - 1, 0):
        return "BL"
    raise AssertionError("marker must lie on a bbox corner")


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    assert 4 <= len(comps) <= 5, "expected 4-5 separated sprites"

    sprites: list[dict] = []
    seen_areas: set[int] = set()
    for coords in comps:
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(g, bbox)
        h, w = patch.shape
        assert h != w, "sprites must be non-square"
        assert np.all(patch != BG), "sprite bbox must be a filled rectangle"

        marker_coords = np.argwhere(patch == MARKER)
        assert marker_coords.shape[0] == 1, "each sprite must have exactly one grey marker"
        mr, mc = map(int, marker_coords[0])
        corner = _corner_name((mr, mc), patch.shape)

        colors = sorted(set(map(int, patch.flat)) - {MARKER})
        assert len(colors) == 1, "each sprite must have one main color plus the grey marker"
        color = colors[0]
        area = int(h * w)
        assert area not in seen_areas, "sprite areas must be unique"
        seen_areas.add(area)

        rotated = _rotate_to_marker_tl(patch, corner)
        rotated[rotated == MARKER] = color

        sprites.append(
            {
                "bbox": bbox,
                "input_shape": (h, w),
                "color": color,
                "area": area,
                "corner": corner,
                "rotated": rotated,
            }
        )

    sprites.sort(key=lambda item: item["area"], reverse=True)
    for i in range(1, len(sprites)):
        prev_h, prev_w = sprites[i - 1]["rotated"].shape
        cur_h, cur_w = sprites[i]["rotated"].shape
        assert cur_h < prev_h - 2 and cur_w < prev_w - 2, "rotated sizes must form a strict nesting chain"
    return sprites


def _render_nested(rotated_patches: list[np.ndarray]) -> np.ndarray:
    outer_h, outer_w = rotated_patches[0].shape
    out = np.full((outer_h, outer_w), BG, dtype=int)
    for idx, patch in enumerate(rotated_patches):
        h, w = patch.shape
        r0 = idx
        c0 = idx
        assert r0 + h <= outer_h and c0 + w <= outer_w, "nested patch must fit inside the outer patch"
        out[r0 : r0 + h, c0 : c0 + w] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each marked rectangle so its grey corner reaches the top-left, then nest the rectangles by descending area.

    Input structure:
    - The grid has black background and 4-5 separated solid non-black rectangular sprites.
    - Each sprite uses one unique main color and has exactly one grey marker pixel at one bbox corner.
    - Every sprite is non-square, and sprite areas are all different.

    Rule:
    - For each sprite, read which bbox corner holds the grey marker.
    - Rotate the sprite just enough to move that marker to the top-left corner of the sprite bbox.
    - Replace the grey marker with the sprite's main color, sort the rotated sprites by area descending,
      then place them at offsets (0,0), (1,1), (2,2), ... with later sprites overwriting earlier ones.
    - Crop the output to the rotated bbox size of the largest sprite.

    Key insight:
    - The single grey corner is a local orientation code: it determines each non-square rectangle's rotated width
      and height, and those rotated dimensions in turn determine the final nested geometry.
    """
    sprites = _extract_sprites(np.asarray(grid, dtype=int))
    return _render_nested([item["rotated"] for item in sprites])


def _build_rotated_dims(count: int) -> list[tuple[int, int]] | None:
    outer_h_min = 2 + 3 * (count - 1)
    outer_w_min = 3 + 3 * (count - 1)
    for _ in range(200):
        outer_h = random.randint(outer_h_min, min(18, outer_h_min + 4))
        outer_w = random.randint(outer_w_min, min(20, outer_w_min + 5))
        if outer_h == outer_w:
            continue

        dims = [(outer_h, outer_w)]
        used_areas = {outer_h * outer_w}
        ok = True
        for idx in range(1, count):
            remaining = count - idx - 1
            h_low = 2 + 3 * remaining
            w_low = 2 + 3 * remaining
            h_high = dims[-1][0] - 3
            w_high = dims[-1][1] - 3
            candidates = [
                (h, w)
                for h in range(h_low, h_high + 1)
                for w in range(w_low, w_high + 1)
                if h != w and h * w not in used_areas
            ]
            if not candidates:
                ok = False
                break
            h, w = random.choice(candidates)
            dims.append((h, w))
            used_areas.add(h * w)
        if ok:
            return dims
    return None


def _make_input_patch(rotated_shape: tuple[int, int], color: int, corner: str) -> tuple[np.ndarray, np.ndarray]:
    rh, rw = rotated_shape
    if corner in ("TL", "BR"):
        h, w = rh, rw
    else:
        h, w = rw, rh

    patch = np.full((h, w), int(color), dtype=int)
    if corner == "TL":
        patch[0, 0] = MARKER
    elif corner == "TR":
        patch[0, w - 1] = MARKER
    elif corner == "BR":
        patch[h - 1, w - 1] = MARKER
    else:
        patch[h - 1, 0] = MARKER

    rotated = _rotate_to_marker_tl(patch, corner)
    rotated[rotated == MARKER] = int(color)
    return patch, rotated


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not np.any(occ[rr0:rr1, cc0:cc1])


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    base = max(max_h, max_w, int(np.ceil(np.sqrt(total_area * 1.7))))
    h_min = max(base + random.randint(0, 2), max_h + 2)
    w_min = max(base + random.randint(0, 2), max_w + 2)
    if h_min > 30 or w_min > 30:
        return None
    grid_h = random.randint(h_min, 30)
    grid_w = random.randint(w_min, 30)

    for _ in range(PLACE_ATTEMPTS):
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        occ = np.zeros((grid_h, grid_w), dtype=bool)
        placed = True
        order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            done = False
            for _ in range(PLACE_ATTEMPTS):
                r0 = random.randint(0, grid_h - ph)
                c0 = random.randint(0, grid_w - pw)
                if not _can_place(occ, r0, c0, ph, pw, pad=1):
                    continue
                grid[r0 : r0 + ph, c0 : c0 + pw] = patch
                occ[r0 : r0 + ph, c0 : c0 + pw] = True
                done = True
                break
            if not done:
                placed = False
                break
        if placed:
            return grid
    return None


def _shortcut_without_rotation(grid: np.ndarray) -> np.ndarray:
    sprites = _extract_sprites(grid)
    patches = []
    for item in sprites:
        patch = _crop_bbox(grid, item["bbox"])
        patch = patch.copy()
        patch[patch == MARKER] = item["color"]
        patches.append(patch)
    return _render_nested(patches)


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> None:
    sprites = _extract_sprites(grid)
    assert len(sprites) in (4, 5)
    assert max(grid.shape) <= 30
    assert max(expected.shape) <= 30
    assert {item["corner"] for item in sprites} == set(CORNERS), "every sample should expose all four corner cases"
    assert any(item["corner"] in {"TR", "BL"} for item in sprites), "rotation must include width/height swaps"
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match generator ground truth"
    shortcut = _shortcut_without_rotation(grid)
    assert not np.array_equal(shortcut, expected), "generated samples must reject the no-rotation shortcut"
    for i in range(1, len(sprites)):
        prev_h, prev_w = sprites[i - 1]["rotated"].shape
        cur_h, cur_w = sprites[i]["rotated"].shape
        assert cur_h < prev_h - 2 and cur_w < prev_w - 2


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        count = random.choice((4, 5))
        rotated_dims = _build_rotated_dims(count)
        if rotated_dims is None:
            continue

        colors = random.sample(MAIN_COLORS, count)
        corners = list(CORNERS)
        random.shuffle(corners)
        while len(corners) < count:
            corners.append(random.choice(CORNERS))

        input_patches: list[np.ndarray] = []
        rotated_patches: list[np.ndarray] = []
        for dims, color, corner in zip(rotated_dims, colors, corners):
            patch, rotated = _make_input_patch(dims, color, corner)
            input_patches.append(patch)
            rotated_patches.append(rotated)

        grid = _place_patches(input_patches)
        if grid is None:
            continue

        expected = _render_nested(rotated_patches)
        try:
            _validate_generated(grid, expected)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid sample")
