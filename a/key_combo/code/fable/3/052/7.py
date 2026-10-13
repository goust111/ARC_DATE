from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

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
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _is_hollow_rectangle_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    return bool(np.array_equal(mask, perimeter))


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox_coords(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if _is_hollow_rectangle_mask(local):
                frames.append(
                    {
                        "color": int(color),
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    }
                )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_sprites(grid: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    outside = np.ones(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    sprites: list[dict] = []
    for idx, coords in enumerate(_components((grid != int(bg)) & outside)):
        patch = _crop_component(grid, coords, bg)
        bbox = _bbox_coords(coords)
        colors = sorted(set(map(int, patch.ravel())) - {int(bg)})
        sprites.append(
            {
                "id": int(idx),
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "area": int(coords.shape[0]),
                "colors": colors,
            }
        )

    sprites.sort(key=lambda item: (item["area"], item["bbox"][0], item["bbox"][1]))
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """Place each ranked sprite into the matching frame, using same-color dot counts as ordinal indices.

    Input structure:
    - The grid has a dominant background, several one-cell-thick hollow rectangular frames, and several disconnected sprites outside those frames.
    - Each frame contains 1 to 4 single-cell dots of the same color as the frame plus additional distractor dots of other colors.
    - Sprite areas are all distinct, and frames are large enough to center their chosen sprite by bbox.

    Rule:
    - Detect every hollow frame and count how many interior pixels share that frame's color; call that count k.
    - Ignore all dots inside the frames, sort the outside sprites by pixel count from smallest to largest, and select the k-th sprite for that frame.
    - Clear every frame interior, erase each selected sprite from its original location, and paste that sprite centered inside every frame that points to it.

    Key insight:
    - The same-color dot count is not a size match; it is an ordinal rank into the globally sorted sprite list.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one hollow frame"

    sprites = _find_sprites(g, bg, frames)
    assert len(sprites) > len(frames), "expected extra sprites outside the frames"
    areas = [sprite["area"] for sprite in sprites]
    assert len(areas) == len(set(areas)), "sprite areas must be unique"

    out = g.copy()
    selected_sprite_ids: set[int] = set()
    placements: list[tuple[np.ndarray, int, int]] = []

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        k = int(np.count_nonzero(inner == int(frame["color"])))
        assert 1 <= k <= len(sprites), "frame dot count must index the sorted sprite list"

        sprite = sprites[k - 1]
        selected_sprite_ids.add(int(sprite["id"]))
        out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)

        patch = sprite["patch"]
        ph, pw = patch.shape
        ih, iw = inner.shape
        assert ph <= ih and pw <= iw, "selected sprite must fit inside the frame interior"
        assert (ih - ph) % 2 == 0 and (iw - pw) % 2 == 0, "selected sprite must be centerable"
        dst_r = r0 + 1 + (ih - ph) // 2
        dst_c = c0 + 1 + (iw - pw) // 2
        placements.append((patch, int(dst_r), int(dst_c)))

    for sprite in sprites:
        if int(sprite["id"]) not in selected_sprite_ids:
            continue
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for patch, dst_r, dst_c in placements:
        _paste(out, patch, dst_r, dst_c, bg)

    return out


def _random_free_bbox(
    occupied: np.ndarray,
    h: int,
    w: int,
    *,
    border: int = 1,
    padding: int = 1,
) -> tuple[int, int]:
    hh, ww = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, hh - border - h + 1):
        for c0 in range(border, ww - border - w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(hh, r0 + h + padding)
            cc1 = min(ww, c0 + w + padding)
            if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        raise ValueError("no free location")
    return random.choice(candidates)


def _shape_from_cells(cells: set[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return patch


def _generate_sprite_patch(area: int, color: int, bg: int) -> np.ndarray:
    choices = [(h, w) for h in range(1, 6) for w in range(1, 6) if h * w >= area]
    for _ in range(200):
        h, w = random.choice(choices)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        patch = _shape_from_cells(cells, color, bg)
        mask = patch != int(bg)
        if _is_hollow_rectangle_mask(mask):
            continue
        if area >= 4 and (mask.shape[0] == 1 or mask.shape[1] == 1) and random.random() < 0.8:
            continue
        return patch
    raise RuntimeError(f"failed to generate sprite patch with area {area}")


def _pick_inner_dim(sprite_dim: int, min_inner: int) -> int:
    lo = max(int(sprite_dim), int(min_inner))
    choices = [dim for dim in range(lo, 10) if (dim - int(sprite_dim)) % 2 == 0]
    if not choices:
        raise ValueError("no centerable inner dimension")
    return random.choice(choices[:3])


def _pick_spread_positions(cells: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    for _ in range(40):
        order = cells[:]
        random.shuffle(order)
        picked: list[tuple[int, int]] = []
        for cell in order:
            if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in picked):
                picked.append(cell)
                if len(picked) == n:
                    return picked
    return random.sample(cells, n)


def _build_frame_patch(
    frame_color: int,
    inner_h: int,
    inner_w: int,
    k: int,
    palette: list[int],
    bg: int,
) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)

    safe_cells = [(r, c) for r in range(1, inner_h - 1) for c in range(1, inner_w - 1)]
    same_color_cells = _pick_spread_positions(safe_cells, k)
    remaining = [cell for cell in safe_cells if cell not in same_color_cells]
    n_noise = random.randint(1, min(3, len(remaining)))
    noise_cells = random.sample(remaining, n_noise)
    noise_palette = [color for color in palette if color != int(frame_color)]

    inner = patch[1:-1, 1:-1]
    for r, c in same_color_cells:
        inner[r, c] = int(frame_color)
    for r, c in noise_cells:
        inner[r, c] = int(random.choice(noise_palette))
    return patch


def _build_area_list(tricky_k: int, n_sprites: int) -> list[int]:
    if tricky_k == 3:
        areas = [2, 3, 4, 6]
        pool = [5, 7, 8, 9]
    else:
        areas = [2, 3, 4, 5]
        pool = [6, 7, 8, 9]
    while len(areas) < n_sprites:
        area = random.choice([value for value in pool if value not in areas])
        areas.append(area)
    areas.sort()
    return areas


def _choose_frame_counts(n_frames: int) -> tuple[list[int], int]:
    tricky_k = random.choice([3, 4])
    counts = [tricky_k]
    while len(counts) < n_frames:
        if random.random() < 0.35:
            counts.append(random.choice(counts))
        else:
            counts.append(random.randint(1, 4))
    if n_frames > 1 and len(set(counts)) == 1:
        counts[-1] = random.choice([value for value in (1, 2, 3, 4) if value != counts[0]])
    random.shuffle(counts)
    return counts, tricky_k


def _place_patches_with_bg(
    grid: np.ndarray,
    patches: list[np.ndarray],
    occupied: np.ndarray,
    bg: int,
) -> list[tuple[int, int]]:
    locations: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1],
        reverse=True,
    )
    for idx in order:
        patch = patches[idx]
        r0, c0 = _random_free_bbox(occupied, patch.shape[0], patch.shape[1], border=1, padding=1)
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = True
        _paste(grid, patch, r0, c0, bg)
        locations[idx] = (int(r0), int(c0))
    return [loc for loc in locations if loc is not None]


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(ALL_COLORS)
    palette = [color for color in ALL_COLORS if color != int(bg)]

    n_frames = random.randint(2, 4)
    frame_counts, tricky_k = _choose_frame_counts(n_frames)
    max_k = max(frame_counts)
    n_sprites = random.randint(max(5, n_frames + 1, max_k + 1), 6)
    areas = _build_area_list(tricky_k, n_sprites)

    frame_colors = random.sample(palette, n_frames)
    sprite_colors = [random.choice(palette) for _ in range(n_sprites)]

    decoy_frame_idx = next(idx for idx, value in enumerate(frame_counts) if value == tricky_k)
    true_sprite_idx = tricky_k - 1
    decoy_sprite_idx = random.choice([idx for idx in range(n_sprites) if idx != true_sprite_idx])
    sprite_colors[decoy_sprite_idx] = frame_colors[decoy_frame_idx]
    if sprite_colors[true_sprite_idx] == frame_colors[decoy_frame_idx]:
        choices = [color for color in palette if color != frame_colors[decoy_frame_idx]]
        sprite_colors[true_sprite_idx] = random.choice(choices)

    try:
        sprite_patches = [
            _generate_sprite_patch(area, color, bg)
            for area, color in zip(areas, sprite_colors)
        ]
    except RuntimeError:
        return None

    frame_patches: list[np.ndarray] = []
    for frame_color, k in zip(frame_colors, frame_counts):
        sprite_patch = sprite_patches[k - 1]
        inner_h = _pick_inner_dim(sprite_patch.shape[0], 6 if k >= 3 else 5)
        inner_w = _pick_inner_dim(sprite_patch.shape[1], 6 if k >= 3 else 5)
        frame_patches.append(_build_frame_patch(frame_color, inner_h, inner_w, k, palette, bg))

    reserve_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in frame_patches + sprite_patches)
    max_dim = max(max(patch.shape) for patch in frame_patches + sprite_patches) + 2
    min_side = max(max_dim, int(np.ceil(np.sqrt(reserve_area))))
    if min_side > 30:
        return None

    h = random.randint(max(20, min_side), 30)
    w = random.randint(max(20, min_side), 30)
    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    try:
        _place_patches_with_bg(grid, frame_patches, occupied, bg)
        _place_patches_with_bg(grid, sprite_patches, occupied, bg)
    except ValueError:
        return None

    counts = Counter(map(int, grid.ravel()))
    if counts[int(bg)] <= max(count for color, count in counts.items() if int(color) != int(bg)):
        return None

    frames = _find_frames(grid, bg)
    sprites = _find_sprites(grid, bg, frames)
    if len(frames) != n_frames or len(sprites) != n_sprites:
        return None
    if [sprite["area"] for sprite in sprites] != areas:
        return None

    if not any(k in areas and areas.index(k) + 1 != k for k in frame_counts):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
