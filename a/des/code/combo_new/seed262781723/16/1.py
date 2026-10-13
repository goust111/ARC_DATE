import random

import numpy as np


_DIRECTIONS = ("up", "right", "down", "left")
_DIR_TO_ROT_K = {"up": 0, "right": 3, "down": 2, "left": 1}
_SPRITE_SIZES = (3, 5)
_MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _slot_start_from_hub(hub_r: int, hub_c: int, size: int, direction: str) -> tuple[int, int]:
    gap = size // 2 + 1
    half = size // 2
    if direction == "up":
        return hub_r - (gap + size), hub_c - half
    if direction == "down":
        return hub_r + gap + 1, hub_c - half
    if direction == "left":
        return hub_r - half, hub_c - (gap + size)
    if direction == "right":
        return hub_r - half, hub_c + gap + 1
    raise ValueError(direction)


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    foreground = grid != bg
    frame_mask = np.zeros_like(foreground, dtype=bool)
    frames: list[dict] = []

    for coords in _components(foreground):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5 or h != w or h % 2 == 0:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if coords.shape[0] != int(border.sum()):
            continue
        if np.any(patch[~border] != bg):
            continue
        if not np.all(patch[border] != bg):
            continue

        colors, counts = np.unique(patch[border], return_counts=True)
        if len(colors) != 2 or int(counts.min()) != 1:
            continue
        marker_color = int(colors[np.argmin(counts)])
        marker_pos = np.argwhere(patch == marker_color)
        if marker_pos.shape[0] != 1:
            continue
        mr, mc = map(int, marker_pos[0])
        mid = h // 2
        if mr == 0 and mc == mid:
            direction = "up"
        elif mr == h - 1 and mc == mid:
            direction = "down"
        elif mc == 0 and mr == mid:
            direction = "left"
        elif mc == w - 1 and mr == mid:
            direction = "right"
        else:
            continue

        frame_mask[coords[:, 0], coords[:, 1]] = True
        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "size": h - 2,
                "direction": direction,
            }
        )

    assert frames, "expected at least one hollow frame"
    sizes = {frame["size"] for frame in frames}
    assert len(sizes) == 1, "all frames must share the same interior size"
    return frames, frame_mask


def _find_legend_patches(
    grid: np.ndarray, bg: int, frame_mask: np.ndarray, size: int
) -> dict[str, np.ndarray]:
    residual = (grid != bg) & ~frame_mask
    comps = _components(residual)
    h, w = grid.shape

    for coords in comps:
        if coords.shape[0] != 1:
            continue
        hub_r, hub_c = map(int, coords[0])
        cover = np.zeros_like(residual, dtype=bool)
        cover[hub_r, hub_c] = True
        patches: dict[str, np.ndarray] = {}
        ok = True

        for direction in _DIRECTIONS:
            r0, c0 = _slot_start_from_hub(hub_r, hub_c, size, direction)
            if r0 < 0 or c0 < 0 or r0 + size > h or c0 + size > w:
                ok = False
                break
            patch = grid[r0 : r0 + size, c0 : c0 + size].copy()
            mask = patch != bg
            if not np.any(mask):
                ok = False
                break
            if len(np.unique(patch[mask])) != 1:
                ok = False
                break
            if len(_components(mask)) != 1:
                ok = False
                break
            patches[direction] = patch
            cover[r0 : r0 + size, c0 : c0 + size] = True

        if ok and not np.any(residual & ~cover):
            return patches

    raise AssertionError("failed to locate the corner legend")


def _fill_frames(
    grid: np.ndarray, bg: int, frames: list[dict], legend_patches: dict[str, np.ndarray]
) -> np.ndarray:
    out = grid.copy()
    size = int(frames[0]["size"])
    for frame in frames:
        direction = str(frame["direction"])
        patch = np.rot90(legend_patches[direction], _DIR_TO_ROT_K[direction])
        mask = patch != bg
        r0 = int(frame["r0"]) + 1
        c0 = int(frame["c0"]) + 1
        out[r0 : r0 + size, c0 : c0 + size] = bg
        out[r0 : r0 + size, c0 : c0 + size][mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame with the corner-legend sprite chosen by its border marker and rotated to that side.

    Input structure:
    - The background is the most common color.
    - Several same-sized hollow square frames are drawn in one color; each frame has exactly one different-colored pixel at the midpoint of its top, bottom, left, or right border.
    - In one corner, four small square sprites surround a single hub pixel; the sprite above, below, left, and right of the hub defines the legend entry for that direction.

    Rule:
    - Detect the four legend sprites from their positions around the hub.
    - For each frame, read which side contains the special border pixel.
    - Select the sprite stored in the matching legend direction, rotate that upright sprite so it faces the same direction, and copy its non-background cells into the frame interior.
    - Keep the source legend, the hub pixel, the frame borders, and the marker pixels unchanged.

    Key insight: The border marker is a direction selector, while the corner legend is a lookup table from direction to sprite identity; the final fill requires both lookup and rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames, frame_mask = _find_frames(g, bg)
    size = int(frames[0]["size"])
    legend_patches = _find_legend_patches(g, bg, frame_mask, size)
    return _fill_frames(g, bg, frames, legend_patches)


# --------------- generator ---------------


def _mask_rotation_key(mask: np.ndarray) -> bytes:
    return min(np.rot90(mask, k).tobytes() for k in range(4))


def _random_up_mask(rng: random.Random, size: int) -> np.ndarray | None:
    center = size // 2
    for _ in range(200):
        segments: list[tuple[int, int]] = [(center, center)]
        prev_left = center
        prev_right = center
        for row in range(1, size):
            prev_len = prev_right - prev_left + 1
            candidates: list[tuple[int, int]] = []
            min_len = max(1, prev_len - 1)
            max_len = min(size, prev_len + 2)
            for length in range(min_len, max_len + 1):
                for left in range(0, size - length + 1):
                    right = left + length - 1
                    if right < prev_left or left > prev_right:
                        continue
                    candidates.append((left, right))
            if not candidates:
                break
            prev_left, prev_right = rng.choice(candidates)
            segments.append((prev_left, prev_right))
        if len(segments) != size:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for row, (left, right) in enumerate(segments):
            mask[row, left : right + 1] = True

        if int(mask.sum()) >= size * size or int(mask.sum()) < size + 2:
            continue
        if int(mask[0].sum()) != 1 or int(mask[-1].sum()) < 2:
            continue
        if len({_rot.tobytes() for _rot in (np.rot90(mask, k) for k in range(4))}) != 4:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _choose_sprite_masks(rng: random.Random, size: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(500):
        mask = _random_up_mask(rng, size)
        if mask is None:
            continue
        key = _mask_rotation_key(mask)
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
        if len(masks) == 4:
            return masks
    return None


def _patch_from_mask(bg: int, color: int, mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _legend_corner_start(height: int, width: int, side: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - side
    if corner == "bl":
        return height - side, 0
    if corner == "br":
        return height - side, width - side
    raise ValueError(corner)


def _place_legend(
    grid: np.ndarray,
    blocked: np.ndarray,
    bg: int,
    hub_color: int,
    legend_patches: dict[str, np.ndarray],
    corner: str,
) -> None:
    size = next(iter(legend_patches.values())).shape[0]
    gap = size // 2 + 1
    side = 2 * size + 2 * gap + 1
    start_r, start_c = _legend_corner_start(grid.shape[0], grid.shape[1], side, corner)
    hub_r = start_r + size + gap
    hub_c = start_c + size + gap
    grid[hub_r, hub_c] = hub_color
    blocked[start_r : start_r + side, start_c : start_c + side] = True
    for direction, patch in legend_patches.items():
        local_r, local_c = _slot_start_from_hub(size + gap, size + gap, size, direction)
        r0 = start_r + local_r
        c0 = start_c + local_c
        mask = patch != bg
        grid[r0 : r0 + size, c0 : c0 + size][mask] = patch[mask]


def _find_free_top_left(
    rng: random.Random, blocked: np.ndarray, obj_h: int, obj_w: int, pad: int = 1, edge_margin: int = 1
) -> tuple[int, int] | None:
    h, w = blocked.shape
    candidates: list[tuple[int, int]] = []
    for r in range(edge_margin, h - obj_h - edge_margin + 1):
        for c in range(edge_margin, w - obj_w - edge_margin + 1):
            rr0 = max(0, r - pad)
            rr1 = min(h, r + obj_h + pad)
            cc0 = max(0, c - pad)
            cc1 = min(w, c + obj_w + pad)
            if not np.any(blocked[rr0:rr1, cc0:cc1]):
                candidates.append((r, c))
    if not candidates:
        return None
    return rng.choice(candidates)


def _draw_frame(
    grid: np.ndarray, top: int, left: int, outer: int, frame_color: int, marker_color: int, direction: str
) -> None:
    bottom = top + outer - 1
    right = left + outer - 1
    grid[top, left : right + 1] = frame_color
    grid[bottom, left : right + 1] = frame_color
    grid[top : bottom + 1, left] = frame_color
    grid[top : bottom + 1, right] = frame_color
    mid = outer // 2
    if direction == "up":
        grid[top, left + mid] = marker_color
    elif direction == "down":
        grid[bottom, left + mid] = marker_color
    elif direction == "left":
        grid[top + mid, left] = marker_color
    else:
        grid[top + mid, right] = marker_color


def _build_expected_output(
    grid: np.ndarray, bg: int, frames: list[dict], legend_patches: dict[str, np.ndarray]
) -> np.ndarray:
    return _fill_frames(grid, bg, frames, legend_patches)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.choice(_SPRITE_SIZES)
    masks = _choose_sprite_masks(rng, size)
    if masks is None:
        return None

    frame_count = rng.choice([4, 5, 6, 7] if size == 3 else [4, 5, 6])
    color_pool = rng.sample(list(range(10)), 8)
    bg, frame_color, marker_color, hub_color = color_pool[:4]
    sprite_colors = color_pool[4:]
    legend_patches = {
        direction: _patch_from_mask(bg, color, mask)
        for direction, color, mask in zip(_DIRECTIONS, sprite_colors, masks)
    }

    outer = size + 2
    gap = size // 2 + 1
    legend_side = 2 * size + 2 * gap + 1
    min_side = max(
        18,
        legend_side + outer + 2,
        outer * 2 + 6,
        18 + 2 * max(0, frame_count - 4) + (4 if size == 5 else 0),
    )
    if min_side > 30:
        return None
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)

    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    _place_legend(grid, blocked, bg, hub_color, legend_patches, corner)

    directions = list(_DIRECTIONS)
    extra_count = frame_count - 4
    directions.extend(rng.choices(_DIRECTIONS, k=extra_count))
    rng.shuffle(directions)

    frames: list[dict] = []
    for direction in directions:
        pos = _find_free_top_left(rng, blocked, outer, outer, pad=1, edge_margin=1)
        if pos is None:
            return None
        top, left = pos
        _draw_frame(grid, top, left, outer, frame_color, marker_color, direction)
        blocked[top : top + outer, left : left + outer] = True
        frames.append({"r0": top, "c0": left, "size": size, "direction": direction})

    expected = _build_expected_output(grid, bg, frames, legend_patches)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
