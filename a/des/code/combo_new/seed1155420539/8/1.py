from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _bg_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    components: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        visited = np.zeros((h, w), dtype=bool)
        cells = np.argwhere(grid == color)
        for sr, sc in cells:
            sr = int(sr)
            sc = int(sc)
            if visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            mask, box = _component_mask(arr)
            components.append({"color": color, "coords": arr, "mask": mask, "bbox": box})
    return components


def _frame_gap_direction(mask: np.ndarray, grid_patch: np.ndarray, bg: int) -> str | None:
    h, w = mask.shape
    if h < 4 or w < 4:
        return None
    if mask[1:-1, 1:-1].any():
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None
    if np.any(grid_patch[1:-1, 1:-1] != bg):
        return None

    sides = {
        "up": mask[0, 1:-1],
        "down": mask[-1, 1:-1],
        "left": mask[1:-1, 0],
        "right": mask[1:-1, -1],
    }
    missing = [name for name, strip in sides.items() if not strip.any()]
    if len(missing) != 1:
        return None
    for name, strip in sides.items():
        if name != missing[0] and not strip.all():
            return None
    return missing[0]


def _protrusion_direction(mask: np.ndarray) -> str:
    counts = {
        "up": int(mask[0, :].sum()),
        "down": int(mask[-1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, -1].sum()),
    }
    best = min(counts.values())
    dirs = [name for name, value in counts.items() if value == best]
    assert len(dirs) == 1, f"sprite must have a unique protrusion side, got {counts}"
    return dirs[0]


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flipped in (False, True):
        base = np.fliplr(mask) if flipped else mask
        for k in range(4):
            variant = np.rot90(base, k)
            key = (variant.shape, variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant)
    return variants


def _oriented_sprite(mask: np.ndarray, target_dir: str) -> np.ndarray:
    for variant in _mask_variants(mask):
        if _protrusion_direction(variant) == target_dir:
            return variant
    raise AssertionError(f"no orientation maps sprite to {target_dir}")


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = mask.shape
    dst_h, dst_w = target_shape
    if (src_h, src_w) == (dst_h, dst_w):
        return mask.copy()
    row_idx = np.array(
        [0 if dst_h == 1 else int(round(r * (src_h - 1) / (dst_h - 1))) for r in range(dst_h)],
        dtype=int,
    )
    col_idx = np.array(
        [0 if dst_w == 1 else int(round(c * (src_w - 1) / (dst_w - 1))) for c in range(dst_w)],
        dtype=int,
    )
    return mask[row_idx][:, col_idx]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each open rectangular frame with the edge sprite whose protrusion direction matches the opposite side, then recolor it with the frame border color.

    Input structure:
    - The dominant color is the background.
    - Several monochrome rectangular frames are scattered in the grid; each has a 1-pixel border, keeps its four corners, and is missing exactly one side segment (top, bottom, left, or right).
    - The interior of every frame is background.
    - Small monochrome sprites sit near the grid edges; each sprite is 4-connected and has exactly one uniquely protruding bbox side.

    Rule:
    - Detect each frame and read the direction of its missing side segment.
    - Among the edge sprites, find the one whose protrusion direction is opposite that frame gap, then rotate or flip it so the protrusion points toward the gap.
    - Map the oriented sprite bbox onto the frame interior, draw its foreground cells using the frame's own border color, keep background cells empty, and erase the original sprite from the grid.

    Key insight:
    - The frame gap does not directly specify the drawn shape; it indexes the sprite whose protrusion points the other way, and the chosen sprite inherits the destination frame color instead of keeping its original color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    frames: list[dict] = []
    sprites: dict[str, dict] = {}

    for comp in _components_by_color(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        gap_dir = _frame_gap_direction(comp["mask"], g[r0 : r1 + 1, c0 : c1 + 1], bg)
        if gap_dir is not None:
            frames.append(
                {
                    "color": int(comp["color"]),
                    "bbox": comp["bbox"],
                    "gap": gap_dir,
                }
            )
            continue

        sprite_dir = _protrusion_direction(comp["mask"])
        assert sprite_dir not in sprites, f"duplicate sprite protrusion direction: {sprite_dir}"
        sprites[sprite_dir] = comp

    assert frames, "expected at least one frame"
    assert set(sprites) == set(DIRS), f"expected one sprite per direction, got {sorted(sprites)}"

    for sprite in sprites.values():
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = bg

    for frame in frames:
        source = sprites[OPPOSITE[frame["gap"]]]
        oriented = _oriented_sprite(source["mask"], frame["gap"])
        r0, c0, r1, c1 = frame["bbox"]
        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        mapped = _resize_mask(oriented, (interior_h, interior_w))
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[mapped] = int(frame["color"])
        patch[~mapped] = bg

    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    for variant in _mask_variants(mask):
        key = (variant.shape, variant.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _make_sprite_mask(size: int, direction: str, rng: random.Random) -> np.ndarray:
    for _ in range(100):
        mask = np.zeros((size, size), dtype=bool)
        if direction == "up":
            span = rng.randint(1, size - 2)
            start = rng.randint(1, size - span - 1)
            mask[1:, :] = True
            mask[0, start : start + span] = True
        elif direction == "down":
            span = rng.randint(1, size - 2)
            start = rng.randint(1, size - span - 1)
            mask[:-1, :] = True
            mask[-1, start : start + span] = True
        elif direction == "left":
            span = rng.randint(1, size - 2)
            start = rng.randint(1, size - span - 1)
            mask[:, 1:] = True
            mask[start : start + span, 0] = True
        else:
            span = rng.randint(1, size - 2)
            start = rng.randint(1, size - span - 1)
            mask[:, :-1] = True
            mask[start : start + span, -1] = True
        if _protrusion_direction(mask) == direction:
            return mask
    raise RuntimeError(f"failed to build sprite for {direction}")


def _make_frame_patch(size: int, gap_dir: str) -> np.ndarray:
    outer = size + 2
    patch = np.zeros((outer, outer), dtype=bool)
    patch[0, :] = True
    patch[-1, :] = True
    patch[:, 0] = True
    patch[:, -1] = True
    if gap_dir == "up":
        patch[0, 1:-1] = False
    elif gap_dir == "down":
        patch[-1, 1:-1] = False
    elif gap_dir == "left":
        patch[1:-1, 0] = False
    else:
        patch[1:-1, -1] = False
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _place_edge_sprites(
    h: int, w: int, size: int, sprite_masks: dict[str, np.ndarray], sprite_colors: dict[str, int], rng: random.Random
) -> tuple[np.ndarray, dict[str, dict]]:
    bg_grid = np.full((h, w), -1, dtype=int)
    placements: dict[str, dict] = {}
    side_order = list(DIRS)
    rng.shuffle(side_order)
    side_for_dir = dict(zip(DIRS, side_order))

    side_boxes: dict[str, tuple[int, int, int, int]] = {}
    top_c = rng.randint(size + 1, w - 2 * size - 1)
    bottom_c = rng.randint(size + 1, w - 2 * size - 1)
    left_r = rng.randint(size + 1, h - 2 * size - 1)
    right_r = rng.randint(size + 1, h - 2 * size - 1)
    side_boxes["up"] = (0, top_c, size - 1, top_c + size - 1)
    side_boxes["down"] = (h - size, bottom_c, h - 1, bottom_c + size - 1)
    side_boxes["left"] = (left_r, 0, left_r + size - 1, size - 1)
    side_boxes["right"] = (right_r, w - size, right_r + size - 1, w - 1)

    assert not _boxes_overlap(side_boxes["up"], side_boxes["left"])
    assert not _boxes_overlap(side_boxes["up"], side_boxes["right"])
    assert not _boxes_overlap(side_boxes["down"], side_boxes["left"])
    assert not _boxes_overlap(side_boxes["down"], side_boxes["right"])

    for direction in DIRS:
        sr0, sc0, sr1, sc1 = side_boxes[side_for_dir[direction]]
        mask = sprite_masks[direction]
        color = sprite_colors[direction]
        bg_grid[sr0 : sr1 + 1, sc0 : sc1 + 1][mask] = color
        coords = np.argwhere(mask) + np.array([sr0, sc0])
        placements[direction] = {
            "mask": mask,
            "color": color,
            "bbox": (sr0, sc0, sr1, sc1),
            "coords": coords.astype(int),
        }
    return bg_grid, placements


def _place_frames(
    grid: np.ndarray,
    size: int,
    frame_specs: list[tuple[str, int]],
    forbidden: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> list[dict] | None:
    h, w = grid.shape
    outer = size + 2
    placed: list[dict] = []
    blocked = forbidden[:]

    for gap_dir, color in frame_specs:
        ok = False
        for _ in range(300):
            r0 = rng.randint(1, h - outer - 1)
            c0 = rng.randint(1, w - outer - 1)
            box = (r0, c0, r0 + outer - 1, c0 + outer - 1)
            if any(_boxes_overlap(box, other, pad=1) for other in blocked):
                continue
            patch = _make_frame_patch(size, gap_dir)
            grid[r0 : r0 + outer, c0 : c0 + outer][patch] = color
            placed.append({"gap": gap_dir, "color": color, "bbox": box})
            blocked.append(box)
            ok = True
            break
        if not ok:
            return None
    return placed


def _build_expected(
    inp: np.ndarray,
    bg: int,
    frames: list[dict],
    sprites: dict[str, dict],
) -> np.ndarray:
    out = inp.copy()
    for sprite in sprites.values():
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = bg
    for frame in frames:
        source_mask = sprites[OPPOSITE[frame["gap"]]]["mask"]
        oriented = _oriented_sprite(source_mask, frame["gap"])
        r0, c0, r1, c1 = frame["bbox"]
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[oriented] = int(frame["color"])
        patch[~oriented] = bg
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.randint(5, 6)
    n_frames = rng.randint(4, 5)
    outer = size + 2
    min_dim = max(3 * size + 8, 2 * outer + 8)
    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors[0]
    frame_palette = colors[1:5]
    sprite_palette = colors[5:9]

    sprite_masks: dict[str, np.ndarray] = {}
    sprite_shape_keys: set[tuple[tuple[int, int], bytes]] = set()
    for direction in DIRS:
        for _ in range(80):
            mask = _make_sprite_mask(size, direction, rng)
            key = _canonical_shape_key(mask)
            if key in sprite_shape_keys:
                continue
            sprite_shape_keys.add(key)
            sprite_masks[direction] = mask
            break
        else:
            return None

    sprite_colors = {direction: sprite_palette[i] for i, direction in enumerate(DIRS)}
    inp = np.full((h, w), bg, dtype=int)
    sprite_layer, sprite_places = _place_edge_sprites(h, w, size, sprite_masks, sprite_colors, rng)
    inp[sprite_layer != -1] = sprite_layer[sprite_layer != -1]

    frame_dirs = list(DIRS) + [rng.choice(DIRS) for _ in range(n_frames - 4)]
    rng.shuffle(frame_dirs)
    frame_specs = [(gap_dir, rng.choice(frame_palette)) for gap_dir in frame_dirs]
    sprite_boxes = [info["bbox"] for info in sprite_places.values()]
    frames = _place_frames(inp, size, frame_specs, sprite_boxes, rng)
    if frames is None:
        return None

    if _bg_color(inp) != bg:
        return None

    expected = _build_expected(inp, bg, frames, sprite_places)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
