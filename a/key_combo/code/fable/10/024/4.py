from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(cells, dtype=int))
    return parts


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    pieces = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _components(grid == color):
            pieces.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return pieces


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return mask


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    base = _crop_bool(mask.astype(bool))
    variants = []
    for k in range(4):
        rot = np.rot90(base, k)
        for flip in (False, True):
            view = np.fliplr(rot) if flip else rot
            tight = _crop_bool(view.astype(bool))
            variants.append((tight.shape, tuple(int(v) for v in tight.ravel())))
    return min(variants)


def _enclosed_holes(region: np.ndarray, bg: int) -> list[np.ndarray]:
    holes = []
    for coords in _components(region == bg):
        rows = coords[:, 0]
        cols = coords[:, 1]
        touches_border = (
            np.any(rows == 0)
            or np.any(rows == region.shape[0] - 1)
            or np.any(cols == 0)
            or np.any(cols == region.shape[1] - 1)
        )
        if not touches_border:
            holes.append(coords)
    return holes


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames: list[dict] = []
    sprites: list[dict] = []

    for comp in _find_monochrome_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        holes = _enclosed_holes(region, bg)

        if len(holes) == 1:
            hole_local = holes[0]
            hole_mask = _coords_to_mask(hole_local)
            hole_abs = hole_local + np.array([r0, c0], dtype=int)
            frames.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "hole_abs": hole_abs,
                    "hole_mask": hole_mask,
                    "sig": _shape_signature(hole_mask),
                }
            )
        else:
            assert len(holes) == 0, "non-frame component should not contain a hole"
            sprite_mask = _coords_to_mask(comp["coords"])
            sprites.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "mask": sprite_mask,
                    "sig": _shape_signature(sprite_mask),
                }
            )

    return frames, sprites, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each free sprite to the congruent frame hole, then swap the frame/sprite colors.

    Input structure:
    - A dominant background contains several separated single-color hollow frames; each encloses one background hole.
    - Elsewhere lie one-color solid free sprites, with exactly one more sprite than frame.
    - Each hole shape matches exactly one sprite up to rotation or reflection; one sprite matches nothing.

    Rule:
    - For every hollow frame, extract the enclosed background hole as the lock shape.
    - Find the unique free sprite congruent to that hole under dihedral transforms, erase the sprite, fill the hole with the frame's original color, and recolor the frame border to the sprite's color.
    - Keep the single unmatched sprite unchanged.

    Key insight:
    - The visible frame border is not what matches the sprite; the decisive shape is the negative-space hole inside the frame, and solving the task requires both matching that hidden shape and performing the resulting two-color swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, sprites, bg = _analyze_scene(g)
    assert frames, "expected at least one hollow frame"
    assert len(sprites) == len(frames) + 1, "expected exactly one extra sprite"

    sig_to_sprites: dict[tuple[tuple[int, int], tuple[int, ...]], list[int]] = defaultdict(list)
    for idx, sprite in enumerate(sprites):
        sig_to_sprites[sprite["sig"]].append(idx)

    out = g.copy()
    used: set[int] = set()
    for frame in frames:
        candidates = [idx for idx in sig_to_sprites.get(frame["sig"], []) if idx not in used]
        assert len(candidates) == 1, "each hole must match exactly one remaining sprite"
        sprite = sprites[candidates[0]]
        used.add(candidates[0])

        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = bg
        out[frame["coords"][:, 0], frame["coords"][:, 1]] = int(sprite["color"])
        out[frame["hole_abs"][:, 0], frame["hole_abs"][:, 1]] = int(frame["color"])

    assert len(sprites) - len(used) == 1, "expected exactly one unmatched sprite"
    return out


SHAPES: dict[str, np.ndarray] = {
    "square2": np.array([[1, 1], [1, 1]], dtype=bool),
    "rect23": np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    "L4": np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    "T4": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "Z4": np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    "P5": np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    "U5": np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    "plus5": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "L5": np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=bool),
}


def _shape_meta() -> dict[str, dict]:
    meta = {}
    for name, mask in SHAPES.items():
        meta[name] = {
            "mask": mask.copy(),
            "sig": _shape_signature(mask),
            "area": int(mask.sum()),
            "dims": tuple(sorted(mask.shape)),
            "rect": bool(mask.all()),
        }
    return meta


SHAPE_META = _shape_meta()


def _same_box_group() -> list[list[str]]:
    groups: dict[tuple[int, tuple[int, int]], list[str]] = defaultdict(list)
    for name, info in SHAPE_META.items():
        groups[(info["area"], info["dims"])].append(name)
    return [names for names in groups.values() if len(names) >= 2]


CONFUSION_GROUPS = _same_box_group()


def _orient_mask(mask: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(mask.astype(bool), int(k))
    out = np.fliplr(out) if flip else out
    return out.astype(bool)


def _unique_oriented_masks(mask: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    base_sig = _shape_signature(mask)
    choices = []
    seen = set()
    for k in range(4):
        for flip in (False, True):
            oriented = _orient_mask(mask, k, flip)
            key = (oriented.shape, tuple(int(v) for v in oriented.ravel()))
            if key not in seen:
                seen.add(key)
                choices.append((oriented, _shape_signature(oriented) != base_sig or key != (mask.shape, tuple(int(v) for v in mask.ravel()))))
    return choices


def _build_frame_patch(hole_mask: np.ndarray, color: int) -> np.ndarray:
    h, w = hole_mask.shape
    padded = np.pad(hole_mask.astype(bool), 1)
    dilated = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in np.argwhere(padded):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                dilated[r + dr, c + dc] = True
    frame = dilated & ~padded
    patch = np.zeros((h + 2, w + 2), dtype=int)
    patch[frame] = int(color)
    return patch


def _paint_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _random_location(occupied: np.ndarray, patch_shape: tuple[int, int], *, border: int = 1, gap: int = 1) -> tuple[int, int]:
    h, w = patch_shape
    H, W = occupied.shape
    candidates = []
    for r0 in range(border, H - border - h + 1):
        for c0 in range(border, W - border - w + 1):
            rr0 = max(0, r0 - gap)
            cc0 = max(0, c0 - gap)
            rr1 = min(H, r0 + h + gap)
            cc1 = min(W, c0 + w + gap)
            if not occupied[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no free location")
    return random.choice(candidates)


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int]:
    r0, c0 = _random_location(occupied, patch.shape, border=1, gap=1)
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w][patch != 0] = patch[patch != 0]
    occupied[r0 : r0 + h, c0 : c0 + w] = True
    return r0, c0


def _choose_shapes(n_frames: int) -> tuple[list[str], str]:
    big_groups = [g for g in CONFUSION_GROUPS if len(g) >= 3]
    source_group = random.choice(big_groups if n_frames >= 3 and big_groups else CONFUSION_GROUPS)

    hole_names = []
    take = 2 if len(source_group) >= 3 and n_frames >= 2 else 1
    hole_names.extend(random.sample(source_group, take))

    extra_name = random.choice([name for name in source_group if name not in hole_names])
    remaining = [name for name in SHAPES if name not in hole_names and name != extra_name]
    random.shuffle(remaining)
    for name in remaining:
        if len(hole_names) == n_frames:
            break
        hole_names.append(name)

    if len(hole_names) != n_frames:
        raise RuntimeError("failed to sample enough hole shapes")
    if all(SHAPE_META[name]["rect"] for name in hole_names):
        raise RuntimeError("all holes became rectangles")
    return hole_names, extra_name


def _simulate_output(grid: np.ndarray, frame_parts: list[dict], sprite_parts: list[dict], extra_idx: int) -> np.ndarray:
    out = grid.copy()
    for frame, sprite in zip(frame_parts, sprite_parts):
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = 0
        out[frame["frame_coords"][:, 0], frame["frame_coords"][:, 1]] = int(sprite["color"])
        out[frame["hole_coords"][:, 0], frame["hole_coords"][:, 1]] = int(frame["color"])
    return out


def _is_confuser(extra_mask: np.ndarray, hole_masks: list[np.ndarray]) -> bool:
    extra_sig = _shape_signature(extra_mask)
    extra_area = int(extra_mask.sum())
    extra_dims = tuple(sorted(extra_mask.shape))
    for hole_mask in hole_masks:
        if _shape_signature(hole_mask) == extra_sig:
            return False
        if int(hole_mask.sum()) == extra_area and tuple(sorted(hole_mask.shape)) == extra_dims:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    try:
        hole_names, extra_name = _choose_shapes(n_frames)
    except RuntimeError:
        return None

    total_objects = 2 * n_frames + 1
    colors = random.sample(range(1, 10), total_objects)
    frame_colors = colors[:n_frames]
    sprite_colors = colors[n_frames : 2 * n_frames]
    extra_color = colors[-1]

    frame_specs = []
    sprite_specs = []
    transformed_used = False
    for idx, name in enumerate(hole_names):
        hole_mask = SHAPE_META[name]["mask"]
        frame_patch = _build_frame_patch(hole_mask, frame_colors[idx])

        sprite_choices = []
        base_key = (hole_mask.shape, tuple(int(v) for v in hole_mask.ravel()))
        seen = set()
        for k in range(4):
            for flip in (False, True):
                oriented = _orient_mask(hole_mask, k, flip)
                key = (oriented.shape, tuple(int(v) for v in oriented.ravel()))
                if key not in seen:
                    seen.add(key)
                    sprite_choices.append((oriented, key != base_key))

        if idx == 0:
            nontrivial = [item for item in sprite_choices if item[1]]
            if nontrivial:
                sprite_mask, changed = random.choice(nontrivial)
            else:
                sprite_mask, changed = random.choice(sprite_choices)
        else:
            sprite_mask, changed = random.choice(sprite_choices)
        transformed_used |= bool(changed)

        frame_specs.append(
            {
                "name": name,
                "color": frame_colors[idx],
                "patch": frame_patch,
                "hole_mask": hole_mask,
            }
        )
        sprite_specs.append(
            {
                "name": name,
                "color": sprite_colors[idx],
                "patch": _paint_patch(sprite_mask, sprite_colors[idx]),
                "mask": sprite_mask,
            }
        )

    extra_mask = SHAPE_META[extra_name]["mask"]
    if not _is_confuser(extra_mask, [spec["hole_mask"] for spec in frame_specs]):
        return None
    extra_patch = _paint_patch(extra_mask, extra_color)

    min_side = 16 + 2 * n_frames
    H = random.randint(min_side, 30)
    W = random.randint(min_side, 30)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    placements = []
    objects = []
    for frame in frame_specs:
        objects.append(("frame", frame))
    for sprite in sprite_specs:
        objects.append(("sprite", sprite))
    objects.append(("extra", {"name": extra_name, "color": extra_color, "patch": extra_patch, "mask": extra_mask}))
    random.shuffle(objects)
    objects.sort(key=lambda item: item[1]["patch"].shape[0] * item[1]["patch"].shape[1], reverse=True)

    try:
        for kind, obj in objects:
            r0, c0 = _place_patch(grid, occupied, obj["patch"])
            placements.append((kind, obj, r0, c0))
    except ValueError:
        return None

    frame_parts = []
    sprite_parts = []
    extra_idx = -1
    frame_by_name = {}
    sprite_by_name = {}

    for kind, obj, r0, c0 in placements:
        patch = obj["patch"]
        fg_coords = np.argwhere(patch != 0) + np.array([r0, c0], dtype=int)
        if kind == "frame":
            hole_coords = np.argwhere(patch == 0) + np.array([r0, c0], dtype=int)
            info = {
                "name": obj["name"],
                "color": obj["color"],
                "frame_coords": fg_coords,
                "hole_coords": hole_coords,
                "hole_mask": obj["hole_mask"],
            }
            frame_parts.append(info)
            frame_by_name[obj["name"]] = info
        elif kind == "sprite":
            info = {
                "name": obj["name"],
                "color": obj["color"],
                "coords": fg_coords,
                "mask": obj["mask"],
            }
            sprite_parts.append(info)
            sprite_by_name[obj["name"]] = info
        else:
            extra_idx = len(sprite_parts)
            sprite_parts.append(
                {
                    "name": obj["name"],
                    "color": obj["color"],
                    "coords": fg_coords,
                    "mask": obj["mask"],
                }
            )

    if set(frame_by_name) != set(sprite_by_name):
        return None
    ordered_frames = [frame_by_name[name] for name in hole_names]
    ordered_sprites = [sprite_by_name[name] for name in hole_names]
    expected = _simulate_output(grid, ordered_frames, ordered_sprites, extra_idx)

    if not transformed_used:
        return None

    bg_counts = np.bincount(grid.ravel(), minlength=10)
    if int(bg_counts[0]) <= int(max(bg_counts[1:])):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        frames, sprites, _ = _analyze_scene(grid)
    except AssertionError:
        return None
    if len(frames) != n_frames or len(sprites) != n_frames + 1:
        return None
    if len({frame["sig"] for frame in frames}) != n_frames:
        return None
    if all(frame["hole_mask"].all() for frame in frames):
        return None
    hole_masks = [frame["hole_mask"] for frame in frames]
    extra_candidates = [sprite for sprite in sprites if sprite["sig"] not in {frame["sig"] for frame in frames}]
    if len(extra_candidates) != 1:
        return None
    if not _is_confuser(extra_candidates[0]["mask"], hole_masks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
