from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


BG = 0
STRUCT8 = np.ones((3, 3), dtype=int)
PALETTE = tuple(range(1, 10))
POSES = (
    "rot90",
    "rot180",
    "rot270",
    "flipud",
    "fliplr",
    "transpose",
    "anti_transpose",
)
OFFSETS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


@dataclass(frozen=True)
class Component:
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    sprite: np.ndarray
    center2: tuple[int, int]


def _apply_pose(arr: np.ndarray, pose: str) -> np.ndarray:
    if pose == "rot90":
        return np.rot90(arr, 1)
    if pose == "rot180":
        return np.rot90(arr, 2)
    if pose == "rot270":
        return np.rot90(arr, 3)
    if pose == "flipud":
        return np.flipud(arr)
    if pose == "fliplr":
        return np.fliplr(arr)
    if pose == "transpose":
        return arr.T
    if pose == "anti_transpose":
        return np.fliplr(np.flipud(arr)).T
    raise ValueError(f"unknown pose: {pose}")


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray) -> list[Component]:
    lab, n = label(grid != BG, structure=STRUCT8)
    comps: list[Component] = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == k)
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        sprite = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        comps.append(
            Component(
                coords=coords,
                bbox=bbox,
                sprite=sprite,
                center2=(int(r0 + r1), int(c0 + c1)),
            )
        )
    return comps


def _pair_components(
    comps: list[Component], shape: tuple[int, int]
) -> tuple[list[tuple[int, int]], int]:
    h, w = map(int, shape)
    lookup = {comp.center2: i for i, comp in enumerate(comps)}
    used = set()
    pairs: list[tuple[int, int]] = []
    widows: list[int] = []
    for i, comp in enumerate(comps):
        if i in used:
            continue
        rr = 2 * (h - 1) - comp.center2[0]
        cc = 2 * (w - 1) - comp.center2[1]
        j = lookup.get((rr, cc))
        if j is None or j == i:
            widows.append(i)
            used.add(i)
            continue
        used.add(i)
        used.add(j)
        if comp.center2 < comps[j].center2:
            pairs.append((i, j))
        else:
            pairs.append((j, i))
    assert len(widows) == 1
    assert len(pairs) * 2 + 1 == len(comps)
    return pairs, widows[0]


def _recolor(arr: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    for src, dst in color_map.items():
        out[arr == int(src)] = int(dst)
    return out


def _infer_color_map(
    source: np.ndarray,
    target: np.ndarray,
    pose: str,
    mapping: dict[int, int],
) -> dict[int, int] | None:
    transformed = _apply_pose(source, pose)
    if transformed.shape != target.shape:
        return None
    src_mask = transformed != BG
    tgt_mask = target != BG
    if not np.array_equal(src_mask, tgt_mask):
        return None

    trial = dict(mapping)
    for r, c in np.argwhere(src_mask):
        src_color = int(transformed[r, c])
        tgt_color = int(target[r, c])
        prev = trial.get(src_color)
        if prev is None:
            trial[src_color] = tgt_color
        elif prev != tgt_color:
            return None
    return trial


def _reflected_top_left(
    src_bbox: tuple[int, int, int, int],
    dst_shape: tuple[int, int],
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = map(int, grid_shape)
    dst_h, dst_w = map(int, dst_shape)
    src_center2_r = int(src_bbox[0] + src_bbox[2])
    src_center2_c = int(src_bbox[1] + src_bbox[3])
    dst_center2_r = 2 * (h - 1) - src_center2_r
    dst_center2_c = 2 * (w - 1) - src_center2_c
    r_num = dst_center2_r - (dst_h - 1)
    c_num = dst_center2_c - (dst_w - 1)
    if r_num % 2 or c_num % 2:
        return None
    r0 = r_num // 2
    c0 = c_num // 2
    if r0 < 0 or c0 < 0 or r0 + dst_h > h or c0 + dst_w > w:
        return None
    return int(r0), int(c0)


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    window = dst[top : top + h, left : left + w]
    mask = patch != BG
    if np.any(window[mask] != BG):
        return False
    window[mask] = patch[mask]
    return True


def _candidate_outputs(grid: np.ndarray) -> list[np.ndarray]:
    comps = _find_components(grid)
    assert len(comps) >= 3 and len(comps) % 2 == 1
    pairs, widow = _pair_components(comps, grid.shape)

    outputs: dict[bytes, np.ndarray] = {}
    widow_sprite = comps[widow].sprite
    widow_colors = {int(v) for v in widow_sprite.ravel() if int(v) != BG}
    assert widow_colors

    for pose in POSES:
        mapping: dict[int, int] = {}
        ok = True
        for src_idx, tgt_idx in pairs:
            maybe = _infer_color_map(comps[src_idx].sprite, comps[tgt_idx].sprite, pose, mapping)
            if maybe is None:
                ok = False
                break
            mapping = maybe
        if not ok:
            continue
        if not widow_colors <= set(mapping):
            continue

        widow_target = _recolor(_apply_pose(widow_sprite, pose), mapping)
        loc = _reflected_top_left(comps[widow].bbox, widow_target.shape, grid.shape)
        if loc is None:
            continue
        out = grid.copy()
        if not _paste_nonzero(out, widow_target, loc[0], loc[1]):
            continue
        key = out.tobytes() + str(out.shape).encode()
        outputs[key] = out
    return list(outputs.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the shared pose-change and recoloring shown by center-symmetric sprite pairs, then draw the widow's missing twin.

    Input structure:
    - A black background contains an odd number of separated, non-symmetric multicolor sprites.
    - All but one sprite form center-symmetric position pairs: the mate of a sprite is placed at the point-reflected location across the whole-grid center.
    - Within every complete pair, the mate is produced from its canonical-side partner by one shared rotation-or-reflection and one shared color-substitution table.

    Rule:
    - Pair sprites by point-symmetric placement across the grid center; the unique unpaired sprite is the widow.
    - Compare the completed pairs to infer the single common pose transform and the single common color map.
    - Apply that same composite transform to the widow and place the result at the widow's point-symmetric position. Keep all existing pixels unchanged.

    Key insight:
    - The visible complete pairs are implicit demonstrations of a global transform; simple center-copying fails because the missing twin must imitate the learned pose and recoloring, not the widow's original appearance.
    """
    g = np.asarray(grid, dtype=int)
    candidates = _candidate_outputs(g)
    assert len(candidates) == 1
    return candidates[0]


def _is_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    _, n = label(mask.astype(bool), structure=STRUCT8)
    return int(n) == 1


def _is_d4_asymmetric(mask: np.ndarray) -> bool:
    base = mask.astype(int)
    for pose in POSES:
        other = _apply_pose(base, pose)
        if other.shape == base.shape and np.array_equal(other, base):
            return False
    return True


def _random_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    for _ in range(300):
        density = float(rng.uniform(0.42, 0.68))
        mask = rng.random((h, w)) < density
        if int(mask.sum()) < max(5, (h * w) // 3) or int(mask.sum()) >= h * w:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _is_connected(mask):
            continue
        if not _is_d4_asymmetric(mask):
            continue
        return mask
    return None


def _colorize_mask(
    rng: np.random.Generator,
    mask: np.ndarray,
    palette: tuple[int, ...],
    min_colors: int = 2,
    max_colors: int = 4,
) -> np.ndarray:
    coords = np.argwhere(mask)
    color_count = int(rng.integers(min_colors, min(max_colors, len(coords), len(palette)) + 1))
    colors = list(rng.choice(np.array(palette), size=color_count, replace=False))
    rng.shuffle(coords)
    arr = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords[:color_count], colors):
        arr[int(r), int(c)] = int(color)
    for r, c in coords[color_count:]:
        arr[int(r), int(c)] = int(colors[int(rng.integers(len(colors)))])
    return arr


def _random_sprite(
    rng: np.random.Generator,
    pose: str,
    color_map: dict[int, int],
    palette: tuple[int, ...],
) -> np.ndarray | None:
    dims = [(3, 3), (3, 5), (5, 3), (5, 5), (4, 4)]
    for _ in range(300):
        h, w = dims[int(rng.integers(len(dims)))]
        mask = _random_mask(rng, h, w)
        if mask is None:
            continue
        sprite = _colorize_mask(rng, mask, palette)
        if len({int(v) for v in sprite.ravel() if int(v) != BG}) < 2:
            continue
        transformed = _recolor(_apply_pose(sprite, pose), color_map)
        if transformed.shape == sprite.shape and np.array_equal(transformed, sprite):
            continue
        return sprite
    return None


def _absolute_coords(patch: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords = set()
    for r, c in np.argwhere(patch != BG):
        coords.add((top + int(r), left + int(c)))
    return coords


def _blocked_from_reserved(reserved: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set()
    for r, c in reserved:
        for dr, dc in OFFSETS8:
            blocked.add((r + dr, c + dc))
    return blocked


def _placement_candidates(
    src: np.ndarray,
    dst: np.ndarray,
    grid_shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    border: int = 1,
) -> list[tuple[int, int, int, int, set[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = map(int, grid_shape)
    src_h, src_w = src.shape
    candidates = []
    for top in range(border, h - src_h - border + 1):
        for left in range(border, w - src_w - border + 1):
            src_center2_r = 2 * top + src_h - 1
            src_center2_c = 2 * left + src_w - 1
            if (src_center2_r, src_center2_c) >= (
                2 * (h - 1) - src_center2_r,
                2 * (w - 1) - src_center2_c,
            ):
                continue
            dst_loc = _reflected_top_left((top, left, top + src_h - 1, left + src_w - 1), dst.shape, grid_shape)
            if dst_loc is None:
                continue
            dst_top, dst_left = dst_loc
            if (
                dst_top < border
                or dst_left < border
                or dst_top + dst.shape[0] > h - border
                or dst_left + dst.shape[1] > w - border
            ):
                continue
            src_coords = _absolute_coords(src, top, left)
            dst_coords = _absolute_coords(dst, dst_top, dst_left)
            all_coords = src_coords | dst_coords
            if len(all_coords) != len(src_coords) + len(dst_coords):
                continue
            if all_coords & blocked:
                continue
            candidates.append((top, left, dst_top, dst_left, src_coords, dst_coords))
    return candidates


def _pose_and_map(rng: np.random.Generator) -> tuple[str, tuple[int, ...], dict[int, int]]:
    pose = POSES[int(rng.integers(len(POSES)))]
    palette_size = int(rng.integers(4, 6))
    palette = tuple(int(v) for v in rng.choice(np.array(PALETTE), size=palette_size, replace=False))
    shifted = palette[1:] + palette[:1]
    color_map = {int(src): int(dst) for src, dst in zip(palette, shifted)}
    return pose, palette, color_map


def _unique_sprite_family(
    rng: np.random.Generator,
    pose: str,
    color_map: dict[int, int],
    palette: tuple[int, ...],
    count: int,
) -> list[np.ndarray] | None:
    sprites: list[np.ndarray] = []
    seen_masks: set[bytes] = set()
    for _ in range(400):
        if len(sprites) == count:
            demo_colors = {
                int(v)
                for sprite in sprites[:-1]
                for v in sprite.ravel()
                if int(v) != BG
            }
            widow_colors = {int(v) for v in sprites[-1].ravel() if int(v) != BG}
            if set(palette) <= demo_colors and widow_colors <= demo_colors:
                return sprites
            sprites = []
            seen_masks.clear()
        sprite = _random_sprite(rng, pose, color_map, palette)
        if sprite is None:
            continue
        mask_key = (sprite != BG).astype(np.uint8).tobytes() + str(sprite.shape).encode()
        if mask_key in seen_masks:
            continue
        seen_masks.add(mask_key)
        sprites.append(sprite)
    return None


def _place_family(
    rng: np.random.Generator,
    sprites: list[np.ndarray],
    pose: str,
    color_map: dict[int, int],
    grid_shape: tuple[int, int],
    pair_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    transformed = [_recolor(_apply_pose(sprite, pose), color_map) for sprite in sprites]
    order = sorted(range(len(sprites)), key=lambda i: -int((sprites[i] != BG).sum() + (transformed[i] != BG).sum()))
    placements: dict[int, tuple[int, int, int, int, set[tuple[int, int]], set[tuple[int, int]]]] = {}
    reserved: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    def backtrack(k: int) -> bool:
        nonlocal reserved, blocked
        if k == len(order):
            return True
        idx = order[k]
        candidates = _placement_candidates(sprites[idx], transformed[idx], grid_shape, blocked)
        if not candidates:
            return False
        rng.shuffle(candidates)
        for cand in candidates:
            top, left, dst_top, dst_left, src_coords, dst_coords = cand
            placements[idx] = cand
            old_reserved = reserved
            reserved = reserved | src_coords | dst_coords
            old_blocked = blocked
            blocked = _blocked_from_reserved(reserved)
            if backtrack(k + 1):
                return True
            reserved = old_reserved
            blocked = old_blocked
            placements.pop(idx, None)
        return False

    if not backtrack(0):
        return None

    grid = np.zeros(grid_shape, dtype=int)
    expected = np.zeros(grid_shape, dtype=int)
    widow_idx = len(sprites) - 1
    for idx, sprite in enumerate(sprites):
        top, left, dst_top, dst_left, _src_coords, _dst_coords = placements[idx]
        _paste_nonzero(grid, sprite, top, left)
        _paste_nonzero(expected, sprite, top, left)
        if idx != widow_idx:
            _paste_nonzero(grid, transformed[idx], dst_top, dst_left)
        _paste_nonzero(expected, transformed[idx], dst_top, dst_left)

    if len(_find_components(grid)) != 2 * pair_count + 1:
        return None
    return grid, expected


def _validate_example(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    comps = _find_components(inp)
    if len(comps) < 5 or len(comps) % 2 != 1:
        return False
    if any(len({int(v) for v in comp.sprite.ravel() if int(v) != BG}) < 2 for comp in comps):
        return False
    if any(not _is_d4_asymmetric(comp.sprite != BG) for comp in comps):
        return False
    pairs, widow = _pair_components(comps, inp.shape)
    if len(pairs) < 2:
        return False
    if comps[widow].center2 == (inp.shape[0] - 1, inp.shape[1] - 1):
        return False
    candidates = _candidate_outputs(inp)
    if len(candidates) != 1:
        return False
    if not np.array_equal(candidates[0], expected):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid_shape = (
            int(rng.integers(19, 28)),
            int(rng.integers(19, 28)),
        )
        pair_count = int(rng.integers(2, 4))
        pose, palette, color_map = _pose_and_map(rng)
        sprites = _unique_sprite_family(rng, pose, color_map, palette, pair_count + 1)
        if sprites is None:
            continue
        placed = _place_family(rng, sprites, pose, color_map, grid_shape, pair_count)
        if placed is None:
            continue
        inp, expected = placed
        if _validate_example(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid widow-twin ARC input")
