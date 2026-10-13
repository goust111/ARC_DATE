from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


# Minimal self-contained helper subset mirroring the canonical arc_utils APIs.


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    r1 = int(r0) + patch.shape[0]
    c1 = int(c0) + patch.shape[1]
    dst[int(r0) : r1, int(c0) : c1] = patch


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    if int(connectivity) == 8:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _all_colors(grid: np.ndarray, *, exclude: Iterable[int] = ()) -> list[int]:
    banned = set(map(int, exclude))
    return sorted(int(v) for v in np.unique(grid) if int(v) not in banned)


def _is_solid_square(patch: np.ndarray, color: int) -> bool:
    h, w = patch.shape
    return h >= 2 and h == w and bool(np.all(patch == int(color)))


def _is_hollow_square_frame(patch: np.ndarray, color: int, bg: int) -> bool:
    h, w = patch.shape
    if h < 4 or h != w:
        return False
    if not bool(np.all(patch[0, :] == int(color))):
        return False
    if not bool(np.all(patch[-1, :] == int(color))):
        return False
    if not bool(np.all(patch[:, 0] == int(color))):
        return False
    if not bool(np.all(patch[:, -1] == int(color))):
        return False
    return bool(np.all(patch[1:-1, 1:-1] == int(bg)))


def _find_route_pairs(grid: np.ndarray) -> tuple[int, dict[int, dict[str, tuple[int, int, int, int]]]]:
    bg = _get_bg(grid)
    pairs: dict[int, dict[str, tuple[int, int, int, int]]] = {}
    for color in _all_colors(grid, exclude={bg}):
        solids: list[tuple[int, int, int, int]] = []
        frames: list[tuple[int, int, int, int]] = []
        for coords in _components_mask(grid == int(color), connectivity=4):
            bbox = _bbox_coords(coords)
            patch = _crop_bbox(grid, bbox)
            if _is_solid_square(patch, color):
                solids.append(bbox)
            elif _is_hollow_square_frame(patch, color, bg):
                frames.append(bbox)
        if len(solids) == 1 and len(frames) == 1:
            sb = solids[0]
            fb = frames[0]
            solid_size = sb[2] - sb[0] + 1
            frame_size = fb[2] - fb[0] + 1
            if frame_size == solid_size + 2:
                pairs[int(color)] = {"occluder": sb, "frame": fb}
    return int(bg), pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each hidden symmetric patch and deliver it into the matching same-color hollow frame.

    Input structure:
    - A central multicolor pattern is 180-degree rotationally symmetric about the whole-grid center.
    - Several unique-color solid square occluders cover square patches of that pattern.
    - Outside the pattern sit same-color hollow square frames whose interiors match the occluder sizes
      and are filled with the background.

    Rule:
    - Identify each occluder color together with its same-color hollow frame.
    - Rotate the whole grid by 180 degrees and crop the pixels that land on the occluder bbox;
      this yields the hidden patch in the puzzle's global reference frame.
    - Paste that recovered patch unchanged into the interior of the matching frame.
    - Leave every other pixel unchanged.

    Key insight:
    - The occluder color is not image content; it is only a routing key. The missing patch can be
      recovered only from the pattern's global 180-degree symmetry, not from any local cue.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _bg, pairs = _find_route_pairs(g)
    rotated = np.rot90(g, 2)

    for color in sorted(pairs):
        occluder = pairs[color]["occluder"]
        frame = pairs[color]["frame"]
        patch = _crop_bbox(rotated, occluder)
        fr0, fc0, fr1, fc1 = frame
        expected_side = occluder[2] - occluder[0] + 1
        if patch.shape != (expected_side, expected_side):
            continue
        if fr1 - fr0 - 1 != expected_side or fc1 - fc0 - 1 != expected_side:
            continue
        _paste(out, patch, fr0 + 1, fc0 + 1)
    return out


ALL_COLORS = tuple(range(10))
GEN_RETRIES = 500
SCENE_RETRIES = 120
PATCH_RETRIES = 120
PLACEMENT_RETRIES = 200


def _rot_bbox(bbox: tuple[int, int, int, int], H: int, W: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return int(H - 1 - r1), int(W - 1 - c1), int(H - 1 - r0), int(W - 1 - c0)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 <= ir0 and ic0 >= oc0 and ir1 <= or1 and ic1 <= oc1


def _paint_pattern(grid: np.ndarray, bbox: tuple[int, int, int, int], colors: list[int], rng: np.random.Generator) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(colors[0])

    n_rects = int(rng.integers(18, 34))
    max_h = min(4, r1 - r0 + 1)
    max_w = min(4, c1 - c0 + 1)
    for _ in range(n_rects):
        color = int(rng.choice(colors))
        sh = int(rng.integers(1, max_h + 1))
        sw = int(rng.integers(1, max_w + 1))
        rr = int(rng.integers(r0, r1 - sh + 2))
        cc = int(rng.integers(c0, c1 - sw + 2))
        grid[rr : rr + sh, cc : cc + sw] = color
        rr2 = H - (rr + sh)
        cc2 = W - (cc + sw)
        grid[rr2 : rr2 + sh, cc2 : cc2 + sw] = color

    n_lines = int(rng.integers(8, 16))
    for _ in range(n_lines):
        color = int(rng.choice(colors))
        if bool(rng.integers(0, 2)):
            sh = 1
            sw = int(rng.integers(2, min(6, c1 - c0 + 1) + 1))
        else:
            sh = int(rng.integers(2, min(6, r1 - r0 + 1) + 1))
            sw = 1
        rr = int(rng.integers(r0, r1 - sh + 2))
        cc = int(rng.integers(c0, c1 - sw + 2))
        grid[rr : rr + sh, cc : cc + sw] = color
        rr2 = H - (rr + sh)
        cc2 = W - (cc + sw)
        grid[rr2 : rr2 + sh, cc2 : cc2 + sw] = color


def _sample_sizes(rng: np.random.Generator, num_pairs: int, max_size: int) -> list[int] | None:
    options = [s for s in (2, 3, 4) if s <= int(max_size)]
    if not options:
        return None
    repeated = int(rng.choice(options))
    sizes = [repeated, repeated]
    if num_pairs == 3:
        sizes.append(int(rng.choice(options)))
    rng.shuffle(sizes)
    return [int(v) for v in sizes]


def _place_occluders(
    rng: np.random.Generator,
    pattern_bbox: tuple[int, int, int, int],
    H: int,
    W: int,
    route_colors: list[int],
    sizes: list[int],
) -> list[dict[str, object]] | None:
    pr0, pc0, pr1, pc1 = pattern_bbox
    reserved: list[tuple[int, int, int, int]] = []
    specs: list[dict[str, object]] = []

    for color, size in zip(route_colors, sizes):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = int(rng.integers(pr0 + 1, pr1 - size + 1))
            c0 = int(rng.integers(pc0 + 1, pc1 - size + 1))
            occ = (r0, c0, r0 + size - 1, c0 + size - 1)
            vis = _rot_bbox(occ, H, W)
            if not _bbox_inside(occ, pattern_bbox) or not _bbox_inside(vis, pattern_bbox):
                continue
            if _boxes_overlap(occ, vis, pad=0):
                continue
            if any(_boxes_overlap(occ, old, pad=1) or _boxes_overlap(vis, old, pad=1) for old in reserved):
                continue
            specs.append(
                {
                    "color": int(color),
                    "size": int(size),
                    "occluder_bbox": occ,
                    "visible_bbox": vis,
                }
            )
            reserved.extend([occ, vis])
            placed = True
            break
        if not placed:
            return None
    return specs


def _design_patch(
    rng: np.random.Generator,
    size: int,
    pattern_colors: list[int],
    taken: set[bytes],
) -> np.ndarray | None:
    for _ in range(PATCH_RETRIES):
        n_colors = 2 if size == 2 else int(rng.choice([2, 3]))
        chosen = list(map(int, rng.choice(pattern_colors, size=n_colors, replace=False)))
        patch = rng.choice(chosen, size=(size, size)).astype(int)
        if len(np.unique(patch)) < 2:
            continue
        if np.array_equal(patch, np.rot90(patch, 2)):
            continue
        sig = patch.tobytes()
        rot_sig = np.rot90(patch, 2).tobytes()
        if sig in taken or rot_sig in taken:
            continue
        return patch
    return None


def _fill_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _set_patch(grid: np.ndarray, bbox: tuple[int, int, int, int], patch: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _mark_occupied(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    H, W = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H - 1, r1 + pad)
    cc1 = min(W - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _frame_candidates(
    occupied: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    outer_size: int,
) -> list[tuple[int, int, int, int]]:
    H, W = occupied.shape
    out: list[tuple[int, int, int, int]] = []
    for r0 in range(1, H - outer_size):
        for c0 in range(1, W - outer_size):
            bbox = (r0, c0, r0 + outer_size - 1, c0 + outer_size - 1)
            if _boxes_overlap(bbox, pattern_bbox, pad=1):
                continue
            if bool(occupied[r0 : r0 + outer_size, c0 : c0 + outer_size].any()):
                continue
            out.append(bbox)
    return out


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for spec in specs:
        frame = spec["frame_bbox"]
        patch = spec["patch"]
        fr0, fc0, fr1, fc1 = frame
        _paste(out, patch, fr0 + 1, fc0 + 1)
        assert fr1 - fr0 - 1 == patch.shape[0]
        assert fc1 - fc0 - 1 == patch.shape[1]
    return out


def _scene_valid(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    bg: int,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if _get_bg(grid) != int(bg):
        return False

    route_colors = {int(spec["color"]) for spec in specs}
    _, pairs = _find_route_pairs(grid)
    if set(pairs) != route_colors:
        return False

    rotated = np.rot90(grid, 2)
    for spec in specs:
        color = int(spec["color"])
        occ = spec["occluder_bbox"]
        vis = spec["visible_bbox"]
        frame = spec["frame_bbox"]
        patch = spec["patch"]
        if pairs[color]["occluder"] != occ or pairs[color]["frame"] != frame:
            return False
        if len(np.unique(patch)) < 2:
            return False
        recovered = _crop_bbox(rotated, occ)
        if not np.array_equal(recovered, patch):
            return False
        visible = _crop_bbox(grid, vis)
        if not np.array_equal(visible, np.rot90(patch, 2)):
            return False
        if np.array_equal(visible, patch):
            return False
        if bool(np.isin(visible, list(route_colors)).any()):
            return False

    if len(specs) > 1:
        sizes = [int(spec["size"]) for spec in specs]
        if len(set(sizes)) == len(sizes):
            return False
        occ_order = [int(spec["color"]) for spec in sorted(specs, key=lambda s: tuple(s["occluder_bbox"][:2]))]
        frame_order = [int(spec["color"]) for spec in sorted(specs, key=lambda s: tuple(s["frame_bbox"][:2]))]
        if occ_order == frame_order:
            return False

    return bool(np.array_equal(main(grid), _expected_output(grid, specs)))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_pairs = int(rng.choice([2, 3]))
    margin_r = int(rng.integers(6, 8))
    margin_c = int(rng.integers(6, 8))
    pattern_h = int(rng.integers(12, 17))
    pattern_w = int(rng.integers(12, 17))
    H = pattern_h + 2 * margin_r
    W = pattern_w + 2 * margin_c
    if H > 30 or W > 30:
        return None

    bg = int(rng.integers(0, 10))
    palette = [c for c in ALL_COLORS if c != bg]
    rng.shuffle(palette)
    route_colors = [int(v) for v in palette[:num_pairs]]
    pattern_colors = [int(v) for v in palette[num_pairs : num_pairs + 4]]
    if len(pattern_colors) < 3:
        return None

    pattern_bbox = (margin_r, margin_c, margin_r + pattern_h - 1, margin_c + pattern_w - 1)
    sizes = _sample_sizes(rng, num_pairs=num_pairs, max_size=min(4, pattern_h - 2, pattern_w - 2))
    if sizes is None:
        return None

    specs = _place_occluders(rng, pattern_bbox, H, W, route_colors, sizes)
    if specs is None:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    _paint_pattern(grid, pattern_bbox, pattern_colors, rng)

    taken: set[bytes] = set()
    for spec in specs:
        patch = _design_patch(rng, int(spec["size"]), pattern_colors, taken)
        if patch is None:
            return None
        spec["patch"] = patch
        taken.add(patch.tobytes())
        taken.add(np.rot90(patch, 2).tobytes())
        _set_patch(grid, spec["occluder_bbox"], patch)
        _set_patch(grid, spec["visible_bbox"], np.rot90(patch, 2))

    for spec in specs:
        _fill_bbox(grid, spec["occluder_bbox"], int(spec["color"]))

    occupied = np.zeros((H, W), dtype=bool)
    _mark_occupied(occupied, pattern_bbox, pad=1)
    for idx in rng.permutation(len(specs)):
        spec = specs[int(idx)]
        outer = int(spec["size"]) + 2
        candidates = _frame_candidates(occupied, pattern_bbox, outer)
        if not candidates:
            return None
        frame = candidates[int(rng.integers(0, len(candidates)))]
        spec["frame_bbox"] = frame
        _draw_frame(grid, frame, int(spec["color"]))
        _mark_occupied(occupied, frame, pad=1)

    if not _scene_valid(grid, specs, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scene")
