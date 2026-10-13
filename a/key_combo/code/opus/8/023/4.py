from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MISSING_CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]
OPPOSITE_QUADRANT_TOPLEFT = {
    (0, 0): (1, 1),  # missing UL -> stamp in LR
    (0, 1): (1, 0),  # missing UR -> stamp in LL
    (1, 0): (0, 1),  # missing LL -> stamp in UR
    (1, 1): (0, 0),  # missing LR -> stamp in UL
}


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def components_of_color(grid: np.ndarray, color: int, *, connectivity: int = 4) -> list[dict]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    comps: list[dict] = []
    nbrs = DIR4 if connectivity == 4 else [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append({"coords": coords, "bbox": bbox_coords(coords), "area": int(coords.shape[0]), "color": int(color)})
    return comps


def _center2_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 + r1), int(c0 + c1)


def _manhattan_center2(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _has_enclosed_hole(grid: np.ndarray, comp: dict) -> bool:
    bbox = comp["bbox"]
    crop = crop_bbox(grid, bbox)
    h, w = crop.shape
    walls = crop == RED

    local = np.zeros((h + 2, w + 2), dtype=bool)
    local[1 : h + 1, 1 : w + 1] = walls

    seen = np.zeros_like(local, dtype=bool)
    q = deque()
    for r in range(h + 2):
        for c in (0, w + 1):
            if not local[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w + 2):
        for r in (0, h + 1):
            if not local[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not local[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    for r in range(h):
        for c in range(w):
            if crop[r, c] == BLACK and not seen[r + 1, c + 1]:
                return True
    return False


def _sprite_missing_corner(grid: np.ndarray, comp: dict) -> tuple[int, int]:
    bbox = comp["bbox"]
    r0, c0, r1, c1 = bbox
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "green sprite must fit in 2x2"
    sprite = crop_bbox(grid, bbox)
    missing = np.argwhere(sprite == BLACK)
    assert missing.shape == (1, 2), "green sprite must have one missing corner"
    return int(missing[0, 0]), int(missing[0, 1])


def _sprite_blue_patch(missing: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), BLUE, dtype=int)
    patch[missing] = BLACK
    return patch


def _stamp_topleft_for_target(bbox: tuple[int, int, int, int], missing: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    quad_r, quad_c = OPPOSITE_QUADRANT_TOPLEFT[missing]
    top = r0 + 1 if quad_r == 0 else r1 - 2
    left = c0 + 1 if quad_c == 0 else c1 - 2
    return int(top), int(left)


def _pair_sprites_to_targets(sprites: list[dict], targets: list[dict]) -> list[tuple[dict, dict]]:
    target_centers2 = [_center2_of_bbox(obj["bbox"]) for obj in targets]
    used_targets: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    for sprite in sprites:
        center2 = _center2_of_bbox(sprite["bbox"])
        dists = [_manhattan_center2(center2, tc) for tc in target_centers2]
        best = min(dists)
        winners = [i for i, d in enumerate(dists) if d == best]
        assert len(winners) == 1, "sprite-to-target pairing must be uniquely nearest"
        idx = winners[0]
        assert idx not in used_targets, "unique nearest pairing must be bijective"
        used_targets.add(idx)
        pairs.append((sprite, targets[idx]))
    assert len(used_targets) == len(targets) == len(sprites)
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the red frames with enclosed holes and stamp each paired L sprite into the opposite target quadrant.

    Input structure:
    - The grid has a black background, several red rectangular outline frames, and several green 3-cell L sprites.
    - Each red frame occupies its own bounding box; some frames are closed and enclose black interior holes, while others have a boundary gap so their interior connects to the outside.
    - Each green sprite is a 3-pixel monochrome L in a 2x2 bounding box, with one missing black corner.

    Rule:
    - Identify the red frames whose black pixels inside the frame bbox are not all 4-connected to the outside when the red pixels act as walls.
    - Ignore every red frame without such an enclosed black region.
    - Pair each green L sprite to the uniquely nearest valid red frame by Manhattan distance between sprite-center and frame-center.
    - Remove the original green sprite, then copy its 2x2 L shape in blue into the paired frame interior corner belonging to the quadrant opposite the sprite's missing corner.

    Key insight:
    - Target selection is topological rather than geometric, while placement depends only on the sprite orientation; both clues are required to know which frames are active and where each blue stamp goes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_components = components_of_color(g, RED, connectivity=4)
    green_components = components_of_color(g, GREEN, connectivity=4)
    assert red_components, "expected red frames"
    assert green_components, "expected green sprites"

    valid_targets = [comp for comp in red_components if _has_enclosed_hole(g, comp)]
    sprites = []
    for comp in green_components:
        assert comp["area"] == 3, "green component must contain exactly three cells"
        missing = _sprite_missing_corner(g, comp)
        sprites.append({**comp, "missing": missing})

    assert len(valid_targets) == len(sprites), "sprite count must match number of valid target frames"
    pairs = _pair_sprites_to_targets(sprites, valid_targets)

    out = g.copy()
    out[out == GREEN] = BLACK
    for sprite, target in pairs:
        patch = _sprite_blue_patch(sprite["missing"])
        rr, cc = _stamp_topleft_for_target(target["bbox"], sprite["missing"])
        paste(out, patch, rr, cc, transparent=BLACK)
    return out


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _can_place_bbox(placed: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int], *, pad: int = 1) -> bool:
    return all(not _rects_too_close(other, bbox, pad=pad) for other in placed)


def _draw_frame_patch(height: int, width: int, *, closed: bool) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = RED
    patch[-1, :] = RED
    patch[:, 0] = RED
    patch[:, -1] = RED
    if not closed:
        side = random.choice(["top", "bottom", "left", "right"])
        gap_len = random.choice([1, 1, 2])
        if side in ("top", "bottom"):
            row = 0 if side == "top" else height - 1
            start = random.randint(1, width - 1 - gap_len)
            patch[row, start : start + gap_len] = BLACK
        else:
            col = 0 if side == "left" else width - 1
            start = random.randint(1, height - 1 - gap_len)
            patch[start : start + gap_len, col] = BLACK
    return patch


def _draw_sprite_patch(missing: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), GREEN, dtype=int)
    patch[missing] = BLACK
    return patch


def _enumerate_sprite_positions(
    shape: tuple[int, int],
    placed: list[tuple[int, int, int, int]],
    target_bbox: tuple[int, int, int, int],
    target_idx: int,
    valid_bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    target_center2 = _center2_of_bbox(target_bbox)
    candidates: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            bbox = (r, c, r + 1, c + 1)
            if not _can_place_bbox(placed, bbox, pad=1):
                continue
            center2 = _center2_of_bbox(bbox)
            dists = [_manhattan_center2(center2, _center2_of_bbox(frame_bbox)) for frame_bbox in valid_bboxes]
            best = min(dists)
            if dists[target_idx] != best or sum(d == best for d in dists) != 1:
                continue
            if not (6 <= _manhattan_center2(center2, target_center2) <= 18):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _analysis(grid: np.ndarray) -> tuple[list[dict], list[dict], list[tuple[dict, dict]]]:
    red_components = components_of_color(grid, RED, connectivity=4)
    green_components = components_of_color(grid, GREEN, connectivity=4)
    valid_targets = [comp for comp in red_components if _has_enclosed_hole(grid, comp)]
    sprites = [{**comp, "missing": _sprite_missing_corner(grid, comp)} for comp in green_components]
    pairs = _pair_sprites_to_targets(sprites, valid_targets)
    return valid_targets, sprites, pairs


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(17, 24)
    width = random.randint(18, 26)
    total_frames = random.randint(3, 5)
    valid_count = random.randint(1, min(4, total_frames - 1))

    frame_specs = []
    frame_states = [True] * valid_count + [False] * (total_frames - valid_count)
    random.shuffle(frame_states)
    for closed in frame_states:
        frame_specs.append((random.randint(5, 8), random.randint(5, 8), closed))

    placed: list[tuple[int, int, int, int]] = []
    frame_records: list[dict] = []
    grid = np.zeros((height, width), dtype=int)

    for fh, fw, closed in sorted(frame_specs, key=lambda item: item[0] * item[1], reverse=True):
        placed_here = False
        for _ in range(250):
            r0 = random.randint(0, height - fh)
            c0 = random.randint(0, width - fw)
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if not _can_place_bbox(placed, bbox, pad=1):
                continue
            patch = _draw_frame_patch(fh, fw, closed=closed)
            paste(grid, patch, r0, c0, transparent=BLACK)
            placed.append(bbox)
            frame_records.append({"bbox": bbox, "closed": closed})
            placed_here = True
            break
        if not placed_here:
            return None

    valid_frame_records = [rec for rec in frame_records if rec["closed"]]
    random.shuffle(valid_frame_records)
    orientations = random.sample(MISSING_CORNERS, k=valid_count)

    for target_idx, (target, missing) in enumerate(zip(valid_frame_records, orientations)):
        candidates = _enumerate_sprite_positions(
            grid.shape,
            placed,
            target["bbox"],
            target_idx,
            [rec["bbox"] for rec in valid_frame_records],
        )
        if not candidates:
            return None
        chosen = None
        for r0, c0 in candidates[:80]:
            bbox = (r0, c0, r0 + 1, c0 + 1)
            if _can_place_bbox(placed, bbox, pad=1):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None
        patch = _draw_sprite_patch(missing)
        paste(grid, patch, chosen[0], chosen[1], transparent=BLACK)
        placed.append((chosen[0], chosen[1], chosen[0] + 1, chosen[1] + 1))

    try:
        valid_targets, sprites, pairs = _analysis(grid)
    except AssertionError:
        return None
    if len(valid_targets) != valid_count or len(sprites) != valid_count:
        return None
    if {sprite["missing"] for sprite in sprites} != set(orientations):
        return None

    out = main(grid)
    if np.any(out == GREEN):
        return None
    if np.count_nonzero(out == BLUE) != 3 * valid_count:
        return None

    paired_target_bboxes = {tuple(target["bbox"]) for _, target in pairs}
    if len(paired_target_bboxes) != valid_count:
        return None
    blue_mask = out == BLUE
    allowed_blue = np.zeros_like(out, dtype=bool)
    for sprite, target in pairs:
        rr, cc = _stamp_topleft_for_target(target["bbox"], sprite["missing"])
        patch = _sprite_blue_patch(sprite["missing"])
        expected = np.zeros_like(out)
        paste(expected, patch, rr, cc, transparent=BLACK)
        if not np.array_equal(out[rr : rr + 2, cc : cc + 2], expected[rr : rr + 2, cc : cc + 2]):
            return None
        allowed_blue[rr : rr + 2, cc : cc + 2] |= patch == BLUE

    if np.any(blue_mask & ~allowed_blue):
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
