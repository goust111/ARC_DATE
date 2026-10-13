from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
SPRITE_COLORS = (1, 2, 3, 4)
FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class SpriteInfo:
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    patch: np.ndarray
    height: int
    width: int


@dataclass(frozen=True)
class FrameInfo:
    bbox: tuple[int, int, int, int]
    inner_bbox: tuple[int, int, int, int]
    inner_height: int
    inner_width: int


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BACKGROUND) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_sprites(grid: np.ndarray) -> list[SpriteInfo]:
    sprites: list[SpriteInfo] = []
    for color in SPRITE_COLORS:
        comps = _components_of_color(grid, color)
        if not comps:
            continue
        assert len(comps) == 1, "each sprite color must appear in exactly one component"
        coords = comps[0]
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(grid, bbox)
        sprites.append(
            SpriteInfo(
                color=int(color),
                coords=coords,
                bbox=bbox,
                patch=patch,
                height=patch.shape[0],
                width=patch.shape[1],
            )
        )
    assert 2 <= len(sprites) <= 4, "expected 2-4 colored sprites"
    return sorted(sprites, key=lambda s: s.color)


def _extract_frames(grid: np.ndarray) -> list[FrameInfo]:
    frames: list[FrameInfo] = []
    for coords in _components_of_color(grid, FRAME_COLOR):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        outer_h = r1 - r0 + 1
        outer_w = c1 - c0 + 1
        assert outer_h >= 5 and outer_w >= 5, "frames must have a non-empty interior"
        sub = _crop_bbox(grid, bbox)
        border = _frame_border_mask(outer_h, outer_w)
        assert int(coords.shape[0]) == int(border.sum()), "grey component must be a 1-pixel hollow frame"
        assert np.all(sub[border] == FRAME_COLOR), "frame border must be grey"
        assert np.all(sub[~border] == BACKGROUND), "frame interior must be empty"
        frames.append(
            FrameInfo(
                bbox=bbox,
                inner_bbox=(r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                inner_height=outer_h - 2,
                inner_width=outer_w - 2,
            )
        )
    assert 3 <= len(frames) <= 5, "expected 3-5 grey frames"
    return sorted(frames, key=lambda f: f.bbox)


def _center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 + r1), int(c0 + c1)


def _compatible_options(sprite: SpriteInfo, frames: list[FrameInfo]) -> list[tuple[int, int, int]]:
    sr2, sc2 = _center2(sprite.bbox)
    options: list[tuple[int, int, int]] = []
    for frame_idx, frame in enumerate(frames):
        if sprite.height > frame.inner_height or sprite.width > frame.inner_width:
            continue
        gap = (frame.inner_height - sprite.height) + (frame.inner_width - sprite.width)
        fr2, fc2 = _center2(frame.inner_bbox)
        dist2 = (sr2 - fr2) ** 2 + (sc2 - fc2) ** 2
        options.append((int(gap), int(dist2), int(frame_idx)))
    return options


def _match_sprites_to_frames(sprites: list[SpriteInfo], frames: list[FrameInfo]) -> dict[int, int]:
    assignment: dict[int, int] = {}
    for sprite_idx, sprite in enumerate(sprites):
        options = _compatible_options(sprite, frames)
        assert options, "every sprite must fit at least one frame"
        min_gap = min(gap for gap, _dist2, _frame_idx in options)
        tied_by_gap = [item for item in options if item[0] == min_gap]
        min_dist2 = min(dist2 for _gap, dist2, _frame_idx in tied_by_gap)
        winners = [frame_idx for _gap, dist2, frame_idx in tied_by_gap if dist2 == min_dist2]
        assert len(winners) == 1, "distance tiebreak must resolve to one frame"
        assignment[sprite_idx] = winners[0]
    assert len(set(assignment.values())) == len(sprites), "frame matches must be one-to-one"
    return assignment


def _placement_in_frame(sprite: SpriteInfo, frame: FrameInfo) -> tuple[int, int]:
    ir0, ic0, _ir1, _ic1 = frame.inner_bbox
    top = ir0 + (frame.inner_height - sprite.height) // 2
    left = ic0 + (frame.inner_width - sprite.width) // 2
    return int(top), int(left)


def _render_output(shape: tuple[int, int], sprites: list[SpriteInfo], frames: list[FrameInfo], assignment: dict[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for sprite_idx, frame_idx in assignment.items():
        sprite = sprites[sprite_idx]
        frame = frames[frame_idx]
        r0, c0, r1, c1 = frame.bbox
        outer_h = r1 - r0 + 1
        outer_w = c1 - c0 + 1
        border = _frame_border_mask(outer_h, outer_w)
        patch = np.zeros((outer_h, outer_w), dtype=int)
        patch[border] = sprite.color
        _paste(out, patch, r0, c0)
        top, left = _placement_in_frame(sprite, frame)
        _paste(out, sprite.patch, top, left)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place each colored sprite into the best-fitting hollow frame and recolor the frame border to the sprite color.

    Input structure:
    - The grid has a black background with 2-4 single-color 4-connected sprites in colors 1-4.
    - It also has 3-5 grey hollow rectangular frames whose interiors are empty black rectangles.
    - Every sprite is separate from every frame, and at least one frame has no sprite match.

    Rule:
    - For each sprite, compute its bounding-box height and width, and compare them with every frame's interior size.
    - Keep only frames whose interior contains the sprite bbox; among them, choose the frame with minimum total slack
      `(frame_h - sprite_h) + (frame_w - sprite_w)`, breaking equal-slack ties by minimum Euclidean distance between
      the sprite bbox center and the frame-interior center.
    - Move each sprite, without rotation, so its bbox is centered inside the matched frame interior, biasing upward and
      leftward when exact centering is impossible. Recolor the matched frame border to the sprite color, erase unmatched
      frames, and erase every sprite from its original location.

    Key insight:
    - The same size-based matching decision controls both where a sprite moves and which frame gets its color, with
      distance mattering only when several frames offer the same minimum slack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sprites = _extract_sprites(g)
    frames = _extract_frames(g)
    assignment = _match_sprites_to_frames(sprites, frames)
    return _render_output(g.shape, sprites, frames, assignment)


def _random_connected_mask(height: int, width: int, area: int, rng: np.random.Generator) -> np.ndarray | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(80):
        start = cells[int(rng.integers(len(cells)))]
        chosen = {start}
        frontier = set()
        while len(chosen) < area:
            frontier.clear()
            for r, c in chosen:
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cell = tuple(frontier)[int(rng.integers(len(frontier)))]
            chosen.add(cell)
        if len(chosen) != area:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if _bbox_coords(np.argwhere(mask)) == (0, 0, height - 1, width - 1):
            return mask
    return None


def _random_sprite_patch(height: int, width: int, color: int, rng: np.random.Generator) -> np.ndarray | None:
    min_area = max(3, height + width - 1)
    max_area = min(8, height * width)
    if min_area > max_area:
        return None
    area = int(rng.integers(min_area, max_area + 1))
    mask = _random_connected_mask(height, width, area, rng)
    if mask is None:
        return None
    patch = np.zeros((height, width), dtype=int)
    patch[mask] = int(color)
    return patch


def _sample_sprite_specs(count: int, colors: Iterable[int], rng: np.random.Generator) -> list[np.ndarray] | None:
    dim_pool = [(h, w) for h in range(2, 6) for w in range(2, 6) if h + w <= 9]
    if len(dim_pool) < count:
        return None
    dims_idx = rng.permutation(len(dim_pool))[:count]
    patches: list[np.ndarray] = []
    for dim_idx, color in zip(dims_idx, colors):
        h, w = dim_pool[int(dim_idx)]
        patch = _random_sprite_patch(h, w, int(color), rng)
        if patch is None:
            return None
        patches.append(patch)
    return patches


def _valid_frame_sizes_for_sprite(sprite_h: int, sprite_w: int) -> list[tuple[int, int]]:
    sizes = []
    for inner_h in range(3, 8):
        for inner_w in range(3, 8):
            if sprite_h <= inner_h <= sprite_h + 2 and sprite_w <= inner_w <= sprite_w + 2:
                sizes.append((inner_h, inner_w))
    return sizes


def _sample_frame_sizes(
    sprite_patches: list[np.ndarray], extra_count: int, require_tie: bool, rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    target_sizes: list[tuple[int, int]] = []
    for patch in sprite_patches:
        candidates = _valid_frame_sizes_for_sprite(*patch.shape)
        if not candidates:
            return None
        scores = np.array([(fh - patch.shape[0]) + (fw - patch.shape[1]) for fh, fw in candidates], dtype=int)
        weight_map = {0: 2, 1: 4, 2: 4, 3: 2, 4: 1}
        weights = np.array([weight_map.get(int(score), 1) for score in scores], dtype=float)
        weights /= weights.sum()
        target_sizes.append(candidates[int(rng.choice(len(candidates), p=weights))])

    extras: list[tuple[int, int]] = []
    if require_tie:
        tie_candidates: list[tuple[int, list[tuple[int, int]]]] = []
        for sprite_idx, patch in enumerate(sprite_patches):
            sh, sw = patch.shape
            target_h, target_w = target_sizes[sprite_idx]
            target_gap = (target_h - sh) + (target_w - sw)
            alts = [
                size
                for size in _valid_frame_sizes_for_sprite(sh, sw)
                if size != (target_h, target_w) and (size[0] - sh) + (size[1] - sw) == target_gap
            ]
            if alts:
                tie_candidates.append((sprite_idx, alts))
        if not tie_candidates:
            return None
        _sprite_idx, alts = tie_candidates[int(rng.integers(len(tie_candidates)))]
        extras.append(alts[int(rng.integers(len(alts)))])

    while len(extras) < extra_count:
        extras.append((int(rng.integers(3, 8)), int(rng.integers(3, 8))))
    frame_sizes = target_sizes + extras
    rng.shuffle(frame_sizes)
    return frame_sizes


def _find_random_location(
    occupied: np.ndarray, patch_h: int, patch_w: int, rng: np.random.Generator, *, border: int = 1, padding: int = 1
) -> tuple[int, int] | None:
    h, w = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, h - border - patch_h + 1):
        for c0 in range(border, w - border - patch_w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + patch_h + padding)
            cc1 = min(w, c0 + patch_w + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _stamp_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    patch = np.zeros((outer_h, outer_w), dtype=int)
    patch[_frame_border_mask(outer_h, outer_w)] = FRAME_COLOR
    _paste(grid, patch, top, left)


def _place_objects(
    sprite_patches: list[np.ndarray], frame_sizes: list[tuple[int, int]], rng: np.random.Generator
) -> np.ndarray | None:
    grid_h = int(rng.integers(22, 29))
    grid_w = int(rng.integers(22, 29))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    items: list[tuple[str, int, int, int]] = []
    for idx, patch in enumerate(sprite_patches):
        items.append(("sprite", idx, patch.shape[0], patch.shape[1]))
    for idx, (inner_h, inner_w) in enumerate(frame_sizes):
        items.append(("frame", idx, inner_h + 2, inner_w + 2))
    items.sort(key=lambda item: item[2] * item[3], reverse=True)

    sprite_pos: dict[int, tuple[int, int]] = {}
    frame_pos: dict[int, tuple[int, int]] = {}
    for kind, idx, h, w in items:
        loc = _find_random_location(occupied, h, w, rng, border=1, padding=1)
        if loc is None:
            return None
        r0, c0 = loc
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        if kind == "sprite":
            sprite_pos[idx] = (r0, c0)
        else:
            frame_pos[idx] = (r0, c0)

    for idx, patch in enumerate(sprite_patches):
        _paste(grid, patch, *sprite_pos[idx])
    for idx, (inner_h, inner_w) in enumerate(frame_sizes):
        _stamp_frame(grid, *frame_pos[idx], inner_h, inner_w)
    return grid


def _has_distance_tiebreak(sprites: list[SpriteInfo], frames: list[FrameInfo], assignment: dict[int, int]) -> bool:
    for sprite_idx, sprite in enumerate(sprites):
        options = _compatible_options(sprite, frames)
        min_gap = min(gap for gap, _dist2, _frame_idx in options)
        tied_by_gap = [item for item in options if item[0] == min_gap]
        if len(tied_by_gap) < 2:
            continue
        dists = sorted({dist2 for _gap, dist2, _frame_idx in tied_by_gap})
        if len(dists) < 2:
            continue
        chosen_frame = assignment[sprite_idx]
        chosen_dist = next(dist2 for _gap, dist2, frame_idx in tied_by_gap if frame_idx == chosen_frame)
        if chosen_dist == dists[0]:
            return True
    return False


def _has_nonzero_gap_match(sprites: list[SpriteInfo], frames: list[FrameInfo], assignment: dict[int, int]) -> bool:
    for sprite_idx, frame_idx in assignment.items():
        sprite = sprites[sprite_idx]
        frame = frames[frame_idx]
        if frame.inner_height != sprite.height or frame.inner_width != sprite.width:
            return True
    return False


def _semantic_checks(inp: np.ndarray, out: np.ndarray, require_tie: bool) -> bool:
    sprites = _extract_sprites(inp)
    frames = _extract_frames(inp)
    assignment = _match_sprites_to_frames(sprites, frames)
    if require_tie and not _has_distance_tiebreak(sprites, frames, assignment):
        return False
    if not _has_nonzero_gap_match(sprites, frames, assignment):
        return False

    expected = _render_output(inp.shape, sprites, frames, assignment)
    if not np.array_equal(out, expected):
        return False

    matched_frames = set(assignment.values())
    for frame_idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame.bbox
        box = out[r0 : r1 + 1, c0 : c1 + 1]
        if frame_idx not in matched_frames and np.any(box != BACKGROUND):
            return False
    if np.any(out[(expected == BACKGROUND)] != BACKGROUND):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        sprite_count = int(rng.integers(2, 5))
        extra_count = int(rng.integers(1, 6 - sprite_count))
        require_tie = bool(rng.integers(2))
        colors = rng.choice(np.array(SPRITE_COLORS, dtype=int), size=sprite_count, replace=False)
        sprite_patches = _sample_sprite_specs(sprite_count, colors, rng)
        if sprite_patches is None:
            continue
        frame_sizes = _sample_frame_sizes(sprite_patches, extra_count, require_tie, rng)
        if frame_sizes is None:
            continue
        grid = _place_objects(sprite_patches, frame_sizes, rng)
        if grid is None:
            continue
        try:
            out = main(grid)
            if _semantic_checks(grid, out, require_tie):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid sprite-to-frame matching task")
