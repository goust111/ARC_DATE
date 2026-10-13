from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg


GREY = Color.GRAY
RED = Color.RED
BLUE = Color.BLUE
BLACK = Color.BLACK
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_GEN_RETRY_LIMIT = 400


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return mask


def _extract_sprite(coords: np.ndarray) -> dict[str, object]:
    coords = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = _mask_from_coords(coords)
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "area": int(coords.shape[0]),
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
    }


def _extract_frame(coords: np.ndarray) -> dict[str, object]:
    coords = np.asarray(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h >= 3 and w >= 3, "frame must have a non-empty interior"

    mask = _mask_from_coords(coords)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.array_equal(mask, border), "red object must be a 1-pixel hollow rectangle"

    inner_h = h - 2
    inner_w = w - 2
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "h": h,
        "w": w,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "inner_area": int(inner_h * inner_w),
    }


def _find_sprites(grid: np.ndarray) -> list[dict[str, object]]:
    return [_extract_sprite(coords) for coords in components_mask(grid == GREY, connectivity=4)]


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    return [_extract_frame(coords) for coords in components_mask(grid == RED, connectivity=4)]


def _mirrored_coords_in_frame(sprite: dict[str, object], frame: dict[str, object]) -> list[tuple[int, int]]:
    mask = np.asarray(sprite["mask"], dtype=bool)
    sprite_h = int(sprite["h"])
    sprite_w = int(sprite["w"])
    inner_h = int(frame["inner_h"])
    inner_w = int(frame["inner_w"])
    assert sprite_h <= inner_h and sprite_w <= inner_w, "sprite must fit inside the paired frame"

    r0, c0, _r1, _c1 = map(int, frame["bbox"])
    top = r0 + 1 + (inner_h - sprite_h) // 2
    left = c0 + 1 + (inner_w - sprite_w) // 2

    placed: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(mask):
        out_r = top + int(rr)
        out_c = left + (sprite_w - 1 - int(cc))
        placed.append((out_r, out_c))
    return placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair gray sprites with red frames by descending area, then mirror each sprite into its matched frame.

    Input structure:
    - The grid has black background, several gray 4-connected sprites, and the same number of red hollow rectangles.
    - Sprite pixel counts are all distinct, and frame interior areas are all distinct.
    - Sprites and frames are spatially separated, so proximity alone does not determine the correct pairing.

    Rule:
    - Compute each gray sprite's pixel area and each red frame's interior area.
    - Sort sprites and frames from largest to smallest by those areas and pair equal ranks.
    - For each pair, place a horizontally mirrored blue copy of the sprite inside the frame interior, centered within that interior.
    - Erase the original gray sprites and keep the red frame borders.

    Key insight:
    - The decisive relation is rank-by-area, not nearest-neighbor matching; the frame only tells you where the mirrored copy must land.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    sprites = _find_sprites(g)
    frames = _find_frames(g)
    assert sprites and len(sprites) == len(frames), "expected the same number of sprites and frames"

    sprite_areas = [int(sprite["area"]) for sprite in sprites]
    frame_areas = [int(frame["inner_area"]) for frame in frames]
    assert len(set(sprite_areas)) == len(sprite_areas), "sprite areas must be unique"
    assert len(set(frame_areas)) == len(frame_areas), "frame interior areas must be unique"

    ranked_sprites = sorted(
        sprites,
        key=lambda sprite: (-int(sprite["area"]), int(sprite["bbox"][0]), int(sprite["bbox"][1])),
    )
    ranked_frames = sorted(
        frames,
        key=lambda frame: (-int(frame["inner_area"]), int(frame["bbox"][0]), int(frame["bbox"][1])),
    )

    out = g.copy()
    out[out == GREY] = bg

    for sprite, frame in zip(ranked_sprites, ranked_frames):
        for rr, cc in _mirrored_coords_in_frame(sprite, frame):
            r0, c0, r1, c1 = map(int, frame["bbox"])
            assert r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1, "blue copy must stay inside the frame"
            assert out[rr, cc] == bg, "mirrored sprites must not overlap borders or one another"
            out[rr, cc] = BLUE

    return out


def _shape_neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in _NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _random_connected_shape(area: int) -> np.ndarray:
    for _ in range(300):
        h = random.randint(2, min(5, area))
        w = random.randint(2, min(5, area))
        if area > h * w:
            continue

        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = set(_shape_neighbors(seed[0], seed[1], h, w))
        while len(cells) < area and frontier:
            candidates = list(frontier)
            weights = [
                1 + sum((nr, nc) in cells for nr, nc in _shape_neighbors(rr, cc, h, w))
                for rr, cc in candidates
            ]
            rr, cc = random.choices(candidates, weights=weights, k=1)[0]
            frontier.remove((rr, cc))
            if (rr, cc) in cells:
                continue
            cells.add((rr, cc))
            for nb in _shape_neighbors(rr, cc, h, w):
                if nb not in cells:
                    frontier.add(nb)

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        mask = _trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if mask.all():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        return mask

    raise RuntimeError(f"failed to sample a usable sprite of area {area}")


def _choose_frame_specs(sprite_specs: list[dict[str, object]], n: int) -> list[dict[str, object]] | None:
    if n == 3:
        inner_h_values = range(3, 7)
        inner_w_values = range(4, 9)
    else:
        inner_h_values = range(3, 6)
        inner_w_values = range(4, 9)

    candidates = [
        (int(inner_h * inner_w), int(inner_h), int(inner_w))
        for inner_h in inner_h_values
        for inner_w in inner_w_values
    ]
    candidates = sorted(set(candidates), reverse=True)

    def search(rank: int, max_area: int, used: set[int]) -> list[dict[str, object]] | None:
        if rank == len(sprite_specs):
            return []

        sprite = sprite_specs[rank]
        options = [
            (area, inner_h, inner_w)
            for area, inner_h, inner_w in candidates
            if area < max_area
            and area not in used
            and area > int(sprite["area"])
            and inner_h >= int(sprite["h"])
            and inner_w >= int(sprite["w"])
        ]
        random.shuffle(options)
        for area, inner_h, inner_w in options:
            used.add(area)
            rest = search(rank + 1, area, used)
            if rest is not None:
                frame = {
                    "rank": rank,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "inner_area": area,
                    "h": inner_h + 2,
                    "w": inner_w + 2,
                }
                return [frame] + rest
            used.remove(area)
        return None

    return search(0, 10**9, set())


def _stack_tops(heights: list[int], canvas_h: int, min_gap: int, margin: int = 1) -> list[int] | None:
    required = sum(heights) + min_gap * (len(heights) - 1) + 2 * margin
    slack = canvas_h - required
    if slack < 0:
        return None

    extras = [0] * (len(heights) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1

    tops: list[int] = []
    cursor = margin + extras[0]
    for idx, height in enumerate(heights):
        tops.append(cursor)
        cursor += height + min_gap + extras[idx + 1]
    return tops


def _non_identity_permutation(n: int) -> list[int]:
    base = list(range(n))
    for _ in range(30):
        perm = base.copy()
        random.shuffle(perm)
        if perm != base:
            return perm
    return base[1:] + base[:1]


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = map(int, bbox)
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _render_input(sprites: list[dict[str, object]], frames: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)

    for frame in frames:
        top = int(frame["top"])
        left = int(frame["left"])
        h = int(frame["h"])
        w = int(frame["w"])
        grid[top, left : left + w] = RED
        grid[top + h - 1, left : left + w] = RED
        grid[top : top + h, left] = RED
        grid[top : top + h, left + w - 1] = RED

    for sprite in sprites:
        top = int(sprite["top"])
        left = int(sprite["left"])
        mask = np.asarray(sprite["mask"], dtype=bool)
        region = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
        assert not np.any(region[mask] != BLACK)
        region[mask] = GREY

    return grid


def _render_expected(input_grid: np.ndarray, sprites: list[dict[str, object]], frames: list[dict[str, object]]) -> np.ndarray:
    out = input_grid.copy()
    out[out == GREY] = BLACK

    ranked_sprites = sorted(sprites, key=lambda sprite: -int(sprite["area"]))
    ranked_frames = sorted(frames, key=lambda frame: -int(frame["inner_area"]))
    for sprite, frame in zip(ranked_sprites, ranked_frames):
        for rr, cc in _mirrored_coords_in_frame(sprite, frame):
            assert out[rr, cc] == BLACK
            out[rr, cc] = BLUE
    return out


def _has_area_vs_bbox_counterexample(sprite_specs: list[dict[str, object]]) -> bool:
    bbox_rank = sorted(
        range(len(sprite_specs)),
        key=lambda idx: (
            -(int(sprite_specs[idx]["h"]) * int(sprite_specs[idx]["w"])),
            -int(sprite_specs[idx]["h"]),
            -int(sprite_specs[idx]["w"]),
            idx,
        ),
    )
    return bbox_rank != list(range(len(sprite_specs)))


def _has_non_nearest_pairing(sprites: list[dict[str, object]], frames: list[dict[str, object]]) -> bool:
    ranked_sprites = sorted(sprites, key=lambda sprite: -int(sprite["area"]))
    ranked_frames = sorted(frames, key=lambda frame: -int(frame["inner_area"]))
    true_pairs = {int(sprite["rank"]): int(frame["rank"]) for sprite, frame in zip(ranked_sprites, ranked_frames)}

    frame_centers = {int(frame["rank"]): _center(frame["bbox"]) for frame in frames}
    for sprite in sprites:
        sr, sc = _center(sprite["bbox"])
        nearest_frame_rank = min(
            frame_centers,
            key=lambda rank: (sr - frame_centers[rank][0]) ** 2 + (sc - frame_centers[rank][1]) ** 2,
        )
        if nearest_frame_rank != true_pairs[int(sprite["rank"])]:
            return True
    return False


def _scene_semantics_hold(input_grid: np.ndarray, expected: np.ndarray, sprites: list[dict[str, object]], frames: list[dict[str, object]]) -> bool:
    parsed_sprites = _find_sprites(input_grid)
    parsed_frames = _find_frames(input_grid)
    if len(parsed_sprites) != len(sprites) or len(parsed_frames) != len(frames):
        return False
    if len(set(int(sprite["area"]) for sprite in parsed_sprites)) != len(parsed_sprites):
        return False
    if len(set(int(frame["inner_area"]) for frame in parsed_frames)) != len(parsed_frames):
        return False
    if not _has_non_nearest_pairing(sprites, frames):
        return False
    if not np.all(expected[input_grid == RED] == RED):
        return False
    if np.any(expected[input_grid == GREY] != BLACK):
        return False

    blue_components = components_mask(expected == BLUE, connectivity=4)
    if len(blue_components) != len(sprites):
        return False
    if sorted(len(component) for component in blue_components) != sorted(int(sprite["area"]) for sprite in sprites):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    sprite_areas = sorted(random.sample(range(3, 11), n), reverse=True)

    sprite_specs: list[dict[str, object]] = []
    for rank, area in enumerate(sprite_areas):
        mask = _random_connected_shape(area)
        sprite_specs.append(
            {
                "rank": rank,
                "mask": mask,
                "area": area,
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
            }
        )
    if not _has_area_vs_bbox_counterexample(sprite_specs):
        return None

    frame_specs = _choose_frame_specs(sprite_specs, n)
    if frame_specs is None:
        return None

    sprite_perm = _non_identity_permutation(n)
    frame_perm = _non_identity_permutation(n)
    sprite_heights = [int(sprite_specs[idx]["h"]) for idx in sprite_perm]
    frame_heights = [int(frame_specs[idx]["h"]) for idx in frame_perm]

    min_h = max(
        20,
        sum(sprite_heights) + 2 * (n - 1) + 2,
        sum(frame_heights) + 2 * (n - 1) + 2,
    )
    if min_h > 30:
        return None
    canvas_h = random.randint(min_h, 30)

    sprite_tops = _stack_tops(sprite_heights, canvas_h, min_gap=2)
    frame_tops = _stack_tops(frame_heights, canvas_h, min_gap=2)
    if sprite_tops is None or frame_tops is None:
        return None

    max_sprite_w = max(int(spec["w"]) for spec in sprite_specs)
    max_frame_w = max(int(spec["w"]) for spec in frame_specs)
    min_w = max(22, max_sprite_w + max_frame_w + 10)
    if min_w > 30:
        return None
    canvas_w = random.randint(min_w, 30)

    left_zone_right = canvas_w // 2 - 2
    right_zone_left = canvas_w // 2 + 1
    if left_zone_right < max_sprite_w + 1 or right_zone_left > canvas_w - max_frame_w - 1:
        return None

    sprites: list[dict[str, object]] = []
    for top, rank in zip(sprite_tops, sprite_perm):
        spec = dict(sprite_specs[rank])
        left_max = min(4, left_zone_right - int(spec["w"]) + 1)
        if left_max < 1:
            return None
        left = random.randint(1, left_max)
        spec["top"] = top
        spec["left"] = left
        spec["bbox"] = (top, left, top + int(spec["h"]) - 1, left + int(spec["w"]) - 1)
        sprites.append(spec)

    frames: list[dict[str, object]] = []
    for top, rank in zip(frame_tops, frame_perm):
        spec = dict(frame_specs[rank])
        right_margin = random.randint(1, 3)
        left = canvas_w - right_margin - int(spec["w"])
        if left < right_zone_left:
            return None
        spec["top"] = top
        spec["left"] = left
        spec["bbox"] = (top, left, top + int(spec["h"]) - 1, left + int(spec["w"]) - 1)
        frames.append(spec)

    input_grid = _render_input(sprites, frames, (canvas_h, canvas_w))
    expected = _render_expected(input_grid, sprites, frames)
    if not _scene_semantics_hold(input_grid, expected, sprites, frames):
        return None
    if not np.array_equal(main(input_grid), expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-ranked mirror puzzle")
