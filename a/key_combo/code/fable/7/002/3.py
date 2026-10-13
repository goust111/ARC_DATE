from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, in_bounds, paste, random_free_location


BG = Color.BLACK
FRAME_SIZE = 4
SPRITE_SIZE = 2
FRAME_AREA = 12
SPRITE_AREA = 3
MISSING_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
NON_BG_COLORS = tuple(range(1, 10))

GEN_HEIGHT_RANGE = (18, 28)
GEN_WIDTH_RANGE = (18, 28)
GEN_FRAME_COUNT_RANGE = (2, 4)
GEN_FRAME_BORDER = 1
GEN_FRAME_PADDING = 2
GEN_SPRITE_PADDING = 1
GEN_RETRY_LIMIT = 600


def _frame_patch(color: int) -> np.ndarray:
    patch = np.full((FRAME_SIZE, FRAME_SIZE), int(color), dtype=int)
    patch[1:-1, 1:-1] = BG
    return patch


def _sprite_patch(color: int, missing: tuple[int, int]) -> np.ndarray:
    patch = np.full((SPRITE_SIZE, SPRITE_SIZE), int(color), dtype=int)
    patch[int(missing[0]), int(missing[1])] = BG
    return patch


def _is_frame_crop(crop: np.ndarray, color: int) -> bool:
    if crop.shape != (FRAME_SIZE, FRAME_SIZE):
        return False
    if np.any(crop[0, :] != color) or np.any(crop[-1, :] != color):
        return False
    if np.any(crop[:, 0] != color) or np.any(crop[:, -1] != color):
        return False
    return bool(np.all(crop[1:-1, 1:-1] == BG))


def _is_sprite_crop(crop: np.ndarray, color: int) -> bool:
    if crop.shape != (SPRITE_SIZE, SPRITE_SIZE):
        return False
    return bool(np.count_nonzero(crop == BG) == 1 and np.count_nonzero(crop == color) == SPRITE_AREA)


def _missing_corner(crop: np.ndarray) -> tuple[int, int]:
    holes = np.argwhere(crop == BG)
    assert holes.shape == (1, 2), "expected exactly one missing corner"
    return int(holes[0, 0]), int(holes[0, 1])


def _direction_from_missing(missing: tuple[int, int]) -> tuple[int, int]:
    mr, mc = map(int, missing)
    return (-1 if mr == 0 else 1, -1 if mc == 0 else 1)


def _frame_border_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _parse_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid must fit ARC size limits"

    objects = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    assert objects, "expected at least one foreground object"

    frames: list[dict] = []
    sprites: list[dict] = []
    for obj in objects:
        color = int(obj["color"])
        bbox = tuple(map(int, obj["bbox"]))
        area = int(obj["area"])
        crop = crop_bbox(g, bbox)

        if area == FRAME_AREA and _is_frame_crop(crop, color):
            r0, c0, r1, c1 = bbox
            frames.append(
                {
                    "bbox": bbox,
                    "color": color,
                    "cavity_tl": (r0 + 1, c0 + 1),
                    "border": _frame_border_coords(bbox),
                }
            )
            continue

        if area == SPRITE_AREA and _is_sprite_crop(crop, color):
            r0, c0, r1, c1 = bbox
            missing = _missing_corner(crop)
            sprites.append(
                {
                    "bbox": bbox,
                    "color": color,
                    "missing": missing,
                    "gap": (r0 + missing[0], c0 + missing[1]),
                    "direction": _direction_from_missing(missing),
                }
            )
            continue

        raise AssertionError(f"unexpected foreground object with bbox={bbox}, area={area}, color={color}")

    assert frames, "expected at least one hollow frame"
    assert sprites, "expected at least one L sprite"

    frames.sort(key=lambda obj: obj["bbox"])
    sprites.sort(key=lambda obj: obj["bbox"])

    frame_colors = {frame["color"] for frame in frames}
    sprite_colors = {sprite["color"] for sprite in sprites}
    assert len(frame_colors) == len(frames), "frame colors must be distinct"
    assert frame_colors.isdisjoint(sprite_colors), "sprite colors must differ from all frame colors"
    return frames, sprites


def _build_frame_coord_map(frames: list[dict]) -> dict[tuple[int, int], int]:
    coord_map: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for coord in frame["border"]:
            assert coord not in coord_map, "frame borders must not overlap"
            coord_map[coord] = idx
    return coord_map


def _trace_distinct_hits(
    shape: tuple[int, int],
    gap: tuple[int, int],
    direction: tuple[int, int],
    frame_coord_map: dict[tuple[int, int], int],
) -> list[int]:
    r, c = map(int, gap)
    dr, dc = map(int, direction)
    hits: list[int] = []
    seen: set[int] = set()

    while True:
        r += dr
        c += dc
        if not in_bounds(shape, r, c):
            break
        frame_idx = frame_coord_map.get((r, c))
        if frame_idx is not None and frame_idx not in seen:
            seen.add(frame_idx)
            hits.append(int(frame_idx))
    return hits


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Move each L sprite into the hollow rectangle first hit by the diagonal ray implied by its missing corner.

    Input structure:
    - The background is black.
    - There are several monochrome 4x4 hollow rectangular frames with distinct colors and 2x2 empty cavities.
    - There are one or more monochrome 2x2 L sprites, each occupying three cells with one missing corner, and sprite colors differ from all frame colors.

    Rule:
    - For each sprite, treat its missing corner cell as the ray origin and use that corner as a diagonal direction.
    - Extend the ray cell by cell; the unique frame whose border is hit first is that sprite's destination.
    - Erase every sprite from its original position, then place the same L orientation inside the destination frame's 2x2 cavity, recolored to the frame's color, leaving the corresponding missing corner black.

    Key insight:
    - The missing corner is not damage to repair locally; it is a directional cue that invisibly pairs each sprite with a distant container.
    """
    g = np.asarray(input_grid, dtype=int)
    frames, sprites = _parse_objects(g)
    frame_coord_map = _build_frame_coord_map(frames)

    out = g.copy()
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    used_targets: set[int] = set()
    for sprite in sprites:
        hits = _trace_distinct_hits(g.shape, sprite["gap"], sprite["direction"], frame_coord_map)
        assert len(hits) == 1, "each sprite must hit exactly one frame"
        target_idx = hits[0]
        assert target_idx not in used_targets, "different sprites must dock into different frames"
        used_targets.add(target_idx)

        frame = frames[target_idx]
        cavity_r, cavity_c = frame["cavity_tl"]
        cavity = out[cavity_r : cavity_r + SPRITE_SIZE, cavity_c : cavity_c + SPRITE_SIZE]
        assert np.all(cavity == BG), "frame cavity must be empty before docking"
        patch = _sprite_patch(frame["color"], sprite["missing"])
        paste(out, patch, cavity_r, cavity_c, transparent=BG)

    return out


def _bbox_clear(
    reserved: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    *,
    padding: int,
    border: int = 0,
) -> bool:
    H, W = reserved.shape
    r0 = int(r0)
    c0 = int(c0)
    h = int(h)
    w = int(w)
    if r0 < border or c0 < border or r0 + h > H - border or c0 + w > W - border:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _place_frames(height: int, width: int, n_frames: int, frame_colors: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=int)
    marker = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=int)
    frames: list[dict] = []

    for color in frame_colors:
        try:
            r0, c0 = random_free_location(
                reserved,
                marker,
                transparent=BG,
                border=GEN_FRAME_BORDER,
                padding=GEN_FRAME_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(reserved, marker, r0, c0)
        paste(grid, _frame_patch(color), r0, c0, transparent=BG)
        bbox = (int(r0), int(c0), int(r0 + FRAME_SIZE - 1), int(c0 + FRAME_SIZE - 1))
        frames.append(
            {
                "bbox": bbox,
                "color": int(color),
                "cavity_tl": (int(r0 + 1), int(c0 + 1)),
                "border": _frame_border_coords(bbox),
            }
        )

    frames.sort(key=lambda obj: obj["bbox"])
    assert len(frames) == n_frames
    return grid, reserved, frames


def _available_orientations(
    shape: tuple[int, int],
    reserved: np.ndarray,
    frames: list[dict],
    frame_coord_map: dict[tuple[int, int], int],
) -> dict[int, list[tuple[int, int]]]:
    available: dict[int, list[tuple[int, int]]] = {}
    H, W = shape
    for frame_idx in range(len(frames)):
        options: list[tuple[int, int]] = []
        for missing in MISSING_CORNERS:
            found = False
            direction = _direction_from_missing(missing)
            for r0 in range(H - SPRITE_SIZE + 1):
                if found:
                    break
                for c0 in range(W - SPRITE_SIZE + 1):
                    if not _bbox_clear(
                        reserved,
                        r0,
                        c0,
                        SPRITE_SIZE,
                        SPRITE_SIZE,
                        padding=GEN_SPRITE_PADDING,
                        border=0,
                    ):
                        continue
                    gap = (r0 + missing[0], c0 + missing[1])
                    if _trace_distinct_hits(shape, gap, direction, frame_coord_map) == [frame_idx]:
                        options.append(missing)
                        found = True
                        break
        available[frame_idx] = options
    return available


def _candidate_positions(
    shape: tuple[int, int],
    reserved: np.ndarray,
    frame_coord_map: dict[tuple[int, int], int],
    missing: tuple[int, int],
    target_idx: int,
) -> list[tuple[int, int]]:
    H, W = shape
    direction = _direction_from_missing(missing)
    candidates: list[tuple[int, int]] = []
    for r0 in range(H - SPRITE_SIZE + 1):
        for c0 in range(W - SPRITE_SIZE + 1):
            if not _bbox_clear(
                reserved,
                r0,
                c0,
                SPRITE_SIZE,
                SPRITE_SIZE,
                padding=GEN_SPRITE_PADDING,
                border=0,
            ):
                continue
            gap = (r0 + missing[0], c0 + missing[1])
            if _trace_distinct_hits(shape, gap, direction, frame_coord_map) == [target_idx]:
                candidates.append((int(r0), int(c0)))
    return candidates


def _diversify_orientations(plans: list[dict], available: dict[int, list[tuple[int, int]]]) -> None:
    if len(plans) < 2 or len({plan["missing"] for plan in plans}) > 1:
        return
    random.shuffle(plans)
    for plan in plans:
        alternatives = [missing for missing in available[plan["target_idx"]] if missing != plan["missing"]]
        if alternatives:
            plan["missing"] = random.choice(alternatives)
            return


def _place_sprites(
    grid: np.ndarray,
    reserved: np.ndarray,
    frames: list[dict],
    plans: list[dict],
) -> list[dict] | None:
    frame_coord_map = _build_frame_coord_map(frames)
    placements: list[dict | None] = [None] * len(plans)
    marker = np.ones((SPRITE_SIZE, SPRITE_SIZE), dtype=int)

    def backtrack(remaining: list[int]) -> bool:
        if not remaining:
            return True

        best_idx = -1
        best_candidates: list[tuple[int, int]] | None = None
        for plan_idx in remaining:
            plan = plans[plan_idx]
            candidates = _candidate_positions(
                grid.shape,
                reserved,
                frame_coord_map,
                plan["missing"],
                plan["target_idx"],
            )
            if not candidates:
                return False
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_idx = plan_idx
                best_candidates = candidates

        assert best_candidates is not None
        random.shuffle(best_candidates)
        plan = plans[best_idx]
        patch = _sprite_patch(plan["color"], plan["missing"])
        next_remaining = [idx for idx in remaining if idx != best_idx]

        for r0, c0 in best_candidates:
            paste(grid, patch, r0, c0, transparent=BG)
            paste(reserved, marker, r0, c0)
            placements[best_idx] = {
                "bbox": (int(r0), int(c0), int(r0 + 1), int(c0 + 1)),
                "color": int(plan["color"]),
                "missing": plan["missing"],
                "gap": (int(r0 + plan["missing"][0]), int(c0 + plan["missing"][1])),
                "direction": _direction_from_missing(plan["missing"]),
                "target_idx": int(plan["target_idx"]),
            }
            if backtrack(next_remaining):
                return True
            placements[best_idx] = None
            reserved[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = BG
            grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE][patch != BG] = BG
        return False

    if not backtrack(list(range(len(plans)))):
        return None
    return [placement for placement in placements if placement is not None]


def _expected_output_from_spec(grid: np.ndarray, frames: list[dict], placements: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for placement in placements:
        r0, c0, r1, c1 = placement["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for placement in placements:
        frame = frames[placement["target_idx"]]
        cavity_r, cavity_c = frame["cavity_tl"]
        patch = _sprite_patch(frame["color"], placement["missing"])
        paste(out, patch, cavity_r, cavity_c, transparent=BG)
    return out


def _validate_generated_instance(grid: np.ndarray, frames: list[dict], placements: list[dict]) -> bool:
    expected = _expected_output_from_spec(grid, frames, placements)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    parsed_frames, parsed_sprites = _parse_objects(grid)
    if len(parsed_frames) != len(frames) or len(parsed_sprites) != len(placements):
        return False

    frame_coord_map = _build_frame_coord_map(parsed_frames)
    sprite_by_gap = {sprite["gap"]: sprite for sprite in parsed_sprites}
    if len(sprite_by_gap) != len(parsed_sprites):
        return False

    target_to_sprite: dict[int, dict] = {}
    for placement in placements:
        sprite = sprite_by_gap.get(placement["gap"])
        if sprite is None:
            return False
        hits = _trace_distinct_hits(grid.shape, sprite["gap"], sprite["direction"], frame_coord_map)
        if hits != [placement["target_idx"]]:
            return False
        target_to_sprite[placement["target_idx"]] = sprite

    for sprite in parsed_sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        if np.any(solved[r0 : r1 + 1, c0 : c1 + 1] != BG):
            return False

    for frame_idx, frame in enumerate(parsed_frames):
        r0, c0, r1, c1 = frame["bbox"]
        if not np.array_equal(solved[r0, c0 : c1 + 1], grid[r0, c0 : c1 + 1]):
            return False
        if not np.array_equal(solved[r1, c0 : c1 + 1], grid[r1, c0 : c1 + 1]):
            return False
        if not np.array_equal(solved[r0 : r1 + 1, c0], grid[r0 : r1 + 1, c0]):
            return False
        if not np.array_equal(solved[r0 : r1 + 1, c1], grid[r0 : r1 + 1, c1]):
            return False

        cavity = solved[r0 + 1 : r1, c0 + 1 : c1]
        if frame_idx not in target_to_sprite:
            if np.any(cavity != BG):
                return False
            continue
        sprite = target_to_sprite[frame_idx]
        expected_cavity = _sprite_patch(frame["color"], sprite["missing"])
        if not np.array_equal(cavity, expected_cavity):
            return False
    return True


def _sample_sprite_count(selectable_frames: int) -> int:
    assert selectable_frames >= 1
    if selectable_frames == 1:
        return 1
    if random.random() < 0.75:
        return random.randint(1, selectable_frames - 1)
    return random.randint(1, selectable_frames)


def _generate_once() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    n_frames = random.randint(*GEN_FRAME_COUNT_RANGE)
    if n_frames * (FRAME_SIZE + GEN_FRAME_PADDING) > max(height, width) + FRAME_SIZE:
        return None

    frame_colors = random.sample(NON_BG_COLORS, n_frames)
    placed = _place_frames(height, width, n_frames, frame_colors)
    if placed is None:
        return None
    grid, reserved, frames = placed

    frame_coord_map = _build_frame_coord_map(frames)
    available = _available_orientations(grid.shape, reserved, frames, frame_coord_map)
    selectable = [frame_idx for frame_idx, options in available.items() if options]
    if not selectable:
        return None

    n_sprites = _sample_sprite_count(len(selectable))
    target_indices = random.sample(selectable, n_sprites)
    sprite_colors = random.sample([color for color in NON_BG_COLORS if color not in frame_colors], n_sprites)
    plans = [
        {
            "target_idx": int(target_idx),
            "missing": random.choice(available[target_idx]),
            "color": int(color),
        }
        for target_idx, color in zip(target_indices, sprite_colors)
    ]
    _diversify_orientations(plans, available)

    placements = _place_sprites(grid, reserved, frames, plans)
    if placements is None:
        return None
    if not _validate_generated_instance(grid, frames, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
