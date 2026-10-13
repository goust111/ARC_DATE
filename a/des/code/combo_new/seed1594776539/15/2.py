from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

BODY_COLORS = (RED, GREEN, BLUE, MAGENTA, ORANGE, MAROON)
PRIORITY = {
    GREY: 0,
    RED: 1,
    GREEN: 2,
    BLUE: 3,
    MAGENTA: 4,
    ORANGE: 5,
    MAROON: 6,
}
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_PATTERNS = {
    "up": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "down": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    "left": {(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)},
    "right": {(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)},
}

GEN_RETRIES = 400
GRID_LIMIT = 30
MARGIN_RANGE = (1, 3)
GAP_RANGE = (1, 3)
SPRITE_COUNT_RANGE = (2, 5)
SPRITE_SIZE_MIN = 5
SPRITE_SIZE_MAX = 8


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _priority(color: int) -> int:
    if color == BLACK:
        return -1
    if color not in PRIORITY:
        raise AssertionError(f"unsupported overlay color: {color}")
    return PRIORITY[color]


def _is_target_patch(patch: np.ndarray) -> bool:
    h, w = patch.shape
    if h < 3 or w < 3:
        return False
    if not np.all((patch == BLACK) | (patch == YELLOW)):
        return False
    if not np.all(patch[0, :] == YELLOW):
        return False
    if not np.all(patch[-1, :] == YELLOW):
        return False
    if not np.all(patch[:, 0] == YELLOW):
        return False
    if not np.all(patch[:, -1] == YELLOW):
        return False
    if not np.all(patch[1:-1, 1:-1] == BLACK):
        return False
    return True


def _detect_arrow_direction(patch: np.ndarray) -> str:
    grey = np.argwhere(patch == GREY)
    if grey.shape[0] != 5:
        raise AssertionError("sprite must contain exactly five grey arrow pixels")
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    if int(r1 - r0 + 1) != 3 or int(c1 - c0 + 1) != 3:
        raise AssertionError("arrow must occupy a translated 3x3 box")
    rel = {(int(r - r0), int(c - c0)) for r, c in grey}
    for name, pattern in ARROW_PATTERNS.items():
        if rel == pattern:
            return name
    raise AssertionError("grey pixels do not match a valid arrow")


def _analyze(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    non_black = g != BLACK
    comps = _components(non_black)

    target_bbox: tuple[int, int, int, int] | None = None
    sprites: list[dict[str, object]] = []

    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        vals = set(int(v) for v in np.unique(patch))

        if YELLOW in vals:
            if target_bbox is not None:
                raise AssertionError("expected a unique yellow target region")
            if not _is_target_patch(patch):
                raise AssertionError("yellow component is not a valid target frame")
            target_bbox = (r0, c0, r1, c1)
            continue

        if BLACK in vals:
            raise AssertionError("sprites must be solid rectangles without black holes")
        if len(comp) != patch.size:
            raise AssertionError("sprite component must fill its bounding box")
        if GREY not in vals or len(vals) != 2:
            raise AssertionError("sprite must contain one body color plus grey arrow pixels")

        body = next(v for v in vals if v != GREY)
        if body not in BODY_COLORS:
            raise AssertionError(f"unsupported body color: {body}")
        direction = _detect_arrow_direction(patch)
        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch.copy(),
                "body": int(body),
                "direction": direction,
            }
        )

    if target_bbox is None:
        raise AssertionError("missing yellow target frame")
    if len(sprites) < 2:
        raise AssertionError("expected multiple sprites")

    tr0, tc0, tr1, tc1 = target_bbox
    th = tr1 - tr0 + 1
    tw = tc1 - tc0 + 1
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]  # type: ignore[index]
        if (r1 - r0 + 1, c1 - c0 + 1) != (th, tw):
            raise AssertionError("all sprites must match the target size")
        dr = tr0 - r0
        dc = tc0 - c0
        direction = sprite["direction"]  # type: ignore[index]
        step = DIRECTIONS[direction]
        if (dr == 0 and dc == 0) or not ((dr == 0) ^ (dc == 0)):
            raise AssertionError("sprite-to-target offset must be purely cardinal and non-zero")
        if step[0] != 0:
            if dc != 0 or (dr > 0) - (dr < 0) != step[0]:
                raise AssertionError("vertical arrow direction does not match target offset")
        else:
            if dr != 0 or (dc > 0) - (dc < 0) != step[1]:
                raise AssertionError("horizontal arrow direction does not match target offset")

    return target_bbox, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate arrow-marked rectangular sprites onto the yellow target frame and merge them by fixed color priority.

    Input structure:
    - The background is black.
    - Several same-sized solid rectangular sprites appear apart from one another; each sprite has one body color and a grey arrow drawn inside it.
    - The grey arrow is a 5-pixel cardinal symbol: a 3-pixel shaft plus one tip pixel on each side of the front end.
    - A single yellow rectangular outline marks the destination region; its size matches every sprite and its interior is black.

    Rule:
    - Detect each sprite and read the direction of its internal grey arrow.
    - Translate each entire sprite along that direction until its bounding box coincides exactly with the yellow target box.
    - Remove the yellow outline and leave the rest of the old sprite locations black.
    - Overlay all translated sprites on the target box; for each overlapped cell keep the visible color with higher fixed priority: grey < red < green < blue < magenta < orange < maroon.

    Key insight:
    - The grey arrow is both part of the sprite and the movement instruction, so solving requires first extracting the arrow orientation and then compositing all aligned sprite patches with the explicit color-priority rule.
    """
    g = np.asarray(grid, dtype=int)
    target_bbox, sprites = _analyze(g)
    tr0, tc0, tr1, tc1 = target_bbox
    th = tr1 - tr0 + 1
    tw = tc1 - tc0 + 1

    out = np.full(g.shape, BLACK, dtype=int)
    for sprite in sprites:
        patch = sprite["patch"]  # type: ignore[index]
        for rr in range(th):
            for cc in range(tw):
                color = int(patch[rr, cc])
                if _priority(color) > _priority(int(out[tr0 + rr, tc0 + cc])):
                    out[tr0 + rr, tc0 + cc] = color
    return out


def _sample_directions(rng: random.Random) -> list[str]:
    n = rng.randint(*SPRITE_COUNT_RANGE)
    names = list(DIRECTIONS)
    while True:
        dirs = [rng.choice(names) for _ in range(n)]
        counts = Counter(dirs)
        if max(counts.values()) > 2:
            continue
        if len(counts) < 2:
            continue
        return dirs


def _sample_gaps(count: int, rng: random.Random) -> list[int]:
    return [rng.randint(*GAP_RANGE) for _ in range(count)]


def _make_sprite_patch(height: int, width: int, body: int, direction: str, rng: random.Random) -> np.ndarray:
    patch = np.full((height, width), body, dtype=int)
    ar = rng.randint(1, height - 4)
    ac = rng.randint(1, width - 4)
    for dr, dc in ARROW_PATTERNS[direction]:
        patch[ar + dr, ac + dc] = GREY
    return patch


def _draw_target(grid: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    grid[r0, c0 : c0 + width] = YELLOW
    grid[r0 + height - 1, c0 : c0 + width] = YELLOW
    grid[r0 : r0 + height, c0] = YELLOW
    grid[r0 : r0 + height, c0 + width - 1] = YELLOW


def _place_side(
    side_direction: str,
    count: int,
    gaps: list[int],
    target_r: int,
    target_c: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    if side_direction == "right":
        cursor = target_c
        for gap in gaps:
            col = cursor - gap - width
            coords.append((target_r, col))
            cursor = col
    elif side_direction == "left":
        cursor = target_c + width
        for gap in gaps:
            col = cursor + gap
            coords.append((target_r, col))
            cursor = col + width
    elif side_direction == "down":
        cursor = target_r
        for gap in gaps:
            row = cursor - gap - height
            coords.append((row, target_c))
            cursor = row
    elif side_direction == "up":
        cursor = target_r + height
        for gap in gaps:
            row = cursor + gap
            coords.append((row, target_c))
            cursor = row + height
    else:
        raise AssertionError(f"unknown direction: {side_direction}")
    if len(coords) != count:
        raise AssertionError("placement count mismatch")
    return coords


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        dirs = _sample_directions(rng)
        counts = Counter(dirs)

        left_count = counts["right"]
        right_count = counts["left"]
        top_count = counts["down"]
        bottom_count = counts["up"]

        left_margin = rng.randint(*MARGIN_RANGE)
        right_margin = rng.randint(*MARGIN_RANGE)
        top_margin = rng.randint(*MARGIN_RANGE)
        bottom_margin = rng.randint(*MARGIN_RANGE)

        left_gaps = _sample_gaps(left_count, rng)
        right_gaps = _sample_gaps(right_count, rng)
        top_gaps = _sample_gaps(top_count, rng)
        bottom_gaps = _sample_gaps(bottom_count, rng)

        width_const = left_margin + right_margin + sum(left_gaps) + sum(right_gaps)
        width_mult = left_count + right_count + 1
        max_width = min(SPRITE_SIZE_MAX, (GRID_LIMIT - width_const) // width_mult)

        height_const = top_margin + bottom_margin + sum(top_gaps) + sum(bottom_gaps)
        height_mult = top_count + bottom_count + 1
        max_height = min(SPRITE_SIZE_MAX, (GRID_LIMIT - height_const) // height_mult)

        if max_width < SPRITE_SIZE_MIN or max_height < SPRITE_SIZE_MIN:
            continue

        width = rng.randint(SPRITE_SIZE_MIN, max_width)
        height = rng.randint(SPRITE_SIZE_MIN, max_height)

        left_extent = left_count * width + sum(left_gaps)
        right_extent = right_count * width + sum(right_gaps)
        top_extent = top_count * height + sum(top_gaps)
        bottom_extent = bottom_count * height + sum(bottom_gaps)

        grid_h = top_margin + top_extent + height + bottom_extent + bottom_margin
        grid_w = left_margin + left_extent + width + right_extent + right_margin
        if grid_h > GRID_LIMIT or grid_w > GRID_LIMIT:
            continue

        target_r = top_margin + top_extent
        target_c = left_margin + left_extent
        grid = np.full((grid_h, grid_w), BLACK, dtype=int)
        _draw_target(grid, target_r, target_c, height, width)

        placements = {
            "right": _place_side("right", left_count, left_gaps, target_r, target_c, height, width),
            "left": _place_side("left", right_count, right_gaps, target_r, target_c, height, width),
            "down": _place_side("down", top_count, top_gaps, target_r, target_c, height, width),
            "up": _place_side("up", bottom_count, bottom_gaps, target_r, target_c, height, width),
        }

        color_pool = list(BODY_COLORS)
        rng.shuffle(color_pool)
        body_colors = color_pool[: len(dirs)]
        dir_slots = {name: list(placements[name]) for name in placements}

        for direction, body in zip(dirs, body_colors, strict=False):
            if not dir_slots[direction]:
                raise AssertionError("missing placement slot for sampled direction")
            r0, c0 = dir_slots[direction].pop()
            patch = _make_sprite_patch(height, width, body, direction, rng)
            if not np.all(grid[r0 : r0 + height, c0 : c0 + width] == BLACK):
                raise AssertionError("sprite placement overlaps an existing object")
            grid[r0 : r0 + height, c0 : c0 + width] = patch

        try:
            target_bbox, sprites = _analyze(grid)
        except AssertionError:
            continue
        if len(sprites) != len(dirs):
            continue
        if target_bbox != (target_r, target_c, target_r + height - 1, target_c + width - 1):
            continue

        out = main(grid)
        if out.shape != grid.shape:
            continue
        if np.any(out == YELLOW):
            continue
        if np.count_nonzero(out) != height * width:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC puzzle instance")
