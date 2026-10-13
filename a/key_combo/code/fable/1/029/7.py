from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
WATER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


Coord = tuple[int, int]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return True
    start = tuple(map(int, coords[0]))
    todo = deque([start])
    seen = {start}
    h, w = mask.shape
    while todo:
        r, c = todo.popleft()
        for dr, dc in NEIGHBORS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                todo.append((rr, cc))
    return len(seen) == int(mask.sum())


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + int(c)] = int(color)


def _shape_cells(mask: np.ndarray) -> tuple[Coord, ...]:
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _reflection_rel_cells(mask: np.ndarray) -> tuple[Coord, ...]:
    h, _w = mask.shape
    return tuple((int(h - 1 - r), int(c)) for r, c in np.argwhere(mask))


def _placed_cells(mask: np.ndarray, top: int, left: int) -> tuple[Coord, ...]:
    return tuple((top + r, left + c) for r, c in _shape_cells(mask))


def _reflection_cells(mask: np.ndarray, line_row: int, left: int) -> tuple[Coord, ...]:
    return tuple((line_row + 1 + r, left + c) for r, c in _reflection_rel_cells(mask))


def _find_waterline(grid: np.ndarray) -> int:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == WATER)]
    assert len(rows) == 1, "expected exactly one full grey waterline"
    return rows[0]


def _extract_sprites(grid: np.ndarray, line_row: int) -> list[dict]:
    sky = grid[:line_row, :]
    colors = sorted(set(map(int, sky.flat)) - {BG, WATER})
    sprites = []
    for color in colors:
        mask = sky == color
        box = _bbox(mask)
        assert box is not None, "sprite color must appear above the waterline"
        r0, c0, r1, c1 = box
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert _connected(cropped), "each sprite must be one connected monochrome object"
        sprites.append(
            {
                "color": int(color),
                "mask": cropped,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    assert sprites, "expected at least one sprite"
    return sprites


def _extract_fragments(grid: np.ndarray, line_row: int) -> dict[int, tuple[Coord, ...]]:
    water = grid[line_row + 1 :, :]
    colors = sorted(set(map(int, water.flat)) - {BG, WATER})
    fragments: dict[int, tuple[Coord, ...]] = {}
    for color in colors:
        coords = np.argwhere(water == color)
        assert coords.size > 0, "fragment color must appear below the waterline"
        fragments[int(color)] = tuple((line_row + 1 + int(r), int(c)) for r, c in coords)
    return fragments


def _valid_columns(mask: np.ndarray, fragments: tuple[Coord, ...], line_row: int, width: int) -> list[int]:
    h, w = mask.shape
    assert h > 0 and w > 0
    rel_cells = _reflection_rel_cells(mask)
    fragment_set = set((int(r), int(c)) for r, c in fragments)
    valid = []
    for left in range(width - w + 1):
        cells = {(line_row + 1 + rr, left + cc) for rr, cc in rel_cells}
        if fragment_set.issubset(cells):
            valid.append(int(left))
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each floating monochrome sprite onto the waterline, infer its horizontal landing from same-colored reflection shards, and complete the full mirrored reflection.

    Input structure:
    - One full grey row is the waterline.
    - Above it, the black sky contains several disconnected monochrome sprites with distinct colors.
    - Below it, each sprite color appears only as a few scattered shard cells belonging to that sprite's eventual water reflection.

    Rule:
    - For each sprite color, keep the sprite shape and orientation fixed, move it horizontally, and place it so its bottom edge sits immediately above the waterline.
    - Choose the unique horizontal landing whose mirror image across the waterline contains every shard of that same color.
    - Erase the original floating sprites, keep the grey waterline, draw each relocated sprite above the line, and draw its complete mirrored reflection below the line.

    Key insight:
    - The shards are not standalone objects; they are partial evidence of the final mirrored image, so the task is solved by reversing the reflection to recover the landing column and then reconstructing the whole symmetric pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = map(int, g.shape)
    line_row = _find_waterline(g)
    sprites = _extract_sprites(g, line_row)
    fragments = _extract_fragments(g, line_row)

    sprite_colors = {int(sprite["color"]) for sprite in sprites}
    assert sprite_colors == set(fragments), "every sprite color must have exactly one fragment color below"

    out = np.full((h, w), BG, dtype=int)
    out[line_row, :] = WATER

    for sprite in sorted(sprites, key=lambda item: int(item["color"])):
        color = int(sprite["color"])
        mask = np.asarray(sprite["mask"], dtype=bool)
        valid = _valid_columns(mask, fragments[color], line_row, w)
        assert len(valid) == 1, "reflection shards must determine a unique landing column"
        left = int(valid[0])
        top = int(line_row - mask.shape[0])

        for r, c in _placed_cells(mask, top, left):
            assert out[r, c] in (BG, color), "sprites must not overlap after relocation"
            out[r, c] = color
        for r, c in _reflection_cells(mask, line_row, left):
            assert out[r, c] in (BG, color), "reflections must not overlap after relocation"
            out[r, c] = color

    return out


def _rotations_and_flips(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    k = rng.randrange(4)
    out = np.rot90(out, k=k)
    return np.array(out, dtype=bool)


def _rectangle_mask(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask


def _l_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    if width >= 3 and height >= 3:
        mask[-2, 1] = True
    return mask


def _t_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mid = width // 2
    mask[0, :] = True
    mask[:, mid] = True
    if height >= 3 and width >= 3:
        lo = max(0, mid - 1)
        hi = min(width, mid + 2)
        mask[-1, lo:hi] = True
    return mask


def _stairs_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        c = min(width - 1, int(round(r * (width - 1) / max(1, height - 1))))
        mask[r, : c + 1] = True
    return mask


def _blob_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    target = rng.randint(max(height + width - 1, (height * width + 1) // 2), height * width)
    cells = {(0, 0)}
    r = c = 0
    steps = [(1, 0)] * (height - 1) + [(0, 1)] * (width - 1)
    rng.shuffle(steps)
    for dr, dc in steps:
        r += dr
        c += dc
        cells.add((r, c))
    while len(cells) < target:
        frontier = []
        for cr, cc in cells:
            for dr, dc in NEIGHBORS4:
                rr = cr + dr
                cc2 = cc + dc
                if 0 <= rr < height and 0 <= cc2 < width and (rr, cc2) not in cells:
                    frontier.append((rr, cc2))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    return mask


def _random_sprite_mask(
    rng: random.Random,
    *,
    max_height: int,
    max_width: int,
    challenging: bool,
) -> np.ndarray | None:
    if max_height < 2 or max_width < 2:
        return None

    if challenging:
        height = rng.randint(2, min(4, max_height))
        width = rng.randint(3, min(5, max_width))
        family = "rect" if rng.random() < 0.7 else "frame"
    else:
        family = rng.choice(("rect", "frame", "l", "t", "stairs", "blob"))
        height = rng.randint(2, min(5, max_height))
        width = rng.randint(2, min(6, max_width))
        if family in {"frame", "t"}:
            height = max(height, 3)
            width = max(width, 3)
        if family == "blob":
            height = rng.randint(2, min(5, max_height))
            width = rng.randint(2, min(6, max_width))

    if height > max_height or width > max_width:
        return None

    if family == "rect":
        mask = _rectangle_mask(height, width)
    elif family == "frame":
        mask = _frame_mask(height, width)
    elif family == "l":
        mask = _l_mask(height, width)
    elif family == "t":
        mask = _t_mask(height, width)
    elif family == "stairs":
        mask = _stairs_mask(height, width)
    else:
        mask = _blob_mask(height, width, rng)

    mask = _rotations_and_flips(mask, rng)
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    if mask.shape[0] > max_height or mask.shape[1] > max_width:
        return None
    if not _connected(mask):
        return None
    if int(mask.sum()) < 3:
        return None
    return np.array(mask, dtype=bool)


def _cells_clear(blocked: np.ndarray, cells: tuple[Coord, ...]) -> bool:
    return all(not blocked[int(r), int(c)] for r, c in cells)


def _mark_blocked(blocked: np.ndarray, cells: tuple[Coord, ...]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _fragment_subset(
    mask: np.ndarray,
    true_left: int,
    line_row: int,
    width: int,
    rng: random.Random,
    *,
    require_multi: bool,
    require_necessary: bool,
) -> tuple[Coord, ...] | None:
    full = sorted(_reflection_cells(mask, line_row, true_left))
    min_k = 2 if require_multi else 1
    max_k = min(4, len(full))
    if min_k > max_k:
        return None

    for size in range(min_k, max_k + 1):
        combos = list(itertools.combinations(full, size))
        rng.shuffle(combos)
        for combo in combos:
            if _valid_columns(mask, combo, line_row, width) != [true_left]:
                continue
            if require_necessary:
                keep = True
                for idx in range(len(combo)):
                    reduced = combo[:idx] + combo[idx + 1 :]
                    if len(_valid_columns(mask, reduced, line_row, width)) <= 1:
                        keep = False
                        break
                if not keep:
                    continue
            return tuple(combo)
    return None


def _build_output(specs: list[dict], shape: tuple[int, int], line_row: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[line_row, :] = WATER
    for spec in specs:
        color = int(spec["color"])
        mask = np.asarray(spec["mask"], dtype=bool)
        left = int(spec["final_left"])
        top = int(line_row - mask.shape[0])
        _paint_mask(out, mask, top, left, color)
        refl = np.flipud(mask.astype(int)).astype(bool)
        _paint_mask(out, refl, line_row + 1, left, color)
    return out


def _build_input(specs: list[dict], shape: tuple[int, int], line_row: int) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[line_row, :] = WATER
    for spec in specs:
        _paint_mask(grid, np.asarray(spec["mask"], dtype=bool), int(spec["start_top"]), int(spec["start_left"]), int(spec["color"]))
        for r, c in spec["fragments"]:
            grid[int(r), int(c)] = int(spec["color"])
    return grid


def _wrong_output_from_original_columns(specs: list[dict], shape: tuple[int, int], line_row: int) -> np.ndarray:
    wrong = np.full(shape, BG, dtype=int)
    wrong[line_row, :] = WATER
    for spec in specs:
        color = int(spec["color"])
        mask = np.asarray(spec["mask"], dtype=bool)
        left = int(spec["start_left"])
        top = int(line_row - mask.shape[0])
        _paint_mask(wrong, mask, top, left, color)
        refl = np.flipud(mask.astype(int)).astype(bool)
        _paint_mask(wrong, refl, line_row + 1, left, color)
    return wrong


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(14, 24)
    width = rng.randint(14, 28)
    line_row = rng.randint(5, height - 6)
    max_sprite_height = min(5, line_row - 1, height - line_row - 1)
    max_sprite_width = min(6, width - 2)
    if max_sprite_height < 2 or max_sprite_width < 3:
        return None

    sprite_count = rng.choices([2, 3, 4], weights=[3, 5, 2], k=1)[0]
    colors = rng.sample(PALETTE, sprite_count)

    specs: list[dict] = []
    challenge_mask = _random_sprite_mask(
        rng,
        max_height=max_sprite_height,
        max_width=max_sprite_width,
        challenging=True,
    )
    if challenge_mask is None:
        return None
    specs.append({"color": int(colors[0]), "mask": challenge_mask, "challenge": True})
    for color in colors[1:]:
        mask = _random_sprite_mask(
            rng,
            max_height=max_sprite_height,
            max_width=max_sprite_width,
            challenging=False,
        )
        if mask is None:
            return None
        specs.append({"color": int(color), "mask": mask, "challenge": False})

    blocked = np.zeros((height, width), dtype=bool)
    blocked[line_row, :] = True
    order = sorted(
        range(len(specs)),
        key=lambda idx: (0 if specs[idx]["challenge"] else 1, -int(np.sum(specs[idx]["mask"]))),
    )
    for idx in order:
        mask = np.asarray(specs[idx]["mask"], dtype=bool)
        h, w = mask.shape
        top = int(line_row - h)
        candidates = []
        for left in range(width - w + 1):
            if specs[idx]["challenge"] and left in (0, width - w):
                continue
            cells = _placed_cells(mask, top, left) + _reflection_cells(mask, line_row, left)
            if _cells_clear(blocked, cells):
                candidates.append(int(left))
        if not candidates:
            return None
        final_left = rng.choice(candidates)
        specs[idx]["final_left"] = int(final_left)
        _mark_blocked(blocked, _placed_cells(mask, top, final_left) + _reflection_cells(mask, line_row, final_left))

    challenge_multi = False
    for spec in specs:
        fragments = _fragment_subset(
            np.asarray(spec["mask"], dtype=bool),
            int(spec["final_left"]),
            line_row,
            width,
            rng,
            require_multi=bool(spec["challenge"]),
            require_necessary=bool(spec["challenge"]),
        )
        if fragments is None:
            return None
        spec["fragments"] = fragments
        if spec["challenge"]:
            challenge_multi = len(fragments) >= 2
        if len(fragments) >= int(np.sum(spec["mask"])):
            return None

    if not challenge_multi:
        return None

    sky_blocked = np.zeros((line_row, width), dtype=bool)
    for idx in sorted(range(len(specs)), key=lambda i: -int(np.sum(specs[i]["mask"]))):
        mask = np.asarray(specs[idx]["mask"], dtype=bool)
        h, w = mask.shape
        candidates = []
        for top in range(line_row - h):
            if top + h > line_row - 1:
                continue
            for left in range(width - w + 1):
                if left == int(specs[idx]["final_left"]):
                    continue
                cells = _placed_cells(mask, top, left)
                if all(not sky_blocked[r, c] for r, c in cells):
                    candidates.append((int(top), int(left)))
        if not candidates:
            return None
        start_top, start_left = rng.choice(candidates)
        specs[idx]["start_top"] = int(start_top)
        specs[idx]["start_left"] = int(start_left)
        _mark_blocked(sky_blocked, _placed_cells(mask, start_top, start_left))

    inp = _build_input(specs, (height, width), line_row)
    expected = _build_output(specs, (height, width), line_row)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(expected, _wrong_output_from_original_columns(specs, (height, width), line_row)):
        return None

    for spec in specs:
        valid = _valid_columns(np.asarray(spec["mask"], dtype=bool), spec["fragments"], line_row, width)
        if valid != [int(spec["final_left"])]:
            return None
        if spec["challenge"]:
            if len(spec["fragments"]) < 2:
                return None
            for idx in range(len(spec["fragments"])):
                reduced = spec["fragments"][:idx] + spec["fragments"][idx + 1 :]
                if len(_valid_columns(np.asarray(spec["mask"], dtype=bool), reduced, line_row, width)) <= 1:
                    return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid water-reflection puzzle")
