from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
TIP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

_COUNT_CYCLE = (4, 3, 2, 4, 3, 4)
_GEN_CALLS = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch, (r0, c0, r1, c1)


def _rotate_cw(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, k=(-int(turns)) % 4)


def _stamp(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    ph, pw = patch.shape
    target = out[top : top + ph, left : left + pw]
    mask = patch != bg
    assert target.shape == patch.shape, "patch must fit the destination grid"
    assert not np.any((target != bg) & mask), "stamped objects must stay disjoint"
    target[mask] = patch[mask]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_rot_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k=k)) for k in (1, 2, 3))


def _make_patch_from_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _component_signature(component: dict) -> tuple:
    return (
        component["color"],
        component["top"],
        component["left"],
        tuple(map(tuple, np.argwhere(component["patch"] != BG))),
    )


def _parse_objects(grid: np.ndarray) -> tuple[int, dict[int, dict], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    bars: dict[int, dict] = {}
    sprites: dict[int, dict] = {}

    for coords in _components(g != bg):
        patch, bbox = _crop_component(g, coords, bg)
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        rows = sorted(int(r) for r, _c in coords)
        cols = {int(c) for _r, c in coords}
        colors = {int(g[int(r), int(c)]) for r, c in coords}

        is_bar = (
            width == 1
            and len(cols) == 1
            and len(rows) >= 2
            and rows == list(range(rows[0], rows[0] + len(rows)))
        )
        if is_bar:
            col = next(iter(cols))
            top = rows[0]
            ordered = [int(g[r, col]) for r in rows]
            if ordered[0] != bg and ordered[0] != GREY and all(color == GREY for color in ordered[1:]):
                tip_color = ordered[0]
                assert tip_color not in bars, "tip colors must be unique"
                bars[tip_color] = {
                    "color": tip_color,
                    "height": len(rows),
                    "top": top,
                    "col": col,
                }
                continue

        assert len(colors) == 1, "sprites must be monochrome connected components"
        sprite_color = next(iter(colors))
        assert sprite_color not in {bg, GREY}, "sprite colors must match tip colors and differ from bar grey"
        assert sprite_color not in sprites, "sprite colors must be unique"
        sprites[sprite_color] = {
            "color": sprite_color,
            "top": r0,
            "left": c0,
            "bbox": bbox,
            "patch": patch,
        }

    assert 2 <= len(bars) <= 4, "expected two to four vertical bars"
    assert len(bars) == len(sprites), "bar count and sprite count must match"
    assert set(bars) == set(sprites), "tip colors must pair one-to-one with sprite colors"
    heights = [info["height"] for info in bars.values()]
    assert len(set(heights)) == len(heights), "bar heights must be distinct"
    return bg, bars, sprites


def _rank_map_from_bars(bars: dict[int, dict]) -> dict[int, int]:
    ranked = sorted(bars.items(), key=lambda item: item[1]["height"])
    return {color: rank for rank, (color, _info) in enumerate(ranked, start=1)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each sprite clockwise by the height-rank of the same-colored tipped bar, then erase the bars.

    Input structure:
    - The grid has a black background with two to four one-cell-wide vertical bars whose bodies are grey and whose top cells are uniquely colored tips.
    - Bar heights are all distinct.
    - Elsewhere, there are equally many monochrome connected sprites; each sprite color matches exactly one tip color.
    - The sprite shapes have no rotational symmetry.

    Rule:
    - Rank the tipped bars from shortest to tallest, with the shortest bar in first place.
    - For each sprite, find the bar whose tip has the same color, and rotate the sprite clockwise by that bar's rank in quarter-turns.
    - Rotate each sprite inside its own bounding-box patch and place the rotated patch back with its top-left corner anchored to the original bounding-box top-left.
    - Erase every bar and keep only the rotated sprites on the background.

    Key insight:
    - The tip color gives the sprite-to-bar pairing, but the action is controlled by relative height rank rather than raw bar height or left-to-right order.
    """
    bg, bars, sprites = _parse_objects(np.asarray(grid, dtype=int))
    rank_by_color = _rank_map_from_bars(bars)

    out = np.full_like(np.asarray(grid, dtype=int), bg)
    for color, sprite in sprites.items():
        rotated = _rotate_cw(sprite["patch"], rank_by_color[color])
        _stamp(out, rotated, sprite["top"], sprite["left"], bg)
    return out


def _sample_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(candidates, count))
        if all(cols[i + 1] - cols[i] >= 3 for i in range(count - 1)):
            return cols
    return None


def _sample_heights(height: int, count: int, cols: list[int]) -> list[int] | None:
    options = list(range(2, min(height - 2, 10) + 1))
    if len(options) < count:
        return None
    for _ in range(200):
        picked = random.sample(options, count)
        by_col = [h for _c, h in sorted(zip(cols, picked))]
        if by_col != sorted(by_col):
            return picked
    return None


def _expand_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(blocked.shape[0], r0 + h + margin)
    cc1 = min(blocked.shape[1], c0 + w + margin)
    blocked[rr0:rr1, cc0:cc1] = True


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = []
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return _trim_mask(mask)


def _sample_sprite_patch(color: int, bg: int) -> np.ndarray | None:
    dims = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
    for _ in range(200):
        h, w = random.choice(dims)
        area = random.randint(3, min(6, h * w - 1))
        mask = _random_connected_mask(h, w, area)
        mh, mw = mask.shape
        if mh < 2 or mw < 2:
            continue
        if not _mask_connected(mask):
            continue
        if not _is_rot_asymmetric(mask):
            continue
        return _make_patch_from_mask(mask, color, bg)
    return None


def _place_sprite(
    blocked: np.ndarray,
    grid_shape: tuple[int, int],
    input_patch: np.ndarray,
    output_patch: np.ndarray,
) -> tuple[int, int] | None:
    h, w = grid_shape
    reserve_h = max(input_patch.shape[0], output_patch.shape[0])
    reserve_w = max(input_patch.shape[1], output_patch.shape[1])
    for _ in range(400):
        top = random.randint(0, h - reserve_h)
        left = random.randint(0, w - reserve_w)
        if np.any(blocked[top : top + reserve_h, left : left + reserve_w]):
            continue
        _expand_blocked(blocked, top, left, reserve_h, reserve_w, margin=1)
        return top, left
    return None


def _compose_output(shape: tuple[int, int], bg: int, sprites: list[dict], turns_by_color: dict[int, int]) -> np.ndarray | None:
    out = np.full(shape, bg, dtype=int)
    for sprite in sprites:
        rotated = _rotate_cw(sprite["input_patch"], turns_by_color[sprite["color"]])
        ph, pw = rotated.shape
        top, left = sprite["top"], sprite["left"]
        if top + ph > shape[0] or left + pw > shape[1]:
            return None
        target = out[top : top + ph, left : left + pw]
        mask = rotated != bg
        if np.any((target != bg) & mask):
            return None
        target[mask] = rotated[mask]
    return out


def _semantic_checks(inp: np.ndarray, sprites: list[dict], bars: list[dict], correct: np.ndarray) -> bool:
    turns_correct = {bar["color"]: bar["rank"] for bar in bars}
    try:
        if not np.array_equal(main(inp), correct):
            return False
    except AssertionError:
        return False

    turns_height = {bar["color"]: bar["height"] for bar in bars}
    height_shortcut = _compose_output(inp.shape, BG, sprites, turns_height)
    if height_shortcut is not None and np.array_equal(height_shortcut, correct):
        return False

    turns_left_to_right = {
        bar["color"]: rank for rank, bar in enumerate(sorted(bars, key=lambda item: item["col"]), start=1)
    }
    lr_shortcut = _compose_output(inp.shape, BG, sprites, turns_left_to_right)
    if lr_shortcut is not None and np.array_equal(lr_shortcut, correct):
        return False

    sprite_lr = [sprite["color"] for sprite in sorted(sprites, key=lambda item: (item["left"], item["top"]))]
    bar_lr = [bar["color"] for bar in sorted(bars, key=lambda item: item["col"])]
    if sprite_lr == bar_lr:
        return False

    if len({tuple(correct.flatten())}) != 1 and np.array_equal(inp, correct):
        return False

    # At least one bar must disagree with raw-height quarter-turns.
    if not any((bar["height"] - bar["rank"]) % 4 != 0 for bar in bars):
        return False

    return True


def _attempt(preferred_count: int | None = None) -> np.ndarray | None:
    count = preferred_count if preferred_count is not None else random.choice((2, 3, 4))
    height = random.randint(13, 19)
    width = random.randint(14, 22)

    cols = _sample_columns(width, count)
    if cols is None:
        return None
    heights = _sample_heights(height, count, cols)
    if heights is None:
        return None
    colors = random.sample(TIP_COLORS, count)

    inp = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    bars: list[dict] = []

    for color, col, bar_h in zip(colors, cols, heights):
        top = height - bar_h
        inp[top, col] = color
        inp[top + 1 : height, col] = GREY
        _expand_blocked(blocked, top, col, bar_h, 1, margin=1)
        bars.append({"color": color, "height": bar_h, "col": col})

    for rank, bar in enumerate(sorted(bars, key=lambda item: item["height"]), start=1):
        bar["rank"] = rank

    sprites: list[dict] = []
    for color in random.sample(colors, len(colors)):
        bar = next(item for item in bars if item["color"] == color)
        output_patch = _sample_sprite_patch(color, BG)
        if output_patch is None:
            return None
        input_patch = np.rot90(output_patch, k=bar["rank"] % 4)
        placement = _place_sprite(blocked, inp.shape, input_patch, output_patch)
        if placement is None:
            return None
        top, left = placement
        _stamp(inp, input_patch, top, left, BG)
        sprites.append(
            {
                "color": color,
                "top": top,
                "left": left,
                "input_patch": input_patch,
            }
        )

    correct = _compose_output(inp.shape, BG, sprites, {bar["color"]: bar["rank"] for bar in bars})
    if not _semantic_checks(inp, sprites, bars, correct):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    preferred_count = _COUNT_CYCLE[_GEN_CALLS % len(_COUNT_CYCLE)]
    _GEN_CALLS += 1

    for _ in range(600):
        grid = _attempt(preferred_count=preferred_count)
        if grid is not None:
            return grid

    for _ in range(600):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bar-rank rotation input")
