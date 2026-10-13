from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BACKGROUND = 0
RED = 2
BODY_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("up", "down", "left", "right")
SIDE_SHIFT = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MOVE_FROM_DAMAGE_SIDE = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

GRID_SIDE_RANGE = (19, 29)
SPRITE_COUNT_RANGE = (2, 5)
SHAPE_SIDE_RANGE = (3, 6)
PLACEMENT_MARGIN = 1
MAX_DAMAGE_PIXELS = 4
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 400
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            cells.append((cr, cc2))
            for nr, nc in ((cr - 1, cc2), (cr + 1, cc2), (cr, cc2 - 1), (cr, cc2 + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(cells, dtype=int))
    return out


def _damage_side(coords: np.ndarray, damage_coords: np.ndarray) -> str:
    center_r = float(coords[:, 0].mean())
    center_c = float(coords[:, 1].mean())
    options: list[tuple[float, str]] = []
    if np.all(damage_coords[:, 0] < center_r):
        options.append((center_r - float(damage_coords[:, 0].mean()), "up"))
    if np.all(damage_coords[:, 0] > center_r):
        options.append((float(damage_coords[:, 0].mean()) - center_r, "down"))
    if np.all(damage_coords[:, 1] < center_c):
        options.append((center_c - float(damage_coords[:, 1].mean()), "left"))
    if np.all(damage_coords[:, 1] > center_c):
        options.append((float(damage_coords[:, 1].mean()) - center_c, "right"))
    if not options:
        raise AssertionError("damage pixels must lie on one side of the sprite centroid")
    options.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return options[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each damaged sprite opposite to the side where its red motor pixels are clustered, then repair the damage.

    Input structure:
    - A black background contains several separated monochrome sprites.
    - Inside each sprite, some body pixels are replaced by red damage pixels.
    - The red pixels of one sprite all lie on the same side of that sprite's centroid.

    Rule:
    - Treat each non-black connected component as one rigid sprite.
    - For each sprite, infer the red cluster side relative to the centroid; move the whole sprite in the opposite direction by exactly the number of red pixels.
    - Erase the source position and draw only the translated repaired sprite in its body color.

    Key insight:
    - The red pixels are a two-part instruction: their side encodes direction, and their count encodes distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    out = np.full_like(g, bg)

    for coords in _components4(g != bg):
        values = g[coords[:, 0], coords[:, 1]]
        damage_coords = coords[values == RED]
        assert len(damage_coords) > 0, "each sprite must contain red damage pixels"

        body_colors = sorted(set(map(int, values)) - {RED})
        assert len(body_colors) == 1, "each sprite must have exactly one body color"
        body_color = body_colors[0]

        side = _damage_side(coords, damage_coords)
        dr, dc = MOVE_FROM_DAMAGE_SIDE[side]
        distance = int(len(damage_coords))
        shifted = coords + np.array([dr * distance, dc * distance], dtype=int)

        assert np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < g.shape[0]))
        assert np.all((0 <= shifted[:, 1]) & (shifted[:, 1] < g.shape[1]))
        assert np.all(out[shifted[:, 0], shifted[:, 1]] == bg), "translated sprites must not overlap"

        out[shifted[:, 0], shifted[:, 1]] = body_color

    return out


def _random_shape() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(*SHAPE_SIDE_RANGE)
        w = random.randint(*SHAPE_SIDE_RANGE)
        mask = np.ones((h, w), dtype=bool)
        variant = random.choice(("rect", "corner_bite", "corner_bite", "slab_bite"))

        if variant == "corner_bite" and h >= 4 and w >= 4:
            bite_h = random.randint(1, h - 2)
            bite_w = random.randint(1, w - 2)
            corner = random.choice(("tl", "tr", "bl", "br"))
            row_slice = slice(0, bite_h) if "t" in corner else slice(h - bite_h, h)
            col_slice = slice(0, bite_w) if "l" in corner else slice(w - bite_w, w)
            mask[row_slice, col_slice] = False
        elif variant == "slab_bite" and h >= 5 and w >= 5:
            if random.random() < 0.5:
                bite_h = random.randint(1, h - 2)
                bite_w = random.randint(1, max(1, w // 2))
                row0 = random.randint(1, h - bite_h - 1)
                if random.random() < 0.5:
                    mask[row0 : row0 + bite_h, :bite_w] = False
                else:
                    mask[row0 : row0 + bite_h, w - bite_w : w] = False
            else:
                bite_h = random.randint(1, max(1, h // 2))
                bite_w = random.randint(1, w - 2)
                col0 = random.randint(1, w - bite_w - 1)
                if random.random() < 0.5:
                    mask[:bite_h, col0 : col0 + bite_w] = False
                else:
                    mask[h - bite_h : h, col0 : col0 + bite_w] = False

        coords = np.argwhere(mask)
        if len(coords) < 5:
            continue
        if len(_components4(mask)) != 1:
            continue
        return coords.astype(int)
    raise RuntimeError("failed to build a connected sprite shape")


def _frontier_candidates(coords: np.ndarray, side: str) -> list[tuple[int, int]]:
    center_r = float(coords[:, 0].mean())
    center_c = float(coords[:, 1].mean())
    points = [tuple(map(int, rc)) for rc in coords.tolist()]

    if side == "up":
        pool = [pt for pt in points if pt[0] < center_r]
        if not pool:
            return []
        rows = sorted({pt[0] for pt in pool})[:2]
        pool = [pt for pt in pool if pt[0] in rows]
        pool.sort(key=lambda pt: (pt[0], abs(pt[1] - center_c), pt[1]))
        return pool

    if side == "down":
        pool = [pt for pt in points if pt[0] > center_r]
        if not pool:
            return []
        rows = sorted({pt[0] for pt in pool})[-2:]
        pool = [pt for pt in pool if pt[0] in rows]
        pool.sort(key=lambda pt: (-pt[0], abs(pt[1] - center_c), pt[1]))
        return pool

    if side == "left":
        pool = [pt for pt in points if pt[1] < center_c]
        if not pool:
            return []
        cols = sorted({pt[1] for pt in pool})[:2]
        pool = [pt for pt in pool if pt[1] in cols]
        pool.sort(key=lambda pt: (pt[1], abs(pt[0] - center_r), pt[0]))
        return pool

    pool = [pt for pt in points if pt[1] > center_c]
    if not pool:
        return []
    cols = sorted({pt[1] for pt in pool})[-2:]
    pool = [pt for pt in pool if pt[1] in cols]
    pool.sort(key=lambda pt: (-pt[1], abs(pt[0] - center_r), pt[0]))
    return pool


def _choose_damage(coords: np.ndarray, side: str) -> np.ndarray | None:
    center_r = float(coords[:, 0].mean())
    center_c = float(coords[:, 1].mean())
    pool = _frontier_candidates(coords, side)
    if not pool:
        return None

    choices: list[tuple[tuple[float, int, int, int], np.ndarray]] = []
    max_k = min(MAX_DAMAGE_PIXELS, len(pool))
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(pool, k):
            damage = np.array(combo, dtype=int)
            if side == "up" and not np.all(damage[:, 0] < center_r):
                continue
            if side == "down" and not np.all(damage[:, 0] > center_r):
                continue
            if side == "left" and not np.all(damage[:, 1] < center_c):
                continue
            if side == "right" and not np.all(damage[:, 1] > center_c):
                continue

            if side in ("up", "down"):
                if np.all(damage[:, 1] < center_c) or np.all(damage[:, 1] > center_c):
                    continue
                extreme = float(np.mean(np.abs(damage[:, 0] - center_r)))
                spread = int(damage[:, 1].max() - damage[:, 1].min())
            else:
                if np.all(damage[:, 0] < center_r) or np.all(damage[:, 0] > center_r):
                    continue
                extreme = float(np.mean(np.abs(damage[:, 1] - center_c)))
                spread = int(damage[:, 0].max() - damage[:, 0].min())

            compactness = int((damage[:, 0].max() - damage[:, 0].min()) + (damage[:, 1].max() - damage[:, 1].min()))
            score = (extreme, spread, -compactness, -k)
            choices.append((score, damage))

    if not choices:
        return None

    choices.sort(key=lambda item: item[0], reverse=True)
    top_choices = choices[: min(12, len(choices))]
    return random.choice(top_choices)[1]


def _sweep_cells(final_coords: np.ndarray, side: str, distance: int) -> set[tuple[int, int]]:
    dr, dc = SIDE_SHIFT[side]
    sweep: set[tuple[int, int]] = set()
    for step in range(distance + 1):
        shifted = final_coords + np.array([dr * step, dc * step], dtype=int)
        for r, c in shifted:
            sweep.add((int(r), int(c)))
    return sweep


def _halo(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                out.add((nr, nc))
    return out


def _place_sprite(height: int, width: int, forbidden: set[tuple[int, int]]) -> dict | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        final_coords = _random_shape()
        side = random.choice(SIDES)
        damage_local = _choose_damage(final_coords, side)
        if damage_local is None:
            continue

        distance = int(len(damage_local))
        relative_sweep = _sweep_cells(final_coords, side, distance)
        rel_rows = [r for r, _ in relative_sweep]
        rel_cols = [c for _, c in relative_sweep]

        row_lo = PLACEMENT_MARGIN - min(rel_rows)
        row_hi = (height - 1 - PLACEMENT_MARGIN) - max(rel_rows)
        col_lo = PLACEMENT_MARGIN - min(rel_cols)
        col_hi = (width - 1 - PLACEMENT_MARGIN) - max(rel_cols)
        if row_lo > row_hi or col_lo > col_hi:
            continue

        anchors = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        random.shuffle(anchors)
        for top, left in anchors[:150]:
            anchor = np.array([top, left], dtype=int)
            abs_final = final_coords + anchor
            sweep = _sweep_cells(abs_final, side, distance)
            if any(cell in forbidden for cell in sweep):
                continue

            start_shift = np.array(SIDE_SHIFT[side], dtype=int) * distance
            abs_damage = damage_local + anchor + start_shift
            return {
                "final_coords": abs_final,
                "damage_coords": abs_damage,
                "side": side,
                "distance": distance,
                "halo": _halo(sweep),
            }
    return None


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    sprite_count = random.randint(*SPRITE_COUNT_RANGE)

    sprites: list[dict] = []
    forbidden: set[tuple[int, int]] = set()
    for body_color in random.sample(BODY_COLORS, sprite_count):
        sprite = _place_sprite(height, width, forbidden)
        if sprite is None:
            return None
        sprite["body_color"] = int(body_color)
        sprites.append(sprite)
        forbidden |= sprite["halo"]

    if len({sprite["side"] for sprite in sprites}) < min(2, sprite_count):
        return None
    if len({sprite["distance"] for sprite in sprites}) < min(2, sprite_count):
        return None

    inp = np.full((height, width), BACKGROUND, dtype=int)
    expected = np.full((height, width), BACKGROUND, dtype=int)

    for sprite in sprites:
        body_color = int(sprite["body_color"])
        final_coords = sprite["final_coords"]
        expected[final_coords[:, 0], final_coords[:, 1]] = body_color

        start_shift = np.array(SIDE_SHIFT[sprite["side"]], dtype=int) * int(sprite["distance"])
        input_coords = final_coords + start_shift
        if np.any(inp[input_coords[:, 0], input_coords[:, 1]] != BACKGROUND):
            return None
        inp[input_coords[:, 0], input_coords[:, 1]] = body_color
        inp[sprite["damage_coords"][:, 0], sprite["damage_coords"][:, 1]] = RED

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid damaged-motor ARC input")
