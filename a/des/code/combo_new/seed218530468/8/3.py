import random
from collections import Counter

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
ROTATIONS = {"top": 0, "right": 3, "bottom": 2, "left": 1}
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
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
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _opened_rectangle_info(coords: np.ndarray) -> dict | None:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    sub = np.zeros((h, w), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True

    missing = ~sub
    box = _bbox_from_mask(missing)
    if box is None:
        return None
    mr0, mc0, mr1, mc1 = box
    if int(missing.sum()) != (mr1 - mr0 + 1) * (mc1 - mc0 + 1):
        return None

    touching = []
    if mr0 == 0:
        touching.append("top")
    if mr1 == h - 1:
        touching.append("bottom")
    if mc0 == 0:
        touching.append("left")
    if mc1 == w - 1:
        touching.append("right")
    if len(touching) != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "hole_bbox": (r0 + mr0, c0 + mc0, r0 + mr1, c0 + mc1),
        "side": touching[0],
        "area": int(coords.shape[0]),
        "coords": coords,
    }


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    region[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored sprite to the same-colored opened rectangle, rotate it by the opening direction, and place it into the hole.

    Input structure:
    - The background is black.
    - Several distinct non-background colors each form one large monochrome rectangle-like component whose bbox is full except for one rectangular background hole touching exactly one side.
    - For every such color, a separate small sprite of the same color sits elsewhere on the background; its tight bbox is the template canvas to be rotated and inserted.

    Rule:
    - For each non-background color, identify the large opened rectangle and the same-colored sprite outside it.
    - Read the hole opening direction from which side of the large component's bbox is removed.
    - Treat a top opening as 0 degrees, then rotate the sprite clockwise by 90, 180, or 270 degrees for right, bottom, or left openings respectively, place it into the hole, and erase the original sprite from its source location.

    Key insight:
    - Color decides which sprite belongs to which rectangle, while the side on which the rectangular hole opens encodes the sprite rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        color_mask = g == color
        comps = _components(color_mask)
        candidates = []
        for comp in comps:
            info = _opened_rectangle_info(comp)
            if info is not None:
                candidates.append(info)
        assert len(candidates) == 1, "expected one opened rectangle per color"
        rect = candidates[0]

        rect_mask = np.zeros_like(color_mask, dtype=bool)
        rect_mask[rect["coords"][:, 0], rect["coords"][:, 1]] = True
        sprite_mask = color_mask & ~rect_mask
        sprite_box = _bbox_from_mask(sprite_mask)
        assert sprite_box is not None, "expected one same-colored sprite outside the rectangle"
        sr0, sc0, sr1, sc1 = sprite_box
        sprite = sprite_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]

        side = rect["side"]
        rotated = np.rot90(sprite, ROTATIONS[side])
        hr0, hc0, hr1, hc1 = rect["hole_bbox"]
        hole_h = hr1 - hr0 + 1
        hole_w = hc1 - hc0 + 1
        assert rotated.shape == (hole_h, hole_w), "rotated sprite must match hole size"

        out[sprite_mask] = bg
        hole_region = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
        hole_region[:] = bg
        hole_region[rotated] = color

    return out


def _random_sprite_mask(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)

        forced = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        for r, c in forced:
            mask[r, c] = True

        density = rng.uniform(0.30, 0.70)
        for r in range(h):
            for c in range(w):
                if not mask[r, c] and rng.random() < density:
                    mask[r, c] = True

        area = int(mask.sum())
        if area < 2 or area == h * w:
            continue
        if not (~mask).any():
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if h == w and any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    return None


def _build_object(color: int, side: str, rng: random.Random) -> dict | None:
    sh = rng.randint(2, 5)
    sw = rng.randint(2, 5)
    sprite_mask = _random_sprite_mask(sh, sw, rng)
    if sprite_mask is None:
        return None

    rotated = np.rot90(sprite_mask, ROTATIONS[side])
    hole_h, hole_w = rotated.shape

    if side in ("top", "bottom"):
        left = rng.randint(1, 3)
        right = rng.randint(1, 3)
        other = rng.randint(1, 3)
        rect_h = hole_h + other
        rect_w = left + hole_w + right
        hole_r0 = 0 if side == "top" else rect_h - hole_h
        hole_c0 = left
    else:
        top = rng.randint(1, 3)
        bottom = rng.randint(1, 3)
        other = rng.randint(1, 3)
        rect_h = top + hole_h + bottom
        rect_w = hole_w + other
        hole_r0 = top
        hole_c0 = 0 if side == "left" else rect_w - hole_w

    hole_r1 = hole_r0 + hole_h
    hole_c1 = hole_c0 + hole_w

    rect_input = np.full((rect_h, rect_w), color, dtype=int)
    rect_input[hole_r0:hole_r1, hole_c0:hole_c1] = BG
    rect_output = rect_input.copy()
    rect_output[hole_r0:hole_r1, hole_c0:hole_c1][rotated] = color

    sprite_patch = np.where(sprite_mask, color, BG).astype(int)
    return {
        "color": color,
        "side": side,
        "rect_input": rect_input,
        "rect_output": rect_output,
        "sprite_input": sprite_patch,
    }


def _candidate_positions(occupied: np.ndarray, shape: tuple[int, int], pad: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = shape
    pos = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - pad)
            cc0 = max(0, c0 - pad)
            rr1 = min(h, r0 + ph + pad)
            cc1 = min(w, c0 + pw + pad)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            pos.append((r0, c0))
    return pos


def _reserve(occupied: np.ndarray, r0: int, c0: int, shape: tuple[int, int], pad: int) -> None:
    h, w = occupied.shape
    ph, pw = shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + ph + pad)
    cc1 = min(w, c0 + pw + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _sample_grid_shape(objects: list[dict], rng: random.Random) -> tuple[int, int] | None:
    dims = []
    for obj in objects:
        dims.append((obj["rect_input"].shape[0] + 2, obj["rect_input"].shape[1] + 2))
        dims.append((obj["sprite_input"].shape[0] + 2, obj["sprite_input"].shape[1] + 2))

    min_h = max(12, max(h for h, _ in dims))
    min_w = max(12, max(w for _, w in dims))
    area_need = sum(h * w for h, w in dims) + 20 * len(objects)

    candidates = [
        (h, w)
        for h in range(min_h, 31)
        for w in range(min_w, 31)
        if h * w >= area_need
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def _assemble_grids(objects: list[dict], shape: tuple[int, int], rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    items = []
    for obj in objects:
        items.append(("rect", obj, obj["rect_input"].shape[0] * obj["rect_input"].shape[1]))
    for obj in objects:
        items.append(("sprite", obj, obj["sprite_input"].shape[0] * obj["sprite_input"].shape[1]))
    items.sort(key=lambda item: (-item[2], item[0] != "rect"))

    for kind, obj, _area in items:
        patch_in = obj["rect_input"] if kind == "rect" else obj["sprite_input"]
        patch_out = obj["rect_output"] if kind == "rect" else None
        positions = _candidate_positions(occupied, patch_in.shape, pad=1)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _place_patch(inp, patch_in, r0, c0)
        if patch_out is not None:
            _place_patch(out, patch_out, r0, c0)
        _reserve(occupied, r0, c0, patch_in.shape, pad=1)

    return inp, out


def _valid_background_dominance(grid: np.ndarray) -> bool:
    return _background_color(grid) == BG


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(1, 4)
    colors = rng.sample(list(range(1, 10)), n_objects)
    sides = rng.sample(list(SIDES), n_objects)

    objects = []
    for color, side in zip(colors, sides):
        obj = _build_object(color, side, rng)
        if obj is None:
            return None
        objects.append(obj)

    shape = _sample_grid_shape(objects, rng)
    if shape is None:
        return None

    assembled = _assemble_grids(objects, shape, rng)
    if assembled is None:
        return None
    inp, expected = assembled

    if not _valid_background_dominance(inp):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
