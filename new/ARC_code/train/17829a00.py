from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def _is_full_width_border(comp: dict, h: int, w: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return (
        int(comp["area"]) == int(w)
        and int(r0) == int(r1)
        and int(c0) == 0
        and int(c1) == int(w - 1)
        and int(r0) in {0, int(h - 1)}
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored object vertically until it touches the matching top or bottom border bar.

    Input structure:
    - The background is the dominant color.
    - The top row is a full-width bar in one foreground color and the bottom row is a full-width bar in another.
    - All other foreground pixels are 8-connected monochrome objects whose color matches one of those bars.

    Rule:
    - Top-bar-colored objects move upward so their bbox top is row 1.
    - Bottom-bar-colored objects move downward so their bbox bottom is the row above the bottom bar.
    - A downward-moving one-column segment whose landing overlaps its source becomes the continuous vertical segment swept between them.

    Key insight:
    The horizontal bars are destination anchors: color determines which border an object slides toward, while columns and object shape are otherwise preserved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 1

    bg = int(get_bg(g))
    top_color = int(g[0, 0])
    bottom_color = int(g[h - 1, 0])

    assert top_color != bg
    assert bottom_color != bg
    assert top_color != bottom_color
    assert bool(np.all(g[0, :] == top_color))
    assert bool(np.all(g[h - 1, :] == bottom_color))
    assert set(map(int, np.unique(g))) == {bg, top_color, bottom_color}

    for c in range(w):
        top_rows = [r for r in range(1, h - 1) if int(g[r, c]) == top_color]
        bot_rows = [r for r in range(1, h - 1) if int(g[r, c]) == bottom_color]
        if top_rows and bot_rows:
            assert max(top_rows) < min(bot_rows), "top-color objects must be above bottom-color objects in every column"

    out = np.full_like(g, bg)
    out[0, :] = top_color
    out[h - 1, :] = bottom_color

    for color, target_side in ((top_color, "top"), (bottom_color, "bottom")):
        components = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[color],
            return_format="dict",
        )
        for comp in components:
            if _is_full_width_border(comp, h, w):
                continue

            coords = np.asarray(comp["coords"], dtype=int)
            r0, c0, r1, c1 = map(int, comp["bbox"])
            assert 0 < r0 <= r1 < h - 1

            if target_side == "top":
                dr = 1 - r0
                assert dr <= 0
                paint = coords + np.array([dr, 0], dtype=int)
            else:
                dr = (h - 2) - r1
                assert dr >= 0
                paint = coords + np.array([dr, 0], dtype=int)
                if dr > 0 and r0 + dr <= r1:
                    assert c0 == c1
                    rows = np.arange(r0, r1 + dr + 1, dtype=int)
                    paint = np.column_stack([rows, np.full(rows.shape, c0, dtype=int)])

            assert bool(np.all((0 <= paint[:, 0]) & (paint[:, 0] < h)))
            assert bool(np.all((0 <= paint[:, 1]) & (paint[:, 1] < w)))
            assert bool(np.all(out[paint[:, 0], paint[:, 1]] == bg))
            out[paint[:, 0], paint[:, 1]] = color

    return out

# --------------- generator (o5x) ---------------

from scipy.ndimage import binary_dilation

# -- Grid constants --
BG = 7
H = 16
W = 16

# -- Sprite geometry --
SPRITE_AREA_MIN = 1
SPRITE_AREA_MAX = 6
SPRITE_H_MAX = 4       # bbox height upper bound (exclusive in range)
SPRITE_W_MAX = 5       # bbox width upper bound (exclusive in range)
SPRITE_DENSITY_MIN = 0.33  # min fill ratio inside bbox

# -- Object counts per side --
OBJ_COUNT_MIN = 3      # per color
OBJ_COUNT_MAX = 5      # per color (inclusive)
OBJ_TOTAL_MAX = 9      # top_count + bottom_count

# -- Swept segment (single-column overlap case) --
SWEPT_PROB = 0.15       # probability of including one swept-segment object
SWEPT_H_MIN = 2        # min input height for swept column
SWEPT_H_MAX = 4        # max input height (inclusive)
SWEPT_TOTAL_MAX = 5    # input_h + offset ceiling

# -- Input derivation --
SLIDE_OFFSET_MAX = 10  # exclusive upper bound for slide offsets (range(1, N))

# -- Retry budgets --
PLACE_RETRIES = 60     # sprite placement attempts per object
GEN_RETRIES = 300      # outer generation attempts


def _corner_path(h: int, w: int) -> set[tuple[int, int]]:
    """Generate a diagonal path from one corner of an (h,w) bbox to the opposite."""
    if h == 1 and w == 1:
        return {(0, 0)}
    r0 = random.choice([0, h - 1])
    c0 = random.choice([0, w - 1])
    r1 = h - 1 - r0
    c1 = w - 1 - c0
    steps = max(h, w)
    cells: set[tuple[int, int]] = set()
    for k in range(steps):
        t = 0.0 if steps == 1 else k / (steps - 1)
        cells.add((int(round(r0 + (r1 - r0) * t)),
                    int(round(c0 + (c1 - c0) * t))))
    return cells


def _random_sprite(color: int) -> np.ndarray:
    """Create a small random 8-connected sprite with the given color on BG."""
    area = random.randint(SPRITE_AREA_MIN, SPRITE_AREA_MAX)
    choices = [
        (sh, sw)
        for sh in range(1, SPRITE_H_MAX + 1)
        for sw in range(1, SPRITE_W_MAX + 1)
        if max(sh, sw) <= area <= sh * sw
        and area / (sh * sw) >= SPRITE_DENSITY_MIN
    ]
    sh, sw = random.choice(choices)
    cells = _corner_path(sh, sw)
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < sh and 0 <= cc < sw and (rr, cc) not in cells:
                        frontier.append((rr, cc))
        cells.add(random.choice(frontier))

    sprite = np.full((sh, sw), BG, dtype=int)
    rs, cs = zip(*cells)
    sprite[rs, cs] = color
    return sprite


def _coords_from_patch(patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    """Return (N,2) absolute coords of non-BG pixels in patch placed at (r0,c0)."""
    return np.argwhere(patch != BG) + np.array([r0, c0], dtype=int)


def _is_bar(comp: dict) -> bool:
    """True if a component dict represents a full-width border bar."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (
        int(comp["area"]) == W
        and r0 == r1
        and c0 == 0
        and c1 == W - 1
        and r0 in {0, H - 1}
    )


def _object_count(grid: np.ndarray, color: int) -> int:
    """Count non-bar objects of the given color."""
    comps = find_components(
        grid, bg=BG, connectivity=8, monochromatic=True,
        colors=[color], return_format="dict",
    )
    return sum(1 for comp in comps if not _is_bar(comp))


_STRUCT8 = np.ones((3, 3), dtype=bool)


def _touches_same(mask: np.ndarray, coords: np.ndarray) -> bool:
    """Check if any coord is 8-adjacent to an existing same-color pixel."""
    probe = np.zeros((H, W), dtype=bool)
    probe[coords[:, 0], coords[:, 1]] = True
    dilated = binary_dilation(probe, structure=_STRUCT8)
    dilated[coords[:, 0], coords[:, 1]] = False  # exclude the coords themselves
    return bool((dilated & mask).any())


def _place_output_object(out: np.ndarray, color: int, side: str) -> dict | None:
    """Try to place a random sprite on the output grid touching the bar on `side`."""
    for _ in range(PLACE_RETRIES):
        patch = _random_sprite(color)
        ph, pw = map(int, patch.shape)
        r0 = 1 if side == "top" else H - 1 - ph
        cols = list(range(0, W - pw + 1))
        random.shuffle(cols)
        for c0 in cols:
            coords = _coords_from_patch(patch, r0, c0)
            if bool(np.all(out[coords[:, 0], coords[:, 1]] == BG)):
                paste(out, patch, r0, c0, transparent=BG)
                return {
                    "color": color, "side": side, "coords": coords,
                    "height": ph, "width": pw, "special": False,
                }
    return None


def _place_swept_output_object(out: np.ndarray, color: int) -> dict | None:
    """Try to place a single-column swept-segment bottom object."""
    choices = [
        (input_h, offset)
        for input_h in range(SWEPT_H_MIN, SWEPT_H_MAX + 1)
        for offset in range(1, input_h)
        if input_h + offset <= SWEPT_TOTAL_MAX
    ]
    random.shuffle(choices)
    cols = list(range(W))
    random.shuffle(cols)
    for input_h, offset in choices:
        out_h = input_h + offset
        patch = np.full((out_h, 1), color, dtype=int)
        r0 = H - 1 - out_h
        for c0 in cols:
            coords = _coords_from_patch(patch, r0, c0)
            if bool(np.all(out[coords[:, 0], coords[:, 1]] == BG)):
                paste(out, patch, r0, c0, transparent=BG)
                input_rows = np.arange(r0, r0 + input_h, dtype=int)
                input_coords = np.column_stack(
                    [input_rows, np.full(input_rows.shape, c0, dtype=int)]
                )
                return {
                    "color": color, "side": "bottom", "coords": coords,
                    "input_coords": input_coords,
                    "height": input_h, "width": 1, "special": True,
                }
    return None


def _can_paint_input(
    occupied: np.ndarray, color_mask: np.ndarray, coords: np.ndarray
) -> bool:
    """Check that coords are unoccupied and not 8-adjacent to same-color pixels."""
    if not bool(np.all(occupied[coords[:, 0], coords[:, 1]] == 0)):
        return False
    return not _touches_same(color_mask, coords)


def _check_column_order(
    coords: np.ndarray, color: int, top_color: int,
    top_max: np.ndarray, bot_min: np.ndarray,
) -> bool:
    """Verify that placing coords respects per-column top/bottom ordering."""
    rs, cs = coords[:, 0], coords[:, 1]
    if color == top_color:
        return bool(np.all(rs < bot_min[cs]))
    return bool(np.all(rs > top_max[cs]))


def _update_column_bounds(
    coords: np.ndarray, color: int, top_color: int,
    top_max: np.ndarray, bot_min: np.ndarray,
) -> None:
    """Update per-column top_max / bot_min after placing coords."""
    for r, c in coords:
        r, c = int(r), int(c)
        if color == top_color:
            top_max[c] = max(top_max[c], r)
        else:
            bot_min[c] = min(bot_min[c], r)


def _derive_input(
    objects: list[dict], top_color: int, bottom_color: int,
) -> np.ndarray | None:
    """Slide output objects away from their bars to produce a valid input grid."""
    inp = np.full((H, W), BG, dtype=int)
    inp[0, :] = top_color
    inp[H - 1, :] = bottom_color
    occupied = inp != BG
    masks = {top_color: inp == top_color, bottom_color: inp == bottom_color}

    top_max = np.full(W, 0, dtype=int)
    bot_min = np.full(W, H - 1, dtype=int)

    ordered = objects[:]
    random.shuffle(ordered)
    for obj in ordered:
        color = int(obj["color"])
        if obj["special"]:
            coord_choices = [obj["input_coords"]]
        elif obj["side"] == "top":
            offsets = list(range(1, SLIDE_OFFSET_MAX))
            random.shuffle(offsets)
            coord_choices = [obj["coords"] + np.array([off, 0], dtype=int) for off in offsets]
        else:
            min_offset = int(obj["height"])
            offsets = list(range(min_offset, SLIDE_OFFSET_MAX))
            random.shuffle(offsets)
            coord_choices = [obj["coords"] - np.array([off, 0], dtype=int) for off in offsets]

        placed = False
        for coords in coord_choices:
            if not bool(np.all((1 <= coords[:, 0]) & (coords[:, 0] <= H - 2))):
                continue
            if not _check_column_order(coords, color, top_color, top_max, bot_min):
                continue
            if _can_paint_input(occupied, masks[color], coords):
                inp[coords[:, 0], coords[:, 1]] = color
                occupied[coords[:, 0], coords[:, 1]] = True
                masks[color][coords[:, 0], coords[:, 1]] = True
                _update_column_bounds(coords, color, top_color, top_max, bot_min)
                placed = True
                break
        if not placed:
            return None

    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build output grid, then derive a valid input."""
    colors = [c for c in range(10) if c != BG]
    top_color, bottom_color = random.sample(colors, 2)

    count_pairs = [
        (a, b)
        for a in range(OBJ_COUNT_MIN, OBJ_COUNT_MAX + 1)
        for b in range(OBJ_COUNT_MIN, OBJ_COUNT_MAX + 1)
        if a + b <= OBJ_TOTAL_MAX
    ]
    top_count, bottom_count = random.choice(count_pairs)
    include_swept = random.random() < SWEPT_PROB

    out = np.full((H, W), BG, dtype=int)
    out[0, :] = top_color
    out[H - 1, :] = bottom_color

    objects: list[dict] = []
    for _ in range(top_count):
        obj = _place_output_object(out, top_color, "top")
        if obj is None:
            return None
        objects.append(obj)

    if include_swept:
        obj = _place_swept_output_object(out, bottom_color)
        if obj is None:
            return None
        objects.append(obj)

    for _ in range(bottom_count - int(include_swept)):
        obj = _place_output_object(out, bottom_color, "bottom")
        if obj is None:
            return None
        objects.append(obj)

    inp = _derive_input(objects, top_color, bottom_color)
    if inp is None:
        return None
    if _object_count(inp, top_color) != top_count:
        return None
    if _object_count(inp, bottom_color) != bottom_count:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 17829a00 input")
