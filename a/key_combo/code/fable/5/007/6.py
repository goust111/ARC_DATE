from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


# The helpers below follow arc_utils' canonical conventions: coordinates are
# (row, column), and bounding boxes have inclusive endpoints.  They are kept
# local so this file can be loaded directly with SourceFileLoader.


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    unseen = mask.astype(bool).copy()
    result: list[np.ndarray] = []
    while unseen.any():
        start = tuple(map(int, np.argwhere(unseen)[0]))
        unseen[start] = False
        queue = deque([start])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and unseen[rr, cc]:
                    unseen[rr, cc] = False
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinates to their inclusive bbox and return its mask."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _bottom_profile(sprite: np.ndarray) -> np.ndarray:
    assert sprite.ndim == 2 and sprite.any()
    assert np.all(sprite.any(axis=0)), "sprite bbox may not contain an empty column"
    return np.asarray(
        [int(np.flatnonzero(sprite[:, c])[-1]) for c in range(sprite.shape[1])],
        dtype=int,
    )


def _surface(terrain: np.ndarray) -> np.ndarray:
    """Top occupied row of the composite terrain in every global column."""
    assert np.all(terrain.any(axis=0)), "the terrain must support every column"
    return np.asarray(
        [int(np.flatnonzero(terrain[:, c])[0]) for c in range(terrain.shape[1])],
        dtype=int,
    )


def _snug_landings(sprite: np.ndarray, terrain: np.ndarray) -> list[tuple[int, int]]:
    """Enumerate (top row, left column) placements with full bottom contact."""
    h, w = terrain.shape
    sh, sw = sprite.shape
    bottom = _bottom_profile(sprite)
    top_surface = _surface(terrain)
    coords = np.argwhere(sprite)
    landings: list[tuple[int, int]] = []

    for left in range(w - sw + 1):
        # Full contact means every sprite bottom is exactly one cell above the
        # current top surface.  Therefore these offsets must all agree.
        offsets = top_surface[left : left + sw] - bottom
        if not np.all(offsets == offsets[0]):
            continue
        top = int(offsets[0] - 1)
        if top < 0 or top + sh > h:
            continue
        placed = coords + np.array([top, left], dtype=int)
        if terrain[placed[:, 0], placed[:, 1]].any():
            continue
        landings.append((top, left))
    return landings


def _extract_problem(grid: np.ndarray) -> tuple[int, int, np.ndarray, list[dict]]:
    """Identify the bottom-connected base and the differently colored sprites."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background(g)
    bottom_colors = {int(v) for v in g[-1] if int(v) != bg}
    assert len(bottom_colors) == 1, "expected one base color across the bottom"
    base_color = bottom_colors.pop()
    assert np.all(g[-1] == base_color), "the base must span the full bottom row"

    terrain = g == base_color
    assert len(_components(terrain)) == 1, "the base must be 4-connected"
    assert np.all(terrain.any(axis=0)), "the base must occur in every column"

    sprite_colors = sorted(set(map(int, np.unique(g))) - {bg, base_color})
    assert len(sprite_colors) >= 2, "expected several differently colored sprites"
    sprites: list[dict] = []
    for color in sprite_colors:
        comps = _components(g == color)
        assert len(comps) == 1, "each sprite color must form one component"
        coords = comps[0]
        mask = _crop_mask(coords)
        assert mask.shape[1] >= 2
        sprites.append({"color": color, "coords": coords, "mask": mask})
    return bg, base_color, terrain, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop floating sprites in the uniquely forced sequence of snug contour fits.

    Input structure:
    - A dominant background contains several separate monochrome sprites, each
      with its own color, above a differently colored 4-connected base.
    - The base spans the bottom row and has an uneven top surface in the global
      row/column coordinate system.

    Rule:
    - Remove all sprites from their display positions.  At each step, test every
      remaining sprite at every horizontal translation.  A snug landing has no
      overlap and places the bottommost cell in every sprite column exactly one
      row above the current composite terrain surface in that global column.
    - Exactly one remaining sprite has exactly one snug landing; place it there
      in its original color, update the composite surface, and repeat.

    Key insight:
    - The order is not encoded by color or position: each landed sprite changes
      the contour that uniquely unlocks the next sprite's interlocking bottom.
    """
    g = np.asarray(grid, dtype=int)
    bg, base_color, terrain, sprites = _extract_problem(g)

    out = np.full_like(g, bg)
    out[terrain] = base_color
    remaining = list(sprites)

    while remaining:
        choices: list[tuple[int, tuple[int, int]]] = []
        for i, sprite in enumerate(remaining):
            landings = _snug_landings(sprite["mask"], terrain)
            if len(landings) == 1:
                choices.append((i, landings[0]))
        assert len(choices) == 1, "expected exactly one uniquely fitting sprite"

        index, (top, left) = choices[0]
        sprite = remaining.pop(index)
        coords = np.argwhere(sprite["mask"]) + np.array([top, left], dtype=int)
        assert not terrain[coords[:, 0], coords[:, 1]].any()
        terrain[coords[:, 0], coords[:, 1]] = True
        out[coords[:, 0], coords[:, 1]] = int(sprite["color"])

    return out


# ---------------------------------------------------------------------------
# Generator: construct a contour-changing chain, then independently audit it.
# ---------------------------------------------------------------------------


_MAX_ATTEMPTS = 1200


def _profile_landings(bottom: np.ndarray, surface: np.ndarray) -> list[tuple[int, int]]:
    """Profile-only semantic oracle, independent of the rendered terrain mask."""
    width = len(bottom)
    result: list[tuple[int, int]] = []
    for left in range(len(surface) - width + 1):
        delta = surface[left : left + width] - bottom
        if np.all(delta == delta[0]) and int(delta[0]) >= 2:
            result.append((int(delta[0] - 1), left))
    return result


def _connected_column_sprite(bottom: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Choose connected filled column intervals with the requested bottom."""
    for _ in range(80):
        top = np.asarray([random.randint(0, int(v)) for v in bottom], dtype=int)
        # A zero-height point of bottom forces normalization to top row zero.
        if int(top.min()) != 0:
            continue
        if any(
            max(int(top[c]), int(top[c + 1]))
            > min(int(bottom[c]), int(bottom[c + 1]))
            for c in range(len(bottom) - 1)
        ):
            continue
        if int(np.sum(bottom - top + 1)) < len(bottom) + 2:
            continue
        mask = np.zeros((int(bottom.max()) + 1, len(bottom)), dtype=bool)
        for c, (u, b) in enumerate(zip(top, bottom)):
            mask[int(u) : int(b) + 1, c] = True
        if len(_components(mask)) == 1:
            return mask, top
    return None


def _initial_surface(height: int, width: int) -> np.ndarray:
    low, high = height - 9, height - 4
    values = [random.randint(low, high)]
    for _ in range(1, width):
        step = random.choice((-1, 0, 1))
        values.append(min(high, max(low, values[-1] + step)))
    return np.asarray(values, dtype=int)


def _build_chain(height: int, width: int, count: int):
    """Return masks, planned placements, and every pre-placement surface."""
    for _ in range(180):
        surface = _initial_surface(height, width)
        surfaces = [surface.copy()]
        masks: list[np.ndarray] = []
        placements: list[tuple[int, int]] = []
        previous_interval: tuple[int, int] | None = None

        for _index in range(count):
            options: list[tuple[int, int]] = []
            widths = list(range(4, min(8, width - 1)))
            random.shuffle(widths)
            for sprite_width in widths:
                for left in range(1, width - sprite_width):
                    segment = surface[left : left + sprite_width]
                    relief = int(segment.max() - segment.min())
                    if not (1 <= relief <= 5) or int(segment.min()) < 4:
                        continue
                    if previous_interval is not None:
                        a0, a1 = previous_interval
                        overlap = min(left + sprite_width, a1) - max(left, a0)
                        if overlap < 2:
                            continue
                    options.append((sprite_width, left))
            if not options:
                break

            random.shuffle(options)
            made = False
            for sprite_width, left in options:
                segment = surface[left : left + sprite_width]
                bottom = segment - int(segment.min())
                built = _connected_column_sprite(bottom)
                if built is None:
                    continue
                mask, top_profile = built
                landing = (int(segment.min() - 1), left)
                if _profile_landings(bottom, surface) != [landing]:
                    continue

                next_surface = surface.copy()
                next_surface[left : left + sprite_width] = landing[0] + top_profile
                if int(next_surface.min()) < 2:
                    continue
                masks.append(mask)
                placements.append(landing)
                surface = next_surface
                surfaces.append(surface.copy())
                previous_interval = (left, left + sprite_width)
                made = True
                break
            if not made:
                break

        if len(masks) != count:
            continue
        if len({mask.shape[1] for mask in masks}) < 2:
            continue
        if len({left for _top, left in placements}) < 2:
            continue

        # At every stage only the intended sprite is a unique fit.  Stronger
        # anti-shortcut condition: every later sprite has no fit at all yet.
        valid = True
        for i in range(count):
            for j in range(i, count):
                landings = _profile_landings(_bottom_profile(masks[j]), surfaces[i])
                if j == i:
                    if landings != [placements[i]]:
                        valid = False
                elif landings:
                    valid = False
            if not valid:
                break
        if not valid:
            continue

        # Explicitly prove that each predecessor changes the target segment
        # and is necessary to unlock its successor.
        for i in range(1, count):
            bottom = _bottom_profile(masks[i])
            if _profile_landings(bottom, surfaces[i - 1]):
                valid = False
                break
            if _profile_landings(bottom, surfaces[i]) != [placements[i]]:
                valid = False
                break
        if valid:
            return masks, placements, surfaces
    return None


def _pack_floating(
    grid: np.ndarray, masks: list[np.ndarray], colors: list[int], surface: np.ndarray
) -> list[np.ndarray] | None:
    """Place visible sprites above the base with one-cell bbox separation."""
    h, w = grid.shape
    order = sorted(range(len(masks)), key=lambda i: masks[i].size, reverse=True)
    reserved = np.zeros((h, w), dtype=bool)
    locations: list[tuple[int, int] | None] = [None] * len(masks)

    candidates: dict[int, list[tuple[int, int]]] = {}
    for i in order:
        sh, sw = masks[i].shape
        spots: list[tuple[int, int]] = []
        for top in range(1, h - sh):
            for left in range(1, w - sw):
                if top + sh + 1 >= int(surface[left : left + sw].min()):
                    continue
                spots.append((top, left))
        random.shuffle(spots)
        candidates[i] = spots

    def place(k: int) -> bool:
        if k == len(order):
            return True
        i = order[k]
        sh, sw = masks[i].shape
        for top, left in candidates[i][:240]:
            r0, r1 = top - 1, top + sh
            c0, c1 = left - 1, left + sw
            if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            reserved[r0 : r1 + 1, c0 : c1 + 1] = True
            locations[i] = (top, left)
            if place(k + 1):
                return True
            reserved[r0 : r1 + 1, c0 : c1 + 1] = False
            locations[i] = None
        return False

    if not place(0):
        return None

    absolute: list[np.ndarray] = []
    for mask, color, location in zip(masks, colors, locations):
        assert location is not None
        top, left = location
        coords = np.argwhere(mask) + np.array([top, left], dtype=int)
        assert np.all(grid[coords[:, 0], coords[:, 1]] == _background(grid))
        grid[coords[:, 0], coords[:, 1]] = color
        absolute.append(coords)
    return absolute


def _render_base(
    height: int, width: int, bg: int, base_color: int, surface: np.ndarray
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for c, top in enumerate(surface):
        grid[int(top) :, c] = base_color
    return grid


def _semantic_audit(
    inp: np.ndarray,
    expected: np.ndarray,
    masks: list[np.ndarray],
    colors: list[int],
    placements: list[tuple[int, int]],
    surfaces: list[np.ndarray],
) -> bool:
    """Check rule semantics, counterfactual dependencies, and rendered output."""
    if inp.ndim != 2 or max(inp.shape) > 30 or expected.shape != inp.shape:
        return False
    if len(masks) < 3 or np.array_equal(inp, expected):
        return False

    # Counterfactual audit independent of main(): no later piece fits before
    # its predecessor, while exactly the planned fit exists immediately after.
    for i, mask in enumerate(masks):
        bottom = _bottom_profile(mask)
        if _profile_landings(bottom, surfaces[i]) != [placements[i]]:
            return False
        if i and _profile_landings(bottom, surfaces[i - 1]):
            return False

    # Every key transition visibly changes the next piece's full target range.
    for i in range(1, len(masks)):
        left = placements[i][1]
        width = masks[i].shape[1]
        if np.array_equal(
            surfaces[i - 1][left : left + width],
            surfaces[i][left : left + width],
        ):
            return False

    try:
        actual = main(inp)
        _bg, _base, _terrain, parsed = _extract_problem(inp)
    except (AssertionError, IndexError, ValueError):
        return False
    if not np.array_equal(actual, expected):
        return False
    if {int(s["color"]) for s in parsed} != set(colors):
        return False
    for color, mask in zip(colors, masks):
        if int(np.count_nonzero(inp == color)) != int(mask.sum()):
            return False
        if int(np.count_nonzero(actual == color)) != int(mask.sum()):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(25, 30)
    width = random.randint(17, 26)
    count = random.randint(3, 5)
    chain = _build_chain(height, width, count)
    if chain is None:
        return None
    masks, placements, surfaces = chain

    palette = random.sample(range(10), count + 2)
    bg, base_color = palette[:2]
    colors = palette[2:]
    inp = _render_base(height, width, bg, base_color, surfaces[0])
    displayed = _pack_floating(inp, masks, colors, surfaces[0])
    if displayed is None:
        return None

    # The display layout is deliberately not an order marker: neither a
    # top-to-bottom nor a left-to-right scan gives the forced landing order.
    sequence = list(range(count))
    by_top = sorted(
        sequence,
        key=lambda i: (int(displayed[i][:, 0].min()), int(displayed[i][:, 1].min())),
    )
    by_left = sorted(
        sequence,
        key=lambda i: (int(displayed[i][:, 1].min()), int(displayed[i][:, 0].min())),
    )
    if by_top == sequence or by_left == sequence:
        return None

    expected = _render_base(height, width, bg, base_color, surfaces[0])
    terrain = expected != bg
    for mask, color, (top, left) in zip(masks, colors, placements):
        coords = np.argwhere(mask) + np.array([top, left], dtype=int)
        if terrain[coords[:, 0], coords[:, 1]].any():
            return None
        terrain[coords[:, 0], coords[:, 1]] = True
        expected[coords[:, 0], coords[:, 1]] = color

    # Background must remain the visually dominant color used by main().
    if _background(inp) != bg:
        return None
    if not _semantic_audit(inp, expected, masks, colors, placements, surfaces):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sequential snug-fit puzzle")
