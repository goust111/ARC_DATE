from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
FILL_COLORS = (2, 3, 4, 6, 7, 8, 9)
CUP_WIDTHS = (3, 4, 5)
CUP_HEIGHTS = (3, 4)
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _cup_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[-1, :] = True
    return mask


def _mirror_box(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0, width - 1 - c1, r1, width - 1 - c0


def _find_cups(grid: np.ndarray) -> list[dict]:
    blue_components = _components(grid == BLUE)
    assert blue_components, "expected at least one blue cup"

    cups = []
    for coords in blue_components:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        height, width = patch.shape
        assert height in CUP_HEIGHTS and width in CUP_WIDTHS, "unexpected cup bbox"
        assert np.array_equal(patch == BLUE, _cup_mask(height, width)), "blue object must be a U-shaped cup"
        cups.append({"bbox": box})

    assert len(cups) == len(blue_components), "every blue component must be a cup"
    return cups


def _cup_marker(grid: np.ndarray, box: tuple[int, int, int, int]) -> int | None:
    r0, c0, r1, c1 = box
    interior = grid[r0:r1, c0 + 1:c1]
    colors = set(map(int, np.unique(interior))) - {BG, BLUE}
    assert len(colors) <= 1, "cup interior marker must have at most one color"
    return next(iter(colors)) if colors else None


def _paint_filled_cup(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0:r1, c0 + 1:c1] = int(color)
    if r0 > 0:
        grid[r0 - 1, c0 + 1:c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every blue cup from its own or mirrored partner's marker color, then add a matching cap above the opening.

    Input structure:
    - A black background contains several blue U-shaped cup outlines.
    - Cups are arranged in left-right mirror pairs about the grid's vertical midline.
    - Some cups contain a single non-black, non-blue marker pixel inside; their mirrored partners may be unmarked.

    Rule:
    - Detect each blue cup by its U-shaped bbox geometry.
    - Read a cup's interior marker color if it has one; otherwise mirror its bbox across the grid's vertical axis and take the partner cup's marker color.
    - Fill the whole interior of each cup with its resolved color and draw one horizontal cap row of the same color directly above the opening.

    Key insight:
    - The missing fill color is not local to an unmarked cup; it is recovered by pairing cups through global left-right reflection symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cups = _find_cups(g)
    by_box = {cup["bbox"]: cup for cup in cups}

    for cup in cups:
        cup["marker"] = _cup_marker(g, cup["bbox"])

    out = g.copy()
    width = g.shape[1]
    for cup in cups:
        color = cup["marker"]
        if color is None:
            partner_box = _mirror_box(cup["bbox"], width)
            assert partner_box in by_box, "every cup must have a mirrored partner"
            partner_color = by_box[partner_box]["marker"]
            assert partner_color is not None, "an unmarked cup must mirror a marked cup"
            color = partner_color
        _paint_filled_cup(out, cup["bbox"], color)

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _pad_box(box: tuple[int, int, int, int], pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _pair_candidates(height: int, width: int) -> list[dict]:
    candidates: list[dict] = []
    for cup_h in CUP_HEIGHTS:
        for cup_w in CUP_WIDTHS:
            for r0 in range(1, height - cup_h + 1):
                for c0 in range(width - cup_w + 1):
                    c1 = c0 + cup_w - 1
                    mirror_c0 = width - cup_w - c0
                    mirror_c1 = mirror_c0 + cup_w - 1
                    if c0 >= mirror_c0:
                        continue
                    if c1 + 1 >= mirror_c0:
                        continue
                    left_box = (r0, c0, r0 + cup_h - 1, c1)
                    right_box = (r0, mirror_c0, r0 + cup_h - 1, mirror_c1)
                    candidates.append(
                        {
                            "left": left_box,
                            "right": right_box,
                            "left_keep": _pad_box(left_box, 1),
                            "right_keep": _pad_box(right_box, 1),
                        }
                    )
    return candidates


def _choose_layout(height: int, width: int, pair_count: int) -> list[dict] | None:
    candidates = _pair_candidates(height, width)
    for _ in range(300):
        order = _RNG.permutation(len(candidates))
        chosen: list[dict] = []
        keepout: list[tuple[int, int, int, int]] = []
        for idx in order:
            candidate = candidates[int(idx)]
            if any(_boxes_overlap(candidate["left_keep"], box) for box in keepout):
                continue
            if any(_boxes_overlap(candidate["right_keep"], box) for box in keepout):
                continue
            chosen.append(candidate)
            keepout.extend((candidate["left_keep"], candidate["right_keep"]))
            if len(chosen) == pair_count:
                return chosen
    return None


def _draw_cup(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE


def _place_marker(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    marker_row = r1 - 1
    marker_col = int(_RNG.integers(c0 + 1, c1))
    grid[marker_row, marker_col] = int(color)


def _sample_pair_states(pair_count: int) -> list[str]:
    states = [str(_RNG.choice(["left", "right"]))]
    while len(states) < pair_count:
        states.append(str(_RNG.choice(["left", "right", "both"], p=[0.35, 0.35, 0.30])))
    _RNG.shuffle(states)
    return states


def _pick_color(exclude: tuple[int, ...] = ()) -> int:
    options = [color for color in FILL_COLORS if color not in set(map(int, exclude))]
    return int(_RNG.choice(options))


def _build_expected(inp: np.ndarray, placements: list[dict], states: list[str], fills: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for placement, state, (left_fill, right_fill) in zip(placements, states, fills):
        left_box = placement["left"]
        right_box = placement["right"]
        if state == "left":
            right_fill = left_fill
        elif state == "right":
            left_fill = right_fill
        _paint_filled_cup(out, left_box, left_fill)
        _paint_filled_cup(out, right_box, right_fill)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = int(_RNG.integers(13, 16))
    width = int(_RNG.integers(13, 16))
    pair_count = int(_RNG.integers(2, 5))
    placements = _choose_layout(height, width, pair_count)
    if placements is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    states = _sample_pair_states(pair_count)
    fills: list[tuple[int, int]] = []

    for placement, state in zip(placements, states):
        left_box = placement["left"]
        right_box = placement["right"]
        _draw_cup(inp, left_box)
        _draw_cup(inp, right_box)

        left_fill = _pick_color()
        if state == "left":
            right_fill = left_fill
            _place_marker(inp, left_box, left_fill)
        elif state == "right":
            right_fill = _pick_color()
            left_fill = right_fill
            _place_marker(inp, right_box, right_fill)
        else:
            if len(FILL_COLORS) > 1 and _RNG.random() < 0.7:
                right_fill = _pick_color((left_fill,))
            else:
                right_fill = _pick_color()
            _place_marker(inp, left_box, left_fill)
            _place_marker(inp, right_box, right_fill)
        fills.append((left_fill, right_fill))

    expected = _build_expected(inp, placements, states, fills)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid symmetric-cup sample")
