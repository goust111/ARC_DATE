from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _perimeter_mask(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_hollow_square_frame(g: np.ndarray, box: BBox, bg: int) -> bool:
    r0, c0, r1, c1 = box
    side = r1 - r0 + 1
    if side < 3 or side != c1 - c0 + 1:
        return False

    color = int(g[r0, c0])
    if color == bg:
        return False

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    perim = _perimeter_mask(side)
    if not np.all(sub[perim] == color):
        return False

    # The frame is hollow with respect to its own border color. Other frames or
    # pointer pixels may lie inside larger frames, so only this color is banned.
    return not np.any(sub[~perim] == color)


def _find_frames(g: np.ndarray, bg: int) -> dict[int, BBox]:
    h, w = g.shape
    by_color: dict[int, list[BBox]] = {}
    for side in range(3, min(h, w) + 1):
        for r0 in range(h - side + 1):
            r1 = r0 + side - 1
            for c0 in range(w - side + 1):
                c1 = c0 + side - 1
                box = (r0, c0, r1, c1)
                if _is_hollow_square_frame(g, box, bg):
                    color = int(g[r0, c0])
                    by_color.setdefault(color, []).append(box)

    assert by_color, "expected at least one hollow square frame"
    frames: dict[int, BBox] = {}
    for color, boxes in by_color.items():
        assert len(boxes) == 1, f"frame color {color} must identify one square"
        frames[int(color)] = boxes[0]
    return frames


def _border_cells(frames: dict[int, BBox]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, r1, c1 in frames.values():
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        for r in range(r0 + 1, r1):
            cells.add((r, c0))
            cells.add((r, c1))
    return cells


def _inside(box: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def _assign_markers(
    g: np.ndarray, bg: int, frames: dict[int, BBox]
) -> dict[int, int | None]:
    frame_colors = set(frames)
    border = _border_cells(frames)
    owners: dict[int, list[int]] = {color: [] for color in frames}

    marker_coords = [
        (int(r), int(c))
        for r, c in np.argwhere(g != bg)
        if (int(r), int(c)) not in border
    ]
    assert len(marker_coords) == len(frames) - 1, "expected exactly one empty frame"

    seen_marker_colors: set[int] = set()
    for r, c in marker_coords:
        marker_color = int(g[r, c])
        assert marker_color in frame_colors, "marker color must name a frame color"
        assert marker_color not in seen_marker_colors, "marker colors must be unique"
        seen_marker_colors.add(marker_color)

        containing = [
            (box[2] - box[0] + 1, color)
            for color, box in frames.items()
            if _inside(box, r, c)
        ]
        assert containing, "marker must be inside a frame"
        containing.sort()
        owner = containing[0][1]
        assert owner != marker_color, "a frame cannot point to itself"
        owners[owner].append(marker_color)

    empty = [color for color, vals in owners.items() if not vals]
    assert len(empty) == 1, "there must be one empty innermost frame"

    edges: dict[int, int | None] = {}
    for color, vals in owners.items():
        assert len(vals) <= 1, "each nonempty frame must contain one marker"
        edges[color] = vals[0] if vals else None
    return edges


def _chain_from_edges(edges: dict[int, int | None]) -> list[int]:
    pointed = {target for target in edges.values() if target is not None}
    heads = [color for color in edges if color not in pointed]
    assert len(heads) == 1, "pointer graph must have one chain head"

    chain: list[int] = []
    seen: set[int] = set()
    cur = heads[0]
    while True:
        assert cur not in seen, "pointer graph must be acyclic"
        assert cur in edges, "pointer target must be a frame"
        seen.add(cur)
        chain.append(cur)
        nxt = edges[cur]
        if nxt is None:
            break
        cur = int(nxt)

    assert len(chain) == len(edges), "pointer graph must cover every frame"
    return chain


def _concentric(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.empty((side, side), dtype=int)
    for layer, color in enumerate(colors):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild pointer-linked hollow square frames as a compact concentric square.

    Input structure:
    - The dominant color is the background.
    - Foreground objects include several one-cell-thick hollow square frames,
      each with a different border color.
    - Exactly one frame has no interior pointer pixel. Every other frame has one
      interior marker pixel whose color is another frame's border color, and
      marker colors are unique.

    Rule:
    - Read a marker of color B inside frame A as "the B frame is the next inner
      layer after A".
    - The chain starts at the only frame whose color is not used by any marker
      and ends at the only empty frame.
    - For n frames, return a (2n-1) x (2n-1) square whose layers are filled from
      outside to inside with the chain colors.

    Key insight:
    - Frame sizes, positions, and geometric containment are decoys; the visible
      pointer pixels alone determine the layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    edges = _assign_markers(g, bg, frames)
    chain = _chain_from_edges(edges)
    return _concentric(chain)


# --------------- generator ---------------

GEN_ATTEMPTS = 500
GRID_H_RANGE = (21, 30)
GRID_W_RANGE = (21, 30)
FRAME_COUNT_RANGE = (3, 6)
FRAME_SIDE_RANGE = (3, 9)
PLACEMENT_PAD = 1
COLOR_COUNT = 10


def _nonmonotonic(values: list[int]) -> bool:
    return values != sorted(values) and values != sorted(values, reverse=True)


def _expected_from_chain(chain: list[int]) -> np.ndarray:
    return _concentric(chain)


def _box_with_pad(box: BBox, pad: int, shape: tuple[int, int]) -> BBox:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _overlap(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_frames(h: int, w: int, items: list[tuple[int, int]]) -> dict[int, BBox] | None:
    placed: dict[int, BBox] = {}
    blocked: list[BBox] = []
    order = sorted(items, key=lambda item: item[1], reverse=True)
    for color, side in order:
        choices: list[BBox] = []
        for r0 in range(1, h - side):
            for c0 in range(1, w - side):
                box = (r0, c0, r0 + side - 1, c0 + side - 1)
                padded = _box_with_pad(box, PLACEMENT_PAD, (h, w))
                if any(_overlap(padded, other) for other in blocked):
                    continue
                choices.append(box)
        if not choices:
            return None
        box = random.choice(choices)
        placed[color] = box
        blocked.append(_box_with_pad(box, PLACEMENT_PAD, (h, w)))
    return placed


def _place_frames_with_nested_pair(
    h: int, w: int, items: list[tuple[int, int]]
) -> dict[int, BBox] | None:
    pair_options = [
        (outer, inner)
        for outer in items
        for inner in items
        if outer[0] != inner[0] and outer[1] >= inner[1] + 4
    ]
    if not pair_options:
        return None

    outer, inner = random.choice(pair_options)
    outer_color, outer_side = outer
    inner_color, inner_side = inner

    outer_choices = [
        (r, c, r + outer_side - 1, c + outer_side - 1)
        for r in range(1, h - outer_side)
        for c in range(1, w - outer_side)
    ]
    if not outer_choices:
        return None
    outer_box = random.choice(outer_choices)
    or0, oc0, or1, oc1 = outer_box

    inner_choices = [
        (r, c, r + inner_side - 1, c + inner_side - 1)
        for r in range(or0 + 2, or1 - inner_side)
        for c in range(oc0 + 2, oc1 - inner_side)
    ]
    if not inner_choices:
        return None
    inner_box = random.choice(inner_choices)

    placed = {outer_color: outer_box, inner_color: inner_box}
    blocked = [_box_with_pad(outer_box, PLACEMENT_PAD, (h, w))]
    remaining = [item for item in items if item[0] not in placed]
    remaining.sort(key=lambda item: item[1], reverse=True)
    for color, side in remaining:
        choices: list[BBox] = []
        for r0 in range(1, h - side):
            for c0 in range(1, w - side):
                box = (r0, c0, r0 + side - 1, c0 + side - 1)
                padded = _box_with_pad(box, PLACEMENT_PAD, (h, w))
                if any(_overlap(padded, other) for other in blocked):
                    continue
                choices.append(box)
        if not choices:
            return None
        box = random.choice(choices)
        placed[color] = box
        blocked.append(_box_with_pad(box, PLACEMENT_PAD, (h, w)))
    return placed


def _draw_frame(g: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color


def _smallest_owner(frames: dict[int, BBox], r: int, c: int) -> int | None:
    containing = [
        (box[2] - box[0] + 1, color)
        for color, box in frames.items()
        if _inside(box, r, c)
    ]
    if not containing:
        return None
    containing.sort()
    return containing[0][1]


def _marker_cell(
    g: np.ndarray,
    frames: dict[int, BBox],
    owner: int,
    bg: int,
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = frames[owner]
    cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in used
        and int(g[r, c]) == bg
        and _smallest_owner(frames, r, c) == owner
    ]
    if not cells:
        return None
    return random.choice(cells)


def _render_input(
    h: int,
    w: int,
    bg: int,
    frames: dict[int, BBox],
    chain: list[int],
) -> np.ndarray | None:
    g = np.full((h, w), int(bg), dtype=int)
    for color, box in frames.items():
        _draw_frame(g, box, color)

    used_markers: set[tuple[int, int]] = set()
    for owner, target in zip(chain, chain[1:]):
        cell = _marker_cell(g, frames, owner, bg, used_markers)
        if cell is None:
            return None
        r, c = cell
        g[r, c] = int(target)
        used_markers.add(cell)
    return g


def _geometry_decoys(chain: list[int], sizes: dict[int, int], frames: dict[int, BBox]) -> bool:
    size_seq = [sizes[color] for color in chain]
    if not _nonmonotonic(size_seq):
        return False

    row_seq = [frames[color][0] for color in chain]
    col_seq = [frames[color][1] for color in chain]
    # At least one major geometric cue must disagree with chain order.
    return _nonmonotonic(row_seq) or _nonmonotonic(col_seq)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n = random.randint(*FRAME_COUNT_RANGE)

    bg = random.randrange(COLOR_COUNT)
    palette = [color for color in range(COLOR_COUNT) if color != bg]
    colors = random.sample(palette, n)
    chain = random.sample(colors, n)

    sides = random.sample(range(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1), n)
    random.shuffle(sides)
    sizes = dict(zip(colors, sides))
    if not _nonmonotonic([sizes[color] for color in chain]):
        return None

    items = [(color, sizes[color]) for color in colors]
    if random.random() < 0.35:
        frames = _place_frames_with_nested_pair(h, w, items)
    else:
        frames = _place_frames(h, w, items)
    if frames is None:
        return None
    if not _geometry_decoys(chain, sizes, frames):
        return None

    grid = _render_input(h, w, bg, frames, chain)
    if grid is None:
        return None

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    expected = _expected_from_chain(chain)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Semantic guards: the generated answer depends on pointer order, not size.
    size_sorted = sorted(chain, key=lambda color: sizes[color], reverse=True)
    if np.array_equal(_expected_from_chain(size_sorted), expected):
        return None
    assert max(grid.shape) <= 30 and max(expected.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
