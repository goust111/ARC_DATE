import itertools
from collections import deque
from functools import lru_cache

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

INTERIOR_HEIGHT_RANGE = (12, 16)
INTERIOR_WIDTH_RANGE = (5, 7)
PANEL_COUNT_OPTIONS = (3, 4)
BAR_GROWTH_OPTIONS = (1, 2)
SPRITE_AREA_RANGE = (4, 6)
SPRITE_BBOX_OPTIONS = (
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
)
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _separator_columns(grid: np.ndarray, frame_color: int) -> list[int]:
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == frame_color)]
    assert len(cols) >= 4
    assert cols[0] == 0 and cols[-1] == grid.shape[1] - 1
    strides = [b - a for a, b in zip(cols, cols[1:])]
    assert len(set(strides)) == 1
    return cols


def _extract_panel(panel: np.ndarray, frame_color: int) -> dict:
    h, w = panel.shape
    assert h >= 4 and w >= 4
    assert np.all(panel[0, :] == frame_color)
    assert np.all(panel[-1, :] == frame_color)
    assert np.all(panel[:, 0] == frame_color)
    assert np.all(panel[:, -1] == frame_color)

    interior = panel[1:-1, 1:-1]
    top_row = interior[0]
    bar_color = int(top_row[0])
    assert bar_color != frame_color

    bar_len = 1
    while bar_len < top_row.size and int(top_row[bar_len]) == bar_color:
        bar_len += 1

    non_bar_mask = np.ones_like(interior, dtype=bool)
    non_bar_mask[0, :bar_len] = False
    bg_color = _most_common(interior[non_bar_mask])

    sprite_mask = interior != bg_color
    sprite_mask[0, :bar_len] = False
    coords = np.argwhere(sprite_mask)
    assert coords.size > 0

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    crop = interior[r0 : r1 + 1, c0 : c1 + 1].copy()
    mask = sprite_mask[r0 : r1 + 1, c0 : c1 + 1].copy()

    return {
        "bg": int(bg_color),
        "bar_color": int(bar_color),
        "bar_len": int(bar_len),
        "top": int(r0),
        "left": int(c0),
        "sprite_crop": crop,
        "sprite_mask": mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frame_color = int(g[0, 0])
    assert np.all(g[0, :] == frame_color)
    assert np.all(g[-1, :] == frame_color)

    separators = _separator_columns(g, frame_color)
    panel_w = separators[1] - separators[0] + 1
    panels = [g[:, c0 : c1 + 1] for c0, c1 in zip(separators, separators[1:])]

    infos = [_extract_panel(panel, frame_color) for panel in panels]
    bar_lens = [info["bar_len"] for info in infos]
    bar_steps = [b - a for a, b in zip(bar_lens, bar_lens[1:])]
    assert bar_steps and len(set(bar_steps)) == 1
    delta = int(bar_steps[0])

    tops = [info["top"] for info in infos]
    drops = [b - a for a, b in zip(tops, tops[1:])]
    assert drops == bar_lens[:-1]

    bar_color = infos[0]["bar_color"]
    bg_color = infos[0]["bg"]
    left = infos[0]["left"]
    sprite_crop = infos[0]["sprite_crop"]
    sprite_mask = infos[0]["sprite_mask"]

    for info in infos[1:]:
        assert info["bar_color"] == bar_color
        assert info["bg"] == bg_color
        assert info["left"] == left
        assert np.array_equal(info["sprite_mask"], sprite_mask)
        assert np.array_equal(info["sprite_crop"][info["sprite_mask"]], sprite_crop[sprite_mask])

    next_bar_len = int(bar_lens[-1] + delta)
    next_top = int(tops[-1] + bar_lens[-1])

    out = np.full((g.shape[0], panel_w), bg_color, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    out[1, 1 : 1 + next_bar_len] = bar_color

    sprite_h, sprite_w = sprite_crop.shape
    r0 = 1 + next_top
    c0 = 1 + left
    assert r0 >= 1 and c0 >= 1
    assert r0 + sprite_h <= out.shape[0] - 1
    assert c0 + sprite_w <= out.shape[1] - 1
    target = out[r0 : r0 + sprite_h, c0 : c0 + sprite_w]
    target[sprite_mask] = sprite_crop[sprite_mask]
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return len(seen) == int(mask.sum())


@lru_cache(maxsize=None)
def _valid_masks(height: int, width: int, area: int) -> tuple[np.ndarray, ...]:
    cells = range(height * width)
    masks = []
    for combo in itertools.combinations(cells, area):
        mask = np.zeros((height, width), dtype=bool)
        for idx in combo:
            mask[idx // width, idx % width] = True
        if not _is_connected(mask):
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        masks.append(mask)
    return tuple(masks)


def _candidate_shapes(max_width: int) -> list[np.ndarray]:
    shapes = []
    for h, w in SPRITE_BBOX_OPTIONS:
        if w > max_width:
            continue
        for area in range(SPRITE_AREA_RANGE[0], SPRITE_AREA_RANGE[1] + 1):
            if area > h * w:
                continue
            shapes.extend(_valid_masks(h, w, area))
    return shapes


def _build_shared_grid(
    interior_h: int,
    interior_w: int,
    bar_lengths: list[int],
    sprite_mask: np.ndarray,
    sprite_left: int,
    sprite_tops: list[int],
) -> np.ndarray:
    panel_h = interior_h + 2
    panel_w = interior_w + 2
    panel_count = len(bar_lengths)
    total_w = panel_count * (panel_w - 1) + 1
    grid = np.full((panel_h, total_w), GREY, dtype=int)

    for idx, (bar_len, top) in enumerate(zip(bar_lengths, sprite_tops)):
        c0 = idx * (panel_w - 1)
        grid[1:-1, c0 + 1 : c0 + 1 + interior_w] = BLACK
        grid[1, c0 + 1 : c0 + 1 + bar_len] = YELLOW
        sr0 = 1 + top
        sc0 = c0 + 1 + sprite_left
        patch = grid[sr0 : sr0 + sprite_mask.shape[0], sc0 : sc0 + sprite_mask.shape[1]]
        patch[sprite_mask] = RED

    return grid


def _build_output_panel(
    interior_h: int,
    interior_w: int,
    next_bar_len: int,
    sprite_mask: np.ndarray,
    sprite_left: int,
    next_top: int,
) -> np.ndarray:
    out = np.full((interior_h + 2, interior_w + 2), BLACK, dtype=int)
    out[0, :] = GREY
    out[-1, :] = GREY
    out[:, 0] = GREY
    out[:, -1] = GREY
    out[1, 1 : 1 + next_bar_len] = YELLOW
    patch = out[
        1 + next_top : 1 + next_top + sprite_mask.shape[0],
        1 + sprite_left : 1 + sprite_left + sprite_mask.shape[1],
    ]
    patch[sprite_mask] = RED
    return out


def _valid_length_choices(panel_count: int, growth: int) -> list[tuple[int, int]]:
    choices = []
    for interior_w in range(INTERIOR_WIDTH_RANGE[0], INTERIOR_WIDTH_RANGE[1] + 1):
        for start_len in range(1, 6):
            lengths = [start_len + i * growth for i in range(panel_count)]
            total_drop = int(sum(lengths))
            if max(lengths) > 5:
                continue
            if start_len + panel_count * growth > interior_w:
                continue
            if total_drop + 3 > INTERIOR_HEIGHT_RANGE[1]:
                continue
            choices.append((interior_w, start_len))
    return choices


def _sample_case() -> tuple[np.ndarray, np.ndarray]:
    pair = _RNG.integers(0, 3)
    if int(pair) == 0:
        panel_count, growth = 3, 1
    elif int(pair) == 1:
        panel_count, growth = 3, 2
    else:
        panel_count, growth = 4, 1

    length_choices = _valid_length_choices(panel_count, growth)
    interior_w, start_len = length_choices[int(_RNG.integers(len(length_choices)))]
    bar_lengths = [start_len + i * growth for i in range(panel_count)]
    total_drop = int(sum(bar_lengths))

    shapes = _candidate_shapes(interior_w)
    _RNG.shuffle(shapes)

    for sprite_mask in shapes:
        sprite_h, sprite_w = sprite_mask.shape
        min_h = max(INTERIOR_HEIGHT_RANGE[0], sprite_h + total_drop + 1)
        if min_h > INTERIOR_HEIGHT_RANGE[1]:
            continue
        interior_h = int(_RNG.integers(min_h, INTERIOR_HEIGHT_RANGE[1] + 1))
        max_top0 = interior_h - sprite_h - total_drop
        assert max_top0 >= 1
        top0 = int(_RNG.integers(1, max_top0 + 1))
        sprite_tops = [top0]
        for length in bar_lengths[:-1]:
            sprite_tops.append(sprite_tops[-1] + length)
        next_top = sprite_tops[-1] + bar_lengths[-1]
        sprite_left = int(_RNG.integers(0, interior_w - sprite_w + 1))

        inp = _build_shared_grid(
            interior_h=interior_h,
            interior_w=interior_w,
            bar_lengths=bar_lengths,
            sprite_mask=sprite_mask,
            sprite_left=sprite_left,
            sprite_tops=sprite_tops,
        )
        out = _build_output_panel(
            interior_h=interior_h,
            interior_w=interior_w,
            next_bar_len=bar_lengths[-1] + growth,
            sprite_mask=sprite_mask,
            sprite_left=sprite_left,
            next_top=next_top,
        )
        return inp, out

    raise RuntimeError("failed to sample a valid case")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, expected = _sample_case()
        actual = main(inp)
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
