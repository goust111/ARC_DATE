from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
CORNERS = ["tl", "tr", "br", "bl"]
CORNER_TO_K = {"tl": 0, "tr": 3, "br": 2, "bl": 1}
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(w):
        col = grid[:, c]
        if int(col[0]) == int(bg):
            continue
        if bool(np.all(col == col[0])):
            left_has_fg = c > 0 and bool(np.any(grid[:, :c] != bg))
            right_has_fg = c + 1 < w and bool(np.any(grid[:, c + 1 :] != bg))
            if left_has_fg and right_has_fg:
                candidates.append(int(c))
    assert candidates, "expected a full-height separator column"
    if DIVIDER in [int(grid[0, c]) for c in candidates]:
        for c in candidates:
            if int(grid[0, c]) == DIVIDER:
                return int(c)
    return int(candidates[0])


def _rotation_corner(marker_r: int, marker_c: int, h: int, w: int) -> str:
    corner_map = {
        (0, 0): "tl",
        (0, w - 1): "tr",
        (h - 1, w - 1): "br",
        (h - 1, 0): "bl",
    }
    assert (marker_r, marker_c) in corner_map, "marker must sit on a bbox corner"
    return corner_map[(marker_r, marker_c)]


def _paste_without_marker(
    out: np.ndarray,
    sprite: np.ndarray,
    top: int,
    left: int,
    bg: int,
    marker: int,
) -> None:
    h, w = out.shape
    for r in range(sprite.shape[0]):
        for c in range(sprite.shape[1]):
            val = int(sprite[r, c])
            if val == int(bg) or val == int(marker):
                continue
            rr = int(top + r)
            cc = int(left + c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = val


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each left sprite has exactly one differently colored marker pixel.
    The marker color matches one anchor on the right canvas, while the
    marker's bbox-corner position encodes the rotation to apply before
    pasting. The marker itself is used only for alignment and is omitted
    from the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    sep = _find_separator(g, int(bg))
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    anchors: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(right != bg):
        color = int(right[r, c])
        assert color not in anchors, "anchor colors must be unique"
        anchors[color] = (int(r), int(c))
    assert anchors, "expected at least one anchor on the canvas"

    out = np.full_like(right, int(bg))

    for comp in _components(left != bg):
        r0, c0, r1, c1 = _bbox(comp)
        sprite = left[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = [int(left[r, c]) for r, c in comp]
        counts = Counter(colors)
        assert len(counts) == 2, "sprite must use one body color and one marker color"
        marker_candidates = [color for color, count in counts.items() if int(count) == 1]
        assert len(marker_candidates) == 1, "sprite must have exactly one unique marker pixel"
        marker = int(marker_candidates[0])
        assert marker in anchors, "every marker color must have a matching anchor"

        marker_abs = next((r, c) for r, c in comp if int(left[r, c]) == int(marker))
        marker_r = int(marker_abs[0] - r0)
        marker_c = int(marker_abs[1] - c0)
        corner = _rotation_corner(int(marker_r), int(marker_c), int(sprite.shape[0]), int(sprite.shape[1]))
        rotated = np.rot90(sprite, CORNER_TO_K[corner])

        marker_pos = np.argwhere(rotated == marker)
        assert marker_pos.shape[0] == 1
        mr, mc = map(int, marker_pos[0])
        ar, ac = anchors[marker]
        _paste_without_marker(out, rotated, int(ar - mr), int(ac - mc), int(bg), int(marker))

    return out


def _corner_cell(corner: str, h: int, w: int) -> tuple[int, int]:
    return {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "br": (h - 1, w - 1),
        "bl": (h - 1, 0),
    }[corner]


def _opposite_corner(corner: str, h: int, w: int) -> tuple[int, int]:
    return {
        "tl": (h - 1, w - 1),
        "tr": (h - 1, 0),
        "br": (0, 0),
        "bl": (0, w - 1),
    }[corner]


def _random_sprite(
    body_color: int,
    marker_color: int,
    corner: str,
    max_dim: int,
    rng: random.Random,
) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(3, max_dim)
        w = rng.randint(3, max_dim)
        start = _corner_cell(corner, int(h), int(w))
        end = _opposite_corner(corner, int(h), int(w))

        dr = 1 if end[0] > start[0] else -1
        dc = 1 if end[1] > start[1] else -1
        moves = [(dr, 0)] * abs(end[0] - start[0]) + [(0, dc)] * abs(end[1] - start[1])
        rng.shuffle(moves)

        cells = {start}
        r, c = start
        for step_r, step_c in moves:
            r += step_r
            c += step_c
            cells.add((int(r), int(c)))

        area_cap = min(int(h * w), len(cells) + rng.randint(2, 6))
        target_area = rng.randint(len(cells), area_cap)
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for rr, cc in cells:
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and (int(nr), int(nc)) not in cells:
                        frontier.add((int(nr), int(nc)))
            assert frontier
            cells.add(rng.choice(sorted(frontier)))

        rows = [rr for rr, _cc in cells]
        cols = [cc for _rr, cc in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        if not any((nr, nc) in cells for nr, nc in _neighbors4(*start)):
            continue

        sprite = np.full((h, w), BG, dtype=int)
        for rr, cc in cells:
            sprite[rr, cc] = int(body_color)
        sprite[start] = int(marker_color)
        return sprite
    raise RuntimeError("failed to generate a valid sprite")


def _slot_origins(slot_size: int, slot_gap: int) -> list[tuple[int, int]]:
    step = int(slot_size + slot_gap)
    return [(0, 0), (0, step), (step, 0), (step, step)]


def _place_sprite_in_slot(
    base: np.ndarray,
    sprite: np.ndarray,
    slot_origin: tuple[int, int],
    slot_size: int,
    rng: random.Random,
) -> tuple[int, int]:
    h, w = sprite.shape
    max_r = int(slot_size - h)
    max_c = int(slot_size - w)
    top = int(slot_origin[0] + rng.randint(0, max_r))
    left = int(slot_origin[1] + rng.randint(0, max_c))
    for r in range(h):
        for c in range(w):
            val = int(sprite[r, c])
            if val != BG:
                base[top + r, left + c] = val
    return int(top), int(left)


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    n_sprites = rng.randint(2, 4)
    slot_size = rng.choice([6, 7])
    slot_gap = rng.choice([1, 2])
    region_size = 2 * slot_size + slot_gap
    left_w = region_size
    canvas_w = region_size
    height = region_size

    colors = rng.sample(PALETTE, 2 * n_sprites)
    body_colors = colors[:n_sprites]
    marker_colors = colors[n_sprites:]

    sprites: list[dict[str, object]] = []
    for idx in range(n_sprites):
        corner = rng.choice(CORNERS)
        sprite = _random_sprite(
            int(body_colors[idx]),
            int(marker_colors[idx]),
            str(corner),
            min(5, int(slot_size)),
            rng,
        )
        rotated = np.rot90(sprite, CORNER_TO_K[str(corner)])
        marker_pos = np.argwhere(rotated == int(marker_colors[idx]))
        assert marker_pos.shape[0] == 1
        sprites.append(
            {
                "body": int(body_colors[idx]),
                "marker": int(marker_colors[idx]),
                "corner": str(corner),
                "sprite": sprite,
                "rotated": rotated,
                "marker_pos": tuple(map(int, marker_pos[0])),
            }
        )

    left_slots = rng.sample(range(4), n_sprites)
    right_slots = rng.sample(range(4), n_sprites)
    origins = _slot_origins(int(slot_size), int(slot_gap))

    left = np.full((height, left_w), BG, dtype=int)
    canvas = np.full((height, canvas_w), BG, dtype=int)
    expected = np.full((height, canvas_w), BG, dtype=int)

    for spec, slot_id in zip(sprites, left_slots, strict=False):
        _place_sprite_in_slot(left, np.asarray(spec["sprite"], dtype=int), origins[int(slot_id)], int(slot_size), rng)

    anchors: dict[int, tuple[int, int]] = {}
    for spec, slot_id in zip(sprites, right_slots, strict=False):
        rotated = np.asarray(spec["rotated"], dtype=int)
        top, left_pos = _place_sprite_in_slot(expected, rotated, origins[int(slot_id)], int(slot_size), rng)
        marker = int(spec["marker"])
        mr, mc = map(int, spec["marker_pos"])
        anchor = (int(top + mr), int(left_pos + mc))
        assert int(expected[anchor]) == int(marker)
        expected[anchor] = BG
        anchors[marker] = anchor

    for marker, (r, c) in anchors.items():
        canvas[int(r), int(c)] = int(marker)

    full = np.full((height, left_w + 1 + canvas_w), BG, dtype=int)
    full[:, left_w] = DIVIDER
    full[:, :left_w] = left
    full[:, left_w + 1 :] = canvas
    return full, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        inp, expected = _build_case(rng)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
