from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_GEN_ATTEMPTS = 400
TOP_HEIGHT_CHOICES = (2, 3)
WIDTH_RANGE = (16, 24)
MAIN_HEIGHT_RANGE = (11, 18)
PAIR_COUNT_CHOICES = (3, 4)
ROOM_COUNT_CHOICES = (2, 3, 4)
ROOM_SIDE_RANGE = (5, 8)  # outer wall size, inclusive
ROOM_GAP = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _full_grey_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]


def _parse_mapping(top: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for row in top:
        c = 0
        while c < row.shape[0]:
            if int(row[c]) == BLACK:
                c += 1
                continue
            start = c
            while c < row.shape[0] and int(row[c]) != BLACK:
                c += 1
            run = row[start:c]
            assert len(run) == 2, "dictionary entries must be 2-pixel horizontal pairs"
            src = int(run[0])
            dst = int(run[1])
            assert src != GREY and dst != GREY and src != dst
            assert src not in mapping, "source colors must be unique"
            mapping[src] = dst
    assert len(mapping) >= 2, "expected at least two mapping pairs"
    return mapping


def _transform_bottom(bottom: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    h, w = bottom.shape
    non_wall = bottom != GREY
    comps = _components(non_wall)
    assert comps, "expected non-wall space"

    out = bottom.copy()
    enclosed_count = 0
    for coords in comps:
        touches_border = bool(
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        colors = {
            int(bottom[r, c])
            for r, c in coords
            if int(bottom[r, c]) not in {BLACK, GREY}
        }
        if touches_border:
            assert len(colors) == 0, "exterior background must not contain seeds"
            continue

        enclosed_count += 1
        assert len(colors) == 1, "each enclosed region must contain exactly one seed color"
        seed_color = next(iter(colors))
        assert seed_color in mapping, "every seed color must be covered by the dictionary"
        fill_color = mapping[seed_color]
        for r, c in coords:
            if int(out[r, c]) == BLACK:
                out[r, c] = fill_color
    assert enclosed_count >= 1, "expected at least one enclosed region"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn source-to-target colors from the top dictionary and flood-fill each enclosed seed room with the mapped color.

    Input structure:
    - One full grey separator row splits the grid into a top dictionary zone and a lower wall zone.
    - In the dictionary zone, each mapping entry is a horizontal 2-pixel color pair on black background.
    - In the lower zone, grey wall pixels enclose one or more non-border regions; each enclosed region has exactly one colored seed and otherwise black interior.

    Rule:
    - Read every horizontal dictionary pair as source color on the left and target color on the right.
    - Keep the dictionary zone, separator row, and all grey wall pixels unchanged.
    - In the lower zone, treat grey as walls and split the non-grey cells into 4-connected regions.
    - For each enclosed region, find its unique seed color, map that seed color through the dictionary, and repaint only the black cells of that region with the mapped color while leaving the seed unchanged; the border-connected exterior stays black.

    Key insight:
    - The fill color is not the seed color itself; the seed only selects a color through the learned top-row dictionary, and the walls are needed to bound where that mapped flood-fill may spread.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    full_grey = _full_grey_rows(g)
    assert len(full_grey) == 1, "expected exactly one full grey separator row"
    sep_r = full_grey[0]
    assert 1 <= sep_r <= g.shape[0] - 2, "separator must split top and bottom zones"

    top = g[:sep_r]
    bottom = g[sep_r + 1 :]
    mapping = _parse_mapping(top)

    out = g.copy()
    out[sep_r + 1 :] = _transform_bottom(bottom, mapping)
    return out


def _room_boxes_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _random_derangement(colors: list[int]) -> list[int]:
    if len(colors) == 2:
        return [colors[1], colors[0]]
    perm = colors[:]
    while True:
        random.shuffle(perm)
        if all(a != b for a, b in zip(colors, perm)):
            return perm[:]


def _sample_mapping(pair_count: int) -> dict[int, int]:
    src_colors = random.sample(PALETTE, pair_count)
    dst_colors = _random_derangement(src_colors)
    return {src: dst for src, dst in zip(src_colors, dst_colors)}


def _dictionary_slots(width: int) -> list[int]:
    return [c for c in range(1, width - 1, 4) if c + 1 < width - 1]


def _draw_dictionary(grid: np.ndarray, mapping: dict[int, int], top_h: int) -> None:
    slots = _dictionary_slots(grid.shape[1])
    assert len(slots) >= len(mapping)
    placements = random.sample(
        [(r, c) for r in range(top_h) for c in slots],
        len(mapping),
    )
    for (src, dst), (r, c) in zip(mapping.items(), placements):
        grid[r, c] = src
        grid[r, c + 1] = dst


def _place_rooms(main_h: int, main_w: int, room_count: int) -> list[tuple[int, int, int, int]] | None:
    rooms: list[tuple[int, int, int, int]] = []
    sizes = [
        (
            random.randint(*ROOM_SIDE_RANGE),
            random.randint(*ROOM_SIDE_RANGE),
        )
        for _ in range(room_count)
    ]
    sizes.sort(reverse=True)
    for room_h, room_w in sizes:
        candidates: list[tuple[int, int, int, int]] = []
        for r0 in range(1, main_h - room_h):
            for c0 in range(1, main_w - room_w):
                box = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
                if all(_room_boxes_far(box, other, ROOM_GAP) for other in rooms):
                    candidates.append(box)
        if not candidates:
            return None
        rooms.append(random.choice(candidates))
    return rooms


def _choose_seed_sequence(sources: list[int], room_count: int) -> list[int]:
    used_count = random.randint(2, min(len(sources), room_count))
    used_sources = random.sample(sources, used_count)
    seeds = used_sources[:]
    while len(seeds) < room_count:
        seeds.append(random.choice(used_sources))
    random.shuffle(seeds)
    return seeds


def _render_rooms(main: np.ndarray, room_boxes: list[tuple[int, int, int, int]], seed_colors: list[int]) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    for box, seed_color in zip(room_boxes, seed_colors):
        r0, c0, r1, c1 = box
        main[r0 : r1 + 1, c0] = GREY
        main[r0 : r1 + 1, c1] = GREY
        main[r0, c0 : c1 + 1] = GREY
        main[r1, c0 : c1 + 1] = GREY

        sr = random.randint(r0 + 2, r1 - 2)
        sc = random.randint(c0 + 2, c1 - 2)
        main[sr, sc] = seed_color
        rooms.append({"box": box, "seed": (sr, sc), "seed_color": seed_color})
    return rooms


def _build_expected_from_layout(
    height: int,
    width: int,
    top_h: int,
    mapping: dict[int, int],
    room_specs: list[dict[str, object]],
    inp: np.ndarray,
) -> np.ndarray:
    out = inp.copy()
    for spec in room_specs:
        r0, c0, r1, c1 = spec["box"]  # type: ignore[index]
        sr, sc = spec["seed"]  # type: ignore[index]
        fill = mapping[int(spec["seed_color"])]  # type: ignore[index]
        rr0 = top_h + 1 + r0 + 1
        rr1 = top_h + 1 + r1 - 1
        cc0 = c0 + 1
        cc1 = c1 - 1
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = fill
        out[top_h + 1 + sr, sc] = int(spec["seed_color"])  # type: ignore[index]
    return out


def _semantic_validate(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        produced = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(produced, expected):
        return False

    sep_r = _full_grey_rows(inp)[0]
    mapping = _parse_mapping(inp[:sep_r])
    bottom = inp[sep_r + 1 :]
    solved_bottom = _transform_bottom(bottom, mapping)
    if not np.array_equal(produced[sep_r + 1 :], solved_bottom):
        return False

    non_wall = bottom != GREY
    comps = _components(non_wall)
    enclosed = 0
    exterior = 0
    for coords in comps:
        touches_border = bool(
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == bottom.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == bottom.shape[1] - 1)
        )
        seeds = [
            (int(r), int(c))
            for r, c in coords
            if int(bottom[r, c]) not in {BLACK, GREY}
        ]
        if touches_border:
            exterior += 1
            if seeds:
                return False
        else:
            enclosed += 1
            if len(seeds) != 1:
                return False
            sr, sc = seeds[0]
            seed_color = int(bottom[sr, sc])
            fill_color = mapping[seed_color]
            for r, c in coords:
                if (r, c) == (sr, sc):
                    if int(produced[sep_r + 1 + r, c]) != seed_color:
                        return False
                elif int(bottom[r, c]) == BLACK:
                    if int(produced[sep_r + 1 + r, c]) != fill_color:
                        return False
        if not touches_border:
            for sr, sc in seeds:
                for nr, nc in _neighbors4(sr, sc):
                    if not (0 <= nr < bottom.shape[0] and 0 <= nc < bottom.shape[1]):
                        return False
                    if int(bottom[nr, nc]) == GREY:
                        return False
    return enclosed >= 2 and exterior == 1


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*WIDTH_RANGE)
    top_h = random.choice(TOP_HEIGHT_CHOICES)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    if len(_dictionary_slots(width)) < pair_count:
        return None

    main_h = random.randint(*MAIN_HEIGHT_RANGE)
    room_count = random.choice(ROOM_COUNT_CHOICES)
    room_boxes = _place_rooms(main_h, width, room_count)
    if room_boxes is None:
        return None

    mapping = _sample_mapping(pair_count)
    seed_colors = _choose_seed_sequence(list(mapping.keys()), room_count)

    inp = np.zeros((top_h + 1 + main_h, width), dtype=int)
    inp[top_h, :] = GREY
    _draw_dictionary(inp, mapping, top_h)
    room_specs = _render_rooms(inp[top_h + 1 :], room_boxes, seed_colors)

    expected = _build_expected_from_layout(inp.shape[0], width, top_h, mapping, room_specs, inp)
    if not _semantic_validate(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
