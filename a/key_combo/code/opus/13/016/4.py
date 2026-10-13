from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5

LEGEND_MARKER_ORDER = [2, 7, 8, 9]  # left, up, right, down
DIR_TO_VEC = {
    "left": (0, -1),
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
}
DIR_TO_MARKER = dict(zip(("left", "up", "right", "down"), LEGEND_MARKER_ORDER))
SPRITE_COLOR_POOL = [1, 3, 4, 6]

LEGEND_HEIGHT = 6
LEGEND_SPRITE_SIZE = 3
LEGEND_ENTRY_STRIDE = 4
LEGEND_WIDTH = 4 * LEGEND_SPRITE_SIZE + 3

GRID_H_RANGE = (17, 22)
GRID_W_RANGE = (25, 29)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _solid_separator_row(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(rows) == 1, "expected exactly one solid grey separator row"
    return int(rows[0])


def _legend_mapping(g: np.ndarray, sep: int) -> dict[int, np.ndarray]:
    top = g[:sep]
    comps = _components((top != BLACK) & (top != GREY))
    sprites = []
    markers = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        area = int(len(comp))
        if area == 1:
            markers.append((int(comp[0, 0]), int(comp[0, 1]), int(top[comp[0, 0], comp[0, 1]])))
        else:
            assert area == 9 and (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "legend sprites must be 3x3"
            sprites.append((c0, top[r0 : r1 + 1, c0 : c1 + 1].copy()))

    assert len(sprites) == 4 and len(markers) == 4, "expected four legend sprites and four legend markers"
    sprites.sort(key=lambda item: item[0])
    markers.sort(key=lambda item: item[1])

    mapping: dict[int, np.ndarray] = {}
    for (_, sprite), (mr, mc, color) in zip(sprites, markers):
        assert mr == sep - 1, "legend marker must sit directly above the separator"
        mapping[int(color)] = sprite
    return mapping


def _find_cavity(local: np.ndarray) -> tuple[int, int]:
    zero_comps = _components(local == BLACK)
    candidates = []
    for comp in zero_comps:
        r0, c0, r1, c1 = _bbox(comp)
        if (
            len(comp) == 9
            and (r1 - r0 + 1, c1 - c0 + 1) == (3, 3)
            and r0 > 0
            and c0 > 0
            and r1 < local.shape[0] - 1
            and c1 < local.shape[1] - 1
        ):
            candidates.append((r0, c0))
    assert len(candidates) == 1, "each structure must contain exactly one enclosed 3x3 cavity"
    return candidates[0]


def _first_hit(g: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int, int] | None:
    h, w = g.shape
    dr, dc = direction
    r, c = start
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        value = int(g[r, c])
        if value not in {BLACK, GREY}:
            return r, c, value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each internal slot with the legend sprite selected by the ray-visible direction marker.

    Input structure:
    - The background is black, with a top legend area separated from the main field by one solid grey row.
    - The legend contains four 3x3 sprites, each paired with a singleton marker pixel directly above the separator.
    - Below the separator are 2-3 grey closed structures, each with one enclosed 3x3 black cavity and one external colored marker.

    Rule:
    - Read the legend as a mapping from marker color to the 3x3 sprite above it.
    - For each grey structure, locate the enclosed 3x3 cavity and cast rays from its center in the four cardinal directions.
    - The unique first non-black, non-grey pixel hit by those rays selects a legend sprite; stamp that sprite into the cavity and erase the external marker to black.

    Key insight:
    - The external marker does not specify the fill directly; it is only a lookup key, and the fill comes from the legend entry associated with that marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "unexpected ARC color id"

    sep = _solid_separator_row(g)
    mapping = _legend_mapping(g, sep)
    out = g.copy()

    lower = g[sep + 1 :]
    structures = _components(lower == GREY)
    assert 2 <= len(structures) <= 3, "expected 2-3 grey structures below the separator"

    for comp in structures:
        lr0, lc0, lr1, lc1 = _bbox(comp)
        gr0 = lr0 + sep + 1
        gc0 = lc0
        gr1 = lr1 + sep + 1
        gc1 = lc1
        local = g[gr0 : gr1 + 1, gc0 : gc1 + 1]
        cav_r0, cav_c0 = _find_cavity(local)
        slot_r0 = gr0 + cav_r0
        slot_c0 = gc0 + cav_c0
        center = (slot_r0 + 1, slot_c0 + 1)

        hits = []
        for direction in DIR_TO_VEC.values():
            hit = _first_hit(g, center, direction)
            if hit is not None:
                hits.append(hit)

        assert len(hits) == 1, "each structure must see exactly one colored marker"
        hit_r, hit_c, hit_color = hits[0]
        assert hit_r > sep, "the hit marker must lie in the main field, not in the legend"
        assert hit_color in mapping, "ray marker color must be present in the legend"

        out[slot_r0 : slot_r0 + 3, slot_c0 : slot_c0 + 3] = mapping[hit_color]
        out[hit_r, hit_c] = BLACK

    return out


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    padding: int = 1,
) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _make_structure_patch() -> tuple[np.ndarray, tuple[int, int]]:
    top_ext = random.randint(0, 1)
    bottom_ext = random.randint(0, 1)
    left_ext = random.randint(0, 1)
    right_ext = random.randint(0, 1)

    h = 5 + top_ext + bottom_ext
    w = 5 + left_ext + right_ext
    patch = np.zeros((h, w), dtype=int)

    rr = top_ext
    cc = left_ext
    patch[rr, cc : cc + 5] = GREY
    patch[rr + 4, cc : cc + 5] = GREY
    patch[rr : rr + 5, cc] = GREY
    patch[rr : rr + 5, cc + 4] = GREY

    if top_ext:
        c0 = cc + random.randint(0, 2)
        c1 = min(cc + 4, c0 + random.randint(1, 2))
        patch[0, c0 : c1 + 1] = GREY
    if bottom_ext:
        c0 = cc + random.randint(0, 2)
        c1 = min(cc + 4, c0 + random.randint(1, 2))
        patch[h - 1, c0 : c1 + 1] = GREY
    if left_ext:
        r0 = rr + random.randint(0, 2)
        r1 = min(rr + 4, r0 + random.randint(1, 2))
        patch[r0 : r1 + 1, 0] = GREY
    if right_ext:
        r0 = rr + random.randint(0, 2)
        r1 = min(rr + 4, r0 + random.randint(1, 2))
        patch[r0 : r1 + 1, w - 1] = GREY

    return patch, (top_ext + 2, left_ext + 2)


def _place_legend(g: np.ndarray, legend_left: int, sprite_colors: list[int]) -> None:
    for i, sprite_color in enumerate(sprite_colors):
        c0 = legend_left + i * LEGEND_ENTRY_STRIDE
        g[0:3, c0 : c0 + 3] = int(sprite_color)
        g[4, c0 + 1] = LEGEND_MARKER_ORDER[i]
    g[LEGEND_HEIGHT - 1, :] = GREY


def _choose_marker_position(
    bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    direction: str,
    h: int,
    w: int,
    sep: int,
    main_left: int,
    forbidden_rows: set[int],
    forbidden_cols: set[int],
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    cr, cc = center

    if direction == "up":
        candidates = [(r, cc) for r in range(sep + 1, r0 - 1) if r not in forbidden_rows]
    elif direction == "down":
        candidates = [(r, cc) for r in range(r1 + 2, h) if r not in forbidden_rows]
    elif direction == "left":
        candidates = [(cr, c) for c in range(main_left, c0 - 1) if c not in forbidden_cols]
    else:
        candidates = [(cr, c) for c in range(c1 + 2, w) if c not in forbidden_cols]

    if not candidates:
        return None
    return random.choice(candidates)


def _ray_lines_clear(
    existing_bboxes: list[tuple[int, int, int, int]],
    existing_centers: list[tuple[int, int]],
    center: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> bool:
    cr, cc = center
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in existing_bboxes:
        if or0 <= cr <= or1 or oc0 <= cc <= oc1:
            return False
    for er, ec in existing_centers:
        if r0 <= er <= r1 or c0 <= ec <= c1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    g = np.zeros((h, w), dtype=int)

    legend_left = random.randint(0, 2)
    main_left = legend_left + LEGEND_WIDTH + 1
    if main_left + 5 >= w:
        return None

    sprite_colors = random.sample(SPRITE_COLOR_POOL, 4)
    _place_legend(g, legend_left, sprite_colors)

    count_options = [2]
    if h >= 19 and w >= 27:
        count_options.append(3)
    structure_count = random.choice(count_options)
    directions = random.sample(list(DIR_TO_VEC), structure_count)

    bboxes: list[tuple[int, int, int, int]] = []
    centers: list[tuple[int, int]] = []
    forbidden_rows: set[int] = set()
    forbidden_cols: set[int] = set()

    for direction in directions:
        placed = False
        marker_color = DIR_TO_MARKER[direction]
        for _ in range(200):
            patch, center_offset = _make_structure_patch()
            ph, pw = patch.shape

            min_cr = LEGEND_HEIGHT + center_offset[0]
            max_cr = h - 1 - (ph - 1 - center_offset[0])
            min_cc = main_left + center_offset[1]
            max_cc = w - 1 - (pw - 1 - center_offset[1])
            if min_cr > max_cr or min_cc > max_cc:
                continue

            center = (random.randint(min_cr, max_cr), random.randint(min_cc, max_cc))
            if center[0] in forbidden_rows or center[1] in forbidden_cols:
                continue

            top = center[0] - center_offset[0]
            left = center[1] - center_offset[1]
            bbox = (top, left, top + ph - 1, left + pw - 1)

            if any(_boxes_too_close(bbox, other, padding=1) for other in bboxes):
                continue
            if not _ray_lines_clear(bboxes, centers, center, bbox):
                continue

            marker_pos = _choose_marker_position(
                bbox, center, direction, h, w, LEGEND_HEIGHT - 1, main_left, forbidden_rows, forbidden_cols
            )
            if marker_pos is None:
                continue

            hit_r, hit_c = marker_pos
            if g[hit_r, hit_c] != BLACK:
                continue

            rr, cc = np.where(patch == GREY)
            global_rr = rr + top
            global_cc = cc + left
            if np.any(g[global_rr, global_cc] != BLACK):
                continue

            g[global_rr, global_cc] = GREY
            g[hit_r, hit_c] = marker_color

            bboxes.append(bbox)
            centers.append(center)
            forbidden_rows.add(center[0])
            forbidden_cols.add(center[1])
            forbidden_rows.add(hit_r)
            forbidden_cols.add(hit_c)
            placed = True
            break

        if not placed:
            return None

    try:
        out = main(g)
    except Exception:
        return None

    if np.array_equal(out, g):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-legend cavity puzzle")
