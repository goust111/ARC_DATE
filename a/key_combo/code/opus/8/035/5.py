from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SIDE_SPECS = {
    RED: ("top", "h"),
    BLUE: ("right", "v"),
    GREEN: ("bottom", "h"),
    YELLOW: ("left", "v"),
}

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_RETRIES = 400
PLACE_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = list(map(tuple, np.argwhere(mask)))
    assert coords
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _segment_info(grid: np.ndarray, color: int) -> dict:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    assert coords, f"missing side color {color}"
    name, expected_orientation = SIDE_SPECS[color]
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})

    if expected_orientation == "h":
        assert len(rows) == 1, f"{name} side must be horizontal"
        assert cols == list(range(cols[0], cols[-1] + 1)), f"{name} side must be contiguous"
        return {
            "name": name,
            "orientation": "h",
            "row": rows[0],
            "c0": cols[0],
            "c1": cols[-1],
            "length": len(coords),
            "color": color,
        }

    assert len(cols) == 1, f"{name} side must be vertical"
    assert rows == list(range(rows[0], rows[-1] + 1)), f"{name} side must be contiguous"
    return {
        "name": name,
        "orientation": "v",
        "col": cols[0],
        "r0": rows[0],
        "r1": rows[-1],
        "length": len(coords),
        "color": color,
    }


def _rank_color_map(grid: np.ndarray) -> dict[int, int]:
    infos = [_segment_info(grid, color) for color in (RED, BLUE, GREEN, YELLOW)]
    lengths = [info["length"] for info in infos]
    assert len(set(lengths)) == 4, "side lengths must be pairwise distinct"
    infos.sort(key=lambda item: item["length"])
    return {rank + 1: int(info["color"]) for rank, info in enumerate(infos)}


def _gray_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        comps.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "holes": _count_holes(mask),
            }
        )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray sprite by querying the side-length-ranked palette with its hole count.

    Input structure:
    - The background is black.
    - Four colored one-pixel-thick side segments act as an open frame: top red, right blue, bottom green,
      and left yellow. Their endpoints do not meet, their corners remain black, and the four segment lengths
      are all different.
    - Inside the open frame sit several disconnected gray 4-connected sprites, each possibly enclosing
      one or more black holes.

    Rule:
    - Measure the pixel length of the four colored side segments and rank them from shortest to longest.
    - This ranking defines a dynamic palette: rank 1, rank 2, rank 3, rank 4 map to the colors of those
      side segments in ascending-length order.
    - For each gray sprite, count how many enclosed black 4-connected hole regions it has inside its own
      bounding box. Recolor the whole sprite with the color at rank (hole_count + 1).
    - Keep the black holes black, and leave the side segments and all other background pixels unchanged.

    Key insight:
    - Hole counts do not map to fixed colors directly; they index a palette that must first be decoded
      from the geometry of the four colored side segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rank_to_color = _rank_color_map(g)
    out = g.copy()
    for comp in _gray_components(g):
        holes = int(comp["holes"])
        assert 0 <= holes <= 3, "gray sprite hole count must be in 0..3"
        new_color = rank_to_color[holes + 1]
        for r, c in comp["coords"]:
            out[r, c] = new_color
    return out


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for _ in range(random.randint(0, 3)):
        out = np.rot90(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _random_zero_hole_sprite() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        canvas = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        canvas[start] = True
        cells = [start]
        target = random.randint(5, min(10, h * w - 1))
        while len(cells) < target:
            r, c = random.choice(cells)
            candidates = []
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not canvas[nr, nc]:
                    candidates.append((nr, nc))
            if not candidates:
                if len(cells) == canvas.sum():
                    frontier = []
                    for rr, cc in cells:
                        for dr, dc in NEIGHBORS4:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not canvas[nr, nc]:
                                frontier.append((nr, nc))
                    if not frontier:
                        break
                    nr, nc = random.choice(frontier)
                else:
                    break
            else:
                nr, nc = random.choice(candidates)
            canvas[nr, nc] = True
            cells.append((nr, nc))
        sprite = _crop_mask(canvas)
        if _count_holes(sprite) == 0 and 5 <= int(sprite.sum()) <= 10:
            return _transform_mask(sprite)
    raise RuntimeError("failed to generate 0-hole sprite")


def _base_holey_sprite(hole_count: int) -> np.ndarray:
    width = 2 * hole_count + 1
    base = np.ones((3, width), dtype=bool)
    base[1, 1::2] = False
    return base


def _augment_sprite(mask: np.ndarray, hole_count: int) -> np.ndarray:
    canvas = np.pad(mask.astype(bool), 2, constant_values=False)
    max_extra = random.randint(1, 4)
    for _ in range(max_extra):
        comp_cells = [tuple(map(int, rc)) for rc in np.argwhere(canvas)]
        candidates: list[tuple[int, int]] = []
        for r, c in comp_cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and not canvas[nr, nc]:
                    candidates.append((nr, nc))
        random.shuffle(candidates)
        placed = False
        for nr, nc in candidates:
            trial = canvas.copy()
            trial[nr, nc] = True
            cropped = _crop_mask(trial)
            if _count_holes(cropped) == hole_count:
                canvas = trial
                placed = True
                break
        if not placed:
            break
    return _crop_mask(canvas)


def _make_sprite(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        return _random_zero_hole_sprite()
    sprite = _base_holey_sprite(hole_count)
    sprite = _augment_sprite(sprite, hole_count)
    sprite = _transform_mask(sprite)
    assert _count_holes(sprite) == hole_count
    return sprite


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _paste_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = shape.shape
    patch = grid[top : top + h, left : left + w]
    patch[shape] = color


def _place_sprite(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: np.ndarray,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
) -> bool:
    sh, sw = shape.shape
    positions = [
        (r, c)
        for r in range(row_lo, row_hi - sh + 2)
        for c in range(col_lo, col_hi - sw + 2)
    ]
    random.shuffle(positions)
    for r, c in positions[:PLACE_RETRIES]:
        window = blocked[r : r + sh, c : c + sw]
        if np.any(window & shape):
            continue
        _paste_shape(grid, shape, r, c, GREY)
        footprint = np.zeros_like(grid, dtype=bool)
        footprint[r : r + sh, c : c + sw] = shape
        blocked |= _dilate8(footprint)
        return True
    return False


def _segment_layout(h: int, w: int) -> dict | None:
    top_r = random.randint(1, 3)
    bottom_r = random.randint(h - 4, h - 2)
    left_c = random.randint(1, 3)
    right_c = random.randint(w - 4, w - 2)

    inner_h = bottom_r - top_r - 1
    inner_w = right_c - left_c - 1
    if inner_h < 11 or inner_w < 11:
        return None

    lengths = random.sample(list(range(4, 10)), 4)
    top_len, right_len, bottom_len, left_len = random.sample(lengths, 4)
    if top_len > inner_w or bottom_len > inner_w or right_len > inner_h or left_len > inner_h:
        return None

    top_c0 = random.randint(left_c + 1, right_c - top_len)
    bottom_c0 = random.randint(left_c + 1, right_c - bottom_len)
    left_r0 = random.randint(top_r + 1, bottom_r - left_len)
    right_r0 = random.randint(top_r + 1, bottom_r - right_len)

    return {
        "top": {"row": top_r, "c0": top_c0, "length": top_len, "color": RED},
        "right": {"col": right_c, "r0": right_r0, "length": right_len, "color": BLUE},
        "bottom": {"row": bottom_r, "c0": bottom_c0, "length": bottom_len, "color": GREEN},
        "left": {"col": left_c, "r0": left_r0, "length": left_len, "color": YELLOW},
        "bounds": (top_r, left_c, bottom_r, right_c),
    }


def _draw_segments(grid: np.ndarray, layout: dict) -> np.ndarray:
    occupied = np.zeros_like(grid, dtype=bool)
    top = layout["top"]
    grid[top["row"], top["c0"] : top["c0"] + top["length"]] = top["color"]
    occupied[top["row"], top["c0"] : top["c0"] + top["length"]] = True

    bottom = layout["bottom"]
    grid[bottom["row"], bottom["c0"] : bottom["c0"] + bottom["length"]] = bottom["color"]
    occupied[bottom["row"], bottom["c0"] : bottom["c0"] + bottom["length"]] = True

    left = layout["left"]
    grid[left["r0"] : left["r0"] + left["length"], left["col"]] = left["color"]
    occupied[left["r0"] : left["r0"] + left["length"], left["col"]] = True

    right = layout["right"]
    grid[right["r0"] : right["r0"] + right["length"], right["col"]] = right["color"]
    occupied[right["r0"] : right["r0"] + right["length"], right["col"]] = True

    return occupied


def _analyze(grid: np.ndarray) -> dict:
    rank_to_color = _rank_color_map(grid)
    sprites = _gray_components(grid)
    return {"rank_to_color": rank_to_color, "sprites": sprites}


def _validate_semantics(grid: np.ndarray) -> None:
    analysis = _analyze(grid)
    sprites = analysis["sprites"]
    holes = sorted(int(sprite["holes"]) for sprite in sprites)
    assert holes == [0, 1, 2, 3], f"expected hole counts [0,1,2,3], got {holes}"

    top_info = _segment_info(grid, RED)
    bottom_info = _segment_info(grid, GREEN)
    left_info = _segment_info(grid, YELLOW)
    right_info = _segment_info(grid, BLUE)
    top_r = top_info["row"]
    bottom_r = bottom_info["row"]
    left_c = left_info["col"]
    right_c = right_info["col"]

    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        assert top_r < r0 <= r1 < bottom_r
        assert left_c < c0 <= c1 < right_c

    out = main(grid)
    non_grey = grid != GREY
    assert np.array_equal(out[non_grey], grid[non_grey]), "non-grey pixels must stay unchanged"
    for sprite in sprites:
        color = analysis["rank_to_color"][sprite["holes"] + 1]
        for r, c in sprite["coords"]:
            assert int(out[r, c]) == int(color)
        mask = sprite["mask"]
        hole_coords = np.argwhere(~mask)
        r0, c0, _, _ = sprite["bbox"]
        for rr, cc in hole_coords:
            gr = r0 + int(rr)
            gc = c0 + int(cc)
            if int(grid[gr, gc]) == BG:
                assert int(out[gr, gc]) == BG


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(19, 26)
    w = random.randint(19, 26)
    grid = np.zeros((h, w), dtype=int)

    layout = _segment_layout(h, w)
    if layout is None:
        return None

    occupied = _draw_segments(grid, layout)
    blocked = _dilate8(occupied)

    top_r, left_c, bottom_r, right_c = layout["bounds"]
    row_lo = top_r + 2
    row_hi = bottom_r - 2
    col_lo = left_c + 2
    col_hi = right_c - 2
    if row_hi - row_lo < 6 or col_hi - col_lo < 8:
        return None

    sprites = [_make_sprite(hole_count) for hole_count in range(4)]
    sprites.sort(key=lambda shape: shape.shape[0] * shape.shape[1], reverse=True)
    for shape in sprites:
        if not _place_sprite(grid, blocked, shape, row_lo, row_hi, col_lo, col_hi):
            return None

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
