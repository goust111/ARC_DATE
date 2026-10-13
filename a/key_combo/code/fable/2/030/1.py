from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, find_components, get_bg


SIDES = ("left", "right", "up", "down")
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 500
SHAPE_ATTEMPTS = 250


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> None:
    for r, c in np.asarray(coords, dtype=int):
        assert 0 <= int(r) < grid.shape[0] and 0 <= int(c) < grid.shape[1]
        assert int(grid[int(r), int(c)]) in (int(bg), int(color))
        grid[int(r), int(c)] = int(color)


def _singleton_side(singleton: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = map(int, singleton)
    r0, c0, r1, c1 = map(int, bbox)
    if r0 <= r <= r1 and c < c0:
        return "left"
    if r0 <= r <= r1 and c > c1:
        return "right"
    if c0 <= c <= c1 and r < r0:
        return "up"
    if c0 <= c <= c1 and r > r1:
        return "down"
    raise AssertionError("singleton must lie in exactly one outward strip of its main bbox")


def _mirror_coords(coords: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> np.ndarray:
    mirrored = np.asarray(coords, dtype=int).copy()
    r0, c0, r1, c1 = map(int, bbox)
    if side in ("left", "right"):
        mirrored[:, 1] = c0 + c1 - mirrored[:, 1]
    else:
        mirrored[:, 0] = r0 + r1 - mirrored[:, 0]
    return mirrored


def _slide_delta(coords: np.ndarray, singleton: tuple[int, int], side: str) -> int:
    r, c = map(int, singleton)
    pts = np.asarray(coords, dtype=int)
    if side == "left":
        same = pts[pts[:, 0] == r, 1]
        assert len(same) > 0
        delta = int(same.min()) - c
    elif side == "right":
        same = pts[pts[:, 0] == r, 1]
        assert len(same) > 0
        delta = c - int(same.max())
    elif side == "up":
        same = pts[pts[:, 1] == c, 0]
        assert len(same) > 0
        delta = int(same.min()) - r
    else:
        same = pts[pts[:, 1] == c, 0]
        assert len(same) > 0
        delta = r - int(same.max())
    assert delta > 0
    return int(delta)


def _shift_vector(side: str, delta: int) -> tuple[int, int]:
    if side == "left":
        return 0, -int(delta)
    if side == "right":
        return 0, int(delta)
    if side == "up":
        return -int(delta), 0
    return int(delta), 0


def _apply_color_rule(main_comp: dict, singleton: tuple[int, int] | None) -> np.ndarray:
    coords = np.asarray(main_comp["coords"], dtype=int)
    if singleton is None:
        return coords
    side = _singleton_side(singleton, main_comp["bbox"])
    mirrored = _mirror_coords(coords, main_comp["bbox"], side)
    delta = _slide_delta(mirrored, singleton, side)
    dr, dc = _shift_vector(side, delta)
    return mirrored + np.array([dr, dc], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each colored main object inside its own bbox, then slide it onto its same-color singleton target.

    Input structure:
    - The background is black.
    - Each foreground color has at most one 8-connected main component of area greater than one.
    - Some colors also have exactly one same-color singleton pixel in the top, bottom, left, or right strip outside that main component's bbox.
    - Some colors may appear only as a singleton noise pixel.

    Rule:
    - Process each non-background color independently using 8-connected same-color components.
    - A singleton on the left or right selects a left-right mirror inside the main component's own bbox; a singleton above or below selects a top-bottom mirror.
    - After mirroring, translate the main component one cell at a time toward the singleton until the first pixel lands on the singleton cell; the singleton is absorbed and disappears.
    - A main component without a singleton stays unchanged, while a singleton without a same-color main component is deleted.

    Key insight:
    - The isolated pixel is both an instruction and a target: its side relative to the main bbox chooses the mirror axis, and its location fixes how far the mirrored shape must slide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    for color in colors_present(g, exclude={bg}):
        comps = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[color],
            min_area=1,
        )
        mains = [comp for comp in comps if int(comp["area"]) > 1]
        singles = [comp for comp in comps if int(comp["area"]) == 1]
        assert len(mains) <= 1
        assert len(singles) <= 1

        if not mains:
            continue

        singleton = None if not singles else tuple(map(int, singles[0]["coords"][0]))
        result_coords = _apply_color_rule(mains[0], singleton)
        _paint_coords(out, result_coords, int(color), bg)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _random_connected_mask(min_side: int = 2, max_side: int = 5) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(min_side, max_side)
        w = random.randint(min_side, max_side)
        max_area = min(h * w - 1, 9)
        min_area = max(3, min(h * w - 2, 4))
        if min_area > max_area:
            continue
        target = random.randint(min_area, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) <= 2:
            continue
        if int(mask.sum()) == mask.size:
            continue
        return mask
    raise RuntimeError("failed to sample a connected mask")


def _axis_asymmetric_mask(axis: str) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        mask = _random_connected_mask()
        if axis == "horizontal" and not np.array_equal(mask, np.fliplr(mask)):
            return mask
        if axis == "vertical" and not np.array_equal(mask, np.flipud(mask)):
            return mask
    raise RuntimeError("failed to sample an axis-asymmetric mask")


def _coords_bounds(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _to_array(coords: list[tuple[int, int]]) -> np.ndarray:
    if not coords:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(coords, dtype=int)


def _make_transformed_item(color: int, axis: str) -> dict:
    side = random.choice(("left", "right")) if axis == "horizontal" else random.choice(("up", "down"))
    mask = _axis_asymmetric_mask(axis)
    mirrored = np.fliplr(mask) if side in ("left", "right") else np.flipud(mask)
    h, w = mask.shape

    if side in ("left", "right"):
        rows = [r for r in range(h) if bool(mirrored[r].any())]
        row = random.choice(rows)
        cols = np.flatnonzero(mirrored[row])
        edge = int(cols.min()) if side == "left" else int(cols.max())
        min_gap = edge + 2 if side == "left" else w + 1 - edge
        gap = random.randint(min_gap, min_gap + 3)
        singleton = (row, edge - gap) if side == "left" else (row, edge + gap)
        shift = (0, -gap) if side == "left" else (0, gap)
    else:
        cols = [c for c in range(w) if bool(mirrored[:, c].any())]
        col = random.choice(cols)
        rows = np.flatnonzero(mirrored[:, col])
        edge = int(rows.min()) if side == "up" else int(rows.max())
        min_gap = edge + 2 if side == "up" else h + 1 - edge
        gap = random.randint(min_gap, min_gap + 3)
        singleton = (edge - gap, col) if side == "up" else (edge + gap, col)
        shift = (-gap, 0) if side == "up" else (gap, 0)

    input_main = _mask_coords(mask)
    output_main = [(r + shift[0], c + shift[1]) for r, c in _mask_coords(mirrored)]
    input_cells = input_main + [singleton]
    assert singleton not in input_main
    return {
        "kind": "transformed",
        "axis": axis,
        "side": side,
        "color": int(color),
        "input_cells": input_cells,
        "output_cells": output_main,
        "input_main": input_main,
        "singleton": singleton,
    }


def _make_static_item(color: int) -> dict:
    mask = _random_connected_mask()
    cells = _mask_coords(mask)
    return {
        "kind": "static",
        "axis": None,
        "side": None,
        "color": int(color),
        "input_cells": cells,
        "output_cells": cells[:],
        "input_main": cells[:],
        "singleton": None,
    }


def _make_noise_item(color: int) -> dict:
    return {
        "kind": "noise",
        "axis": None,
        "side": None,
        "color": int(color),
        "input_cells": [(0, 0)],
        "output_cells": [],
        "input_main": [],
        "singleton": (0, 0),
    }


def _item_span(item: dict) -> int:
    coords = item["input_cells"] + item["output_cells"]
    r0, c0, r1, c1 = _coords_bounds(coords)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _coords_clear(coords: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        rr0 = max(0, r - PLACEMENT_PADDING)
        rr1 = min(h, r + PLACEMENT_PADDING + 1)
        cc0 = max(0, c - PLACEMENT_PADDING)
        cc1 = min(w, c + PLACEMENT_PADDING + 1)
        if bool(occupied[rr0:rr1, cc0:cc1].any()):
            return False
    return True


def _mark_coords(occupied: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        occupied[int(r), int(c)] = True


def _anchor_candidates(item: dict, height: int, width: int, inp_occ: np.ndarray, out_occ: np.ndarray) -> list[tuple[int, int]]:
    min_r = 0
    max_r = height - 1
    min_c = 0
    max_c = width - 1

    for coords in (item["input_cells"], item["output_cells"]):
        if not coords:
            continue
        r0, c0, r1, c1 = _coords_bounds(coords)
        min_r = max(min_r, -r0)
        max_r = min(max_r, height - 1 - r1)
        min_c = max(min_c, -c0)
        max_c = min(max_c, width - 1 - c1)

    if min_r > max_r or min_c > max_c:
        return []

    anchors = []
    for ar in range(min_r, max_r + 1):
        for ac in range(min_c, max_c + 1):
            abs_in = [(ar + r, ac + c) for r, c in item["input_cells"]]
            abs_out = [(ar + r, ac + c) for r, c in item["output_cells"]]
            if not _coords_clear(abs_in, inp_occ):
                continue
            if not _coords_clear(abs_out, out_occ):
                continue
            anchors.append((ar, ac))
    return anchors


def _translate_item(item: dict, anchor: tuple[int, int]) -> dict:
    ar, ac = map(int, anchor)

    def shift(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(ar + r, ac + c) for r, c in coords]

    placed = dict(item)
    placed["anchor"] = anchor
    placed["input_cells"] = shift(item["input_cells"])
    placed["output_cells"] = shift(item["output_cells"])
    placed["input_main"] = shift(item["input_main"])
    placed["singleton"] = None if item["singleton"] is None else shift([item["singleton"]])[0]
    return placed


def _paint_item_cells(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    if not coords:
        return
    arr = _to_array(coords)
    grid[arr[:, 0], arr[:, 1]] = int(color)


def _remove_singletons_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    for color in colors_present(g, exclude={bg}):
        comps = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[color],
            min_area=1,
        )
        for comp in comps:
            if int(comp["area"]) > 1:
                _paint_coords(out, np.asarray(comp["coords"], dtype=int), int(color), bg)
    return out


def _slide_without_mirror(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)
    sentinel = np.full(g.shape, -1, dtype=int)
    for color in colors_present(g, exclude={bg}):
        comps = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[color],
            min_area=1,
        )
        mains = [comp for comp in comps if int(comp["area"]) > 1]
        singles = [comp for comp in comps if int(comp["area"]) == 1]
        if not mains:
            continue
        coords = np.asarray(mains[0]["coords"], dtype=int)
        if singles:
            singleton = tuple(map(int, singles[0]["coords"][0]))
            side = _singleton_side(singleton, mains[0]["bbox"])
            delta = _slide_delta(coords, singleton, side)
            dr, dc = _shift_vector(side, delta)
            coords = coords + np.array([dr, dc], dtype=int)
        if (
            len(coords) > 0
            and (
                int(coords[:, 0].min()) < 0
                or int(coords[:, 1].min()) < 0
                or int(coords[:, 0].max()) >= g.shape[0]
                or int(coords[:, 1].max()) >= g.shape[1]
            )
        ):
            return sentinel
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if int(out[rr, cc]) not in (bg, int(color)):
                return sentinel
            out[rr, cc] = int(color)
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, placed_items: list[dict]) -> None:
    transformed = [item for item in placed_items if item["kind"] == "transformed"]
    static = [item for item in placed_items if item["kind"] == "static"]
    noise = [item for item in placed_items if item["kind"] == "noise"]

    assert any(item["axis"] == "horizontal" for item in transformed)
    assert any(item["axis"] == "vertical" for item in transformed)
    assert static
    assert noise

    got = main(inp)
    assert np.array_equal(got, expected)
    assert not np.array_equal(got, _remove_singletons_only(inp))
    assert not np.array_equal(got, _slide_without_mirror(inp))

    bg = int(get_bg(inp))
    for item in transformed:
        color = int(item["color"])
        main_coords = _to_array(item["input_main"])
        rr = main_coords[:, 0]
        cc = main_coords[:, 1]
        bbox = (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max()))
        singleton = item["singleton"]
        assert singleton is not None
        side = _singleton_side(singleton, bbox)
        assert side == item["side"]
        if side == "left":
            assert singleton[1] <= bbox[1] - 2
        elif side == "right":
            assert singleton[1] >= bbox[3] + 2
        elif side == "up":
            assert singleton[0] <= bbox[0] - 2
        else:
            assert singleton[0] >= bbox[2] + 2

        comps = find_components(inp, bg=bg, connectivity=8, monochromatic=True, colors=[color], min_area=1)
        main_parts = [comp for comp in comps if int(comp["area"]) > 1]
        singles = [comp for comp in comps if int(comp["area"]) == 1]
        assert len(main_parts) == 1 and len(singles) == 1


def _build_items(colors: list[int]) -> list[dict]:
    items = [
        _make_transformed_item(colors[0], "horizontal"),
        _make_transformed_item(colors[1], "vertical"),
        _make_static_item(colors[2]),
        _make_noise_item(colors[3]),
    ]

    idx = 4
    extras = []
    if idx < len(colors) and random.random() < 0.55:
        axis = random.choice(("horizontal", "vertical"))
        extras.append(_make_transformed_item(colors[idx], axis))
        idx += 1
    if idx < len(colors) and random.random() < 0.45:
        extras.append(_make_static_item(colors[idx]))
        idx += 1
    if idx < len(colors) and random.random() < 0.45:
        extras.append(_make_noise_item(colors[idx]))
        idx += 1

    items.extend(extras)
    items.sort(key=_item_span, reverse=True)
    return items


def _attempt_generation() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    n_colors = random.randint(4, 7)
    colors = random.sample(list(range(1, 10)), n_colors)
    items = _build_items(colors)

    inp_occ = np.zeros((height, width), dtype=bool)
    out_occ = np.zeros((height, width), dtype=bool)
    placed_items = []

    for item in items:
        anchors = _anchor_candidates(item, height, width, inp_occ, out_occ)
        if not anchors:
            return None
        anchor = random.choice(anchors)
        placed = _translate_item(item, anchor)
        placed_items.append(placed)
        _mark_coords(inp_occ, placed["input_cells"])
        _mark_coords(out_occ, placed["output_cells"])

    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    for item in placed_items:
        _paint_item_cells(inp, item["input_cells"], int(item["color"]))
        _paint_item_cells(expected, item["output_cells"], int(item["color"]))

    if len(np.unique(inp)) < 4:
        return None
    _validate_semantics(inp, expected, placed_items)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generation()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-and-slide ARC input")
