from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_FROM_EDGE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
GEN_ATTEMPTS = 700

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr or dc
        ]

    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[Coord] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _diagonal_step(orient: str) -> Coord:
    return (1, -1) if orient == "/" else (1, 1)


def _mirror_orientation(coords: np.ndarray) -> str:
    cells = {tuple(map(int, rc)) for rc in coords}
    assert len(cells) >= 3, "mirror segments must have at least three cells"
    sums = {r + c for r, c in cells}
    diffs = {r - c for r, c in cells}
    if len(sums) == 1:
        orient = "/"
    elif len(diffs) == 1:
        orient = "\\"
    else:
        raise AssertionError("gray mirrors must be straight 45-degree segments")

    dr, dc = _diagonal_step(orient)
    ordered = sorted(cells)
    for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
        assert (r1 - r0, c1 - c0) == (dr, dc), "mirror segment cells must be contiguous"
    return orient


def _mirror_map(g: np.ndarray) -> dict[Coord, str]:
    mirrors: dict[Coord, str] = {}
    for comp in _components(g == GREY, connectivity=8):
        orient = _mirror_orientation(comp)
        for r, c in comp:
            pos = (int(r), int(c))
            assert pos not in mirrors, "mirror cells must be unique"
            mirrors[pos] = orient
    assert mirrors, "expected at least one diagonal mirror"
    return mirrors


def _reflect(direction: Coord, orient: str) -> Coord:
    dr, dc = direction
    if orient == "/":
        return -dc, -dr
    if orient == "\\":
        return dc, dr
    raise ValueError(orient)


def _diag_flip(patch: np.ndarray, orient: str) -> np.ndarray:
    if orient == "\\":
        return patch.T.copy()
    if orient == "/":
        return np.fliplr(np.flipud(patch)).T.copy()
    raise ValueError(orient)


def _swap_colors(patch: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    out = patch.copy()
    ma = patch == a
    mb = patch == b
    out[ma] = b
    out[mb] = a
    return out


def _apply_sequence(patch: np.ndarray, colors: tuple[int, int], sequence: list[str], *, swap: bool) -> np.ndarray:
    out = patch.copy()
    for orient in sequence:
        out = _diag_flip(out, orient)
        if swap:
            out = _swap_colors(out, colors)
    return out


def _touching_edge(b: BBox, shape: tuple[int, int]) -> tuple[str, Coord]:
    h, w = shape
    r0, c0, r1, c1 = b
    touches = []
    if r0 == 0:
        touches.append("top")
    if r1 == h - 1:
        touches.append("bottom")
    if c0 == 0:
        touches.append("left")
    if c1 == w - 1:
        touches.append("right")
    assert len(touches) == 1, "the sprite bbox must touch exactly one grid edge"
    side = touches[0]
    return side, DIR_FROM_EDGE[side]


def _find_sprite(g: np.ndarray) -> dict:
    sprite_mask = (g != BG) & (g != GREY)
    assert np.any(sprite_mask), "expected one colored sprite"
    comps = _components(sprite_mask, connectivity=4)
    assert len(comps) == 1, "all non-gray colored cells must form one sprite"

    colors = tuple(map(int, np.unique(g[sprite_mask]).tolist()))
    assert len(colors) == 2, "the sprite must use exactly two colors"

    b = _bbox(sprite_mask)
    r0, c0, r1, c1 = b
    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert not np.any(patch == GREY), "mirrors must not lie inside the sprite bbox"
    ph, pw = patch.shape
    assert ph % 2 == 1 and pw % 2 == 1, "sprite bbox must have a cell center"
    assert int(patch[ph // 2, pw // 2]) in colors, "sprite center must be visible"

    side, direction = _touching_edge(b, g.shape)
    center = (r0 + ph // 2, c0 + pw // 2)
    return {
        "bbox": b,
        "patch": patch,
        "colors": colors,
        "side": side,
        "center": center,
        "direction": direction,
    }


def _hit_is_interior(mirrors: dict[Coord, str], pos: Coord, orient: str) -> bool:
    r, c = pos
    dr, dc = _diagonal_step(orient)
    return mirrors.get((r + dr, c + dc)) == orient and mirrors.get((r - dr, c - dc)) == orient


def _trace(g: np.ndarray, sprite: dict, mirrors: dict[Coord, str]) -> dict:
    pos = sprite["center"]
    direction = sprite["direction"]
    patch = sprite["patch"].copy()
    colors = sprite["colors"]
    h, w = g.shape

    visited_states: set[tuple[Coord, Coord]] = set()
    visited_cells: set[Coord] = {pos}
    hits: list[tuple[Coord, str]] = []

    for _ in range(h * w * 4 + 10):
        state = (pos, direction)
        assert state not in visited_states, "trajectory must leave the grid instead of cycling"
        visited_states.add(state)

        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        if not _in_bounds(g.shape, nr, nc):
            assert hits, "trajectory must reflect at least once before exiting"
            return {
                "exit_cell": pos,
                "exit_dir": direction,
                "patch": patch,
                "hits": hits,
                "path": visited_cells,
            }

        pos = (nr, nc)
        assert pos not in visited_cells, "trajectory must not self-intersect"
        visited_cells.add(pos)

        if pos in mirrors:
            orient = mirrors[pos]
            assert _hit_is_interior(mirrors, pos, orient), "trajectory must not hit a mirror endpoint"
            hits.append((pos, orient))
            patch = _diag_flip(patch, orient)
            patch = _swap_colors(patch, colors)
            direction = _reflect(direction, orient)

    raise AssertionError("trajectory did not leave the grid")


def _exit_placement(shape: tuple[int, int], patch_shape: tuple[int, int], exit_cell: Coord, exit_dir: Coord) -> Coord:
    h, w = shape
    ph, pw = patch_shape
    er, ec = exit_cell
    if exit_dir == (-1, 0):
        assert er == 0, "top exit must leave from row 0"
        r0, c0 = 0, ec - pw // 2
    elif exit_dir == (1, 0):
        assert er == h - 1, "bottom exit must leave from the last row"
        r0, c0 = h - ph, ec - pw // 2
    elif exit_dir == (0, -1):
        assert ec == 0, "left exit must leave from column 0"
        r0, c0 = er - ph // 2, 0
    elif exit_dir == (0, 1):
        assert ec == w - 1, "right exit must leave from the last column"
        r0, c0 = er - ph // 2, w - pw
    else:
        raise AssertionError("exit direction must be cardinal")
    assert 0 <= r0 <= h - ph and 0 <= c0 <= w - pw, "exit sprite must fit on the canvas"
    return int(r0), int(c0)


def _paste_sprite(out: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    for rr, cc in np.argwhere(patch != BG):
        r = r0 + int(rr)
        c = c0 + int(cc)
        assert int(out[r, c]) != GREY, "exit sprite must not cover a mirror"
        out[r, c] = int(patch[rr, cc])


def main(grid: np.ndarray) -> np.ndarray:
    """Bounce an edge sprite through diagonal mirrors and draw its transformed exit copy.

    Input structure:
    - The background is black. Exactly one two-color sprite has an odd-sized
      bounding box touching one grid edge; its launch direction is perpendicular
      to that edge and points inward.
    - The remaining foreground cells are gray 45-degree mirror segments, each a
      contiguous one-cell-wide "/" or "\\" diagonal, away from the sprite.
    - The ray from the sprite center hits mirror interiors in sequence and then
      leaves the grid without self-intersection.

    Rule:
    - Trace the sprite center one cell at a time in its current cardinal
      direction. On a gray mirror cell, reflect the direction by that mirror's
      diagonal orientation.
    - Each reflection also applies the same diagonal flip to the sprite patch
      and swaps the sprite's two colors.
    - Clear the original sprite cells, preserve all mirrors, and paste the
      accumulated sprite patch flush to the exit edge with its center aligned to
      the exit row or column.

    Key insight:
    - The mirrors are both route changers and object-transform instructions:
      the exit location requires following the reflected trajectory, while the
      final sprite is the ordered composition of mirror flips plus color-swap
      parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))).issubset({BG, GREY, *SPRITE_COLORS}), "unexpected ARC color"

    sprite = _find_sprite(g)
    mirrors = _mirror_map(g)
    traced = _trace(g, sprite, mirrors)
    r0, c0 = _exit_placement(g.shape, traced["patch"].shape, traced["exit_cell"], traced["exit_dir"])

    out = g.copy()
    out[(g != BG) & (g != GREY)] = BG
    _paste_sprite(out, traced["patch"], r0, c0)
    return out


def _cell_in_bbox(cell: Coord, b: BBox) -> bool:
    r, c = cell
    r0, c0, r1, c1 = b
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sprite_bbox(side: str, center: Coord, size: int, shape: tuple[int, int]) -> BBox:
    h, w = shape
    r, c = center
    half = size // 2
    if side == "top":
        return 0, c - half, size - 1, c + half
    if side == "bottom":
        return h - size, c - half, h - 1, c + half
    if side == "left":
        return r - half, 0, r + half, size - 1
    if side == "right":
        return r - half, w - size, r + half, w - 1
    raise ValueError(side)


def _cells_on_ray(start: Coord, direction: Coord, shape: tuple[int, int]) -> list[Coord]:
    cells: list[Coord] = []
    r, c = start
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _segment_cells(hit: Coord, orient: str, length: int, offset: int) -> list[Coord]:
    dr, dc = _diagonal_step(orient)
    hr, hc = hit
    return [(hr + (i - offset) * dr, hc + (i - offset) * dc) for i in range(length)]


def _valid_segment_options(
    hit: Coord,
    orient: str,
    shape: tuple[int, int],
    sprite_box: BBox,
    path_cells: set[Coord],
    mirror_cells: set[Coord],
) -> list[list[Coord]]:
    options: list[list[Coord]] = []
    for length in (3, 5):
        for offset in range(1, length - 1):
            cells = _segment_cells(hit, orient, length, offset)
            if not all(_in_bounds(shape, r, c) for r, c in cells):
                continue
            if any(_cell_in_bbox(cell, sprite_box) for cell in cells):
                continue
            if any(cell in mirror_cells for cell in cells):
                continue
            if any(cell != hit and cell in path_cells for cell in cells):
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for r, c in cells for rr, cc in mirror_cells):
                continue
            options.append(cells)
    return options


def _choose_center(side: str, size: int, shape: tuple[int, int]) -> Coord:
    h, w = shape
    half = size // 2
    margin = half + 1
    if side == "top":
        return half, random.randint(margin, w - margin - 1)
    if side == "bottom":
        return h - half - 1, random.randint(margin, w - margin - 1)
    if side == "left":
        return random.randint(margin, h - margin - 1), half
    if side == "right":
        return random.randint(margin, h - margin - 1), w - half - 1
    raise ValueError(side)


def _build_plan(shape: tuple[int, int], size: int, side: str, sequence: list[str]) -> dict | None:
    for _ in range(180):
        center = _choose_center(side, size, shape)
        sprite_box = _sprite_bbox(side, center, size, shape)
        direction = DIR_FROM_EDGE[side]
        pos = center
        path_cells: set[Coord] = {pos}
        mirror_cells: set[Coord] = set()
        mirrors: list[dict] = []
        ok = True

        for orient in sequence:
            ray = _cells_on_ray(pos, direction, shape)
            choices: list[tuple[Coord, list[Coord], list[Coord]]] = []
            for i, hit in enumerate(ray):
                cells_to_hit = ray[: i + 1]
                if len(cells_to_hit) < size // 2 + 2:
                    continue
                if _cell_in_bbox(hit, sprite_box):
                    continue
                if any(cell in path_cells for cell in cells_to_hit):
                    continue
                if any(cell in mirror_cells for cell in cells_to_hit):
                    continue
                segment_options = _valid_segment_options(hit, orient, shape, sprite_box, path_cells, mirror_cells)
                for segment in segment_options:
                    choices.append((hit, cells_to_hit, segment))

            if not choices:
                ok = False
                break
            hit, cells_to_hit, segment = random.choice(choices)
            for cell in cells_to_hit:
                path_cells.add(cell)
            for cell in segment:
                mirror_cells.add(cell)
            mirrors.append({"hit": hit, "orient": orient, "cells": segment})
            pos = hit
            direction = _reflect(direction, orient)

        if not ok:
            continue

        final_ray = _cells_on_ray(pos, direction, shape)
        if not final_ray:
            continue
        if any(cell in path_cells for cell in final_ray):
            continue
        if any(cell in mirror_cells for cell in final_ray):
            continue
        exit_cell = final_ray[-1]
        try:
            out_r0, out_c0 = _exit_placement(shape, (size, size), exit_cell, direction)
        except AssertionError:
            continue
        output_box = (out_r0, out_c0, out_r0 + size - 1, out_c0 + size - 1)
        if _bbox_overlap(sprite_box, output_box):
            continue
        if any(_cell_in_bbox(cell, output_box) for cell in mirror_cells):
            continue

        return {
            "side": side,
            "center": center,
            "sprite_box": sprite_box,
            "sequence": list(sequence),
            "mirrors": mirrors,
            "exit_cell": exit_cell,
            "exit_dir": direction,
            "output_box": output_box,
        }
    return None


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask, connectivity=4)) == 1


def _random_sprite(size: int, colors: tuple[int, int], sequence: list[str]) -> np.ndarray | None:
    for _ in range(80):
        patch = np.zeros((size, size), dtype=int)
        for r in range(size):
            for c in range(size):
                if random.random() < 0.72:
                    patch[r, c] = random.choice(colors)

        mid = size // 2
        anchors = [(mid, mid), (0, mid), (size - 1, mid), (mid, 0), (mid, size - 1)]
        for i, (r, c) in enumerate(anchors):
            patch[r, c] = colors[i % 2]

        if set(map(int, np.unique(patch[patch != BG]))) != set(colors):
            continue
        if not _is_connected(patch != BG):
            continue

        transformed = _apply_sequence(patch, colors, sequence, swap=True)
        no_swap = _apply_sequence(patch, colors, sequence, swap=False)
        if np.array_equal(transformed, patch):
            continue
        if len(sequence) % 2 == 1 and np.array_equal(transformed, no_swap):
            continue
        if len(sequence) % 2 == 0 and not np.array_equal(transformed, no_swap):
            continue
        return patch
    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    for rr, cc in np.argwhere(patch != BG):
        grid[r0 + int(rr), c0 + int(cc)] = int(patch[rr, cc])


def _semantic_generation_check(grid: np.ndarray, patch: np.ndarray, colors: tuple[int, int], plan: dict) -> bool:
    try:
        sprite = _find_sprite(grid)
        mirrors = _mirror_map(grid)
        traced = _trace(grid, sprite, mirrors)
        expected_patch = _apply_sequence(patch, colors, plan["sequence"], swap=True)
        actual_hits = [(pos, orient) for pos, orient in traced["hits"]]
        planned_hits = [(m["hit"], m["orient"]) for m in plan["mirrors"]]
        if actual_hits != planned_hits:
            return False
        if traced["exit_cell"] != plan["exit_cell"] or traced["exit_dir"] != plan["exit_dir"]:
            return False
        if not np.array_equal(traced["patch"], expected_patch):
            return False
        if np.array_equal(main(grid), grid):
            return False
    except Exception:
        return False
    return True


_GEN_CALL_COUNT = 0


def _sequence_for_call() -> list[str]:
    forced = [
        ["/"],
        ["\\"],
        ["/", "\\"],
        ["\\", "/"],
    ]
    if _GEN_CALL_COUNT < len(forced):
        return forced[_GEN_CALL_COUNT]
    if random.random() < 0.55:
        return [random.choice(["/", "\\"])]
    return random.choice([["/", "\\"], ["\\", "/"]])


def _gen_attempt(sequence: list[str]) -> np.ndarray | None:
    size = random.choice([3, 5])
    min_side = 14 if size == 3 else 17
    h = random.randint(min_side, 28)
    w = random.randint(min_side, 28)
    side = random.choice(["top", "bottom", "left", "right"])

    colors = tuple(random.sample(SPRITE_COLORS, 2))
    patch = _random_sprite(size, colors, sequence)
    if patch is None:
        return None

    plan = _build_plan((h, w), size, side, sequence)
    if plan is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for mirror in plan["mirrors"]:
        for r, c in mirror["cells"]:
            if grid[r, c] != BG:
                return None
            grid[r, c] = GREY

    r0, c0, _r1, _c1 = plan["sprite_box"]
    _place_patch(grid, patch, r0, c0)

    if not _semantic_generation_check(grid, patch, colors, plan):
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    sequence = _sequence_for_call()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(sequence)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid diagonal mirror bounce input")
