from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR_BY_CORNER = {
    "ul": (1, 1),
    "ur": (1, -1),
    "ll": (-1, 1),
    "lr": (-1, -1),
}
CORNER_BY_SIDE = {
    "upper": ("ul", "ur"),
    "lower": ("ll", "lr"),
}

BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _find_wall(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and bool(np.all(grid[r] == grid[r, 0]))
    ]
    assert len(candidates) == 1, "expected exactly one full-width wall row"
    wall_row = int(candidates[0])
    return wall_row, int(grid[wall_row, 0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _local_cells(height: int, width: int, missing: str) -> tuple[tuple[int, int], ...]:
    miss = {
        "ul": (0, 0),
        "ur": (0, width - 1),
        "ll": (height - 1, 0),
        "lr": (height - 1, width - 1),
    }[missing]
    return tuple(
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) != miss
    )


def _translate_cells(
    cells: tuple[tuple[int, int], ...], row0: int, col0: int
) -> tuple[tuple[int, int], ...]:
    return tuple((row0 + r, col0 + c) for r, c in cells)


def _analyze_sprite(
    coords: np.ndarray,
    color: int,
    wall_row: int,
) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height >= 2 and width >= 2

    mask = np.zeros((height, width), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    assert int(mask.sum()) == height * width - 1

    holes = np.argwhere(~mask)
    assert holes.shape == (1, 2)
    hole_r, hole_c = map(int, holes[0])
    hole_to_corner = {
        (0, 0): "ul",
        (0, width - 1): "ur",
        (height - 1, 0): "ll",
        (height - 1, width - 1): "lr",
    }
    assert (hole_r, hole_c) in hole_to_corner
    missing = hole_to_corner[(hole_r, hole_c)]
    dr, dc = DIR_BY_CORNER[missing]

    if r1 < wall_row:
        side = "upper"
        assert dr == 1
        distance = wall_row - 1 - r1
    elif r0 > wall_row:
        side = "lower"
        assert dr == -1
        distance = r0 - (wall_row + 1)
    else:
        raise AssertionError("sprites must stay entirely on one side of the wall")
    assert distance >= 0

    translated = tuple(
        (int(r) + distance * dr, int(c) + distance * dc) for r, c in coords
    )

    return {
        "color": int(color),
        "coords": tuple((int(r), int(c)) for r, c in coords),
        "bbox": (r0, c0, r1, c1),
        "height": height,
        "width": width,
        "missing": missing,
        "side": side,
        "dr": int(dr),
        "dc": int(dc),
        "distance": int(distance),
        "translated": translated,
    }


def _extract_sprites(grid: np.ndarray, bg: int, wall_row: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    sprites: list[dict[str, object]] = []

    for r in range(h):
        if r == wall_row:
            continue
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and nr != wall_row
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            sprites.append(_analyze_sprite(np.asarray(cells, dtype=int), color, wall_row))
    return sprites


def _sprite_sort_key(sprite: dict[str, object]) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = sprite["bbox"]
    return (-int(sprite["distance"]), int(r0), int(c0), int(r1), int(c1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each corner-missing rectangular sprite diagonally toward the grey wall.

    The missing corner encodes the diagonal direction; each sprite slides along
    that 45-degree diagonal until it becomes adjacent to the wall row. Original
    sprites remain in place, while projected copies are painted from farthest to
    nearest so nearer copies overwrite farther ones where they overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    wall_row, _wall_color = _find_wall(g, bg)
    sprites = _extract_sprites(g, bg, wall_row)
    assert sprites

    out = g.copy()
    for sprite in sorted(sprites, key=_sprite_sort_key):
        for r, c in sprite["translated"]:
            out[int(r), int(c)] = int(sprite["color"])
    return out


def _boxes_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 0,
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _max_step(side: str, wall_row: int, height: int, total_rows: int) -> int:
    if side == "upper":
        return int(wall_row - height)
    return int(total_rows - wall_row - height - 1)


def _build_spec(
    side: str,
    color: int,
    total_rows: int,
    total_cols: int,
    wall_row: int,
    height: int,
    width: int,
    missing: str,
    step: int,
    final_col0: int,
) -> dict[str, object] | None:
    dr, dc = DIR_BY_CORNER[missing]
    final_row0 = wall_row - height if side == "upper" else wall_row + 1
    orig_row0 = final_row0 - step * dr
    orig_col0 = final_col0 - step * dc
    if not (
        0 <= orig_row0 <= total_rows - height
        and 0 <= orig_col0 <= total_cols - width
        and 0 <= final_col0 <= total_cols - width
    ):
        return None

    local = _local_cells(height, width, missing)
    orig_cells = _translate_cells(local, orig_row0, orig_col0)
    final_cells = _translate_cells(local, final_row0, final_col0)
    orig_bbox = (orig_row0, orig_col0, orig_row0 + height - 1, orig_col0 + width - 1)
    final_bbox = (final_row0, final_col0, final_row0 + height - 1, final_col0 + width - 1)

    if side == "upper":
        if orig_bbox[2] >= wall_row or final_bbox[2] != wall_row - 1:
            return None
    else:
        if orig_bbox[0] <= wall_row or final_bbox[0] != wall_row + 1:
            return None

    return {
        "side": side,
        "color": int(color),
        "height": int(height),
        "width": int(width),
        "missing": missing,
        "dr": int(dr),
        "dc": int(dc),
        "step": int(step),
        "local": local,
        "orig_cells": orig_cells,
        "final_cells": final_cells,
        "orig_bbox": orig_bbox,
        "final_bbox": final_bbox,
    }


def _compatible_spec(
    candidate: dict[str, object],
    placed: list[dict[str, object]],
    overlap_target: dict[str, object] | None,
) -> bool:
    target_hit = overlap_target is None
    cand_final = set(candidate["final_cells"])

    for other in placed:
        if _boxes_intersect(candidate["orig_bbox"], other["orig_bbox"], gap=1):
            return False
        if _boxes_intersect(candidate["orig_bbox"], other["final_bbox"]):
            return False
        if _boxes_intersect(candidate["final_bbox"], other["orig_bbox"]):
            return False

        overlap = bool(cand_final & set(other["final_cells"]))
        if overlap_target is not None and other is overlap_target:
            if int(candidate["step"]) == int(other["step"]):
                return False
            if not overlap:
                return False
            target_hit = True
        elif overlap or _boxes_intersect(candidate["final_bbox"], other["final_bbox"]):
            return False

    return target_hit


def _place_sprite(
    side: str,
    color: int,
    total_rows: int,
    total_cols: int,
    wall_row: int,
    placed: list[dict[str, object]],
    overlap_target: dict[str, object] | None = None,
) -> dict[str, object] | None:
    for _ in range(80):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        missing = random.choice(CORNER_BY_SIDE[side])
        max_step = _max_step(side, wall_row, height, total_rows)
        if max_step < 1:
            continue

        steps = list(range(1, max_step + 1))
        random.shuffle(steps)
        for step in steps:
            final_cols = list(range(total_cols - width + 1))
            random.shuffle(final_cols)
            candidates: list[dict[str, object]] = []
            for final_col0 in final_cols:
                spec = _build_spec(
                    side,
                    color,
                    total_rows,
                    total_cols,
                    wall_row,
                    height,
                    width,
                    missing,
                    step,
                    final_col0,
                )
                if spec is not None and _compatible_spec(spec, placed, overlap_target):
                    candidates.append(spec)
            if candidates:
                return random.choice(candidates)
    return None


def _render_input(
    total_rows: int,
    total_cols: int,
    wall_row: int,
    specs: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((total_rows, total_cols), BG, dtype=int)
    grid[wall_row, :] = WALL
    for spec in specs:
        for r, c in spec["orig_cells"]:
            grid[int(r), int(c)] = int(spec["color"])
    return grid


def _render_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in sorted(
        specs,
        key=lambda s: (
            -int(s["step"]),
            int(s["orig_bbox"][0]),
            int(s["orig_bbox"][1]),
            int(s["orig_bbox"][2]),
            int(s["orig_bbox"][3]),
        ),
    ):
        for r, c in spec["final_cells"]:
            out[int(r), int(c)] = int(spec["color"])
    return out


def _has_projection_overlap(specs: list[dict[str, object]]) -> bool:
    final_sets = [set(spec["final_cells"]) for spec in specs]
    for i in range(len(final_sets)):
        for j in range(i + 1, len(final_sets)):
            if final_sets[i] & final_sets[j]:
                return True
    return False


def _side_counts() -> tuple[int, int]:
    return random.choice(
        [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (1, 3),
            (3, 1),
            (2, 3),
            (3, 2),
        ]
    )


def _gen_attempt() -> np.ndarray | None:
    total_rows = random.randint(14, 22)
    total_cols = random.randint(14, 22)
    wall_row = random.randint(5, total_rows - 6)

    upper_count, lower_count = _side_counts()
    total_count = upper_count + lower_count
    if total_count > len(PALETTE):
        return None

    want_overlap = random.random() < 0.5
    overlap_side = None
    if want_overlap:
        overlap_options = [
            side
            for side, count in (("upper", upper_count), ("lower", lower_count))
            if count >= 2
        ]
        if overlap_options:
            overlap_side = random.choice(overlap_options)
        else:
            want_overlap = False

    colors = random.sample(PALETTE, total_count)
    side_to_colors = {
        "upper": colors[:upper_count],
        "lower": colors[upper_count:],
    }

    specs: list[dict[str, object]] = []
    for side in random.sample(["upper", "lower"], 2):
        colors_for_side = side_to_colors[side]
        if side == overlap_side:
            first = _place_sprite(side, colors_for_side[0], total_rows, total_cols, wall_row, specs)
            if first is None:
                return None
            specs.append(first)

            second = _place_sprite(
                side,
                colors_for_side[1],
                total_rows,
                total_cols,
                wall_row,
                specs,
                overlap_target=first,
            )
            if second is None:
                return None
            specs.append(second)

            for color in colors_for_side[2:]:
                spec = _place_sprite(side, color, total_rows, total_cols, wall_row, specs)
                if spec is None:
                    return None
                specs.append(spec)
        else:
            for color in colors_for_side:
                spec = _place_sprite(side, color, total_rows, total_cols, wall_row, specs)
                if spec is None:
                    return None
                specs.append(spec)

    inp = _render_input(total_rows, total_cols, wall_row, specs)
    expected = _render_output(inp, specs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    if _has_projection_overlap(specs) != want_overlap:
        return None

    if np.array_equal(inp, expected):
        return None

    bg = _most_common_color(inp)
    found_wall_row, found_wall_color = _find_wall(inp, bg)
    if bg != BG or found_wall_row != wall_row or found_wall_color != WALL:
        return None

    extracted = _extract_sprites(inp, bg, wall_row)
    if len(extracted) != total_count:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
