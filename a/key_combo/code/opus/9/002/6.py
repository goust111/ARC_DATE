from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
BODY_COLORS = (1, 2, 3, 6, 7, 8, 9)
SHAPE_BOX = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-2, -1, 0, 1, 2))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_axis(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected a unique full grey symmetry axis"
    return int(cols[0])


def _mirror_coord(coord: tuple[int, int], axis_col: int) -> tuple[int, int]:
    r, c = coord
    return int(r), int(2 * axis_col - c)


def _extract_sprites(grid: np.ndarray, axis_col: int) -> list[dict]:
    mask = grid != BLACK
    mask[:, axis_col] = False
    sprites = []
    for coords in _components_mask(mask):
        marker_cells = [(int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) == GREY]
        assert len(marker_cells) == 1, "each sprite must contain exactly one grey marker"
        r0, c0, r1, c1 = _bbox_coords(coords)
        if c1 < axis_col:
            side = "L"
        elif c0 > axis_col:
            side = "R"
        else:
            raise AssertionError("sprites must lie strictly on one side of the axis")
        cells = {(int(r), int(c)): int(grid[int(r), int(c)]) for r, c in coords}
        sprites.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "marker": marker_cells[0],
                "side": side,
                "cells": cells,
            }
        )
    return sprites


def _pair_map(sprites: list[dict], axis_col: int) -> dict[int, int | None]:
    marker_to_idx = {sprite["marker"]: idx for idx, sprite in enumerate(sprites)}
    assert len(marker_to_idx) == len(sprites), "marker positions must be unique"

    partner: dict[int, int | None] = {}
    for idx, sprite in enumerate(sprites):
        mirrored_marker = _mirror_coord(sprite["marker"], axis_col)
        other = marker_to_idx.get(mirrored_marker)
        if other is None:
            partner[idx] = None
            continue
        assert other != idx, "a sprite marker cannot mirror onto itself"
        assert sprites[other]["side"] != sprite["side"], "paired sprites must lie on opposite sides"
        partner[idx] = other

    for idx, other in partner.items():
        if other is not None:
            assert partner[other] == idx, "pairing must be symmetric"
    return partner


def _mirror_cells(cell_map: dict[tuple[int, int], int], axis_col: int) -> dict[tuple[int, int], int]:
    return {_mirror_coord(coord, axis_col): color for coord, color in cell_map.items()}


def _apply_transform(grid: np.ndarray, sprites: list[dict], partner: dict[int, int | None], axis_col: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    used: set[int] = set()
    for idx, other in partner.items():
        if other is None or idx in used:
            continue
        used.add(idx)
        used.add(other)
        for mirrored in (_mirror_cells(sprites[idx]["cells"], axis_col), _mirror_cells(sprites[other]["cells"], axis_col)):
            for (r, c), color in mirrored.items():
                out[r, c] = int(color)
        for sprite_idx in (idx, other):
            mr, mc = sprites[sprite_idx]["marker"]
            out[mr, mc] = YELLOW
    return np.concatenate([out[:, :axis_col], out[:, axis_col + 1 :]], axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each sprite across the vertical grey axis and overlay it onto the opposite-side sprite only when the mirrored marker lands on that sprite's marker.

    Input structure:
    - A black background contains one unique full-height grey column that acts as
      the vertical symmetry axis.
    - On both sides of that axis sit several small 4-connected multi-color
      sprites; each sprite contains exactly one grey marker pixel.
    - Sprites do not touch each other, and every sprite lies strictly on one side
      of the axis.

    Rule:
    - Find the unique full grey column and treat it as the mirror axis.
    - For every sprite, reflect each pixel `(r, c)` to `(r, 2*axis_col - c)`.
    - If the mirrored position of that sprite's grey marker exactly matches the
      original grey marker of another sprite, those two sprites form a pair.
    - For each paired sprite, transparently overlay its mirrored non-black pixels
      onto the partner side, overwriting any partner colors at overlapped cells.
      After both overlay directions are applied, recolor both original marker
      pixels of the pair to yellow.
    - Sprites whose mirrored marker matches no other marker remain unchanged.
    - Remove the grey axis column from the final grid by shifting the right side
      one column left.

    Key insight:
    - The mirror location alone is not enough: overlay happens only for sprites
      whose mirrored marker coincides with another sprite's marker, so the marker
      acts as a gate on whether the mirrored copy becomes visible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    axis_col = _find_axis(g)
    sprites = _extract_sprites(g, axis_col)
    partner = _pair_map(sprites, axis_col)
    return _apply_transform(g, sprites, partner, axis_col)


def _shape_bbox(rel_cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in rel_cells]
    cols = [c for _r, c in rel_cells]
    return min(rows), min(cols), max(rows), max(cols)


def _random_shape(size: int) -> set[tuple[int, int]]:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in SHAPE_BOX and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != size:
            continue
        r0, c0, r1, c1 = _shape_bbox(cells)
        if r1 - r0 < 1 or c1 - c0 < 1:
            continue
        return cells
    raise RuntimeError("failed to sample a connected sprite shape")


def _random_sprite_spec() -> dict:
    size = random.randint(3, 8)
    rel_cells = _random_shape(size)
    body = sorted(cell for cell in rel_cells if cell != (0, 0))
    n_colors = 2 if len(body) <= 3 else random.choice((2, 3))
    palette = random.sample(BODY_COLORS, n_colors)
    random.shuffle(body)
    body_colors = {}
    for idx, cell in enumerate(body):
        body_colors[cell] = palette[idx % len(palette)]
    random.shuffle(body)
    rel_map = {(0, 0): GREY}
    rel_map.update(body_colors)
    return {"rel_cells": rel_cells, "rel_map": rel_map}


def _sprite_cells_at(spec: dict, marker: tuple[int, int]) -> dict[tuple[int, int], int]:
    mr, mc = marker
    return {(mr + dr, mc + dc): color for (dr, dc), color in spec["rel_map"].items()}


def _touching_or_overlapping(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in b:
                    return True
    return False


def _in_bounds_side(cell_map: dict[tuple[int, int], int], height: int, width: int, axis_col: int, side: str) -> bool:
    for r, c in cell_map:
        if not (0 <= r < height and 0 <= c < width):
            return False
        if side == "L" and c >= axis_col:
            return False
        if side == "R" and c <= axis_col:
            return False
    return True


def _mirror_hits_only_allowed(
    cell_map: dict[tuple[int, int], int],
    axis_col: int,
    occupied: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
) -> bool:
    mirrored_coords = set(_mirror_cells(cell_map, axis_col))
    return len((mirrored_coords & occupied) - allowed) == 0


def _overlay_changes_target(
    source_map: dict[tuple[int, int], int],
    target_map: dict[tuple[int, int], int],
    target_marker: tuple[int, int],
    axis_col: int,
) -> bool:
    mirrored = _mirror_cells(source_map, axis_col)
    for coord, color in mirrored.items():
        if coord == target_marker:
            continue
        if target_map.get(coord, BLACK) != color:
            return True
    return False


def _left_marker_choices(spec: dict, axis_col: int) -> list[int]:
    cols = []
    for marker_col in range(1, axis_col):
        cell_map = _sprite_cells_at(spec, (3, marker_col))
        if all(c < axis_col for _r, c in cell_map):
            cols.append(marker_col)
    return cols


def _right_marker_choices(spec: dict, axis_col: int, width: int) -> list[int]:
    cols = []
    for marker_col in range(axis_col + 1, width - 1):
        cell_map = _sprite_cells_at(spec, (3, marker_col))
        if all(c > axis_col for _r, c in cell_map):
            cols.append(marker_col)
    return cols


def _try_place_pair(
    row: int,
    height: int,
    width: int,
    axis_col: int,
    placed: list[dict],
) -> tuple[dict, dict] | None:
    occupied = set().union(*(set(item["cells"]) for item in placed)) if placed else set()
    for _ in range(400):
        left_spec = _random_sprite_spec()
        right_spec = _random_sprite_spec()
        cols = _left_marker_choices(left_spec, axis_col)
        random.shuffle(cols)
        for left_col in cols:
            left_marker = (row, left_col)
            right_marker = _mirror_coord(left_marker, axis_col)
            left_map = _sprite_cells_at(left_spec, left_marker)
            right_map = _sprite_cells_at(right_spec, right_marker)
            left_coords = set(left_map)
            right_coords = set(right_map)
            if not _in_bounds_side(left_map, height, width, axis_col, "L"):
                continue
            if not _in_bounds_side(right_map, height, width, axis_col, "R"):
                continue
            if _touching_or_overlapping(left_coords, right_coords):
                continue
            if any(_touching_or_overlapping(left_coords, set(item["cells"])) for item in placed):
                continue
            if any(_touching_or_overlapping(right_coords, set(item["cells"])) for item in placed):
                continue
            if not _mirror_hits_only_allowed(left_map, axis_col, occupied | right_coords, right_coords):
                continue
            if not _mirror_hits_only_allowed(right_map, axis_col, occupied | left_coords, left_coords):
                continue
            if not _overlay_changes_target(left_map, right_map, right_marker, axis_col):
                continue
            if not _overlay_changes_target(right_map, left_map, left_marker, axis_col):
                continue
            return (
                {"side": "L", "marker": left_marker, "cells": left_map},
                {"side": "R", "marker": right_marker, "cells": right_map},
            )
    return None


def _try_place_single(
    side: str,
    row: int,
    height: int,
    width: int,
    axis_col: int,
    placed: list[dict],
) -> dict | None:
    occupied = set().union(*(set(item["cells"]) for item in placed)) if placed else set()
    for _ in range(400):
        spec = _random_sprite_spec()
        cols = _left_marker_choices(spec, axis_col) if side == "L" else _right_marker_choices(spec, axis_col, width)
        random.shuffle(cols)
        for marker_col in cols:
            marker = (row, marker_col)
            cell_map = _sprite_cells_at(spec, marker)
            coords = set(cell_map)
            if not _in_bounds_side(cell_map, height, width, axis_col, side):
                continue
            if any(_touching_or_overlapping(coords, set(item["cells"])) for item in placed):
                continue
            if not _mirror_hits_only_allowed(cell_map, axis_col, occupied, set()):
                continue
            return {"side": side, "marker": marker, "cells": cell_map}
    return None


def _render_grid(height: int, width: int, axis_col: int, placed: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[:, axis_col] = GREY
    for item in placed:
        for (r, c), color in item["cells"].items():
            grid[r, c] = int(color)
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    axis_col = _find_axis(grid)
    sprites = _extract_sprites(grid, axis_col)
    left_count = sum(sprite["side"] == "L" for sprite in sprites)
    right_count = sum(sprite["side"] == "R" for sprite in sprites)
    assert 2 <= left_count <= 4
    assert 2 <= right_count <= 4
    assert all(3 <= len(sprite["cells"]) <= 8 for sprite in sprites)

    for i, sprite_a in enumerate(sprites):
        coords_a = set(sprite_a["cells"])
        for j, sprite_b in enumerate(sprites):
            if i >= j:
                continue
            assert not _touching_or_overlapping(coords_a, set(sprite_b["cells"]))

    partner = _pair_map(sprites, axis_col)
    pair_ids = {tuple(sorted((idx, other))) for idx, other in partner.items() if other is not None}
    assert pair_ids
    assert len(pair_ids) <= min(left_count, right_count) - 1

    all_coords = set().union(*(set(sprite["cells"]) for sprite in sprites))
    for idx, sprite in enumerate(sprites):
        other = partner[idx]
        mirrored_coords = set(_mirror_cells(sprite["cells"], axis_col))
        if other is None:
            assert not (mirrored_coords & all_coords)
            continue
        target = sprites[other]
        assert (mirrored_coords & all_coords) <= set(target["cells"])
        assert _overlay_changes_target(sprite["cells"], target["cells"], target["marker"], axis_col)

    raw_without_axis = np.concatenate([grid[:, :axis_col], grid[:, axis_col + 1 :]], axis=1)
    transformed = main(grid)
    assert not np.array_equal(raw_without_axis, transformed)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(23, 25)
    width = random.choice([w for w in range(19, 26) if w % 2 == 1])
    axis_col = width // 2

    left_count = random.randint(2, 4)
    right_count = random.randint(2, 4)
    pair_count = random.randint(1, min(left_count, right_count) - 1)
    left_unpaired = left_count - pair_count
    right_unpaired = right_count - pair_count

    needed_rows = pair_count + left_unpaired + right_unpaired
    row_pool = list(range(2, height - 2, 3))
    if len(row_pool) < needed_rows:
        return None
    random.shuffle(row_pool)
    pair_rows = row_pool[:pair_count]
    left_rows = row_pool[pair_count : pair_count + left_unpaired]
    right_rows = row_pool[pair_count + left_unpaired : needed_rows]

    placed: list[dict] = []
    for row in pair_rows:
        pair = _try_place_pair(row, height, width, axis_col, placed)
        if pair is None:
            return None
        placed.extend(pair)
    for row in left_rows:
        item = _try_place_single("L", row, height, width, axis_col, placed)
        if item is None:
            return None
        placed.append(item)
    for row in right_rows:
        item = _try_place_single("R", row, height, width, axis_col, placed)
        if item is None:
            return None
        placed.append(item)

    grid = _render_grid(height, width, axis_col, placed)
    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-gated overlay puzzle")
