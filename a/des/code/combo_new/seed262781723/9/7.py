from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BG = 0
LINE = 1
FACE = 5
PORTAL_COLORS = (2, 3, 4, 6, 7, 8, 9)

RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
UP = (-1, 0)
DIRS = (RIGHT, DOWN, LEFT, UP)
PERP_DIRS = {
    RIGHT: (UP, DOWN),
    LEFT: (UP, DOWN),
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
}

RNG = np.random.default_rng()
MAX_GEN_ATTEMPTS = 400
MAX_PLACE_ATTEMPTS = 200


def _in_bounds(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _add(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * steps, cell[1] + direction[1] * steps


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _find_portals(
    grid: np.ndarray,
) -> tuple[
    dict[tuple[int, int], dict[str, object]],
    dict[tuple[int, int], dict[str, object]],
    dict[int, list[dict[str, object]]],
]:
    h, w = grid.shape
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, LINE, FACE)]
    portals_by_color: dict[int, list[dict[str, object]]] = {}
    portal_by_face: dict[tuple[int, int], dict[str, object]] = {}
    partner_by_face: dict[tuple[int, int], dict[str, object]] = {}
    used_faces: set[tuple[int, int]] = set()

    for color in colors:
        markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(markers) != 2:
            raise AssertionError(f"portal color {color} must appear exactly twice")
        portals: list[dict[str, object]] = []
        for marker in markers:
            faces = []
            for direction in DIRS:
                face = _add(marker, direction)
                if _in_bounds(face, (h, w)) and int(grid[face]) == FACE:
                    faces.append(face)
            if len(faces) != 1:
                raise AssertionError(f"portal marker {marker} must have exactly one adjacent grey face")
            face = faces[0]
            if face in used_faces:
                raise AssertionError("grey face cannot belong to more than one portal")
            used_faces.add(face)
            portal = {
                "color": int(color),
                "marker": marker,
                "face": face,
                "direction": (face[0] - marker[0], face[1] - marker[1]),
            }
            portals.append(portal)
        portals_by_color[int(color)] = portals

    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == FACE)}
    if grey_cells != used_faces:
        raise AssertionError("all grey cells must be portal faces")

    for portals in portals_by_color.values():
        a, b = portals
        portal_by_face[a["face"]] = a
        portal_by_face[b["face"]] = b
        partner_by_face[a["face"]] = b
        partner_by_face[b["face"]] = a

    return portal_by_face, partner_by_face, portals_by_color


def _find_segments(grid: np.ndarray) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for coords in _components(grid == LINE):
        if len(coords) < 2:
            raise AssertionError("blue segments must have length at least 2")
        rows = sorted({r for r, _ in coords})
        cols = sorted({c for _, c in coords})
        if len(rows) == 1:
            row = rows[0]
            sorted_cols = sorted(c for _, c in coords)
            if sorted_cols != list(range(sorted_cols[0], sorted_cols[-1] + 1)):
                raise AssertionError("horizontal segment must be contiguous")
            segments.append(
                {
                    "axis": "h",
                    "length": len(coords),
                    "cells": {(row, c) for c in sorted_cols},
                    "endpoints": ((row, sorted_cols[0]), (row, sorted_cols[-1])),
                }
            )
        elif len(cols) == 1:
            col = cols[0]
            sorted_rows = sorted(r for r, _ in coords)
            if sorted_rows != list(range(sorted_rows[0], sorted_rows[-1] + 1)):
                raise AssertionError("vertical segment must be contiguous")
            segments.append(
                {
                    "axis": "v",
                    "length": len(coords),
                    "cells": {(r, col) for r in sorted_rows},
                    "endpoints": ((sorted_rows[0], col), (sorted_rows[-1], col)),
                }
            )
        else:
            raise AssertionError("blue component must be a straight segment")
    segments.sort(key=lambda seg: tuple(sorted(seg["cells"])))
    return segments


def _segment_entries(
    segment: dict[str, object],
    portal_by_face: dict[tuple[int, int], dict[str, object]],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    axis = str(segment["axis"])
    first, last = segment["endpoints"]
    checks: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]
    if axis == "h":
        checks = [
            (first, (first[0], first[1] - 1), RIGHT),
            (last, (last[0], last[1] + 1), LEFT),
        ]
    else:
        checks = [
            (first, (first[0] - 1, first[1]), DOWN),
            (last, (last[0] + 1, last[1]), UP),
        ]

    for endpoint, face, outward in checks:
        portal = portal_by_face.get(face)
        if portal is None:
            continue
        if tuple(portal["direction"]) != outward:
            continue
        entries.append({"endpoint": endpoint, "face": face, "portal": portal})
    return entries


def _draw_extension(
    out: np.ndarray,
    base: np.ndarray,
    face: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> None:
    cursor = face
    for _ in range(length):
        cursor = _add(cursor, direction)
        if not _in_bounds(cursor, base.shape):
            break
        if int(base[cursor]) != BG:
            break
        out[cursor] = LINE


def main(grid: np.ndarray) -> np.ndarray:
    """Continue blue line segments through matched portal pairs using the exit portal's facing direction.

    Input structure:
    - The background is black.
    - Several portal colors appear exactly twice; each colored portal marker has one orthogonally adjacent grey face pixel.
    - Blue pixels form straight horizontal or vertical line segments; some segments terminate against a portal face on that face's open side.

    Rule:
    - Pair portal markers by color and infer each portal's facing direction from marker to grey face.
    - If a blue segment endpoint touches a portal face on the open side, keep the original segment unchanged and continue the segment from the paired portal's grey face.
    - The continued segment uses the paired portal's facing direction and has the same blue-pixel length as the original segment, stopping early at the grid boundary or before any original non-black pixel.
    - Portal markers and grey faces remain unchanged; only the added blue continuation is drawn.

    Key insight:
    - Solving requires three linked inferences: color pairing identifies portal partners, marker-face geometry gives the exit direction, and the original segment length controls how far the teleported continuation travels.
    """
    g = np.asarray(grid, dtype=int)
    portal_by_face, partner_by_face, _ = _find_portals(g)
    segments = _find_segments(g)
    out = g.copy()

    for segment in segments:
        entries = _segment_entries(segment, portal_by_face)
        for entry in entries:
            partner = partner_by_face[entry["face"]]
            _draw_extension(
                out,
                g,
                tuple(partner["face"]),
                tuple(partner["direction"]),
                int(segment["length"]),
            )

    return out


def _max_ahead(face: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    steps = 0
    cursor = face
    while True:
        cursor = _add(cursor, direction)
        if not _in_bounds(cursor, shape):
            return steps
        steps += 1


def _face_candidates(
    shape: tuple[int, int],
    direction: tuple[int, int],
    min_ahead: int = 0,
    max_ahead: int | None = None,
) -> list[tuple[tuple[int, int], int]]:
    h, w = shape
    cands: list[tuple[tuple[int, int], int]] = []
    for r in range(h):
        for c in range(w):
            face = (r, c)
            marker = _add(face, (-direction[0], -direction[1]))
            if not _in_bounds(marker, shape):
                continue
            ahead = _max_ahead(face, direction, shape)
            if ahead < min_ahead:
                continue
            if max_ahead is not None and ahead > max_ahead:
                continue
            cands.append((face, ahead))
    return cands


def _path_from_face(face: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [_add(face, direction, step) for step in range(1, length + 1)]


def _place_cells(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for cell in cells:
        grid[cell] = int(color)


def _all_free(
    grid: np.ndarray,
    exact_cells: Iterable[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
) -> bool:
    seen: set[tuple[int, int]] = set()
    for cell in exact_cells:
        if cell in seen:
            return False
        seen.add(cell)
        if not _in_bounds(cell, grid.shape):
            return False
        if int(grid[cell]) != BG:
            return False
        if cell in reserved_black:
            return False
    return True


def _reserve_free(
    grid: np.ndarray,
    reserve_cells: Iterable[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    blocked_now: set[tuple[int, int]],
) -> bool:
    for cell in reserve_cells:
        if not _in_bounds(cell, grid.shape):
            return False
        if int(grid[cell]) != BG:
            return False
        if cell in reserved_black:
            return False
        if cell in blocked_now:
            return False
    return True


def _choose_modes(count: int) -> list[str]:
    base = ["clear", "edge", "object"]
    if count <= 0:
        return []
    if count == 1:
        return [str(RNG.choice(base))]
    if count == 2:
        picks = RNG.choice(base, size=2, replace=False)
        return [str(x) for x in picks]
    modes = list(RNG.permutation(base))
    while len(modes) < count:
        modes.append(str(RNG.choice(base)))
    return modes[:count]


def _make_centered_blocker(
    block: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    options = list(PERP_DIRS[direction])
    order = RNG.permutation(len(options))
    for idx in order:
        perp = options[int(idx)]
        cells = [_add(block, perp, -1), block, _add(block, perp, 1)]
        if all(_in_bounds(cell, shape) for cell in cells):
            return cells
    return None


def _touches_existing_portal(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    axis: str,
    reserved_faces: dict[tuple[int, int], dict[str, object]],
) -> bool:
    if axis == "h":
        row = cells[0][0]
        cols = sorted(c for _, c in cells)
        checks = [((row, cols[0] - 1), RIGHT), ((row, cols[-1] + 1), LEFT)]
    else:
        col = cells[0][1]
        rows = sorted(r for r, _ in cells)
        checks = [((rows[0] - 1, col), DOWN), ((rows[-1] + 1, col), UP)]
    for face, outward in checks:
        portal = reserved_faces.get(face)
        if portal is not None and tuple(portal["direction"]) == outward:
            return True
    return False


def _place_idle_pair(
    grid: np.ndarray,
    color: int,
    reserved_black: set[tuple[int, int]],
) -> bool:
    shape = grid.shape
    for _ in range(MAX_PLACE_ATTEMPTS):
        d1 = DIRS[int(RNG.integers(0, len(DIRS)))]
        d2 = DIRS[int(RNG.integers(0, len(DIRS)))]
        cand1 = _face_candidates(shape, d1)
        cand2 = _face_candidates(shape, d2)
        if not cand1 or not cand2:
            return False
        face1, _ = cand1[int(RNG.integers(0, len(cand1)))]
        face2, _ = cand2[int(RNG.integers(0, len(cand2)))]
        marker1 = _add(face1, (-d1[0], -d1[1]))
        marker2 = _add(face2, (-d2[0], -d2[1]))
        cells = [marker1, face1, marker2, face2]
        if not _all_free(grid, cells, reserved_black):
            continue
        if _add(face1, d1) == face2 or _add(face2, d2) == face1:
            continue
        grid[marker1] = color
        grid[face1] = FACE
        grid[marker2] = color
        grid[face2] = FACE
        return True
    return False


def _place_active_pair(
    grid: np.ndarray,
    color: int,
    mode: str,
    reserved_black: set[tuple[int, int]],
    portal_by_face: dict[tuple[int, int], dict[str, object]] | None = None,
) -> bool:
    shape = grid.shape
    max_len = min(7, max(3, min(shape) - 3))
    for _ in range(MAX_PLACE_ATTEMPTS):
        length = int(RNG.integers(2, max_len))
        d_in = DIRS[int(RNG.integers(0, len(DIRS)))]
        d_out = DIRS[int(RNG.integers(0, len(DIRS)))]

        entry_cands = _face_candidates(shape, d_in, min_ahead=length)
        if not entry_cands:
            continue
        face_a, _ = entry_cands[int(RNG.integers(0, len(entry_cands)))]
        marker_a = _add(face_a, (-d_in[0], -d_in[1]))
        entry_line = _path_from_face(face_a, d_in, length)
        exact_entry = [marker_a, face_a, *entry_line]
        if not _all_free(grid, exact_entry, reserved_black):
            continue

        reserve_cells: list[tuple[int, int]] = []
        blocker_cells: list[tuple[int, int]] = []

        if mode == "clear":
            exit_cands = _face_candidates(shape, d_out, min_ahead=length)
            if not exit_cands:
                continue
            face_b, _ = exit_cands[int(RNG.integers(0, len(exit_cands)))]
            reserve_cells = _path_from_face(face_b, d_out, length)
        elif mode == "edge":
            exit_cands = _face_candidates(shape, d_out, min_ahead=1, max_ahead=length - 1)
            if not exit_cands:
                continue
            face_b, ahead = exit_cands[int(RNG.integers(0, len(exit_cands)))]
            reserve_cells = _path_from_face(face_b, d_out, int(ahead))
        else:
            clear_k = int(RNG.integers(1, length))
            exit_cands = _face_candidates(shape, d_out, min_ahead=clear_k + 1)
            if not exit_cands:
                continue
            face_b, _ = exit_cands[int(RNG.integers(0, len(exit_cands)))]
            block = _add(face_b, d_out, clear_k + 1)
            blocker_cells = _make_centered_blocker(block, d_out, shape) or []
            if not blocker_cells:
                continue
            reserve_cells = _path_from_face(face_b, d_out, clear_k)

        marker_b = _add(face_b, (-d_out[0], -d_out[1]))
        exact_exit = [marker_b, face_b, *blocker_cells]
        combined_exact = exact_entry + exact_exit
        if not _all_free(grid, combined_exact, reserved_black):
            continue
        blocked_now = set(combined_exact)
        if not _reserve_free(grid, reserve_cells, reserved_black, blocked_now):
            continue
        if portal_by_face is not None and blocker_cells:
            axis = "v" if blocker_cells[0][1] == blocker_cells[1][1] else "h"
            if _touches_existing_portal(grid, blocker_cells, axis, portal_by_face):
                continue
        if set(exact_entry) & set(exact_exit):
            continue
        if set(reserve_cells) & set(combined_exact):
            continue

        grid[marker_a] = color
        grid[face_a] = FACE
        _place_cells(grid, entry_line, LINE)
        grid[marker_b] = color
        grid[face_b] = FACE
        _place_cells(grid, blocker_cells, LINE)
        reserved_black.update(reserve_cells)
        return True
    return False


def _place_free_line(
    grid: np.ndarray,
    reserved_black: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    portal_by_face, _, _ = _find_portals(grid)
    for _ in range(MAX_PLACE_ATTEMPTS):
        axis = "h" if int(RNG.integers(0, 2)) == 0 else "v"
        if axis == "h":
            length = int(RNG.integers(2, min(7, w)))
            row = int(RNG.integers(0, h))
            start = int(RNG.integers(0, w - length + 1))
            cells = [(row, c) for c in range(start, start + length)]
        else:
            length = int(RNG.integers(2, min(7, h)))
            col = int(RNG.integers(0, w))
            start = int(RNG.integers(0, h - length + 1))
            cells = [(r, col) for r in range(start, start + length)]
        if not _all_free(grid, cells, reserved_black):
            continue
        if _touches_existing_portal(grid, cells, axis, portal_by_face):
            continue
        _place_cells(grid, cells, LINE)
        return True
    return False


def _validate_generated(
    grid: np.ndarray,
    expected_active_colors: set[int],
    total_pairs: int,
) -> bool:
    try:
        portal_by_face, _, portals_by_color = _find_portals(grid)
        if len(portals_by_color) != total_pairs:
            return False
        segments = _find_segments(grid)
        used_colors: set[int] = set()
        active_segments = 0
        for segment in segments:
            entries = _segment_entries(segment, portal_by_face)
            if len(entries) > 1:
                return False
            if entries:
                active_segments += 1
                used_colors.add(int(entries[0]["portal"]["color"]))
        if used_colors != expected_active_colors:
            return False
        if active_segments != len(expected_active_colors):
            return False
        out = main(grid)
    except AssertionError:
        return False
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(14, 25))
    w = int(RNG.integers(14, 25))
    grid = np.full((h, w), BG, dtype=int)

    active_count = int(RNG.integers(1, 4))
    idle_cap = min(2, len(PORTAL_COLORS) - active_count)
    idle_count = int(RNG.integers(0, idle_cap + 1))
    free_line_count = int(RNG.integers(0, 3))
    total_pairs = active_count + idle_count
    colors = list(RNG.choice(PORTAL_COLORS, size=total_pairs, replace=False))
    active_colors = set(colors[:active_count])
    idle_colors = colors[active_count:]

    reserved_black: set[tuple[int, int]] = set()

    for color in idle_colors:
        if not _place_idle_pair(grid, int(color), reserved_black):
            return None

    portal_view = None
    for color, mode in zip(colors[:active_count], _choose_modes(active_count)):
        if portal_view is None:
            portal_view = {}
        if not _place_active_pair(grid, int(color), mode, reserved_black, portal_view):
            return None
        try:
            portal_view, _, _ = _find_portals(grid)
        except AssertionError:
            return None

    for _ in range(free_line_count):
        if not _place_free_line(grid, reserved_black):
            break

    if not _validate_generated(grid, active_colors, total_pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-line ARC sample")
