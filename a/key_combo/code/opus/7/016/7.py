import itertools
import random

import numpy as np


BG = 5
BLACK = 0
RED = 2
ORANGE = 7
MAROON = 9

TARGET_OUTLINE = np.array(
    [
        [MAROON, MAROON, MAROON],
        [MAROON, BG, MAROON],
        [MAROON, MAROON, MAROON],
    ],
    dtype=int,
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask has no bbox")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _single_component_mask(grid: np.ndarray, color: int) -> np.ndarray:
    mask = grid == int(color)
    comps = _components(mask)
    assert len(comps) == 1, f"expected exactly one component of color {color}"
    return mask


def _gate_opening_indices(line: np.ndarray) -> list[int] | None:
    line = np.asarray(line, dtype=int)
    n = int(line.size)
    top = 0
    while top < n and int(line[top]) == BLACK:
        top += 1
    bottom = 0
    while bottom < n and int(line[n - 1 - bottom]) == BLACK:
        bottom += 1
    if top == 0 or bottom == 0 or top + bottom >= n:
        return None
    middle = line[top : n - bottom]
    if np.any(middle == BLACK):
        return None
    return list(range(top, n - bottom))


def _rotate_cw(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, -int(turns) % 4)


def _shape_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _build_shape_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    for area in (4, 5, 6):
        for cells in itertools.combinations(range(9), area):
            mask = np.zeros((3, 3), dtype=bool)
            for idx in cells:
                mask[idx // 3, idx % 3] = True
            if not mask[0].any() or not mask[2].any() or not mask[:, 0].any() or not mask[:, 2].any():
                continue
            if len(_components(mask)) != 1:
                continue
            keys = {np.rot90(mask, k).tobytes() for k in range(4)}
            if len(keys) != 4:
                continue
            shapes.append(mask)
    assert shapes, "expected a non-empty shape library"
    return shapes


SHAPES = _build_shape_library()


def _movement_axis(
    src_box: tuple[int, int, int, int], dst_box: tuple[int, int, int, int]
) -> tuple[str, int, range, tuple[int, int]]:
    sr0, sc0, sr1, sc1 = src_box
    tr0, tc0, tr1, tc1 = dst_box
    if sr0 == tr0 and sr1 == tr1:
        if sc1 < tc0:
            return "horizontal", 1, range(sc1 + 1, tc0), (sr0, sr1)
        if tc1 < sc0:
            return "horizontal", -1, range(tc1 + 1, sc0), (sr0, sr1)
    if sc0 == tc0 and sc1 == tc1:
        if sr1 < tr0:
            return "vertical", 1, range(sr1 + 1, tr0), (sc0, sc1)
        if tr1 < sr0:
            return "vertical", -1, range(tr1 + 1, sr0), (sc0, sc1)
    raise AssertionError("source and target must share exactly one axis-aligned corridor")


def _find_gates(
    grid: np.ndarray,
    axis: str,
    corridor_steps: range,
    span: tuple[int, int],
) -> tuple[int, list[tuple[int, int]]]:
    gate_count = 0
    openings: list[tuple[int, int]] = []
    a0, a1 = span
    if axis == "horizontal":
        for c in corridor_steps:
            opening_rows = _gate_opening_indices(grid[a0 : a1 + 1, c])
            if opening_rows is None:
                continue
            gate_count += 1
            for rr in opening_rows:
                openings.append((a0 + rr, c))
    else:
        for r in corridor_steps:
            opening_cols = _gate_opening_indices(grid[r, a0 : a1 + 1])
            if opening_cols is None:
                continue
            gate_count += 1
            for cc in opening_cols:
                openings.append((r, a0 + cc))
    return gate_count, openings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the orange sprite through a gated corridor, count the gates, and use that count to rotate it at the destination.

    Input structure:
    - The background is gray.
    - One small orange 4-connected sprite sits at one end of a straight corridor.
    - One maroon 3x3 outline marks the destination box at the other end and shares the source's row span or column span.
    - Inside the corridor, black obstacle caps leave one or more narrow openings that act as gates.

    Rule:
    - Determine whether the orange sprite moves horizontally or vertically by the shared span of the source and destination boxes.
    - For each corridor cross-section containing black cells on both outer sides and a non-black opening in the middle, count one gate and recolor the opening cells red.
    - Let G be the number of gates and rotate the orange sprite clockwise by (G mod 4) quarter-turns about its 3x3 source box.
    - Erase the original orange sprite and the maroon destination outline.
    - Place the rotated orange sprite centered on the destination box while preserving the black obstacles and the new red gate marks.

    Key insight:
    - The black gate caps serve two roles at once: their openings become visible red markers, and their count determines the sprite's final rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    src_mask = _single_component_mask(g, ORANGE)
    dst_mask = _single_component_mask(g, MAROON)
    src_box = _bbox(src_mask)
    dst_box = _bbox(dst_mask)

    axis, _direction, corridor_steps, span = _movement_axis(src_box, dst_box)
    gate_count, openings = _find_gates(g, axis, corridor_steps, span)
    assert gate_count >= 1

    out = g.copy()
    out[src_mask] = BG
    out[dst_mask] = BG
    for r, c in openings:
        out[r, c] = RED

    src_patch = _crop(g, src_box)
    rotated = _rotate_cw(src_patch, gate_count % 4)
    mask = rotated == ORANGE

    dr0, dc0, dr1, dc1 = dst_box
    center_r = (dr0 + dr1) // 2
    center_c = (dc0 + dc1) // 2
    h, w = rotated.shape
    place_r = center_r - (h - 1) // 2
    place_c = center_c - (w - 1) // 2
    out[place_r : place_r + h, place_c : place_c + w][mask] = ORANGE
    return out


def _pick_gate_positions(corridor_len: int, gate_count: int) -> list[int]:
    candidates = []
    for combo in itertools.combinations(range(1, corridor_len - 1), gate_count):
        if all(b - a >= 2 for a, b in zip(combo, combo[1:])):
            candidates.append(combo)
    assert candidates
    return list(random.choice(candidates))


def _build_case() -> tuple[np.ndarray, dict]:
    source_shape = random.choice(SHAPES)
    source_rot = random.randrange(4)
    source_mask = np.rot90(source_shape, -source_rot)
    gate_count = random.randint(1, 4)
    corridor_len = random.randint(2 * gate_count + 1, 2 * gate_count + 4)
    gate_positions = _pick_gate_positions(corridor_len, gate_count)
    orientation = random.choice(("horizontal", "vertical"))

    if orientation == "horizontal":
        h = random.randint(11, 18)
        total_w = 3 + corridor_len + 3
        w = random.randint(total_w + 2, min(30, total_w + 7))
        row0 = random.randint(1, h - 4)
        left_anchor = random.randint(1, w - total_w - 1)
        source_is_left = random.choice((True, False))
        if source_is_left:
            src_r, src_c = row0, left_anchor
            dst_r, dst_c = row0, src_c + 3 + corridor_len
        else:
            dst_r, dst_c = row0, left_anchor
            src_r, src_c = row0, dst_c + 3 + corridor_len
        corridor_origin = min(src_c, dst_c) + 3
    else:
        w = random.randint(11, 18)
        total_h = 3 + corridor_len + 3
        h = random.randint(total_h + 2, min(30, total_h + 7))
        col0 = random.randint(1, w - 4)
        top_anchor = random.randint(1, h - total_h - 1)
        source_is_top = random.choice((True, False))
        if source_is_top:
            src_r, src_c = top_anchor, col0
            dst_r, dst_c = src_r + 3 + corridor_len, col0
        else:
            dst_r, dst_c = top_anchor, col0
            src_r, src_c = dst_r + 3 + corridor_len, col0
        corridor_origin = min(src_r, dst_r) + 3

    grid = np.full((h, w), BG, dtype=int)
    grid[src_r : src_r + 3, src_c : src_c + 3][source_mask] = ORANGE
    grid[dst_r : dst_r + 3, dst_c : dst_c + 3] = TARGET_OUTLINE

    gate_cells: list[tuple[int, int]] = []
    if orientation == "horizontal":
        for pos in gate_positions:
            c = corridor_origin + pos
            grid[src_r, c] = BLACK
            grid[src_r + 2, c] = BLACK
            gate_cells.append((src_r + 1, c))
    else:
        for pos in gate_positions:
            r = corridor_origin + pos
            grid[r, src_c] = BLACK
            grid[r, src_c + 2] = BLACK
            gate_cells.append((r, src_c + 1))

    return grid, {
        "source_box": (src_r, src_c, src_r + 2, src_c + 2),
        "target_box": (dst_r, dst_c, dst_r + 2, dst_c + 2),
        "source_mask": source_mask.copy(),
        "gate_count": gate_count,
        "gate_cells": gate_cells,
    }


def _validate_case(grid: np.ndarray, spec: dict) -> bool:
    out = main(grid)
    src_r0, src_c0, src_r1, src_c1 = spec["source_box"]
    dst_r0, dst_c0, dst_r1, dst_c1 = spec["target_box"]
    gate_count = int(spec["gate_count"])

    if out.shape != grid.shape or out.ndim != 2 or max(out.shape) > 30:
        return False
    if np.any(out[src_r0 : src_r1 + 1, src_c0 : src_c1 + 1] == ORANGE):
        return False
    for r, c in spec["gate_cells"]:
        if int(out[r, c]) != RED:
            return False

    expected_patch = _shape_from_mask(np.rot90(spec["source_mask"], -gate_count % 4), ORANGE)
    placed_patch = out[dst_r0 : dst_r1 + 1, dst_c0 : dst_c1 + 1]
    if not np.array_equal(placed_patch, expected_patch):
        return False

    wrong_turns = [k for k in range(4) if k != gate_count % 4]
    if not all(
        not np.array_equal(
            placed_patch,
            _shape_from_mask(np.rot90(spec["source_mask"], -k), ORANGE),
        )
        for k in wrong_turns
    ):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid, spec = _build_case()
        if _validate_case(grid, spec):
            return grid
    raise RuntimeError("failed to generate a valid gated-corridor puzzle")
