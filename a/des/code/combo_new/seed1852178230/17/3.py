from __future__ import annotations

import numpy as np

MAROON = 9
AZURE = 8
STRIPE_COLORS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
SHIFT_VALUES = (1, 2, 3)
MAX_GEN_ATTEMPTS = 300


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    neighbors8 = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
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
                for dr, dc in neighbors8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _mode_value(values: np.ndarray) -> int:
    uniq, counts = np.unique(values.astype(int), return_counts=True)
    return int(uniq[np.argmax(counts)])


def _uniform_count(values: np.ndarray) -> bool:
    return bool(values.size > 0 and np.all(values == values[0]))


def _classify_arrow(cells: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(cells) != 3:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    rel = frozenset((r - r0, c - c0) for r, c in cells)
    patterns = {
        frozenset({(0, 1), (1, 0), (1, 2)}): ("down", (0, 1)),
        frozenset({(0, 0), (0, 2), (1, 1)}): ("up", (1, 1)),
        frozenset({(0, 1), (1, 0), (2, 1)}): ("right", (1, 0)),
        frozenset({(0, 0), (1, 1), (2, 0)}): ("left", (1, 1)),
    }
    if rel not in patterns:
        return None
    direction, tip_rel = patterns[rel]
    tip = (r0 + tip_rel[0], c0 + tip_rel[1])
    return {
        "direction": direction,
        "tip": tip,
        "cells": tuple(sorted((int(r), int(c)) for r, c in cells)),
    }


def _detect_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _components_of_color(grid, MAROON):
        arrow = _classify_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
    arrows.sort(key=lambda arrow: (str(arrow["direction"]), tuple(arrow["tip"])))
    return arrows


def _infer_orientation(grid: np.ndarray, marker_mask: np.ndarray) -> str:
    row_score = 0
    for r in range(grid.shape[0]):
        visible = grid[r, ~marker_mask[r]]
        row_score += int(_uniform_count(visible))

    col_score = 0
    for c in range(grid.shape[1]):
        visible = grid[~marker_mask[:, c], c]
        col_score += int(_uniform_count(visible))

    if row_score > col_score:
        return "horizontal"
    if col_score > row_score:
        return "vertical"
    raise AssertionError("failed to infer stripe orientation")


def _restore_base(grid: np.ndarray, marker_mask: np.ndarray, orientation: str) -> np.ndarray:
    h, w = grid.shape
    out = np.empty_like(grid)
    if orientation == "horizontal":
        for r in range(h):
            visible = grid[r, ~marker_mask[r]]
            color = _mode_value(visible) if visible.size else _mode_value(grid[r])
            out[r, :] = color
        return out

    for c in range(w):
        visible = grid[~marker_mask[:, c], c]
        color = _mode_value(visible) if visible.size else _mode_value(grid[:, c])
        out[:, c] = color
    return out


def _shift_from_marker(
    arrow: dict[str, object],
    azure_cells: set[tuple[int, int]],
) -> int:
    tip_r, tip_c = arrow["tip"]
    direction = str(arrow["direction"])
    distances: list[int] = []
    if direction in ("down", "up"):
        for r, c in azure_cells:
            if r == tip_r:
                d = abs(c - tip_c)
                if 1 <= d <= 3:
                    distances.append(d)
    else:
        for r, c in azure_cells:
            if c == tip_c:
                d = abs(r - tip_r)
                if 1 <= d <= 3:
                    distances.append(d)
    if not distances:
        raise AssertionError("each arrow needs one nearby azure marker")
    return int(min(distances))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cyclically shift selected stripe lines by reading edge arrows and their nearby distance markers.

    Input structure:
    - The whole grid is filled with either horizontal or vertical color stripes; there is no shared background.
    - Maroon three-pixel arrowheads sit on the grid edge and point inward.
    - Each arrow has one nearby azure edge pixel whose offset encodes a shift amount from 1 to 3.

    Rule:
    - Detect whether the stripes are horizontal (uniform rows) or vertical (uniform columns).
    - Remove the maroon and azure markers conceptually, restoring the covered stripe colors.
    - For every arrow, read its direction and the row or column it addresses, then cyclically shift that full row or column by the encoded amount.
    - All parameters are read from the input before applying the shifts; the output contains only the shifted stripe pattern.

    Key insight:
    - The visible markers are not part of the pattern itself: they temporarily overwrite a globally striped canvas and encode which stripe line to rotate, in which direction, and by how much.
    """
    g = np.asarray(grid, dtype=int)
    arrows = _detect_arrows(g)
    if not arrows:
        return g.copy()

    marker_mask = (g == MAROON) | (g == AZURE)
    orientation = _infer_orientation(g, marker_mask)
    base = _restore_base(g, marker_mask, orientation)
    azure_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == AZURE)}

    out = base.copy()
    if orientation == "horizontal":
        col_shifts: dict[int, int] = {}
        for arrow in arrows:
            direction = str(arrow["direction"])
            if direction not in ("down", "up"):
                continue
            d = _shift_from_marker(arrow, azure_cells)
            col = int(arrow["tip"][1])
            signed = d if direction == "down" else -d
            col_shifts[col] = col_shifts.get(col, 0) + signed
        for col, shift in col_shifts.items():
            out[:, col] = np.roll(base[:, col], shift)
        return out

    row_shifts: dict[int, int] = {}
    for arrow in arrows:
        direction = str(arrow["direction"])
        if direction not in ("right", "left"):
            continue
        d = _shift_from_marker(arrow, azure_cells)
        row = int(arrow["tip"][0])
        signed = d if direction == "right" else -d
        row_shifts[row] = row_shifts.get(row, 0) + signed
    for row, shift in row_shifts.items():
        out[row, :] = np.roll(base[row, :], shift)
    return out


def _build_stripes(
    rng: np.random.Generator,
    height: int,
    width: int,
    orientation: str,
) -> np.ndarray:
    cycle_len = int(rng.integers(4, 6))
    colors = rng.choice(STRIPE_COLORS, size=cycle_len, replace=False)
    phase = int(rng.integers(0, cycle_len))
    if orientation == "horizontal":
        row_colors = np.array([colors[(phase + r) % cycle_len] for r in range(height)], dtype=int)
        return np.repeat(row_colors[:, None], width, axis=1)

    col_colors = np.array([colors[(phase + c) % cycle_len] for c in range(width)], dtype=int)
    return np.repeat(col_colors[None, :], height, axis=0)


def _collect_candidates(
    length: int,
    sides: tuple[str, str],
    reserved: dict[str, set[int]],
    used_targets: set[int],
) -> list[tuple[str, int, int, int]]:
    candidates: list[tuple[str, int, int, int]] = []
    for side in sides:
        for pos in range(1, length - 1):
            if pos in used_targets:
                continue
            neighborhood = range(max(0, pos - 3), min(length, pos + 4))
            if any(edge_pos in reserved[side] for edge_pos in neighborhood):
                continue
            for distance in SHIFT_VALUES:
                for sign in (-1, 1):
                    marker_pos = pos + sign * distance
                    if 0 <= marker_pos < length:
                        candidates.append((side, pos, distance, sign))
    return candidates


def _place_arrows(
    rng: np.random.Generator,
    height: int,
    width: int,
    orientation: str,
) -> list[dict[str, int | str]]:
    if orientation == "horizontal":
        length = width
        sides = ("top", "bottom")
        max_total = min(4, 2 * ((width + 1) // 4))
    else:
        length = height
        sides = ("left", "right")
        max_total = min(4, 2 * ((height + 1) // 4))

    target_count = int(rng.integers(1, max_total + 1))
    reserved = {side: set() for side in sides}
    used_targets: set[int] = set()
    arrows: list[dict[str, int | str]] = []

    while len(arrows) < target_count:
        candidates = _collect_candidates(length, sides, reserved, used_targets)
        if not candidates:
            break
        distances = sorted({cand[2] for cand in candidates})
        distance = int(distances[int(rng.integers(0, len(distances)))])
        distance_candidates = [cand for cand in candidates if cand[2] == distance]
        side, pos, distance, sign = distance_candidates[int(rng.integers(0, len(distance_candidates)))]
        arrows.append(
            {
                "side": side,
                "pos": int(pos),
                "distance": int(distance),
                "sign": int(sign),
            }
        )
        used_targets.add(int(pos))
        for edge_pos in range(max(0, pos - 3), min(length, pos + 4)):
            reserved[side].add(int(edge_pos))

    if not arrows:
        raise AssertionError("generator failed to place any arrows")
    return arrows


def _draw_arrow(grid: np.ndarray, arrow: dict[str, int | str]) -> None:
    h, w = grid.shape
    side = str(arrow["side"])
    pos = int(arrow["pos"])
    distance = int(arrow["distance"])
    sign = int(arrow["sign"])

    if side == "top":
        cells = [(0, pos), (1, pos - 1), (1, pos + 1)]
        marker = (0, pos + sign * distance)
    elif side == "bottom":
        cells = [(h - 1, pos), (h - 2, pos - 1), (h - 2, pos + 1)]
        marker = (h - 1, pos + sign * distance)
    elif side == "left":
        cells = [(pos, 0), (pos - 1, 1), (pos + 1, 1)]
        marker = (pos + sign * distance, 0)
    else:
        cells = [(pos, w - 1), (pos - 1, w - 2), (pos + 1, w - 2)]
        marker = (pos + sign * distance, w - 1)

    for r, c in cells:
        grid[r, c] = MAROON
    grid[marker] = AZURE


def _gen_attempt(rng: np.random.Generator) -> np.ndarray:
    height = int(rng.integers(8, 19))
    width = int(rng.integers(8, 19))
    orientation = str(rng.choice(["horizontal", "vertical"]))
    base = _build_stripes(rng, height, width, orientation)
    arrows = _place_arrows(rng, height, width, orientation)
    grid = base.copy()
    for arrow in arrows:
        _draw_arrow(grid, arrow)

    out = main(grid)
    if np.array_equal(out, _restore_base(grid, (grid == MAROON) | (grid == AZURE), orientation)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-shift input")
