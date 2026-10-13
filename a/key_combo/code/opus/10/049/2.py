from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIVIDER = 1
WALL = 5
OUTPUT = 6
PANEL_H = 12
PANEL_W = 12
DIVIDER_W = 1
INPUT_SHAPE = (PANEL_H, PANEL_W * 2 + DIVIDER_W)
OUTPUT_SHAPE = (PANEL_H - 1, PANEL_W)
PALETTE = (2, 3, 4, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 600


def _moving_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BG) & (grid != DIVIDER) & (grid != WALL)


def _find_divider_column(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(candidates) == 1, "expected exactly one full-height blue divider"
    return int(candidates[0])


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    sep = _find_divider_column(g)
    assert sep * 2 + 1 == g.shape[1], "divider must split two equal-width panels"
    left = g[:, :sep].copy()
    right = g[:, sep + 1 :].copy()

    for panel in (left, right):
        assert panel.shape[0] >= 3 and panel.shape[1] >= 2, "panel is too small"
        assert np.all(panel[-1, :] == WALL), "each panel must end with a full grey wall"
        assert not np.any(panel[:-1, :] == WALL), "grey may only appear in the bottom wall"
        assert not np.any(panel == DIVIDER), "divider color must not appear inside panels"
    return left, right


def _find_dominoes(panel: np.ndarray) -> list[dict]:
    h, w = panel.shape
    visited = np.zeros((h, w), dtype=bool)
    dominoes: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color in (BG, DIVIDER, WALL) or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(panel[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            coords = np.array(cells, dtype=int)
            assert coords.shape[0] == 2, "every moving object must be a domino"
            (r0, c0), (r1, c1) = coords
            assert abs(int(r0) - int(r1)) + abs(int(c0) - int(c1)) == 1, "domino must be 4-connected"
            dominoes.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (
                        int(coords[:, 0].min()),
                        int(coords[:, 1].min()),
                        int(coords[:, 0].max()),
                        int(coords[:, 1].max()),
                    ),
                }
            )

    colors = [domino["color"] for domino in dominoes]
    assert len(colors) == len(set(colors)), "each panel must use distinct domino colors"
    return dominoes


def _settle_key(domino: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = domino["bbox"]
    return (-int(r1), int(c0), int(r0), int(c1))


def _settle_panel(panel: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    g = np.asarray(panel, dtype=int)
    h, w = g.shape
    assert np.all(g[-1, :] == WALL), "panel must contain a bottom wall"

    dominoes = _find_dominoes(g)
    out = np.full((h, w), BG, dtype=int)
    out[-1, :] = WALL
    occupied = np.zeros((h, w), dtype=bool)
    occupied[-1, :] = True
    final_by_color: dict[int, np.ndarray] = {}

    for domino in sorted(dominoes, key=_settle_key):
        coords = domino["coords"].copy()
        while True:
            nxt = coords + np.array([1, 0], dtype=int)
            if np.any(nxt[:, 0] >= h) or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt

        color = int(domino["color"])
        assert not np.any(occupied[coords[:, 0], coords[:, 1]]), "settled domino collision"
        out[coords[:, 0], coords[:, 1]] = color
        occupied[coords[:, 0], coords[:, 1]] = True
        final_by_color[color] = coords.copy()

    return out, final_by_color


def _union_from_settled(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    mask = _moving_mask(left[:-1, :]) | _moving_mask(right[:-1, :])
    out = np.full(mask.shape, BG, dtype=int)
    out[mask] = OUTPUT
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the dominoes inside each side panel, then union the two settled masks into one magenta panel.

    Input structure:
    - The grid contains two equal-width black panels separated by one full-height blue divider column.
    - Each panel has a solid grey bottom wall and several isolated monochromatic dominoes above it.
    - Every moving object is exactly two 4-connected cells, and the divider and wall colors appear only in their structural roles.

    Rule:
    - Process the left and right panels independently.
    - In each panel, settle dominoes downward toward the grey wall: nearer dominoes settle first, and equal-distance ties are resolved from left to right.
    - After both panels settle, remove the divider and wall row, align the two panels cell by cell, and paint magenta wherever either settled panel has foreground; otherwise paint black.

    Key insight:
    - The boolean union is taken after the per-panel gravity simulation, so the output depends on the physically settled layouts rather than the original scattered masks.
    """
    left, right = _split_panels(np.asarray(grid, dtype=int))
    left_settled, _ = _settle_panel(left)
    right_settled, _ = _settle_panel(right)
    return _union_from_settled(left_settled, right_settled)


def _mark_blocked(blocked: np.ndarray, cells: tuple[tuple[int, int], tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr, dc in ((0, 0), *DIR4):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True


def _candidate_cells(
    blocked: np.ndarray,
    orient: str,
    col_range: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = blocked.shape
    c0, c1 = col_range
    c0 = max(0, int(c0))
    c1 = min(w - 1, int(c1))
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if orient == "h":
        for r in range(0, h - 2):
            for c in range(c0, c1):
                cells = ((r, c), (r, c + 1))
                if all(not blocked[rr, cc] for rr, cc in cells):
                    candidates.append(cells)
    else:
        for r in range(0, h - 3):
            for c in range(c0, c1 + 1):
                cells = ((r, c), (r + 1, c))
                if all(not blocked[rr, cc] for rr, cc in cells):
                    candidates.append(cells)

    random.shuffle(candidates)
    return candidates


def _choose_candidate(
    candidates: list[tuple[tuple[int, int], tuple[int, int]]],
    placed: list[dict],
    *,
    prefer_tie: bool,
    prefer_stack: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if not candidates:
        return None

    best = None
    best_score = None
    for cells in candidates:
        rows = [r for r, _ in cells]
        cols = {c for _, c in cells}
        bottom = max(rows)
        score = random.random()
        score += 0.25 * (PANEL_H - 2 - bottom)
        if prefer_tie and any(bottom == item["bottom"] for item in placed):
            score += 2.0
        if prefer_stack and any(cols & item["cols"] for item in placed):
            score += 2.5
        if best_score is None or score > best_score:
            best_score = score
            best = cells
    return best


def _render_panel(dominoes: list[dict]) -> np.ndarray:
    panel = np.full((PANEL_H, PANEL_W), BG, dtype=int)
    panel[-1, :] = WALL
    for domino in dominoes:
        coords = np.asarray(domino["coords"], dtype=int)
        panel[coords[:, 0], coords[:, 1]] = int(domino["color"])
    return panel


def _panel_stats(panel: np.ndarray) -> dict:
    settled, final_by_color = _settle_panel(panel)
    initial = {domino["color"]: domino["coords"].copy() for domino in _find_dominoes(panel)}
    moving = _moving_mask(settled)
    moved = 0
    stacked = False

    for color, start_coords in initial.items():
        final_coords = final_by_color[color]
        if set(map(tuple, start_coords)) != set(map(tuple, final_coords)):
            moved += 1

        other_mask = moving.copy()
        other_mask[final_coords[:, 0], final_coords[:, 1]] = False
        nxt = final_coords + np.array([1, 0], dtype=int)
        in_bounds = nxt[:, 0] < panel.shape[0]
        if np.any(in_bounds) and np.any(other_mask[nxt[in_bounds, 0], nxt[in_bounds, 1]]):
            stacked = True

    bottoms = [int(domino["coords"][:, 0].max()) for domino in _find_dominoes(panel)]
    tie = len(bottoms) != len(set(bottoms))
    return {
        "settled": settled,
        "moved": moved,
        "stacked": stacked,
        "tie": tie,
    }


def _generate_panel(
    count: int,
    col_range: tuple[int, int],
    *,
    prefer_tie: bool,
    prefer_stack: bool,
) -> np.ndarray | None:
    colors = random.sample(PALETTE, count)
    orientations = ["h", "v"] + [random.choice(("h", "v")) for _ in range(count - 2)]
    random.shuffle(orientations)

    blocked = np.zeros((PANEL_H, PANEL_W), dtype=bool)
    blocked[-2:, :] = True
    placed: list[dict] = []

    for color, orient in zip(colors, orientations):
        candidates = _candidate_cells(blocked, orient, col_range)
        cells = _choose_candidate(candidates, placed, prefer_tie=prefer_tie, prefer_stack=prefer_stack)
        if cells is None:
            return None
        _mark_blocked(blocked, cells)
        rows = [r for r, _ in cells]
        cols = {c for _, c in cells}
        placed.append({"color": color, "coords": np.array(cells, dtype=int), "bottom": max(rows), "cols": cols})

    return _render_panel(placed)


def _assemble_grid(left_panel: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
    grid = np.full(INPUT_SHAPE, BG, dtype=int)
    grid[:, :PANEL_W] = left_panel
    grid[:, PANEL_W] = DIVIDER
    grid[:, PANEL_W + 1 :] = right_panel
    return grid


def _semantic_check(grid: np.ndarray, *, want_overlap: bool, want_tie: bool) -> bool:
    left, right = _split_panels(grid)
    left_stats = _panel_stats(left)
    right_stats = _panel_stats(right)

    raw_union = _moving_mask(left[:-1, :]) | _moving_mask(right[:-1, :])
    left_final = _moving_mask(left_stats["settled"][:-1, :])
    right_final = _moving_mask(right_stats["settled"][:-1, :])
    overlap = int(np.count_nonzero(left_final & right_final))
    output = main(grid)

    if output.shape != OUTPUT_SHAPE:
        return False
    if np.array_equal(raw_union, output == OUTPUT):
        return False
    if not np.array_equal(output, _union_from_settled(left_stats["settled"], right_stats["settled"])):
        return False
    if left_stats["moved"] < 1 or right_stats["moved"] < 1:
        return False
    if left_stats["moved"] + right_stats["moved"] < 3:
        return False
    if not (left_stats["stacked"] or right_stats["stacked"]):
        return False
    if not np.any(left_final & ~right_final):
        return False
    if not np.any(right_final & ~left_final):
        return False
    if want_overlap != (overlap > 0):
        return False
    if want_tie and not (left_stats["tie"] or right_stats["tie"]):
        return False
    return True


def _generation_ranges(want_overlap: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    if want_overlap:
        start = random.randint(1, 4)
        span = random.randint(6, 7)
        end = min(PANEL_W - 1, start + span - 1)
        shared = (start, end)
        return shared, shared

    if random.random() < 0.5:
        return (0, 5), (6, 11)
    return (6, 11), (0, 5)


def _gen_attempt() -> np.ndarray | None:
    want_overlap = random.random() < 0.5
    want_tie = random.random() < 0.6
    left_range, right_range = _generation_ranges(want_overlap)
    left_count = random.randint(4, 6)
    right_count = random.randint(4, 6)
    tie_side = random.choice(("left", "right"))
    stack_side = random.choice(("left", "right"))

    left = _generate_panel(
        left_count,
        left_range,
        prefer_tie=want_tie and tie_side == "left",
        prefer_stack=stack_side == "left",
    )
    if left is None:
        return None

    right = _generate_panel(
        right_count,
        right_range,
        prefer_tie=want_tie and tie_side == "right",
        prefer_stack=stack_side == "right",
    )
    if right is None:
        return None

    grid = _assemble_grid(left, right)
    if not _semantic_check(grid, want_overlap=want_overlap, want_tie=want_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-panel gravity-union input")
