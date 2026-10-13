from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

_RNG = np.random.default_rng()

_HEIGHT_RANGE = (9, 14)
_PANEL_WIDTH_RANGE = (4, 7)
_COMPONENT_COUNT_OPTIONS = (2, 3)
_COMPONENT_AREA_RANGE = (3, 9)
_MAX_COMPONENT_BBOX = 4
_MAX_PANEL_RETRIES = 120
_MAX_GRID_RETRIES = 600


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the right panel by 180 degrees, then encode left-only/right-only/intersection cells with three output colors.

    Input structure:
    - The grid has odd width and a solid gray separator column in the middle.
    - The separator splits the canvas into equal-size left and right panels.
    - The left panel uses black background with blue foreground cells, and the right panel uses black background with green foreground cells.

    Rule:
    - Extract the right panel and rotate it 180 degrees within its own panel coordinates.
    - Align the rotated right panel with the left panel cell-by-cell.
    - Output a single-panel grid where left-only foreground becomes red, right-only foreground becomes yellow, overlap becomes orange, and double-background stays black.

    Key insight:
    - The output colors do not preserve input colors; they encode the set relationship created only after the right panel is half-turned into alignment.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    height, width = map(int, g.shape)
    if width < 3 or width % 2 != 1:
        raise AssertionError("expected odd grid width with a separator column")

    mid = width // 2
    if not np.all(g[:, mid] == GREY):
        raise AssertionError("expected a solid gray separator column")

    left = g[:, :mid]
    right = g[:, mid + 1 :]
    if left.shape != right.shape:
        raise AssertionError("left and right panels must match")

    left_fg = left == BLUE
    right_fg = right == GREEN
    rotated_right_fg = np.rot90(right_fg, 2)

    out = np.zeros(left.shape, dtype=int)
    out[left_fg & ~rotated_right_fg] = RED
    out[~left_fg & rotated_right_fg] = YELLOW
    out[left_fg & rotated_right_fg] = ORANGE
    return out


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_of_coords(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _random_component_patch(max_h: int, max_w: int) -> np.ndarray:
    dims: list[tuple[int, int, int]] = []
    h_cap = min(_MAX_COMPONENT_BBOX, max_h)
    w_cap = min(_MAX_COMPONENT_BBOX, max_w)
    for h in range(2, h_cap + 1):
        for w in range(2, w_cap + 1):
            for area in range(_COMPONENT_AREA_RANGE[0], _COMPONENT_AREA_RANGE[1]):
                if area > h * w or area == h * w:
                    continue
                dims.append((h, w, area))
    if not dims:
        raise RuntimeError("panel too small for component generation")

    for _ in range(200):
        h, w, area = dims[int(_RNG.integers(len(dims)))]
        patch = np.zeros((h, w), dtype=bool)
        seed = (int(_RNG.integers(h)), int(_RNG.integers(w)))
        patch[seed] = True

        while int(patch.sum()) < area:
            frontier: list[tuple[int, int]] = []
            used: set[tuple[int, int]] = set()
            for r, c in np.argwhere(patch):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not patch[nr, nc] and (nr, nc) not in used:
                        used.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick_r, pick_c = frontier[int(_RNG.integers(len(frontier)))]
            patch[pick_r, pick_c] = True

        coords = [tuple(map(int, rc)) for rc in np.argwhere(patch)]
        if len(coords) != area:
            continue
        if len(_find_components(patch)) != 1:
            continue
        bbox_h, bbox_w = _bbox_of_coords(coords)
        if bbox_h < 2 or bbox_w < 2:
            continue
        if area == bbox_h * bbox_w:
            continue
        return patch

    raise RuntimeError("failed to sample an irregular component patch")


def _valid_placements(panel_mask: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = panel_mask.shape
    ph, pw = patch.shape
    blocked = _dilate4(panel_mask)
    placements: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            window = blocked[r0 : r0 + ph, c0 : c0 + pw]
            if not np.any(window & patch):
                placements.append((r0, c0))
    return placements


def _build_panel_mask(height: int, width: int, component_count: int) -> np.ndarray:
    for _ in range(_MAX_PANEL_RETRIES):
        mask = np.zeros((height, width), dtype=bool)
        ok = True
        for _ in range(component_count):
            placed = False
            for _ in range(120):
                patch = _random_component_patch(height, width)
                placements = _valid_placements(mask, patch)
                if not placements:
                    continue
                r0, c0 = placements[int(_RNG.integers(len(placements)))]
                ph, pw = patch.shape
                mask[r0 : r0 + ph, c0 : c0 + pw] |= patch
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        comps = _find_components(mask)
        if len(comps) != component_count:
            continue
        if any(not (_COMPONENT_AREA_RANGE[0] <= len(comp) < _COMPONENT_AREA_RANGE[1]) for comp in comps):
            continue
        return mask

    raise RuntimeError("failed to build a panel mask")


def _render_input(left_mask: np.ndarray, right_input_mask: np.ndarray) -> np.ndarray:
    h, w = left_mask.shape
    grid = np.zeros((h, 2 * w + 1), dtype=int)
    grid[:, w] = GREY
    grid[:, :w][left_mask] = BLUE
    grid[:, w + 1 :][right_input_mask] = GREEN
    return grid


def _overlay_from_masks(left_mask: np.ndarray, transformed_right_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(left_mask.shape, dtype=int)
    out[left_mask & ~transformed_right_mask] = RED
    out[~left_mask & transformed_right_mask] = YELLOW
    out[left_mask & transformed_right_mask] = ORANGE
    return out


def _alternative_outputs(right_input_mask: np.ndarray, left_mask: np.ndarray) -> list[np.ndarray]:
    transforms = [
        right_input_mask,
        np.flipud(right_input_mask),
        np.fliplr(right_input_mask),
    ]
    return [_overlay_from_masks(left_mask, mask) for mask in transforms]


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, total_w = grid.shape
    if h > 30 or total_w > 30 or total_w % 2 != 1:
        return False

    panel_w = total_w // 2
    if panel_w < _PANEL_WIDTH_RANGE[0] or panel_w >= _PANEL_WIDTH_RANGE[1]:
        return False
    if not np.all(grid[:, panel_w] == GREY):
        return False

    left = grid[:, :panel_w]
    right = grid[:, panel_w + 1 :]
    if set(np.unique(left)) - {BLACK, BLUE}:
        return False
    if set(np.unique(right)) - {BLACK, GREEN}:
        return False

    left_mask = left == BLUE
    right_input_mask = right == GREEN
    rotated_right_mask = np.rot90(right_input_mask, 2)

    left_comps = _find_components(left_mask)
    right_comps = _find_components(right_input_mask)
    if len(left_comps) not in _COMPONENT_COUNT_OPTIONS or len(right_comps) not in _COMPONENT_COUNT_OPTIONS:
        return False
    if any(len(comp) < _COMPONENT_AREA_RANGE[0] or len(comp) >= _COMPONENT_AREA_RANGE[1] for comp in left_comps + right_comps):
        return False

    both = left_mask & rotated_right_mask
    left_only = left_mask & ~rotated_right_mask
    right_only = ~left_mask & rotated_right_mask
    if int(both.sum()) < 2 or int(left_only.sum()) < 2 or int(right_only.sum()) < 2:
        return False

    correct = _overlay_from_masks(left_mask, rotated_right_mask)
    if not np.array_equal(correct, main(grid)):
        return False

    for alt in _alternative_outputs(right_input_mask, left_mask):
        if np.array_equal(alt, correct):
            return False

    if len(set(np.unique(correct)) - {BLACK, RED, YELLOW, ORANGE}) != 0:
        return False
    if RED not in correct or YELLOW not in correct or ORANGE not in correct:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(*_HEIGHT_RANGE))
    panel_w = int(_RNG.integers(*_PANEL_WIDTH_RANGE))
    left_count = int(_RNG.choice(_COMPONENT_COUNT_OPTIONS))
    right_count = int(_RNG.choice(_COMPONENT_COUNT_OPTIONS))

    try:
        left_mask = _build_panel_mask(height, panel_w, left_count)
        rotated_right_mask = _build_panel_mask(height, panel_w, right_count)
    except RuntimeError:
        return None

    if int(left_mask.sum()) + int(rotated_right_mask.sum()) > (height * panel_w) // 2:
        return None

    right_input_mask = np.rot90(rotated_right_mask, 2)
    grid = _render_input(left_mask, right_input_mask)
    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GRID_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
