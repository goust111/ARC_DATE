from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "new" / "solvers"
    if _candidate.is_dir():
        candidate_str = str(_candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        break

from arc_utils import Color, bbox_coords, components_mask, crop_bbox


PANEL_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE]
DIR_TO_TURNS = {"up": 0, "right": 3, "down": 2, "left": 1}
OTHER_MARKER_COUNTS = [0, 2, 3]
PANEL_SIDE_RANGE = (7, 10)
GEN_ATTEMPTS = 400


def _find_dividers(grid: np.ndarray) -> tuple[int, int]:
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == Color.GREY)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == Color.GREY)]
    assert len(full_rows) == 1 and len(full_cols) == 1, "expected one grey row and one grey column divider"
    return int(full_rows[0]), int(full_cols[0])


def _panel_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mid_r, mid_c = _find_dividers(grid)
    return [
        (0, mid_r - 1, 0, mid_c - 1),
        (0, mid_r - 1, mid_c + 1, grid.shape[1] - 1),
        (mid_r + 1, grid.shape[0] - 1, 0, mid_c - 1),
        (mid_r + 1, grid.shape[0] - 1, mid_c + 1, grid.shape[1] - 1),
    ]


def _crop_component(panel: np.ndarray, coords: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(panel, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return crop_bbox(panel, bbox_coords(coords))


def _largest_sprite_component(panel: np.ndarray) -> np.ndarray:
    mask = (panel != Color.BLACK) & (panel != Color.RED)
    comps = components_mask(mask, connectivity=4)
    assert comps, "selected panel must contain a sprite"
    areas = [int(coords.shape[0]) for coords in comps]
    largest = max(areas)
    winners = [coords for coords in comps if int(coords.shape[0]) == largest]
    assert len(winners) == 1, "largest sprite component must be unique"
    return winners[0]


def _direction_from_marker(mr: int, mc: int, sr: int, sc: int) -> str:
    dr = int(mr) - int(sr)
    dc = int(mc) - int(sc)
    assert dr != 0 and dc != 0, "marker must not share row or column with sprite center"
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the uniquely singly-marked panel and rotate its sprite using the marker direction code.

    Input structure:
    - One grey horizontal divider and one grey vertical divider split the grid into four equal panels.
    - Each panel has black background, one colored 4-connected sprite, and 0 to 3 red marker pixels.
    - Exactly one panel contains exactly one red marker; the other panels contain 0, 2, or 3.

    Rule:
    - Split the grid along the grey divider row and column to isolate the four panels.
    - Count red marker pixels in each panel and select the unique panel whose count is exactly 1.
    - In that panel, take the largest 4-connected non-black, non-red component as the sprite and compute its bbox center with floor rounding.
    - Compare the single red marker to that center: vertical dominates when |dr| >= |dc|, giving up/down; otherwise horizontal gives left/right.
    - Rotate the sprite crop by the encoded clockwise angle: up -> 0 degrees, right -> 90 degrees, down -> 180 degrees, left -> 270 degrees.

    Key insight:
    - The red marker has two roles at once: its count picks the source panel, and its position relative to the sprite center encodes the rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    chosen_panel = None
    chosen_marker = None
    for r0, r1, c0, c1 in _panel_boxes(g):
        panel = g[r0 : r1 + 1, c0 : c1 + 1]
        markers = np.argwhere(panel == Color.RED)
        if int(markers.shape[0]) == 1:
            assert chosen_panel is None, "there must be exactly one singly-marked panel"
            chosen_panel = panel
            chosen_marker = tuple(map(int, markers[0]))

    assert chosen_panel is not None and chosen_marker is not None, "no singly-marked panel found"

    sprite_coords = _largest_sprite_component(chosen_panel)
    sprite_box = bbox_coords(sprite_coords)
    sprite = crop_bbox(chosen_panel, sprite_box)
    sr = (sprite_box[0] + sprite_box[2]) // 2
    sc = (sprite_box[1] + sprite_box[3]) // 2
    direction = _direction_from_marker(chosen_marker[0], chosen_marker[1], sr, sc)
    return np.rot90(sprite, DIR_TO_TURNS[direction]).copy()


def _neighbors4_in_box(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _grow_shape(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4_in_box(start[0], start[1], h, w))
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        frontier.update(_neighbors4_in_box(cell[0], cell[1], h, w))
        frontier -= cells
    if len(cells) != area:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _rotationally_distinct(mask: np.ndarray) -> bool:
    base = mask.astype(int)
    for turns in (1, 2, 3):
        rot = np.rot90(base, turns)
        if base.shape == rot.shape and np.array_equal(base, rot):
            return False
    return True


def _random_sprite_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(6, h * w - 1)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _grow_shape(h, w, area)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        crop = crop_bbox(mask.astype(int), bbox_coords(coords)).astype(bool)
        if min(crop.shape) < 2:
            continue
        if int(crop.sum()) == crop.size:
            continue
        if not _rotationally_distinct(crop):
            continue
        return crop
    raise RuntimeError("failed to sample a non-degenerate sprite")


def _dilate4(mask: np.ndarray) -> np.ndarray:
    blocked = mask.copy()
    blocked[1:, :] |= mask[:-1, :]
    blocked[:-1, :] |= mask[1:, :]
    blocked[:, 1:] |= mask[:, :-1]
    blocked[:, :-1] |= mask[:, 1:]
    return blocked


def _too_close(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in chosen)


def _sprite_bbox_center(sprite_mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(sprite_mask)
    box = bbox_coords(coords)
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def _marker_candidates(sprite_mask: np.ndarray) -> list[tuple[int, int]]:
    blocked = _dilate4(sprite_mask)
    cells = np.argwhere(~blocked)
    return [tuple(map(int, cell)) for cell in cells]


def _place_markers(
    sprite_mask: np.ndarray,
    count: int,
    desired_direction: str | None,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    center = _sprite_bbox_center(sprite_mask)
    candidates = _marker_candidates(sprite_mask)
    if desired_direction is not None:
        candidates = [
            (r, c)
            for r, c in candidates
            if r != center[0] and c != center[1] and _direction_from_marker(r, c, center[0], center[1]) == desired_direction
        ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if _too_close(cell, chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen, center
    return None


def _render_panel(
    panel_h: int,
    panel_w: int,
    color: int,
    marker_count: int,
    desired_direction: str | None,
) -> tuple[np.ndarray, dict] | None:
    for _ in range(300):
        sprite = _random_sprite_mask()
        sh, sw = sprite.shape
        placements = [(r0, c0) for r0 in range(panel_h - sh + 1) for c0 in range(panel_w - sw + 1)]
        random.shuffle(placements)
        for r0, c0 in placements:
            sprite_mask = np.zeros((panel_h, panel_w), dtype=bool)
            sprite_mask[r0 : r0 + sh, c0 : c0 + sw] = sprite
            placed = _place_markers(sprite_mask, marker_count, desired_direction)
            if marker_count > 0 and placed is None:
                continue

            panel = np.zeros((panel_h, panel_w), dtype=int)
            panel[sprite_mask] = int(color)
            markers: list[tuple[int, int]] = []
            if placed is not None:
                markers, _center = placed
                for mr, mc in markers:
                    panel[mr, mc] = Color.RED

            sprite_crop = sprite.astype(int) * int(color)
            return panel, {"sprite": sprite_crop, "markers": markers, "center": _sprite_bbox_center(sprite_mask)}
    return None


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(*PANEL_SIDE_RANGE)
    panel_w = random.randint(*PANEL_SIDE_RANGE)
    chosen_panel = random.randrange(4)
    chosen_direction = random.choice(list(DIR_TO_TURNS))
    panel_colors = random.sample(PANEL_COLORS, 4)
    other_counts = OTHER_MARKER_COUNTS[:]
    random.shuffle(other_counts)

    panels: list[np.ndarray] = []
    expected = None
    other_i = 0
    selected_sprite = None
    selected_direction = None
    for idx in range(4):
        marker_count = 1 if idx == chosen_panel else other_counts[other_i]
        if idx != chosen_panel:
            other_i += 1
        desired = chosen_direction if idx == chosen_panel else None
        result = _render_panel(panel_h, panel_w, panel_colors[idx], marker_count, desired)
        if result is None:
            return None
        panel, meta = result
        panels.append(panel)
        if idx == chosen_panel:
            selected_sprite = meta["sprite"]
            marker = meta["markers"][0]
            center = meta["center"]
            selected_direction = _direction_from_marker(marker[0], marker[1], center[0], center[1])
            expected = np.rot90(selected_sprite, DIR_TO_TURNS[selected_direction]).copy()

    assert expected is not None and selected_sprite is not None and selected_direction is not None
    if selected_direction != chosen_direction:
        return None
    if selected_direction != "up" and np.array_equal(expected, selected_sprite):
        return None

    grid = np.full((panel_h * 2 + 1, panel_w * 2 + 1), Color.BLACK, dtype=int)
    grid[panel_h, :] = Color.GREY
    grid[:, panel_w] = Color.GREY
    grid[0:panel_h, 0:panel_w] = panels[0]
    grid[0:panel_h, panel_w + 1 :] = panels[1]
    grid[panel_h + 1 :, 0:panel_w] = panels[2]
    grid[panel_h + 1 :, panel_w + 1 :] = panels[3]

    counts = [int(np.count_nonzero(panel == Color.RED)) for panel in panels]
    if counts.count(1) != 1:
        return None
    if 0 not in counts or max(counts) < 2:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid panel-marker rotation task")
