from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MARKER_COLORS = (BLUE, RED, GREEN, ORANGE)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == YELLOW)]
    assert len(rows) == 1, "expected exactly one solid yellow separator row"
    sep = rows[0]
    top = grid[:sep].copy()
    bottom = grid[sep + 1 :].copy()
    assert top.shape == bottom.shape and top.size > 0, "expected equal non-empty panels"
    return top, bottom


def _winner_from_markers(
    component: list[tuple[int, int]], top: np.ndarray, bottom: np.ndarray
) -> int:
    touches: dict[int, tuple[int, int, int]] = {}
    for panel_idx, panel in enumerate((top, bottom)):
        for r, c in component:
            for nr, nc in neighbors4(r, c):
                if not _in_bounds(panel.shape, nr, nc):
                    continue
                color = int(panel[nr, nc])
                if color not in MARKER_COLORS:
                    continue
                key = (panel_idx, nr, nc)
                if key in touches:
                    continue
                touches[key] = (color, nr, nc)

    assert touches, "every confirmation component must touch at least one marker"

    counts = Counter(color for color, _r, _c in touches.values())
    tied_colors = [color for color, count in counts.items() if count == max(counts.values())]
    if len(tied_colors) == 1:
        return tied_colors[0]

    tied_positions = {
        color: min((r, c, panel_idx) for (panel_idx, r, c), (marker_color, _r, _c) in touches.items() if marker_color == color)
        for color in tied_colors
    }
    return min(tied_colors, key=lambda color: tied_positions[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay two separated panels, keep only their shared grey cells, and recolor each surviving component by adjacent marker majority.

    Input structure:
    - The input grid is split into equal top and bottom panels by a single solid yellow separator row.
    - Each panel uses black background, contains one to three grey 4-connected shapes, and may contain isolated single-cell colored markers.
    - Markers use colors from {blue, red, green, orange}; grey pixels are the only structural shape color.

    Rule:
    - Remove the yellow separator and align the two panels cell by cell.
    - Keep a cell only where both panels contain grey at that coordinate; all other coordinates become black.
    - Split that shared-grey mask into 4-connected components.
    - For each component, collect every marker pixel from either panel that is 4-neighbor adjacent to any cell of the component, count marker colors, and repaint the whole component with the unique majority color.
    - Marker pixels themselves are not drawn in the output.

    Key insight:
    - Geometry is decided by a positional AND of the two grey masks, while color is decided afterward from marker votes gathered across both aligned panels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top, bottom = _find_separator(g)
    confirm = (top == GREY) & (bottom == GREY)
    out = np.full(confirm.shape, BLACK, dtype=int)

    for comp in _components(confirm):
        fill_color = _winner_from_markers(comp, top, bottom)
        for r, c in comp:
            out[r, c] = fill_color
    return out


def _box_dims(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, r1, c0, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _valid_box(box: tuple[int, int, int, int], min_h: int = 4, min_w: int = 4) -> bool:
    h, w = _box_dims(box)
    return h >= min_h and w >= min_w


def _zone_layouts(h: int, w: int, k: int) -> list[list[tuple[int, int, int, int]]]:
    layouts: list[list[tuple[int, int, int, int]]] = []

    if k == 2:
        mid_c = w // 2
        vertical = [(1, h - 2, 1, mid_c - 2), (1, h - 2, mid_c + 1, w - 2)]
        if all(_valid_box(box) for box in vertical):
            layouts.append(vertical)

        mid_r = h // 2
        horizontal = [(1, mid_r - 2, 1, w - 2), (mid_r + 1, h - 2, 1, w - 2)]
        if all(_valid_box(box) for box in horizontal):
            layouts.append(horizontal)

        diagonal = [(1, h // 2 - 1, 1, w // 2 - 1), (h // 2 + 1, h - 2, w // 2 + 1, w - 2)]
        if all(_valid_box(box) for box in diagonal):
            layouts.append(diagonal)
    else:
        third_c1 = 1 + (w - 4) // 3
        third_c2 = third_c1 + 2 + (w - 4) // 3
        vertical3 = [(1, h - 2, 1, third_c1), (1, h - 2, third_c1 + 2, third_c2), (1, h - 2, third_c2 + 2, w - 2)]
        if all(_valid_box(box) for box in vertical3):
            layouts.append(vertical3)

        third_r1 = 1 + (h - 4) // 3
        third_r2 = third_r1 + 2 + (h - 4) // 3
        horizontal3 = [(1, third_r1, 1, w - 2), (third_r1 + 2, third_r2, 1, w - 2), (third_r2 + 2, h - 2, 1, w - 2)]
        if all(_valid_box(box) for box in horizontal3):
            layouts.append(horizontal3)

        top_two_bottom = [
            (1, h // 2 - 1, 1, w // 2 - 2),
            (1, h // 2 - 1, w // 2 + 1, w - 2),
            (h // 2 + 1, h - 2, w // 4, w - w // 4 - 1),
        ]
        if all(_valid_box(box) for box in top_two_bottom):
            layouts.append(top_two_bottom)

    return layouts


def _random_component(
    box: tuple[int, int, int, int], target_area: int, rng: random.Random
) -> set[tuple[int, int]] | None:
    r0, r1, c0, c1 = box
    inner_r0 = r0 + 1 if r1 - r0 + 1 >= 5 else r0
    inner_r1 = r1 - 1 if r1 - r0 + 1 >= 5 else r1
    inner_c0 = c0 + 1 if c1 - c0 + 1 >= 5 else c0
    inner_c1 = c1 - 1 if c1 - c0 + 1 >= 5 else c1
    seed = (rng.randint(inner_r0, inner_r1), rng.randint(inner_c0, inner_c1))
    cells = {seed}
    while len(cells) < target_area:
        frontier = []
        for r, c in list(cells):
            for nr, nc in neighbors4(r, c):
                if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        rng.shuffle(frontier)
        frontier.sort(key=lambda pos: abs(pos[0] - seed[0]) + abs(pos[1] - seed[1]))
        cells.add(frontier[0])
    return cells


def _direction_score(pos: tuple[int, int], center: tuple[float, float], direction: str) -> float:
    r, c = pos
    cr, cc = center
    if direction == "north":
        return cr - r
    if direction == "south":
        return r - cr
    if direction == "west":
        return cc - c
    return c - cc


def _expand_component(
    base: set[tuple[int, int]],
    box: tuple[int, int, int, int],
    target_area: int,
    direction: str,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    cells = set(base)
    center = (
        sum(r for r, _ in base) / len(base),
        sum(c for _, c in base) / len(base),
    )
    r0, r1, c0, c1 = box
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                    continue
                if (nr, nc) in cells or (nr, nc) in forbidden:
                    continue
                frontier.append((nr, nc))
        if not frontier:
            return None
        frontier = list(dict.fromkeys(frontier))
        rng.shuffle(frontier)
        frontier.sort(key=lambda pos: (-_direction_score(pos, center, direction), abs(pos[0] - center[0]) + abs(pos[1] - center[1])))
        cells.add(frontier[0])
    return cells


def _adjacent_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if _in_bounds(shape, nr, nc) and (nr, nc) not in cells:
                out.add((nr, nc))
    return out


def _make_vote_plan(k: int, rng: random.Random) -> list[list[int]]:
    if k == 2:
        winners = rng.sample(MARKER_COLORS, 2 if rng.random() < 0.8 else 1)
        if len(winners) == 1:
            winners = [winners[0], winners[0]]
    else:
        if rng.random() < 0.6:
            a, b = rng.sample(MARKER_COLORS, 2)
            winners = [a, b, a]
        else:
            winners = rng.sample(MARKER_COLORS, 3)

    plans: list[list[int]] = []
    for idx in range(k):
        winner = winners[idx]
        losers = [color for color in MARKER_COLORS if color != winner]
        if idx == 0:
            votes = [winner, winner]
            if rng.random() < 0.75:
                votes.append(rng.choice(losers))
        else:
            votes = [winner, winner]
            if rng.random() < 0.55:
                votes.append(rng.choice(losers))
            if rng.random() < 0.20:
                votes.append(winner)
        plans.append(votes)

    return plans


def _pick_marker_pos(
    candidates: list[tuple[int, int]],
    used: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    pool = list(candidates)
    rng.shuffle(pool)
    for pos in pool:
        if pos in used:
            continue
        if any(abs(pos[0] - r) + abs(pos[1] - c) == 1 for r, c in used):
            continue
        return pos
    return None


def _place_distractor_markers(
    panel: np.ndarray,
    used: set[tuple[int, int]],
    grey_mask: np.ndarray,
    count_needed: int,
    rng: random.Random,
) -> bool:
    h, w = panel.shape
    grey_adjacent = _adjacent_cells(set(map(tuple, np.argwhere(grey_mask))), panel.shape)
    choices = []
    for r in range(h):
        for c in range(w):
            if panel[r, c] != BLACK or (r, c) in grey_adjacent:
                continue
            if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in used):
                continue
            choices.append((r, c))
    rng.shuffle(choices)
    if len(choices) < count_needed:
        return False
    for pos in choices[:count_needed]:
        used.add(pos)
        panel[pos] = rng.choice(MARKER_COLORS)
    return True


def _assemble_attempt(rng: random.Random) -> np.ndarray | None:
    k = 3 if rng.random() < 0.45 else 2
    h = rng.randint(9, 12) if k == 3 else rng.randint(7, 12)
    w = rng.randint(11, 14) if k == 3 else rng.randint(8, 14)
    layouts = _zone_layouts(h, w, k)
    if not layouts:
        return None
    zones = rng.choice(layouts)

    base_components: list[set[tuple[int, int]]] = []
    top_components: list[set[tuple[int, int]]] = []
    bottom_components: list[set[tuple[int, int]]] = []
    cross_top_options: list[list[tuple[int, int]]] = []
    cross_bottom_options: list[list[tuple[int, int]]] = []
    generic_top_options: list[list[tuple[int, int]]] = []
    generic_bottom_options: list[list[tuple[int, int]]] = []
    top_mask = np.zeros((h, w), dtype=bool)
    bottom_mask = np.zeros((h, w), dtype=bool)

    direction_pairs = [("north", "south"), ("south", "north"), ("west", "east"), ("east", "west")]

    for zone in zones:
        zone_h, zone_w = _box_dims(zone)
        max_base = min(5, zone_h * zone_w // 4)
        min_base = 2
        if max_base < min_base:
            return None
        base = _random_component(zone, rng.randint(min_base, max_base), rng)
        if base is None:
            return None

        top_target = rng.randint(max(4, len(base) + 1), min(12, len(base) + 4))
        bottom_target = rng.randint(max(4, len(base) + 1), min(12, len(base) + 4))
        top_dir, bottom_dir = rng.choice(direction_pairs)
        top_comp = _expand_component(base, zone, top_target, top_dir, set(), rng)
        if top_comp is None:
            return None
        top_extra = top_comp - base
        bottom_comp = _expand_component(base, zone, bottom_target, bottom_dir, top_extra, rng)
        if bottom_comp is None:
            return None
        if bottom_comp & top_extra:
            return None

        for r, c in top_comp:
            top_mask[r, c] = True
        for r, c in bottom_comp:
            bottom_mask[r, c] = True

        base_components.append(base)
        top_components.append(top_comp)
        bottom_components.append(bottom_comp)

    confirm_mask = top_mask & bottom_mask
    confirm_components = _components(confirm_mask)
    if len(confirm_components) != k:
        return None

    comp_sets = [set(comp) for comp in confirm_components]
    if sorted(len(comp) for comp in comp_sets) != sorted(len(comp) for comp in base_components):
        return None

    for idx, comp in enumerate(comp_sets):
        all_adj = _adjacent_cells(comp, (h, w))
        generic_top = [pos for pos in all_adj if not top_mask[pos]]
        generic_bottom = [pos for pos in all_adj if not bottom_mask[pos]]
        cross_top = [pos for pos in all_adj if bottom_mask[pos] and not top_mask[pos]]
        cross_bottom = [pos for pos in all_adj if top_mask[pos] and not bottom_mask[pos]]

        if len(generic_top) < 2 or len(generic_bottom) < 2:
            return None
        generic_top_options.append(generic_top)
        generic_bottom_options.append(generic_bottom)
        cross_top_options.append(cross_top)
        cross_bottom_options.append(cross_bottom)

    top = np.full((h, w), BLACK, dtype=int)
    bottom = np.full((h, w), BLACK, dtype=int)
    top[top_mask] = GREY
    bottom[bottom_mask] = GREY

    vote_plans = _make_vote_plan(k, rng)
    top_marker_used: set[tuple[int, int]] = set()
    bottom_marker_used: set[tuple[int, int]] = set()
    top_marker_count = 0
    bottom_marker_count = 0
    cross_panel_ok = False

    for idx, votes in enumerate(vote_plans):
        forced = []
        if idx == 0:
            forced = [("top", votes[0], True), ("bottom", votes[1], True)]
            remaining_votes = votes[2:]
        else:
            remaining_votes = votes

        for panel_name, color, require_cross in forced:
            if panel_name == "top":
                pool = cross_top_options[idx] if require_cross else generic_top_options[idx]
                pos = _pick_marker_pos(pool, top_marker_used, rng)
                if pos is None:
                    return None
                top[pos] = color
                top_marker_used.add(pos)
                top_marker_count += 1
                cross_panel_ok = cross_panel_ok or require_cross
            else:
                pool = cross_bottom_options[idx] if require_cross else generic_bottom_options[idx]
                pos = _pick_marker_pos(pool, bottom_marker_used, rng)
                if pos is None:
                    return None
                bottom[pos] = color
                bottom_marker_used.add(pos)
                bottom_marker_count += 1
                cross_panel_ok = cross_panel_ok or require_cross

        for color in remaining_votes:
            prefer_top = top_marker_count < bottom_marker_count
            panel_choices = ["top", "bottom"] if prefer_top else ["bottom", "top"]
            if rng.random() < 0.35:
                panel_choices.reverse()

            placed = False
            for panel_name in panel_choices:
                if panel_name == "top":
                    pos = _pick_marker_pos(generic_top_options[idx], top_marker_used, rng)
                    if pos is None:
                        continue
                    top[pos] = color
                    top_marker_used.add(pos)
                    top_marker_count += 1
                else:
                    pos = _pick_marker_pos(generic_bottom_options[idx], bottom_marker_used, rng)
                    if pos is None:
                        continue
                    bottom[pos] = color
                    bottom_marker_used.add(pos)
                    bottom_marker_count += 1
                placed = True
                break
            if not placed:
                return None

    if not cross_panel_ok or not cross_top_options[0] or not cross_bottom_options[0]:
        return None
    if top_marker_count > 6 or bottom_marker_count > 6:
        return None

    if top_marker_count < 2 and not _place_distractor_markers(top, top_marker_used, top_mask, 2 - top_marker_count, rng):
        return None
    if bottom_marker_count < 2 and not _place_distractor_markers(bottom, bottom_marker_used, bottom_mask, 2 - bottom_marker_count, rng):
        return None

    top_total = int(np.isin(top, MARKER_COLORS).sum())
    bottom_total = int(np.isin(bottom, MARKER_COLORS).sum())
    if not (2 <= top_total <= 6 and 2 <= bottom_total <= 6):
        return None

    if len(_components(top == GREY)) != k or len(_components(bottom == GREY)) != k:
        return None
    if any(not (4 <= len(comp) <= 12) for comp in _components(top == GREY)):
        return None
    if any(not (4 <= len(comp) <= 12) for comp in _components(bottom == GREY)):
        return None

    grid = np.full((2 * h + 1, w), BLACK, dtype=int)
    grid[:h] = top
    grid[h] = YELLOW
    grid[h + 1 :] = bottom

    out = main(grid)
    out_comps = _components(out != BLACK)
    if len(out_comps) != k:
        return None
    out_colors = {int(out[r, c]) for comp in out_comps for r, c in comp}
    if len(out_colors) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _assemble_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-overlay voting input")
