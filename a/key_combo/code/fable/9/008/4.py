import itertools
import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _support_count(mask: np.ndarray) -> np.ndarray:
    support = np.zeros(mask.shape, dtype=int)
    support[1:, :] += mask[:-1, :]
    support[:-1, :] += mask[1:, :]
    support[:, 1:] += mask[:, :-1]
    support[:, :-1] += mask[:, 1:]
    return support * mask.astype(int)


def _count_components(mask: np.ndarray) -> int:
    mask = mask.astype(bool)
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _dilate4(mask: np.ndarray) -> np.ndarray:
    near = np.zeros(mask.shape, dtype=bool)
    near[1:, :] |= mask[:-1, :]
    near[:-1, :] |= mask[1:, :]
    near[:, 1:] |= mask[:, :-1]
    near[:, :-1] |= mask[:, 1:]
    return near


def _frontier(mask: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    return _dilate4(mask) & allowed & ~mask


def _pick_true(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r, c = coords[random.randrange(len(coords))]
    return int(r), int(c)


def _split_positive(total: int, parts: int) -> list[int] | None:
    if parts <= 0 or total < parts:
        return None
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    out = [cuts[0]]
    out.extend(cuts[i] - cuts[i - 1] for i in range(1, len(cuts)))
    out.append(total - cuts[-1])
    return out


def _grow_mask(shape: tuple[int, int], target: int, max_components: int) -> np.ndarray | None:
    if target <= 0:
        return np.zeros(shape, dtype=bool)

    allowed = np.ones(shape, dtype=bool)
    component_goal = random.randint(1, min(max_components, target))
    sizes = _split_positive(target, component_goal)
    if sizes is None:
        return None

    labels = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for label in range(1, component_goal + 1):
        choices = allowed & ~occupied & ~_dilate4(occupied)
        seed = _pick_true(choices)
        if seed is None:
            return None
        labels[seed] = label
        occupied[seed] = True

    stuck: set[int] = set()
    while int((labels > 0).sum()) < target:
        growable = [
            label
            for label in range(1, component_goal + 1)
            if int((labels == label).sum()) < sizes[label - 1] and label not in stuck
        ]
        if not growable:
            return None
        label = random.choice(growable)
        own = labels == label
        other = (labels > 0) & ~own
        choices = _frontier(own, allowed & (labels == 0)) & ~_dilate4(other)
        picked = _pick_true(choices)
        if picked is None:
            stuck.add(label)
            continue
        labels[picked] = label
        stuck.clear()

    mask = labels > 0
    if _count_components(mask) != component_goal:
        return None

    # Add or remove a few border cells to diversify local support patterns.
    for _ in range(random.randint(0, 3)):
        if random.random() < 0.6:
            choices = _frontier(mask, allowed)
            picked = _pick_true(choices)
            if picked is not None:
                trial = mask.copy()
                trial[picked] = True
                if _count_components(trial) <= max_components:
                    mask = trial
        else:
            removable = []
            for r, c in np.argwhere(mask):
                rr, cc = int(r), int(c)
                if _support_count(mask)[rr, cc] >= 4:
                    continue
                trial = mask.copy()
                trial[rr, cc] = False
                if trial.any() and _count_components(trial) == _count_components(mask):
                    removable.append((rr, cc))
            if removable:
                rr, cc = random.choice(removable)
                mask[rr, cc] = False
    return mask


def _find_full_cross(grid: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        rows = np.where(np.all(grid == int(color), axis=1))[0]
        cols = np.where(np.all(grid == int(color), axis=0))[0]
        for r in rows:
            for c in cols:
                candidates.append((int(color), int(r), int(c)))
    assert len(candidates) == 1
    return candidates[0]


def _split_panels(grid: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    cross_color, cross_r, cross_c = _find_full_cross(grid)
    h, w = grid.shape
    assert h == 2 * cross_r + 1 and w == 2 * cross_c + 1
    panels = [
        grid[:cross_r, :cross_c],
        grid[:cross_r, cross_c + 1 :],
        grid[cross_r + 1 :, :cross_c],
        grid[cross_r + 1 :, cross_c + 1 :],
    ]
    assert len({panel.shape for panel in panels}) == 1
    return cross_color, 0, panels


def _common_bg_and_colors(panels: list[np.ndarray]) -> tuple[int, list[int]]:
    color_sets = [set(map(int, np.unique(panel))) for panel in panels]
    common = set.intersection(*color_sets)
    assert len(common) == 1
    bg = common.pop()
    fg_colors: list[int] = []
    for panel in panels:
        colors = [int(color) for color in np.unique(panel) if int(color) != bg]
        assert len(colors) == 1
        fg_colors.append(colors[0])
    assert len(set(fg_colors)) == 4
    return bg, fg_colors


def _compose_support_overlay(
    masks: list[np.ndarray], fg_colors: list[int], bg: int
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    shape = masks[0].shape
    supports = [_support_count(mask) for mask in masks]
    out = np.full(shape, bg, dtype=int)
    winner_idx = np.full(shape, -1, dtype=int)
    conflicts: list[dict] = []

    for r in range(shape[0]):
        for c in range(shape[1]):
            contenders = [i for i, mask in enumerate(masks) if mask[r, c]]
            if len(contenders) == 0:
                continue
            if len(contenders) == 1:
                winner = contenders[0]
                out[r, c] = fg_colors[winner]
                winner_idx[r, c] = winner
                continue

            support_values = {i: int(supports[i][r, c]) for i in contenders}
            max_support = max(support_values.values())
            winners = [i for i in contenders if support_values[i] == max_support]
            if len(winners) != 1:
                return None

            winner = winners[0]
            out[r, c] = fg_colors[winner]
            winner_idx[r, c] = winner
            conflicts.append(
                {
                    "pos": (r, c),
                    "contenders": tuple(contenders),
                    "winner": winner,
                    "support_values": support_values,
                }
            )
    return out, winner_idx, conflicts


def _compose_fixed_order(masks: list[np.ndarray], fg_colors: list[int], bg: int, order) -> np.ndarray:
    out = np.full(masks[0].shape, bg, dtype=int)
    for idx in order:
        out[masks[idx]] = fg_colors[idx]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse four cross-separated panels into one panel, resolving each overlap by local 4-neighbor support.

    Input structure:
    - A single full-row/full-column cross splits the grid into four equal panels.
    - Each panel contains one shared background color and one panel-specific foreground color.
    - Corresponding local coordinates across the four panels refer to the same output cell.

    Rule:
    - Remove the cross and align the four panels on one canvas of panel size.
    - If exactly one panel has foreground at a local cell, copy that panel's foreground color there.
    - If multiple panels overlap at a local cell, count for each contender how many 4-neighbors of that cell are also foreground inside that same panel; the unique highest-support panel wins that cell.
    - Cells with no panel foreground stay background.

    Key insight:
    - Overlap is not resolved by a fixed quadrant stack; each contested cell is judged independently by how locally embedded that pixel is inside its own panel shape.
    """
    g = np.asarray(grid, dtype=int)
    _, _, panels = _split_panels(g)
    bg, fg_colors = _common_bg_and_colors(panels)
    masks = [panel != bg for panel in panels]
    composed = _compose_support_overlay(masks, fg_colors, bg)
    assert composed is not None
    out, _, _ = composed
    return out


PANEL_MIN = 4
PANEL_MAX = 7
FILL_RATIO_MIN = 0.22
FILL_RATIO_MAX = 0.50
MAX_COMPONENTS = 3
GENERATION_ATTEMPTS = 2000


def _random_panel_mask(shape: tuple[int, int]) -> np.ndarray | None:
    area = shape[0] * shape[1]
    lo = max(4, int(np.ceil(area * FILL_RATIO_MIN)))
    hi = min(area - 2, int(np.floor(area * FILL_RATIO_MAX)))
    if lo > hi:
        return None
    target = random.randint(lo, hi)
    return _grow_mask(shape, target, MAX_COMPONENTS)


def _pairwise_overlap_pairs(masks: list[np.ndarray]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for a in range(4):
        for b in range(a + 1, 4):
            if np.any(masks[a] & masks[b]):
                pairs.add((a, b))
    return pairs


def _contradicts_global_area(conflicts: list[dict], areas: list[int]) -> bool:
    for info in conflicts:
        contenders = info["contenders"]
        max_area = max(areas[i] for i in contenders)
        biggest = [i for i in contenders if areas[i] == max_area]
        if len(biggest) == 1 and biggest[0] != info["winner"]:
            return True
    return False


def _semantic_accepts(
    masks: list[np.ndarray],
    fg_colors: list[int],
    bg: int,
    require_high_order: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    composed = _compose_support_overlay(masks, fg_colors, bg)
    if composed is None:
        return None
    out, winner_idx, conflicts = composed
    area = out.shape[0] * out.shape[1]
    if not conflicts:
        return None
    if len(conflicts) < 2 or len(conflicts) > max(14, area // 2):
        return None
    if require_high_order and not any(len(info["contenders"]) >= 3 for info in conflicts):
        return None
    if not require_high_order and not any(len(info["contenders"]) == 2 for info in conflicts):
        return None
    if np.sum(out == bg) < 2:
        return None
    if len(set(map(int, out.flatten())) - {bg}) != 4:
        return None

    winning_panels = {info["winner"] for info in conflicts}
    if len(winning_panels) < 2:
        return None

    overlap_pairs = _pairwise_overlap_pairs(masks)
    if len(overlap_pairs) < 2:
        return None

    areas = [int(mask.sum()) for mask in masks]
    if not _contradicts_global_area(conflicts, areas):
        return None
    if random.random() < 0.35:
        for order in itertools.permutations(range(4)):
            if np.array_equal(_compose_fixed_order(masks, fg_colors, bg, order), out):
                return None

    # Make sure both direct-copy cells and support-resolved cells matter.
    if not np.any(winner_idx == -1):
        return None
    unique_only = np.zeros(out.shape, dtype=bool)
    for mask in masks:
        unique_only |= mask
    overlap_count = sum(mask.astype(int) for mask in masks)
    if not np.any(overlap_count == 1):
        return None

    return out, winner_idx, conflicts


def _render_input(
    masks: list[np.ndarray],
    bg: int,
    cross_color: int,
    fg_colors: list[int],
) -> np.ndarray:
    h, w = masks[0].shape
    grid = np.full((2 * h + 1, 2 * w + 1), cross_color, dtype=int)
    slices = [
        (slice(0, h), slice(0, w)),
        (slice(0, h), slice(w + 1, 2 * w + 1)),
        (slice(h + 1, 2 * h + 1), slice(0, w)),
        (slice(h + 1, 2 * h + 1), slice(w + 1, 2 * w + 1)),
    ]
    for idx, (rs, cs) in enumerate(slices):
        panel = np.full((h, w), bg, dtype=int)
        panel[masks[idx]] = fg_colors[idx]
        grid[rs, cs] = panel
    return grid


def _attempt_generate() -> np.ndarray | None:
    panel_h = random.randint(PANEL_MIN, PANEL_MAX)
    panel_w = random.randint(PANEL_MIN, PANEL_MAX)
    palette = random.sample(range(10), 6)
    bg, cross_color, *fg_colors = palette
    require_high_order = random.random() < 0.45

    masks: list[np.ndarray] = []
    for _ in range(4):
        mask = _random_panel_mask((panel_h, panel_w))
        if mask is None:
            return None
        masks.append(mask)

    accepted = _semantic_accepts(masks, fg_colors, bg, require_high_order)
    if accepted is None:
        return None
    expected, _, _ = accepted

    grid = _render_input(masks, bg, cross_color, fg_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid local-support overlay instance")
