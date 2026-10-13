from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

PANEL_COUNT = 3
PAIR_INDEXES = ((0, 1), (0, 2), (1, 2))
VOTE_TRIPLES = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
)

PANEL_HEIGHT_RANGE = (8, 9)
PANEL_WIDTH_RANGE = (10, 12)
EXTRA_NOISE_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _extract_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert width > 0 and (height - 2) % PANEL_COUNT == 0, "invalid stacked-panel layout"

    panel_h = (height - 2) // PANEL_COUNT
    assert panel_h > 0
    separator_rows = (panel_h, 2 * panel_h + 1)
    for rr in separator_rows:
        assert len({int(v) for v in g[rr, :]}) == 1, "separator rows must be full-width solid lines"

    return [
        g[:panel_h, :].copy(),
        g[panel_h + 1 : 2 * panel_h + 1, :].copy(),
        g[2 * panel_h + 2 :, :].copy(),
    ]


def _candidate_blocks(panel: np.ndarray, bg: int) -> list[Coord]:
    blocks: list[Coord] = []
    for coords in _components(panel != bg):
        if len(coords) != 4:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        if np.all(panel[r0 : r1 + 1, c0 : c1 + 1] != bg):
            blocks.append((r0, c0))
    return blocks


def _panel_votes(panels: list[np.ndarray], bg: int) -> tuple[list[int], list[list[Coord]]]:
    block_lists = [_candidate_blocks(panel, bg) for panel in panels]
    fg_masks = [panel != bg for panel in panels]

    votes: list[int] = []
    for panel_index, blocks in enumerate(block_lists):
        count = 0
        for r0, c0 in blocks:
            contested = False
            for other_index in range(PANEL_COUNT):
                if other_index == panel_index:
                    continue
                if np.any(fg_masks[other_index][r0 : r0 + 2, c0 : c0 + 2]):
                    contested = True
                    break
            if not contested:
                count += 1
        votes.append(count)

    return votes, block_lists


def _order_from_counts(counts: tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    order = tuple(sorted(range(PANEL_COUNT), key=lambda idx: int(counts[idx])))
    assert len({int(counts[idx]) for idx in order}) == PANEL_COUNT, "counts must be pairwise distinct"
    return order


def _overlay_panels(panels: list[np.ndarray], order: tuple[int, int, int], bg: int) -> np.ndarray:
    out = np.full_like(panels[0], int(bg))
    for idx in order:
        mask = panels[int(idx)] != bg
        out[mask] = panels[int(idx)][mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay three stacked panels after ordering them by their uncontested 2x2 block votes.

    Input structure:
    - The grid is three equal-size panels stacked vertically, with a full-width separator line between panels.
    - All panels share one background color, and each panel uses exactly one foreground color for its own objects.
    - Foreground objects inside a panel are separated 4-connected pieces that include solid 2x2 blocks and small distractor fragments.

    Rule:
    - For each panel, find every foreground component that is exactly a solid 2x2 block.
    - A 2x2 block counts as a vote only when the corresponding 2x2 cells are background in both other aligned panels.
    - Sort the three panels by vote count from fewest to most, then transparently paint the whole panels in that order so later panels overwrite earlier ones.

    Key insight:
    - Vote counts cannot be read from one panel in isolation: blocks overlapped by any other panel lose their vote, and only those cross-panel-filtered counts determine the final layer order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    panels = _extract_panels(g)

    shapes = {panel.shape for panel in panels}
    assert len(shapes) == 1, "all panels must have the same shape"
    for panel in panels:
        fg_colors = {int(v) for v in np.unique(panel) if int(v) != bg}
        assert len(fg_colors) == 1, "each panel must contain exactly one foreground color"

    votes, _ = _panel_votes(panels, bg)
    order = _order_from_counts(votes)
    return _overlay_panels(panels, order, bg)


def _slot_anchors(panel_h: int, width: int) -> list[Coord]:
    rows = list(range(0, panel_h - 1, 3))
    cols = list(range(0, width - 1, 3))
    return [(r, c) for r in rows for c in cols]


def _witness_coords(panel_h: int, width: int) -> list[Coord]:
    assert panel_h >= 3
    cols = [c for c in range(width) if c % 3 == 2]
    assert len(cols) >= len(PAIR_INDEXES)
    chosen = random.sample(cols, len(PAIR_INDEXES))
    return [(2, c) for c in chosen]


def _choose_vote_plan(slot_count: int) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    plans: list[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = []
    for base in VOTE_TRIPLES:
        for votes in set(itertools.permutations(base)):
            vote_order = _order_from_counts(votes)
            for extras in itertools.product(range(4), repeat=PANEL_COUNT):
                if sum(extras) < 2:
                    continue
                raw = tuple(votes[i] + extras[i] for i in range(PANEL_COUNT))
                if min(raw) < 1:
                    continue
                if len(set(raw)) != PANEL_COUNT:
                    continue
                if sum(raw) > slot_count:
                    continue
                if _order_from_counts(raw) == vote_order:
                    continue
                plans.append((votes, extras, raw))
    assert plans, "no non-degenerate vote plans available"
    return random.choice(plans)


def _paint_block(panel: np.ndarray, color: int, slot: Coord) -> None:
    r0, c0 = slot
    panel[r0 : r0 + 2, c0 : c0 + 2] = int(color)


def _free_pixel_cells_for_blockers(panel: np.ndarray, slot: Coord, bg: int) -> list[Coord]:
    r0, c0 = slot
    cells = [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]
    return [(r, c) for r, c in cells if int(panel[r, c]) == int(bg)]


def _non_adjacent_safe_cells(panel_h: int, width: int, reserved: set[Coord]) -> list[Coord]:
    cells: list[Coord] = []
    for r in range(panel_h):
        if r % 3 != 2:
            continue
        for c in range(width):
            if c % 3 != 2:
                continue
            coord = (r, c)
            if coord not in reserved:
                cells.append(coord)
    return cells


def _can_place_isolated_pixel(panel: np.ndarray, coord: Coord, bg: int) -> bool:
    r, c = coord
    if int(panel[r, c]) != int(bg):
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < panel.shape[0] and 0 <= nc < panel.shape[1] and int(panel[nr, nc]) != int(bg):
            return False
    return True


def _build_instance(
    panel_h: int,
    width: int,
    bg: int,
    separator: int,
    colors: tuple[int, int, int],
    votes: tuple[int, int, int],
    extras: tuple[int, int, int],
) -> tuple[np.ndarray, dict]:
    slots = _slot_anchors(panel_h, width)
    random.shuffle(slots)
    slot_cursor = 0

    panels = [np.full((panel_h, width), int(bg), dtype=int) for _ in range(PANEL_COUNT)]
    vote_slots: list[list[Coord]] = [[] for _ in range(PANEL_COUNT)]
    extra_specs: list[dict] = []

    owner_order = random.sample(range(PANEL_COUNT), PANEL_COUNT)
    for owner in owner_order:
        for _ in range(votes[owner]):
            vote_slots[owner].append(slots[slot_cursor])
            slot_cursor += 1

    for owner in random.sample(range(PANEL_COUNT), PANEL_COUNT):
        for _ in range(extras[owner]):
            blocker_count = 2 if random.random() < 0.35 else 1
            blockers = tuple(sorted(random.sample([q for q in range(PANEL_COUNT) if q != owner], blocker_count)))
            extra_specs.append({"owner": owner, "slot": slots[slot_cursor], "blockers": blockers})
            slot_cursor += 1

    for owner, panel in enumerate(panels):
        for slot in vote_slots[owner]:
            _paint_block(panel, colors[owner], slot)
        for spec in extra_specs:
            if spec["owner"] == owner:
                _paint_block(panel, colors[owner], spec["slot"])

    for spec in extra_specs:
        chosen_cells: list[Coord] = []
        for blocker in spec["blockers"]:
            free_cells = _free_pixel_cells_for_blockers(panels[blocker], spec["slot"], bg)
            assert free_cells, "blocker placement unexpectedly collided"
            coord = random.choice(free_cells)
            panels[blocker][coord] = int(colors[blocker])
            chosen_cells.append(coord)
        spec["blocker_cells"] = tuple(chosen_cells)

    witness_coords = _witness_coords(panel_h, width)
    witness_map: dict[tuple[int, int], Coord] = {}
    for pair, coord in zip(PAIR_INDEXES, witness_coords):
        witness_map[pair] = coord
        for owner in pair:
            assert int(panels[owner][coord]) == int(bg)
            panels[owner][coord] = int(colors[owner])

    reserved = set(witness_coords)
    for spec in extra_specs:
        r0, c0 = spec["slot"]
        for dr in range(2):
            for dc in range(2):
                reserved.add((r0 + dr, c0 + dc))
    for owner_slots in vote_slots:
        for r0, c0 in owner_slots:
            for dr in range(2):
                for dc in range(2):
                    reserved.add((r0 + dr, c0 + dc))

    noise_candidates = _non_adjacent_safe_cells(panel_h, width, reserved)
    for panel_index, panel in enumerate(panels):
        count = random.randint(*EXTRA_NOISE_RANGE)
        random.shuffle(noise_candidates)
        added = 0
        for coord in noise_candidates:
            if added >= count:
                break
            if _can_place_isolated_pixel(panel, coord, bg):
                panel[coord] = int(colors[panel_index])
                added += 1

    sep_row = np.full((1, width), int(separator), dtype=int)
    grid = np.vstack([panels[0], sep_row, panels[1], sep_row.copy(), panels[2]])
    meta = {
        "bg": int(bg),
        "separator": int(separator),
        "colors": tuple(map(int, colors)),
        "votes": tuple(map(int, votes)),
        "extras": tuple(map(int, extras)),
        "raw": tuple(int(votes[i] + extras[i]) for i in range(PANEL_COUNT)),
        "witness_map": witness_map,
    }
    return grid, meta


def _validate_instance(grid: np.ndarray, meta: dict) -> bool:
    bg = _get_bg(grid)
    if bg != int(meta["bg"]):
        return False

    panels = _extract_panels(grid)
    for panel_index, panel in enumerate(panels):
        fg_colors = {int(v) for v in np.unique(panel) if int(v) != bg}
        if fg_colors != {int(meta["colors"][panel_index])}:
            return False

    votes, blocks = _panel_votes(panels, bg)
    raw = tuple(len(block_list) for block_list in blocks)
    vote_order = _order_from_counts(votes)
    raw_order = _order_from_counts(raw)

    if tuple(votes) != tuple(meta["votes"]):
        return False
    if raw != tuple(meta["raw"]):
        return False
    if raw_order == vote_order:
        return False

    expected = _overlay_panels(panels, vote_order, bg)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(_overlay_panels(panels, raw_order, bg), expected):
        return False

    outputs = set()
    for perm in itertools.permutations(range(PANEL_COUNT)):
        outputs.add(_overlay_panels(panels, perm, bg).tobytes())
    if len(outputs) != 6:
        return False

    for pair, coord in meta["witness_map"].items():
        a, b = pair
        if int(panels[a][coord]) == bg or int(panels[b][coord]) == bg:
            return False
        other = next(idx for idx in range(PANEL_COUNT) if idx not in pair)
        if int(panels[other][coord]) != bg:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    width = random.randint(*PANEL_WIDTH_RANGE)
    slots = _slot_anchors(panel_h, width)
    if len(slots) < 7:
        return None

    palette = random.sample(range(10), 5)
    bg, separator, c0, c1, c2 = palette
    votes, extras, _raw = _choose_vote_plan(len(slots))

    grid, meta = _build_instance(
        panel_h=panel_h,
        width=width,
        bg=bg,
        separator=separator,
        colors=(c0, c1, c2),
        votes=votes,
        extras=extras,
    )
    if not _validate_instance(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
