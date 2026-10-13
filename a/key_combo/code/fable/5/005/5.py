from __future__ import annotations

import random

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()
_GENERATION_COUNT = 0
_HOLE_PROFILES = ((1, 1, 2), (2, 3, 3), (1, 2, 3))


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)
    assert len(counts) == 1 or counts[order[-1]] > counts[order[-2]], (
        "background must be the unique most common color"
    )
    return int(colors[order[-1]])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        stack = [(r0, c0)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _hole_count(mask: np.ndarray) -> int:
    """Count enclosed 4-connected background regions inside a tight object crop."""
    rows, cols = np.nonzero(mask)
    assert rows.size > 0
    sub = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    h, w = sub.shape
    outside = ~sub
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0, c0 in np.argwhere(outside):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        stack = [(r0, c0)]
        touches_border = False
        while stack:
            r, c = stack.pop()
            touches_border |= r == 0 or c == 0 or r == h - 1 or c == w - 1
            for dr, dc in _DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and outside[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if not touches_border:
            holes += 1
    return holes


def _split_panels(grid: np.ndarray, bg: int) -> tuple[list[np.ndarray], str]:
    """Detect the two full separator bars and return three panels in display order."""
    h, w = grid.shape
    layouts: list[tuple[list[np.ndarray], str]] = []

    if w >= 5 and (w - 2) % 3 == 0:
        pw = (w - 2) // 3
        c1, c2 = pw, 2 * pw + 1
        sep = int(grid[0, c1])
        if (
            sep != bg
            and np.all(grid[:, c1] == sep)
            and np.all(grid[:, c2] == sep)
        ):
            layouts.append(
                (
                    [grid[:, :pw], grid[:, pw + 1 : 2 * pw + 1], grid[:, 2 * pw + 2 :]],
                    "vertical",
                )
            )

    if h >= 5 and (h - 2) % 3 == 0:
        ph = (h - 2) // 3
        r1, r2 = ph, 2 * ph + 1
        sep = int(grid[r1, 0])
        if (
            sep != bg
            and np.all(grid[r1, :] == sep)
            and np.all(grid[r2, :] == sep)
        ):
            layouts.append(
                (
                    [grid[:ph, :], grid[ph + 1 : 2 * ph + 1, :], grid[2 * ph + 2 :, :]],
                    "horizontal",
                )
            )

    assert len(layouts) == 1, "exactly one orientation must contain two equal-panel separator bars"
    panels, orientation = layouts[0]
    assert panels[0].shape == panels[1].shape == panels[2].shape
    return panels, orientation


def _panel_objects(panel: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    colors = np.unique(panel[panel != bg])
    assert colors.size == 1, "each panel's four objects must share one foreground color"
    color = int(colors[0])
    comps = _components(panel == color)
    assert len(comps) == 4, "each panel must contain exactly four 4-connected objects"

    records: list[dict[str, object]] = []
    for coords in comps:
        mask = np.zeros(panel.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        records.append(
            {
                "mask": mask,
                "area": int(coords.shape[0]),
                "holes": _hole_count(mask),
            }
        )

    pointers = [obj for obj in records if int(obj["holes"]) > 0]
    solids = [obj for obj in records if int(obj["holes"]) == 0]
    assert len(pointers) == 1 and len(solids) == 3
    holes = int(pointers[0]["holes"])
    assert 1 <= holes <= 3, "pointer hole count must be from one to three"
    solids.sort(key=lambda obj: int(obj["area"]), reverse=True)
    areas = [int(obj["area"]) for obj in solids]
    assert len(set(areas)) == 3, "solid-object areas must be pairwise distinct"
    return holes, solids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each holed pointer to select an area-ranked solid in the cyclic next panel, then merge the three selections in place.

    Input structure:
    - Two one-cell solid separator bars divide the grid into three equal panels, horizontally or vertically.
    - Every panel has one background, one object color, and exactly four 4-connected objects of that color.
    - Exactly one object in each panel has one to three enclosed holes; the other three are hole-free and have distinct areas.

    Rule:
    - A pointer with h holes selects the h-th-largest solid object in the next panel, with panel three pointing back to panel one.
    - Copy the three selected solids to a background canvas the size of one panel, preserving their panel-relative coordinates.
    - The valid selections occupy disjoint cells, so all three panel colors remain visible in the merged output; pointers are discarded.

    Key insight:
    - Hole count encodes an area rank, but its referent is the cyclically next panel rather than the pointer's own panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _background(g)
    panels, _orientation = _split_panels(g, bg)

    holes: list[int] = []
    solids_by_panel: list[list[dict[str, object]]] = []
    colors: list[int] = []
    for panel in panels:
        h, solids = _panel_objects(panel, bg)
        holes.append(h)
        solids_by_panel.append(solids)
        colors.append(int(panel[panel != bg][0]))

    out = np.full(panels[0].shape, bg, dtype=int)
    occupied = np.zeros(panels[0].shape, dtype=bool)
    # The pointer in source i acts on target (i + 1) mod 3.
    for source in range(3):
        target = (source + 1) % 3
        chosen = solids_by_panel[target][holes[source] - 1]
        mask = np.asarray(chosen["mask"], dtype=bool)
        assert not np.any(occupied & mask), "selected objects must not overlap in valid inputs"
        out[mask] = colors[target]
        occupied |= mask

    assert len(np.unique(out[occupied])) == 3, "all three selected panel colors must remain visible"
    return out


def _random_polyomino(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Grow a random hole-free 4-connected object and return its tight mask."""
    for _ in range(80):
        cells = {(random.randrange(max_h), random.randrange(max_w))}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in _DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < max_h and 0 <= cc < max_w and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            # Sometimes favor compact growth, sometimes produce arms and corners.
            if random.random() < 0.45:
                scores = {
                    cell: sum((cell[0] + dr, cell[1] + dc) in cells for dr, dc in _DIR4)
                    for cell in frontier
                }
                best = max(scores.values())
                choices = [cell for cell, score in scores.items() if score == best]
                cells.add(random.choice(choices))
            else:
                cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        patch = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=bool)
        for r, c in cells:
            patch[r - min(rows), c - min(cols)] = True
        if _hole_count(patch) == 0:
            return patch
    return None


def _pointer_patch(holes: int) -> np.ndarray:
    """Make a visibly framed pointer with exactly `holes` isolated one-cell holes."""
    width = 2 * holes + 1
    base = np.ones((3, width), dtype=bool)
    base[1, 1::2] = False
    if random.random() < 0.65:
        patch = np.zeros((4, width), dtype=bool)
        if random.random() < 0.5:
            patch[1:, :] = base
            row = 0
        else:
            patch[:3, :] = base
            row = 3
        center = width // 2
        bump_width = random.choice((1, 2, 3))
        left = max(0, min(width - bump_width, center - bump_width // 2))
        patch[row, left : left + bump_width] = True
        assert _hole_count(patch) == holes
        return patch
    return base


def _slot_starts(long_side: int) -> list[int]:
    """Four height-4 slots separated from each other and the panel border."""
    gaps = [1, 1, 1, 1, 1]
    for _ in range(long_side - 21):
        gaps[random.randrange(5)] += 1
    starts: list[int] = []
    cursor = gaps[0]
    for i in range(4):
        starts.append(cursor)
        cursor += 4 + gaps[i + 1]
    return starts


def _place_patch(panel: np.ndarray, patch: np.ndarray, slot_row: int, color: int) -> np.ndarray:
    ph, pw = patch.shape
    assert ph <= 4 and pw <= panel.shape[1] - 2
    top = slot_row + random.randrange(5 - ph)
    left = random.randint(1, panel.shape[1] - pw - 1)
    panel[top : top + ph, left : left + pw][patch] = color
    mask = np.zeros(panel.shape, dtype=bool)
    mask[top : top + ph, left : left + pw] = patch
    return mask


def _merge_from_ranks(
    records: list[dict[str, object]], ranks_for_target: list[int], bg: int
) -> np.ndarray:
    shape = np.asarray(records[0]["pointer"], dtype=bool).shape
    out = np.full(shape, bg, dtype=int)
    for target in range(3):
        rank = ranks_for_target[target]
        mask = np.asarray(records[target]["solids"][rank - 1], dtype=bool)
        out[mask] = int(records[target]["color"])
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    canonical_records: list[dict[str, object]],
    holes: list[int],
    bg: int,
    transposed: bool,
) -> bool:
    """Reject oracle-consistent but semantically degenerate or shortcut-friendly cases."""
    if any(h not in (1, 2, 3) for h in holes) or len(set(holes)) == 1:
        return False

    correct_ranks = [holes[(target - 1) % 3] for target in range(3)]
    local_ranks = holes[:]
    ascending_ranks = [4 - rank for rank in correct_ranks]
    panel_index_ranks = [1, 2, 3]
    canonical_expected = _merge_from_ranks(canonical_records, correct_ranks, bg)
    wrong_outputs = [
        _merge_from_ranks(canonical_records, local_ranks, bg),
        _merge_from_ranks(canonical_records, ascending_ranks, bg),
        _merge_from_ranks(canonical_records, panel_index_ranks, bg),
    ]
    if any(np.array_equal(canonical_expected, wrong) for wrong in wrong_outputs):
        return False

    occupied = np.zeros(canonical_expected.shape, dtype=bool)
    for target, rank in enumerate(correct_ranks):
        solids = canonical_records[target]["solids"]
        areas = [int(np.sum(mask)) for mask in solids]
        if not (areas[0] > areas[1] > areas[2]):
            return False
        chosen = np.asarray(solids[rank - 1], dtype=bool)
        if np.any(occupied & chosen):
            return False
        occupied |= chosen
        # A counterfactual change in this source pointer's hole count changes
        # which genuinely different object is selected in the next panel.
        alternative = np.asarray(solids[rank % 3], dtype=bool)
        if np.array_equal(chosen, alternative):
            return False

    if len(np.unique(canonical_expected[occupied])) != 3 or int(occupied.sum()) < 16:
        return False

    # Disjointness keeps the merge visually unambiguous, but it must not reveal
    # the answer by itself: require another all-disjoint rank triple besides the
    # hole-directed one.
    disjoint_rank_triples = 0
    for r0 in (1, 2, 3):
        for r1 in (1, 2, 3):
            for r2 in (1, 2, 3):
                masks = [
                    np.asarray(canonical_records[i]["solids"][rank - 1], dtype=bool)
                    for i, rank in enumerate((r0, r1, r2))
                ]
                if not any(np.any(masks[a] & masks[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
                    disjoint_rank_triples += 1
    if disjoint_rank_triples < 2:
        return False
    oriented_expected = canonical_expected.T if transposed else canonical_expected
    if not np.array_equal(oriented_expected, expected):
        return False
    try:
        actual = main(inp)
    except (AssertionError, IndexError, ValueError):
        return False
    return np.array_equal(actual, expected)


def _generation_attempt(hole_profile: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    long_side = random.randint(21, 28)
    short_side = 9
    bg, separator, *panel_colors = random.sample(range(10), 5)
    holes = list(hole_profile)
    random.shuffle(holes)
    starts = _slot_starts(long_side)

    # Choose each panel's pointer lane and rank-to-lane mapping until the three
    # cross-panel selections land in different spatial lanes.
    lane_data: list[tuple[int, list[int]]] | None = None
    for _ in range(100):
        trial: list[tuple[int, list[int]]] = []
        for _panel in range(3):
            pointer_lane = random.randrange(4)
            rank_lanes = [lane for lane in range(4) if lane != pointer_lane]
            random.shuffle(rank_lanes)
            trial.append((pointer_lane, rank_lanes))
        selected_lanes = [trial[t][1][holes[(t - 1) % 3] - 1] for t in range(3)]
        if len(set(selected_lanes)) == 3:
            lane_data = trial
            break
    if lane_data is None:
        return None

    panels: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for i in range(3):
        panel = np.full((long_side, short_side), bg, dtype=int)
        pointer_lane, rank_lanes = lane_data[i]
        pointer = _pointer_patch(holes[i])
        pointer_mask = _place_patch(panel, pointer, starts[pointer_lane], panel_colors[i])

        areas = [random.randint(9, 12), random.randint(5, 8), random.randint(2, 4)]
        solid_masks: list[np.ndarray] = []
        for area, lane in zip(areas, rank_lanes):
            patch = _random_polyomino(area, 4, short_side - 2)
            if patch is None:
                return None
            solid_masks.append(_place_patch(panel, patch, starts[lane], panel_colors[i]))

        # Validate topology from the rendered panel, not merely from templates.
        try:
            actual_holes, actual_solids = _panel_objects(panel, bg)
        except AssertionError:
            return None
        if actual_holes != holes[i] or [int(obj["area"]) for obj in actual_solids] != areas:
            return None
        panels.append(panel)
        records.append(
            {
                "color": panel_colors[i],
                "pointer": pointer_mask,
                "solids": solid_masks,
            }
        )

    combined = np.full((long_side, 3 * short_side + 2), bg, dtype=int)
    combined[:, short_side] = separator
    combined[:, 2 * short_side + 1] = separator
    for i, panel in enumerate(panels):
        combined[:, i * (short_side + 1) : i * (short_side + 1) + short_side] = panel

    correct_ranks = [holes[(target - 1) % 3] for target in range(3)]
    expected = _merge_from_ranks(records, correct_ranks, bg)
    transposed = random.random() < 0.5
    inp = combined.T.copy() if transposed else combined
    expected = expected.T.copy() if transposed else expected
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_checks(inp, expected, records, holes, bg, transposed):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a diverse valid instance with explicit anti-shortcut validation."""
    global _GENERATION_COUNT
    profile = _HOLE_PROFILES[_GENERATION_COUNT % len(_HOLE_PROFILES)]
    for _ in range(400):
        generated = _generation_attempt(profile)
        if generated is None:
            continue
        inp, expected = generated
        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        _GENERATION_COUNT += 1
        return inp
    raise RuntimeError("failed to generate a non-degenerate hole-pointer panel instance")
