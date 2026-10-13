from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

LOWER_COLORS = (GREEN, YELLOW, MAGENTA)
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

# Generator configuration.
GEN_PANEL_H_RANGE = (5, 7)
GEN_PANEL_W_RANGE = (8, 10)
GEN_MARGIN_OPTIONS = (0, 1)
GEN_TOP_COUNT_RANGE = (6, 10)
GEN_MID_COUNT_RANGE = (6, 10)
GEN_SOURCE_COUNT_RANGE = (3, 5)
GEN_MAX_ATTEMPTS = 600


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _is_isolated8(cell: tuple[int, int], points: set[tuple[int, int]]) -> bool:
    return all(_chebyshev(cell, other) > 1 for other in points)


def _is_isolated4(cell: tuple[int, int], points: set[tuple[int, int]]) -> bool:
    return all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in points)


def _is_full_grey_row(row: np.ndarray) -> bool:
    return bool(np.all(row == GREY))


def _split_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_rows = [r for r in range(g.shape[0]) if _is_full_grey_row(g[r])]
    assert len(separator_rows) == 2, "expected exactly two grey separator rows"
    sep0, sep1 = separator_rows
    panel_h = sep1 - sep0 - 1
    assert panel_h > 0, "separator rows must enclose a non-empty middle panel"

    top_start = sep0 - panel_h
    bottom_end = sep1 + 1 + panel_h
    assert top_start >= 0, "top panel must fit above the first separator"
    assert bottom_end <= g.shape[0], "bottom panel must fit below the second separator"

    if top_start > 0:
        assert np.all(g[:top_start] == BLACK), "top margin must be black"
    if bottom_end < g.shape[0]:
        assert np.all(g[bottom_end:] == BLACK), "bottom margin must be black"

    top = g[top_start:sep0]
    mid = g[sep0 + 1 : sep1]
    low = g[sep1 + 1 : bottom_end]
    assert top.shape == mid.shape == low.shape, "expected three equal-size panels"
    assert np.count_nonzero(top == GREY) == 0 and np.count_nonzero(mid == GREY) == 0 and np.count_nonzero(low == GREY) == 0
    return top, mid, low, sep1 + 1


def _anchors_from_panels(top: np.ndarray, mid: np.ndarray) -> list[tuple[int, int]]:
    assert set(map(int, np.unique(top))) <= {BLACK, BLUE}, "top panel may only use black and blue"
    assert set(map(int, np.unique(mid))) <= {BLACK, RED}, "middle panel may only use black and red"
    anchors = [tuple(map(int, p)) for p in np.argwhere((top == BLACK) & (mid == BLACK))]
    assert anchors, "expected at least one background-intersection anchor"
    return anchors


def _lower_sources(low: np.ndarray) -> list[dict[str, int]]:
    colors = set(map(int, np.unique(low)))
    assert colors <= {BLACK, *LOWER_COLORS}, "bottom panel may only use black or lower-source colors"
    coords = [tuple(map(int, p)) for p in np.argwhere(low != BLACK)]
    assert 3 <= len(coords) <= 5, "expected 3 to 5 lower singleton pixels"
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            assert _chebyshev(a, b) > 1, "lower singleton pixels must not touch"
    return [{"row": r, "col": c, "color": int(low[r, c])} for r, c in coords]


def _best_anchor(
    src: dict[str, int], anchors: list[tuple[int, int]]
) -> tuple[tuple[int, int], int, int]:
    best_anchor = None
    best_key = None
    min_dist = None
    min_count = 0
    for anchor in anchors:
        dist = abs(src["row"] - anchor[0]) + abs(src["col"] - anchor[1])
        key = (dist, anchor[1], anchor[0])
        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_count = 1
        elif dist == min_dist:
            min_count += 1
        if best_key is None or key < best_key:
            best_key = key
            best_anchor = anchor
    assert best_anchor is not None and min_dist is not None
    return best_anchor, min_dist, min_count


def _resolve_assignments(
    sources: list[dict[str, int]], anchors: list[tuple[int, int]]
) -> tuple[dict[tuple[int, int], dict[str, int]], list[dict[str, object]]]:
    proposed: list[dict[str, object]] = []
    for index, src in enumerate(sources):
        anchor, dist, min_count = _best_anchor(src, anchors)
        proposed.append(
            {
                "index": index,
                "row": src["row"],
                "col": src["col"],
                "color": src["color"],
                "anchor": anchor,
                "dist": dist,
                "min_anchor_count": min_count,
            }
        )

    winners: dict[tuple[int, int], dict[str, int]] = {}
    for item in proposed:
        anchor = item["anchor"]
        current = winners.get(anchor)
        if current is None:
            winners[anchor] = item
            continue
        current_key = (current["dist"], current["row"], current["col"])
        candidate_key = (item["dist"], item["row"], item["col"])
        if candidate_key < current_key:
            winners[anchor] = item
    return winners, proposed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Generate anchor positions from the background intersection of two panels, then snap lower singleton pixels onto those anchors.

    Input structure:
    - The grid has black background and exactly two solid grey separator rows.
    - The separator rows carve out three equal-size panels: a top blue-on-black panel, a middle red-on-black panel, and a lower panel with 3-5 isolated green/yellow/magenta singleton pixels.
    - Optional extra rows above the top panel or below the bottom panel are black margins only.

    Rule:
    - Remove the separator rows and align the top and middle panels cell by cell.
    - Mark every position where both aligned panel cells are black; these positions are the available anchors in the output-sized panel.
    - For each lower singleton, map its row into bottom-panel local coordinates and choose the anchor with minimum Manhattan distance; ties choose the smaller anchor column, then the smaller anchor row.
    - If several lower singletons choose the same anchor, keep only the one with smaller anchor distance; equal distances keep the singleton with smaller lower-panel row, then smaller column. Discard losers and paint winners at their anchors on an otherwise black output grid.

    Key insight:
    - The top and middle panels never appear in the output directly; they only define the hidden anchor set, and the lower singleton pixels matter only through the distance-based snapping and conflict resolution on that anchor set.
    """
    top, mid, low, _bottom_start = _split_regions(np.asarray(grid, dtype=int))
    anchors = _anchors_from_panels(top, mid)
    sources = _lower_sources(low)
    assert len(anchors) >= len(sources), "anchor count must be at least the number of lower sources"

    winners, _proposed = _resolve_assignments(sources, anchors)
    out = np.zeros(top.shape, dtype=int)
    for anchor, item in winners.items():
        out[anchor] = int(item["color"])
    return out


def _sample_tie_pattern(
    h: int,
    w: int,
    sources: list[tuple[int, int]],
    required_holes: set[tuple[int, int]],
    protected_anchors: set[tuple[int, int]],
) -> tuple[str, tuple[int, int]] | None:
    modes = ["horizontal", "vertical"]
    random.shuffle(modes)
    for mode in modes:
        candidates = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if _is_isolated8((r, c), set(sources))
        ]
        random.shuffle(candidates)
        for r, c in candidates:
            if mode == "horizontal":
                holes = {(r, c), (r - 1, c), (r + 1, c)}
                anchors = {(r, c - 1), (r, c + 1)}
            else:
                holes = {(r, c), (r, c - 1), (r, c + 1)}
                anchors = {(r - 1, c), (r + 1, c)}
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in holes | anchors):
                continue
            if holes & protected_anchors:
                continue
            sources.append((r, c))
            required_holes.update(holes)
            protected_anchors.update(anchors)
            return mode, (r, c)
    return None


def _sample_conflict_pattern(
    h: int,
    w: int,
    sources: list[tuple[int, int]],
    required_holes: set[tuple[int, int]],
    protected_anchors: set[tuple[int, int]],
) -> tuple[str, tuple[int, int]] | None:
    modes = ["equal", "distance"]
    random.shuffle(modes)
    source_set = set(sources)
    for mode in modes:
        if mode == "equal":
            centers = [
                (r, c)
                for r in range(1, h - 1)
                for c in range(2, w - 2)
            ]
        else:
            centers = [
                (r, c)
                for r in range(1, h - 1)
                for c in range(2, w - 4)
            ]
        random.shuffle(centers)
        for r, c in centers:
            if mode == "equal":
                srcs = [(r, c - 1), (r, c + 1)]
                holes = {
                    (r, c - 2),
                    (r - 1, c - 1),
                    (r + 1, c - 1),
                    (r, c - 1),
                    (r, c + 1),
                    (r, c + 2),
                    (r - 1, c + 1),
                    (r + 1, c + 1),
                }
            else:
                srcs = [(r, c - 1), (r, c + 2)]
                holes = {
                    (r, c - 2),
                    (r - 1, c - 1),
                    (r + 1, c - 1),
                    (r, c - 1),
                    (r, c + 1),
                    (r, c + 2),
                    (r, c + 3),
                    (r - 1, c + 2),
                    (r + 1, c + 2),
                }
            anchor = (r, c)
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in holes | {anchor} | set(srcs)):
                continue
            if anchor in required_holes:
                continue
            if holes & protected_anchors:
                continue
            if any(not _is_isolated8(src, source_set) for src in srcs):
                continue
            if _chebyshev(srcs[0], srcs[1]) <= 1:
                continue
            for src in srcs:
                if not _is_isolated8(src, source_set | (set(srcs) - {src})):
                    break
            else:
                sources.extend(srcs)
                source_set.update(srcs)
                required_holes.update(holes)
                protected_anchors.add(anchor)
                return mode, anchor
    return None


def _sample_extra_sources(
    h: int,
    w: int,
    count: int,
    sources: list[tuple[int, int]],
    required_holes: set[tuple[int, int]],
    protected_anchors: set[tuple[int, int]],
) -> bool:
    source_set = set(sources)
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for cell in cells:
        if len(sources) >= count:
            return True
        if cell in protected_anchors:
            continue
        if not _is_isolated8(cell, source_set):
            continue
        sources.append(cell)
        source_set.add(cell)
        required_holes.add(cell)
    return len(sources) == count


def _assign_required_holes(
    h: int,
    w: int,
    required_holes: set[tuple[int, int]],
    top_target: int,
    mid_target: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    ordered = sorted(required_holes, key=lambda cell: sum(_chebyshev(cell, other) == 1 for other in required_holes), reverse=True)

    def fits(cell: tuple[int, int], chosen: set[tuple[int, int]]) -> bool:
        return _is_isolated4(cell, chosen)

    def search(
        index: int,
        top_fg: set[tuple[int, int]],
        mid_fg: set[tuple[int, int]],
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
        if len(top_fg) > top_target or len(mid_fg) > mid_target:
            return None
        if index == len(ordered):
            return top_fg, mid_fg

        cell = ordered[index]
        choices = [0, 1]
        random.shuffle(choices)
        for side in choices:
            if side == 0:
                if fits(cell, top_fg):
                    out = search(index + 1, top_fg | {cell}, mid_fg)
                    if out is not None:
                        return out
            else:
                if fits(cell, mid_fg):
                    out = search(index + 1, top_fg, mid_fg | {cell})
                    if out is not None:
                        return out
        return None

    return search(0, set(), set())


def _fill_panel_extras(
    h: int,
    w: int,
    target: int,
    existing: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    chosen = set(existing)
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for cell in cells:
        if len(chosen) >= target:
            return chosen
        if cell in forbidden:
            continue
        if cell in chosen:
            continue
        if not _is_isolated4(cell, chosen):
            continue
        chosen.add(cell)
    return chosen if len(chosen) == target else None


def _anchors_from_sets(
    h: int,
    w: int,
    top_fg: set[tuple[int, int]],
    mid_fg: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    anchors = []
    for r in range(h):
        for c in range(w):
            if (r, c) not in top_fg and (r, c) not in mid_fg:
                anchors.append((r, c))
    return anchors


def _build_panels(
    h: int,
    w: int,
    required_holes: set[tuple[int, int]],
    protected_anchors: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    top_target = random.randint(*GEN_TOP_COUNT_RANGE)
    mid_target = random.randint(*GEN_MID_COUNT_RANGE)
    if len(required_holes) > top_target + mid_target:
        return None

    assigned = _assign_required_holes(h, w, required_holes, top_target, mid_target)
    if assigned is None:
        return None
    top_fg, mid_fg = assigned

    top_fg = _fill_panel_extras(h, w, top_target, top_fg, protected_anchors)
    if top_fg is None:
        return None
    mid_fg = _fill_panel_extras(h, w, mid_target, mid_fg, protected_anchors)
    if mid_fg is None:
        return None

    anchors = _anchors_from_sets(h, w, top_fg, mid_fg)
    if len(anchors) < 3:
        return None
    return top_fg, mid_fg


def _render_input(
    h: int,
    w: int,
    top_fg: set[tuple[int, int]],
    mid_fg: set[tuple[int, int]],
    sources: list[tuple[int, int]],
    source_colors: list[int],
    top_margin: int,
    bottom_margin: int,
) -> np.ndarray:
    total_h = top_margin + h + 1 + h + 1 + h + bottom_margin
    grid = np.zeros((total_h, w), dtype=int)
    sep0 = top_margin + h
    sep1 = sep0 + 1 + h
    grid[sep0, :] = GREY
    grid[sep1, :] = GREY

    for r, c in top_fg:
        grid[top_margin + r, c] = BLUE
    for r, c in mid_fg:
        grid[sep0 + 1 + r, c] = RED
    low_start = sep1 + 1
    for (r, c), color in zip(sources, source_colors):
        grid[low_start + r, c] = color
    return grid


def _semantic_signature(
    sources: list[dict[str, int]], proposals: list[dict[str, object]], winners: dict[tuple[int, int], dict[str, int]]
) -> dict[str, object]:
    winner_ids = {int(item["index"]) for item in winners.values()}
    tied_sources = sum(int(item["min_anchor_count"] > 1) for item in proposals)
    contested_anchors = defaultdict(list)
    for item in proposals:
        contested_anchors[item["anchor"]].append(item)
    conflict_anchor_counts = [len(items) for items in contested_anchors.values() if len(items) > 1]
    discarded = len(sources) - len(winner_ids)
    return {
        "tied_sources": tied_sources,
        "conflict_anchors": len(conflict_anchor_counts),
        "discarded": discarded,
        "retained": len(winner_ids),
        "max_conflict_size": max(conflict_anchor_counts, default=1),
    }


def _independent_expected(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    top, mid, low, _ = _split_regions(grid)
    anchors = _anchors_from_panels(top, mid)
    sources = _lower_sources(low)
    winners, proposals = _resolve_assignments(sources, anchors)
    out = np.zeros(top.shape, dtype=int)
    for anchor, item in winners.items():
        out[anchor] = int(item["color"])
    return out, _semantic_signature(sources, proposals, winners)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_PANEL_H_RANGE)
    w = random.randint(*GEN_PANEL_W_RANGE)
    source_count = random.randint(*GEN_SOURCE_COUNT_RANGE)
    top_margin = random.choice(GEN_MARGIN_OPTIONS)
    bottom_margin = random.choice(GEN_MARGIN_OPTIONS)

    scenario = random.choices(
        ["random", "tie", "conflict", "both"],
        weights=[1, 2, 2, 3],
        k=1,
    )[0]

    sources: list[tuple[int, int]] = []
    required_holes: set[tuple[int, int]] = set()
    protected_anchors: set[tuple[int, int]] = set()
    expected_tie = False
    expected_conflict = False
    tie_mode = None
    conflict_mode = None

    if scenario in {"tie", "both"}:
        tie = _sample_tie_pattern(h, w, sources, required_holes, protected_anchors)
        if tie is None:
            return None
        tie_mode, _ = tie
        expected_tie = True

    if scenario in {"conflict", "both"}:
        conflict = _sample_conflict_pattern(h, w, sources, required_holes, protected_anchors)
        if conflict is None:
            return None
        conflict_mode, _ = conflict
        expected_conflict = True

    if not _sample_extra_sources(h, w, source_count, sources, required_holes, protected_anchors):
        return None

    panels = _build_panels(h, w, required_holes, protected_anchors)
    if panels is None:
        return None
    top_fg, mid_fg = panels
    anchors = _anchors_from_sets(h, w, top_fg, mid_fg)
    if len(anchors) < len(sources):
        return None

    placeholder_low = np.zeros((h, w), dtype=int)
    for r, c in sources:
        placeholder_low[r, c] = GREEN
    placeholder_sources = _lower_sources(placeholder_low)
    winners, proposals = _resolve_assignments(placeholder_sources, anchors)
    metrics = _semantic_signature(placeholder_sources, proposals, winners)

    if metrics["retained"] < 2:
        return None
    if expected_tie and metrics["tied_sources"] == 0:
        return None
    if expected_conflict and metrics["conflict_anchors"] == 0:
        return None
    if conflict_mode == "distance":
        groups = defaultdict(list)
        for item in proposals:
            groups[item["anchor"]].append(item)
        if not any(len(items) > 1 and len({int(entry["dist"]) for entry in items}) > 1 for items in groups.values()):
            return None
    if tie_mode == "vertical":
        if not any(
            int(item["min_anchor_count"]) > 1
            and item["anchor"][1] == item["col"]
            for item in proposals
        ):
            return None

    winner_indices = [int(item["index"]) for item in winners.values()]
    colors = [random.choice(LOWER_COLORS) for _ in sources]
    if len(winner_indices) >= 2:
        first, second = winner_indices[:2]
        colors[first] = GREEN
        colors[second] = YELLOW if first != second else MAGENTA
    grid = _render_input(h, w, top_fg, mid_fg, sources, colors, top_margin, bottom_margin)

    expected, semantic = _independent_expected(grid)
    if not np.array_equal(main(grid), expected):
        return None

    if expected_tie and semantic["tied_sources"] == 0:
        return None
    if expected_conflict and semantic["conflict_anchors"] == 0:
        return None
    if expected_conflict and semantic["discarded"] == 0:
        return None
    if semantic["retained"] < 2:
        return None
    if len(set(map(int, np.unique(expected))) - {BLACK}) < 2:
        return None
    if not (6 <= len(top_fg) <= 10 and 6 <= len(mid_fg) <= 10):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid puzzle instance")
