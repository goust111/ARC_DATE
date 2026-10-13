from __future__ import annotations

import os
import random
import sys
from collections import Counter

import numpy as np


_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_PATH not in sys.path:
    sys.path.append(_ARC_UTILS_PATH)

from arc_utils import Color, find_components, get_bg, paste, random_free_location


def _transform_panel(panel: np.ndarray, rot90: int, flip_lr: bool) -> np.ndarray:
    out = np.array(panel, copy=True)
    if flip_lr:
        out = np.fliplr(out)
    return np.rot90(out, k=rot90)


def _key_pair_at_topleft(panel: np.ndarray) -> int | None:
    for color in np.unique(panel):
        coords = np.argwhere(panel == color)
        if len(coords) != 2:
            continue
        (r0, c0), (r1, c1) = [tuple(map(int, v)) for v in coords]
        if abs(r0 - r1) + abs(c0 - c1) != 1:
            continue
        if set(((r0, c0), (r1, c1))) == {(0, 0), (0, 1)}:
            return int(color)
    return None


def _candidate_normalizations(panel: np.ndarray) -> list[tuple[np.ndarray, int, bool, int]]:
    candidates: list[tuple[np.ndarray, int, bool, int]] = []
    for rot90 in range(4):
        for flip_lr in (False, True):
            transformed = _transform_panel(panel, rot90, flip_lr)
            key = _key_pair_at_topleft(transformed)
            if key is None:
                continue
            candidates.append((transformed, int(key), bool(flip_lr), int(rot90)))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize each panel by a unique corner-key pose and overlay their interior drawings.

    Input structure:
    - The most common color is a separator surrounding several same-size rectangular panels.
    - Each panel is built from one shared blank color plus sparse colored marks.
    - Each panel has one two-cell corner key: one corner pixel and its right-adjacent partner, both same key color.

    Rule:
    - For each panel, apply the unique dihedral transform that maps its corner key to the top-left with the partner at top-right.
    - Create one output panel of that canonical panel shape and blank color.
    - Overlay each panel's non-blank, non-key marks at their normalized coordinates.
    - Omit all key cells from the output.

    Key insight:
    - The two-cell corner key uniquely determines each panel's pose; only after pose normalization do the fragments align.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert comps, "expected at least one panel"

    normalized: list[tuple[np.ndarray, int, int, np.ndarray]] = []
    per_panel_candidates: list[list[tuple[np.ndarray, int, bool, int]]] = []
    panel_sizes = Counter()
    blank_color = None

    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        panel = g[r0 : r1 + 1, c0 : c1 + 1]
        coords = np.asarray(comp["coords"], dtype=int)
        assert panel.size == coords.shape[0], "panel should be rectangular with no holes"

        candidates = _candidate_normalizations(panel)
        assert candidates, "failed to locate corner-key pose"
        per_panel_candidates.append(candidates)
        for transformed_shape in {tuple(map(int, transformed.shape)) for transformed, _key, _flip_lr, _rot90 in candidates}:
            panel_sizes[transformed_shape] += 1

    if not panel_sizes:
        raise AssertionError("no panel candidates found")
    expected_panel_count = len(per_panel_candidates)
    shared_shapes = [shape for shape, count in panel_sizes.items() if count == expected_panel_count]
    assert shared_shapes, "failed to find a common normalized shape across all panels"
    target_shape = sorted(shared_shapes)[0]
    h, w = target_shape

    target_candidates_by_panel: list[list[tuple[np.ndarray, int, int, bool, int]]] = []
    for candidates in per_panel_candidates:
        candidates_for_target = [
            (transformed, key, flip_lr, rot90)
            for transformed, key, flip_lr, rot90 in candidates
            if tuple(map(int, transformed.shape)) == target_shape
        ]
        assert candidates_for_target, "failed to choose a normalized candidate for a panel"
        enriched = []
        for transformed, key, flip_lr, rot90 in candidates_for_target:
            color_counts = Counter(map(int, transformed.ravel()))
            blank = next((color for color, _cnt in color_counts.most_common() if color != key), None)
            if blank is None:
                continue
            if color_counts[key] != 2:
                continue
            enriched.append((transformed, int(key), int(blank), bool(flip_lr), int(rot90)))
        if not enriched:
            raise AssertionError("failed to derive normalized panel candidates")
        target_candidates_by_panel.append(enriched)

    pair_votes = Counter()
    for panel_cands in target_candidates_by_panel:
        for key, blank in {(cand[1], cand[2]) for cand in panel_cands}:
            pair_votes[(key, blank)] += 1
    selected_key, selected_blank = pair_votes.most_common(1)[0][0]

    for panel_cands in target_candidates_by_panel:
        filtered = [cand for cand in panel_cands if cand[1] == selected_key and cand[2] == selected_blank]
        if not filtered:
            filtered = [cand for cand in panel_cands if cand[1] == selected_key]
        if not filtered:
            filtered = panel_cands

        transformed, key, blank, _flip_lr, _rot90 = sorted(
            filtered, key=lambda t: (t[3], t[2], tuple(int(v) for v in t[0].ravel()))
        )[0]

        if blank_color is None:
            blank_color = blank
        else:
            assert blank == blank_color, "all panels must share the same blank color"

        mark_mask = (transformed != blank) & (transformed != key)
        assert mark_mask.any(), "panel must contain at least one drawing mark"
        normalized.append((transformed, blank, key, mark_mask))

    assert blank_color is not None

    out = np.full((h, w), int(blank_color), dtype=int)
    for norm, _blank, key, mark_mask in normalized:
        painted = out != int(blank_color)
        assert not np.any(painted & mark_mask), "drawing marks must not overlap between panels"
        out[mark_mask] = norm[mark_mask]
        key_mask = norm == key
        assert np.all(out[key_mask] == int(blank_color)), "corner key pixels should be ignored"

    return out


PANEL_H_RANGE = (3, 9)
PANEL_W_RANGE = (3, 9)
PANEL_COUNT_RANGE = (2, 6)
CANVAS_H_RANGE = (18, 30)
CANVAS_W_RANGE = (18, 30)
GEN_ATTEMPTS = 500


def _random_connected_cells(
    h: int,
    w: int,
    target: int,
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    if len(free) < target:
        return set()

    cells = {random.choice(free)}
    for _ in range(1, target):
        frontier = []
        for r, c in list(cells):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and (nr, nc) not in cells
                    and (nr, nc) not in forbidden
                ):
                    frontier.append((nr, nc))
        if not frontier:
            return set()
        cells.add(random.choice(frontier))
    return cells


def _partition_mark_cells(panel_count: int, total: int) -> list[int]:
    assert panel_count >= 1
    assert total >= panel_count

    sizes = [1] * panel_count
    left = total - panel_count
    for i in range(panel_count - 1):
        slots_left = panel_count - i - 1
        max_add = min(5, left - slots_left)
        if max_add < 0:
            return []
        add = random.randint(0, max_add)
        sizes[i] += add
        left -= add
    sizes[-1] += left
    return sizes


def _generate_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    panel_count = random.randint(*PANEL_COUNT_RANGE)

    sep = random.choice([c for c in Color.NOT_BLACK if c != 0])
    blank = random.choice([c for c in Color.NOT_BLACK if c != sep])
    key_color = random.choice([c for c in Color.NOT_BLACK if c not in (sep, blank)])
    mark_palette = [c for c in Color.NOT_BLACK if c not in (sep, blank, key_color)]
    if len(mark_palette) < 2:
        return None
    random.shuffle(mark_palette)
    mark_palette = mark_palette[: random.randint(2, min(4, len(mark_palette)))]

    max_mark = min(panel_h * panel_w - 2, panel_count * 6)
    min_mark = panel_count
    if max_mark < min_mark:
        return None
    total_mark_cells = random.randint(min_mark, max_mark)

    panel_sizes = _partition_mark_cells(panel_count, total_mark_cells)
    if not panel_sizes:
        return None

    key_cells = {(0, 0), (0, 1)}
    if key_cells == set():
        return None

    panel_assignments: list[set[tuple[int, int]]] = [set() for _ in range(panel_count)]
    forbidden: set[tuple[int, int]] = set(key_cells)
    for i, size in enumerate(panel_sizes):
        stroke = _random_connected_cells(panel_h, panel_w, size, forbidden)
        if not stroke:
            return None
        panel_assignments[i] = stroke
        forbidden.update(stroke)

    canonical_panels: list[np.ndarray] = []
    expected = np.full((panel_h, panel_w), blank, dtype=int)
    for cells in panel_assignments:
        panel = np.full((panel_h, panel_w), blank, dtype=int)
        panel[0, 0] = key_color
        panel[0, 1] = key_color
        for cell in cells:
            if cell in key_cells:
                return None
            panel[cell[0], cell[1]] = random.choice(mark_palette)
            if expected[cell[0], cell[1]] != blank:
                return None
            expected[cell[0], cell[1]] = panel[cell[0], cell[1]]
        canonical_panels.append(panel)

    transforms = [(random.choice((0, 2)), random.choice((False, True))) for _ in canonical_panels]
    oriented_panels = [_transform_panel(panel, rot, flip) for panel, (rot, flip) in zip(canonical_panels, transforms)]

    canvas_h = random.randint(*CANVAS_H_RANGE)
    canvas_w = random.randint(*CANVAS_W_RANGE)
    grid = np.full((canvas_h, canvas_w), sep, dtype=int)
    for panel in oriented_panels:
        if panel.size == 0:
            return None
        try:
            r0, c0 = random_free_location(
                grid,
                panel,
                transparent=sep,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, panel, r0, c0)

    if get_bg(grid) != sep:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for task 8")
