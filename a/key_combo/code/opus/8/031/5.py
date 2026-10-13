from __future__ import annotations

import random

import numpy as np


PANEL_SIZE = 7
NUM_PANELS = 3
SEPARATOR = 0
PANEL_BG = 5
INPUT_SHAPE = (7, 23)
OUTPUT_SHAPE = (7, 7)
PANEL_STARTS = (0, 8, 16)
SEPARATOR_COLS = (7, 15)
FG_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
BIG_SIZE_CHOICES = tuple(range(4, 10))
SMALL_RELATIONS = ("row_left", "col_above", "upper_left", "upper_right", "lower_left")


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _top_left_residue(coords: np.ndarray) -> int:
    min_r = int(coords[:, 0].min())
    min_c = int(coords[:, 1].min())
    return (min_r + min_c) % 3


def _bbox_corner_residue(coords: np.ndarray, which: str) -> int:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    if which == "tr":
        return (r0 + c1) % 3
    if which == "bl":
        return (r1 + c0) % 3
    if which == "br":
        return (r1 + c1) % 3
    raise ValueError(which)


def _panel_info(panel: np.ndarray) -> dict:
    colors = [int(v) for v in np.unique(panel) if int(v) not in (PANEL_BG, SEPARATOR)]
    assert len(colors) == 1, "each panel must contain exactly one foreground color"
    color = colors[0]
    comps = _components_of_color(panel, color)
    assert len(comps) == 2, "each panel must contain exactly two 4-connected components"
    comps = sorted(comps, key=lambda arr: int(arr.shape[0]), reverse=True)
    big, small = comps
    assert int(big.shape[0]) > int(small.shape[0]), "largest component must be unique"
    return {
        "color": color,
        "big": big,
        "small": small,
        "big_size": int(big.shape[0]),
        "small_size": int(small.shape[0]),
        "big_residue": _top_left_residue(big),
        "small_residue": _top_left_residue(small),
        "big_anchor": (int(big[:, 0].min()), int(big[:, 1].min())),
        "small_anchor": (int(small[:, 0].min()), int(small[:, 1].min())),
    }


def _stripe_output(colors: list[int], residues: list[int]) -> np.ndarray:
    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    for color, residue in zip(colors, residues):
        out[(np.indices(OUTPUT_SHAPE).sum(axis=0) % 3) == int(residue)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep each panel's largest component and use its local top-left offset to seed diagonal stripe classes.

    Input structure:
    - The input is a 7x23 grid made of three 7x7 panels separated by one black column each.
    - Every panel has grey background cells and exactly one non-grey foreground color.
    - That foreground color is split into exactly two 4-connected components of different sizes.

    Rule:
    - Split the input into the three 7x7 panels.
    - In each panel, keep only the larger foreground component and discard the smaller distractor component.
    - For the kept component, take its color and the top-left corner of its occupied cells, defined by the component's minimum row and minimum column.
    - On a new 7x7 black canvas, fill every cell whose (row + col) mod 3 matches that panel's phase with the kept component's color.

    Key insight:
    - The larger component supplies both visible parameters of the answer: its color chooses the stripe color, and its local top-left position chooses which diagonal congruence class that color occupies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == INPUT_SHAPE, "expected three 7x7 panels with separator columns"
    assert np.all(g[:, SEPARATOR_COLS[0]] == SEPARATOR) and np.all(
        g[:, SEPARATOR_COLS[1]] == SEPARATOR
    ), "separator columns must be black"

    colors: list[int] = []
    residues: list[int] = []
    for c0 in PANEL_STARTS:
        panel = g[:, c0 : c0 + PANEL_SIZE]
        info = _panel_info(panel)
        colors.append(int(info["color"]))
        residues.append(int(info["big_residue"]))

    assert len(set(colors)) == NUM_PANELS, "panel colors must be distinct"
    assert sorted(residues) == [0, 1, 2], "largest-component phases must cover the three mod-3 classes"
    return _stripe_output(colors, residues)


def _blocked_with_gap(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < PANEL_SIZE and 0 <= nc < PANEL_SIZE:
                blocked.add((nr, nc))
    return blocked


def _grow_component(
    anchor: tuple[int, int], size: int, blocked: set[tuple[int, int]]
) -> set[tuple[int, int]] | None:
    ar, ac = anchor
    allowed = {
        (r, c)
        for r in range(ar, PANEL_SIZE)
        for c in range(ac, PANEL_SIZE)
        if (r, c) not in blocked
    }
    if anchor not in allowed or len(allowed) < size:
        return None

    cells = {anchor}
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors4(ar, ac)
        if (nr, nc) in allowed
    }
    while len(cells) < size:
        if not frontier:
            return None
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        for nr, nc in _neighbors4(*nxt):
            if (nr, nc) in allowed and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _fits_relation(
    big_anchor: tuple[int, int], small_anchor: tuple[int, int], relation: str
) -> bool:
    br, bc = big_anchor
    sr, sc = small_anchor
    if relation == "row_left":
        return sr == br and sc < bc
    if relation == "col_above":
        return sc == bc and sr < br
    if relation == "upper_left":
        return sr < br and sc < bc
    if relation == "upper_right":
        return sr < br and sc > bc
    if relation == "lower_left":
        return sr > br and sc < bc
    raise ValueError(relation)


def _candidate_anchors(size: int, residue: int) -> list[tuple[int, int]]:
    out = []
    for r in range(PANEL_SIZE):
        for c in range(PANEL_SIZE):
            if (r + c) % 3 == residue and (PANEL_SIZE - r) * (PANEL_SIZE - c) >= size:
                out.append((r, c))
    random.shuffle(out)
    return out


def _place_panel(color: int, big_size: int, big_residue: int, relation: str) -> np.ndarray | None:
    for _ in range(250):
        big_anchors = _candidate_anchors(big_size, big_residue)
        if relation in ("row_left", "upper_left", "upper_right"):
            big_anchors = [a for a in big_anchors if a[1] >= 1]
        if relation in ("col_above", "upper_left", "upper_right"):
            big_anchors = [a for a in big_anchors if a[0] >= 1]
        if relation == "lower_left":
            big_anchors = [a for a in big_anchors if a[1] >= 1 and a[0] <= PANEL_SIZE - 2]
        if not big_anchors:
            continue

        big_anchor = random.choice(big_anchors)
        big_cells = _grow_component(big_anchor, big_size, blocked=set())
        if big_cells is None:
            continue

        small_max = min(5, big_size - 2)
        small_size = random.randint(1, small_max)
        blocked = _blocked_with_gap(big_cells)
        small_candidates: list[tuple[int, int]] = []
        for residue in (0, 1, 2):
            if residue == big_residue:
                continue
            for anchor in _candidate_anchors(small_size, residue):
                if anchor in blocked:
                    continue
                if _fits_relation(big_anchor, anchor, relation):
                    small_candidates.append(anchor)
        random.shuffle(small_candidates)
        for small_anchor in small_candidates:
            small_cells = _grow_component(small_anchor, small_size, blocked)
            if small_cells is None:
                continue
            panel = np.full((PANEL_SIZE, PANEL_SIZE), PANEL_BG, dtype=int)
            for r, c in big_cells | small_cells:
                panel[r, c] = color
            info = _panel_info(panel)
            if info["big_size"] != big_size:
                continue
            if info["big_residue"] != big_residue:
                continue
            if info["big_size"] - info["small_size"] < 2:
                continue
            if (info["big_anchor"][0], info["big_anchor"][1]) not in set(map(tuple, info["big"])):
                continue
            if info["small_residue"] == info["big_residue"]:
                continue
            return panel
    return None


def _wrong_output_variants(grid: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    infos = [_panel_info(grid[:, c0 : c0 + PANEL_SIZE]) for c0 in PANEL_STARTS]
    colors = [int(info["color"]) for info in infos]

    variants.append(_stripe_output(colors, [int(info["small_residue"]) for info in infos]))

    union_residues = []
    for info in infos:
        all_coords = np.vstack([info["big"], info["small"]])
        union_residues.append(_top_left_residue(all_coords))
    variants.append(_stripe_output(colors, union_residues))

    for corner in ("tr", "bl", "br"):
        variants.append(
            _stripe_output(colors, [_bbox_corner_residue(info["big"], corner) for info in infos])
        )
    return variants


def _validate_example(grid: np.ndarray) -> None:
    assert grid.shape == INPUT_SHAPE
    assert np.all(grid[:, SEPARATOR_COLS[0]] == SEPARATOR)
    assert np.all(grid[:, SEPARATOR_COLS[1]] == SEPARATOR)

    infos = [_panel_info(grid[:, c0 : c0 + PANEL_SIZE]) for c0 in PANEL_STARTS]
    big_sizes = [int(info["big_size"]) for info in infos]
    residues = [int(info["big_residue"]) for info in infos]
    colors = [int(info["color"]) for info in infos]

    assert len(set(colors)) == NUM_PANELS
    assert len(set(big_sizes)) == NUM_PANELS
    assert sorted(residues) == [0, 1, 2]

    for info in infos:
        assert info["big_size"] - info["small_size"] >= 2
        assert info["small_residue"] != info["big_residue"]
        br, bc = info["big_anchor"]
        sr, sc = info["small_anchor"]
        assert sr < br or sc < bc, "small component must change the overall top-left foreground cue"

    out = main(grid)
    assert out.shape == OUTPUT_SHAPE
    assert np.count_nonzero(out == 0) == 0
    expected = _stripe_output(colors, residues)
    assert np.array_equal(out, expected)

    for wrong in _wrong_output_variants(grid):
        assert not np.array_equal(wrong, out), "example must reject common shortcuts"


def generate_input() -> np.ndarray:
    for _ in range(400):
        colors = random.sample(FG_COLORS, NUM_PANELS)
        residues = random.sample((0, 1, 2), NUM_PANELS)
        big_sizes = random.sample(BIG_SIZE_CHOICES, NUM_PANELS)
        relations = [random.choice(SMALL_RELATIONS) for _ in range(NUM_PANELS)]

        panels: list[np.ndarray] = []
        for color, big_size, residue, relation in zip(colors, big_sizes, residues, relations):
            panel = _place_panel(color, big_size, residue, relation)
            if panel is None:
                break
            panels.append(panel)
        if len(panels) != NUM_PANELS:
            continue

        grid = np.zeros(INPUT_SHAPE, dtype=int)
        grid[:, SEPARATOR_COLS[0]] = SEPARATOR
        grid[:, SEPARATOR_COLS[1]] = SEPARATOR
        for c0, panel in zip(PANEL_STARTS, panels):
            grid[:, c0 : c0 + PANEL_SIZE] = panel

        try:
            _validate_example(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid example")
