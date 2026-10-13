from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 7)

PANEL_HEIGHT_RANGE = (7, 10)
PANEL_WIDTH_RANGE = (7, 9)
PANEL_COUNT = 3

MIN_FOREGROUND = 8
MAX_FOREGROUND_FRACTION = 0.42
MIN_TIES = 2
LAYOUT_RETRIES = 200
PLACEMENT_RETRIES = 60
GEN_RETRIES = 400

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    sep_cols = np.flatnonzero(np.all(g == SEPARATOR, axis=0))
    assert sep_cols.size == 2, "expected exactly two full-height separator columns"
    left_sep, right_sep = map(int, sep_cols)
    panel_w = left_sep
    assert panel_w > 0, "left panel must be non-empty"
    assert right_sep - left_sep - 1 == panel_w, "panels must have equal width"
    assert w - right_sep - 1 == panel_w, "panels must have equal width"
    panels = [
        g[:, :panel_w],
        g[:, left_sep + 1 : right_sep],
        g[:, right_sep + 1 :],
    ]
    assert all(panel.shape == (h, panel_w) for panel in panels), "panel extraction failed"
    return panels


def _vote_color(values: tuple[int, int, int]) -> int:
    color, count = Counter(values).most_common(1)[0]
    return int(color) if count >= 2 else BACKGROUND


def _independent_vote(panels: list[np.ndarray]) -> np.ndarray:
    h, w = panels[0].shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            values = tuple(int(panel[r, c]) for panel in panels)
            out[r, c] = _vote_color(values)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge three aligned panels by pixelwise majority vote, erasing three-way color conflicts.

    Input structure:
    - The input is a black background grid split by two full-height gray separator columns into three equal-width panels.
    - The three panels are positionally aligned: each panel cell contributes to the same output coordinate.
    - Foreground colors come from the ARC palette, while black acts as background.

    Rule:
    - Remove the two gray separator columns and align the three panels cell-by-cell.
    - For each aligned triplet of values, output the color that appears at least twice.
    - If all three values are different, output black; this also keeps fully empty positions black.

    Key insight:
    - The panels are not separate objects to select from; they are three aligned votes for every pixel, and a three-way disagreement is resolved by deletion rather than by choosing one panel.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    left, middle, right = panels
    out = np.where(
        left == middle,
        left,
        np.where(left == right, left, np.where(middle == right, middle, BACKGROUND)),
    )
    return np.asarray(out, dtype=int)


def _grow_blob(height: int, width: int, area: int) -> np.ndarray | None:
    if area < 1 or area > height * width:
        return None

    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}
    frontier = set(_neighbors4(*seed))
    frontier = {(r, c) for r, c in frontier if 0 <= r < height and 0 <= c < width}

    while len(cells) < area:
        if not frontier:
            return None
        weighted: list[tuple[int, int]] = []
        for r, c in frontier:
            shared = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
            weighted.extend([(r, c)] * (1 + shared))
        r, c = random.choice(weighted)
        frontier.remove((r, c))
        cells.add((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    blob = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        blob[r, c] = True
    return blob


def _can_stamp(existing: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = existing.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and existing[rr, cc]:
                    return False
    return True


def _place_blob(mask: np.ndarray, blob: np.ndarray) -> bool:
    h, w = mask.shape
    bh, bw = blob.shape
    offsets = [tuple(map(int, rc)) for rc in np.argwhere(blob)]
    positions = [
        (top, left)
        for top in range(h - bh + 1)
        for left in range(w - bw + 1)
    ]
    random.shuffle(positions)
    for top, left in positions[:PLACEMENT_RETRIES]:
        cells = [(top + r, left + c) for r, c in offsets]
        if not _can_stamp(mask, cells):
            continue
        for r, c in cells:
            mask[r, c] = True
        return True
    return False


def _sample_shared_mask(height: int, width: int) -> np.ndarray | None:
    max_foreground = max(MIN_FOREGROUND, int(height * width * MAX_FOREGROUND_FRACTION))
    for _ in range(LAYOUT_RETRIES):
        mask = np.zeros((height, width), dtype=bool)
        target_blobs = random.randint(2, 4)
        ok = True
        for blob_index in range(target_blobs):
            remaining = target_blobs - blob_index - 1
            box_h = random.randint(2, min(5, height))
            box_w = random.randint(2, min(5, width))
            area_hi = min(box_h * box_w, 8, max_foreground - int(mask.sum()) - 3 * remaining)
            if area_hi < 3:
                ok = False
                break
            area = random.randint(3, area_hi)
            blob = _grow_blob(box_h, box_w, area)
            if blob is None or not _place_blob(mask, blob):
                ok = False
                break
        foreground = int(mask.sum())
        if not ok or foreground < MIN_FOREGROUND or foreground > max_foreground:
            continue
        if np.any(mask.all(axis=0)):
            continue
        return mask
    return None


def _random_color(exclude: set[int] | None = None) -> int:
    blocked = set() if exclude is None else set(exclude)
    choices = [color for color in PALETTE if color not in blocked]
    return int(random.choice(choices))


def _sample_triples(mask: np.ndarray) -> list[np.ndarray]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    assert len(coords) >= MIN_FOREGROUND

    triples: dict[tuple[int, int], tuple[int, int, int]] = {}

    # Force each panel to disagree with the majority at least once.
    for minority_panel, coord in enumerate(coords[:3]):
        majority_color = _random_color()
        minority_color = _random_color({majority_color})
        triplet = [majority_color, majority_color, majority_color]
        triplet[minority_panel] = minority_color
        triples[coord] = tuple(triplet)

    # Force visible three-way conflict cells.
    max_forced_ties = min(5, len(coords) - 4)
    tie_count = random.randint(MIN_TIES, max_forced_ties)
    cursor = 3
    for coord in coords[cursor : cursor + tie_count]:
        triples[coord] = tuple(random.sample(PALETTE, 3))
    cursor += tie_count

    # Force at least one unanimous foreground vote.
    unanimous_coord = coords[cursor]
    unanimous_color = _random_color()
    triples[unanimous_coord] = (unanimous_color, unanimous_color, unanimous_color)
    cursor += 1

    for coord in coords[cursor:]:
        mode = random.choices(
            ("majority", "unanimous", "tie"),
            weights=(6, 2, 1),
            k=1,
        )[0]
        if mode == "tie":
            triples[coord] = tuple(random.sample(PALETTE, 3))
            continue
        if mode == "unanimous":
            color = _random_color()
            triples[coord] = (color, color, color)
            continue
        majority_color = _random_color()
        minority_color = _random_color({majority_color})
        minority_panel = random.randrange(PANEL_COUNT)
        triplet = [majority_color, majority_color, majority_color]
        triplet[minority_panel] = minority_color
        triples[coord] = tuple(triplet)

    panels = [np.zeros(mask.shape, dtype=int) for _ in range(PANEL_COUNT)]
    for (r, c), triplet in triples.items():
        for panel_index in range(PANEL_COUNT):
            panels[panel_index][r, c] = int(triplet[panel_index])
    return panels


def _render_input(panels: list[np.ndarray]) -> np.ndarray:
    h, w = panels[0].shape
    grid = np.zeros((h, PANEL_COUNT * w + 2), dtype=int)
    grid[:, w] = SEPARATOR
    grid[:, 2 * w + 1] = SEPARATOR
    grid[:, :w] = panels[0]
    grid[:, w + 1 : 2 * w + 1] = panels[1]
    grid[:, 2 * w + 2 :] = panels[2]
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    panels = _split_panels(grid)
    supports = [(panel != BACKGROUND) for panel in panels]
    assert np.array_equal(supports[0], supports[1])
    assert np.array_equal(supports[1], supports[2])
    mask = supports[0]

    unanimous = 0
    majority = 0
    ties = 0
    minority_panels: set[int] = set()
    output_colors: set[int] = set()
    for r, c in np.argwhere(mask):
        triplet = tuple(int(panel[r, c]) for panel in panels)
        assert all(color in PALETTE for color in triplet)
        counts = Counter(triplet)
        winner, copies = counts.most_common(1)[0]
        if copies == 1:
            ties += 1
            continue
        output_colors.add(int(winner))
        if copies == 3:
            unanimous += 1
            continue
        majority += 1
        minority = [i for i, color in enumerate(triplet) if color != winner]
        assert len(minority) == 1
        minority_panels.add(int(minority[0]))

    assert int(mask.sum()) >= MIN_FOREGROUND
    assert ties >= MIN_TIES
    assert unanimous >= 1
    assert majority >= 3
    assert unanimous + majority > ties, "most foreground cells should have a real majority"
    assert minority_panels == {0, 1, 2}, "every panel must lose at least one vote"
    assert len(output_colors) >= 2, "output should keep at least two non-black colors"

    oracle = _independent_vote(panels)
    actual = main(grid)
    assert np.array_equal(actual, oracle), "main must match the semantic vote oracle"
    for panel in panels:
        assert not np.array_equal(panel, actual), "output must differ from every single panel"
    assert np.any(oracle == BACKGROUND), "conflicts should erase at least one foreground cell"
    assert np.any(oracle != BACKGROUND), "output must retain some voted foreground"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(*PANEL_HEIGHT_RANGE)
        width = random.randint(*PANEL_WIDTH_RANGE)
        if PANEL_COUNT * width + 2 > 30:
            continue
        mask = _sample_shared_mask(height, width)
        if mask is None:
            continue
        panels = _sample_triples(mask)
        grid = _render_input(panels)
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid three-panel voting puzzle")
