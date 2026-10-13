"""ARC task: rotate an anchored sample motif and stamp it into other quadrants."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue: deque[Coord] = deque([(sr, sc)])
        cells: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _quadrant(r: int, c: int, center_r: int, center_c: int) -> int:
    """Clockwise quadrant index: NW=0, NE=1, SE=2, SW=3."""
    assert r != center_r and c != center_c, "objects cannot lie on the separator cross"
    if r < center_r:
        return 0 if c < center_c else 1
    return 3 if c < center_c else 2


def _rotate_offset_clockwise(dr: int, dc: int, quarter_turns: int) -> Coord:
    """Rotate a row/column offset clockwise about the origin."""
    k = int(quarter_turns) % 4
    if k == 0:
        return int(dr), int(dc)
    if k == 1:
        return int(dc), int(-dr)
    if k == 2:
        return int(-dr), int(-dc)
    return int(-dc), int(dr)


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    height, width = map(int, g.shape)
    assert height % 2 == 1 and width % 2 == 1, "the separator cross must pass through the grid center"
    center_r, center_c = height // 2, width // 2

    counts = Counter(map(int, g.ravel()))
    bg, bg_count = counts.most_common(1)[0]
    assert sum(value == bg_count for value in counts.values()) == 1, "background must be uniquely dominant"
    assert np.all(g[center_r, :] == bg), "center row must be the background separator"
    assert np.all(g[:, center_c] == bg), "center column must be the background separator"

    components = _components(g != bg)
    samples = [component for component in components if len(component) > 1]
    singles = [component for component in components if len(component) == 1]
    assert len(samples) == 1, "expected one and only one complete sample motif"
    assert len(singles) >= 3, "expected several isolated marker pixels"
    sample = samples[0]

    sample_quadrants = {
        _quadrant(int(r), int(c), center_r, center_c) for r, c in sample
    }
    assert len(sample_quadrants) == 1, "sample motif must stay inside one quadrant"
    sample_q = next(iter(sample_quadrants))

    sample_color_counts = Counter(int(g[int(r), int(c)]) for r, c in sample)
    assert len(sample_color_counts) == 2, "sample motif must use one body color and one anchor color"
    anchor_colors = [color for color, count in sample_color_counts.items() if count == 1]
    body_colors = [color for color, count in sample_color_counts.items() if count > 1]
    assert len(anchor_colors) == 1 and len(body_colors) == 1, "sample anchor must be the unique odd-colored cell"
    anchor_color = anchor_colors[0]
    anchor_cells = [(int(r), int(c)) for r, c in sample if int(g[int(r), int(c)]) == anchor_color]
    assert len(anchor_cells) == 1
    anchor = anchor_cells[0]

    markers: list[tuple[int, int, int, int]] = []
    marker_colors: set[int] = set()
    target_quadrants: set[int] = set()
    for component in singles:
        r, c = map(int, component[0])
        color = int(g[r, c])
        q = _quadrant(r, c, center_r, center_c)
        assert q != sample_q, "the sample quadrant cannot contain marker pixels"
        assert color not in sample_color_counts, "marker colors must differ from sample colors"
        assert color not in marker_colors, "every marker pixel must have a distinct color"
        marker_colors.add(color)
        target_quadrants.add(q)
        markers.append((r, c, color, q))
    assert target_quadrants == ({0, 1, 2, 3} - {sample_q}), "all three target quadrants must be represented"

    return {
        "grid": g,
        "bg": int(bg),
        "center": (center_r, center_c),
        "sample": sample,
        "sample_q": int(sample_q),
        "anchor": anchor,
        "markers": markers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a uniquely anchored sample motif into each other quadrant with the quadrant-directed rotation.

    Input structure:
    - A background-colored center row and center column divide an odd-sized grid into four quadrants.
    - Exactly one quadrant contains a connected two-color sample motif; its singleton color is the anchor pixel.
    - The other quadrants contain isolated marker pixels, each with its own color, and every target quadrant is represented.

    Rule:
    - For each marker, rotate the sample clockwise by the number of quarter-turns from the sample quadrant to the marker quadrant.
    - Recolor every occupied sample cell, including the anchor, with the marker color, translate its rotated anchor onto the marker, and overlay the stamp.
    - Keep the original sample motif and the background separator cross unchanged.

    Key insight:
    - The marker's quadrant supplies the rotation, while the sample's internal odd-colored pixel—not a bounding-box corner or center—supplies the translation anchor.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    center_r, center_c = scene["center"]
    anchor_r, anchor_c = scene["anchor"]
    sample_q = int(scene["sample_q"])

    relative = [
        (int(r) - anchor_r, int(c) - anchor_c)
        for r, c in scene["sample"]
    ]
    assignments: dict[Coord, int] = {}
    owners: dict[Coord, Coord] = {}
    for marker_r, marker_c, marker_color, target_q in scene["markers"]:
        turns = (int(target_q) - sample_q) % 4
        marker = (int(marker_r), int(marker_c))
        for dr, dc in relative:
            rr, cc = _rotate_offset_clockwise(dr, dc, turns)
            out_r, out_c = marker_r + rr, marker_c + cc
            assert 0 <= out_r < g.shape[0] and 0 <= out_c < g.shape[1], "stamp must fit inside the grid"
            assert _quadrant(out_r, out_c, center_r, center_c) == target_q, "stamp must remain in its marker quadrant"
            existing = int(g[out_r, out_c])
            assert existing == bg or (out_r, out_c) == marker, "stamps cannot cover samples or other markers"
            cell = (int(out_r), int(out_c))
            assert cell not in assignments or owners[cell] == marker, "differently colored stamps cannot overlap"
            assignments[cell] = int(marker_color)
            owners[cell] = marker

    out = g.copy()
    for (r, c), color in assignments.items():
        out[r, c] = color
    return out


# ------------------------------ generator ------------------------------

MAX_GENERATION_ATTEMPTS = 500


def _mask_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _make_motif() -> tuple[np.ndarray, Coord] | None:
    height, width = random.choice([(4, 5), (5, 4), (5, 5)])
    interior = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (2 * r != height - 1 or 2 * c != width - 1)
    ]
    if not interior:
        return None
    anchor = random.choice(interior)
    mask = np.ones((height, width), dtype=bool)

    removable = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if r in (0, height - 1) or c in (0, width - 1)
    ]
    random.shuffle(removable)
    target_removed = random.randint(3, min(7, len(removable) - 4))
    removed = 0
    for r, c in removable:
        trial = mask.copy()
        trial[r, c] = False
        coords = np.argwhere(trial)
        if (
            _mask_connected(trial)
            and int(coords[:, 0].min()) == 0
            and int(coords[:, 0].max()) == height - 1
            and int(coords[:, 1].min()) == 0
            and int(coords[:, 1].max()) == width - 1
        ):
            mask = trial
            removed += 1
            if removed == target_removed:
                break
    if removed < 3:
        return None

    ar, ac = anchor
    if not all(mask[ar + dr, ac + dc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
        return None
    relative = {(int(r) - ar, int(c) - ac) for r, c in np.argwhere(mask)}
    if any(
        {_rotate_offset_clockwise(dr, dc, k) for dr, dc in relative} == relative
        for k in (1, 2, 3)
    ):
        return None
    return mask, (int(ar), int(ac))


def _quadrant_bounds(q: int, center_r: int, center_c: int, height: int, width: int) -> tuple[int, int, int, int]:
    if q == 0:
        return 0, center_r - 1, 0, center_c - 1
    if q == 1:
        return 0, center_r - 1, center_c + 1, width - 1
    if q == 2:
        return center_r + 1, height - 1, center_c + 1, width - 1
    return center_r + 1, height - 1, 0, center_c - 1


def _candidate_anchors(
    relative: set[Coord],
    bounds: tuple[int, int, int, int],
    occupied: set[Coord],
) -> list[tuple[Coord, set[Coord]]]:
    r0, r1, c0, c1 = bounds
    candidates: list[tuple[Coord, set[Coord]]] = []
    for anchor_r in range(r0 + 1, r1):
        for anchor_c in range(c0 + 1, c1):
            cells = {(anchor_r + dr, anchor_c + dc) for dr, dc in relative}
            if not cells or any(not (r0 + 1 <= r <= r1 - 1 and c0 + 1 <= c <= c1 - 1) for r, c in cells):
                continue
            if any(
                max(abs(r - rr), abs(c - cc)) <= 1
                for r, c in cells
                for rr, cc in occupied
            ):
                continue
            candidates.append(((anchor_r, anchor_c), cells))
    return candidates


def _expected_from_mask(
    inp: np.ndarray,
    mask: np.ndarray,
    local_anchor: Coord,
    sample_q: int,
    markers: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[set[Coord]]]:
    """Independent oracle using np.rot90 on the bitmap and an anchor one-hot mask."""
    out = inp.copy()
    stamp_sets: list[set[Coord]] = []
    anchor_bitmap = np.zeros(mask.shape, dtype=bool)
    anchor_bitmap[local_anchor] = True
    for marker_r, marker_c, color, target_q in markers:
        turns = (target_q - sample_q) % 4
        rotated = np.rot90(mask, k=-turns)
        rotated_anchor = np.argwhere(np.rot90(anchor_bitmap, k=-turns))
        assert len(rotated_anchor) == 1
        ar, ac = map(int, rotated_anchor[0])
        top, left = marker_r - ar, marker_c - ac
        cells = {
            (top + int(r), left + int(c))
            for r, c in np.argwhere(rotated)
        }
        for r, c in cells:
            out[r, c] = color
        stamp_sets.append(cells)
    return out, stamp_sets


def _semantic_validate(
    inp: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    local_anchor: Coord,
    sample_q: int,
    markers: list[tuple[int, int, int, int]],
    stamp_sets: list[set[Coord]],
) -> bool:
    try:
        scene = _parse_scene(inp)
        if scene["sample_q"] != sample_q or len(scene["markers"]) != len(markers):
            return False
        if {q for _r, _c, _color, q in markers} != ({0, 1, 2, 3} - {sample_q}):
            return False
        if {(q - sample_q) % 4 for _r, _c, _color, q in markers} != {1, 2, 3}:
            return False

        # No two output stamps overlap or even touch; this removes overlay-order ambiguity.
        for i, cells in enumerate(stamp_sets):
            for other in stamp_sets[i + 1 :]:
                if any(max(abs(r - rr), abs(c - cc)) <= 1 for r, c in cells for rr, cc in other):
                    return False

        # The internal anchor is neither a bbox corner nor its geometric center.
        ar, ac = local_anchor
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (ar, ac) in {(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))}:
            return False
        if 2 * ar == int(r0 + r1) and 2 * ac == int(c0 + c1):
            return False

        # Explicit counterexamples: fixed orientation and bbox-corner alignment differ.
        relative = {(int(r) - ar, int(c) - ac) for r, c in coords}
        rotation_is_needed = False
        anchor_is_needed = False
        for (mr, mc, _color, q), correct in zip(markers, stamp_sets):
            fixed = {(mr + dr, mc + dc) for dr, dc in relative}
            rotation_is_needed |= fixed != correct
            turns = (q - sample_q) % 4
            rotated = {_rotate_offset_clockwise(dr, dc, turns) for dr, dc in relative}
            min_r = min(dr for dr, _dc in rotated)
            min_c = min(dc for _dr, dc in rotated)
            corner_aligned = {(mr + dr - min_r, mc + dc - min_c) for dr, dc in rotated}
            anchor_is_needed |= corner_aligned != correct
        if not rotation_is_needed or not anchor_is_needed:
            return False

        for (_r, _c, color, _q), cells in zip(markers, stamp_sets):
            if int(np.count_nonzero(expected == color)) != len(cells):
                return False
        if np.array_equal(inp, expected):
            return False
        return np.array_equal(main(inp), expected)
    except (AssertionError, IndexError, ValueError):
        return False


def _gen_attempt() -> np.ndarray | None:
    motif = _make_motif()
    if motif is None:
        return None
    mask, local_anchor = motif

    height = random.choice([19, 21, 23, 25, 27, 29])
    width = random.choice([19, 21, 23, 25, 27, 29])
    center_r, center_c = height // 2, width // 2
    sample_q = random.randrange(4)

    palette = list(range(10))
    random.shuffle(palette)
    bg, body_color, anchor_color = palette[:3]
    extra_count = random.choices([0, 1, 2], weights=[3, 4, 3], k=1)[0]
    marker_colors = palette[3 : 6 + extra_count]

    relative = {(int(r) - local_anchor[0], int(c) - local_anchor[1]) for r, c in np.argwhere(mask)}
    sample_candidates = _candidate_anchors(
        relative,
        _quadrant_bounds(sample_q, center_r, center_c, height, width),
        set(),
    )
    if not sample_candidates:
        return None
    sample_anchor, sample_cells = random.choice(sample_candidates)
    occupied = set(sample_cells)

    target_qs = [q for q in range(4) if q != sample_q]
    target_qs.extend(random.choice(target_qs) for _ in range(extra_count))
    random.shuffle(target_qs)
    markers: list[tuple[int, int, int, int]] = []
    for q, color in zip(target_qs, marker_colors):
        turns = (q - sample_q) % 4
        rotated_relative = {_rotate_offset_clockwise(dr, dc, turns) for dr, dc in relative}
        candidates = _candidate_anchors(
            rotated_relative,
            _quadrant_bounds(q, center_r, center_c, height, width),
            occupied,
        )
        if not candidates:
            return None
        marker, cells = random.choice(candidates)
        occupied.update(cells)
        markers.append((marker[0], marker[1], int(color), q))

    inp = np.full((height, width), int(bg), dtype=int)
    sample_top = sample_anchor[0] - local_anchor[0]
    sample_left = sample_anchor[1] - local_anchor[1]
    for r, c in np.argwhere(mask):
        inp[sample_top + int(r), sample_left + int(c)] = int(body_color)
    inp[sample_anchor] = int(anchor_color)
    for r, c, color, _q in markers:
        inp[r, c] = int(color)

    expected, stamp_sets = _expected_from_mask(inp, mask, local_anchor, sample_q, markers)
    if not _semantic_validate(inp, expected, mask, local_anchor, sample_q, markers, stamp_sets):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid quadrant-anchor stamping input")
